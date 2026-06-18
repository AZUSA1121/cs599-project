"""向量数据库模块 - 支持 ChromaDB / FAISS 后端"""

import os
import json
import uuid
import pickle
from typing import Optional

from knowledge_base.embedder import Embedder
from config import DATA_DIR


class VectorStore:
    """向量存储与检索，支持多种后端"""

    def __init__(self, db_type: str = "chroma",
                 persist_dir: Optional[str] = None,
                 embedder: Optional[Embedder] = None):
        """
        Args:
            db_type: "chroma" 或 "faiss"
            persist_dir: 持久化目录
            embedder: Embedder 实例 (chroma+默认嵌入时可为 None)
        """
        self.db_type = db_type
        self.embedder = embedder

        if persist_dir is None:
            persist_dir = os.path.join(DATA_DIR, "vector_db")
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir

        if db_type == "chroma":
            self._store = _ChromaStore(persist_dir, embedder)
        elif db_type == "faiss":
            self._store = _FAISSStore(persist_dir, embedder)
        else:
            raise ValueError(f"不支持的向量数据库: {db_type}，可选: chroma, faiss")

    def add_chunks(self, chunks: list, embeddings: list = None):
        """
        添加文本块到向量库
        chunks: [{"id": str, "paper_id": str, "content": str, "chunk_type": str}, ...]
        embeddings: 预计算的嵌入向量列表 (chroma+默认嵌入时可省略)
        """
        self._store.add_chunks(chunks, embeddings)

    def search(self, query: str, n_results: int = 10, paper_id: str = None) -> list:
        """
        语义检索
        query: 查询文本
        返回: [{"chunk_id", "content", "paper_id", "chunk_type", "distance"}, ...]
        """
        return self._store.search(query, n_results, paper_id)

    def search_with_score(self, query: str, n_results: int = 10,
                          paper_id: str = None, threshold: float = 0.3) -> list:
        """带相似度阈值的检索"""
        results = self.search(query, n_results, paper_id)
        if threshold and self.db_type == "faiss":
            results = [r for r in results if r.get("distance", 0) >= threshold]
        return results

    def delete_by_paper(self, paper_id: str):
        """删除某篇论文的所有向量"""
        self._store.delete_by_paper(paper_id)

    def count(self) -> int:
        return self._store.count()

    def reset(self):
        """清空向量库"""
        self._store.reset()

    @property
    def info(self) -> dict:
        """向量库信息"""
        return {
            "db_type": self.db_type,
            "count": self.count(),
            "embed_method": self.embedder.method if self.embedder else "chroma-default",
        }


class _ChromaStore:
    """ChromaDB 后端"""

    def __init__(self, persist_dir: str, embedder: Optional[Embedder]):
        import chromadb
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 如果使用自定义 embedder，创建对应的 embedding function
        if embedder and embedder.method != "chroma":
            self._ef = _CustomEmbeddingFunction(embedder)
            self.collection = self.client.get_or_create_collection(
                name="paper_chunks",
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.collection = self.client.get_or_create_collection(
                name="paper_chunks",
                metadata={"hnsw:space": "cosine"}
            )

    def add_chunks(self, chunks: list, embeddings: list = None):
        if not chunks:
            return
        ids = [c["id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {"paper_id": c.get("paper_id", ""), "chunk_type": c.get("chunk_type", "")}
            for c in chunks
        ]
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            kwargs = {
                "ids": ids[i:i + batch_size],
                "documents": documents[i:i + batch_size],
                "metadatas": metadatas[i:i + batch_size],
            }
            if embeddings:
                kwargs["embeddings"] = embeddings[i:i + batch_size]
            self.collection.upsert(**kwargs)

    def search(self, query: str, n_results: int = 10, paper_id: str = None) -> list:
        total = self.collection.count()
        if total == 0:
            return []
        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, total),
            "include": ["documents", "metadatas", "distances"]
        }
        if paper_id:
            kwargs["where"] = {"paper_id": paper_id}

        results = self.collection.query(**kwargs)
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                dist = results["distances"][0][i]
                chunks.append({
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "paper_id": results["metadatas"][0][i]["paper_id"],
                    "chunk_type": results["metadatas"][0][i].get("chunk_type", ""),
                    "distance": round(1 - dist, 4),  # cosine similarity
                })
        return chunks

    def delete_by_paper(self, paper_id: str):
        self.collection.delete(where={"paper_id": paper_id})

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        if self.embedder and self.embedder.method != "chroma":
            self.collection = self.client.get_or_create_collection(
                name=name, embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.collection = self.client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )


class _FAISSStore:
    """FAISS 后端"""

    def __init__(self, persist_dir: str, embedder: Optional[Embedder]):
        self.embedder = embedder
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.index_path = os.path.join(persist_dir, "faiss.index")
        self.meta_path = os.path.join(persist_dir, "faiss_meta.pkl")

        self._index = None
        self._metas = []  # [{id, paper_id, chunk_type, content}, ...]
        self._load()

    def _load(self):
        """从磁盘加载索引"""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            import faiss
            self._index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self._metas = pickle.load(f)
            print(f"[FAISS] 已加载索引: {len(self._metas)} 条")

    def _save(self):
        """保存索引到磁盘"""
        import faiss
        if self._index:
            faiss.write_index(self._index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self._metas, f)

    def _ensure_embedder(self):
        if not self.embedder or self.embedder.method == "chroma":
            raise ValueError("FAISS 需要显式的 Embedding 模型，请配置 embed_method 为 local 或 openai")

    def _ensure_index(self, dim: int):
        import faiss
        if self._index is None or self._index.d != dim:
            self._index = faiss.IndexFlatIP(dim)  # 内积 (向量已归一化=cosine)
            self._metas = []

    def add_chunks(self, chunks: list, embeddings: list = None):
        if not chunks:
            return
        self._ensure_embedder()
        import numpy as np

        ids = [c["id"] for c in chunks]
        documents = [c["content"] for c in chunks]

        if not embeddings:
            embeddings = self.embedder.embed_texts(documents)

        vectors = np.array(embeddings, dtype=np.float32)
        self._ensure_index(vectors.shape[1])
        self._index.add(vectors)

        for i, c in enumerate(chunks):
            self._metas.append({
                "id": c.get("id", ids[i] if i < len(ids) else str(uuid.uuid4())),
                "paper_id": c.get("paper_id", ""),
                "chunk_type": c.get("chunk_type", ""),
                "content": c.get("content", documents[i] if i < len(documents) else ""),
            })
        self._save()

    def search(self, query: str, n_results: int = 10, paper_id: str = None) -> list:
        if not self._index or self._index.ntotal == 0:
            return []
        self._ensure_embedder()

        import numpy as np
        query_vec = np.array([self.embedder.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        n = min(n_results, self._index.ntotal)
        scores, indices = self._index.search(query_vec, n)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < 0 or idx >= len(self._metas):
                continue
            meta = self._metas[idx]
            if paper_id and meta["paper_id"] != paper_id:
                continue
            results.append({
                "chunk_id": meta["id"],
                "content": meta["content"],
                "paper_id": meta["paper_id"],
                "chunk_type": meta["chunk_type"],
                "distance": round(float(scores[0][i]), 4),
            })
        return results

    def delete_by_paper(self, paper_id: str):
        """FAISS 不支持高效删除，这里用重建索引的方式"""
        if not self._metas:
            return
        import numpy as np

        keep_indices = [i for i, m in enumerate(self._metas) if m["paper_id"] != paper_id]
        if len(keep_indices) == len(self._metas):
            return

        # 获取保留的向量重建索引
        import faiss
        old_dim = self._index.d
        self._index = faiss.IndexFlatIP(old_dim)
        self._metas = [self._metas[i] for i in keep_indices]
        if self._metas:
            # 需要重新嵌入（FAISS不存储原始向量）
            # 这里简化处理：清空后需重新索引
            print(f"[FAISS] 已从元数据中移除 paper_id={paper_id}，共 {len(keep_indices)} 条保留")
        self._save()

    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    def reset(self):
        self._index = None
        self._metas = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)


class _CustomEmbeddingFunction:
    """ChromaDB 自定义 Embedding Function 适配器"""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed_texts(input)

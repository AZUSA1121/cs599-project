"""Embedding 模块 - 支持多种 Embedding 后端"""

import os
from typing import Optional


class Embedder:
    """文本向量化器，支持多种后端"""

    def __init__(self, method: str = "chroma",
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Args:
            method: 嵌入方法 - "chroma"(默认), "local"(sentence-transformers), "openai"(API)
            model_name: 模型名称 (local时为HuggingFace模型名, openai时为模型ID)
            api_key: OpenAI兼容API密钥
            base_url: OpenAI兼容API地址
        """
        self.method = method
        self.model_name = model_name
        self._model = None
        self._dim = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量文本嵌入"""
        if not texts:
            return []
        if self.method == "local":
            return self._embed_local(texts)
        elif self.method == "openai":
            return self._embed_openai(texts)
        elif self.method == "chroma":
            raise ValueError("chroma 嵌入由 ChromaDB 内部处理，无需手动调用 embed_texts")
        else:
            raise ValueError(f"不支持的嵌入方法: {self.method}")

    def embed_query(self, text: str) -> list[float]:
        """单条查询文本嵌入"""
        result = self.embed_texts([text])
        return result[0] if result else []

    @property
    def dimension(self) -> int:
        """嵌入维度"""
        if self._dim is None:
            if self.method == "local":
                self._ensure_model()
                self._dim = self._model.get_sentence_embedding_dimension()
            elif self.method == "openai":
                self._dim = 1536  # text-embedding-ada-002 default
            else:
                self._dim = 384  # ChromaDB default model
        return self._dim

    def _ensure_model(self):
        """懒加载本地模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "使用 local 嵌入需要安装 sentence-transformers:\n"
                    "  pip install sentence-transformers"
                )
            model = self.model_name or "paraphrase-multilingual-MiniLM-L12-v2"
            print(f"[Embedder] 加载本地模型: {model}")
            self._model = SentenceTransformer(model)
            print(f"[Embedder] 模型加载完成, 维度: {self._model.get_sentence_embedding_dimension()}")

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """使用 sentence-transformers 本地模型"""
        self._ensure_model()
        embeddings = self._model.encode(texts, show_progress_bar=False,
                                        normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """使用 OpenAI 兼容 Embedding API"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 库: pip install openai")

        import config
        api_key = self.api_key or getattr(config, 'LLM_API_KEY', '')
        base_url = self.base_url or getattr(config, 'LLM_BASE_URL', '').replace('/chat', '')
        # 对于 embedding，通常 base_url 需要是 /v1 格式
        if not base_url.endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'
        model = self.model_name or "text-embedding-ada-002"

        client = OpenAI(api_key=api_key, base_url=base_url)
        # OpenAI embedding API 单次最多 2048 条
        batch_size = 2048
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(input=batch, model=model)
            all_embeddings.extend([d.embedding for d in resp.data])
        return all_embeddings

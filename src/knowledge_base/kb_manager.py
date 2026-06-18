"""本地知识库管理模块 - SQLite + ChromaDB 向量检索"""

import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import Optional

from knowledge_base.vector_store import VectorStore
from config import KB_DIR


class KnowledgeBase:
    """本地论文知识库，SQLite 存储 + ChromaDB 向量检索"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(KB_DIR, "kb.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.vector_store = VectorStore()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS categories (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT DEFAULT '',
                color TEXT DEFAULT '#6366f1',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                file_path TEXT,
                raw_text TEXT,
                decomposed_json TEXT,
                keywords TEXT,
                abstract TEXT,
                category_id TEXT DEFAULT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(id)
            );

            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            );

            CREATE TABLE IF NOT EXISTS tags (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS paper_tags (
                paper_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                PRIMARY KEY (paper_id, tag_id),
                FOREIGN KEY (paper_id) REFERENCES papers(id),
                FOREIGN KEY (tag_id) REFERENCES tags(id)
            );

            CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);
            CREATE INDEX IF NOT EXISTS idx_papers_keywords ON papers(keywords);
            CREATE INDEX IF NOT EXISTS idx_chunks_type ON knowledge_chunks(chunk_type);
            CREATE INDEX IF NOT EXISTS idx_chunks_paper ON knowledge_chunks(paper_id);
        """)

        # 迁移：为已有的 papers 表补充 category_id 列
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(papers)").fetchall()]
        if "category_id" not in cols:
            cursor.execute("ALTER TABLE papers ADD COLUMN category_id TEXT DEFAULT NULL")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_category ON papers(category_id)")

        conn.commit()
        conn.close()

    # ---- 论文 CRUD ----

    @staticmethod
    def _normalize_text_field(value) -> str:
        """将列表/字典等结构化值归一化为可存储的文本。"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value if item is not None)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def add_paper(self, title: str, authors: str = "", file_path: str = "",
                  raw_text: str = "", decomposed: dict = None, keywords: list = None,
                  category_id: str = None, paper_id: str = None) -> str:
        """添加论文到知识库"""
        paper_id = paper_id or str(uuid.uuid4())
        now = datetime.now().isoformat()
        decomposed_json = json.dumps(decomposed, ensure_ascii=False) if decomposed else "{}"
        title = self._normalize_text_field(title)
        authors = self._normalize_text_field(authors)
        file_path = self._normalize_text_field(file_path)
        raw_text = self._normalize_text_field(raw_text)
        keywords_str = self._normalize_text_field(keywords)
        abstract = self._normalize_text_field(decomposed.get("abstract", "") if decomposed else "")

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO papers (id, title, authors, file_path, raw_text, decomposed_json, keywords, abstract, category_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (paper_id, title, authors, file_path, raw_text, decomposed_json,
              keywords_str, abstract, category_id, now, now))
        conn.commit()
        conn.close()
        return paper_id

    def update_decomposition(self, paper_id: str, decomposed: dict):
        """更新论文的拆解结果"""
        now = datetime.now().isoformat()
        decomposed_json = json.dumps(decomposed, ensure_ascii=False)
        keywords_str = self._normalize_text_field(decomposed.get("keywords", []))
        abstract = self._normalize_text_field(decomposed.get("abstract", ""))

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE papers SET decomposed_json=?, keywords=?, abstract=?, updated_at=?
            WHERE id=?
        """, (decomposed_json, keywords_str, abstract, now, paper_id))
        conn.commit()
        conn.close()

    def get_paper(self, paper_id: str) -> Optional[dict]:
        """获取单篇论文"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE id=?", (paper_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            result = dict(row)
            result["decomposed"] = json.loads(result["decomposed_json"])
            return result
        return None

    def list_papers(self, limit: int = 50, offset: int = 0, category_id: str = None) -> list:
        """列出所有论文，可按分类过滤"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if category_id == "__uncategorized__":
            cursor.execute(
                "SELECT id, title, authors, keywords, abstract, category_id, created_at FROM papers WHERE category_id IS NULL OR category_id='' ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset))
        elif category_id:
            cursor.execute(
                "SELECT id, title, authors, keywords, abstract, category_id, created_at FROM papers WHERE category_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (category_id, limit, offset))
        else:
            cursor.execute(
                "SELECT id, title, authors, keywords, abstract, category_id, created_at FROM papers ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def delete_paper(self, paper_id: str) -> bool:
        """删除论文及其关联数据（含向量库）"""
        self.vector_store.delete_by_paper(paper_id)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM knowledge_chunks WHERE paper_id=?", (paper_id,))
        cursor.execute("DELETE FROM paper_tags WHERE paper_id=?", (paper_id,))
        cursor.execute("DELETE FROM papers WHERE id=?", (paper_id,))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    # ---- 知识块 ----

    def add_chunks(self, paper_id: str, chunks: list):
        """添加知识块（拆解后的各部分内容），同时写入向量库"""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        vector_chunks = []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            chunk_type = self._normalize_text_field(chunk["type"])
            content = self._normalize_text_field(chunk["content"])
            cursor.execute("""
                INSERT INTO knowledge_chunks (id, paper_id, chunk_type, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, paper_id, chunk_type, content,
                  json.dumps(chunk.get("metadata", {}), ensure_ascii=False), now))
            vector_chunks.append({
                "id": chunk_id,
                "paper_id": paper_id,
                "type": chunk_type,
                "content": content
            })
        conn.commit()
        conn.close()

        # 写入向量库
        self.vector_store.add_chunks(vector_chunks)

    # ---- 搜索 ----

    def search(self, query: str, limit: int = 5, use_vector: bool = True) -> list:
        """搜索知识库：优先使用向量语义检索"""
        if use_vector:
            return self._vector_search(query, limit)
        return self._keyword_search(query, limit)

    def _vector_search(self, query: str, limit: int = 5) -> list:
        """基于向量语义搜索"""
        # 向量检索相关知识块
        vec_chunks = self.vector_store.search(query, n_results=limit * 2)

        # 收集涉及的 paper_id
        paper_ids = list(set(c["paper_id"] for c in vec_chunks))

        # 从 SQLite 获取论文信息
        papers = []
        if paper_ids:
            conn = self._get_conn()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(paper_ids))
            cursor.execute(f"""
                SELECT id, title, authors, abstract, keywords, decomposed_json, created_at
                FROM papers WHERE id IN ({placeholders})
                ORDER BY created_at DESC
            """, paper_ids)
            papers = [dict(row) for row in cursor.fetchall()]
            conn.close()

        # 格式化 chunks，附带论文信息
        paper_map = {p["id"]: p for p in papers}
        chunks_with_info = []
        for c in vec_chunks[:limit]:
            p = paper_map.get(c["paper_id"], {})
            chunks_with_info.append({
                "paper_id": c["paper_id"],
                "paper_title": p.get("title", ""),
                "chunk_type": c["chunk_type"],
                "content": c["content"],
                "score": c["distance"]
            })

        return {"papers": papers, "chunks": chunks_with_info}

    def _keyword_search(self, query: str, limit: int = 5) -> list:
        """基于关键词搜索知识库（全文检索）"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 在论文标题、关键词、摘要中搜索
        cursor.execute("""
            SELECT p.id, p.title, p.authors, p.abstract, p.keywords,
                   p.decomposed_json, p.created_at
            FROM papers p
            WHERE p.title LIKE ? OR p.keywords LIKE ? OR p.abstract LIKE ?
            ORDER BY p.created_at DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
        papers = [dict(row) for row in cursor.fetchall()]

        # 在知识块中搜索
        cursor.execute("""
            SELECT kc.paper_id, kc.chunk_type, kc.content
            FROM knowledge_chunks kc
            WHERE kc.content LIKE ?
            LIMIT ?
        """, (f"%{query}%", limit))
        chunks = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {"papers": papers, "chunks": chunks}

    def get_all_keywords(self) -> list:
        """获取所有关键词（去重）"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT keywords FROM papers WHERE keywords != ''")
        rows = cursor.fetchall()
        conn.close()
        keywords = set()
        for row in rows:
            if row["keywords"]:
                keywords.update(k.strip() for k in row["keywords"].split(",") if k.strip())
        return sorted(keywords)

    # ---- 分类管理 ----

    def create_category(self, name: str, description: str = "", color: str = "#6366f1") -> dict:
        """创建知识库分类"""
        cat_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO categories (id, name, description, color, created_at) VALUES (?, ?, ?, ?, ?)",
                (cat_id, name, description, color, now))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            raise ValueError(f"分类 '{name}' 已存在")
        conn.close()
        return {"id": cat_id, "name": name, "description": description, "color": color, "created_at": now}

    def list_categories(self) -> list:
        """列出所有分类及论文数量"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.id, c.name, c.description, c.color, c.created_at,
                   COUNT(p.id) as paper_count
            FROM categories c
            LEFT JOIN papers p ON p.category_id = c.id
            GROUP BY c.id
            ORDER BY c.created_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_category(self, cat_id: str) -> Optional[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM categories WHERE id=?", (cat_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_category(self, cat_id: str, name: str = None, description: str = None, color: str = None) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        sets = []
        vals = []
        if name is not None:
            sets.append("name=?"); vals.append(name)
        if description is not None:
            sets.append("description=?"); vals.append(description)
        if color is not None:
            sets.append("color=?"); vals.append(color)
        if not sets:
            conn.close(); return False
        vals.append(cat_id)
        cursor.execute(f"UPDATE categories SET {','.join(sets)} WHERE id=?", vals)
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    def delete_category(self, cat_id: str) -> bool:
        """删除分类（论文的 category_id 置空，不删论文）"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("UPDATE papers SET category_id=NULL WHERE category_id=?", (cat_id,))
        cursor.execute("DELETE FROM categories WHERE id=?", (cat_id,))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    def set_paper_category(self, paper_id: str, category_id: str = None) -> bool:
        """设置论文的分类"""
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("UPDATE papers SET category_id=?, updated_at=? WHERE id=?", (category_id, now, paper_id))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    def get_papers_by_category(self, cat_id: str) -> list:
        """获取某分类下所有论文的 ID"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if cat_id == "__uncategorized__":
            cursor.execute("SELECT id FROM papers WHERE category_id IS NULL OR category_id=''")
        else:
            cursor.execute("SELECT id FROM papers WHERE category_id=?", (cat_id,))
        rows = cursor.fetchall()
        conn.close()
        return [row["id"] for row in rows]

    def stats(self) -> dict:
        """知识库统计信息"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM papers")
        paper_count = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) as cnt FROM knowledge_chunks")
        chunk_count = cursor.fetchone()["cnt"]
        conn.close()
        return {
            "paper_count": paper_count,
            "chunk_count": chunk_count,
            "vector_count": self.vector_store.count()
        }

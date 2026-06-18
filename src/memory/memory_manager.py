"""历史记忆管理模块"""

import sqlite3
import json
import os
import uuid
from datetime import datetime
from typing import Optional

from config import MEMORY_DIR


class MemoryManager:
    """对话历史与记忆管理"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(MEMORY_DIR, "memory.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                is_pinned INTEGER DEFAULT 0,
                pinned_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
        """)
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(sessions)").fetchall()]
        if "is_pinned" not in cols:
            cursor.execute("ALTER TABLE sessions ADD COLUMN is_pinned INTEGER DEFAULT 0")
        if "pinned_at" not in cols:
            cursor.execute("ALTER TABLE sessions ADD COLUMN pinned_at TEXT")
        conn.commit()
        conn.close()

    # ---- 会话管理 ----

    def create_session(self, title: str = "新会话") -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, title, is_pinned, pinned_at, created_at, updated_at) VALUES (?, ?, 0, NULL, ?, ?)",
            (session_id, title, now, now)
        )
        conn.commit()
        conn.close()
        return session_id

    def list_sessions(self, limit: int = 20) -> list:
        """列出历史会话"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, s.title, s.is_pinned, s.pinned_at, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM messages WHERE session_id = s.id) as msg_count
            FROM sessions s
            ORDER BY s.is_pinned DESC, COALESCE(s.pinned_at, '') DESC, s.updated_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        """获取会话详情"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_session_title(self, session_id: str, title: str) -> bool:
        """更新会话标题"""
        now = datetime.now().isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
            (title, now, session_id)
        )
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    def set_session_pinned(self, session_id: str, pinned: bool) -> bool:
        """设置会话置顶状态"""
        pinned_at = datetime.now().isoformat() if pinned else None
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET is_pinned=?, pinned_at=? WHERE id=?",
            (1 if pinned else 0, pinned_at, session_id)
        )
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其所有消息和记忆"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        cursor.execute("DELETE FROM memories WHERE session_id=?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    # ---- 消息管理 ----

    @staticmethod
    def _normalize_message_record(msg: dict) -> dict:
        """兼容旧格式消息，避免把模式标签展示在正文里。"""
        metadata = msg.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}

        content = msg.get("content") or ""
        if msg.get("role") == "user" and content.startswith("[RAG] "):
            content = content[6:]
            metadata.setdefault("mode", "rag")

        msg["content"] = content
        msg["metadata"] = metadata or None
        return msg

    def add_message(self, session_id: str, role: str, content: str, metadata: dict = None) -> str:
        """添加消息"""
        msg_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO messages (id, session_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (msg_id, session_id, role, content,
              json.dumps(metadata, ensure_ascii=False) if metadata else None, now))
        # 更新会话时间
        cursor.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
        conn.commit()
        conn.close()
        return msg_id

    def update_message(self, message_id: str, content: str = None, metadata: dict = None) -> bool:
        """更新已有消息内容与元数据"""
        if content is None and metadata is None:
            return False

        now = datetime.now().isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, metadata FROM messages WHERE id=?", (message_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        updates = []
        values = []

        if content is not None:
            updates.append("content=?")
            values.append(content)

        if metadata is not None:
            updates.append("metadata=?")
            values.append(json.dumps(metadata, ensure_ascii=False) if metadata else None)

        values.append(message_id)
        cursor.execute(f"UPDATE messages SET {', '.join(updates)} WHERE id=?", values)
        cursor.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, row["session_id"]))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    def get_messages(self, session_id: str, limit: int = 50) -> list:
        """获取会话消息（按时间正序）"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, role, content, metadata, created_at
            FROM messages WHERE session_id=?
            ORDER BY created_at ASC LIMIT ?
        """, (session_id, limit))
        rows = cursor.fetchall()
        conn.close()
        results = []
        for row in rows:
            msg = dict(row)
            results.append(self._normalize_message_record(msg))
        return results

    def get_context_window(self, session_id: str, max_tokens: int = 4000) -> list:
        """获取最近的消息上下文窗口（用于 LLM 对话）"""
        messages = self.get_messages(session_id, limit=100)
        # 简单估算 token：1 token ≈ 2 个中文字 / 1.3 个英文词
        result = []
        total_chars = 0
        for msg in reversed(messages):
            char_count = len(msg["content"])
            if total_chars + char_count > max_tokens * 2:
                break
            result.insert(0, msg)
            total_chars += char_count
        return result

    # ---- 记忆管理 ----

    def add_memory(self, content: str, category: str = "general",
                   session_id: str = None, importance: int = 5) -> str:
        """添加一条记忆（用户的偏好、重要发现等）"""
        memory_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memories (id, session_id, category, content, importance, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (memory_id, session_id, category, content, importance, now))
        conn.commit()
        conn.close()
        return memory_id

    def get_memories(self, category: str = None, limit: int = 20) -> list:
        """获取记忆"""
        conn = self._get_conn()
        cursor = conn.cursor()
        if category:
            cursor.execute("""
                SELECT * FROM memories WHERE category=? ORDER BY importance DESC, created_at DESC LIMIT ?
            """, (category, limit))
        else:
            cursor.execute("""
                SELECT * FROM memories ORDER BY importance DESC, created_at DESC LIMIT ?
            """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def search_memories(self, query: str, limit: int = 10) -> list:
        """搜索记忆"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM memories WHERE content LIKE ?
            ORDER BY importance DESC LIMIT ?
        """, (f"%{query}%", limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0

    # ---- 用户偏好 ----

    def set_preference(self, key: str, value: str):
        """设置用户偏好"""
        now = datetime.now().isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?
        """, (key, value, now, value, now))
        conn.commit()
        conn.close()

    def get_preference(self, key: str, default: str = None) -> Optional[str]:
        """获取用户偏好"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM user_preferences WHERE key=?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row["value"] if row else default

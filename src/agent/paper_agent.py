"""核心 Agent 模块 - 论文拆解 Agent + RAG 检索"""

import json
import os
import re
import threading
import uuid
from datetime import datetime
from typing import Optional

from agent.paper_parser import extract_text_from_file, preprocess_text, sanitize_text
from knowledge_base.kb_manager import KnowledgeBase
from knowledge_base.chunker import TextChunker
from knowledge_base.embedder import Embedder
from knowledge_base.vector_store import VectorStore
from memory.memory_manager import MemoryManager
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    DECOMPOSE_PROMPT, KB_QUERY_PROMPT, SYSTEM_PROMPT,
    VECTOR_DB_TYPE, EMBED_METHOD, EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, RAG_TOP_K, RAG_PROMPT
)


class PaperAgent:
    """论文拆解 Agent，集成 LLM + RAG + 知识库 + 记忆"""

    def __init__(self):
        self.kb = KnowledgeBase()
        self.memory = MemoryManager()
        self.session_id = None
        self._llm_client = None
        self._generation_lock = threading.Lock()
        self._generation_states = {}
        # RAG 组件
        self._embedder = None
        self._vector_store = None
        self._chunker = None

    @property
    def llm(self):
        """懒加载 LLM 客户端"""
        if self._llm_client is None:
            self._llm_client = LLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL)
        return self._llm_client

    @property
    def chunker(self):
        if self._chunker is None:
            self._chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return self._chunker

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = Embedder(method=EMBED_METHOD, model_name=EMBED_MODEL)
        return self._embedder

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = VectorStore(
                db_type=VECTOR_DB_TYPE,
                embedder=self.embedder if EMBED_METHOD != "chroma" else None
            )
        return self._vector_store

    # ---- 会话 ----

    def new_session(self, title: str = "新会话") -> str:
        """创建新会话"""
        self.session_id = self.memory.create_session(title)
        return self.session_id

    def load_session(self, session_id: str):
        """加载已有会话"""
        session = self.memory.get_session(session_id)
        if session:
            self.session_id = session_id
            return session
        raise ValueError(f"会话不存在: {session_id}")

    @staticmethod
    def _build_session_title(text: str, max_len: int = 48) -> str:
        """根据首轮对话摘要生成一个简洁标题。"""
        text = re.sub(r"\s+", " ", (text or "").strip())
        text = re.sub(r"^[#>*`\-\[\]()【】]+", "", text).strip()
        if not text:
            return "新会话"
        for sep in ["。", "？", "！", ".", "?", "!", "：", ":"]:
            idx = text.find(sep)
            if 0 < idx <= max_len:
                text = text[:idx]
                break
        return text[:max_len].rstrip("，,;；:： ") or "新会话"

    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = re.sub(r"```[\s\S]*?```", " ", text or "")
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _is_pending_message(message: dict) -> bool:
        metadata = message.get("metadata") or {}
        return metadata.get("status") == "pending"

    @staticmethod
    def _is_stopped_or_error_message(message: dict) -> bool:
        metadata = message.get("metadata") or {}
        return metadata.get("status") in {"stopped", "error"}

    def _extract_title_from_first_round(self, messages: list[dict]) -> str:
        """基于首轮完整对话内容生成会话标题。"""
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        first_assistant = next(
            (
                m for m in messages
                if m.get("role") == "assistant"
                and not self._is_pending_message(m)
                and not self._is_stopped_or_error_message(m)
            ),
            None
        )
        if not first_user and not first_assistant:
            return "新会话"

        assistant_content = (first_assistant or {}).get("content") or ""
        match = re.search(r"##\s*论文拆解完成[:：]\s*(.+)", assistant_content)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip() or "新会话"

        user_text = self._strip_markdown((first_user or {}).get("content") or "")[:400]
        assistant_text = self._strip_markdown(assistant_content)[:900]
        prompt = (
            "请根据下面这轮对话内容，生成一个准确、简洁的中文会话标题。\n"
            "要求：\n"
            "1. 标题要概括对话主题，而不是照抄原句。\n"
            "2. 控制在 8-24 个中文字符内。\n"
            "3. 不要使用引号、句号、冒号、编号。\n"
            "4. 只输出标题本身。\n\n"
            f"用户内容：{user_text or '（无）'}\n"
            f"助手内容：{assistant_text or '（无）'}"
        )
        try:
            title = self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是一个擅长总结主题的助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=40
            )
            title = self._build_session_title(self._strip_markdown(title), max_len=24)
            return title or "新会话"
        except Exception:
            fallback = assistant_text or user_text
            return self._build_session_title(fallback, max_len=24)

    def _maybe_auto_title_session(self, session_id: str = None):
        """在首轮对话后自动为默认会话命名。"""
        sid = session_id or self.session_id
        if not sid:
            return
        session = self.memory.get_session(sid)
        if not session:
            return
        current_title = (session.get("title") or "").strip()
        if current_title not in {"新会话", "Web 会话"} and not current_title.startswith("分析:"):
            return
        messages = self.memory.get_messages(sid, limit=10)
        assistant_messages = [
            m for m in messages
            if m.get("role") == "assistant"
            and not self._is_pending_message(m)
            and not self._is_stopped_or_error_message(m)
        ]
        if not assistant_messages:
            return
        new_title = self._extract_title_from_first_round(messages)
        if new_title and new_title != current_title:
            self.memory.update_session_title(sid, new_title)

    def _register_generation(self, message_id: str, session_id: str, task: str, metadata: dict):
        with self._generation_lock:
            self._generation_states[message_id] = {
                "session_id": session_id,
                "task": task,
                "metadata": dict(metadata or {}),
                "cancelled": False,
            }

    def _peek_generation(self, message_id: str) -> Optional[dict]:
        with self._generation_lock:
            state = self._generation_states.get(message_id)
            return dict(state) if state else None

    def _take_generation(self, message_id: str) -> Optional[dict]:
        with self._generation_lock:
            state = self._generation_states.pop(message_id, None)
            return dict(state) if state else None

    def _generation_cancelled(self, message_id: str) -> bool:
        state = self._peek_generation(message_id)
        return bool(state and state.get("cancelled"))

    def _next_revision_index(self, session_id: str, revision_group_id: str) -> int:
        messages = self.memory.get_messages(session_id, limit=1000)
        max_index = 0
        for msg in messages:
            meta = msg.get("metadata") or {}
            if meta.get("revision_group_id") == revision_group_id:
                try:
                    max_index = max(max_index, int(meta.get("revision_index") or 0))
                except (TypeError, ValueError):
                    continue
        return max_index + 1

    def _start_chat_log(self, user_input: str, session_id: str = None, mode: str = "chat",
                        revision_group_id: str = None, category_id: str = None) -> tuple[str, str, str, int]:
        """先写入用户消息和助手占位消息，避免切换/刷新时丢失中间态。"""
        sid = session_id or self.session_id
        if not sid:
            sid = self.new_session()

        revision_group_id = revision_group_id or str(uuid.uuid4())
        revision_index = self._next_revision_index(sid, revision_group_id)
        turn_id = str(uuid.uuid4())
        base_metadata = {
            "task": mode,
            "mode": mode,
            "turn_id": turn_id,
            "revision_group_id": revision_group_id,
            "revision_index": revision_index,
        }
        if category_id:
            base_metadata["category_id"] = category_id

        user_metadata = dict(base_metadata)
        self.memory.add_message(sid, "user", user_input, user_metadata)
        assistant_placeholder = "正在思考中..." if mode == "chat" else "正在检索知识库并组织回答..."
        assistant_metadata = dict(base_metadata)
        assistant_metadata.update({
            "status": "pending",
            "started_at": datetime.now().isoformat()
        })
        assistant_message_id = self.memory.add_message(sid, "assistant", assistant_placeholder, assistant_metadata)
        self._register_generation(assistant_message_id, sid, mode, assistant_metadata)
        return sid, assistant_message_id, revision_group_id, revision_index

    def _finish_chat_log(self, assistant_message_id: str, response: str, mode: str = "chat",
                         extra_metadata: dict = None):
        state = self._take_generation(assistant_message_id)
        if state and state.get("cancelled"):
            return
        metadata = dict((state or {}).get("metadata") or {})
        metadata.update({"task": mode, "mode": mode, "status": "done"})
        if extra_metadata:
            metadata.update(extra_metadata)
        self.memory.update_message(assistant_message_id, content=response, metadata=metadata)

    def _fail_chat_log(self, assistant_message_id: str, error: str, mode: str = "chat"):
        state = self._take_generation(assistant_message_id)
        if state and state.get("cancelled"):
            return
        metadata = dict((state or {}).get("metadata") or {})
        metadata.update({"task": mode, "mode": mode, "status": "error", "error": error})
        self.memory.update_message(assistant_message_id, content=f"处理失败: {error}", metadata=metadata)

    def stop_session_generation(self, session_id: str, revision_group_id: str = None) -> bool:
        stopped = False
        updates = []
        with self._generation_lock:
            for message_id, state in self._generation_states.items():
                meta = state.get("metadata") or {}
                if state.get("session_id") != session_id:
                    continue
                if revision_group_id and meta.get("revision_group_id") != revision_group_id:
                    continue
                state["cancelled"] = True
                updates.append((message_id, dict(meta), state.get("task")))

        for message_id, meta, task in updates:
            meta.update({"status": "stopped"})
            text = "已停止论文拆解。" if task == "decompose" else "已停止生成。"
            self.memory.update_message(message_id, content=text, metadata=meta)
            stopped = True
        return stopped

    def regenerate_turn(self, session_id: str, revision_group_id: str, user_input: str,
                        use_rag: bool = False, category_id: str = None) -> dict:
        self.stop_session_generation(session_id)
        if use_rag:
            return self.rag_query(
                user_input,
                category_id=category_id,
                session_id=session_id,
                revision_group_id=revision_group_id
            )
        return {
            "answer": self.chat(
                user_input,
                session_id=session_id,
                revision_group_id=revision_group_id
            )
        }

    # ---- 论文拆解（核心功能）----

    def _start_decompose_log(self, file_path: str, session_id: str = None, title_override: str = None) -> tuple[str, str]:
        """为拆解任务创建一个持久化的进行中消息，便于刷新后继续看到状态。"""
        sid = session_id or self.session_id
        if not sid:
            sid = self.new_session(title=f"分析: {os.path.basename(file_path)}")

        display_name = (title_override or "").strip() or os.path.basename(file_path)
        content = f"正在解析 {display_name} 并进行 AI 拆解..."
        metadata = {
            "task": "decompose",
            "status": "pending",
            "file_name": os.path.basename(file_path),
            "display_name": display_name,
            "started_at": datetime.now().isoformat()
        }
        message_id = self.memory.add_message(sid, "assistant", content, metadata)
        self._register_generation(message_id, sid, "decompose", metadata)
        return sid, message_id

    def _log_decompose_error(self, file_path: str, error: str, session_id: str = None,
                             message_id: str = None, title_override: str = None):
        """将拆解失败状态写回会话历史。"""
        sid = session_id or self.session_id
        if not sid:
            sid = self.new_session(title=f"分析: {os.path.basename(file_path)}")

        display_name = (title_override or "").strip() or os.path.basename(file_path)
        content = f"论文拆解失败: {error}"
        metadata = {
            "task": "decompose",
            "status": "error",
            "file_name": os.path.basename(file_path),
            "display_name": display_name,
            "error": error,
        }

        if message_id:
            state = self._take_generation(message_id)
            if state and state.get("cancelled"):
                return
            if state:
                metadata = {**(state.get("metadata") or {}), **metadata}
            self.memory.update_message(message_id, content=content, metadata=metadata)
        else:
            self.memory.add_message(sid, "assistant", content, metadata)

    def decompose_paper(self, file_path: str, save_to_kb: bool = True,
                        title_override: str = None, category_id: str = None,
                        session_id: str = None) -> dict:
        """
        拆解论文：解析文件 -> LLM 分析 -> 结构化输出
        """
        sid, progress_message_id = self._start_decompose_log(
            file_path, session_id=session_id, title_override=title_override
        )
        try:
            print(f"[RAG] 正在解析文件: {os.path.basename(file_path)}")
            raw_text = extract_text_from_file(file_path)
            if self._generation_cancelled(progress_message_id):
                self._take_generation(progress_message_id)
                return {"error": "已停止论文拆解"}
            raw_text = preprocess_text(raw_text)
            print(f"   提取文本: {len(raw_text):,} 字符")

            print("[RAG] 正在调用 LLM 进行论文拆解分析...")
            decomposed = self._call_decompose_llm(raw_text)
            if self._generation_cancelled(progress_message_id):
                self._take_generation(progress_message_id)
                return {"error": "已停止论文拆解"}

            if decomposed is None:
                error = "LLM 分析失败，请检查 API 配置"
                self._log_decompose_error(
                    file_path, error, session_id=sid,
                    message_id=progress_message_id, title_override=title_override
                )
                return {"error": error}

            if title_override and title_override.strip():
                decomposed["title"] = title_override.strip()

            # 保存到知识库
            paper_id = None
            if save_to_kb:
                paper_id = self._save_to_knowledge_base(
                    file_path, raw_text, decomposed, category_id=category_id
                )

            # 记录到对话历史
            self._log_decompose(
                file_path, decomposed, paper_id,
                session_id=sid, message_id=progress_message_id
            )
            self._maybe_auto_title_session(session_id=sid)

            # 提取关键记忆
            self._extract_memories(decomposed)

            return {
                "paper_id": paper_id,
                "file": os.path.basename(file_path),
                "decomposed": decomposed
            }
        except Exception as e:
            self._log_decompose_error(
                file_path, str(e), session_id=sid,
                message_id=progress_message_id, title_override=title_override
            )
            raise

    def _call_decompose_llm(self, text: str) -> Optional[dict]:
        """调用 LLM 进行论文拆解"""
        prompt = DECOMPOSE_PROMPT.format(text=text[:25000])  # 限制 token
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": "你是一个专业的学术论文拆解助手，请始终返回合法的 JSON 格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return self._parse_json_response(response)

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """尝试从 LLM 响应中解析 JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        import re
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取第一个 { ... } 块
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        print(f"[警告] 无法解析 LLM 返回的 JSON，原始响应前200字符: {text[:200]}")
        return None

    def _save_to_knowledge_base(self, file_path: str, raw_text: str, decomposed: dict,
                                category_id: str = None) -> str:
        """保存拆解结果到知识库"""
        paper_id = self.kb.add_paper(
            title=decomposed.get("title", os.path.basename(file_path)),
            authors=decomposed.get("authors", ""),
            file_path=file_path,
            raw_text=raw_text,
            decomposed=decomposed,
            keywords=decomposed.get("keywords", []),
            category_id=category_id,
        )

        # 将各部分保存为知识块
        chunks = []
        section_summary = decomposed.get("section_summary", {})
        for section_name, content in section_summary.items():
            if content:
                chunks.append({"type": section_name, "content": content})

        # 添加其他关键信息块
        for key in ["research_question", "contributions", "key_findings", "limitations", "future_work"]:
            val = decomposed.get(key)
            if val:
                content = val if isinstance(val, str) else "; ".join(val)
                chunks.append({"type": key, "content": content})

        methodology = decomposed.get("methodology", {})
        if methodology:
            chunks.append({"type": "methodology", "content": json.dumps(methodology, ensure_ascii=False)})

        if chunks:
            self.kb.add_chunks(paper_id, chunks)

        print(f"[RAG] 已保存到知识库, 论文ID: {paper_id}")
        return paper_id

    def _log_decompose(self, file_path: str, decomposed: dict, paper_id: str,
                       session_id: str = None, message_id: str = None):
        """记录到对话历史"""
        sid = session_id or self.session_id
        if not sid:
            sid = self.new_session(title=f"分析: {os.path.basename(file_path)}")

        summary = (
            f"## 论文拆解完成: {decomposed.get('title', '未知标题')}\n\n"
            f"**作者**: {decomposed.get('authors', '未知')}\n"
            f"**关键词**: {', '.join(decomposed.get('keywords', []))}\n"
            f"**核心研究问题**: {decomposed.get('research_question', '无')}\n\n"
            f"**主要贡献**:\n"
        )
        for c in decomposed.get("contributions", []):
            summary += f"- {c}\n"

        summary += f"\n**关键发现**:\n"
        for f in decomposed.get("key_findings", []):
            summary += f"- {f}\n"

        summary += f"\n**总体评价**: {decomposed.get('overall_assessment', '无')}"

        metadata = {
            "task": "decompose",
            "status": "done",
            "paper_id": paper_id,
            "file_name": os.path.basename(file_path),
            "title": decomposed.get("title", "未知标题")
        }
        if message_id:
            state = self._take_generation(message_id)
            if state and state.get("cancelled"):
                return
            if state:
                metadata = {**(state.get("metadata") or {}), **metadata}
            self.memory.update_message(message_id, content=summary, metadata=metadata)
        else:
            self.memory.add_message(sid, "assistant", summary, metadata)

    def _extract_memories(self, decomposed: dict):
        """从拆解结果中提取关键记忆"""
        # 记住研究方法和关键词
        methodology = decomposed.get("methodology", {})
        if methodology.get("name"):
            self.memory.add_memory(
                content=f"论文《{decomposed.get('title', '')}》使用了 {methodology['name']} 方法",
                category="methodology",
                importance=7
            )

        for kw in decomposed.get("keywords", [])[:5]:
            self.memory.add_memory(
                content=f"关键词关联: {kw} - 出现在论文《{decomposed.get('title', '')}》",
                category="keyword",
                importance=5
            )

    # ---- 知识库查询 ----

    def query_knowledge_base(self, question: str) -> str:
        """基于知识库回答问题"""
        print(f"[RAG] 搜索知识库: {question}")
        results = self.kb.search(question, limit=5)

        if not results["papers"] and not results["chunks"]:
            answer = "知识库中未找到相关论文。请先使用 `decompose` 命令拆解论文。"
        else:
            # 构建上下文
            context_parts = []
            for paper in results["papers"]:
                context_parts.append(
                    f"论文: {paper['title']}\n作者: {paper['authors']}\n摘要: {paper['abstract']}"
                )
            for chunk in results["chunks"]:
                context_parts.append(
                    f"[{chunk['chunk_type']}] {chunk['content']}"
                )
            context = "\n\n".join(context_parts)

            # 使用 LLM 生成回答
            prompt = KB_QUERY_PROMPT.format(context=context, question=question)
            answer = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

        # 记录到历史
        if self.session_id:
            self.memory.add_message(self.session_id, "user", question)
            self.memory.add_message(self.session_id, "assistant", answer)
            self._maybe_auto_title_session()

        return answer

    # ---- 对话 ----

    def chat(self, user_input: str, session_id: str = None, revision_group_id: str = None) -> str:
        """普通对话（带上下文）"""
        sid, assistant_message_id, _, _ = self._start_chat_log(
            user_input,
            session_id=session_id,
            mode="chat",
            revision_group_id=revision_group_id
        )

        try:
            # 构建消息列表
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # 加载相关记忆作为上下文补充
            memories = self.memory.get_memories(limit=5)
            if memories:
                memory_text = "以下是用户的历史偏好和相关记忆：\n"
                for m in memories:
                    memory_text += f"- [{m['category']}] {m['content']}\n"
                messages.append({"role": "system", "content": memory_text})

            # 加载历史消息，跳过当前占位中的助手消息
            history = self.memory.get_context_window(sid)
            for msg in history:
                if self._is_pending_message(msg):
                    continue
                messages.append({"role": msg["role"], "content": msg["content"]})

            response = self.llm.chat(messages=messages)
            self._finish_chat_log(assistant_message_id, response, mode="chat")
            self._maybe_auto_title_session(session_id=sid)
            return response
        except Exception as e:
            self._fail_chat_log(assistant_message_id, str(e), mode="chat")
            raise

    # ---- RAG 索引 (无需 LLM) ----

    def index_paper(self, file_path: str, category_id: str = None, display_name: str = None) -> dict:
        """
        索引论文: 提取文本 → 分块 → Embedding → 向量库
        不调用 LLM，纯本地处理
        """
        filename = (display_name or "").strip() or os.path.basename(file_path)
        print(f"[RAG] 正在索引: {filename}")

        # 1. 提取文本
        try:
            raw_text = extract_text_from_file(file_path)
        except Exception as e:
            return {"file": filename, "status": "error", "reason": f"文本提取失败: {e}"}

        raw_text = preprocess_text(raw_text)
        print(f"   提取文本: {len(raw_text):,} 字符")

        # 2. 分块
        doc_chunks = self.chunker.chunk_document(raw_text, paper_title=filename)
        print(f"   分块完成: {len(doc_chunks)} 个文本块")

        if not doc_chunks:
            return {"file": filename, "status": "error", "reason": "分块结果为空"}

        # 3. 生成向量并存储
        paper_id = str(uuid.uuid4())
        vector_chunks = []
        for dc in doc_chunks:
            chunk_id = str(uuid.uuid4())
            vector_chunks.append({
                "id": chunk_id,
                "paper_id": paper_id,
                "content": dc["content"],
                "chunk_type": "rag_chunk",
            })

        # 同时写入 SQLite 知识库
        try:
            self.kb.add_paper(
                title=filename,
                file_path=file_path,
                raw_text=raw_text,
                category_id=category_id,
                paper_id=paper_id,
            )
        except Exception:
            pass

        # 写入向量库
        embeddings = None
        if EMBED_METHOD != "chroma":
            print(f"   正在生成 Embedding ({EMBED_METHOD})...")
            try:
                embeddings = self.embedder.embed_texts([c["content"] for c in vector_chunks])
                print(f"   Embedding 完成: {len(embeddings)} 条")
            except Exception as e:
                return {"file": filename, "status": "error", "reason": f"Embedding 失败: {e}"}

        try:
            self.vector_store.add_chunks(vector_chunks, embeddings)
        except Exception as e:
            return {"file": filename, "status": "error", "reason": f"向量库写入失败: {e}"}

        print(f"[RAG] 索引完成, 向量数: {self.vector_store.count()}")
        return {
            "file": filename,
            "status": "ok",
            "paper_id": paper_id,
            "chunks": len(doc_chunks),
        }

    def batch_index(self, file_paths: list[str]) -> dict:
        """批量索引论文"""
        results = []
        for fp in file_paths:
            try:
                result = self.index_paper(fp)
                results.append(result)
            except Exception as e:
                results.append({"file": os.path.basename(fp), "status": "error", "reason": str(e)})

        success = sum(1 for r in results if r["status"] == "ok")
        return {"total": len(results), "success": success, "results": results}

    # ---- RAG 查询 ----

    def rag_query(self, question: str, top_k: int = None,
                  category_id: str = None, session_id: str = None,
                  revision_group_id: str = None) -> dict:
        """
        RAG 查询: Embedding 检索 → 构建 Context → LLM 回答
        Args:
            question: 用户问题
            top_k: 检索返回数量
            category_id: 限定检索的分类 ID
        返回: {"answer": str, "sources": list}
        """
        top_k = top_k or RAG_TOP_K

        # 如果指定了分类，获取该分类下所有论文 ID
        paper_ids = None
        if category_id:
            paper_ids = self.kb.get_papers_by_category(category_id)

        print(f"[RAG] 查询: {question}  (分类: {category_id or '全部'})")

        if self._is_catalog_query(question):
            return self._answer_catalog_query(
                question,
                category_id,
                session_id=session_id,
                revision_group_id=revision_group_id
            )

        sid, assistant_message_id, _, _ = self._start_chat_log(
            question,
            session_id=session_id,
            mode="rag",
            revision_group_id=revision_group_id,
            category_id=category_id
        )

        # 1. 向量检索（可按 paper_id 过滤）
        try:
            if paper_ids is not None:
                if not paper_ids:
                    chunks = []
                else:
                    all_chunks = []
                    for pid in paper_ids:
                        found = self.vector_store.search(question, n_results=top_k, paper_id=pid)
                        all_chunks.extend(found)
                    all_chunks.sort(key=lambda x: x.get("distance", 0), reverse=True)
                    chunks = all_chunks[:top_k]
            else:
                all_chunks = []
                chunks = self.vector_store.search(question, n_results=top_k)
        except Exception as e:
            self._fail_chat_log(assistant_message_id, f"检索出错: {e}", mode="rag")
            return {"answer": f"检索出错: {e}", "sources": []}

        if not chunks:
            answer = "知识库中未找到相关内容。请先导入论文建立索引。"
            self._finish_chat_log(assistant_message_id, answer, mode="rag")
            self._maybe_auto_title_session(session_id=sid)
            return {"answer": answer, "sources": []}

        # 2. 构建 context
        context_parts = []
        sources = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[片段{i + 1}] (相似度: {chunk.get('distance', 'N/A')})\n{chunk['content']}"
            )
            sources.append({
                "paper_title": chunk.get("paper_title", ""),
                "chunk_type": chunk.get("chunk_type", ""),
                "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                "score": chunk.get("distance", 0),
            })

        context = "\n\n---\n\n".join(context_parts)
        print(f"   检索到 {len(chunks)} 个相关片段")

        # 3. LLM 生成回答
        prompt = RAG_PROMPT.format(context=context, question=question)
        try:
            answer = self.llm.chat(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            self._finish_chat_log(
                assistant_message_id,
                answer,
                mode="rag",
                extra_metadata={"sources": sources}
            )
            self._maybe_auto_title_session(session_id=sid)
            return {"answer": answer, "sources": sources}
        except Exception as e:
            self._fail_chat_log(assistant_message_id, str(e), mode="rag")
            raise

    def _is_catalog_query(self, question: str) -> bool:
        """识别询问知识库论文数量/标题列表的目录型问题。"""
        q = re.sub(r"\s+", "", question.lower())
        patterns = [
            r"知识库.*几篇论文",
            r"知识库.*多少篇论文",
            r"知识库.*哪些论文",
            r"知识库.*标题",
            r"几篇论文",
            r"多少篇论文",
            r"论文数量",
            r"论文总数",
            r"论文标题",
            r"标题分别是",
            r"有哪些论文",
            r"列出.*论文",
            r"论文清单",
        ]
        return any(re.search(pattern, q) for pattern in patterns)

    def _answer_catalog_query(self, question: str, category_id: str = None,
                              session_id: str = None, revision_group_id: str = None) -> dict:
        """直接基于论文元数据回答知识库盘点问题。"""
        sid, assistant_message_id, _, _ = self._start_chat_log(
            question,
            session_id=session_id,
            mode="rag",
            revision_group_id=revision_group_id,
            category_id=category_id
        )
        papers = self.kb.list_papers(limit=1000, category_id=category_id)
        if category_id == "__uncategorized__":
            category = None
            scope_name = "未分类知识库"
        else:
            category = self.kb.get_category(category_id) if category_id else None
            scope_name = f"分类“{category['name']}”" if category else "知识库"

        if not papers:
            answer = f"{scope_name}中当前没有论文。"
            sources = []
        else:
            lines = [f"{scope_name}中当前共有 {len(papers)} 篇论文。", "", "标题如下："]
            sources = []
            for idx, paper in enumerate(papers, 1):
                title = (paper.get("title") or "未命名论文").strip()
                authors = (paper.get("authors") or "").strip()
                created_at = (paper.get("created_at") or "")[:19]
                if authors:
                    lines.append(f"{idx}. {title}（作者：{authors}）")
                else:
                    lines.append(f"{idx}. {title}")
                sources.append({
                    "content": f"{title} | 添加时间: {created_at}" if created_at else title,
                    "score": 1.0,
                })
            answer = "\n".join(lines)

        self._finish_chat_log(
            assistant_message_id,
            answer,
            mode="rag",
            extra_metadata={"sources": sources}
        )
        self._maybe_auto_title_session(session_id=sid)
        return {"answer": answer, "sources": sources}

    # ---- RAG 配置 ----

    def get_rag_config(self) -> dict:
        """获取当前 RAG 配置"""
        try:
            vs_info = self.vector_store.info
        except Exception:
            vs_info = {"db_type": VECTOR_DB_TYPE, "count": 0, "embed_method": EMBED_METHOD}
        return {
            "vector_db_type": VECTOR_DB_TYPE,
            "embed_method": EMBED_METHOD,
            "embed_model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "rag_top_k": RAG_TOP_K,
            "vector_count": vs_info.get("count", 0),
        }

    # ---- 会话管理 ----

    def list_sessions(self) -> list:
        return self.memory.list_sessions()

    def rename_session(self, session_id: str, title: str) -> bool:
        return self.memory.update_session_title(session_id, title)

    def pin_session(self, session_id: str, pinned: bool) -> bool:
        return self.memory.set_session_pinned(session_id, pinned)

    def delete_session(self, session_id: str) -> bool:
        deleted = self.memory.delete_session(session_id)
        if deleted and self.session_id == session_id:
            self.session_id = None
        return deleted

    def list_papers(self, limit: int = 20, category_id: str = None) -> list:
        return self.kb.list_papers(limit=limit, category_id=category_id)

    def list_categories(self) -> list:
        return self.kb.list_categories()

    def create_category(self, name: str, description: str = "", color: str = "#6366f1") -> dict:
        return self.kb.create_category(name, description, color)

    def update_category(self, cat_id: str, **kwargs) -> bool:
        return self.kb.update_category(cat_id, **kwargs)

    def delete_category(self, cat_id: str) -> bool:
        return self.kb.delete_category(cat_id)

    def set_paper_category(self, paper_id: str, category_id: str = None) -> bool:
        return self.kb.set_paper_category(paper_id, category_id)

    def get_paper_detail(self, paper_id: str) -> Optional[dict]:
        return self.kb.get_paper(paper_id)

    def show_history(self, session_id: str = None) -> list:
        sid = session_id or self.session_id
        if not sid:
            return []
        return self.memory.get_messages(sid)

    def show_memories(self) -> list:
        return self.memory.get_memories()

    def show_kb_stats(self) -> dict:
        return self.kb.stats()


class LLMClient:
    """LLM API 客户端，支持 OpenAI 兼容接口"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = None

    @property
    def client(self):
        if not self.api_key:
            raise RuntimeError("缺少 LLM_API_KEY，请在 .env 文件或环境变量中配置后再运行")
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("请安装 openai 库: pip install openai")
        return self._client

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """发送聊天请求"""
        try:
            clean_messages = []
            for message in messages:
                clean_message = dict(message)
                if isinstance(clean_message.get("content"), str):
                    clean_message["content"] = sanitize_text(clean_message["content"])
                clean_messages.append(clean_message)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=clean_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return sanitize_text(response.choices[0].message.content.strip())
        except Exception as e:
            return f"[LLM 调用错误] {e}"

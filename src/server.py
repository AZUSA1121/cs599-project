"""
PaperAgent Web Server - FastAPI 后端
启动: python server.py
访问: http://localhost:8000
"""

import os
import sys
import uuid
import tempfile

# 修复 Windows GBK 编码导致 print 中文/emoji 崩溃的问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from typing import List
from agent.paper_agent import PaperAgent

app = FastAPI(title="PaperAgent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 Agent 实例
agent = PaperAgent()

# 静态文件
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


# ─── 页面路由 ───

@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ─── 论文拆解 ───

@app.post("/api/decompose")
async def decompose_paper(
    file: UploadFile = File(...),
    title: str = Form(None),
    category_id: str = Form(None),
    session_id: str = Form(None)
):
    """上传并拆解论文"""
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "仅支持 PDF / TXT / MD 文件")

    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{suffix}")
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        result = await run_in_threadpool(
            agent.decompose_paper,
            tmp_path,
            True,
            (title or "").strip() or None,
            (category_id or "").strip() or None,
            (session_id or "").strip() or None,
        )
        if "error" in result:
            raise HTTPException(500, result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/batch-decompose")
async def batch_decompose_papers(files: List[UploadFile] = File(...)):
    """批量上传并拆解论文"""
    results = []
    for file in files:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in (".pdf", ".txt", ".md"):
            results.append({"file": file.filename, "status": "skipped", "reason": "不支持的格式"})
            continue

        tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{suffix}")
        try:
            content = await file.read()
            with open(tmp_path, "wb") as f:
                f.write(content)

            result = agent.decompose_paper(tmp_path, save_to_kb=True)
            if "error" in result:
                results.append({"file": file.filename, "status": "error", "reason": result["error"]})
            else:
                results.append({
                    "file": file.filename,
                    "status": "ok",
                    "paper_id": result.get("paper_id"),
                    "title": result.get("decomposed", {}).get("title", file.filename)
                })
        except Exception as e:
            results.append({"file": file.filename, "status": "error", "reason": str(e)})
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    success = sum(1 for r in results if r["status"] == "ok")
    return {"total": len(results), "success": success, "results": results}


@app.post("/api/decompose-dir")
async def decompose_directory(body: dict):
    """批量拆解指定目录下的所有论文"""
    dir_path = (body.get("dir_path") or "").strip()
    if not dir_path or not os.path.isdir(dir_path):
        raise HTTPException(400, "目录不存在或路径为空")

    supported = {".pdf", ".txt", ".md"}
    files = [
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not files:
        raise HTTPException(400, "目录下没有 PDF/TXT/MD 文件")

    results = []
    for fname in files:
        file_path = os.path.join(dir_path, fname)
        try:
            result = agent.decompose_paper(file_path, save_to_kb=True)
            if "error" in result:
                results.append({"file": fname, "status": "error", "reason": result["error"]})
            else:
                results.append({
                    "file": fname,
                    "status": "ok",
                    "paper_id": result.get("paper_id"),
                    "title": result.get("decomposed", {}).get("title", fname)
                })
        except Exception as e:
            results.append({"file": fname, "status": "error", "reason": str(e)})

    success = sum(1 for r in results if r["status"] == "ok")
    return {"total": len(results), "success": success, "results": results}


# ─── 知识库 ───

@app.get("/api/papers")
async def list_papers(limit: int = 50, category_id: str = None):
    return agent.list_papers(limit=limit, category_id=category_id)

@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: str):
    paper = agent.get_paper_detail(paper_id)
    if not paper:
        raise HTTPException(404, "论文不存在")
    return paper

@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: str):
    if agent.kb.delete_paper(paper_id):
        return {"ok": True}
    raise HTTPException(404, "论文不存在")

@app.get("/api/search")
async def search_kb(q: str = "", limit: int = 10):
    if not q.strip():
        return {"papers": [], "chunks": []}
    return agent.kb.search(q, limit=limit)

@app.get("/api/keywords")
async def get_keywords():
    return agent.kb.get_all_keywords()


# ─── 知识库分类 ───

@app.get("/api/categories")
async def list_categories():
    return agent.list_categories()


@app.post("/api/categories")
async def create_category(body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "分类名称不能为空")
    try:
        return agent.create_category(
            name=name,
            description=body.get("description", ""),
            color=body.get("color", "#6366f1")
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.put("/api/categories/{cat_id}")
async def update_category(cat_id: str, body: dict):
    ok = agent.update_category(cat_id,
        name=body.get("name"),
        description=body.get("description"),
        color=body.get("color"))
    if not ok:
        raise HTTPException(404, "分类不存在")
    return {"ok": True}


@app.delete("/api/categories/{cat_id}")
async def delete_category(cat_id: str):
    if not agent.delete_category(cat_id):
        raise HTTPException(404, "分类不存在")
    return {"ok": True}


@app.put("/api/papers/{paper_id}/category")
async def set_paper_category(paper_id: str, body: dict):
    category_id = (body.get("category_id") or "").strip() or None
    ok = agent.set_paper_category(paper_id, category_id)
    if not ok:
        raise HTTPException(404, "论文不存在")
    return {"ok": True}


# ─── 对话 ───

@app.post("/api/chat")
async def chat(message: dict):
    text = message.get("text", "").strip()
    if not text:
        raise HTTPException(400, "消息不能为空")
    use_rag = message.get("use_rag", False)
    category_id = message.get("category_id") or None
    session_id = (message.get("session_id") or "").strip() or None
    if use_rag:
        result = await run_in_threadpool(
            agent.rag_query,
            text,
            None,
            category_id,
            session_id,
        )
        return {"reply": result["answer"], "sources": result.get("sources", [])}
    reply = await run_in_threadpool(agent.chat, text, session_id)
    return {"reply": reply}

@app.post("/api/sessions/{session_id}/stop")
async def stop_session_generation(session_id: str, body: dict = None):
    revision_group_id = ((body or {}).get("revision_group_id") or "").strip() or None
    stopped = agent.stop_session_generation(session_id, revision_group_id=revision_group_id)
    return {"ok": True, "stopped": stopped}

@app.post("/api/messages/{revision_group_id}/edit")
async def edit_message(revision_group_id: str, body: dict):
    session_id = (body.get("session_id") or "").strip()
    text = (body.get("text") or "").strip()
    use_rag = bool(body.get("use_rag"))
    category_id = (body.get("category_id") or "").strip() or None
    if not session_id:
        raise HTTPException(400, "缺少 session_id")
    if not text:
        raise HTTPException(400, "消息不能为空")
    result = await run_in_threadpool(
        agent.regenerate_turn,
        session_id,
        revision_group_id,
        text,
        use_rag,
        category_id,
    )
    return {"reply": result.get("answer", ""), "sources": result.get("sources", [])}


# ─── 会话 ───

@app.get("/api/sessions")
async def list_sessions():
    return agent.list_sessions()

@app.post("/api/sessions")
async def create_session(body: dict = None):
    title = (body or {}).get("title", "新会话")
    sid = agent.new_session(title)
    return {"session_id": sid, "title": title}

@app.post("/api/sessions/{session_id}/load")
async def load_session(session_id: str):
    if agent.memory.get_session(session_id):
        return {"ok": True}
    raise HTTPException(404, f"会话不存在: {session_id}")

@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    return agent.show_history(session_id)

@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, body: dict):
    title = body.get("title")
    pinned = body.get("pinned")

    if title is not None:
        title = title.strip()
        if not title:
            raise HTTPException(400, "会话标题不能为空")
        if not agent.rename_session(session_id, title):
            raise HTTPException(404, "会话不存在")

    if pinned is not None:
        if not agent.pin_session(session_id, bool(pinned)):
            raise HTTPException(404, "会话不存在")

    return {"ok": True}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if agent.delete_session(session_id):
        return {"ok": True}
    raise HTTPException(404, "会话不存在")

@app.get("/api/history")
async def get_history():
    return agent.show_history()


# ─── 记忆 ───

@app.get("/api/memories")
async def get_memories():
    return agent.show_memories()


# ─── RAG 索引 (无需 LLM) ───

@app.post("/api/index")
async def index_paper(file: UploadFile = File(...), category_id: str = Form(None)):
    """索引单篇论文: 提取文本 → 分块 → Embedding → 向量库"""
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, "仅支持 PDF / TXT / MD 文件")

    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{suffix}")
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        result = agent.index_paper(tmp_path, category_id=category_id, display_name=file.filename)
        if result["status"] == "error":
            raise HTTPException(500, result["reason"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/batch-index")
async def batch_index_papers(files: List[UploadFile] = File(...), category_id: str = Form(None)):
    """批量索引论文"""
    upload_jobs = []
    for file in files:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in (".pdf", ".txt", ".md"):
            continue
        tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{suffix}")
        try:
            content = await file.read()
            with open(tmp_path, "wb") as f:
                f.write(content)
            upload_jobs.append((tmp_path, file.filename))
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not upload_jobs:
        raise HTTPException(400, "没有可处理的文件")

    results = []
    for fp, original_name in upload_jobs:
        try:
            result = agent.index_paper(fp, category_id=category_id, display_name=original_name)
            results.append(result)
        except Exception as e:
            results.append({"file": original_name, "status": "error", "reason": str(e)})
        finally:
            if os.path.exists(fp):
                os.remove(fp)

    success = sum(1 for r in results if r["status"] == "ok")
    return {"total": len(results), "success": success, "results": results}


@app.post("/api/index-dir")
async def index_directory(body: dict):
    """索引指定目录下的所有论文"""
    dir_path = (body.get("dir_path") or "").strip()
    category_id = (body.get("category_id") or "").strip() or None
    if not dir_path or not os.path.isdir(dir_path):
        raise HTTPException(400, "目录不存在或路径为空")

    supported = {".pdf", ".txt", ".md"}
    files = [
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in supported
    ]
    if not files:
        raise HTTPException(400, "目录下没有 PDF/TXT/MD 文件")

    file_paths = [os.path.join(dir_path, f) for f in files]
    result = agent.batch_index(file_paths)
    if category_id:
        # 批量设置分类
        for r in result.get("results", []):
            if r.get("status") == "ok" and r.get("paper_id"):
                agent.kb.set_paper_category(r["paper_id"], category_id)
    return result


@app.get("/api/rag-config")
async def get_rag_config():
    """获取 RAG 配置"""
    return agent.get_rag_config()


@app.post("/api/rag-reset")
async def reset_rag():
    """重置向量库"""
    agent.vector_store.reset()
    # 重置缓存的 vector_store 以便下次使用新配置
    agent._vector_store = None
    return {"ok": True}


@app.get("/api/stats")
async def get_stats():
    stats = agent.show_kb_stats()
    try:
        stats["vector_count"] = agent.vector_store.count()
    except Exception:
        stats["vector_count"] = 0
    return stats


# 启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

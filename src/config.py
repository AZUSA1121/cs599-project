"""全局配置"""
import os


def load_env_file(path: str):
    """Load simple KEY=VALUE pairs from .env without making dotenv mandatory."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = load_env_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# LLM 配置 (DeepSeek API，兼容 OpenAI 接口格式)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.siliconflow.cn/v1").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V3.2").strip()

# 数据存储目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")
MEMORY_DIR = os.path.join(DATA_DIR, "memory")

# 确保目录存在
for d in [DATA_DIR, KB_DIR, MEMORY_DIR]:
    os.makedirs(d, exist_ok=True)

# 论文拆解提示词模板
DECOMPOSE_PROMPT = """你是一个专业的学术论文拆解助手。请对以下论文内容进行深度拆解分析，输出以下结构化信息（使用 JSON 格式）：

{{
    "title": "论文标题",
    "authors": "作者列表",
    "abstract": "摘要总结（200字以内）",
    "keywords": ["关键词1", "关键词2", ...],
    "research_question": "核心研究问题",
    "methodology": {{
        "name": "方法名称",
        "description": "方法描述",
        "key_techniques": ["关键技术1", "关键技术2", ...]
    }},
    "contributions": ["贡献1", "贡献2", ...],
    "key_findings": ["发现1", "发现2", ...],
    "limitations": ["局限性1", "局限性2", ...],
    "section_summary": {{
        "introduction": "引言要点",
        "related_work": "相关工作要点",
        "methodology": "方法论要点",
        "experiments": "实验要点",
        "results": "结果要点",
        "conclusion": "结论要点"
    }},
    "future_work": "未来工作方向",
    "overall_assessment": "总体评价（100字以内）"
}}

论文内容如下：

{text}
"""

# 知识库查询提示词
KB_QUERY_PROMPT = """基于以下从知识库中检索到的相关论文信息，回答用户的问题。如果检索结果中没有相关信息，请根据你的知识回答并说明。

检索到的相关信息：
{context}

用户问题：{question}
"""

# 多轮对话系统提示词
SYSTEM_PROMPT = """你是一个专业的学术论文分析助手 "PaperAgent"。你的能力包括：

1. **论文拆解**：对论文进行深度结构化分析，提取关键信息
2. **知识管理**：帮助用户构建和管理本地论文知识库
3. **论文对比**：对比分析多篇论文的异同
4. **问答检索**：基于知识库内容回答问题

请用专业但易懂的语言与用户交流。
"""

# ─── RAG 配置 ───

# 向量数据库类型: "chroma" 或 "faiss"
VECTOR_DB_TYPE = os.environ.get("VECTOR_DB_TYPE", "chroma")

# Embedding 方法: "chroma"(ChromaDB内置), "local"(本地模型), "openai"(API)
EMBED_METHOD = os.environ.get("EMBED_METHOD", "chroma")

# 本地 Embedding 模型名 (EMBED_METHOD=local 时生效)
# 推荐: paraphrase-multilingual-MiniLM-L12-v2 (中英文, 384维)
#       all-MiniLM-L6-v2 (英文, 384维, 更快)
#       BAAI/bge-small-zh-v1.5 (中文专用, 512维)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# 文本分块参数
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

# RAG 检索参数
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

# RAG 检索提示词
RAG_PROMPT = """基于以下从知识库中检索到的相关内容，回答用户的问题。
如果检索结果中没有相关信息，请根据你的知识回答并说明。

--- 检索到的相关内容 ---

{context}

--- 用户问题 ---

{question}

请基于上述内容给出准确、专业的回答。如果引用了具体内容，请标注来源。"""

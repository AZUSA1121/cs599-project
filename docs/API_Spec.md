# API Spec

| 接口 | 方法 | 功能 | 主要输入 | 主要输出 |
|---|---|---|---|---|
| `/` | GET | Web 首页 | 无 | HTML 页面 |
| `/api/decompose` | POST | 上传并拆解论文 | file, title, category_id, session_id | paper_id, decomposed |
| `/api/batch-decompose` | POST | 批量拆解论文 | files | total, success, results |
| `/api/index` | POST | 建立 RAG 索引 | file, category_id | paper_id, chunks |
| `/api/chat` | POST | 普通对话或 RAG 问答 | text, use_rag, category_id, session_id | reply, sources |
| `/api/papers` | GET | 论文列表 | limit, category_id | paper list |
| `/api/papers/{paper_id}` | GET | 论文详情 | paper_id | paper detail |
| `/api/categories` | GET/POST | 分类管理 | name, description, color | category |
| `/api/sessions` | GET/POST | 会话管理 | title | session_id |
| `/api/memories` | GET | 获取长期记忆 | 无 | memories |
| `/api/stats` | GET | 知识库统计 | 无 | paper_count, chunk_count, vector_count |

## 错误处理

- 上传不支持的文件格式时返回 400。
- 论文、分类或会话不存在时返回 404。
- LLM 或向量库调用失败时返回错误信息，并在会话中记录失败状态。

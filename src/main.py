"""
PaperAgent - 论文拆解 Agent
功能：
  1. 论文拆解分析（PDF/TXT）
  2. 本地知识库管理
  3. 对话历史与记忆

使用前请先配置 API Key（环境变量或修改 config.py）：
  set LLM_API_KEY=your-key
  set LLM_BASE_URL=https://api.openai.com/v1
  set LLM_MODEL=gpt-3.5-turbo
"""

import os
import sys
import json

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.paper_agent import PaperAgent


def print_banner():
    print("""
╔═══════════════════════════════════════════════════╗
║           📚 PaperAgent - 论文拆解助手            ║
║                                                   ║
║  功能:                                            ║
║    decompose <文件路径>   拆解论文                 ║
║    query <问题>          查询知识库               ║
║    chat <消息>           自由对话                 ║
║    papers                列出知识库论文           ║
║    paper <id>            查看论文详情             ║
║    sessions              查看历史会话             ║
║    session <id>          加载会话                 ║
║    history               查看当前对话记录         ║
║    memories              查看记忆                 ║
║    stats                 知识库统计               ║
║    help                  帮助                     ║
║    quit                  退出                     ║
╚═══════════════════════════════════════════════════╝
    """)


def print_help():
    print("""
📋 可用命令：

  论文分析：
    decompose <文件路径>     拆解论文（PDF/TXT），自动保存到知识库
    decompose <文件路径> --no-save    拆解但不保存到知识库

  知识库：
    query <问题>            在知识库中搜索并回答
    papers                  列出知识库中的所有论文
    paper <论文ID>          查看论文拆解详情
    keywords                列出所有关键词
    stats                   知识库统计信息

  对话：
    chat <消息>             带记忆的智能对话
    history                 查看当前会话的对话记录

  会话管理：
    sessions                列出历史会话
    session <会话ID>        加载历史会话
    new                     创建新会话

  记忆：
    memories                查看系统自动记录的记忆
    forget <记忆ID>         删除指定记忆

  其他：
    help                    显示帮助
    quit / exit             退出程序
    """)


def main():
    print_banner()

    agent = PaperAgent()

    # 尝试恢复最近会话
    sessions = agent.list_sessions()
    if sessions:
        print(f"📂 发现 {len(sessions)} 个历史会话，已创建新会话。")
        print(f"   使用 'sessions' 查看历史，'session <id>' 恢复。\n")
    agent.new_session("新会话")

    while True:
        try:
            user_input = input("\n🔬 PaperAgent > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # 解析命令
        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit"):
            print("👋 再见！")
            break

        elif cmd == "help":
            print_help()

        elif cmd == "decompose":
            if not arg:
                print("❌ 请指定文件路径，例如: decompose ./paper.pdf")
                continue
            save = "--no-save" not in arg
            file_path = arg.replace(" --no-save", "").strip().strip('"').strip("'")
            if not os.path.isfile(file_path):
                print(f"❌ 文件不存在: {file_path}")
                continue
            result = agent.decompose_paper(file_path, save_to_kb=save)
            if "error" in result:
                print(f"❌ {result['error']}")
            else:
                d = result["decomposed"]
                print(f"\n{'='*60}")
                print(f"📄 论文标题: {d.get('title', '未知')}")
                print(f"👤 作者: {d.get('authors', '未知')}")
                print(f"🏷️  关键词: {', '.join(d.get('keywords', []))}")
                print(f"❓ 核心研究问题: {d.get('research_question', '无')}")
                print(f"💡 主要贡献:")
                for c in d.get("contributions", []):
                    print(f"   - {c}")
                print(f"🔬 关键发现:")
                for f in d.get("key_findings", []):
                    print(f"   - {f}")
                print(f"📝 总体评价: {d.get('overall_assessment', '无')}")
                if result.get("paper_id"):
                    print(f"🗂️  知识库ID: {result['paper_id']}")
                print(f"{'='*60}")

        elif cmd == "query":
            if not arg:
                print("❌ 请输入查询问题，例如: query 注意力机制是怎么工作的")
                continue
            answer = agent.query_knowledge_base(arg)
            print(f"\n💡 {answer}")

        elif cmd == "chat":
            if not arg:
                print("❌ 请输入消息内容")
                continue
            response = agent.chat(arg)
            print(f"\n🤖 {response}")

        elif cmd == "papers":
            papers = agent.list_papers()
            if not papers:
                print("📭 知识库为空，请使用 'decompose' 命令添加论文。")
            else:
                print(f"\n📚 知识库中的论文 ({len(papers)} 篇):")
                print("-" * 80)
                for i, p in enumerate(papers, 1):
                    print(f"  {i}. [{p['id'][:8]}] {p['title']}")
                    if p.get("authors"):
                        print(f"     作者: {p['authors']}")
                    if p.get("keywords"):
                        print(f"     关键词: {p['keywords']}")
                    print(f"     添加时间: {p['created_at'][:19]}")
                    print()

        elif cmd == "paper":
            if not arg:
                print("❌ 请指定论文ID")
                continue
            paper = agent.get_paper_detail(arg.strip())
            if not paper:
                print(f"❌ 未找到论文: {arg}")
            else:
                d = paper.get("decomposed", {})
                print(f"\n📄 论文详情: {paper['title']}")
                print(f"   ID: {paper['id']}")
                print(f"   作者: {paper.get('authors', '未知')}")
                print(f"   文件: {paper.get('file_path', '未知')}")
                print(f"   关键词: {paper.get('keywords', '无')}")
                print(f"\n📋 拆解结果:")
                print(json.dumps(d, ensure_ascii=False, indent=2))

        elif cmd == "keywords":
            keywords = agent.kb.get_all_keywords()
            if keywords:
                print(f"\n🏷️  所有关键词 ({len(keywords)} 个):")
                print(", ".join(keywords))
            else:
                print("📭 暂无关键词")

        elif cmd == "sessions":
            sessions = agent.list_sessions()
            if not sessions:
                print("📭 暂无历史会话")
            else:
                print(f"\n📂 历史会话 ({len(sessions)} 个):")
                for s in sessions:
                    current = " ◀ 当前" if s["id"] == agent.session_id else ""
                    print(f"  [{s['id'][:8]}] {s['title']}")
                    print(f"     消息数: {s['msg_count']} | 创建: {s['created_at'][:19]}{current}")

        elif cmd == "session":
            if not arg:
                print("❌ 请指定会话ID")
                continue
            try:
                agent.load_session(arg.strip())
                print(f"✅ 已加载会话")
            except ValueError as e:
                print(f"❌ {e}")

        elif cmd == "new":
            agent.new_session()
            print("✅ 已创建新会话")

        elif cmd == "history":
            messages = agent.show_history()
            if not messages:
                print("📭 当前会话暂无记录")
            else:
                print(f"\n📜 对话记录 (共 {len(messages)} 条):")
                for msg in messages:
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    content = msg["content"][:200]
                    if len(msg["content"]) > 200:
                        content += "..."
                    print(f"  {role_icon} [{msg['created_at'][:19]}] {content}")

        elif cmd == "memories":
            memories = agent.show_memories()
            if not memories:
                print("📭 暂无记忆记录")
            else:
                print(f"\n🧠 记忆记录 (共 {len(memories)} 条):")
                for m in memories:
                    print(f"  [{m['id'][:8]}] [{m['category']}] ⭐{m['importance']} {m['content']}")

        elif cmd == "forget":
            if not arg:
                print("❌ 请指定记忆ID")
                continue
            if agent.memory.delete_memory(arg.strip()):
                print("✅ 记忆已删除")
            else:
                print("❌ 未找到该记忆")

        elif cmd == "stats":
            stats = agent.show_kb_stats()
            print(f"\n📊 知识库统计:")
            print(f"   论文数量: {stats['paper_count']}")
            print(f"   知识块数: {stats['chunk_count']}")

        else:
            # 未知命令当作 chat 处理
            response = agent.chat(user_input)
            print(f"\n🤖 {response}")


if __name__ == "__main__":
    main()

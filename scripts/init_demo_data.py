"""Initialize demo papers for PaperAgent without calling the LLM.

This script indexes the Markdown files in sample_data/papers into the local
SQLite knowledge base and vector database. It is useful for reproducing a small
RAG demo after cloning the repository.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.paper_agent import PaperAgent


def main():
    agent = PaperAgent()
    sample_dir = ROOT / "sample_data" / "papers"
    files = sorted(sample_dir.glob("*.md"))
    if not files:
        print("No sample papers found.")
        return

    for file_path in files:
        result = agent.index_paper(str(file_path), display_name=file_path.name)
        print(f"{file_path.name}: {result.get('status')} ({result.get('chunks', 0)} chunks)")

    print("Demo data initialization complete.")
    print(agent.show_kb_stats())


if __name__ == "__main__":
    main()

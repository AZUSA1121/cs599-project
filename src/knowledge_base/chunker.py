"""文本分块模块 - 将论文文本拆分为适合检索的文本块"""

import re
from typing import Optional


class TextChunker:
    """论文文本分块器，支持按段落/句子分割，带重叠"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: 每个文本块的目标字符数
            chunk_overlap: 相邻块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        """
        将文本分块，优先在段落/句子边界处分割
        返回文本块列表
        """
        if not text or not text.strip():
            return []

        text = re.sub(r'\n{3,}', '\n\n', text.strip())
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not paragraphs:
            paragraphs = [text.strip()]

        # 合并小段落
        merged = []
        current = ""
        for para in paragraphs:
            if not current:
                current = para
            elif len(current) + len(para) + 2 <= self.chunk_size:
                current += "\n\n" + para
            else:
                if current:
                    merged.append(current)
                current = para
        if current:
            merged.append(current)

        # 对过长的块进行二次分割
        chunks = []
        for block in merged:
            if len(block) <= self.chunk_size:
                chunks.append(block)
            else:
                chunks.extend(self._split_long_block(block))

        # 生成重叠块
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0 and self.chunk_overlap > 0:
                prev = chunks[i - 1]
                overlap_text = prev[-self.chunk_overlap:]
                combined = overlap_text + chunk
                if len(combined) <= self.chunk_size * 1.5:
                    overlapped.append(combined)
                else:
                    overlapped.append(chunk)
            else:
                overlapped.append(chunk)

        return overlapped

    def chunk_document(self, text: str, paper_title: str = "",
                       paper_id: str = "") -> list[dict]:
        """
        将完整文档分块并附加元数据
        返回: [{"content": str, "chunk_index": int, "paper_title": str, "paper_id": str}, ...]
        """
        raw_chunks = self.chunk_text(text)
        result = []
        for i, chunk in enumerate(raw_chunks):
            result.append({
                "content": chunk,
                "chunk_index": i,
                "paper_title": paper_title,
                "paper_id": paper_id,
            })
        return result

    def _split_long_block(self, text: str) -> list[str]:
        """对超长文本块按句子边界分割"""
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            # 按固定长度强制分割
            return self._force_split(text)

        chunks = []
        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
            elif len(current) + len(sentence) + 1 <= self.chunk_size:
                current += " " + sentence
            else:
                if current:
                    chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)

        # 对仍然超长的句子进行强制分割
        final = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final.append(chunk)
            else:
                final.extend(self._force_split(chunk))
        return final

    def _force_split(self, text: str) -> list[str]:
        """强制按固定长度分割"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

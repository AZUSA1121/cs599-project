"""论文解析模块 - 支持 PDF 和纯文本"""

import os
import re
from typing import Optional


def extract_text_from_file(file_path: str) -> str:
    """从文件中提取文本内容，支持 PDF 和 TXT"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _extract_from_pdf(file_path)
    elif ext in (".txt", ".md", ".text"):
        return _extract_from_text(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请使用 PDF 或 TXT 文件")


def _extract_from_text(file_path: str) -> str:
    """从纯文本文件提取内容"""
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码文件: {file_path}")


def _extract_from_pdf(file_path: str) -> str:
    """从 PDF 文件提取文本"""
    try:
        import PyPDF2
    except ImportError:
        print("[警告] 未安装 PyPDF2，正在使用备用方案...")
        return _extract_pdf_fallback(file_path)

    text_parts = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- 第 {i + 1} 页 ---\n{page_text}")

    full_text = "\n\n".join(text_parts)
    if not full_text.strip():
        raise ValueError(f"无法从 PDF 提取文本: {file_path}")
    return full_text


def _extract_pdf_fallback(file_path: str) -> str:
    """PDF 提取的备用方案 - 使用 pdfplumber 或其他库"""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- 第 {i + 1} 页 ---\n{page_text}")
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text_parts = []
        for i, page in enumerate(doc):
            text_parts.append(f"--- 第 {i + 1} 页 ---\n{page.get_text()}")
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    raise ImportError(
        "无法提取 PDF 内容。请安装以下任一库：\n"
        "  pip install PyPDF2\n"
        "  pip install pdfplumber\n"
        "  pip install PyMuPDF"
    )


def sanitize_text(text: str) -> str:
    """清理会导致编码失败的非法字符。"""
    if not text:
        return ""

    # 移除 UTF-16 代理字符，避免后续 UTF-8 编码时报错。
    text = re.sub(r"[\ud800-\udfff]", "", text)
    # 去掉常见非法控制字符，但保留换行、制表符等可读空白。
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def preprocess_text(text: str, max_chars: int = 30000) -> str:
    """预处理文本：清理多余空白，截断过长内容"""
    text = sanitize_text(text)

    # 清理多余空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... 内容已截断，原文共 {:,} 字符 ...]".format(len(text))
    return text

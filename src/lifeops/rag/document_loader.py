"""文档加载器 — 支持 PDF/TXT/MD 格式。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
import uuid


@dataclass
class Document:
    """文档对象。"""
    id: str
    title: str
    file_path: str
    file_type: str       # pdf | txt | md
    content: str
    file_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """文档加载器 — 根据文件扩展名自动选择加载方式。"""

    SUPPORTED_FORMATS = {".pdf", ".txt", ".md", ".markdown"}

    def load(self, file_path: str) -> Document:
        """根据扩展名自动选择加载器。

        Args:
            file_path: 文件路径

        Returns:
            Document 对象

        Raises:
            ValueError: 不支持的文件格式
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext in (".txt", ".md", ".markdown"):
            return self.load_text(file_path, "md" if ext in (".md", ".markdown") else "txt")
        else:
            raise ValueError(f"不支持的文件格式：{ext}")

    def load_pdf(self, file_path: str) -> Document:
        """加载 PDF 文档。

        Args:
            file_path: PDF 文件路径

        Returns:
            Document 对象

        Raises:
            RuntimeError: pypdf 未安装
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("pypdf 未安装，无法加载 PDF 文件")

        path = Path(file_path)
        reader = PdfReader(file_path)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_text.append(f"--- 第 {i+1} 页 ---\n{text}")

        content = "\n\n".join(pages_text)
        file_size = path.stat().st_size

        return Document(
            id=f"doc_{uuid.uuid4().hex[:12]}",
            title=path.stem,
            file_path=file_path,
            file_type="pdf",
            content=content,
            file_size=file_size,
            metadata={
                "page_count": len(reader.pages),
                "char_count": len(content),
            },
        )

    def load_text(self, file_path: str, file_type: str = "txt") -> Document:
        """加载纯文本或 Markdown 文档。

        Args:
            file_path: 文本文件路径
            file_type: 文件类型（txt 或 md）

        Returns:
            Document 对象
        """
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        file_size = path.stat().st_size

        return Document(
            id=f"doc_{uuid.uuid4().hex[:12]}",
            title=path.stem,
            file_path=file_path,
            file_type=file_type,
            content=content,
            file_size=file_size,
            metadata={
                "char_count": len(content),
                "line_count": content.count("\n") + 1,
            },
        )

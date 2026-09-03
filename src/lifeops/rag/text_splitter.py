"""文本分块器 — 支持 Markdown 标题感知分块。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import uuid
import re


@dataclass
class TextChunk:
    """文本块。"""
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any]


class TextSplitter:
    """文本分块器。

    对于 Markdown 文件，优先按标题层级分割，保持语义完整性；
    其他格式按字符数分割，并尝试在句子边界处断开。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """初始化分块器。

        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document_id: str, text: str,
              file_type: str = "txt",
              doc_metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """将文本分块。

        对于 Markdown，优先按标题分割；其他格式按字符数分割。

        Args:
            document_id: 所属文档 ID
            text: 待分块的文本内容
            file_type: 文件类型（md / txt / pdf）
            doc_metadata: 文档元数据，会附加到每个 chunk 上

        Returns:
            TextChunk 列表
        """
        if file_type == "md":
            return self._split_markdown(document_id, text, doc_metadata or {})
        else:
            return self._split_by_size(document_id, text, doc_metadata or {})

    def _split_markdown(self, doc_id: str, text: str,
                        doc_meta: Dict[str, Any]) -> List[TextChunk]:
        """Markdown 感知分块：按标题层级分割。

        按 # 标题将文档切分为多个 section，再根据 chunk_size 做进一步合并或拆分。
        """
        # 按 # 标题分割（保留标题行在 section 内）
        sections = re.split(r'\n(?=#{1,4}\s)', text)
        chunks: List[TextChunk] = []
        current_section = ""
        index = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # 如果当前累积 + 新段落不超过 chunk_size，合并
            if len(current_section) + len(section) + 2 <= self.chunk_size:
                if current_section:
                    current_section += "\n\n" + section
                else:
                    current_section = section
            else:
                # 先保存当前块
                if current_section:
                    chunks.append(self._make_chunk(doc_id, current_section, index, doc_meta))
                    index += 1
                    # 重叠：保留末尾 chunk_overlap 字符
                    overlap_text = current_section[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_section = overlap_text + "\n\n" + section if overlap_text else section
                else:
                    current_section = section

            # 如果单个 section 超过 chunk_size，需要进一步拆分
            while len(current_section) > self.chunk_size:
                piece = current_section[:self.chunk_size]
                chunks.append(self._make_chunk(doc_id, piece, index, doc_meta))
                index += 1
                current_section = current_section[self.chunk_size - self.chunk_overlap:]

        if current_section.strip():
            chunks.append(self._make_chunk(doc_id, current_section, index, doc_meta))

        return chunks

    def _split_by_size(self, doc_id: str, text: str,
                       doc_meta: Dict[str, Any]) -> List[TextChunk]:
        """按字符数固定大小分块。

        尝试在句子边界（句号、换行）处断开，保持语义完整。
        """
        chunks: List[TextChunk] = []
        start = 0
        index = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            # 尝试在句子边界断开
            if end < text_len:
                for boundary in ["。\n", "。", ".\n", ". ", "\n\n", "\n"]:
                    pos = text.rfind(boundary, start, end)
                    if pos > start + self.chunk_size // 2:
                        end = pos + len(boundary)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(self._make_chunk(doc_id, chunk_text, index, doc_meta))
                index += 1

            # 重叠
            if end >= text_len:
                break
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return chunks

    def _make_chunk(self, doc_id: str, content: str,
                    index: int, doc_meta: Dict[str, Any]) -> TextChunk:
        """创建一个 TextChunk。"""
        return TextChunk(
            id=f"chk_{uuid.uuid4().hex[:12]}",
            document_id=doc_id,
            content=content.strip(),
            chunk_index=index,
            metadata={
                "char_count": len(content),
                **doc_meta,
            },
        )

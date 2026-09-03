"""RAG 模块单元测试。"""
import pytest
import tempfile
from pathlib import Path

from lifeops.rag import (
    DocumentLoader, Document,
    TextSplitter, TextChunk,
    KnowledgeBase,
    Retriever, RetrievedChunk,
)


# ==========================================
# DocumentLoader 测试
# ==========================================
class TestDocumentLoader:
    """文档加载器测试。"""

    def test_load_txt(self, tmp_path):
        """测试加载纯文本。"""
        f = tmp_path / "test.txt"
        f.write_text("这是测试文档的内容\n第二行内容\n第三行", encoding="utf-8")

        loader = DocumentLoader()
        doc = loader.load(str(f))

        assert isinstance(doc, Document)
        assert doc.title == "test"
        assert doc.file_type == "txt"
        assert "这是测试文档的内容" in doc.content
        assert doc.file_size > 0
        assert doc.metadata["char_count"] == len(doc.content)
        assert doc.metadata["line_count"] == 3

    def test_load_md(self, tmp_path):
        """测试加载 Markdown。"""
        f = tmp_path / "notes.md"
        f.write_text("# 第一章\n\n这是第一章内容。\n\n## 1.1 小节\n\n小节内容。", encoding="utf-8")

        loader = DocumentLoader()
        doc = loader.load(str(f))

        assert doc.file_type == "md"
        assert doc.title == "notes"
        assert "# 第一章" in doc.content

    def test_load_pdf(self, tmp_path):
        """测试加载 PDF（需要 pypdf）。"""
        pytest.importorskip("pypdf")

        from pypdf import PdfWriter
        from pypdf.generic import RectangleObject
        from io import BytesIO

        # 创建一个简单的 PDF
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        # 简单 PDF 空白页测试
        pdf_path = tmp_path / "test.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)

        loader = DocumentLoader()
        doc = loader.load(str(pdf_path))

        assert isinstance(doc, Document)
        assert doc.file_type == "pdf"
        assert doc.title == "test"
        assert doc.metadata["page_count"] >= 1

    def test_unsupported_format(self, tmp_path):
        """测试不支持的格式。"""
        f = tmp_path / "test.docx"
        f.write_text("not supported")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="不支持的文件格式"):
            loader.load(str(f))

    def test_supported_formats(self):
        """测试支持的格式列表。"""
        assert ".pdf" in DocumentLoader.SUPPORTED_FORMATS
        assert ".txt" in DocumentLoader.SUPPORTED_FORMATS
        assert ".md" in DocumentLoader.SUPPORTED_FORMATS


# ==========================================
# TextSplitter 测试
# ==========================================
class TestTextSplitter:
    """文本分块器测试。"""

    def test_split_by_size_small(self):
        """短文本应该只产生一个块。"""
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        text = "这是一段短文本。" * 10  # ~70 chars

        chunks = splitter.split("doc_1", text, file_type="txt")
        assert len(chunks) == 1
        assert isinstance(chunks[0], TextChunk)
        assert chunks[0].document_id == "doc_1"
        assert chunks[0].chunk_index == 0

    def test_split_by_size_large(self):
        """长文本应该分成多块。"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        text = "测试文本。" * 100  # ~500 chars

        chunks = splitter.split("doc_1", text, file_type="txt")
        assert len(chunks) > 1
        # 所有块的长度应 <= chunk_size
        for chunk in chunks:
            assert len(chunk.content) <= 100 + 10  # 允许少量超出边界情况

    def test_split_overlap(self):
        """测试重叠。"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "A" * 40 + "B" * 40 + "C" * 40

        chunks = splitter.split("doc_1", text, file_type="txt")
        assert len(chunks) >= 2

    def test_markdown_split(self):
        """测试 Markdown 标题感知分块。"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        md_text = """# 第一章

这是第一章的内容，有很多很多的文字需要被处理，内容非常丰富，这里写了很多字。

## 1.1 第一节

第一节的内容也很长，有很多知识点需要学习和掌握。

## 1.2 第二节

第二节同样很重要，包含了很多关键的概念和公式。

# 第二章

第二章的内容是全新的领域，需要花时间理解和消化。
"""
        chunks = splitter.split("doc_1", md_text, file_type="md")
        assert len(chunks) >= 2
        # 第一个块应该包含第一章
        assert "第一章" in chunks[0].content

    def test_chunk_ids_unique(self):
        """测试所有块 ID 唯一。"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "测试" * 200

        chunks = splitter.split("doc_1", text, file_type="txt")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))  # 无重复

    def test_chunk_index_ordered(self):
        """测试 chunk_index 是有序的。"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "测试" * 200

        chunks = splitter.split("doc_1", text, file_type="txt")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# ==========================================
# KnowledgeBase 测试
# ==========================================
class TestKnowledgeBase:
    """知识库测试（mock 模式，不需要真实 embedding）。"""

    def test_init(self, tmp_path):
        """测试初始化。"""
        kb = KnowledgeBase(
            persist_dir=str(tmp_path / "chroma"),
            llm=None,  # mock 模式
            collection_name="test_kb",
        )
        assert kb.persist_dir == str(tmp_path / "chroma")
        assert kb.collection_name == "test_kb"

    def test_add_txt_document(self, tmp_path):
        """测试添加文本文档。"""
        kb = KnowledgeBase(
            persist_dir=str(tmp_path / "chroma"),
            llm=None,
            collection_name="test_add",
        )

        # 创建测试文档
        doc_file = tmp_path / "test_doc.txt"
        doc_file.write_text("这是一个测试文档。" * 20, encoding="utf-8")

        doc = kb.add_document(str(doc_file))

        assert isinstance(doc, Document)
        assert doc.title == "test_doc"
        assert doc.file_type == "txt"

    def test_list_documents_no_db(self, tmp_path):
        """测试无数据库时的列表查询。"""
        kb = KnowledgeBase(
            persist_dir=str(tmp_path / "chroma"),
            llm=None,
            collection_name="test_list",
        )

        # 添加两个文档
        for i in range(2):
            f = tmp_path / f"doc{i}.txt"
            f.write_text(f"文档 {i} 的内容。" * 10, encoding="utf-8")
            kb.add_document(str(f))

        docs = kb.list_documents()
        assert len(docs) == 2

    def test_delete_document(self, tmp_path):
        """测试删除文档。"""
        kb = KnowledgeBase(
            persist_dir=str(tmp_path / "chroma"),
            llm=None,
            collection_name="test_del",
        )

        f = tmp_path / "del_test.txt"
        f.write_text("待删除的文档内容。" * 10, encoding="utf-8")
        doc = kb.add_document(str(f))

        # 验证存在
        docs_before = kb.list_documents()
        assert len(docs_before) == 1

        # 删除
        kb.delete_document(doc.id)

        # 验证删除
        docs_after = kb.list_documents()
        assert len(docs_after) == 0


# ==========================================
# Retriever 测试
# ==========================================
class TestRetriever:
    """检索器测试（mock 模式）。"""

    def test_init(self, tmp_path):
        """测试初始化。"""
        kb = KnowledgeBase(
            persist_dir=str(tmp_path / "chroma"),
            llm=None,
            collection_name="test_ret",
        )
        retriever = Retriever(kb, llm=None)
        assert retriever.kb == kb

    def test_search_empty(self, tmp_path):
        """空库搜索。"""
        kb = KnowledgeBase(
            persist_dir=str(tmp_path / "chroma"),
            llm=None,
            collection_name="test_empty_search",
        )
        retriever = Retriever(kb, llm=None)

        results = retriever.search("测试查询", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_with_data(self, tmp_path):
        """有数据时的搜索。"""
        kb = KnowledgeBase(
            persist_dir=str(tmp_path / "chroma"),
            llm=None,
            collection_name="test_search",
        )

        # 添加文档
        f = tmp_path / "search_test.txt"
        f.write_text("机器学习是人工智能的一个分支。深度学习是机器学习的子领域。" * 10, encoding="utf-8")
        kb.add_document(str(f))

        retriever = Retriever(kb, llm=None)
        results = retriever.search("机器学习", top_k=3)

        # mock 模式下结果是随机的，但应该有结果返回
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], RetrievedChunk)
        assert 0.0 <= results[0].score <= 1.0

    def test_build_context(self, tmp_path):
        """测试构建上下文。"""
        kb = KnowledgeBase(
            persist_dir=str(tmp_path / "chroma"),
            llm=None,
            collection_name="test_context",
        )

        f = tmp_path / "ctx_test.txt"
        f.write_text("Python 是一种编程语言。它简洁易读。" * 10, encoding="utf-8")
        kb.add_document(str(f))

        retriever = Retriever(kb, llm=None)
        context_str, references = retriever.build_context("Python 是什么", top_k=3)

        assert isinstance(context_str, str)
        assert isinstance(references, list)
        assert "参考资料" in context_str
        if references:
            assert "index" in references[0]
            assert "document_title" in references[0]

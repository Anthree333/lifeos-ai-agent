"""知识库主类 — 基于 ChromaDB 的向量存储。"""
from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path

from .document_loader import DocumentLoader, Document
from .text_splitter import TextSplitter, TextChunk


class KnowledgeBase:
    """知识库 — 基于 ChromaDB 的向量存储。

    支持文档的上传、删除、列表查询，以及向量索引的自动构建。
    ChromaDB 采用延迟初始化，避免 import 阶段就依赖 chromadb。
    支持无 LLM 的 mock 模式（随机向量），方便测试。
    """

    def __init__(
        self,
        persist_dir: str,
        llm: Any = None,
        db: Any = None,  # Database 实例（可选，用于存元数据）
        collection_name: str = "knowledge",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """初始化知识库。

        Args:
            persist_dir: ChromaDB 持久化目录
            llm: LLM 实例，需实现 embed(texts) -> List[List[float]] 方法
            db: Database 实例，用于存储文档元数据（可选）
            collection_name: ChromaDB 集合名称
            chunk_size: 文本分块大小（字符数）
            chunk_overlap: 分块重叠大小（字符数）
        """
        self.persist_dir = persist_dir
        self.llm = llm
        self.db = db
        self.collection_name = collection_name
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(chunk_size, chunk_overlap)

        # 延迟初始化 ChromaDB
        self._client = None
        self._collection = None
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

    @property
    def client(self):
        """ChromaDB 客户端（延迟初始化）。"""
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    @property
    def collection(self):
        """ChromaDB 集合（延迟初始化）。"""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_document(self, file_path: str) -> Document:
        """上传文档并建立向量索引。

        流程：加载文档 → 分块 → 向量化 → 存入 ChromaDB → 存数据库元数据。

        Args:
            file_path: 文档文件路径

        Returns:
            Document 对象

        Raises:
            ValueError: 文档内容为空
        """
        # 1. 加载文档
        doc = self.loader.load(file_path)

        # 2. 分块
        chunks = self.splitter.split(doc.id, doc.content, doc.file_type, doc.metadata)

        if not chunks:
            raise ValueError("文档内容为空，无法建立索引")

        # 3. 向量化
        if self.llm:
            embeddings = self.llm.embed([c.content for c in chunks])
        else:
            # Mock 模式：使用随机向量（仅测试用）
            import random
            dim = 1536
            embeddings = [[random.uniform(-0.1, 0.1) for _ in range(dim)] for _ in chunks]

        # 4. 存入 ChromaDB
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[{
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "doc_title": doc.title,
                **c.metadata,
            } for c in chunks],
        )

        # 5. 存数据库元数据（如果有 db）
        if self.db:
            self.db.add_knowledge_document(doc, len(chunks))

        return doc

    def delete_document(self, doc_id: str) -> None:
        """删除文档及其所有向量。

        Args:
            doc_id: 文档 ID
        """
        # 从 ChromaDB 删除
        self.collection.delete(
            where={"document_id": doc_id}
        )
        # 从数据库删除
        if self.db:
            self.db.delete_knowledge_document(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档。

        优先从数据库查询；没有 db 时从 ChromaDB 聚合文档信息。

        Returns:
            文档列表，每项包含 id 和 title
        """
        if self.db:
            return self.db.list_knowledge_documents()
        # 没有 db 时从 ChromaDB 聚合
        results = self.collection.get()
        doc_ids = set()
        doc_titles: Dict[str, str] = {}
        for meta in results["metadatas"] or []:
            if meta:
                did = meta.get("document_id")
                if did:
                    doc_ids.add(did)
                    doc_titles[did] = meta.get("doc_title", did)
        return [{"id": did, "title": doc_titles.get(did, did)} for did in doc_ids]

    def get_chunk_count(self, doc_id: str) -> int:
        """获取文档的分块数。

        Args:
            doc_id: 文档 ID

        Returns:
            分块数量
        """
        try:
            res = self.collection.get(where={"document_id": doc_id})
            return len(res.get("ids", []))
        except Exception:
            return 0

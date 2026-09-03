"""向量检索器 — 基于 ChromaDB 的相似度搜索。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class RetrievedChunk:
    """检索结果。"""
    chunk_id: str
    document_id: str
    document_title: str
    content: str
    score: float
    metadata: Dict[str, Any]


class Retriever:
    """向量检索器。

    基于 ChromaDB 集合进行相似度搜索，支持按文档过滤。
    提供 search() 方法返回原始检索结果，以及 build_context() 方法
    将结果组装为 LLM 可用的上下文字符串。
    """

    def __init__(self, knowledge_base: Any, llm: Any = None):
        """初始化检索器。

        Args:
            knowledge_base: KnowledgeBase 实例
            llm: LLM 实例，需实现 embed(texts) 方法（可选，mock 模式下可为 None）
        """
        self.kb = knowledge_base
        self.llm = llm

    def search(self, query: str, top_k: int = 5,
               filter_doc_id: Optional[str] = None) -> List[RetrievedChunk]:
        """相似度搜索。

        Args:
            query: 查询文本
            top_k: 返回结果数
            filter_doc_id: 限定在某个文档内搜索（可选）

        Returns:
            RetrievedChunk 列表，按相似度从高到低排序
        """
        # 0. 知识库为空时跳过，避免无意义的 embedding 调用
        try:
            if self.kb.collection.count() == 0:
                return []
        except Exception:
            return []

        # 1. 生成 query 向量（embedding 服务不可用时降级为无检索结果）
        if self.llm:
            try:
                query_embedding = self.llm.embed([query])[0]
            except Exception:
                return []
        else:
            # Mock 模式
            import random
            query_embedding = [random.uniform(-0.1, 0.1) for _ in range(1536)]

        # 2. 搜索
        where = {"document_id": filter_doc_id} if filter_doc_id else None
        try:
            results = self.kb.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )
        except Exception:
            return []

        # 3. 组装结果
        chunks: List[RetrievedChunk] = []
        ids = results["ids"][0] if results["ids"] else []
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results.get("distances") else []

        for i, cid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            # ChromaDB cosine distance → 相似度分数
            dist = distances[i] if i < len(distances) else 1.0
            score = max(0.0, 1.0 - dist)  # 距离转相似度

            chunks.append(RetrievedChunk(
                chunk_id=cid,
                document_id=meta.get("document_id", ""),
                document_title=meta.get("doc_title", ""),
                content=docs[i] if i < len(docs) else "",
                score=score,
                metadata=meta,
            ))

        return chunks

    def build_context(self, query: str, top_k: int = 5) -> tuple:
        """构建检索上下文，返回 (context_str, references)。

        context_str 格式：
            参考资料：
            [1] 《文档名》：内容片段
            [2] 《文档名》：内容片段
            ...

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            (context_str, references) 元组
            - context_str: 格式化的参考资料文本
            - references: 引用列表，每项包含 index、document_id、document_title、content、score
        """
        chunks = self.search(query, top_k=top_k)
        if not chunks:
            return "", []

        context_parts = ["参考资料："]
        references: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks, 1):
            title = chunk.document_title or chunk.document_id
            # 截断过长内容
            content_preview = chunk.content[:300]
            if len(chunk.content) > 300:
                content_preview += "..."

            context_parts.append(f"[{i}] 《{title}》：{content_preview}")
            references.append({
                "index": i,
                "document_id": chunk.document_id,
                "document_title": title,
                "content": chunk.content,
                "score": chunk.score,
            })

        return "\n\n".join(context_parts), references

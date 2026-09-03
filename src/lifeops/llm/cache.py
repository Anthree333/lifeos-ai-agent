"""LLM 调用缓存。

基于 SQLite 持久化，跨会话复用。
缓存键为 prompt + model + 参数的 SHA256 哈希。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import sqlite3


class LLMCache:
    """LLM 调用缓存。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get(
            "DB_PATH", "./data/lifeos.db"
        )
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_preview TEXT NOT NULL DEFAULT '',
                    response TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, prompt_hash: str, model: str) -> Optional[Dict[str, Any]]:
        """获取缓存。"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM llm_cache WHERE prompt_hash = ? AND model = ?",
                (prompt_hash, model),
            ).fetchone()

            if row is None:
                return None

            # 更新命中计数
            conn.execute(
                """
                UPDATE llm_cache
                SET hit_count = hit_count + 1, last_used_at = ?
                WHERE prompt_hash = ? AND model = ?
                """,
                (datetime.now().isoformat(), prompt_hash, model),
            )
            conn.commit()

            return json.loads(row["response"])

    def set(
        self,
        prompt_hash: str,
        model: str,
        response: Dict[str, Any],
        preview: str = "",
    ) -> None:
        """写入缓存。"""
        now = datetime.now().isoformat()
        response_json = json.dumps(response, ensure_ascii=False)

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (prompt_hash, model, prompt_preview, response, hit_count, created_at, last_used_at)
                VALUES (?, ?, ?, ?, COALESCE(
                    (SELECT hit_count + 1 FROM llm_cache WHERE prompt_hash = ? AND model = ?),
                    1
                ), ?, ?)
                """,
                (
                    prompt_hash,
                    model,
                    preview,
                    response_json,
                    prompt_hash,
                    model,
                    now,
                    now,
                ),
            )
            conn.commit()

    def clear(self) -> int:
        """清空缓存，返回删除的条数。"""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM llm_cache")
            conn.commit()
            return cursor.rowcount

    def stats(self) -> Dict[str, Any]:
        """获取缓存统计。"""
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_entries,
                    SUM(hit_count) as total_hits,
                    COUNT(*) - SUM(CASE WHEN hit_count = 1 THEN 1 ELSE 0 END) as reused_entries
                FROM llm_cache
                """
            ).fetchone()
            return {
                "total_entries": row["total_entries"] or 0,
                "total_hits": row["total_hits"] or 0,
                "reused_entries": row["reused_entries"] or 0,
            }

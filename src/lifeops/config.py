"""配置管理模块。

集中管理所有环境变量，提供类型校验和默认值。
纯标准库实现（dataclass + os.environ），不依赖第三方包。

用法：
    from lifeops.config import settings
    print(settings.LLM_MODEL)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


def _env_bool(key: str, default: bool = False) -> bool:
    """从环境变量读取布尔值。"""
    val = os.environ.get(key, "")
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    """从环境变量读取浮点数，非法值回退默认。"""
    try:
        return float(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    """从环境变量读取整数，非法值回退默认。"""
    try:
        return int(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """LifeOS 全局配置（从环境变量读取，带类型校验）。"""

    # --- 模式开关 ---
    LLM_MOCK_MODE: bool = field(default_factory=lambda: _env_bool("LLM_MOCK_MODE", False))

    # --- LLM 基础配置 ---
    LLM_BASE_URL: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", ""))
    LLM_API_KEY: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    LLM_MODEL: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "deepseek-chat"))
    LLM_TIMEOUT_SECONDS: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 20.0))
    LLM_MAX_RETRIES: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 1))

    # --- 视觉模型 ---
    LLM_VISION_MODEL: str = field(default_factory=lambda: os.environ.get("LLM_VISION_MODEL", ""))
    LLM_VISION_BASE_URL: Optional[str] = field(default_factory=lambda: os.environ.get("LLM_VISION_BASE_URL") or None)
    LLM_VISION_API_KEY: Optional[str] = field(default_factory=lambda: os.environ.get("LLM_VISION_API_KEY") or None)

    # --- Embedding ---
    EMBED_MODEL: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", "text-embedding-v2"))
    EMBED_BASE_URL: Optional[str] = field(default_factory=lambda: os.environ.get("EMBED_BASE_URL") or None)
    EMBED_API_KEY: Optional[str] = field(default_factory=lambda: os.environ.get("EMBED_API_KEY") or None)

    # --- 数据库 ---
    DB_PATH: str = field(default_factory=lambda: os.environ.get("DB_PATH", "./data/lifeos.db"))

    # --- 向量库 ---
    CHROMA_PATH: str = field(default_factory=lambda: os.environ.get("CHROMA_PATH", "./data/knowledge/chroma"))

    # --- MCP ---
    MCP_MODE: str = field(default_factory=lambda: os.environ.get("MCP_MODE", "stdio"))

    # --- 应用 ---
    APP_TIMEZONE: str = field(default_factory=lambda: os.environ.get("APP_TIMEZONE", "Asia/Shanghai"))
    APP_LANGUAGE: str = field(default_factory=lambda: os.environ.get("APP_LANGUAGE", "zh-CN"))

    # --- 日志 ---
    LOG_LEVEL: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO").upper())

    def __post_init__(self) -> None:
        """初始化后校验。"""
        # 非 Mock 模式下 LLM 配置必填
        if not self.LLM_MOCK_MODE:
            if not self.LLM_BASE_URL.strip() or not self.LLM_API_KEY.strip():
                raise ValueError(
                    "非 Mock 模式下 LLM_BASE_URL 和 LLM_API_KEY 必须配置，"
                    "或者设置 LLM_MOCK_MODE=true"
                )

        # MCP_MODE 校验
        if self.MCP_MODE not in ("stdio", "sse"):
            raise ValueError(f"MCP_MODE 必须是 stdio 或 sse，当前: {self.MCP_MODE}")

        # LOG_LEVEL 校验
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.LOG_LEVEL not in valid_levels:
            raise ValueError(f"LOG_LEVEL 必须是 {valid_levels} 之一，当前: {self.LOG_LEVEL}")

        # 超时和重试范围校验
        if self.LLM_TIMEOUT_SECONDS <= 0:
            raise ValueError(f"LLM_TIMEOUT_SECONDS 必须 > 0，当前: {self.LLM_TIMEOUT_SECONDS}")
        if self.LLM_MAX_RETRIES < 0:
            raise ValueError(f"LLM_MAX_RETRIES 必须 >= 0，当前: {self.LLM_MAX_RETRIES}")

    @property
    def is_mock(self) -> bool:
        """是否处于 Mock 模式。"""
        return self.LLM_MOCK_MODE

    @property
    def llm_ready(self) -> bool:
        """LLM 是否可用（Mock 模式或已配置密钥）。"""
        return self.is_mock or bool(self.LLM_BASE_URL and self.LLM_API_KEY)


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例（带缓存）。"""
    return Settings()


# 便捷访问：from lifeops.config import settings
settings = get_settings()

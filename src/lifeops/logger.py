"""统一日志模块。

- 从环境变量 LOG_LEVEL 读取级别（默认 INFO）
- MCP stdio 模式下输出到 stderr，避免污染协议通道
- 格式：时间 [级别] 模块名: 消息
"""
from __future__ import annotations

import os
import sys
import logging
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_default_handler: Optional[logging.Handler] = None


def _get_default_handler() -> logging.Handler:
    """获取（并缓存）默认 handler。"""
    global _default_handler
    if _default_handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        _default_handler = handler
    return _default_handler


def get_logger(name: str = "lifeos") -> logging.Logger:
    """获取配置好的 logger。

    Args:
        name: logger 名称，建议用模块名（如 "lifeos.agent"）

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if not logger.handlers:
        handler = _get_default_handler()
        logger.addHandler(handler)

    # 从环境变量读取日志级别
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    # 不向上传播，避免根 logger 重复输出
    logger.propagate = False

    return logger


def set_level(level: int | str) -> None:
    """动态调整所有 lifeos logger 的级别。"""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("lifeos").setLevel(level)
    for handler in logging.getLogger("lifeos").handlers:
        handler.setLevel(level)

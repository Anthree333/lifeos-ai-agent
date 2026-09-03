"""视觉解析模块。

基于视觉大模型的截图内容结构化解析，支持考试安排、课程表、作业清单等多种类型。
"""

from .screenshot_parser import ParsedItem, ScreenshotParseResult, ScreenshotParser

__all__ = [
    "ParsedItem",
    "ScreenshotParseResult",
    "ScreenshotParser",
]

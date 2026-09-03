"""截图解析器 — 基于视觉大模型的结构化提取。

支持考试安排、课程表、作业清单等多种截图类型的自动识别与结构化解析。
视觉模型失败时可降级为 OCR + 文本 LLM 解析。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm.chat_model import ChatMessage, ChatRole, ChatModel
from .prompts import (
    COURSE_TABLE_PROMPT,
    EXAM_SCHEDULE_PROMPT,
    GENERIC_PARSE_PROMPT,
    HOMEWORK_PROMPT,
)


@dataclass
class ParsedItem:
    """解析出的单个条目。"""

    item_type: str  # exam | course | homework | task | event | commitment
    title: str
    date: Optional[str] = None  # YYYY-MM-DD
    start_time: Optional[str] = None  # HH:MM
    end_time: Optional[str] = None  # HH:MM
    location: Optional[str] = None
    description: Optional[str] = None
    deadline_type: str = "soft"  # hard | soft | none
    recurrence: str = "none"  # none | daily | weekly
    day_of_week: Optional[int] = None  # 0-6, 周一=0
    priority: int = 5
    estimated_minutes: Optional[int] = None
    energy_level: str = "medium"  # high | medium | low
    raw: Dict[str, Any] = field(default_factory=dict)  # 原始数据


@dataclass
class ScreenshotParseResult:
    """截图解析结果。"""

    type: str  # exam_schedule | course_table | homework | generic
    items: List[ParsedItem]
    summary: str
    confidence: float
    raw_text: str = ""
    source_image_hash: str = ""
    parse_method: str = "vision_model"  # vision_model | ocr_fallback


class ScreenshotParser:
    """截图解析器 — 基于视觉大模型的结构化提取。"""

    def __init__(self, llm: Optional[ChatModel] = None, use_ocr_fallback: bool = True):
        self.llm = llm
        self.use_ocr_fallback = use_ocr_fallback

    def parse(self, image_path: str, image_type: str = "auto") -> ScreenshotParseResult:
        """解析截图，返回结构化结果。

        Args:
            image_path: 图片文件路径
            image_type: auto | exam_schedule | course_table | homework | generic
        """
        # 1. 计算图片哈希
        img_hash = self._compute_hash(image_path)

        # 2. 选择 prompt
        prompt = self._select_prompt(image_type)

        # 3. 调用视觉大模型
        try:
            result = self._parse_with_vision(image_path, prompt, img_hash)
            return result
        except Exception as e:
            # 4. 失败则尝试 OCR 兜底
            if self.use_ocr_fallback:
                try:
                    return self._parse_with_ocr(image_path, prompt, img_hash)
                except Exception as e2:
                    raise RuntimeError(f"截图解析失败：视觉模型={e}, OCR={e2}")
            raise

    def _compute_hash(self, image_path: str) -> str:
        """计算图片 SHA-256 哈希（前 16 位）。"""
        with open(image_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

    def _select_prompt(self, image_type: str) -> str:
        """根据图片类型选择对应的 Prompt。"""
        prompts = {
            "exam_schedule": EXAM_SCHEDULE_PROMPT,
            "course_table": COURSE_TABLE_PROMPT,
            "homework": HOMEWORK_PROMPT,
            "auto": GENERIC_PARSE_PROMPT,
            "generic": GENERIC_PARSE_PROMPT,
        }
        return prompts.get(image_type, GENERIC_PARSE_PROMPT)

    def _parse_with_vision(
        self, image_path: str, prompt: str, img_hash: str
    ) -> ScreenshotParseResult:
        """使用视觉大模型解析。"""
        if not self.llm:
            raise RuntimeError("未配置 LLM，无法使用视觉解析")

        result = self.llm.chat_with_image(
            text_prompt=prompt,
            image_path=image_path,
            temperature=0.1,
            max_tokens=2000,
        )

        content = result.get("content", "")
        parsed_data = self._extract_json(content)
        return self._build_result(parsed_data, content, img_hash, "vision_model")

    def _parse_with_ocr(
        self, image_path: str, prompt: str, img_hash: str
    ) -> ScreenshotParseResult:
        """OCR 兜底：提取文字后用文本 LLM 解析。"""
        try:
            import pytesseract
            from PIL import Image

            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        except ImportError:
            raise RuntimeError("pytesseract 未安装，OCR 兜底不可用")
        except Exception as e:
            raise RuntimeError(f"OCR 提取失败：{e}")

        if not self.llm:
            raise RuntimeError("未配置 LLM，无法解析 OCR 文本")

        # 用文本 LLM 解析（给它 OCR 文本 + 同样的 prompt 要求）
        ocr_prompt = f"""以下是从图片中 OCR 提取的文字：
---
{text}
---
请按照以下要求解析这些文字：
{prompt}
"""
        messages = [ChatMessage(ChatRole.USER, ocr_prompt)]
        result = self.llm.chat_json(messages, temperature=0.1)
        return self._build_result(result, text, img_hash, "ocr_fallback")

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """从模型输出中提取 JSON。"""
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试提取第一个 { 到最后一个 }
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass

        return {
            "type": "generic",
            "items": [],
            "summary": content[:100],
            "confidence": 0.3,
        }

    def _build_result(
        self,
        data: Dict[str, Any],
        raw_text: str,
        img_hash: str,
        method: str,
    ) -> ScreenshotParseResult:
        """从 JSON 数据构建解析结果。"""
        items_data = data.get("items", [])
        items: List[ParsedItem] = []
        for item in items_data:
            items.append(
                ParsedItem(
                    item_type=item.get("item_type", "task"),
                    title=item.get("title", ""),
                    date=item.get("date"),
                    start_time=item.get("start_time"),
                    end_time=item.get("end_time"),
                    location=item.get("location"),
                    description=item.get("description"),
                    deadline_type=item.get("deadline_type", "soft"),
                    recurrence=item.get("recurrence", "none"),
                    day_of_week=item.get("day_of_week"),
                    priority=int(item.get("priority", 5)),
                    estimated_minutes=item.get("estimated_minutes"),
                    energy_level=item.get("energy_level", "medium"),
                    raw=item,
                )
            )

        return ScreenshotParseResult(
            type=data.get("type", "generic"),
            items=items,
            summary=data.get("summary", ""),
            confidence=float(data.get("confidence", 0.5)),
            raw_text=raw_text,
            source_image_hash=img_hash,
            parse_method=method,
        )

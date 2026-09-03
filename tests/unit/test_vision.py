"""Vision 模块单元测试。"""
import pytest
import tempfile
from pathlib import Path
from PIL import Image

from lifeops.vision import ScreenshotParser, ParsedItem, ScreenshotParseResult
from lifeops.vision.prompts import (
    GENERIC_PARSE_PROMPT,
    EXAM_SCHEDULE_PROMPT,
    COURSE_TABLE_PROMPT,
    HOMEWORK_PROMPT,
)


class TestPrompts:
    """Prompt 模板测试。"""

    def test_all_prompts_exist(self):
        assert GENERIC_PARSE_PROMPT
        assert EXAM_SCHEDULE_PROMPT
        assert COURSE_TABLE_PROMPT
        assert HOMEWORK_PROMPT

    def test_prompts_contain_json_instruction(self):
        """所有 Prompt 都应该要求 JSON 输出。"""
        for prompt in [GENERIC_PARSE_PROMPT, EXAM_SCHEDULE_PROMPT,
                       COURSE_TABLE_PROMPT, HOMEWORK_PROMPT]:
            assert "json" in prompt.lower() or "JSON" in prompt


class TestParsedItem:
    """ParsedItem 数据类测试。"""

    def test_default_values(self):
        item = ParsedItem(item_type="task", title="测试任务")
        assert item.item_type == "task"
        assert item.title == "测试任务"
        assert item.priority == 5
        assert item.energy_level == "medium"
        assert item.deadline_type == "soft"
        assert item.recurrence == "none"

    def test_full_fields(self):
        item = ParsedItem(
            item_type="exam",
            title="高数期末考试",
            date="2024-06-15",
            start_time="09:00",
            end_time="11:00",
            location="教学楼A101",
            description="闭卷考试",
            deadline_type="hard",
            priority=10,
            estimated_minutes=120,
            energy_level="high",
        )
        assert item.item_type == "exam"
        assert item.date == "2024-06-15"
        assert item.deadline_type == "hard"
        assert item.priority == 10


class TestScreenshotParser:
    """ScreenshotParser 基础测试。"""

    def test_init_without_llm(self):
        parser = ScreenshotParser(llm=None)
        assert parser.llm is None
        assert parser.use_ocr_fallback is True

    def test_compute_hash(self):
        """测试图片哈希计算。"""
        parser = ScreenshotParser(llm=None)
        # 创建一个小的测试图片
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (10, 10), color="red")
            img.save(f.name)
            f.flush()
            f.close()
            h1 = parser._compute_hash(f.name)
            h2 = parser._compute_hash(f.name)
            assert h1 == h2  # 同一张图哈希相同
            assert len(h1) == 16  # 16 字符 hex
            Path(f.name).unlink()

    def test_select_prompt(self):
        """测试 prompt 选择逻辑。"""
        parser = ScreenshotParser(llm=None)
        assert parser._select_prompt("auto") == GENERIC_PARSE_PROMPT
        assert parser._select_prompt("exam_schedule") == EXAM_SCHEDULE_PROMPT
        assert parser._select_prompt("course_table") == COURSE_TABLE_PROMPT
        assert parser._select_prompt("homework") == HOMEWORK_PROMPT
        assert parser._select_prompt("unknown") == GENERIC_PARSE_PROMPT

    def test_extract_json_direct(self):
        """测试直接 JSON 解析。"""
        parser = ScreenshotParser(llm=None)
        data = parser._extract_json('{"type": "exam", "items": []}')
        assert data["type"] == "exam"
        assert data["items"] == []

    def test_extract_json_code_block(self):
        """测试从 ```json``` 块中提取。"""
        parser = ScreenshotParser(llm=None)
        content = '''好的，以下是解析结果：
```json
{"type": "homework", "items": [{"title": "数学作业"}], "summary": "数学作业", "confidence": 0.9}
```
'''
        data = parser._extract_json(content)
        assert data["type"] == "homework"
        assert len(data["items"]) == 1
        assert data["items"][0]["title"] == "数学作业"

    def test_extract_json_braces(self):
        """测试从大括号提取。"""
        parser = ScreenshotParser(llm=None)
        content = '解析结果如下：{"type":"generic","items":[],"summary":"test","confidence":0.5} 完毕'
        data = parser._extract_json(content)
        assert data["type"] == "generic"
        assert data["confidence"] == 0.5

    def test_extract_json_invalid_fallback(self):
        """测试完全无法解析时的 fallback。"""
        parser = ScreenshotParser(llm=None)
        data = parser._extract_json("这不是 JSON")
        assert data["type"] == "generic"
        assert data["items"] == []
        assert data["confidence"] == 0.3

    def test_build_result(self):
        """测试构建解析结果。"""
        parser = ScreenshotParser(llm=None)
        raw_data = {
            "type": "exam_schedule",
            "items": [
                {"item_type": "exam", "title": "高数期末", "date": "2024-06-15",
                 "priority": 9, "deadline_type": "hard"},
                {"item_type": "exam", "title": "物理期末", "date": "2024-06-17",
                 "priority": 8},
            ],
            "summary": "两门期末考试",
            "confidence": 0.92,
        }
        result = parser._build_result(raw_data, "raw text", "abc123", "vision_model")
        assert isinstance(result, ScreenshotParseResult)
        assert result.type == "exam_schedule"
        assert len(result.items) == 2
        assert result.items[0].title == "高数期末"
        assert result.items[0].priority == 9
        assert result.summary == "两门期末考试"
        assert result.confidence == 0.92
        assert result.source_image_hash == "abc123"
        assert result.parse_method == "vision_model"
        assert result.raw_text == "raw text"

    def test_parse_no_llm_raises(self):
        """没有 LLM 时调用 parse 应该抛错（OCR 也没装的话）。"""
        parser = ScreenshotParser(llm=None, use_ocr_fallback=True)
        # 创建测试图片
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (10, 10), color="red")
            img.save(f.name)
            f.flush()
            f.close()
            with pytest.raises(RuntimeError):
                parser.parse(f.name)
            Path(f.name).unlink()

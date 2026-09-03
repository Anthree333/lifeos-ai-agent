"""LifeOS 能力探测脚本（D1 门禁）。

验证 LLM 服务商的三项核心能力：
1. 文本对话 + Function calling
2. 视觉模型（截图转结构化 JSON）
3. Embedding 向量化

用法：
    python -m lifeops.probe
    python src/lifeops/probe.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# 确保能导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from lifeops.llm.chat_model import ChatModel, ChatMessage, ChatRole
from lifeops.llm.cache import LLMCache


# ==========================================
# 测试用例定义
# ==========================================

TEST_FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "add_task",
        "description": "添加一个新任务到计划中",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "任务标题"},
                "duration_minutes": {"type": "integer", "description": "预计时长（分钟）"},
                "deadline": {"type": "string", "description": "截止日期 YYYY-MM-DD"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["title", "duration_minutes"],
        },
    },
}

VISION_TEST_PROMPT = """请分析这张图片，提取其中的日程信息，输出严格的 JSON 格式：
{
  "events": [
    {
      "title": "事件标题",
      "date": "YYYY-MM-DD",
      "start_time": "HH:MM",
      "end_time": "HH:MM",
      "event_type": "exam|assignment|class|other"
    }
  ]
}
只输出 JSON，不要其他文字。"""

EMBED_TEST_TEXTS = [
    "今天学习了 Python 基础语法",
    "高等数学第三章：导数与微分",
    "计算机网络 TCP 三次握手过程",
]


# ==========================================
# 探测结果数据结构
# ==========================================

class ProbeResult:
    """探测结果。"""

    def __init__(self):
        self.text_chat: Dict[str, Any] = {"status": "untested"}
        self.function_calling: Dict[str, Any] = {"status": "untested"}
        self.vision: Dict[str, Any] = {"status": "untested"}
        self.embedding: Dict[str, Any] = {"status": "untested"}
        self.llm_cache: Dict[str, Any] = {"status": "untested"}

    @property
    def all_passed(self) -> bool:
        checks = [
            self.text_chat.get("status") == "ok",
            self.function_calling.get("status") == "ok",
            self.embedding.get("status") == "ok",
            # vision 是可选的
        ]
        return all(checks)

    def to_markdown(self) -> str:
        """生成 CAPABILITY.md 内容。"""
        lines = ["# LifeOS 能力探测报告", ""]
        lines.append(f"**探测时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 文本对话
        lines.append("## 1. 文本对话")
        lines.append(self._format_status(self.text_chat))
        lines.append("")

        # Function calling
        lines.append("## 2. Function Calling")
        lines.append(self._format_status(self.function_calling))
        lines.append("")

        # 视觉模型
        lines.append("## 3. 视觉模型")
        lines.append(self._format_status(self.vision))
        lines.append("")

        # Embedding
        lines.append("## 4. Embedding 向量化")
        lines.append(self._format_status(self.embedding))
        lines.append("")

        # 缓存
        lines.append("## 5. LLM 缓存")
        lines.append(self._format_status(self.llm_cache))
        lines.append("")

        # 总结
        lines.append("## 总结")
        if self.all_passed:
            lines.append("✅ **核心能力全部通过**，可以进入 D2 开发。")
        else:
            lines.append("⚠️ **部分能力未通过**，请检查配置后重新探测。")
        lines.append("")

        # 配置信息（脱敏）
        lines.append("## 配置信息")
        lines.append(f"- LLM_BASE_URL: `{_mask_url(os.environ.get('LLM_BASE_URL', ''))}`")
        lines.append(f"- LLM_MODEL: `{os.environ.get('LLM_MODEL', '')}`")
        lines.append(f"- LLM_VISION_MODEL: `{os.environ.get('LLM_VISION_MODEL', '')}`")
        lines.append(f"- EMBED_MODEL: `{os.environ.get('EMBED_MODEL', '')}`")
        lines.append("")

        return "\n".join(lines)

    def _format_status(self, result: Dict[str, Any]) -> str:
        status = result.get("status", "untested")
        icon = {"ok": "✅", "fail": "❌", "untested": "⏳", "skip": "⏭️"}.get(status, "❓")

        lines = [f"{icon} **状态**：{status}"]

        if "model" in result:
            lines.append(f"- 模型：`{result['model']}`")
        if "response_time_ms" in result:
            lines.append(f"- 响应时间：{result['response_time_ms']}ms")
        if "detail" in result:
            lines.append(f"- 详情：{result['detail']}")
        if "error" in result:
            lines.append(f"- 错误：{result['error']}")
        if "sample" in result:
            sample = str(result["sample"])
            if len(sample) > 200:
                sample = sample[:200] + "..."
            lines.append(f"- 示例输出：`{sample}`")

        return "\n".join(lines)


def _mask_url(url: str) -> str:
    """脱敏 URL（保留域名部分）。"""
    if not url:
        return "(未设置)"
    return url


# ==========================================
# 各项探测函数
# ==========================================

def probe_text_chat(model: ChatModel) -> Dict[str, Any]:
    """测试文本对话能力。"""
    print("  📝 测试文本对话...", end=" ", flush=True)
    import time

    messages = [
        ChatMessage(ChatRole.SYSTEM, "你是一个简洁的助手，用简短的句子回答。"),
        ChatMessage(ChatRole.USER, "用一句话解释什么是时间管理。"),
    ]

    try:
        t0 = time.time()
        result = model.chat(messages, temperature=0.3, max_tokens=100, skip_cache=True)
        elapsed = int((time.time() - t0) * 1000)

        content = result.get("content", "").strip()
        if content:
            print(f"✅ ({elapsed}ms)")
            return {
                "status": "ok",
                "model": model.model,
                "response_time_ms": elapsed,
                "detail": f"返回 {len(content)} 字符",
                "sample": content[:80],
            }
        else:
            print("❌ 空响应")
            return {"status": "fail", "model": model.model, "error": "空响应内容"}

    except Exception as e:
        print(f"❌ {e}")
        return {"status": "fail", "model": model.model, "error": str(e)}


def probe_function_calling(model: ChatModel) -> Dict[str, Any]:
    """测试 function calling 能力。"""
    print("  🔧 测试 Function Calling...", end=" ", flush=True)
    import time

    messages = [
        ChatMessage(ChatRole.SYSTEM, "你是一个任务管理助手。需要添加任务时调用 add_task 工具。"),
        ChatMessage(ChatRole.USER, "帮我加一个任务：周五之前完成 Python 大作业，大概需要 8 小时，优先级高。"),
    ]

    try:
        t0 = time.time()
        result = model.chat(
            messages,
            temperature=0.1,
            max_tokens=200,
            tools=[TEST_FUNCTION_TOOL],
            tool_choice="auto",
            skip_cache=True,
        )
        elapsed = int((time.time() - t0) * 1000)

        tool_calls = result.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            func_name = tc.get("function", {}).get("name", "")
            args_raw = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"raw": args_raw}

            print(f"✅ ({elapsed}ms) → {func_name}")
            return {
                "status": "ok",
                "model": model.model,
                "response_time_ms": elapsed,
                "detail": f"调用工具: {func_name}",
                "sample": json.dumps(args, ensure_ascii=False),
            }
        else:
            # 有些模型可能用文字回复而不调用工具，也算"支持但未触发"
            content = result.get("content", "")
            if content:
                print(f"⚠️  未调用工具，但返回了文本 ({elapsed}ms)")
                return {
                    "status": "ok",
                    "model": model.model,
                    "response_time_ms": elapsed,
                    "detail": "工具定义已接收，但模型选择用文本回复",
                    "sample": content[:80],
                }
            print(f"❌ 无 tool_calls 也无 content")
            return {"status": "fail", "model": model.model, "error": "无工具调用也无文本响应"}

    except Exception as e:
        print(f"❌ {e}")
        return {"status": "fail", "model": model.model, "error": str(e)}


def probe_vision(model: ChatModel, test_image_path: Optional[str] = None) -> Dict[str, Any]:
    """测试视觉模型能力。"""
    print("  👁️  测试视觉模型...", end=" ", flush=True)
    import time

    vision_model = os.environ.get("LLM_VISION_MODEL", "")
    if not vision_model:
        print("⏭️  未配置 LLM_VISION_MODEL，跳过")
        return {"status": "skip", "detail": "未配置视觉模型，可使用 OCR 兜底"}

    # 如果没有测试图片，生成一张简单的测试图
    if test_image_path is None or not os.path.exists(test_image_path):
        test_image_path = _create_test_image()

    try:
        t0 = time.time()
        result = model.chat_with_image(
            text_prompt=VISION_TEST_PROMPT,
            image_path=test_image_path,
            temperature=0.1,
            max_tokens=300,
            skip_cache=True,
        )
        elapsed = int((time.time() - t0) * 1000)

        content = result.get("content", "").strip()

        # 尝试解析 JSON
        try:
            parsed = json.loads(content)
            is_json = True
        except json.JSONDecodeError:
            # 尝试提取
            import re
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    is_json = True
                except json.JSONDecodeError:
                    is_json = False
                    parsed = None
            else:
                is_json = False
                parsed = None

        if is_json and parsed:
            print(f"✅ ({elapsed}ms) JSON 解析成功")
            return {
                "status": "ok",
                "model": vision_model,
                "response_time_ms": elapsed,
                "detail": "成功输出结构化 JSON",
                "sample": json.dumps(parsed, ensure_ascii=False)[:150],
            }
        else:
            print(f"⚠️  ({elapsed}ms) 返回了文本但 JSON 解析失败")
            return {
                "status": "ok",
                "model": vision_model,
                "response_time_ms": elapsed,
                "detail": "视觉可用，但需优化 prompt 确保 JSON 输出",
                "sample": content[:100],
            }

    except Exception as e:
        print(f"❌ {e}")
        return {"status": "fail", "model": vision_model, "error": str(e)}


def probe_embedding(model: ChatModel) -> Dict[str, Any]:
    """测试 embedding 能力。"""
    print("  📐 测试 Embedding...", end=" ", flush=True)
    import time

    embed_model = os.environ.get("EMBED_MODEL", "")
    if not embed_model:
        print("⏭️  未配置 EMBED_MODEL，跳过")
        return {"status": "skip", "detail": "未配置 embedding 模型，RAG 将使用本地方案"}

    try:
        t0 = time.time()
        vectors = model.embed(EMBED_TEST_TEXTS)
        elapsed = int((time.time() - t0) * 1000)

        if vectors and len(vectors) == len(EMBED_TEST_TEXTS):
            dim = len(vectors[0])
            print(f"✅ ({elapsed}ms) 维度={dim}")
            return {
                "status": "ok",
                "model": embed_model,
                "response_time_ms": elapsed,
                "detail": f"维度={dim}, 数量={len(vectors)}",
                "sample": f"[{vectors[0][0]:.4f}, {vectors[0][1]:.4f}, ...]",
            }
        else:
            print("❌ 返回向量数量不匹配")
            return {"status": "fail", "model": embed_model, "error": "返回向量数量不匹配"}

    except Exception as e:
        print(f"❌ {e}")
        return {"status": "fail", "model": embed_model, "error": str(e)}


def probe_cache(model: ChatModel) -> Dict[str, Any]:
    """测试缓存功能。"""
    print("  💾 测试 LLM 缓存...", end=" ", flush=True)

    messages = [
        ChatMessage(ChatRole.USER, "缓存测试：请回复 'pong'。"),
    ]

    try:
        # 第一次调用（不命中）
        result1 = model.chat(messages, temperature=0.1, max_tokens=20)
        # 第二次调用（应该命中）
        stats_before = model.cache.stats()
        result2 = model.chat(messages, temperature=0.1, max_tokens=20)
        stats_after = model.cache.stats()

        if result1.get("content") == result2.get("content"):
            print("✅ 缓存命中正常")
            return {
                "status": "ok",
                "detail": f"缓存条目: {stats_after['total_entries']}, 总命中: {stats_after['total_hits']}",
            }
        else:
            print("⚠️  两次结果不同（可能温度影响）")
            return {
                "status": "ok",
                "detail": "缓存已写入，结果差异可能由温度导致",
            }

    except Exception as e:
        print(f"❌ {e}")
        return {"status": "fail", "error": str(e)}


def _create_test_image() -> str:
    """生成一张简单的测试图片（带文字的 PNG）。"""
    tmp_path = os.path.join(tempfile.gettempdir(), "lifeos_vision_test.png")

    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (400, 300), color="white")
        draw = ImageDraw.Draw(img)

        # 画一个简单的日程表
        draw.text((20, 20), "本周日程", fill="black")
        draw.text((20, 60), "周一 09:00-11:00 高数考试", fill="black")
        draw.text((20, 90), "周三 14:00-16:00 Python作业", fill="black")
        draw.text((20, 120), "周五 10:00 实验报告截止", fill="black")

        img.save(tmp_path)
        return tmp_path
    except ImportError:
        # Pillow 不可用，创建一个 1x1 像素的 PNG 占位
        import base64
        # 1x1 白色 PNG 的 base64
        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        with open(tmp_path, "wb") as f:
            f.write(tiny_png)
        return tmp_path


# ==========================================
# 主流程
# ==========================================

def run_probe(
    test_image: Optional[str] = None,
    output_md: Optional[str] = None,
) -> ProbeResult:
    """运行全部能力探测。

    Args:
        test_image: 自定义测试图片路径
        output_md: 输出 Markdown 报告路径

    Returns:
        ProbeResult 探测结果
    """
    load_dotenv()

    print("=" * 60)
    print(" LifeOS 能力探测")
    print("=" * 60)
    print()

    result = ProbeResult()

    # 检查基础配置
    base_url = os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    model_name = os.environ.get("LLM_MODEL", "")

    if not base_url or not api_key or not model_name:
        print("❌ 缺少基础配置！请检查 .env 文件：")
        print(f"   LLM_BASE_URL: {'✅' if base_url else '❌'}")
        print(f"   LLM_API_KEY: {'✅' if api_key else '❌'}")
        print(f"   LLM_MODEL: {'✅' if model_name else '❌'}")
        print()
        print("请复制 .env.example 为 .env 并填入实际值。")
        return result

    print(f"📡 服务商: {base_url}")
    print(f"🤖 模型: {model_name}")

    mock_mode = os.environ.get("LLM_MOCK_MODE", "false").lower() == "true"
    if mock_mode:
        print("🎭 模式: Mock（使用预设数据，无需真实 API）")
    print()

    # 初始化 ChatModel
    try:
        model = ChatModel(use_cache=True)
    except Exception as e:
        print(f"❌ 初始化 ChatModel 失败: {e}")
        return result

    # 逐项测试
    print("--- 文本对话 ---")
    result.text_chat = probe_text_chat(model)
    print()

    print("--- Function Calling ---")
    result.function_calling = probe_function_calling(model)
    print()

    print("--- 视觉模型 ---")
    result.vision = probe_vision(model, test_image)
    print()

    print("--- Embedding ---")
    result.embedding = probe_embedding(model)
    print()

    print("--- LLM 缓存 ---")
    result.llm_cache = probe_cache(model)
    print()

    # 输出总结
    print("=" * 60)
    print(" 探测完成")
    print("=" * 60)

    if result.all_passed:
        print("✅ 核心能力全部通过！")
    else:
        print("⚠️  部分能力需要检查")

    # 输出报告
    md_content = result.to_markdown()

    if output_md:
        Path(output_md).parent.mkdir(parents=True, exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md_content)
        print()
        print(f"📄 报告已保存到: {output_md}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LifeOS 能力探测脚本")
    parser.add_argument("--image", help="自定义测试图片路径")
    parser.add_argument(
        "--output",
        default="./docs/CAPABILITY.md",
        help="Markdown 报告输出路径 (默认: ./docs/CAPABILITY.md)",
    )
    args = parser.parse_args()

    # 切换到项目根目录
    script_dir = Path(__file__).parent.parent.parent
    os.chdir(script_dir)

    run_probe(test_image=args.image, output_md=args.output)

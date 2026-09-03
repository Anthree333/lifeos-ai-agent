"""统一 LLM 聊天模型客户端。

支持 OpenAI-compatible API，可独立配置文本/视觉/embedding 模型。
所有调用经过 LLMCache，命中直接返回。
"""
from __future__ import annotations

import os
import json
import base64
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI

from .cache import LLMCache


class ChatRole(str, Enum):
    """消息角色。"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage:
    """聊天消息。"""

    def __init__(self, role: ChatRole, content: Union[str, List[Dict[str, Any]]]):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role.value, "content": self.content}


class ChatModel:
    """统一的聊天模型接口。

    支持：
    - 文本对话
    - Function calling / tool calling
    - 视觉理解（通过 image_url 消息）
    - 响应格式（JSON mode）

    所有调用自动经过缓存。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache: Optional[LLMCache] = None,
        use_cache: bool = True,
    ):
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.use_cache = use_cache

        # Mock 模式检测
        self.mock_mode = os.environ.get("LLM_MOCK_MODE", "false").lower() == "true"

        if not self.mock_mode and (not self.base_url or not self.api_key):
            raise ValueError(
                "LLM_BASE_URL 和 LLM_API_KEY 必须配置（环境变量或构造参数），"
                "或者设置 LLM_MOCK_MODE=true 启用 Mock 模式"
            )

        timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "20"))
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", "1"))
        self._client = (
            None
            if self.mock_mode
            else OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
        )
        self.cache = cache if cache is not None else LLMCache()

    @classmethod
    def from_env(cls, **overrides):
        """从环境变量构造 ChatModel，便于 UI/脚本直接接入。"""
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", ""),
            api_key=os.environ.get("LLM_API_KEY", ""),
            model=os.environ.get("LLM_MODEL", ""),
            **overrides,
        )

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,  # "json_object" | None
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        skip_cache: bool = False,
    ) -> Dict[str, Any]:
        """发起聊天对话。

        Args:
            messages: 消息列表
            temperature: 温度
            max_tokens: 最大输出 token 数
            response_format: 响应格式（json_object 表示 JSON 模式）
            tools: 工具定义（function calling）
            tool_choice: 工具选择策略
            skip_cache: 是否跳过缓存

        Returns:
            模型响应字典，包含 content 和/或 tool_calls
        """
        # 构建缓存键
        if self.use_cache and not skip_cache:
            cache_key = self._build_cache_key(
                messages, temperature, max_tokens, response_format, tools, tool_choice
            )
            cached = self.cache.get(cache_key, self.model)
            if cached is not None:
                return cached

        # 构建请求参数
        api_messages = [m.to_dict() for m in messages]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        # Mock 模式：返回预设数据
        if self.mock_mode:
            result = self._mock_chat_response(messages, response_format, tools)
            if self.use_cache and not skip_cache and tools is None:
                cache_key = self._build_cache_key(
                    messages, temperature, max_tokens, response_format, tools, tool_choice
                )
                self.cache.set(cache_key, self.model, result, _preview(messages))
            return result

        # 调用 API
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        result = self._parse_choice(choice)

        # 写入缓存（空回复不缓存，避免污染后续请求）
        has_content = bool((result.get("content") or "").strip())
        if has_content or result.get("tool_calls"):
            if self.use_cache and not skip_cache and tools is None:
                cache_key = self._build_cache_key(
                    messages, temperature, max_tokens, response_format, tools, tool_choice
                )
                self.cache.set(cache_key, self.model, result, _preview(messages))

        return result

    def chat_json(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        skip_cache: bool = False,
    ) -> Dict[str, Any]:
        """对话并解析 JSON 输出。

        优先使用 JSON mode，失败则手动解析。
        """
        result = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json_object",
            skip_cache=skip_cache,
        )
        content = result.get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            return _extract_json(content)

    def chat_with_image(
        self,
        text_prompt: str,
        image_path: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        skip_cache: bool = False,
    ) -> Dict[str, Any]:
        """带图片的视觉对话。

        Args:
            text_prompt: 文本提示
            image_path: 本地图片路径
            temperature: 温度
            max_tokens: 最大 token 数
            model: 视觉模型名（默认用 LLM_VISION_MODEL 环境变量）
            skip_cache: 是否跳过缓存

        Returns:
            模型响应字典
        """
        vision_model = model or os.environ.get("LLM_VISION_MODEL", self.model)

        # 读取图片并转 base64
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # 判断图片格式
        if image_path.lower().endswith(".png"):
            mime = "image/png"
        else:
            mime = "image/jpeg"

        image_url = f"data:{mime};base64,{img_data}"

        content = [
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

        messages = [ChatMessage(ChatRole.USER, content)]

        # 视觉模型可能用不同的 base_url
        vision_base_url = os.environ.get("LLM_VISION_BASE_URL", self.base_url)
        vision_api_key = os.environ.get("LLM_VISION_API_KEY", self.api_key)

        if vision_base_url != self.base_url or vision_api_key != self.api_key:
            # 用独立客户端
            vision_client = OpenAI(base_url=vision_base_url, api_key=vision_api_key)
        else:
            vision_client = self._client

        # 视觉缓存键用图片路径哈希
        if self.use_cache and not skip_cache:
            import hashlib
            with open(image_path, "rb") as f:
                img_hash = hashlib.sha256(f.read()).hexdigest()
            cache_key = f"vision:{vision_model}:{img_hash}:{hashlib.sha256(text_prompt.encode()).hexdigest()}"
            cached = self.cache.get(cache_key, vision_model)
            if cached is not None:
                return cached

        # Mock 模式：返回预设的视觉解析结果
        if self.mock_mode:
            result = self._mock_vision_response(text_prompt)
            if self.use_cache and not skip_cache:
                import hashlib
                with open(image_path, "rb") as f:
                    img_hash = hashlib.sha256(f.read()).hexdigest()
                cache_key = f"vision:{vision_model}:{img_hash}:{hashlib.sha256(text_prompt.encode()).hexdigest()}"
                self.cache.set(cache_key, vision_model, result, f"[视觉] {text_prompt[:50]}")
            return result

        kwargs: Dict[str, Any] = {
            "model": vision_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = vision_client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        result = self._parse_choice(choice)

        # 写入缓存
        if self.use_cache and not skip_cache:
            import hashlib
            with open(image_path, "rb") as f:
                img_hash = hashlib.sha256(f.read()).hexdigest()
            cache_key = f"vision:{vision_model}:{img_hash}:{hashlib.sha256(text_prompt.encode()).hexdigest()}"
            self.cache.set(cache_key, vision_model, result, f"[视觉] {text_prompt[:50]}")

        return result

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """获取文本 embedding。

        Args:
            texts: 文本列表
            model: embedding 模型名

        Returns:
            向量列表
        """
        embed_model = model or os.environ.get("EMBED_MODEL", "text-embedding-v2")
        embed_base_url = os.environ.get("EMBED_BASE_URL", self.base_url)
        embed_api_key = os.environ.get("EMBED_API_KEY", self.api_key)

        if embed_base_url != self.base_url or embed_api_key != self.api_key:
            client = OpenAI(base_url=embed_base_url, api_key=embed_api_key)
        else:
            client = self._client

        # Mock 模式：返回随机向量
        if self.mock_mode:
            import random
            dim = 1536  # 常见 embedding 维度
            return [
                [random.uniform(-0.1, 0.1) for _ in range(dim)]
                for _ in texts
            ]

        response = client.embeddings.create(model=embed_model, input=texts)
        return [item.embedding for item in response.data]

    def _build_cache_key(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[str],
    ) -> str:
        """构建缓存键。"""
        import hashlib
        content = json.dumps(
            {
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "tools": tools,
                "tool_choice": tool_choice,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def _parse_choice(self, choice) -> Dict[str, Any]:
        """解析模型响应为统一字典。"""
        message = choice.message
        result: Dict[str, Any] = {
            "content": message.content or "",
            "role": message.role,
            "finish_reason": choice.finish_reason,
        }

        # 解析 tool_calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            result["tool_calls"] = tool_calls

        return result

    # ==========================================
    # Mock 模式：预设响应（无需真实 API）
    # ==========================================

    def _mock_chat_response(
        self,
        messages: List[ChatMessage],
        response_format: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """生成 Mock 聊天响应。

        根据用户消息内容智能返回不同的 mock 数据，便于开发测试。
        """
        user_text = ""
        for m in messages:
            if m.role == ChatRole.USER and isinstance(m.content, str):
                user_text = m.content
                break

        # 如果有 tools，模拟 function calling
        if tools:
            # 找第一个工具
            tool_name = tools[0]["function"]["name"]
            # 根据工具名生成合理的 mock 参数
            if tool_name == "add_task":
                args = json.dumps({
                    "title": "Python 大作业",
                    "duration_minutes": 480,
                    "deadline": "2024-03-20",
                    "priority": "high"
                }, ensure_ascii=False)
            else:
                args = "{}"

            return {
                "content": "",
                "role": "assistant",
                "finish_reason": "tool_calls",
                "tool_calls": [{
                    "id": "mock_call_001",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": args,
                    },
                }],
            }

        # JSON mode
        if response_format == "json_object":
            # 根据消息关键词返回不同的 mock JSON
            if "解析" in user_text or "事件" in user_text:
                content = json.dumps({
                    "event_type": "progress_report",
                    "task_title": "Python 大作业",
                    "progress": 0.3,
                    "note": "完成了基础框架部分"
                }, ensure_ascii=False)
            elif "分解" in user_text or "任务" in user_text:
                content = json.dumps({
                    "tasks": [
                        {"title": "需求分析", "estimated_minutes": 60, "energy_level": "high", "deadline_type": "soft"},
                        {"title": "编码实现", "estimated_minutes": 240, "energy_level": "high", "deadline_type": "hard"},
                        {"title": "测试调试", "estimated_minutes": 120, "energy_level": "medium", "deadline_type": "hard"},
                        {"title": "文档撰写", "estimated_minutes": 60, "energy_level": "low", "deadline_type": "soft"},
                    ]
                }, ensure_ascii=False)
            elif "计划" in user_text or "变更" in user_text:
                content = json.dumps({
                    "summary": "已根据进度偏差重新调整计划",
                    "reason_tags": ["进度偏差", "硬截止保障"],
                    "changes_count": 3,
                    "sacrificed_count": 1,
                }, ensure_ascii=False)
            else:
                content = json.dumps({
                    "status": "ok",
                    "message": "这是 Mock 模式的 JSON 响应",
                    "mock": True,
                }, ensure_ascii=False)

            return {
                "content": content,
                "role": "assistant",
                "finish_reason": "stop",
            }

        # 普通文本对话
        if "时间管理" in user_text:
            reply = "时间管理就是合理分配精力，优先完成重要且紧急的任务，同时给自己留出缓冲时间应对意外。"
        elif "你好" in user_text or "hi" in user_text.lower():
            reply = "你好！我是 LifeOS 助手，有什么可以帮你的吗？（当前为 Mock 模式）"
        elif "pong" in user_text.lower() or "ping" in user_text.lower():
            reply = "pong"
        else:
            reply = f"[Mock 模式] 收到你的消息：{user_text[:30]}...\n\n这是预设的模拟回复，配置真实 API 后将调用实际大模型。"

        return {
            "content": reply,
            "role": "assistant",
            "finish_reason": "stop",
        }

    def _mock_vision_response(self, text_prompt: str) -> Dict[str, Any]:
        """生成 Mock 视觉解析响应。"""
        result = {
            "events": [
                {
                    "title": "高等数学期中考试",
                    "date": "2024-03-20",
                    "start_time": "09:00",
                    "end_time": "11:00",
                    "event_type": "exam"
                },
                {
                    "title": "Python 大作业截止",
                    "date": "2024-03-18",
                    "start_time": "23:59",
                    "end_time": "23:59",
                    "event_type": "assignment"
                }
            ],
            "confidence": 0.92,
            "mock": True
        }
        return {
            "content": json.dumps(result, ensure_ascii=False, indent=2),
            "role": "assistant",
            "finish_reason": "stop",
        }


def _preview(messages: List[ChatMessage]) -> str:
    """生成消息预览（用于缓存表展示）。"""
    for m in messages:
        if m.role == ChatRole.USER and isinstance(m.content, str):
            return m.content[:100]
    return ""


def _extract_json(text: str) -> Dict[str, Any]:
    """从文本中提取 JSON。"""
    # 尝试找 ```json ... ``` 块
    import re
    match = re.search(r"```json\s*(.+?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {"raw_content": text}

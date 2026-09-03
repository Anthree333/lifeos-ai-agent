"""输入解析器。

将用户自然语言消息解析为结构化的内部状态/事件：
- 进度汇报（"昨天完成了 30%"）
- 新目标（"我要准备高数考试"）
- 事件通知（"考试改期了"）
- 状态查询（"我现在进度怎么样？"）

使用 LLM JSON mode 保证结构化输出。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from ..llm.chat_model import ChatModel, ChatMessage, ChatRole
from ..models import LifeEvent, ExecutionRecord, EventType


class ParseIntent(str, Enum):
    """解析意图类型。"""
    PROGRESS_REPORT = "progress_report"  # 进度汇报
    NEW_GOAL = "new_goal"                # 新目标
    NEW_EVENT = "new_event"              # 新事件
    STATUS_QUERY = "status_query"        # 状态查询
    UNKNOWN = "unknown"                  # 无法识别


@dataclass
class ParseResult:
    """解析结果。"""
    intent: ParseIntent = ParseIntent.UNKNOWN
    confidence: float = 0.0

    # 根据 intent 不同，填充不同字段
    execution_records: List[ExecutionRecord] = field(default_factory=list)
    life_events: List[LifeEvent] = field(default_factory=list)
    goal_data: Optional[Dict] = None
    query_type: Optional[str] = None

    raw_response: Dict = field(default_factory=dict)


SYSTEM_PROMPT = """你是 LifeOS 的输入解析器。你的任务是将用户的自然语言消息解析为结构化数据。

请严格按以下 JSON 格式输出：
{
  "intent": "progress_report | new_goal | new_event | status_query | unknown",
  "confidence": 0.0-1.0,
  "data": { ... }
}

各意图的 data 格式：

1. progress_report（进度汇报）：
   - 当用户说"完成了 X%"、"做了 Y 小时"、"进度..."等时触发
   - data 格式：
   {
     "updates": [
     {
       "task_title": "任务名称（从上下文推断）",
       "progress": 0.3,
       "actual_minutes": 120,
       "note": "用户原话摘要"
     }
   ]
  }
   - progress 表示本次汇报后该任务达到的总进度（30% = 0.3）
   - 只有用户明确说“又/再/额外完成”时才把它理解为增量

2. new_goal（新目标）：
   - 当用户说"我要..."、"准备..."、"目标是..."等时触发
   - data 格式：
   {
     "title": "目标标题",
     "description": "详细描述",
     "deadline": "YYYY-MM-DD 或 null",
     "weight": 0.5
   }

3. new_event（新事件）：
   - 当用户说"考试改期"、"生病了"、"有活动"等时触发
   - data 格式：
   {
     "events": [
       {
         "title": "事件标题",
         "event_type": "exam | assignment | schedule_change | illness | activity | notification | other",
         "event_time": "ISO datetime 或 null",
         "change_content": "变更内容描述"
       }
     ]
   }

4. status_query（状态查询）：
   - 当用户问"怎么样了"、"进度如何"、"看看计划"等时触发
   - data 格式：
   {
     "query_type": "progress | plan | risk | all"
   }

5. unknown（无法识别）：
   - data 为空对象 {}

注意：
- 只输出 JSON，不要其他文字
- 从用户消息中尽可能提取精确信息
- 如果信息不足，留空但不要编造
- 进度是 0-1 之间的小数（30% = 0.3）
- 用户消息中会给出今天的日期和时间。涉及相对/绝对时间（明天、下周六、9月19日等）时，必须结合今天日期换算：
  - new_goal 的 deadline 输出 "YYYY-MM-DD"
  - new_event 的 event_time 输出 "YYYY-MM-DD HH:MM"（24小时制），能推断到具体时刻就给具体时刻，推断不到时刻就只给 "YYYY-MM-DD"
- 一条消息同时提到目标和当前进度时（如"X日前要交Y，已经做了一半"），归类为 new_goal，进度信息留待后续汇报
"""


def _clean_str(value: Any, default: str = "") -> str:
    """清洗 LLM 返回的字符串字段：None/非字符串/空白时用默认值。"""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _clean_float(value: Any, default: float = 0.0) -> float:
    """清洗 LLM 返回的数值字段。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_event_type(value: Any) -> EventType:
    """清洗事件类型，非法值回退为 other。"""
    try:
        return EventType(_clean_str(value, "other"))
    except ValueError:
        return EventType.OTHER


class InputParser:
    """输入解析器。"""

    def __init__(self, llm: Optional[ChatModel] = None):
        self.llm = llm

    def parse(
        self,
        user_message: str,
        context: Optional[Dict] = None,
        now: Optional[datetime] = None,
    ) -> ParseResult:
        """解析用户消息。

        Args:
            user_message: 用户自然语言输入
            context: 可选上下文（当前任务列表等，帮助解析）
            now: 当前时间（用于换算相对日期）

        Returns:
            ParseResult 解析结果
        """
        if now is None:
            now = datetime.now()

        # 明确的目标句优先走确定性规则，避免模型误判成“事件”
        if self._is_explicit_goal_phrase(user_message):
            return self._rule_based_parse(user_message, now=now)

        # Mock 模式或 LLM 不可用时，用规则解析兜底
        if self.llm is None or self.llm.mock_mode:
            return self._rule_based_parse(user_message, now=now)

        # 构建上下文提示
        context_prompt = ""
        if context and context.get("tasks"):
            task_titles = [t["title"] for t in context["tasks"][:5]]
            context_prompt = f"\n当前任务列表：{', '.join(task_titles)}"
        if context and context.get("confirming"):
            context_prompt += (
                "\n注意：这是用户对上一条消息的确认回复，请把上面的消息归类为最具体的意图"
                "（new_goal / new_event / progress_report / status_query），除非它明显不是。"
            )

        weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        date_prompt = (
            f"\n今天是 {now.strftime('%Y-%m-%d %H:%M')}（{weekday_names[now.weekday()]}）。"
        )

        user_prompt = f"用户消息：{user_message}{context_prompt}{date_prompt}\n\n请解析并输出 JSON。"

        messages = [
            ChatMessage(ChatRole.SYSTEM, SYSTEM_PROMPT),
            ChatMessage(ChatRole.USER, user_prompt),
        ]

        # LLM 解析失败时重试一次（跳过缓存），仍失败再用规则兜底
        for attempt, skip in enumerate((False, True)):
            try:
                response = self.llm.chat_json(
                    messages, temperature=0.1, skip_cache=skip
                )
                if response:
                    return self._build_result(response, now=now)
            except Exception:
                if attempt == 1:
                    break

        # LLM 失败，用规则兜底
        return self._rule_based_parse(user_message, now=now)

    def _build_result(
        self,
        response: Dict,
        now: Optional[datetime] = None,
    ) -> ParseResult:
        """从 LLM 响应构建 ParseResult。"""
        if now is None:
            now = datetime.now()
        intent_str = response.get("intent", "unknown")
        try:
            intent = ParseIntent(intent_str)
        except ValueError:
            intent = ParseIntent.UNKNOWN

        confidence = response.get("confidence", 0.5)
        data = response.get("data", {})

        result = ParseResult(
            intent=intent,
            confidence=confidence,
            raw_response=response,
        )

        # 根据意图解析具体数据
        if intent == ParseIntent.PROGRESS_REPORT:
            updates = data.get("updates", [])
            for u in updates:
                record = ExecutionRecord(
                    task_id="",  # 后续匹配任务
                    actual_minutes=int(_clean_float(u.get("actual_minutes"), 0)),
                    progress_delta=_clean_float(u.get("progress"), 0.0),
                    note=_clean_str(u.get("note")),
                )
                # 暂存 task_title 用于后续匹配
                record._task_title_hint = _clean_str(u.get("task_title"))
                result.execution_records.append(record)

        elif intent == ParseIntent.NEW_EVENT:
            events = data.get("events", [])
            for e in events:
                event_time = e.get("event_time")
                event = LifeEvent(
                    title=_clean_str(e.get("title"), "未命名事件"),
                    event_type=_clean_event_type(e.get("event_type")),
                    event_time=event_time if isinstance(event_time, str) else None,
                    change_content=_clean_str(e.get("change_content")),
                    user_confirmed=False,
                    confidence=confidence,
                )
                result.life_events.append(event)

        elif intent == ParseIntent.NEW_GOAL:
            goal_data = dict(data or {})
            goal_data["title"] = _clean_str(goal_data.get("title"), "新目标")
            goal_data["description"] = _clean_str(goal_data.get("description"))
            weight = _clean_float(goal_data.get("weight"), 0.5)
            goal_data["weight"] = min(max(weight, 0.0), 1.0)
            deadline = goal_data.get("deadline")
            if not isinstance(deadline, str) or not deadline.strip():
                deadline = None
            if deadline is None:
                deadline = self._parse_deadline(
                    goal_data.get("description") or "",
                    now=now,
                )
            elif not self._is_iso_date(str(deadline)):
                deadline = self._parse_deadline(str(deadline), now=now)
            goal_data["deadline"] = deadline
            result.goal_data = goal_data

        elif intent == ParseIntent.STATUS_QUERY:
            result.query_type = _clean_str(data.get("query_type"), "all")

        return result

    @staticmethod
    def _is_iso_date(value: str) -> bool:
        """判断字符串是否是 ISO 日期/时间。"""
        if not value:
            return False
        try:
            datetime.fromisoformat(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _is_explicit_goal_phrase(text: str) -> bool:
        """判断是否明显在提出目标，而不是陈述已发生的事件。"""
        if not any(word in text for word in ("我要", "我想", "目标是", "需要")):
            return False
        action_words = (
            "参加", "提交", "准备", "复习", "学习", "完成",
            "考试", "比赛", "报告", "项目", "作业", "论文", "写",
        )
        return any(word in text for word in action_words)

    @staticmethod
    def _clean_goal_title(text: str) -> str:
        """把用户原话整理成更短的标题。"""
        cleaned = text.strip()
        # 去掉开头的日期和“我要/我想/我需要”
        cleaned = re.sub(
            r"^(?:\d{4}\s*年\s*)?\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?"
            r"\s*[，,]?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"^\s*(?:我(?:要|想|需要)?|目标是)\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(r"我(?:要|想|需要)?", "", cleaned)
        cleaned = cleaned.strip(" ，。,.，：:；;")
        return (cleaned or text).strip()[:50]

    @staticmethod
    def _parse_deadline(
        text: str,
        now: Optional[datetime] = None,
    ) -> Optional[str]:
        """从中文日期表达中提取 ISO 日期。"""
        if not text:
            return None
        if now is None:
            now = datetime.now()
        today = now.date()

        # ISO 形式
        iso = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
        if iso:
            try:
                return date(
                    int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
                ).isoformat()
            except ValueError:
                pass

        # 2026年9月19日 / 9月19日 / 9月19
        zh = re.search(
            r"(?:(?P<year>\d{4})\s*年\s*)?"
            r"(?P<month>\d{1,2})\s*月\s*"
            r"(?P<day>\d{1,2})\s*(?:日|号)?",
            text,
        )
        if zh:
            month = int(zh.group("month"))
            day = int(zh.group("day"))
            year = int(zh.group("year")) if zh.group("year") else today.year
            try:
                result = date(year, month, day)
                if not zh.group("year") and result < today:
                    result = date(year + 1, month, day)
                return result.isoformat()
            except ValueError:
                return None

        # 相对日期
        weekday_map = {
            "一": 0, "二": 1, "三": 2, "四": 3,
            "五": 4, "六": 5, "日": 6, "天": 6,
        }
        rel = {
            "明天": 1,
            "后天": 2,
            "大后天": 3,
            "明晚": 1,
        }
        for word, delta in rel.items():
            if word in text:
                return (today + timedelta(days=delta)).isoformat()

        next_week = re.search(r"下(?:周|星期)([一二三四五六日天])", text)
        if next_week:
            target = weekday_map[next_week.group(1)]
            delta = (target - today.weekday()) % 7
            return (today + timedelta(days=7 if delta == 0 else delta)).isoformat()

        this_week = re.search(r"(?:本周|这周|周)([一二三四五六日天])", text)
        if this_week:
            target = weekday_map[this_week.group(1)]
            delta = (target - today.weekday()) % 7
            return (today + timedelta(days=delta)).isoformat()

        return None

    def _rule_based_parse(
        self,
        user_message: str,
        now: Optional[datetime] = None,
    ) -> ParseResult:
        """基于规则的兜底解析（Mock 模式使用）。"""
        msg = user_message.lower()

        # 状态查询检测（先于进度汇报，因为"进度如何"是查询不是汇报）
        if any(w in user_message for w in ["怎么样", "如何", "看看", "状态", "进度如何", "计划", "情况"]):
            # 排除纯数字百分比的情况（那是汇报）
            import re
            if not re.search(r"\d+\s*%", user_message) or "怎么样" in user_message or "如何" in user_message:
                return ParseResult(
                    intent=ParseIntent.STATUS_QUERY,
                    confidence=0.7,
                    query_type="all",
                )

        # 进度汇报检测
        import re
        progress_match = re.search(r"(\d+)\s*%", user_message)
        if "完成" in msg or "做完" in msg or progress_match:
            progress = 0.0
            if progress_match:
                progress = int(progress_match.group(1)) / 100.0

            # 提取任务名（简单 heuristic）
            task_title = "未知任务"
            for keyword in ["大作业", "考试", "复习", "作业", "项目", "报告", "练习", "做题"]:
                if keyword in user_message:
                    # 找关键词前后的词
                    idx = user_message.find(keyword)
                    start = max(0, idx - 6)
                    task_title = user_message[start:idx + len(keyword)]
                    break

            record = ExecutionRecord(
                task_id="",
                actual_minutes=0,
                progress_delta=progress,
                note=user_message,
            )
            record._progress_is_delta = any(
                word in user_message for word in ["又", "再", "追加", "新增", "额外"]
            )
            record._task_title_hint = task_title

            return ParseResult(
                intent=ParseIntent.PROGRESS_REPORT,
                confidence=0.7,
                execution_records=[record],
            )

        # 新目标检测
        if any(w in msg for w in ["我要", "我想", "准备", "目标是", "开始"]):
            return ParseResult(
                intent=ParseIntent.NEW_GOAL,
                confidence=0.6,
                goal_data={
                    "title": self._clean_goal_title(user_message),
                    "description": user_message,
                    "deadline": self._parse_deadline(user_message, now=now),
                    "weight": 0.5,
                },
            )

        # 生病检测
        if any(w in msg for w in ["生病", "感冒", "发烧", "不舒服", "请假"]):
            event = LifeEvent(
                title="生病请假",
                event_type=EventType.ILLNESS,
                change_content=user_message,
                user_confirmed=True,
                confidence=0.9,
            )
            return ParseResult(
                intent=ParseIntent.NEW_EVENT,
                confidence=0.9,
                life_events=[event],
            )

        # 状态查询检测
        if any(w in msg for w in ["怎么样", "如何", "看看", "状态", "进度如何", "计划"]):
            return ParseResult(
                intent=ParseIntent.STATUS_QUERY,
                confidence=0.7,
                query_type="all",
            )

        # 默认：无法识别
        return ParseResult(
            intent=ParseIntent.UNKNOWN,
            confidence=0.3,
        )

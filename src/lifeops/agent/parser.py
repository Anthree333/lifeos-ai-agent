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
    CANCEL_PLAN = "cancel_plan"          # 取消/删除已有计划
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
    cancel_data: Optional[Dict] = None

    raw_response: Dict = field(default_factory=dict)


SYSTEM_PROMPT = """你是 LifeOS 的输入解析器。你的任务是将用户的自然语言消息解析为结构化数据。

请严格按以下 JSON 格式输出：
{
  "intent": "progress_report | new_goal | new_event | status_query | cancel_plan | unknown",
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

5. cancel_plan（取消已有计划/目标/任务/安排）：
   - 当用户说"取消…计划"、"删除…任务"、"别安排…"、"清空所有计划"等时触发
   - data 格式：
   {
     "target_type": "goal | task | commitment | date | all",
     "keyword": "要取消的对象名称关键词，如'高数'；用户没给出名称时为空字符串",
     "date": "YYYY-MM-DD 或 null（只取消某一天的全部安排时才填）"
   }
   - target_type 取值说明：
     - goal：取消某个目标（会连带删除它分解出的任务）
     - task：取消某个/某些任务
     - commitment：取消课表或固定安排
     - date：取消 date 指定那天的全部安排
     - all：清空所有目标与任务
   - 关键词必须来自用户原话，禁止编造；用户没说清是哪个计划时 keyword 留空、target_type 用 goal

6. unknown（无法识别）：
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
- 目标/事件标题必须忠实用户原话或上下文中已有的内容，禁止编造用户没提到的具体事项。
  例如用户说"八点去学习"，标题只能是"学习"这类原话词，绝不能臆造成"打篮球/打球"等。
- "去学习/复习/写作业/背单词"这类用户给自己安排的任务时间点，不是一次性外部活动（activity/event）；
  如果句中同时给出了要做的事和时间点（如"八点去学习英语"、"今晚复习高数"），归类为 new_goal，
  title 取活动内容（如"学习英语"、"复习高数"），deadline/event_time 按给出的时间换算；
  只有既没说做什么、也没说时间（如单纯"去学习"）才输出 unknown，不要虚构事件。
- 消息中没给出的信息（如 event_time、deadline）一律留空，禁止猜测或脑补。
- 用户明确说要"取消/删除/去掉/别安排/停掉"某个计划、目标、任务或安排时，归类为 cancel_plan，
  不要当成 status_query（即使句子里有"计划""安排"等词），也不要当成 new_goal。
- 用户说"不去/不去…了/不参加/不用去"某活动（如"八点不去打篮球了"）同样是取消意图（cancel_plan）：
  keyword 填活动名（如"打篮球"）。这绝不是 new_event，绝不能把它作为新事件加进日程。
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

        # 取消类语句优先用确定性规则识别，避免被"我要取消…"误判成新目标、
        # 或被"…计划"字样误判成状态查询
        cancel_data = self._parse_cancel_request(user_message, now=now)
        if cancel_data is not None:
            return ParseResult(
                intent=ParseIntent.CANCEL_PLAN,
                confidence=0.8,
                cancel_data=cancel_data,
            )

        # 明确的目标句优先走确定性规则，避免模型误判成“事件”
        if self._is_explicit_goal_phrase(user_message):
            return self._rule_based_parse(user_message, now=now)

        # Mock 模式或 LLM 不可用时，用规则解析兜底
        if self.llm is None or self.llm.mock_mode:
            return self._rule_based_parse(user_message, now=now)

        # 构建上下文提示
        context_prompt = ""
        if context and context.get("profile_name"):
            context_prompt += f"\n用户姓名：{context['profile_name']}"
        if context and context.get("tasks"):
            task_titles = [t["title"] for t in context["tasks"][:5]]
            context_prompt += f"\n当前任务列表：{', '.join(task_titles)}"
        if context and context.get("history"):
            rows = []
            for h in context["history"][:8]:
                role = "用户" if h.get("role") in ("user", "用户") else "助手"
                content = (h.get("content") or "").strip()
                if not content:
                    continue
                rows.append(f"{role}：{content[:150]}")
            if rows:
                context_prompt += (
                    "\n\n最近对话（仅供理解语境，当前用户消息才是唯一需要解析的对象）：\n"
                    + "\n".join(rows)
                )
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

        elif intent == ParseIntent.CANCEL_PLAN:
            target_type = _clean_str(data.get("target_type"), "goal")
            if target_type not in ("goal", "task", "commitment", "date", "all"):
                target_type = "goal"
            raw_date = data.get("date")
            result.cancel_data = {
                "target_type": target_type,
                "keyword": _clean_str(data.get("keyword")),
                "date": raw_date if isinstance(raw_date, str) and raw_date.strip() else None,
                "raw": _clean_str(data.get("raw")),
            }

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
        """判断是否明显在提出目标（NEW_GOAL），而非陈述事件或闲聊。

        NEW_GOAL 特征：表达要完成的任务/作品/考试，有隐含或明确的截止日期。
        NEW_EVENT 特征：已经安排好的日程（看医生/聚会/会议），是具体时间点事件。
        """
        # 先排除明显的事件词：已经安排好的日程不是目标
        event_keywords = (
            "去看医生", "去医院", "看病", "聚会", "聚餐", "约会",
            "开会", "会议", "讲座", "上课", "演出", "电影", "演唱会",
        )
        if any(kw in text for kw in event_keywords):
            return False

        # 核心判断：我 + 要/想/需要 + 动作词
        # 中间可以夹日期（如 "我10月15日要交..."）
        if re.search(r"我(?:.+?)(?:要|想|需要)", text):
            action_words = (
                "参加", "提交", "准备", "复习", "学习", "完成",
                "考试", "比赛", "报告", "项目", "作业", "论文",
                "交", "写", "做", "攻克", "拿下", "考研", "保研",
                "答辩", "面试", "出", "通过",
            )
            return any(word in text for word in action_words)
        return False

    @staticmethod
    def _is_affirmative_only(text: str) -> bool:
        """判断是否是纯粹的确认回复（好的/嗯/可以），用于 run() 里优先走确认分支。"""
        affirmative = {
            "好", "好的", "好呀", "好啊", "好嘞", "好哒", "好滴",
            "嗯", "嗯嗯", "行", "行的", "可以", "是的", "是", "对",
            "对的", "要", "需要", "确认", "安排", "ok", "okay", "yes",
            "没问题", "当然", "当然可以", "麻烦你了", "帮我安排",
            "这样安排", "就这么办", "没问题", "开始吧", "开始",
        }
        t = text.strip().lower().strip("，。,.！！?？")
        return len(t) <= 10 and t in affirmative

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

    # 取消意图相关的确定性词表
    _CANCEL_VERBS = (
        "取消", "删除", "去掉", "撤掉", "撤回", "移除", "清空", "别安排",
        "不用安排", "不要安排", "不安排", "停掉", "终止", "不要了",
        # 否定式：说"不去/不参加某活动"同样是在取消，而不是新增
        "不去", "不要去", "不用去", "不参加", "不去了",
    )
    # 这类否定式动词本身就足以表意，不要求句中出现"计划/任务"等名词
    _CANCEL_STRONG_VERBS = ("不去", "不要去", "不用去", "不参加", "不去了")
    _CANCEL_NOUNS = ("计划", "目标", "任务", "安排", "日程", "备考", "课", "课程", "课表")
    _CANCEL_STOPWORDS = (
        "帮我", "帮我把", "请", "把", "的", "了", "吧", "这个", "那个", "一下",
        "我", "要", "想", "给我", "所有", "全部", "今天", "明天", "后天", "当天",
    )
    # 剥离时间表达，让关键词更接近活动名（"八点打篮球" → "打篮球"）
    _TIME_WORDS_RE = re.compile(
        r"\d{1,2}[点:：]\d{0,2}分?\d{0,2}"
        r"|\d{1,2}点(?:半)?"
        r"|[一二两三四五六七八九十]+点(?:半)?"
        r"|早上|上午|中午|下午|傍晚|晚上|今晚|明晚|今早|明早"
    )

    def _parse_cancel_request(
        self,
        user_message: str,
        now: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """识别"取消/删除计划"类语句，返回 cancel_data；不是则返回 None。"""
        msg = user_message.strip()
        if not msg:
            return None

        verb = next((v for v in self._CANCEL_VERBS if v in msg), None)
        if verb is None:
            return None
        # "不去/不参加…"本身就是明确取消；其余动词必须提到被取消的对象
        if (
            verb not in self._CANCEL_STRONG_VERBS
            and not any(n in msg for n in self._CANCEL_NOUNS)
        ):
            return None

        rest = msg
        for v in self._CANCEL_VERBS:
            rest = rest.replace(v, " ")

        # 目标类型判定：整体清空 > 课表 > 任务 > 目标/计划
        if any(w in msg for w in ("所有计划", "全部计划", "所有目标", "全部目标",
                                  "所有任务", "全部任务", "清空")):
            target_type = "all"
        elif any(w in rest for w in ("课表", "课程", "上课", "的课")):
            target_type = "commitment"
        elif "任务" in rest:
            target_type = "task"
        elif any(w in rest for w in ("目标", "计划", "备考")):
            target_type = "goal"
        else:
            # 否定式活动（"不去打篮球了"）按关键词取消，落到 goal/task 兜底匹配
            target_type = "goal"

        # 取消某一天的全部安排
        date_str = None
        if any(w in msg for w in ("今天", "明天", "后天", "当天")):
            target_type = "date"
            parsed = self._parse_deadline(msg, now=now) if now is not None else None
            date_str = parsed[:10] if isinstance(parsed, str) else None

        # 提取关键词：剥掉动词、对象名词、口语停用词和时间表达
        keyword = rest
        for w in set(self._CANCEL_NOUNS) | set(self._CANCEL_STOPWORDS):
            keyword = keyword.replace(w, " ")
        keyword = self._TIME_WORDS_RE.sub(" ", keyword)
        keyword = re.sub(r"[\s，。、！？,.!?~～]+", "", keyword).strip()

        return {
            "target_type": target_type,
            "keyword": keyword,
            "date": date_str,
            "raw": msg,
        }

    def _rule_based_parse(
        self,
        user_message: str,
        now: Optional[datetime] = None,
    ) -> ParseResult:
        """基于规则的兜底解析（Mock 模式使用）。"""
        msg = user_message.lower()

        # 取消计划检测（必须放在状态查询之前，避免"取消…计划"被当成查询）
        cancel_data = self._parse_cancel_request(user_message, now=now)
        if cancel_data is not None:
            return ParseResult(
                intent=ParseIntent.CANCEL_PLAN,
                confidence=0.75,
                cancel_data=cancel_data,
            )

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

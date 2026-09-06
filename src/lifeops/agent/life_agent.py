"""LifeAgent 主类 — 五步执行法。

每轮对话执行固定五步：
1. 解析输入 → 结构化意图 + 事件
2. 状态更新 → 应用进度/事件到内部状态
3. 重规划判断 → 是否需要重新调度
4. 执行调度 → 调用确定性调度器
5. 生成解释 → 中文说明改了什么、为什么、牺牲了什么

维护 PlanSnapshot 快照链，支持回溯。
"""
from __future__ import annotations

import uuid
import difflib
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

from ..models import (
    StudentProfile, Goal, Task, Commitment,
    PlanSnapshot, TimeSlot, ChangeLog, ExecutionRecord, LifeEvent, EventType,
    TaskStatus, DeadlineType,
)
from ..scheduler import Scheduler, ScheduleResult
from ..storage.database import Database

from .parser import InputParser, ParseResult, ParseIntent
from .decomposer import GoalDecomposer
from .explainer import PlanExplainer, PlanDiff


@dataclass
class AgentResult:
    """Agent 执行结果。"""
    reply: str                                  # 给用户的回复（中文）
    snapshot: Optional[PlanSnapshot] = None     # 当前计划快照
    changed: bool = False                       # 计划是否变更
    change_log: Optional[ChangeLog] = None      # 变更记录
    intent: ParseIntent = ParseIntent.UNKNOWN   # 解析出的意图
    schedule_result: Optional[ScheduleResult] = None  # 调度结果
    parsed_screenshot: Optional[Any] = None     # 截图解析结果（供用户确认）
    rag_references: List[Any] = field(default_factory=list)  # RAG 引用列表


@dataclass
class PendingAction:
    """待用户确认的动作。

    - NEW_GOAL / NEW_EVENT：直接保存解析结果，确认时不再二次调用 LLM，避免结果漂移；
    - 闲聊反问（intent 为空）：只存原始消息，确认时再重新解析。
    """

    message: str
    intent: Optional[str] = None                        # ParseIntent.value
    goal_data: Optional[Dict[str, Any]] = None
    life_events: Optional[List[Dict[str, Any]]] = None  # LifeEvent 的 dict 表示
    cancel_data: Optional[Dict[str, Any]] = None        # 取消计划的请求
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "intent": self.intent,
            "goal_data": self.goal_data,
            "life_events": self.life_events,
            "cancel_data": self.cancel_data,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingAction":
        return cls(
            message=data.get("message", ""),
            intent=data.get("intent"),
            goal_data=data.get("goal_data"),
            life_events=data.get("life_events"),
            cancel_data=data.get("cancel_data"),
            created_at=data.get("created_at")
            or datetime.now().isoformat(),
        )


class LifeAgent:
    """LifeOS 智能体 — 五步执行法。"""

    def __init__(
        self,
        profile: StudentProfile,
        db: Optional[Database] = None,
        llm: Optional[Any] = None,
        scheduler: Optional[Scheduler] = None,
        parser: Optional[InputParser] = None,
        decomposer: Optional[GoalDecomposer] = None,
        explainer: Optional[PlanExplainer] = None,
        screenshot_parser: Optional[Any] = None,
        knowledge_base: Optional[Any] = None,
        retriever: Optional[Any] = None,
    ):
        self.profile = profile
        self.db = db
        self.llm = llm
        self.scheduler = scheduler or Scheduler(profile=profile)
        self.parser = parser or InputParser(llm=self.llm)
        self.decomposer = decomposer or GoalDecomposer(llm=self.llm)
        self.explainer = explainer or PlanExplainer(llm=self.llm)
        self.screenshot_parser = screenshot_parser
        self.knowledge_base = knowledge_base
        self.retriever = retriever

        # 待确认的动作：NEW_GOAL/NEW_EVENT/CANCEL_PLAN 或闲聊反问后等待用户确认
        # 会持久化到数据库，进程重启后仍可恢复（不依赖单条内存消息）
        self._pending_action: Optional[PendingAction] = None

        # 最近一次取消操作的摘要与被移除条目数量（用于回复与是否重规划）
        self._cancel_summary: str = ""
        self._cancel_removed_count: int = 0

        # 内存状态（如果没有数据库，则纯内存运行）
        self._goals: List[Goal] = []
        self._tasks: List[Task] = []
        self._commitments: List[Commitment] = []
        self._snapshots: List[PlanSnapshot] = []
        self._execution_records: List[ExecutionRecord] = []

        if self.db is not None:
            self.load_state()

    # ==========================================
    # 状态管理
    # ==========================================

    @property
    def current_snapshot(self) -> Optional[PlanSnapshot]:
        """当前最新快照。"""
        return self._snapshots[-1] if self._snapshots else None

    def plan_is_consistent(self) -> bool:
        """当前最新快照是否与现有任务/承诺一致。

        若最新快照仍引用已被删除的任务或承诺（例如某删除操作未及时触发
        重排、或历史遗留数据），则返回 False。

        若没有任何快照但存在未完成任务（待开始/进行中），也视为不一致，
        以便触发重排生成计划（清空历史快照后有任务却无计划的场景）。
        """
        snap = self.current_snapshot
        if snap is None:
            # 无快照时：若存在未完成任务则需要重排生成计划
            has_active_tasks = any(
                t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
                for t in self._tasks
            )
            return not has_active_tasks
        task_ids = {t.id for t in self._tasks}
        commitment_ids = {c.id for c in self._commitments}
        for ts in snap.time_slots or []:
            src = ts.source.value if hasattr(ts.source, "value") else ts.source
            if src == "task" and ts.task_id and ts.task_id not in task_ids:
                return False
            if src == "commitment" and ts.commitment_id and ts.commitment_id not in commitment_ids:
                return False
        for s in snap.sacrifice_list or []:
            if s.task_id and s.task_id not in task_ids:
                return False
        return True

    def ensure_plan_consistent(self, now: Optional[datetime] = None) -> bool:
        """数据自愈：若最新快照引用了已删除的任务/承诺，按当前数据重排一次。

        Returns:
            True 表示执行了一次校正性重排并生成了新快照；正常时返回 False。
        """
        try:
            if self.plan_is_consistent():
                return False
            self.force_replan(
                reason="计划校正：清理已删除的任务/承诺安排",
                now=now,
            )
            return True
        except Exception:
            return False

    def load_state(self, profile_id: Optional[str] = None) -> None:
        """从 SQLite 加载目标、任务、承诺和快照历史。"""
        if self.db is None:
            return

        pid = profile_id or self.profile.id or "default"
        self._goals = self.db.list_goals(pid)
        self._tasks = self.db.list_tasks(profile_id=pid)
        self._commitments = self.db.list_commitments(pid)
        self._snapshots = self.db.list_snapshots(pid, limit=500)
        # list_snapshots 按时间倒序返回，内存中保持正序快照链
        self._snapshots.sort(key=lambda s: s.created_at)

        # 恢复上次未确认的动作（跨进程/刷新保留对话状态）
        pending_data = self.db.load_pending_action(pid)
        self._pending_action = (
            PendingAction.from_dict(pending_data) if pending_data else None
        )

    def persist_state(self) -> None:
        """将当前内存状态写回 SQLite。"""
        if self.db is None:
            return

        self.db.save_profile(self.profile)
        for goal in self._goals:
            self.db.save_goal(goal)
        for task in self._tasks:
            self.db.save_task(task)
        for commitment in self._commitments:
            self.db.save_commitment(commitment)
        for snapshot in self._snapshots:
            self.db.save_snapshot(snapshot)

    def add_goal(self, goal: Goal) -> None:
        """添加目标。"""
        if not goal.id:
            goal.id = f"goal_{uuid.uuid4().hex[:8]}"
        self._goals.append(goal)

    def add_task(self, task: Task) -> None:
        """添加任务。"""
        if not task.id:
            task.id = f"task_{uuid.uuid4().hex[:8]}"
        self._tasks.append(task)

    def add_commitment(self, commitment: Commitment) -> None:
        """添加固定承诺。"""
        if not commitment.id:
            commitment.id = f"commit_{uuid.uuid4().hex[:8]}"
        self._commitments.append(commitment)

    def update_commitment(
        self,
        commitment_id: str,
        *,
        title: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        recurrence: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """修改一个固定承诺/课程。"""
        for commitment in self._commitments:
            if commitment.id != commitment_id:
                continue
            if title is not None:
                commitment.title = title
            if start_time is not None:
                commitment.start_time = start_time
            if end_time is not None:
                commitment.end_time = end_time
            if recurrence is not None:
                commitment.recurrence = recurrence
            if description is not None:
                commitment.description = description
            return True
        return False

    def remove_commitment(self, commitment_id: str) -> bool:
        """删除一个固定承诺/课程。"""
        before = len(self._commitments)
        self._commitments = [
            c for c in self._commitments if c.id != commitment_id
        ]
        removed = len(self._commitments) < before
        if removed and self.db is not None:
            self.db.delete_commitment(commitment_id)
        return removed

    def remove_task(self, task_id: str) -> bool:
        """删除单个任务（同时从 SQLite 移除）。"""
        before = len(self._tasks)
        self._tasks = [t for t in self._tasks if t.id != task_id]
        removed = len(self._tasks) < before
        if removed and self.db is not None:
            self.db.delete_task(task_id)
        return removed

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        for g in self._goals:
            if g.id == goal_id:
                return g
        return None

    # ==========================================
    # 取消计划：按对话意图删除目标/任务/固定安排
    # ==========================================

    @staticmethod
    def _title_matches(title: str, keyword: str) -> bool:
        """标题是否命中关键词：只要有一方包含另一方即算命中。

        用户常把任务全名说出来（"删除整理知识点任务"），提取出的关键词
        可能与标题互有出入，因此做双向包含匹配。
        """
        kw = (keyword or "").strip().lower()
        name = (title or "").strip().lower()
        if not kw or not name:
            return False
        if kw in name or name in kw:
            return True
        # 用户原话提取出的关键词可能与标题互有出入（口语停用词被剥离），
        # 用相似度兜底，避免"说了全名却匹配不上"
        return difflib.SequenceMatcher(None, kw, name).ratio() >= 0.6

    @classmethod
    def _select_by_keyword(cls, items: List[Any], keyword: str) -> List[Any]:
        """按关键词筛选；没有关键词时，只在候选唯一的情况下才选中（避免误删）。"""
        if keyword:
            return [i for i in items if cls._title_matches(getattr(i, "title", ""), keyword)]
        return list(items) if len(items) == 1 else []

    def _collect_cancel_targets(
        self, cancel_data: Dict[str, Any]
    ) -> tuple:
        """找出本次取消会涉及的目标 / 任务 / 固定安排（不修改状态）。"""
        target_type = (cancel_data or {}).get("target_type") or "goal"
        keyword = ((cancel_data or {}).get("keyword") or "").strip()
        date_str = (cancel_data or {}).get("date")

        goals: List[Goal] = []
        tasks: List[Task] = []
        commitments: List[Commitment] = []

        if target_type == "all":
            goals = list(self._goals)
            tasks = list(self._tasks)
        elif target_type == "date":
            if date_str:
                tasks = [
                    t for t in self._tasks
                    if (t.deadline or "").startswith(str(date_str))
                    and t.status != TaskStatus.COMPLETED
                ]
        elif target_type == "commitment":
            commitments = self._select_by_keyword(self._commitments, keyword)
        elif target_type == "task":
            tasks = self._select_by_keyword(self._tasks, keyword)
        else:
            goals = self._select_by_keyword(self._goals, keyword)
            if not goals and keyword:
                # 目标没匹配上，但任务匹配上了 → 退一步只删这些任务
                tasks = [
                    t for t in self._tasks
                    if self._title_matches(t.title, keyword)
                ]

        return goals, tasks, commitments

    def _has_cancel_targets(self, cancel_data: Dict[str, Any]) -> bool:
        """当前计划里是否存在该取消请求能匹配到的内容。"""
        goals, tasks, commitments = self._collect_cancel_targets(cancel_data)
        return bool(goals or tasks or commitments)

    def _preview_cancel(self, cancel_data: Dict[str, Any]) -> str:
        """生成"将要取消什么"的描述（供确认文案使用，不修改状态）。"""
        goals, tasks, commitments = self._collect_cancel_targets(cancel_data)
        if not (goals or tasks or commitments):
            return self._cancel_not_found_hint(cancel_data)

        parts = []
        if goals:
            for goal in goals:
                count = sum(
                    1 for t in self._tasks if getattr(t, "goal_id", None) == goal.id
                )
                suffix = f"（含 {count} 个任务）" if count else ""
                parts.append(f"取消目标「{goal.title}」{suffix}")
        if tasks:
            preview = "、".join(t.title for t in tasks[:3])
            more = f" 等 {len(tasks)} 个" if len(tasks) > 3 else f" 共 {len(tasks)} 个"
            parts.append(f"删除任务：{preview}{more}")
        if commitments:
            parts.append(
                "删除固定安排：" + "、".join(c.title for c in commitments)
            )
        return "，".join(parts)

    def _cancel_not_found_hint(self, cancel_data: Dict[str, Any]) -> str:
        """没匹配到可取消内容时的提示（列出当前计划供用户选择）。"""
        keyword = ((cancel_data or {}).get("keyword") or "").strip()
        target_type = (cancel_data or {}).get("target_type") or "goal"
        label = {
            "task": "任务", "commitment": "固定安排", "date": "安排", "all": "计划",
        }.get(target_type, "目标/计划")

        if keyword:
            hint = f"没找到名称包含「{keyword}」的{label}。"
        else:
            hint = f"你还没说清要取消哪个{label}。"

        candidates = [g.title for g in self._goals] or [
            t.title for t in self._tasks
        ]
        if candidates:
            hint += "当前计划有：" + "、".join(candidates[:5]) + "。"
        else:
            hint += "当前还没有任何计划。"
        return hint

    def _apply_cancel(self, cancel_data: Dict[str, Any]) -> None:
        """执行取消：删除匹配到的目标 / 任务 / 固定安排，并记录摘要。"""
        goals, tasks, commitments = self._collect_cancel_targets(cancel_data)
        self._cancel_removed_count = 0
        parts = []

        for goal in goals:
            if self.remove_goal(goal.id):
                self._cancel_removed_count += 1
        if goals:
            parts.append("目标：" + "、".join(g.title for g in goals))

        # 目标被删除后，其任务已从列表移除，只删仍然存在的任务
        tasks = [t for t in tasks if any(x.id == t.id for x in self._tasks)]
        for task in tasks:
            if self.remove_task(task.id):
                self._cancel_removed_count += 1
        if tasks:
            parts.append(f"任务 {len(tasks)} 个：" + "、".join(t.title for t in tasks[:3]))

        for commitment in commitments:
            if self.remove_commitment(commitment.id):
                self._cancel_removed_count += 1
        if commitments:
            parts.append("固定安排：" + "、".join(c.title for c in commitments))

        if parts:
            self._cancel_summary = "已取消：" + "；".join(parts)
        else:
            self._cancel_summary = self._cancel_not_found_hint(cancel_data)

    def remove_goal(self, goal_id: str) -> bool:
        """删除目标（连同其下属任务一并移除）。"""
        before = len(self._goals)
        # 移除内存中的目标
        self._goals = [g for g in self._goals if g.id != goal_id]
        # 移除归属于该目标的任务
        self._tasks = [t for t in self._tasks if getattr(t, "goal_id", None) != goal_id]
        removed = len(self._goals) < before
        if removed and self.db is not None:
            self.db.delete_goal(goal_id)
        return removed

    def complete_task(self, task_id: str) -> bool:
        """将任务标记为已完成（status=completed, progress=1.0）。"""
        from ..models.goal import TaskStatus
        task = self.get_task(task_id)
        if task is None:
            return False
        task.status = TaskStatus.COMPLETED
        task.progress = 1.0
        if self.db is not None:
            self.db.save_task(task)
        return True

    def uncomplete_task(self, task_id: str) -> bool:
        """取消完成状态（回退为 in_progress, progress=0.5）。"""
        from ..models.goal import TaskStatus
        task = self.get_task(task_id)
        if task is None:
            return False
        task.status = TaskStatus.IN_PROGRESS
        task.progress = 0.5
        if self.db is not None:
            self.db.save_task(task)
        return True

    def start_task(self, task_id: str) -> bool:
        """把任务标记为「已开始」：仅切换状态为 in_progress。

        与 report_absolute_progress 的区别：那个必须给出具体百分比并会写
        执行记录；这里只表示“开始动手了”，进度保持 0，等用户真正汇报
        进度时再由对话/接口更新，避免凭空捏造一个假的百分比。
        """
        from ..models.goal import TaskStatus
        task = self.get_task(task_id)
        if task is None:
            return False
        if task.status == TaskStatus.COMPLETED:
            return False
        task.status = TaskStatus.IN_PROGRESS
        if self.db is not None:
            self.db.save_task(task)
        return True

    def reset_task_to_pending(self, task_id: str) -> bool:
        """把进行中的任务退回待开始（status=pending, progress=0）。"""
        from ..models.goal import TaskStatus
        task = self.get_task(task_id)
        if task is None:
            return False
        task.status = TaskStatus.PENDING
        task.progress = 0.0
        if self.db is not None:
            self.db.save_task(task)
        return True

    def get_task(self, task_id: str) -> Task | None:
        for t in self._tasks:
            if t.id == task_id:
                return t
        return None

    def get_tasks_by_goal(self, goal_id: str) -> List[Task]:
        return [t for t in self._tasks if t.goal_id == goal_id]

    def create_goal_structured(
        self,
        title: str,
        description: str = "",
        weight: float = 0.5,
        deadline: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> AgentResult:
        """跳过自然语言解析，直接用结构化字段创建目标并排计划。"""
        goal_data = {
            "title": title,
            "description": description,
            "weight": weight,
            "deadline": deadline,
        }
        self._create_goal_and_tasks(goal_data)
        return self.force_replan(
            reason=f"新增目标：{title}",
            now=now,
        )

    def force_replan(
        self,
        reason: str = "计划更新",
        now: Optional[datetime] = None,
    ) -> AgentResult:
        """不经过输入解析，直接基于当前状态重新排计划。"""
        if now is None:
            now = datetime.now()

        old_snapshot = self.current_snapshot
        start_date = now.date()
        end_date = start_date + timedelta(days=7)
        schedule_result = self.scheduler.schedule(
            tasks=self._tasks,
            goals=self._goals,
            commitments=self._commitments,
            start_date=start_date,
            end_date=end_date,
            now=now,
        )
        trigger = "initial" if old_snapshot is None else "manual"
        snapshot = self._create_snapshot(
            schedule_result, now, trigger_reason=trigger
        )
        change_log = self.explainer.explain(
            old_snapshot=old_snapshot,
            new_result=schedule_result,
            reason=reason,
        )
        snapshot.change_log = change_log
        self._snapshots.append(snapshot)
        self.persist_state()

        return AgentResult(
            reply=self._build_reply(change_log, schedule_result, ParseResult()),
            snapshot=snapshot,
            changed=True,
            change_log=change_log,
            intent=ParseIntent.NEW_EVENT,
            schedule_result=schedule_result,
        )

    def report_absolute_progress(
        self,
        task_id: str = "",
        progress: float = 0.0,
        actual_minutes: int = 0,
        note: str = "",
    ) -> AgentResult:
        """按“当前总进度”语义更新任务，供 MCP/UI 直接调用。"""
        if not 0.0 <= float(progress) <= 1.0:
            raise ValueError("progress 必须是 0-1 之间的当前总进度")

        if task_id:
            task = self.get_task(task_id)
        else:
            task = next(
                (
                    t for t in self._tasks
                    if t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
                ),
                None,
            )
        if task is None:
            return AgentResult(
                reply="没有找到可更新的任务，请先创建目标。",
                snapshot=self.current_snapshot,
                changed=False,
                intent=ParseIntent.PROGRESS_REPORT,
            )

        delta = progress - task.progress
        task.progress = round(min(1.0, max(0.0, progress)), 4)
        if task.progress >= 1.0:
            task.status = TaskStatus.COMPLETED
        elif task.progress > 0:
            task.status = TaskStatus.IN_PROGRESS
        task.actual_minutes = (task.actual_minutes or 0) + int(actual_minutes)

        record = ExecutionRecord(
            task_id=task.id,
            progress_delta=delta,
            actual_minutes=int(actual_minutes),
            note=note,
        )
        self._execution_records.append(record)
        if self.db is not None:
            self.db.add_execution_record(record)

        need_replan = abs(delta) >= 0.2 or sum(
            r.progress_delta for r in self._execution_records
        ) >= 0.3
        if need_replan:
            result = self.force_replan(
                reason=f"进度更新：{task.title} 当前 {task.progress * 100:.0f}%"
            )
            result.intent = ParseIntent.PROGRESS_REPORT
            return result

        self.persist_state()
        return AgentResult(
            reply=(
                f"已记录：{task.title} 当前进度 "
                f"{task.progress * 100:.0f}%，变化幅度未触发重排。"
            ),
            snapshot=self.current_snapshot,
            changed=False,
            intent=ParseIntent.PROGRESS_REPORT,
        )

    # ==========================================
    # 主入口：五步执行法
    # ==========================================

    def run(
        self,
        user_message: str,
        now: Optional[datetime] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResult:
        """执行一轮 Agent 循环。

        核心流程：
        1. 若存在待确认动作（_pending_action），肯定词 → 执行，否定词 → 取消，其他 → 作为新输入
        2. 正常解析 → 需要确认的 NEW_GOAL/NEW_EVENT 先反问确认并持久化（不重复解析）
        3. 其他意图直接执行（状态更新 + 规划/回复）

        history: 最近对话（[{role, content}]），注入解析与闲聊上下文，形成"记忆"。
        """
        if now is None:
            now = datetime.now()

        parse_context: Dict[str, Any] = {
            "tasks": [{"title": t.title, "id": t.id} for t in self._tasks],
            "history": [dict(h) for h in (history or [])][-8:],
            "profile_name": self.profile.name,
        }

        # ===== 待确认分支：肯定执行 / 否定取消 / 其他则覆盖 =====
        if self._pending_action is not None:
            if self._is_negative(user_message):
                return self._resolve_pending(confirmed=False, now=now)
            if self._is_affirmative(user_message):
                return self._resolve_pending(confirmed=True, now=now)
            # 非确认/取消内容 → 丢弃旧待确认，当作新输入处理
            self._set_pending(None)

        # ===== 普通输入 =====
        parse_result = self.parser.parse(
            user_message, context=parse_context, now=now
        )

        # 需要用户确认的意图（NEW_GOAL / NEW_EVENT / CANCEL_PLAN）：先反问确认，并保存解析结果
        # 取消属于破坏性操作，无论是否接入 LLM 都必须先确认；
        # 但没匹配到任何可取消内容时直接给提示，不进确认
        needs_confirm = self._needs_confirmation(parse_result)
        if (
            needs_confirm
            and parse_result.intent == ParseIntent.CANCEL_PLAN
            and not self._has_cancel_targets(parse_result.cancel_data or {})
        ):
            needs_confirm = False
        if needs_confirm and (
            self._confirm_enabled()
            or parse_result.intent == ParseIntent.CANCEL_PLAN
        ):
            action = self._pending_action_from_parse(
                parse_result, user_message, now
            )
            self._set_pending(action)
            reply = self._build_confirm_reply(parse_result, user_message)
            return AgentResult(
                reply=reply,
                snapshot=self.current_snapshot,
                changed=False,
                intent=parse_result.intent,
            )

        # 其他意图直接执行
        self._set_pending(None)
        self._apply_parse_result(parse_result, now)
        need_replan = self._should_replan(parse_result)
        if need_replan:
            result = self._do_replan(parse_result, now, user_message)
        elif parse_result.intent == ParseIntent.UNKNOWN:
            result = self._reply_general(user_message, now, history=history)
        else:
            result = self._reply_no_change(parse_result, now)

        if parse_result.intent not in (
            ParseIntent.STATUS_QUERY,
            ParseIntent.UNKNOWN,
        ):
            self.persist_state()
        return result

    # ==========================================
    # 待确认动作管理（持久化 + 确认/取消）
    # ==========================================

    @property
    def pending_action(self) -> Optional[PendingAction]:
        """当前待用户确认的动作（无则为 None）。"""
        return self._pending_action

    def _confirm_enabled(self) -> bool:
        """是否启用"先确认再执行"：有 LLM（真实对话场景）时启用，
        无 LLM 的确定性/演示模式直接执行，保持向后兼容。"""
        return self.llm is not None or self.parser.llm is not None

    def _needs_confirmation(self, parse_result: ParseResult) -> bool:
        """该解析结果是否需要先经过用户确认。"""
        # 取消计划：一律先确认，避免误删
        if parse_result.intent == ParseIntent.CANCEL_PLAN:
            return True
        if parse_result.intent == ParseIntent.NEW_GOAL and parse_result.goal_data:
            return True
        if parse_result.intent == ParseIntent.NEW_EVENT and parse_result.life_events:
            # 纯生病事件直接执行（生病通常无需二次确认）
            if all(
                e.event_type == EventType.ILLNESS
                for e in parse_result.life_events
            ):
                return False
            return True
        return False

    @staticmethod
    def _life_event_to_dict(event: LifeEvent) -> Dict[str, Any]:
        """把 LifeEvent 转成可 JSON 序列化的 dict。"""
        return {
            "title": event.title,
            "event_type": (
                event.event_type.value
                if hasattr(event.event_type, "value")
                else event.event_type
            ),
            "event_time": event.event_time,
            "end_time": event.end_time,
            "change_content": event.change_content,
            "confidence": event.confidence,
        }

    def _pending_action_from_parse(
        self,
        parse_result: ParseResult,
        user_message: str,
        now: datetime,
    ) -> PendingAction:
        """把刚解析出的意图保存为待确认动作（含结构化结果，确认时不再调 LLM）。"""
        life_events = None
        if parse_result.life_events:
            life_events = [
                self._life_event_to_dict(e) for e in parse_result.life_events
            ]
        return PendingAction(
            message=user_message,
            intent=(
                parse_result.intent.value
                if parse_result.intent
                else None
            ),
            goal_data=(
                dict(parse_result.goal_data)
                if parse_result.goal_data
                else None
            ),
            life_events=life_events,
            cancel_data=(
                dict(parse_result.cancel_data)
                if parse_result.cancel_data
                else None
            ),
            created_at=now.isoformat(),
        )

    def _set_pending(self, action: Optional[PendingAction]) -> None:
        """设置/清除待确认动作，并同步持久化。"""
        self._pending_action = action
        if self.db is None:
            return
        pid = self.profile.id or "default"
        if action is None:
            self.db.clear_pending_action(pid)
        else:
            self.db.save_pending_action(pid, action.to_dict())

    def _parse_result_from_action(
        self, action: PendingAction
    ) -> Optional[ParseResult]:
        """把待确认动作中的结构化结果还原为 ParseResult（不重新调用 LLM）。"""
        try:
            intent = ParseIntent(action.intent) if action.intent else None
        except ValueError:
            intent = None

        if intent == ParseIntent.NEW_GOAL and action.goal_data:
            return ParseResult(
                intent=intent,
                confidence=0.9,
                goal_data=dict(action.goal_data),
            )

        if intent == ParseIntent.CANCEL_PLAN and action.cancel_data:
            return ParseResult(
                intent=intent,
                confidence=0.9,
                cancel_data=dict(action.cancel_data),
            )

        if intent == ParseIntent.NEW_EVENT and action.life_events:
            events = []
            for raw in action.life_events:
                try:
                    events.append(LifeEvent(**raw))
                except Exception:
                    continue
            if events:
                return ParseResult(
                    intent=intent,
                    confidence=0.9,
                    life_events=events,
                )
        return None

    def confirm_pending(
        self,
        confirmed: bool = True,
        now: Optional[datetime] = None,
    ) -> AgentResult:
        """显式确认/取消待确认动作（供 UI 按钮、MCP 等直接调用）。"""
        if now is None:
            now = datetime.now()
        if self._pending_action is None:
            return AgentResult(
                reply="当前没有待确认的安排。",
                snapshot=self.current_snapshot,
                changed=False,
                intent=ParseIntent.UNKNOWN,
            )
        return self._resolve_pending(confirmed=confirmed, now=now)

    def _resolve_pending(
        self,
        confirmed: bool,
        now: datetime,
    ) -> AgentResult:
        """执行或取消一条待确认动作。"""
        action = self._pending_action
        self._set_pending(None)
        if action is None:
            return AgentResult(
                reply="没有需要确认的安排。",
                snapshot=self.current_snapshot,
                changed=False,
                intent=ParseIntent.UNKNOWN,
            )

        # 取消
        if not confirmed:
            return AgentResult(
                reply="好的，不安排。需要安排时随时告诉我。",
                snapshot=self.current_snapshot,
                changed=False,
                intent=(
                    ParseIntent(action.intent)
                    if action.intent
                    else ParseIntent.UNKNOWN
                ),
            )

        # 执行：优先复用确认前保存的解析结果（稳定、省一次 LLM 调用）
        parse_result = self._parse_result_from_action(action)
        if parse_result is None:
            # 只有原始消息（如闲聊反问"要不要排计划"）→ 带 confirming 提示重新解析
            parse_context: Dict[str, Any] = {
                "tasks": [
                    {"title": t.title, "id": t.id} for t in self._tasks
                ],
                "profile_name": self.profile.name,
                "confirming": True,
            }
            parse_result = self.parser.parse(
                action.message, context=parse_context, now=now
            )

        # 用户确认后的事件才落地
        for event in parse_result.life_events:
            event.user_confirmed = True

        self._apply_parse_result(parse_result, now)
        need_replan = self._should_replan(parse_result)
        if need_replan:
            result = self._do_replan(parse_result, now, action.message)
        else:
            result = self._reply_no_change(parse_result, now)
        self.persist_state()
        return result

    # ==========================================
    # Step 2: 应用解析结果到状态
    # ==========================================

    def _apply_parse_result(
        self,
        result: ParseResult,
        now: Optional[datetime] = None,
    ) -> None:
        """将解析结果应用到内部状态。"""
        if result.intent == ParseIntent.PROGRESS_REPORT:
            self._apply_progress_updates(result.execution_records)

        elif result.intent == ParseIntent.NEW_GOAL and result.goal_data:
            self._create_goal_and_tasks(result.goal_data)

        elif result.intent == ParseIntent.NEW_EVENT:
            for event in result.life_events:
                self._apply_event(event, now)

        elif result.intent == ParseIntent.CANCEL_PLAN:
            self._apply_cancel(result.cancel_data or {})

    def _apply_progress_updates(self, records: List[ExecutionRecord]) -> None:
        """应用进度更新。"""
        for record in records:
            # 尝试匹配任务
            task = self._match_task(record)
            if task is None:
                continue

            # 默认把用户汇报的百分比当作“当前总进度”；
            # 明确说了“又/再/额外”时才作为增量累加
            previous_progress = task.progress
            reported_progress = record.progress_delta
            if getattr(record, "_progress_is_delta", False):
                target_progress = previous_progress + reported_progress
            else:
                target_progress = reported_progress
            target_progress = min(1.0, max(0.0, target_progress))
            record.progress_delta = round(target_progress - previous_progress, 4)
            task.progress = round(target_progress, 4)
            if task.progress >= 1.0:
                task.status = TaskStatus.COMPLETED
            elif task.progress > 0:
                task.status = TaskStatus.IN_PROGRESS

            # 更新实际花费时间
            task.actual_minutes = (task.actual_minutes or 0) + record.actual_minutes
            record.task_id = task.id
            self._execution_records.append(record)
            if self.db is not None:
                self.db.add_execution_record(record)

    def _match_task(self, record: ExecutionRecord) -> Optional[Task]:
        """根据提示匹配任务。"""
        hint = getattr(record, "_task_title_hint", "")
        task_id = record.task_id

        if task_id:
            return self.get_task(task_id)

        if hint and hint != "未知任务":
            # 模糊匹配
            hint_lower = hint.lower()
            best_match = None
            best_score = 0
            for t in self._tasks:
                if t.status == TaskStatus.COMPLETED:
                    continue
                score = 0
                title_lower = t.title.lower()
                if hint_lower in title_lower:
                    score = 100
                else:
                    # 简单字符重叠
                    common = len(set(hint_lower) & set(title_lower))
                    score = common
                if score > best_score:
                    best_score = score
                    best_match = t
            if best_score >= 2:  # 至少有 2 个字符重叠才算匹配
                return best_match

        # 匹配第一个进行中/待开始的任务
        for t in self._tasks:
            if t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                return t

        return None

    def _create_goal_and_tasks(self, goal_data: Dict) -> None:
        """创建新目标并分解任务。"""
        goal = Goal(
            id=f"goal_{uuid.uuid4().hex[:8]}",
            profile_id=self.profile.id or "default",
            title=goal_data.get("title", "新目标"),
            description=goal_data.get("description", ""),
            weight=goal_data.get("weight") or 0.5,
            deadline=goal_data.get("deadline"),
        )
        self._goals.append(goal)

        # 分解目标为任务
        deadline_str = goal_data.get("deadline")
        deadline = None
        if deadline_str:
            try:
                deadline = datetime.fromisoformat(deadline_str)
            except (ValueError, TypeError):
                pass

        tasks = self.decomposer.decompose(goal, deadline=deadline)

        # 分配真实 ID 并解析依赖
        id_map: Dict[str, str] = {}
        for i, task in enumerate(tasks):
            new_id = f"task_{uuid.uuid4().hex[:8]}"
            id_map[f"__idx_{i}"] = new_id
            task.id = new_id

        # 替换依赖占位符
        for task in tasks:
            new_deps = []
            for dep in task.dependencies:
                if dep in id_map:
                    new_deps.append(id_map[dep])
                else:
                    new_deps.append(dep)
            task.dependencies = new_deps
            self._tasks.append(task)

    def _apply_life_event(
        self,
        event: LifeEvent,
        now: Optional[datetime] = None,
    ) -> None:
        """应用生活事件。

        目前支持的事件类型对调度的影响：
        - schedule_change: 考试改期 → 更新对应任务的 deadline
        - illness: 生病 → 降低可用精力，可能需要重新安排
        - activity: 新增活动 → 作为 commitment 加入
        """
        if now is None:
            now = datetime.now()
        event_type = event.event_type

        if event_type == "schedule_change" or event_type == "exam":
            # 简单处理：标记第一个硬截止任务需要更新 deadline
            if event.event_time:
                for task in self._tasks:
                    if task.deadline_type == DeadlineType.HARD:
                        task.deadline = event.event_time
                        break

        elif event_type == "illness":
            # 生病：当天剩余时间不可用，交给确定性调度器重新安排
            self._block_event_time(
                event=event,
                now=now,
                to_day_end=True,
            )
        elif event_type == "activity":
            # 临时活动：只占用 event_time 到 end_time 这段时间
            self._block_event_time(
                event=event,
                now=now,
                to_day_end=False,
            )

    def _block_event_time(
        self,
        event: LifeEvent,
        now: datetime,
        to_day_end: bool,
    ) -> None:
        """把生病/活动事件转成一次性固定承诺。"""
        from ..models.profile import Commitment, CommitmentType

        try:
            start = (
                datetime.fromisoformat(event.event_time)
                if event.event_time else now
            )
        except (ValueError, TypeError):
            start = now

        if to_day_end:
            end = datetime.combine(start.date(), self.profile.sleep_time)
            if end <= start:
                end = start + timedelta(hours=4)
        elif event.end_time:
            try:
                end = datetime.fromisoformat(event.end_time)
            except (ValueError, TypeError):
                end = start + timedelta(hours=2)
        else:
            end = start + timedelta(hours=2)

        if end <= start:
            end = start + timedelta(hours=1)

        title = event.title or ("生病休息" if event.event_type == "illness" else "临时活动")
        commitment = Commitment(
            profile_id=self.profile.id or "default",
            title=f"{title}（不可用）",
            type=CommitmentType.OTHER,
            start_time=start.isoformat(),
            end_time=end.isoformat(),
            recurrence="none",
            description=event.change_content or event.title or "",
        )
        self.add_commitment(commitment)

    # ==========================================
    # Step 3: 是否需要重规划
    # ==========================================

    def _should_replan(self, result: ParseResult) -> bool:
        """判断是否需要重规划。"""
        # 取消计划 → 只有真的删掉了东西才需要重排（优先判断，否则会被"无任务"分支拦下）
        if result.intent == ParseIntent.CANCEL_PLAN:
            return self._cancel_removed_count > 0

        # 没有任何快照但有任务 → 需要首次规划
        if self._tasks and not self._snapshots:
            return True

        # 无任务 → 需要（首次规划，来自新目标）
        if not self._tasks:
            return result.intent in (ParseIntent.NEW_GOAL, ParseIntent.NEW_EVENT)

        # 新目标 → 需要
        if result.intent == ParseIntent.NEW_GOAL:
            return True

        # 新事件 → 需要
        if result.intent == ParseIntent.NEW_EVENT:
            return True

        # 取消计划 → 只有真的删掉了东西才需要重排
        if result.intent == ParseIntent.CANCEL_PLAN:
            return self._cancel_removed_count > 0

        # 进度偏差 → 检查是否超过阈值
        if result.intent == ParseIntent.PROGRESS_REPORT:
            return self._is_progress_significant(result.execution_records)

        # 状态查询 → 不需要重规划
        if result.intent == ParseIntent.STATUS_QUERY:
            return False

        return False

    def _is_progress_significant(self, records: List[ExecutionRecord]) -> bool:
        """判断进度更新是否足够大，需要触发重规划。

        阈值：任一任务进度变化 >= 20%，或总进度变化 >= 30%。
        """
        total_delta = sum(r.progress_delta for r in records)
        max_delta = max((r.progress_delta for r in records), default=0)
        return total_delta >= 0.3 or max_delta >= 0.2

    # ==========================================
    # Step 4 & 5: 重规划
    # ==========================================

    def _do_replan(
        self,
        parse_result: ParseResult,
        now: datetime,
        user_message: str,
    ) -> AgentResult:
        """执行重规划。"""
        old_snapshot = self.current_snapshot

        # 计算时间范围（默认 7 天）
        start_date = now.date()
        end_date = start_date + timedelta(days=7)

        # 调用调度器
        schedule_result = self.scheduler.schedule(
            tasks=self._tasks,
            goals=self._goals,
            commitments=self._commitments,
            start_date=start_date,
            end_date=end_date,
            now=now,
        )

        # 创建快照
        reason = self._build_reason(parse_result, user_message)
        trigger = "initial" if old_snapshot is None else parse_result.intent.value
        snapshot = self._create_snapshot(schedule_result, now, trigger_reason=trigger)
        self._snapshots.append(snapshot)
        new_tasks = [t for t in self._tasks if t.created_at and 
                     (now - datetime.fromisoformat(t.created_at)).total_seconds() < 60]
        change_log = self.explainer.explain(
            old_snapshot=old_snapshot,
            new_result=schedule_result,
            reason=reason,
            new_tasks=new_tasks,
        )
        snapshot.change_log = change_log

        # 构建回复
        reply = self._build_reply(change_log, schedule_result, parse_result)

        # 取消计划时直接说明"取消了什么"，比调度差异更直观
        if parse_result.intent == ParseIntent.CANCEL_PLAN and self._cancel_summary:
            reply = (
                f"{self._cancel_summary}。"
                f"\n📊 剩余计划：{len(schedule_result.time_slots)} 个时间段，"
                f"共 {schedule_result.total_scheduled_minutes} 分钟。"
            )

        return AgentResult(
            reply=reply,
            snapshot=snapshot,
            changed=True,
            change_log=change_log,
            intent=parse_result.intent,
            schedule_result=schedule_result,
        )

    def _create_snapshot(
        self,
        result: ScheduleResult,
        now: datetime,
        trigger_reason: str = "",
    ) -> PlanSnapshot:
        """创建计划快照。"""
        snapshot = PlanSnapshot(
            id=f"snap_{uuid.uuid4().hex[:8]}",
            profile_id=self.profile.id or "default",
            version=(self.current_snapshot.version + 1
                     if self.current_snapshot else 1),
            created_at=now.isoformat(),
            trigger_reason=trigger_reason,
            time_slots=result.time_slots,
            sacrifice_list=result.sacrifice_list,
            schedule_tracks=result.schedule_tracks,
            hard_deadline_count=sum(
                1 for t in self._tasks if t.deadline_type == DeadlineType.HARD
            ),
            risk_count=len(result.sacrifice_list),
        )
        return snapshot

    def _build_reason(self, parse_result: ParseResult, user_message: str) -> str:
        """构建变更原因。"""
        if parse_result.intent == ParseIntent.CANCEL_PLAN:
            return self._cancel_summary or "取消已有安排"
        if parse_result.intent == ParseIntent.NEW_GOAL:
            return f"新增目标：{parse_result.goal_data.get('title', '新目标') if parse_result.goal_data else '新目标'}"
        elif parse_result.intent == ParseIntent.NEW_EVENT:
            event_titles = [e.title for e in parse_result.life_events]
            return f"新事件：{', '.join(event_titles)}"
        elif parse_result.intent == ParseIntent.PROGRESS_REPORT:
            return f"进度更新：{user_message[:30]}"
        else:
            return user_message[:50]

    def _build_reply(
        self,
        change_log: ChangeLog,
        result: ScheduleResult,
        parse_result: ParseResult,
    ) -> str:
        """构建给用户的回复。"""
        parts = [change_log.detail or change_log.summary]

        # 如果有牺牲，提醒
        if result.sacrifice_list:
            parts.append(f"\n⚠️ 有 {len(result.sacrifice_list)} 个任务暂时搁置了。")

        # 统计信息
        parts.append(f"\n📊 当前计划：{len(result.time_slots)} 个时间段，"
                     f"共 {result.total_scheduled_minutes} 分钟。")

        return "\n".join(parts)

    # ==========================================
    # 无重规划时的回复
    # ==========================================

    def _reply_general(
        self,
        user_message: str,
        now: datetime,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResult:
        """普通闲聊/未命中计划意图时走模型对话，不再返回固定文案。

        会携带最近对话（history）一起发给模型，保持多轮语境连贯。
        """
        llm = self.llm or self.parser.llm or getattr(self.retriever, "llm", None)
        if llm is not None:
            from ..llm.chat_model import ChatMessage, ChatRole

            weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            today_desc = f"{now.strftime('%Y年%m月%d日 %H:%M')}（{weekday_names[now.weekday()]}）"

            # 注入最近对话，让闲聊能接上上下文（"可以/OK"之类才有意义）
            messages = [
                ChatMessage(
                    ChatRole.SYSTEM,
                    f"你是 LifeOS 的中文学习与时间管理助手。今天是 {today_desc}。"
                    "涉及学生目标、课程、任务、进度时先确认是否需要排计划；"
                    "其他问题正常友好回答。回答简洁自然，不要编造今天的日期。",
                )
            ]
            for h in (history or [])[-6:]:
                role = h.get("role", "")
                if role not in ("user", "bot", "assistant"):
                    continue
                content = (h.get("content") or "").strip()[:300]
                if not content:
                    continue
                messages.append(
                    ChatMessage(
                        ChatRole.USER if role == "user" else ChatRole.ASSISTANT,
                        content,
                    )
                )
            messages.append(ChatMessage(ChatRole.USER, user_message))

            try:
                response = llm.chat(messages, temperature=0.7, max_tokens=600)
                content = response.get("content", "").strip()
                if content:
                    # 模型反问"是否需要排计划"时，记下本条消息等待用户确认
                    if "？" in content and any(
                        kw in content for kw in ("计划", "安排", "目标", "确认")
                    ):
                        self._set_pending(PendingAction(message=user_message))
                    return AgentResult(
                        reply=content,
                        snapshot=self.current_snapshot,
                        changed=False,
                        intent=ParseIntent.UNKNOWN,
                    )
            except Exception:
                pass

        return AgentResult(
            reply=(
                "暂时没有理解你的意思，也没连接到可用的模型。"
                "你可以试试说“我要准备期末考试”或“看看我的计划”。"
            ),
            snapshot=self.current_snapshot,
            changed=False,
            intent=ParseIntent.UNKNOWN,
        )

    # 肯定词表：支持整句精确命中，也支持"前缀包含 + 短句"的宽松识别
    _AFFIRMATIVE_REPLIES = {
        "好", "好的", "好呀", "好啊", "好嘞", "好哒", "好滴", "嗯", "嗯嗯",
        "行", "行的", "可以", "可以的", "可以啊", "可以的", "好的呀", "行啊",
        "是的", "是", "对", "对的", "对呀", "要", "需要", "确定",
        "确认", "安排", "安排吧", "ok", "okay", "ok的", "yes", "yeah",
        "没问题", "当然", "当然可以", "麻烦你了", "帮我安排", "这样安排",
        "就这么办", "就这样", "同意", "开始", "开始吧", "快开始吧",
    }

    # 否定/取消词（先于肯定词判断）
    _NEGATIVE_WORDS = (
        "不用", "不需要", "不要", "不安排", "不行", "算了", "取消", "先不了",
        "以后再说", "再想想", "等等", "别", "不是", "不了", "拒绝",
        "改一下", "想改", "不用了",
    )
    _NEGATIVE_EXACT = {
        "不", "不排", "不要", "不用", "不行", "取消", "算了", "别", "先不了",
        "不需要", "no", "不安排",
    }

    @classmethod
    def _normalize_reply(cls, message: str) -> str:
        """去掉常见标点与空白后的小写文本。"""
        return (
            message.strip().lower().strip("。，！？!?~～,.、；;:： \t\n")
        )

    @classmethod
    def _is_negative(cls, message: str) -> bool:
        """判断是否为否定/取消回复。"""
        text = cls._normalize_reply(message)
        if not text:
            return False
        if text in cls._NEGATIVE_EXACT:
            return True
        # 短句包含否定词即视为取消（"不用了先""算了不排了"）
        return (
            len(text) <= 16
            and any(w in text for w in cls._NEGATIVE_WORDS)
        )

    @classmethod
    def _is_affirmative(cls, message: str) -> bool:
        """判断是否为肯定回复（用于确认上一轮的待确认意图）。

        相比精确词表，这里支持"可以啊 / OK 好的 / 行，帮我排吧 / 嗯好"等
        常见变体；同时排除带否定词的表达（如"不用了"）。
        """
        text = cls._normalize_reply(message)
        if not text or len(text) > 20:
            return False
        # 明确的否定词优先（"没问题"含"没"但属于肯定，特例处理）
        if cls._is_negative(text) and text not in ("没问题", "没问题吧"):
            return False
        # 1) 精确命中短词
        if text in cls._AFFIRMATIVE_REPLIES:
            return True
        # 2) 前缀包含式：如"可以，帮我排吧""好的没问题""OK 就这么办"
        if len(text) <= 14:
            for stem in (
                "好的", "好呀", "好啊", "好嘞", "可以", "没问题", "当然",
                "行", "嗯", "对", "ok", "okay", "yes", "要", "确认",
                "安排", "就这样", "就这么办", "同意", "开始", "麻烦你了",
                "帮我安排", "好",
            ):
                if text.startswith(stem):
                    return True
        return False

    def _build_confirm_reply(
        self, parse_result: ParseResult, user_message: str
    ) -> str:
        """生成待确认的回复，引导用户说「好的」再执行。"""
        if parse_result.intent == ParseIntent.NEW_GOAL:
            title = (
                (parse_result.goal_data or {}).get("title")
                or self.parser._clean_goal_title(user_message)
            )
            dl = (parse_result.goal_data or {}).get("deadline", "")
            dl_text = f"（截止 {dl}）" if dl else ""
            return (
                f"我理解你的目标是：{title}{dl_text}。"
                "要帮你安排到每日计划里吗？"
                "同意的话回复「好的 / 可以 / 没问题」，不需要就回复「不用了」。"
            )
        if parse_result.intent == ParseIntent.NEW_EVENT:
            if parse_result.life_events:
                ev = parse_result.life_events[0]
                title = getattr(ev, "title", "") or user_message[:30]
                etime = getattr(ev, "event_time", "")
                etime_text = f"（{etime}）" if etime else ""
            else:
                title = user_message[:30]
                etime_text = ""
            return (
                f"我记下一个事件：{title}{etime_text}。"
                "要把它加入日程并重新安排任务吗？"
                "同意的话回复「好的 / 可以 / 没问题」，不需要就回复「不用了」。"
            )
        if parse_result.intent == ParseIntent.CANCEL_PLAN:
            preview = self._preview_cancel(parse_result.cancel_data or {})
            return (
                f"{preview}。"
                "确认取消吗？取消后我会重新安排剩余时间。"
                "同意的话回复「好的 / 可以 / 没问题」，不改就回复「不用了」。"
            )
        return "好的，我记下了。"

    def _reply_no_change(self, parse_result: ParseResult, now: datetime) -> AgentResult:
        """不需要重规划时的回复。"""
        if parse_result.intent == ParseIntent.STATUS_QUERY:
            return self._reply_status(parse_result, now)

        # 取消了但没匹配到东西：给出提示而不是"记下了"
        if parse_result.intent == ParseIntent.CANCEL_PLAN:
            return AgentResult(
                reply=self._cancel_summary or "没有可取消的内容。",
                snapshot=self.current_snapshot,
                changed=False,
                intent=parse_result.intent,
            )

        return AgentResult(
            reply="好的，我记下了。目前不需要调整计划。",
            snapshot=self.current_snapshot,
            changed=False,
            intent=parse_result.intent,
        )

    def _reply_status(self, parse_result: ParseResult, now: datetime) -> AgentResult:
        """状态查询回复。"""
        snapshot = self.current_snapshot
        if snapshot is None:
            return AgentResult(
                reply="还没有制定计划呢，告诉我你的目标，我来帮你安排！",
                snapshot=None,
                changed=False,
                intent=parse_result.intent,
            )

        # 统计进度
        total_tasks = len(self._tasks)
        done_tasks = sum(1 for t in self._tasks if t.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for t in self._tasks if t.status == TaskStatus.IN_PROGRESS)

        lines = [
            f"📋 当前计划状态：",
            f"• 总任务数：{total_tasks} 个",
            f"• 已完成：{done_tasks} 个",
            f"• 进行中：{in_progress} 个",
            f"• 总安排时长：{snapshot.total_scheduled_minutes} 分钟",
        ]

        if snapshot.sacrifice_list:
            lines.append(f"• 搁置任务：{len(snapshot.sacrifice_list)} 个")

        return AgentResult(
            reply="\n".join(lines),
            snapshot=snapshot,
            changed=False,
            intent=parse_result.intent,
        )

    # ==========================================
    # 截图解析
    # ==========================================

    def process_screenshot(self, image_path: str, image_type: str = "auto") -> AgentResult:
        """处理截图输入，解析出候选条目（不自动应用，需用户确认）。

        返回 AgentResult，其中 parsed_screenshot 字段包含解析结果，
        reply 字段包含中文说明和确认提示。
        """
        if not self.screenshot_parser:
            return AgentResult(
                reply="抱歉，截图解析功能未启用，请先配置视觉模型。",
                intent=ParseIntent.UNKNOWN,
            )

        result = self.screenshot_parser.parse(image_path, image_type)

        # 构建说明
        items_desc = []
        for i, item in enumerate(result.items[:10], 1):
            time_info = ""
            if item.date:
                time_info = f"，{item.date}"
            if item.start_time:
                time_info += f" {item.start_time}"
                if item.end_time:
                    time_info += f"-{item.end_time}"

            type_label = {
                "exam": "考试",
                "course": "课程",
                "homework": "作业",
                "task": "任务",
                "event": "事件",
                "commitment": "承诺",
            }.get(item.item_type, "条目")

            items_desc.append(f"{i}. {type_label}：{item.title}{time_info}")

        if not result.items:
            reply = f"我尝试解析了这张截图，但没有识别出明确的条目。\n\n识别类型：{result.type}\n置信度：{int(result.confidence * 100)}%\n\n你可以描述一下这张图的内容，我来帮你手动添加。"
        else:
            more = f"\n（还有 {len(result.items) - 10} 条...）" if len(result.items) > 10 else ""
            reply = f"我从截图中识别出了 {len(result.items)} 个条目：\n\n" + "\n".join(items_desc) + more + f"\n\n置信度：{int(result.confidence * 100)}%\n\n请确认以上条目是否正确，你可以选择要导入哪些，或者告诉我需要修改的地方。"

        return AgentResult(
            reply=reply,
            snapshot=self.current_snapshot,
            changed=False,
            intent=ParseIntent.NEW_EVENT,
            parsed_screenshot=result,
        )

    def apply_parsed_items(self, items: list) -> AgentResult:
        """将确认后的解析条目应用到计划中。

        items: ParsedItem 列表
        """
        events = []
        goals_to_create = []
        commitments_to_create = []
        created_goal_count = 0

        today = date.today()

        for item in items:
            if item.item_type == "event":
                # 一次性事件先落到事件记录，用户可后续确认具体安排
                from ..models.event import LifeEvent, EventType

                start_dt = None
                end_dt = None
                if item.date:
                    date_part = item.date
                    if item.start_time:
                        start_dt = f"{date_part}T{item.start_time}:00"
                    if item.end_time:
                        end_dt = f"{date_part}T{item.end_time}:00"

                event = LifeEvent(
                    title=item.title,
                    event_type=EventType.OTHER,
                    event_time=start_dt or f"{item.date or today.isoformat()}T09:00:00",
                    end_time=end_dt,
                    change_content=item.description or "",
                    confidence=float(item.priority) / 10.0 if item.priority else 0.7,
                    raw_data={"impact_level": "high" if item.deadline_type == "hard" else "medium"},
                )
                events.append(event)

            elif item.item_type == "course":
                # 课程 → Commitment（每周重复）
                from ..models.profile import CommitmentType, RecurrenceType

                day_map = {0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday", 4: "friday", 5: "saturday", 6: "sunday"}
                day_of_week = item.day_of_week if item.day_of_week is not None else 0

                start_time_str = item.start_time or "08:00"
                end_time_str = item.end_time or "09:00"

                # 计算下一个出现的日期
                today_weekday = today.weekday()
                days_ahead = (day_of_week - today_weekday) % 7
                next_date = today + timedelta(days=days_ahead)

                commitment = {
                    "title": item.title,
                    "type": CommitmentType.CLASS,
                    "start_time": f"{next_date.isoformat()}T{start_time_str}:00",
                    "end_time": f"{next_date.isoformat()}T{end_time_str}:00",
                    "recurrence": RecurrenceType.WEEKLY,
                    "description": item.location or "",
                }
                commitments_to_create.append(commitment)

            elif item.item_type in ("exam", "homework", "task", "goal"):
                # 考试/作业/任务 → 生成为可排期的目标
                goals_to_create.append(item)

        # 应用事件
        for event in events:
            self._apply_event(event)

        # 将考试/作业/任务转成结构化目标
        for item in goals_to_create:
            deadline = None
            if item.date:
                date_part = item.date
                if item.end_time:
                    deadline = f"{date_part}T{item.end_time}:00"
                elif item.start_time:
                    deadline = f"{date_part}T{item.start_time}:00"
                else:
                    deadline = f"{date_part}T23:59:00"

            if item.item_type == "exam":
                title = f"备考：{item.title}"
            elif item.item_type == "homework":
                title = f"完成：{item.title}"
            else:
                title = item.title

            self._create_goal_and_tasks({
                "title": title,
                "description": item.description or "",
                "deadline": deadline,
                "weight": max(0.1, min(1.0, (item.priority or 5) / 10.0)),
            })
            created_goal_count += 1

        # 应用承诺
        for c in commitments_to_create:
            from ..models.profile import Commitment
            commitment = Commitment(
                profile_id=self.profile.id,
                title=c["title"],
                type=c["type"],
                start_time=c["start_time"],
                end_time=c["end_time"],
                recurrence=c["recurrence"],
                description=c.get("description"),
            )
            self.add_commitment(commitment)

        # 触发重规划
        changed = bool(events or commitments_to_create or created_goal_count)
        if changed:
            result = self.force_replan(
                reason=(
                    f"从截图导入了 {created_goal_count} 个目标、"
                    f"{len(commitments_to_create)} 个承诺和 {len(events)} 个事件"
                )
            )
            snapshot = result.snapshot
        else:
            result = None
            snapshot = self.current_snapshot

        # 生成解释
        if snapshot and changed:
            reply = f"已将截图中的内容导入计划：\n"
            if created_goal_count:
                reply += f"- 添加了 {created_goal_count} 个可排期目标\n"
            if commitments_to_create:
                reply += f"- 添加了 {len(commitments_to_create)} 个课程承诺（每周重复）\n"
            if events:
                reply += f"- 记录了 {len(events)} 个事件\n"
            reply += "\n计划已重新安排。"
        else:
            reply = "没有需要导入的条目。"

        return AgentResult(
            reply=reply,
            snapshot=snapshot,
            changed=changed,
            change_log=snapshot.change_log if snapshot else None,
            intent=ParseIntent.NEW_EVENT,
        )

    def _apply_event(
        self,
        event: LifeEvent,
        now: Optional[datetime] = None,
    ) -> None:
        """应用单个事件到内部状态。

        封装 _apply_life_event，保持接口一致。
        """
        self._apply_life_event(event, now)
        if self.db is not None:
            self.db.add_event(event)

    def _reschedule(self, change_log: ChangeLog) -> None:
        """触发重规划，更新计划快照。

        根据当前任务、目标、承诺重新调度，
        生成新快照并附加到快照链。
        """
        old_snapshot = self.current_snapshot
        now = datetime.now()
        start_date = now.date()
        end_date = start_date + timedelta(days=7)

        # 调用调度器
        schedule_result = self.scheduler.schedule(
            tasks=self._tasks,
            goals=self._goals,
            commitments=self._commitments,
            start_date=start_date,
            end_date=end_date,
            now=now,
        )

        # 创建快照
        snapshot = self._create_snapshot(
            schedule_result, now,
            trigger_reason=change_log.trigger_reason,
        )

        # 生成变更说明
        new_tasks = [t for t in self._tasks if t.created_at and
                     (now - datetime.fromisoformat(t.created_at)).total_seconds() < 60]
        snapshot_change_log = self.explainer.explain(
            old_snapshot=old_snapshot,
            new_result=schedule_result,
            reason=change_log.detail,
            new_tasks=new_tasks,
        )
        snapshot.change_log = snapshot_change_log
        self._snapshots.append(snapshot)
        self.persist_state()

    # ==========================================
    # 知识库对话（RAG）
    # ==========================================

    def chat_with_knowledge(
        self,
        user_message: str,
        use_rag: bool = True,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResult:
        """带知识库的对话。

        对于知识类问题，先检索再回答；对于计划类问题，走正常流程。
        """
        # 若正在等确认且这条是确认/取消回复，直接走主流程（不要让"可以"被知识库误判）
        if self._pending_action is not None and (
            self._is_affirmative(user_message)
            or self._is_negative(user_message)
        ):
            return self.run(user_message, history=history)

        # 判断是否需要 RAG
        need_rag = use_rag and self.retriever and self._is_knowledge_query(user_message)

        references = []
        if need_rag:
            context_str, references = self.retriever.build_context(user_message, top_k=5)
            if references:
                llm = getattr(self.parser, "llm", None)
                if llm is None:
                    llm = getattr(self.retriever, "llm", None)

                if llm is not None:
                    from ..llm.chat_model import ChatMessage, ChatRole

                    prompt = (
                        f"用户问题：{user_message}\n\n"
                        f"{context_str}\n\n"
                        "请基于以上资料直接回答用户问题。"
                        "资料没有覆盖的内容要明确说明，不要编造。"
                        "引用时用 [编号] 标注来源。"
                    )
                    messages = [
                        ChatMessage(
                            ChatRole.SYSTEM,
                            "你是 LifeOS 的知识助手，只依据参考资料回答，中文输出，简洁准确。",
                        ),
                        ChatMessage(ChatRole.USER, prompt),
                    ]
                    try:
                        response = llm.chat(messages, temperature=0.3, max_tokens=800)
                        reply = response.get("content", "").strip()
                        if not reply:
                            raise ValueError("模型返回空回复")
                        result = AgentResult(
                            reply=reply,
                            snapshot=self.current_snapshot,
                            changed=False,
                            intent=ParseIntent.UNKNOWN,
                            rag_references=references,
                        )
                        return result
                    except Exception:
                        # 模型调用失败时退化为“资料原文 + 提示”
                        pass

                # 没有可用的回答模型时给出资料摘要而不是误导性的计划回复
                summary = "\n".join(
                    f"[{ref.get('index')}] {ref.get('content', '')[:120]}"
                    for ref in references[:3]
                )
                return AgentResult(
                    reply=(
                        f"已从知识库找到相关片段：\n\n{summary}\n\n"
                        "当前未配置可用的回答模型，请先在 .env 中配置 LLM。"
                    ),
                    snapshot=self.current_snapshot,
                    changed=False,
                    intent=ParseIntent.UNKNOWN,
                    rag_references=references,
                )

        # 不需要 RAG，走正常流程
        return self.run(user_message, history=history)

    def _is_knowledge_query(self, message: str) -> bool:
        """判断是否为知识查询类问题。"""
        knowledge_keywords = [
            "是什么", "什么是", "为什么", "怎么", "如何", "解释",
            "介绍", "概念", "定义", "原理", "公式", "定理",
            "知识点", "考点", "重点", "复习",
            "吗？", "呢？", "？",  # 疑问句
        ]
        msg_lower = message.lower()
        for kw in knowledge_keywords:
            if kw in msg_lower:
                return True

        # 如果消息较短且包含疑问语气，也可能是知识查询
        if len(message) < 30 and ("?" in message or "？" in message):
            return True

        return False

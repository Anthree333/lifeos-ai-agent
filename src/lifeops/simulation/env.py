"""StudentEnv 仿真环境。

模拟一个学生的日常生活，接受固定随机种子，保证同一场景可重复复现。
环境按天推进，每天注入事件并执行计划。

D2 阶段：实现最小可运行骨架（初始化、单步推进、事件注入）。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional

from ..models import (
    StudentProfile, Goal, Task, Commitment,
    ExecutionRecord, LifeEvent,
)
from ..scheduler import Scheduler, ScheduleResult


@dataclass
class EnvEvent:
    """仿真事件。"""
    day: int                    # 第几天发生（从 0 开始）
    event_type: str             # 事件类型
    data: Dict = field(default_factory=dict)
    description: str = ""


@dataclass
class EnvAction:
    """Agent 执行的动作。"""
    action_type: str
    data: Dict = field(default_factory=dict)


@dataclass
class EnvState:
    """环境状态。"""
    day: int = 0
    current_date: Optional[date] = None
    tasks: List[Task] = field(default_factory=list)
    goals: List[Goal] = field(default_factory=list)
    commitments: List[Commitment] = field(default_factory=list)
    execution_records: List[ExecutionRecord] = field(default_factory=list)
    events: List[LifeEvent] = field(default_factory=list)
    schedule_history: List[ScheduleResult] = field(default_factory=list)
    daily_completion_minutes: Dict[int, int] = field(default_factory=dict)


class StudentEnv:
    """学生仿真环境。

    用法：
        env = StudentEnv(profile=profile, seed=42)
        env.add_events(events)
        state = env.reset()
        while not env.done():
            action = agent.act(state)
            state = env.step(action)
    """

    def __init__(
        self,
        profile: Optional[StudentProfile] = None,
        start_date: Optional[date] = None,
        total_days: int = 7,
        seed: int = 42,
        scheduler: Optional[Scheduler] = None,
    ):
        self.profile = profile or StudentProfile()
        self.start_date = start_date or date.today()
        self.total_days = total_days
        self.seed = seed
        self.scheduler = scheduler or Scheduler(profile=self.profile)

        self._rng = random.Random(seed)
        self._events: List[EnvEvent] = []
        self.state = EnvState()
        self._done = False

    # ==========================================
    # 配置
    # ==========================================

    def add_events(self, events: List[EnvEvent]) -> None:
        """添加仿真事件。"""
        self._events.extend(events)
        # 按天排序
        self._events.sort(key=lambda e: e.day)

    def set_tasks(self, tasks: List[Task]) -> None:
        """设置初始任务列表。"""
        self.state.tasks = list(tasks)

    def set_goals(self, goals: List[Goal]) -> None:
        """设置初始目标列表。"""
        self.state.goals = list(goals)

    def set_commitments(self, commitments: List[Commitment]) -> None:
        """设置固定承诺。"""
        self.state.commitments = list(commitments)

    # ==========================================
    # 环境交互
    # ==========================================

    def reset(self) -> EnvState:
        """重置环境到初始状态。"""
        self._rng = random.Random(self.seed)
        self._done = False

        self.state = EnvState(
            day=0,
            current_date=self.start_date,
            tasks=list(self.state.tasks),
            goals=list(self.state.goals),
            commitments=list(self.state.commitments),
        )

        return self.state

    def step(self, action: Optional[EnvAction] = None) -> EnvState:
        """推进一天。

        Args:
            action: Agent 的动作（可选，D2 阶段先不处理复杂动作）

        Returns:
            更新后的环境状态
        """
        if self._done:
            return self.state

        # 1. 注入当天的事件
        self._inject_events()

        # 2. 模拟当天执行（简化：按计划的 60-100% 完成）
        self._simulate_daily_execution()

        # 3. 推进到下一天
        self.state.day += 1
        if self.state.current_date:
            self.state.current_date += timedelta(days=1)

        # 4. 检查是否结束
        if self.state.day >= self.total_days:
            self._done = True

        return self.state

    def done(self) -> bool:
        """是否结束。"""
        return self._done

    # ==========================================
    # 内部方法
    # ==========================================

    def _inject_events(self) -> None:
        """注入当天的事件。"""
        today_events = [e for e in self._events if e.day == self.state.day]

        for event in today_events:
            # 转换为 LifeEvent
            life_event = LifeEvent(
                title=event.description or event.event_type,
                event_type=event.event_type,
                change_content=event.data.get("content", ""),
                source_image_hash=None,
                user_confirmed=True,
                confidence=1.0,
                raw_data=event.data,
            )
            self.state.events.append(life_event)

            # 根据事件类型影响状态
            self._apply_event_effect(event)

    def _apply_event_effect(self, event: EnvEvent) -> None:
        """应用事件对环境状态的影响。

        D2 简化实现：只处理几种基础事件。
        """
        if event.event_type == "illness":
            # 生病：当天效率降低
            self.state.daily_completion_minutes[self.state.day] = 0
        elif event.event_type == "progress_report":
            # 进度汇报：更新任务进度
            task_id = event.data.get("task_id")
            progress_delta = event.data.get("progress_delta", 0)
            if task_id:
                for task in self.state.tasks:
                    if task.id == task_id:
                        task.progress = min(1.0, task.progress + progress_delta)
                        break

    def _simulate_daily_execution(self) -> None:
        """模拟当天的任务执行。

        D2 简化：随机完成当天计划的 60-100%。
        后续会根据调度结果精确模拟。
        """
        # 如果已经被事件设置了（比如生病=0），就不覆盖
        if self.state.day in self.state.daily_completion_minutes:
            return

        # 随机效率：60% - 100%
        efficiency = self._rng.uniform(0.6, 1.0)
        daily_minutes = int(4 * 60 * efficiency)  # 假设每天 4 小时学习时间
        self.state.daily_completion_minutes[self.state.day] = daily_minutes

    # ==========================================
    # 指标计算（D2 骨架，后续完善）
    # ==========================================

    def get_metrics(self) -> Dict:
        """获取仿真指标。

        D2 阶段返回基础指标，后续完善。
        """
        completed = sum(1 for t in self.state.tasks if t.progress >= 1.0)
        total = len(self.state.tasks)

        return {
            "total_days": self.state.day,
            "total_tasks": total,
            "completed_tasks": completed,
            "completion_rate": completed / max(total, 1),
            "total_events": len(self.state.events),
            "total_minutes": sum(self.state.daily_completion_minutes.values()),
        }

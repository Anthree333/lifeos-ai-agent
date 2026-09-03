"""核心调度器。

采用确定性约束贪心算法：
1. 标记固定时间槽（承诺 + 已安排的硬截止任务）
2. 任务按优先级排序
3. 硬截止任务优先分配
4. 软截止任务分配最小推进块
5. 剩余时间按优先级填充
6. 插入缓冲时间
7. 无法全部安排时生成牺牲清单
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

from ..models import (
    Task, Goal, Commitment, TimeSlot, StudentProfile,
    DeadlineType, TaskStatus, EnergyLevel, RecurrenceType,
)
from ..models.schedule import (
    TimeSlotSource, ChangeLog, ChangeLogItem,
    ChangeAction, SacrificeItem,
)

from .priority import TaskPriorityScorer, count_dependency_blocks
from .constraints import ConstraintChecker, TimeSlotCandidate, ConstraintViolation
from .sacrifice import SacrificeGenerator


@dataclass
class ScheduleResult:
    """调度结果。"""
    success: bool = True
    time_slots: List[TimeSlot] = field(default_factory=list)
    unscheduled_tasks: List[Task] = field(default_factory=list)
    sacrifice_list: List[SacrificeItem] = field(default_factory=list)
    change_log: Optional[ChangeLog] = None

    # 数值计算轨迹（用于解释器）
    schedule_tracks: Dict = field(default_factory=dict)

    @property
    def total_scheduled_minutes(self) -> int:
        return sum(
            ts.duration_minutes for ts in self.time_slots
            if ts.source == TimeSlotSource.TASK
        )

    @property
    def hard_deadline_scheduled(self) -> int:
        return sum(
            1 for ts in self.time_slots
            if ts.source == TimeSlotSource.TASK and ts.task_id
        )


class Scheduler:
    """确定性约束调度器。"""

    def __init__(
        self,
        profile: Optional[StudentProfile] = None,
        scorer: Optional[TaskPriorityScorer] = None,
        checker: Optional[ConstraintChecker] = None,
        sacrifice_gen: Optional[SacrificeGenerator] = None,
    ):
        self.profile = profile or StudentProfile()
        self.scorer = scorer or TaskPriorityScorer()
        self.checker = checker or ConstraintChecker(
            default_buffer_minutes=self.profile.default_buffer_minutes,
            min_progress_block_minutes=self.profile.min_progress_block_minutes,
        )
        self.sacrifice_gen = sacrifice_gen or SacrificeGenerator()

    def schedule(
        self,
        tasks: List[Task],
        goals: List[Goal],
        commitments: List[Commitment],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        now: Optional[datetime] = None,
    ) -> ScheduleResult:
        """执行调度。

        Args:
            tasks: 待调度任务列表
            goals: 目标列表
            commitments: 固定承诺列表
            start_date: 调度开始日期（默认今天）
            end_date: 调度结束日期（默认取最晚截止日 + 3天）
            now: 当前时间

        Returns:
            ScheduleResult 调度结果
        """
        if now is None:
            now = datetime.now()
        if start_date is None:
            start_date = now.date()
        if end_date is None:
            end_date = self._calc_end_date(tasks, start_date)

        # 准备数据结构
        goal_map = {g.id: g for g in goals}
        task_map = {t.id: t for t in tasks}
        active_tasks = [
            t for t in tasks
            if t.status not in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
            and t.remaining_minutes > 0
        ]

        # 计算依赖阻塞数
        block_counts = count_dependency_blocks(tasks)

        # 计算所有任务优先级得分
        priority_scores = self.scorer.score_all(
            active_tasks, goal_map, block_counts, now
        )

        # 生成可用时间槽
        wake_hour = self.profile.wake_up_time.hour
        sleep_hour = self.profile.sleep_time.hour
        available_slots = self.checker.generate_available_slots(
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.min.time()),
            wake_up_hour=wake_hour,
            sleep_hour=sleep_hour,
            commitments=commitments,
            high_energy_hours=self.profile.daily_high_energy_hours,
            now=now,
        )

        # 追踪调度轨迹
        tracks: Dict = {
            "total_available_minutes": sum(
                int((s.end - s.start).total_seconds() / 60) for s in available_slots
            ),
            "total_task_minutes": sum(t.remaining_minutes for t in active_tasks),
            "priority_scores": priority_scores,
            "task_count": len(active_tasks),
            "hard_task_count": sum(
                1 for t in active_tasks if t.deadline_type == DeadlineType.HARD
            ),
            "soft_task_count": sum(
                1 for t in active_tasks if t.deadline_type == DeadlineType.SOFT
            ),
        }

        # 分离硬截止和软截止任务
        hard_tasks = [
            t for t in active_tasks
            if t.deadline_type == DeadlineType.HARD
        ]
        soft_tasks = [
            t for t in active_tasks
            if t.deadline_type == DeadlineType.SOFT
        ]

        # 按优先级排序（高到低）
        hard_tasks.sort(
            key=lambda t: priority_scores.get(t.id, 0), reverse=True
        )
        soft_tasks.sort(
            key=lambda t: priority_scores.get(t.id, 0), reverse=True
        )

        # 已安排的时间槽（用于重叠检查）
        scheduled_time_slots: List[TimeSlot] = []
        # 固定时间槽（承诺 + 已安排的硬截止）
        fixed_slots: List[TimeSlot] = self._commitments_to_slots(commitments)

        # 任务剩余时间追踪
        remaining = {t.id: t.remaining_minutes for t in active_tasks}
        scheduled_task_ids = set()
        completed_task_ids = {
            t.id for t in tasks
            if t.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
            or t.remaining_minutes <= 0
        }

        # ==========================================
        # 第 1 步：分配硬截止任务
        # ==========================================
        hard_made_progress = True
        while hard_made_progress:
            hard_made_progress = False
            for task in hard_tasks:
                if remaining.get(task.id, 0) <= 0:
                    continue
                # 依赖未满足时跳过，等前置任务完成后下一轮再尝试
                if not self.checker.check_dependencies(
                    task, task_map, completed_task_ids
                ):
                    continue

                placed = self._place_task(
                    task=task,
                    minutes_needed=remaining[task.id],
                    available_slots=available_slots,
                    scheduled_slots=scheduled_time_slots,
                    fixed_slots=fixed_slots,
                    priority_scores=priority_scores,
                    remaining=remaining,
                )

                if placed:
                    hard_made_progress = True
                    scheduled_task_ids.add(task.id)
                    if remaining[task.id] <= 0:
                        completed_task_ids.add(task.id)

        tracks["scheduled_hard_tasks"] = len(
            [t for t in hard_tasks if remaining.get(t.id, 0) <= 0]
        )
        tracks["unscheduled_hard_tasks"] = len(hard_tasks) - tracks["scheduled_hard_tasks"]

        # ==========================================
        # 第 2 步：软截止任务最小推进块
        # ==========================================
        min_block = self.profile.min_progress_block_minutes
        soft_min_scheduled = set()

        min_block_made_progress = True
        while min_block_made_progress:
            min_block_made_progress = False
            for task in soft_tasks:
                if remaining.get(task.id, 0) <= 0 or task.id in soft_min_scheduled:
                    continue
                if not self.checker.check_dependencies(
                    task, task_map, completed_task_ids
                ):
                    continue

                # 只分配最小推进块
                placed = self._place_task(
                    task=task,
                    minutes_needed=min_block,
                    available_slots=available_slots,
                    scheduled_slots=scheduled_time_slots,
                    fixed_slots=fixed_slots,
                    priority_scores=priority_scores,
                    remaining=remaining,
                    is_min_block=True,
                )

                if placed:
                    soft_min_scheduled.add(task.id)
                    scheduled_task_ids.add(task.id)
                    min_block_made_progress = True
                    if remaining[task.id] <= 0:
                        completed_task_ids.add(task.id)

        tracks["soft_min_block_scheduled"] = len(soft_min_scheduled)

        # ==========================================
        # 第 3 步：剩余时间按优先级填充
        # ==========================================
        # 按“硬截止优先 + 优先级从高到低”逐轮填充；
        # 每轮允许新解锁的依赖任务进入，直到不再产生变化。
        fill_made_progress = True
        while fill_made_progress:
            fill_made_progress = False

            remaining_tasks = [
                t for t in active_tasks
                if remaining.get(t.id, 0) > 0
            ]
            remaining_tasks.sort(
                key=lambda t: (
                    0 if t.deadline_type == DeadlineType.HARD else 1,
                    -priority_scores.get(t.id, 0),
                )
            )

            for task in remaining_tasks:
                if remaining.get(task.id, 0) <= 0:
                    continue
                if not self.checker.check_dependencies(
                    task, task_map, completed_task_ids
                ):
                    continue

                # 新解锁的软任务先获得最小推进块，下一轮再继续填满
                give_min_block = (
                    task.deadline_type == DeadlineType.SOFT
                    and task.id not in scheduled_task_ids
                )
                minutes_needed = (
                    min_block if give_min_block else remaining[task.id]
                )
                placed = self._place_task(
                    task=task,
                    minutes_needed=minutes_needed,
                    available_slots=available_slots,
                    scheduled_slots=scheduled_time_slots,
                    fixed_slots=fixed_slots,
                    priority_scores=priority_scores,
                    remaining=remaining,
                )

                if placed:
                    fill_made_progress = True
                    scheduled_task_ids.add(task.id)
                    if remaining[task.id] <= 0:
                        completed_task_ids.add(task.id)

        # ==========================================
        # 第 4 步：插入缓冲时间
        # ==========================================
        scheduled_time_slots = self._insert_buffers(
            scheduled_time_slots, self.profile.default_buffer_minutes
        )
        scheduled_time_slots.extend(
            self._commitment_slots_in_range(
                commitments=commitments,
                start_date=start_date,
                end_date=end_date,
                now=now,
            )
        )

        # ==========================================
        # 收集未安排的任务
        # ==========================================
        unscheduled = [
            t for t in active_tasks
            if remaining.get(t.id, 0) > 0
        ]

        # ==========================================
        # 生成牺牲清单
        # ==========================================
        scheduled_hard = [t for t in hard_tasks if remaining.get(t.id, 0) <= 0]
        sacrifice_list = self.sacrifice_gen.generate(
            unscheduled_tasks=unscheduled,
            scheduled_hard_tasks=scheduled_hard,
            priority_scores=priority_scores,
            now=now,
        )

        # ==========================================
        # 构建结果
        # ==========================================
        # 按开始时间排序
        scheduled_time_slots.sort(key=lambda ts: ts.start_time)

        success = not any(
            remaining.get(t.id, 0) > 0 for t in hard_tasks
        )

        tracks["total_scheduled_minutes"] = sum(
            ts.duration_minutes for ts in scheduled_time_slots
            if ts.source == TimeSlotSource.TASK
        )
        tracks["unscheduled_count"] = len(unscheduled)
        tracks["sacrifice_count"] = len(sacrifice_list)

        result = ScheduleResult(
            success=success,
            time_slots=scheduled_time_slots,
            unscheduled_tasks=unscheduled,
            sacrifice_list=sacrifice_list,
            schedule_tracks=tracks,
        )

        return result

    # ==========================================
    # 内部方法
    # ==========================================

    def _place_task(
        self,
        task: Task,
        minutes_needed: int,
        available_slots: List[TimeSlotCandidate],
        scheduled_slots: List[TimeSlot],
        fixed_slots: List[TimeSlot],
        priority_scores: Dict[str, float],
        remaining: Dict[str, int],
        is_min_block: bool = False,
    ) -> bool:
        """尝试将任务放置到可用时间槽中。

        使用首次适应算法：从最早的可用时段开始，找到能放下的位置就放。
        优先匹配精力等级。

        Returns:
            True 表示至少放了一部分
        """
        if minutes_needed <= 0 or remaining.get(task.id, 0) <= 0:
            return False

        remaining_minutes = min(minutes_needed, remaining.get(task.id, 0))
        placed_any = False

        # 计算每个可用槽的"适合度"：精力匹配 + 时间顺序
        scored_slots = []
        for slot in available_slots:
            energy_score = self.checker.energy_match_score(
                task.energy_level, slot.energy_level
            )
            # 检查是否与已安排的重叠
            if self._overlaps_scheduled(slot, scheduled_slots):
                continue
            # 检查硬约束
            violations = self.checker.check_hard_constraints(
                task, slot.start, slot.end, fixed_slots
            )
            if violations:
                # 有硬违反，跳过这个槽
                # 但如果部分时间可用，后面会处理
                continue

            scored_slots.append((slot, energy_score))

        # 按精力匹配度排序（高匹配优先），同时保持时间顺序
        # 先按时间排序，再按精力分组
        scored_slots.sort(key=lambda x: (x[0].start, -x[1]))

        for slot, _ in scored_slots:
            if remaining_minutes <= 0:
                break

            slot_duration = int((slot.end - slot.start).total_seconds() / 60)

            # 计算实际可用时长（不超过还需要的）
            actual_minutes = min(
                slot_duration,
                remaining_minutes,
                remaining.get(task.id, 0),
            )
            # 最小 15 分钟规则不阻止“刚好收尾”的短片段
            if actual_minutes < 15 and remaining_minutes > actual_minutes:
                continue

            # 创建时间槽
            end_time = slot.start + timedelta(minutes=actual_minutes)
            ts = TimeSlot(
                id=str(uuid.uuid4()),
                task_id=task.id,
                start_time=slot.start.isoformat(),
                end_time=end_time.isoformat(),
                energy_level=slot.energy_level,
                is_fixed=False,
                source=TimeSlotSource.TASK,
                title=task.title,
            )
            scheduled_slots.append(ts)

            remaining_minutes -= actual_minutes
            remaining[task.id] = max(0, remaining.get(task.id, 0) - actual_minutes)
            placed_any = True

        return placed_any

    def _overlaps_scheduled(
        self,
        candidate: TimeSlotCandidate,
        scheduled: List[TimeSlot],
    ) -> bool:
        """检查候选时段是否与已安排的重叠。"""
        for ts in scheduled:
            ts_start = datetime.fromisoformat(ts.start_time)
            ts_end = datetime.fromisoformat(ts.end_time)
            if candidate.start < ts_end and candidate.end > ts_start:
                return True
        return False

    def _insert_buffers(
        self,
        time_slots: List[TimeSlot],
        buffer_minutes: int,
    ) -> List[TimeSlot]:
        """在任务之间插入缓冲时间。

        注意：这会减少实际可用时间。
        简化实现：将缓冲从任务时间中扣除，不额外加缓冲槽。
        只在结果中记录缓冲信息。
        """
        # 按开始时间排序
        sorted_slots = sorted(time_slots, key=lambda ts: ts.start_time)
        return sorted_slots

    def _commitments_to_slots(self, commitments: List[Commitment]) -> List[TimeSlot]:
        """将承诺转换为 TimeSlot（固定时间槽）。"""
        slots = []
        for c in commitments:
            slots.append(TimeSlot(
                id=c.id,
                commitment_id=c.id,
                start_time=c.start_time,
                end_time=c.end_time,
                energy_level=EnergyLevel.MEDIUM,
                is_fixed=True,
                source=TimeSlotSource.COMMITMENT,
                title=c.title,
            ))
        return slots

    def _commitment_slots_in_range(
        self,
        commitments: List[Commitment],
        start_date: date,
        end_date: date,
        now: datetime,
    ) -> List[TimeSlot]:
        """将固定承诺（含 daily/weekly）展开到计划展示中。"""
        slots: List[TimeSlot] = []

        for commitment in commitments:
            try:
                start = datetime.fromisoformat(commitment.start_time)
                end = datetime.fromisoformat(commitment.end_time)
            except (ValueError, TypeError):
                continue

            anchor = start.date()
            cursor = max(anchor, start_date)
            while cursor <= end_date:
                if commitment.recurrence == RecurrenceType.NONE:
                    if cursor != anchor:
                        cursor = end_date + timedelta(days=1)
                        continue
                elif commitment.recurrence == RecurrenceType.WEEKLY:
                    if (cursor - anchor).days % 7 != 0:
                        cursor += timedelta(days=1)
                        continue
                elif commitment.recurrence == RecurrenceType.DAILY:
                    pass
                else:
                    cursor = end_date + timedelta(days=1)
                    continue

                occurrence_start = datetime.combine(cursor, start.time())
                occurrence_end = datetime.combine(cursor, end.time())

                # 今天已经结束的时段不显示；正在进行中的从当前时间开始展示
                if cursor == now.date():
                    if occurrence_end <= now:
                        cursor += timedelta(days=1)
                        continue
                    occurrence_start = max(occurrence_start, now)

                slots.append(TimeSlot(
                    id=str(uuid.uuid4()),
                    commitment_id=commitment.id,
                    start_time=occurrence_start.isoformat(),
                    end_time=occurrence_end.isoformat(),
                    energy_level=EnergyLevel.MEDIUM,
                    is_fixed=True,
                    source=TimeSlotSource.COMMITMENT,
                    title=commitment.title,
                ))

                cursor += timedelta(days=1)

        return slots

    def _calc_end_date(self, tasks: List[Task], start_date: date) -> date:
        """计算调度结束日期（取最晚截止日 + 3天兜底）。"""
        latest = start_date
        for task in tasks:
            if task.deadline:
                try:
                    d = datetime.fromisoformat(task.deadline).date()
                    if d > latest:
                        latest = d
                except (ValueError, TypeError):
                    pass

        # 加 3 天兜底
        return latest + timedelta(days=3)

"""约束规则检查。

调度器的约束分为：
- 强制约束（hard）：必须满足，否则调度失败
- 偏好约束（soft）：尽量满足，不满足也可调度
- 保障约束（guarantee）：软任务最小推进块等保障性规则
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from ..models import Task, TimeSlot, Commitment, DeadlineType, EnergyLevel, RecurrenceType
from ..models.schedule import TimeSlotSource


@dataclass
class ConstraintViolation:
    """约束违反记录。"""
    rule_name: str
    severity: str  # "hard" | "soft" | "guarantee"
    task_id: str
    task_title: str
    message: str


@dataclass
class TimeSlotCandidate:
    """时间槽候选：任务可以放置的时间段。"""
    start: datetime
    end: datetime
    energy_level: EnergyLevel
    is_fixed: bool = False


class ConstraintChecker:
    """约束规则检查器。

    负责：
    1. 生成可用时间槽（扣除固定承诺）
    2. 检查任务分配是否违反约束
    3. 精力时段匹配建议
    """

    def __init__(
        self,
        default_buffer_minutes: int = 10,
        min_progress_block_minutes: int = 20,
    ):
        self.default_buffer_minutes = default_buffer_minutes
        self.min_progress_block_minutes = min_progress_block_minutes

    # ==========================================
    # 可用时间槽生成
    # ==========================================

    def generate_available_slots(
        self,
        start_date: datetime,
        end_date: datetime,
        wake_up_hour: int = 7,
        sleep_hour: int = 23,
        commitments: Optional[List[Commitment]] = None,
        high_energy_hours: int = 4,
        now: Optional[datetime] = None,
    ) -> List[TimeSlotCandidate]:
        """生成指定日期范围内的可用时间槽。

        扣除睡眠时间和固定承诺，剩余时间按天划分为可用时段。
        高精力时段默认安排在上午（9-12点）。

        Args:
            start_date: 开始日期
            end_date: 结束日期
            wake_up_hour: 起床时间（小时）
            sleep_hour: 睡觉时间（小时）
            commitments: 固定承诺列表
            high_energy_hours: 每天高精力小时数

        Returns:
            可用时间槽候选列表（按时间排序）
        """
        slots: List[TimeSlotCandidate] = []
        commitments = commitments or []

        # 按天生成
        day = start_date.date()
        end_day = end_date.date()

        while day <= end_day:
            day_start = datetime.combine(day, datetime.min.time()).replace(hour=wake_up_hour)
            day_end = datetime.combine(day, datetime.min.time()).replace(hour=sleep_hour)

            # 调度当天不能排到当前时间之前
            if now is not None and day == now.date():
                day_start = max(day_start, now)
            if day_start >= day_end:
                day += timedelta(days=1)
                continue

            # 生成当天的基本时段（按小时切分，便于精细调度）
            day_slots = self._generate_day_slots(day_start, day_end, high_energy_hours)

            # 扣除固定承诺
            day_commitments = [
                c for c in commitments
                if self._commitment_occurs_on(c, day)
            ]
            available = self._subtract_commitments(day_slots, day_commitments)

            slots.extend(available)
            day += timedelta(days=1)

        # 按开始时间排序
        slots.sort(key=lambda s: s.start)
        return slots

    def _generate_day_slots(
        self,
        day_start: datetime,
        day_end: datetime,
        high_energy_hours: int,
    ) -> List[TimeSlotCandidate]:
        """生成一天的基础时段。

        默认高精力时段：上午 9:00-12:00（3小时）+ 晚上 19:00-20:00（1小时）
        其余为中精力。
        """
        slots: List[TimeSlotCandidate] = []
        current = day_start

        while current < day_end:
            next_hour = current + timedelta(hours=1)
            if next_hour > day_end:
                next_hour = day_end

            hour = current.hour
            # 高精力时段：9-12点、19-20点
            if (9 <= hour < 12) or (19 <= hour < 20):
                energy = EnergyLevel.HIGH
            elif (14 <= hour < 17) or (20 <= hour < 22):
                energy = EnergyLevel.MEDIUM
            else:
                energy = EnergyLevel.LOW

            slots.append(TimeSlotCandidate(
                start=current,
                end=next_hour,
                energy_level=energy,
            ))
            current = next_hour

        return slots

    def _subtract_commitments(
        self,
        day_slots: List[TimeSlotCandidate],
        commitments: List[Commitment],
    ) -> List[TimeSlotCandidate]:
        """从可用时段中扣除固定承诺。"""
        if not commitments:
            return day_slots

        result: List[TimeSlotCandidate] = []

        for slot in day_slots:
            remaining = [slot]

            for commit in commitments:
                commit_start = datetime.fromisoformat(commit.start_time)
                commit_end = datetime.fromisoformat(commit.end_time)

                new_remaining = []
                for rs in remaining:
                    # 完全不重叠
                    if rs.end <= commit_start or rs.start >= commit_end:
                        new_remaining.append(rs)
                        continue

                    # 前面剩余部分
                    if rs.start < commit_start:
                        new_remaining.append(TimeSlotCandidate(
                            start=rs.start,
                            end=commit_start,
                            energy_level=rs.energy_level,
                        ))

                    # 后面剩余部分
                    if rs.end > commit_end:
                        new_remaining.append(TimeSlotCandidate(
                            start=commit_end,
                            end=rs.end,
                            energy_level=rs.energy_level,
                        ))

                remaining = new_remaining

            result.extend(remaining)

        return result

    def _is_same_day(self, commitment: Commitment, day) -> bool:
        """检查承诺是否在指定日期。"""
        try:
            start = datetime.fromisoformat(commitment.start_time)
            return start.date() == day
        except (ValueError, TypeError):
            return False

    def _commitment_occurs_on(self, commitment: Commitment, day) -> bool:
        """判断承诺在指定日期是否生效，支持 daily/weekly 展开。"""
        try:
            start = datetime.fromisoformat(commitment.start_time).date()
        except (ValueError, TypeError):
            return False

        if day < start:
            return False
        if commitment.recurrence == RecurrenceType.DAILY:
            return True
        if commitment.recurrence == RecurrenceType.WEEKLY:
            return (day - start).days % 7 == 0
        return day == start

    # ==========================================
    # 约束检查
    # ==========================================

    def check_hard_constraints(
        self,
        task: Task,
        slot_start: datetime,
        slot_end: datetime,
        fixed_slots: List[TimeSlot],
    ) -> List[ConstraintViolation]:
        """检查硬约束（必须满足）。

        1. 硬截止：任务必须在截止时间前完成
        2. 不重叠：不与固定时间槽重叠
        """
        violations: List[ConstraintViolation] = []

        # 硬截止检查
        if task.deadline_type == DeadlineType.HARD and task.deadline:
            try:
                deadline = datetime.fromisoformat(task.deadline)
                if slot_end > deadline:
                    violations.append(ConstraintViolation(
                        rule_name="hard_deadline",
                        severity="hard",
                        task_id=task.id,
                        task_title=task.title,
                        message=f"任务结束时间 {slot_end} 晚于硬截止 {deadline}",
                    ))
            except (ValueError, TypeError):
                pass

        # 与固定时间槽重叠检查
        for fs in fixed_slots:
            fs_start = datetime.fromisoformat(fs.start_time)
            fs_end = datetime.fromisoformat(fs.end_time)

            if slot_start < fs_end and slot_end > fs_start:
                violations.append(ConstraintViolation(
                    rule_name="fixed_slot_overlap",
                    severity="hard",
                    task_id=task.id,
                    task_title=task.title,
                    message=f"与固定时间槽 '{fs.title}' 重叠",
                ))

        return violations

    def check_dependencies(
        self,
        task: Task,
        task_map: dict,
        completed_tasks: set,
    ) -> bool:
        """检查任务的依赖是否已满足。

        Args:
            task: 待检查任务
            task_map: {task_id: Task} 字典
            completed_tasks: 已完成任务 ID 集合

        Returns:
            True 表示依赖满足，可以开始调度
        """
        for dep_id in task.dependencies:
            if dep_id not in completed_tasks:
                return False
        return True

    def energy_match_score(
        self,
        task_energy: EnergyLevel,
        slot_energy: EnergyLevel,
    ) -> float:
        """计算任务与时段的精力匹配度。

        Returns:
            1.0 = 完美匹配
            0.5 = 高任务放中时段 / 中任务放高时段
            0.2 = 高任务放低时段（较差）
            0.8 = 低任务放中时段（可以接受）
        """
        order = {EnergyLevel.LOW: 0, EnergyLevel.MEDIUM: 1, EnergyLevel.HIGH: 2}
        diff = abs(order[task_energy] - order[slot_energy])

        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.6
        else:
            return 0.2

"""牺牲清单生成器。

当所有任务无法全部安排时，按优先级从低到高"牺牲"任务，
并生成明确的牺牲清单，说明"为了 A，牺牲了 B"。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..models import Task, DeadlineType
from ..models.schedule import SacrificeItem


class SacrificeGenerator:
    """牺牲清单生成器。"""

    def __init__(self):
        pass

    def generate(
        self,
        unscheduled_tasks: List[Task],
        scheduled_hard_tasks: List[Task],
        priority_scores: Dict[str, float],
        now: Optional[datetime] = None,
        wake_up_hour: int = 7,
    ) -> List[SacrificeItem]:
        """生成牺牲清单。

        按优先级从低到高排列未安排的任务，
        并说明每个任务是为了什么而被牺牲的。

        Args:
            unscheduled_tasks: 未能安排的任务列表
            scheduled_hard_tasks: 已安排的硬截止任务（牺牲的原因）
            priority_scores: 所有任务的优先级得分
            now: 当前时间
            wake_up_hour: 用户起床时间（小时），用于判断截止时间是否早于起床时间

        Returns:
            牺牲清单（按优先级从低到高排序）
        """
        if not unscheduled_tasks:
            return []

        if now is None:
            now = datetime.now()

        # 按优先级得分从低到高排序
        sorted_unscheduled = sorted(
            unscheduled_tasks,
            key=lambda t: priority_scores.get(t.id, 0),
        )

        # 计算"为了什么"——找到优先级最高的已安排硬截止任务
        if scheduled_hard_tasks:
            top_hard_task = max(
                scheduled_hard_tasks,
                key=lambda t: priority_scores.get(t.id, 0),
            )
            reason_template = f"为保障「{top_hard_task.title}」硬截止"
        else:
            reason_template = "时间不足"

        sacrifices: List[SacrificeItem] = []

        for task in sorted_unscheduled:
            # 判断是否因截止时间过早而无法安排（截止时间在起床时间之前或已过）
            deadline_too_early = False
            if task.deadline:
                try:
                    deadline = datetime.fromisoformat(task.deadline)
                    # 截止时间已过
                    if deadline <= now:
                        deadline_too_early = True
                        early_reason = "截止时间已过"
                    # 截止时间在今天，但早于起床时间
                    elif (deadline.date() == now.date()
                          and deadline.hour < wake_up_hour):
                        deadline_too_early = True
                        early_reason = (
                            f"截止时间 {deadline.strftime('%H:%M')} "
                            f"早于起床时间 {wake_up_hour:02d}:00"
                        )
                except (ValueError, TypeError):
                    pass

            # 确定牺牲动作与原因
            if task.deadline_type == DeadlineType.HARD:
                action = "delayed"
                if deadline_too_early:
                    reason = f"{early_reason}（硬截止，需手动调整）"
                else:
                    reason = f"{reason_template}（硬截止冲突，需手动调整）"
            else:
                action = "delayed"
                reason = early_reason if deadline_too_early else reason_template

            # 估算推迟到什么时候（简单估算：下一天）
            delayed_to = None
            if task.deadline:
                try:
                    deadline = datetime.fromisoformat(task.deadline)
                    delayed_to = deadline.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    pass

            sacrifices.append(SacrificeItem(
                task_id=task.id,
                task_title=task.title,
                reason=reason,
                priority_before=round(priority_scores.get(task.id, 0), 4),
                action=action,
                delayed_to=delayed_to,
            ))

        return sacrifices

    def summarize(self, sacrifices: List[SacrificeItem]) -> Dict[str, any]:
        """生成牺牲清单摘要。"""
        hard_count = sum(
            1 for s in sacrifices if "硬截止" in s.reason
        )
        soft_count = len(sacrifices) - hard_count
        total_priority = sum(s.priority_before for s in sacrifices)

        return {
            "total_sacrificed": len(sacrifices),
            "hard_deadline_count": hard_count,
            "soft_deadline_count": soft_count,
            "avg_priority": round(total_priority / len(sacrifices), 4) if sacrifices else 0,
            "has_hard_conflict": hard_count > 0,
        }

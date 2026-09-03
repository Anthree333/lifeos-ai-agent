"""任务优先级评分。

综合考虑：目标权重、剩余工作量、依赖阻塞数、紧迫程度。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ..models import Task, Goal, DeadlineType


class TaskPriorityScorer:
    """任务优先级评分器。

    评分公式：
    priority_score = goal_weight * 0.4
                   + remaining_ratio * 0.3
                   + dependency_block_count * 0.2
                   + urgency_bonus * 0.1
    """

    def __init__(
        self,
        goal_weight_factor: float = 0.4,
        remaining_factor: float = 0.3,
        dependency_factor: float = 0.2,
        urgency_factor: float = 0.1,
        hard_deadline_bonus: float = 0.5,
    ):
        self.goal_weight_factor = goal_weight_factor
        self.remaining_factor = remaining_factor
        self.dependency_factor = dependency_factor
        self.urgency_factor = urgency_factor
        self.hard_deadline_bonus = hard_deadline_bonus

    def score(
        self,
        task: Task,
        goal: Optional[Goal] = None,
        block_count: int = 0,
        now: Optional[datetime] = None,
    ) -> float:
        """计算单个任务的优先级得分。

        Args:
            task: 任务对象
            goal: 所属目标（可选，用于获取目标权重）
            block_count: 被该任务阻塞的下游任务数
            now: 当前时间（用于计算紧迫度）

        Returns:
            优先级得分（越高越优先）
        """
        if now is None:
            now = datetime.now()

        # 1. 目标权重 (0 ~ 1)
        goal_weight = goal.weight if goal else task.priority_weight

        # 2. 剩余工作量比例 (0 ~ 1)
        # 剩余越多，越需要优先安排（避免临近截止才赶工）
        if task.estimated_minutes > 0:
            remaining_ratio = task.remaining_minutes / task.estimated_minutes
        else:
            remaining_ratio = 0.0

        # 3. 依赖阻塞数（归一化到 0~1）
        # 阻塞越多，越应该优先完成（解锁下游任务）
        block_score = min(block_count / 10.0, 1.0)  # 最多按 10 个阻塞封顶

        # 4. 紧迫度加成
        urgency_bonus = self._calc_urgency_bonus(task, now)

        # 综合得分
        score = (
            goal_weight * self.goal_weight_factor
            + remaining_ratio * self.remaining_factor
            + block_score * self.dependency_factor
            + urgency_bonus * self.urgency_factor
        )

        return round(score, 4)

    def score_all(
        self,
        tasks: List[Task],
        goals: Dict[str, Goal],
        task_blocks: Dict[str, int],
        now: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """批量计算所有任务的优先级得分。

        Args:
            tasks: 任务列表
            goals: 目标字典 {goal_id: Goal}
            task_blocks: 每个任务阻塞的下游数 {task_id: count}
            now: 当前时间

        Returns:
            {task_id: score} 字典
        """
        scores = {}
        for task in tasks:
            if task.status == "completed" or task.status == "cancelled":
                scores[task.id] = -1.0  # 已完成/取消的任务不参与调度
                continue

            goal = goals.get(task.goal_id)
            block_count = task_blocks.get(task.id, 0)
            scores[task.id] = self.score(task, goal, block_count, now)

        return scores

    def _calc_urgency_bonus(self, task: Task, now: datetime) -> float:
        """计算紧迫度加成 (0 ~ 1 + hard_deadline_bonus)。

        - 硬截止额外加分
        - 距离截止越近分越高
        - 没有截止时间的任务得 0 分
        """
        if not task.deadline:
            return 0.0

        try:
            deadline = datetime.fromisoformat(task.deadline)
        except (ValueError, TypeError):
            return 0.0

        days_left = (deadline - now).total_seconds() / 86400.0

        if days_left <= 0:
            # 已过期，最高紧迫度
            urgency = 1.0
        elif days_left >= 30:
            # 30 天以上，紧迫度很低
            urgency = 0.1
        else:
            # 1 ~ 30 天线性递减
            urgency = 1.0 - (days_left / 30.0) * 0.9
            urgency = max(0.1, min(1.0, urgency))

        # 硬截止额外加分
        if task.deadline_type == DeadlineType.HARD:
            urgency += self.hard_deadline_bonus

        return urgency


def count_dependency_blocks(tasks: List[Task]) -> Dict[str, int]:
    """统计每个任务阻塞的下游任务数量。

    即：有多少个任务直接或间接依赖该任务。

    Args:
        tasks: 任务列表

    Returns:
        {task_id: block_count} 字典
    """
    # 构建反向依赖图：task -> 依赖它的任务列表
    reverse_deps: Dict[str, List[str]] = {t.id: [] for t in tasks}
    task_map = {t.id: t for t in tasks}

    for task in tasks:
        for dep_id in task.dependencies:
            if dep_id in reverse_deps:
                reverse_deps[dep_id].append(task.id)

    # DFS 计算每个任务阻塞的总数
    result: Dict[str, int] = {}

    def dfs(task_id: str, visited: set) -> int:
        if task_id in visited:
            return 0
        visited.add(task_id)

        count = 0
        for downstream_id in reverse_deps.get(task_id, []):
            count += 1 + dfs(downstream_id, visited)
        return count

    for task in tasks:
        result[task.id] = dfs(task.id, set())

    return result

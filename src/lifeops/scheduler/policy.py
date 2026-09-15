"""调度策略参数。

由 PlannerAgent 或 LLM 生成，注入确定性调度器。
默认值保证确定性（同输入同输出），LLM 策略仅在显式传入时生效。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SchedulePolicy:
    """调度策略参数。"""

    # 专注块
    max_focus_block_minutes: int = 90      # 单次专注上限（番茄钟）
    min_focus_block_minutes: int = 25      # 最小专注块
    # 每日上限
    max_daily_minutes: int = 360           # 每天学习上限（防过劳）
    # 排列策略
    prefer_contiguous: bool = True         # 优先连续安排（避免碎片化）
    distribute_across_days: bool = True    # 大任务跨天分散
    # 精力匹配
    energy_match_strictness: float = 0.6    # 精力匹配严格度 (0-1)
    # 缓冲
    buffer_between_tasks: int = 10         # 不同任务间缓冲分钟
    # 硬截止安全
    hard_deadline_safety_margin_days: int = 1  # 硬截止前留安全余量天数

    @property
    def is_default(self) -> bool:
        """是否为默认策略（保证确定性）。"""
        return (
            self.max_focus_block_minutes == 90
            and self.min_focus_block_minutes == 25
            and self.max_daily_minutes == 360
            and self.prefer_contiguous is True
            and self.distribute_across_days is True
            and self.energy_match_strictness == 0.6
            and self.buffer_between_tasks == 10
            and self.hard_deadline_safety_margin_days == 1
        )

    @classmethod
    def default(cls) -> "SchedulePolicy":
        """返回默认策略实例。"""
        return cls()

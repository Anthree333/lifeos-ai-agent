"""目标与任务模型。"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from .profile import EnergyLevel


class DeadlineType(str, Enum):
    """截止类型。"""
    HARD = "hard"  # 硬截止
    SOFT = "soft"  # 软截止


class TaskStatus(str, Enum):
    """任务状态。"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DELAYED = "delayed"


class Goal(BaseModel):
    """目标。"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str
    title: str
    description: str = ""
    weight: float = 0.5  # 0-1，目标权重
    deadline: Optional[str] = None  # ISO date string
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class Task(BaseModel):
    """任务。"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str
    parent_id: Optional[str] = None  # 父任务 ID，支持层级
    title: str
    description: str = ""
    estimated_minutes: int = 60
    actual_minutes: int = 0  # 实际已花费分钟数
    energy_level: EnergyLevel = EnergyLevel.MEDIUM
    deadline_type: DeadlineType = DeadlineType.SOFT
    deadline: Optional[str] = None  # ISO datetime string
    priority_weight: float = 0.5  # 0-1，任务自身优先级
    progress: float = 0.0  # 0-1
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(default_factory=list)  # 依赖任务 ID 列表
    order_index: int = 0  # 同层级排序
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def remaining_minutes(self) -> int:
        """剩余分钟数。"""
        return max(0, int(self.estimated_minutes * (1 - self.progress)))

    @property
    def priority(self) -> int:
        """UI 展示用 1-10 优先级（内部权重为 0-1）。"""
        return max(1, min(10, int(round(self.priority_weight * 10))))

    @property
    def is_completed(self) -> bool:
        return self.progress >= 1.0 or self.status == TaskStatus.COMPLETED

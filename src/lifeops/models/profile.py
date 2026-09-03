"""学生档案与承诺模型。"""
from __future__ import annotations

import uuid
from datetime import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EnergyLevel(str, Enum):
    """精力档位。"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CommitmentType(str, Enum):
    """承诺类型。"""
    CLASS = "class"           # 课程
    ACTIVITY = "activity"     # 社团活动
    MEETING = "meeting"       # 会议
    OTHER = "other"           # 其他


class RecurrenceType(str, Enum):
    """重复类型。"""
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"


class StudentProfile(BaseModel):
    """学生基本信息与偏好设置。"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "学生"
    daily_high_energy_hours: int = 4
    wake_up_time: time = Field(default_factory=lambda: time(7, 0))
    sleep_time: time = Field(default_factory=lambda: time(23, 0))
    default_buffer_minutes: int = 10
    min_progress_block_minutes: int = 20  # 软任务最小推进块

    class Config:
        json_encoders = {time: lambda v: v.isoformat()}


class Commitment(BaseModel):
    """固定承诺（课表、固定活动等，不可移动）。"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str
    title: str
    type: CommitmentType = CommitmentType.OTHER
    start_time: str  # ISO datetime string
    end_time: str    # ISO datetime string
    recurrence: RecurrenceType = RecurrenceType.NONE
    description: Optional[str] = None

"""事件与执行记录模型。"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """事件类型。"""
    EXAM = "exam"                       # 考试
    ASSIGNMENT = "assignment"           # 作业
    SCHEDULE_CHANGE = "schedule_change" # 改期
    NOTIFICATION = "notification"       # 通知
    ILLNESS = "illness"                 # 生病
    ACTIVITY = "activity"               # 临时活动
    PROGRESS_REPORT = "progress_report" # 进度汇报
    OTHER = "other"                     # 其他


class LifeEvent(BaseModel):
    """生活事件（从截图/用户输入解析而来）。"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    event_type: EventType = EventType.OTHER
    event_time: Optional[str] = None    # ISO datetime string
    end_time: Optional[str] = None
    change_content: str = ""
    source_image_hash: Optional[str] = None
    user_confirmed: bool = False
    confidence: float = 1.0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    raw_data: dict = Field(default_factory=dict)  # 原始解析数据


class ExecutionRecord(BaseModel):
    """执行记录。"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    record_date: str = Field(default_factory=lambda: date.today().isoformat())
    actual_minutes: int = 0
    progress_delta: float = 0.0  # 进度变化量（-1 到 1）
    note: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

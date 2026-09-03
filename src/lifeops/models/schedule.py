"""计划与时间槽模型。"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field

from .profile import EnergyLevel


class TimeSlotSource(str, Enum):
    """时间槽来源。"""
    TASK = "task"           # 任务安排
    COMMITMENT = "commitment"  # 固定承诺
    BUFFER = "buffer"       # 缓冲时间
    REST = "rest"           # 休息


class TimeSlot(BaseModel):
    """时间片分配。"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: Optional[str] = None
    commitment_id: Optional[str] = None
    start_time: str  # ISO datetime string
    end_time: str    # ISO datetime string
    energy_level: EnergyLevel = EnergyLevel.MEDIUM
    is_fixed: bool = False
    source: TimeSlotSource = TimeSlotSource.TASK
    title: str = ""

    @property
    def duration_minutes(self) -> int:
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        return int((end - start).total_seconds() / 60)


class ChangeAction(str, Enum):
    """变更动作。"""
    ADDED = "added"          # 新增
    MOVED = "moved"          # 后移/前移
    REMOVED = "removed"      # 删除
    SHORTENED = "shortened"  # 缩短
    EXTENDED = "extended"    # 延长
    KEPT = "kept"            # 保留


class ChangeLogItem(BaseModel):
    """变更记录条目。"""
    task_id: str
    task_title: str
    action: ChangeAction
    old_time: Optional[str] = None
    new_time: Optional[str] = None
    old_duration: Optional[int] = None
    new_duration: Optional[int] = None
    reason: str = ""


class SacrificeItem(BaseModel):
    """牺牲清单条目。"""
    task_id: str
    task_title: str
    reason: str              # 为了什么而牺牲
    priority_before: float   # 原优先级得分
    action: str = "delayed"  # delayed | dropped
    delayed_to: Optional[str] = None  # 推迟到哪天


class ChangeLog(BaseModel):
    """变更清单。"""
    trigger_reason: str = ""  # 触发原因标签
    summary: str = ""         # 简短摘要（<= 100 字）
    detail: str = ""          # 详细中文解释
    reason: str = ""          # 变更原因
    affected_tasks: List[str] = Field(default_factory=list)  # 受影响的任务 ID/标题
    items: List[ChangeLogItem] = Field(default_factory=list)
    sacrifice_list: List[SacrificeItem] = Field(default_factory=list)


class PlanSnapshot(BaseModel):
    """计划快照。"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str = ""
    version: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    trigger_reason: str = ""
    change_log: Optional[ChangeLog] = None
    time_slots: List[TimeSlot] = Field(default_factory=list)
    sacrifice_list: List[SacrificeItem] = Field(default_factory=list)
    schedule_tracks: Dict[str, Any] = Field(default_factory=dict)
    hard_deadline_count: int = 0
    risk_count: int = 0

    class Config:
        arbitrary_types_allowed = True

    @property
    def total_scheduled_minutes(self) -> int:
        from .schedule import TimeSlotSource
        return sum(
            ts.duration_minutes for ts in self.time_slots
            if ts.source == TimeSlotSource.TASK
        )

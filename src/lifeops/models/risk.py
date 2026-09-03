"""风险报告模型。"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """风险等级。"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskSuggestionType(str, Enum):
    """风险建议类型。"""
    ADVANCE = "advance"       # 提前安排
    DOWNGRADE = "downgrade"   # 降级（减少工作量）
    DELAY = "delay"           # 延期
    DELEGATE = "delegate"     # 寻求帮助


class RiskReport(BaseModel):
    """风险报告。"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    snapshot_id: Optional[str] = None
    task_id: str
    task_title: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    estimated_finish_date: Optional[str] = None  # ISO date string
    deadline: Optional[str] = None               # ISO date string
    delay_days: float = 0.0
    remaining_minutes: int = 0
    avg_daily_minutes: float = 0.0
    suggestion_type: RiskSuggestionType = RiskSuggestionType.ADVANCE
    suggestion_text: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_overdue(self) -> bool:
        return self.delay_days > 0

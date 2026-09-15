"""Agent 编排器。

串联 IntentAgent、PlannerAgent、SchedulerAgent、ExplainerAgent，
管理对话状态机（正常/澄清中/待确认）。

当前为骨架版本：run() 直通委托给 LifeAgent 原有流程，
后续阶段替换为真正的多 Agent 编排。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import PlanSnapshot, ChangeLog, ExecutionRecord
from .parser import ParseResult, ParseIntent
from .decomposer import GoalDecomposer
from .explainer import PlanExplainer
from ..scheduler import Scheduler, ScheduleResult
from ..scheduler.policy import SchedulePolicy


# 对话状态
DIALOGUE_STATE_NORMAL = "normal"
DIALOGUE_STATE_CLARIFYING = "clarifying"
DIALOGUE_STATE_CONFIRMING = "confirming"


@dataclass
class OrchestrationResult:
    """编排结果（映射为 AgentResult）。"""
    reply: str = ""
    snapshot: Optional[PlanSnapshot] = None
    changed: bool = False
    change_log: Optional[ChangeLog] = None
    intent: ParseIntent = ParseIntent.UNKNOWN
    schedule_result: Optional[ScheduleResult] = None
    decision_trace: List[str] = field(default_factory=list)
    risk_alerts: List[str] = field(default_factory=list)
    coaching_tips: List[str] = field(default_factory=list)
    clarification_slots: List[str] = field(default_factory=list)


class AgentOrchestrator:
    """Agent 编排器。

    串联四个专职 Agent，管理对话状态机。
    当前为骨架版本，run() 由 LifeAgent 委托调用。
    """

    def __init__(
        self,
        parser: Optional[Any] = None,
        decomposer: Optional[GoalDecomposer] = None,
        scheduler: Optional[Scheduler] = None,
        explainer: Optional[PlanExplainer] = None,
    ):
        self.parser = parser
        self.decomposer = decomposer
        self.scheduler = scheduler
        self.explainer = explainer

        # 对话状态
        self._dialogue_state = DIALOGUE_STATE_NORMAL
        # 澄清中的待补槽位
        self._pending_clarification_slots: List[str] = []
        # 澄清中的原始消息
        self._pending_clarification_message: str = ""
        # 当前调度策略（默认）
        self._active_policy: SchedulePolicy = SchedulePolicy.default()
        # 决策链（本轮收集）
        self._decision_trace: List[str] = []

    @property
    def dialogue_state(self) -> str:
        """当前对话状态。"""
        return self._dialogue_state

    @property
    def active_policy(self) -> SchedulePolicy:
        """当前调度策略。"""
        return self._active_policy

    def set_policy(self, policy: SchedulePolicy) -> None:
        """设置调度策略（由 PlannerAgent 调用）。"""
        self._active_policy = policy

    def reset_trace(self) -> None:
        """每轮对话开始时重置决策链。"""
        self._decision_trace = []

    def add_trace(self, step: str) -> None:
        """添加决策链步骤。"""
        self._decision_trace.append(step)

    def get_trace(self) -> List[str]:
        """获取本轮决策链。"""
        return list(self._decision_trace)

    def enter_clarification(self, message: str, slots: List[str]) -> None:
        """进入澄清状态。"""
        self._dialogue_state = DIALOGUE_STATE_CLARIFYING
        self._pending_clarification_message = message
        self._pending_clarification_slots = list(slots)

    def exit_clarification(self) -> None:
        """退出澄清状态。"""
        self._dialogue_state = DIALOGUE_STATE_NORMAL
        self._pending_clarification_slots = []
        self._pending_clarification_message = ""

    def enter_confirming(self) -> None:
        """进入待确认状态。"""
        self._dialogue_state = DIALOGUE_STATE_CONFIRMING

    def exit_confirming(self) -> None:
        """退出待确认状态。"""
        self._dialogue_state = DIALOGUE_STATE_NORMAL

    def run(
        self,
        user_message: str,
        now: Optional[datetime] = None,
        history: Optional[List[Dict]] = None,
        agent: Optional[Any] = None,
    ) -> OrchestrationResult:
        """编排一轮对话。

        当前为骨架版本：直接委托给 LifeAgent 的原有流程。
        后续阶段替换为真正的多 Agent 编排。

        Args:
            user_message: 用户消息
            now: 当前时间
            history: 对话历史
            agent: LifeAgent 实例（用于委托调用）

        Returns:
            OrchestrationResult
        """
        # 骨架版本：直通委托
        # 后续阶段在这里实现真正的编排逻辑：
        # 1. 对话状态判断
        # 2. 意图理解 (IntentAgent)
        # 3. 对话决策（澄清/确认/执行）
        # 4. 策略规划 (PlannerAgent)
        # 5. 状态应用 + 调度执行
        # 6. 风险评估
        # 7. 解释生成 (ExplainerAgent)

        if agent is None:
            return OrchestrationResult(reply="系统初始化中，请稍后重试。")

        # 委托给 LifeAgent 的原有 run 逻辑
        result = agent._run_original(user_message, now=now, history=history)

        # 映射为 OrchestrationResult
        return OrchestrationResult(
            reply=result.reply,
            snapshot=result.snapshot,
            changed=result.changed,
            change_log=result.change_log,
            intent=result.intent,
            schedule_result=result.schedule_result,
        )

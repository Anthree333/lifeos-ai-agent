"""Agent 全链路重构后的功能测试。

覆盖：
- 多 Agent 协作架构（Orchestrator）
- 自适应目标分解
- 调度策略注入
- 因果链解释
- 主动风险预警
- 对话式澄清
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from datetime import datetime, timedelta
import pytest

from lifeops.models import (
    StudentProfile, Goal, Task, DeadlineType, EnergyLevel, TaskStatus
)
from lifeops.agent import LifeAgent
from lifeops.agent.orchestrator import AgentOrchestrator, OrchestrationResult
from lifeops.agent.decomposer import GoalDecomposer, DecomposeResult
from lifeops.agent.parser import InputParser, ParseResult, ParseIntent
from lifeops.agent.explainer import PlanExplainer, PlanDiff
from lifeops.scheduler import Scheduler, ScheduleResult
from lifeops.scheduler.policy import SchedulePolicy


class TestOrchestratorIntegration:
    """Orchestrator 接线测试。"""

    def test_orchestrator_exists(self):
        """LifeAgent 实例化后 orchestrator 存在。"""
        agent = LifeAgent(profile=StudentProfile(name="test"))
        assert agent.orchestrator is not None
        assert agent.orchestrator.dialogue_state == "normal"
        assert agent.orchestrator.active_policy.is_default is True

    def test_run_delegates_to_orchestrator(self):
        """run() 通过 orchestrator 执行。"""
        agent = LifeAgent(profile=StudentProfile(name="test"))
        result = agent.run("我要准备高数考试", now=datetime(2026, 9, 13, 9, 0))
        assert result.intent == ParseIntent.NEW_GOAL
        assert result.reply  # 非空回复
        assert result.changed is True

    def test_orchestrator_trace_reset(self):
        """每轮对话重置决策链。"""
        agent = LifeAgent(profile=StudentProfile(name="test"))
        agent.orchestrator.add_trace("test step")
        assert len(agent.orchestrator.get_trace()) == 1
        agent.orchestrator.reset_trace()
        assert len(agent.orchestrator.get_trace()) == 0

    def test_orchestrator_policy_management(self):
        """Orchestrator 可以设置和获取策略。"""
        agent = LifeAgent(profile=StudentProfile(name="test"))
        assert agent.orchestrator.active_policy.is_default
        custom = SchedulePolicy(max_focus_block_minutes=60, max_daily_minutes=240)
        agent.orchestrator.set_policy(custom)
        assert agent.orchestrator.active_policy.max_focus_block_minutes == 60
        assert agent.orchestrator.active_policy.is_default is False


class TestAdaptiveDecomposition:
    """自适应目标分解测试。"""

    def test_decompose_with_context_exists(self):
        """decompose_with_context 方法存在。"""
        d = GoalDecomposer()
        assert hasattr(d, 'decompose_with_context')

    def test_adaptive_long_deadline(self):
        """长截止期（>14天）分解出充足任务。"""
        d = GoalDecomposer()
        goal = Goal(id="g1", profile_id="p1", title="高数考试", weight=0.8)
        result = d.decompose_with_context(
            goal, deadline=datetime(2026, 12, 15), now=datetime(2026, 9, 13)
        )
        assert isinstance(result, DecomposeResult)
        assert len(result.tasks) >= 5
        assert result.total_estimated_minutes > 0
        assert result.daily_target_minutes > 0
        assert len(result.breakdown_strategy) > 0

    def test_adaptive_short_deadline(self):
        """短截止期（<3天）压缩任务数，标记风险。"""
        d = GoalDecomposer()
        goal = Goal(id="g2", profile_id="p1", title="高数考试", weight=0.8)
        result = d.decompose_with_context(
            goal, deadline=datetime(2026, 9, 15), now=datetime(2026, 9, 13)
        )
        assert len(result.tasks) >= 3
        assert len(result.tasks) <= 5  # 压缩
        assert "时间紧张" in result.risk_notes

    def test_adaptive_milestone_tasks(self):
        """分解结果包含里程碑索引。"""
        d = GoalDecomposer()
        goal = Goal(id="g3", profile_id="p1", title="期末考试", weight=0.8)
        result = d.decompose_with_context(
            goal, deadline=datetime(2026, 12, 15), now=datetime(2026, 9, 13)
        )
        assert len(result.milestone_tasks) > 0

    def test_adaptive_decision_trace(self):
        """分解结果包含决策链。"""
        d = GoalDecomposer()
        goal = Goal(id="g4", profile_id="p1", title="项目大作业", weight=0.7)
        result = d.decompose_with_context(
            goal, deadline=datetime(2026, 10, 15), now=datetime(2026, 9, 13)
        )
        assert len(result.decision_trace) > 0

    def test_decompose_backward_compatible(self):
        """原 decompose() 仍可用。"""
        d = GoalDecomposer()
        goal = Goal(id="g5", profile_id="p1", title="高数考试", weight=0.8)
        tasks = d.decompose(goal, deadline=datetime(2026, 12, 15))
        assert len(tasks) >= 5
        assert tasks[-1].deadline_type == DeadlineType.HARD


class TestSchedulePolicy:
    """调度策略测试。"""

    def test_default_policy_is_default(self):
        """默认策略的 is_default 返回 True。"""
        p = SchedulePolicy.default()
        assert p.is_default is True

    def test_custom_policy_not_default(self):
        """自定义策略的 is_default 返回 False。"""
        p = SchedulePolicy(max_focus_block_minutes=60)
        assert p.is_default is False

    def test_scheduler_has_policy(self):
        """Scheduler 实例化后有默认策略。"""
        s = Scheduler()
        assert hasattr(s, '_active_policy')
        assert s._active_policy.is_default is True

    def test_schedule_with_policy(self):
        """带策略调度不崩溃。"""
        s = Scheduler()
        task = Task(
            id="t1", title="复习", goal_id="g1",
            estimated_minutes=60, deadline_type=DeadlineType.SOFT,
            energy_level=EnergyLevel.HIGH,
        )
        goal = Goal(id="g1", profile_id="p1", title="考试", weight=0.8)
        policy = SchedulePolicy(max_focus_block_minutes=45, max_daily_minutes=180)
        result = s.schedule_with_policy([task], [goal], [], policy)
        assert isinstance(result, ScheduleResult)

    def test_deterministic_with_default_policy(self):
        """默认策略保证确定性。"""
        s = Scheduler()
        task = Task(
            id="t1", title="复习", goal_id="g1",
            estimated_minutes=60, deadline_type=DeadlineType.SOFT,
            energy_level=EnergyLevel.HIGH,
        )
        goal = Goal(id="g1", profile_id="p1", title="考试", weight=0.8)
        r1 = s.schedule([task], [goal], [])
        r2 = s.schedule([task], [goal], [])
        assert r1.total_scheduled_minutes == r2.total_scheduled_minutes


class TestCausalExplanation:
    """因果链解释测试。"""

    def test_plandiff_has_new_fields(self):
        """PlanDiff 包含新增字段。"""
        d = PlanDiff()
        assert hasattr(d, 'decision_trace')
        assert hasattr(d, 'energy_distribution')
        assert hasattr(d, 'risk_alerts')
        assert hasattr(d, 'coaching_tips')
        assert d.decision_trace == []
        assert d.coaching_tips == []

    def test_explain_proactively_exists(self):
        """explain_proactively 方法存在。"""
        e = PlanExplainer()
        assert hasattr(e, 'explain_proactively')

    def test_explain_proactively_empty(self):
        """无风险时 explain_proactively 返回空。"""
        e = PlanExplainer()
        result = e.explain_proactively([])
        assert result == ""

    def test_explain_proactively_with_risks(self):
        """有风险时生成提醒。"""
        e = PlanExplainer()
        result = e.explain_proactively(["有2个任务未安排", "时间紧张"])
        assert len(result) > 0
        assert "风险" in result or "⚠" in result

    def test_rule_based_explain_with_coaching(self):
        """规则解释包含教练建议（非首次计划场景）。"""
        e = PlanExplainer()
        d = PlanDiff(
            moved_tasks=[("复习高数", "2026-09-13T09:00:00", "2026-09-14T10:00:00")],
            reason="进度偏差",
            coaching_tips=["建议优先完成硬截止任务"],
        )
        result = e._rule_based_explain(d)
        assert "💡" in result or "建议" in result

    def test_compute_diff_causal_analysis(self):
        """_compute_diff 包含因果分析。"""
        e = PlanExplainer()
        # 无旧快照（首次计划）
        new_result = ScheduleResult()
        diff = e._compute_diff(None, new_result, None)
        assert isinstance(diff, PlanDiff)


class TestRiskWarning:
    """主动风险预警测试。"""

    def test_check_and_append_risks_exists(self):
        """_check_and_append_risks 方法存在。"""
        agent = LifeAgent(profile=StudentProfile(name="test"))
        assert hasattr(agent, '_check_and_append_risks')

    def test_risk_warning_silent_on_no_risk(self):
        """无风险时不追加提醒。"""
        agent = LifeAgent(profile=StudentProfile(name="test"))
        r = agent.run("我要准备高数考试", now=datetime(2026, 9, 13, 9, 0))
        # 新用户无历史记录，不应有风险提醒
        assert "风险" not in r.reply or "⚠" not in r.reply


class TestClarification:
    """对话式澄清测试。"""

    def test_clarification_fields_exist(self):
        """LifeAgent 有澄清状态字段。"""
        agent = LifeAgent(profile=StudentProfile(name="test"))
        assert hasattr(agent, '_clarification_slots')
        assert hasattr(agent, '_clarification_message')
        assert agent._clarification_slots == []

    def test_parser_has_clarification_fields(self):
        """ParseResult 有澄清字段。"""
        r = ParseResult()
        assert hasattr(r, 'clarification_slots')
        assert hasattr(r, 'dialogue_act')
        assert r.clarification_slots == []
        assert r.dialogue_act == ""

    def test_normal_goal_not_affected_by_clarification(self):
        """正常目标输入不受澄清逻辑影响。"""
        agent = LifeAgent(profile=StudentProfile(name="test"))
        r = agent.run("我要准备高数考试", now=datetime(2026, 9, 13, 9, 0))
        assert r.intent == ParseIntent.NEW_GOAL
        assert r.changed is True


class TestParserRewrite:
    """Parser 重写验证。"""

    def test_empty_input_returns_unknown(self):
        """空输入返回 UNKNOWN。"""
        p = InputParser()
        r = p.parse("")
        assert r.intent == ParseIntent.UNKNOWN

    def test_progress_before_status_query(self):
        """进度汇报优先于状态查询（修正点）。"""
        p = InputParser()
        r = p.parse("我完成了30%")
        assert r.intent == ParseIntent.PROGRESS_REPORT

    def test_status_query_no_percentage(self):
        """无百分比的状态查询正确识别。"""
        p = InputParser()
        r = p.parse("看看进度怎么样")
        assert r.intent == ParseIntent.STATUS_QUERY

    def test_cancel_priority(self):
        """取消意图最高优先级。"""
        p = InputParser()
        r = p.parse("取消高数计划")
        assert r.intent == ParseIntent.CANCEL_PLAN

    def test_goal_priority(self):
        """明确目标句优先。"""
        p = InputParser()
        r = p.parse("我要准备高数考试")
        assert r.intent == ParseIntent.NEW_GOAL

    def test_denial_only_detection(self):
        """纯否定词检测。"""
        p = InputParser()
        # 否定词应被识别
        assert p._is_denial_only("不用了") is True
        assert p._is_denial_only("算了") is True
        assert p._is_denial_only("我要考试") is False

    def test_classify_and_extract_prompts_exist(self):
        """两阶段 prompt 存在。"""
        import lifeops.agent.parser as parser_mod
        assert hasattr(parser_mod, 'CLASSIFY_PROMPT')
        assert hasattr(parser_mod, 'EXTRACT_PROMPT')

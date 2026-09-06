"""Agent 单元测试。

覆盖：
- 输入解析（规则模式）
- 目标分解（规则模式）
- 解释生成
- Agent 完整五步流程
- 快照机制
"""
import sys
from pathlib import Path
from datetime import datetime, date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

from lifeops.models import (
    StudentProfile, Goal, Task, Commitment,
    DeadlineType, TaskStatus, EnergyLevel, CommitmentType,
)
from lifeops.agent import LifeAgent, InputParser, GoalDecomposer, PlanExplainer
from lifeops.agent.parser import ParseIntent
from lifeops.storage.database import Database


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def profile():
    return StudentProfile(
        name="测试学生",
        daily_high_energy_hours=4,
        min_progress_block_minutes=20,
        default_buffer_minutes=10,
    )


@pytest.fixture
def parser():
    return InputParser()  # Mock 模式（无 LLM）


@pytest.fixture
def decomposer():
    return GoalDecomposer()  # Mock 模式


@pytest.fixture
def explainer():
    return PlanExplainer()  # Mock 模式


@pytest.fixture
def agent(profile):
    return LifeAgent(profile=profile)


@pytest.fixture
def now():
    return datetime(2024, 3, 15, 8, 0, 0)  # 周五早上 8 点


# ==========================================
# 输入解析测试
# ==========================================

class TestInputParser:
    def test_progress_report_with_percent(self, parser):
        """解析进度汇报（百分比）。"""
        result = parser.parse("我高数复习完成了 30%")

        assert result.intent == ParseIntent.PROGRESS_REPORT
        assert result.confidence >= 0.5
        assert len(result.execution_records) == 1
        assert result.execution_records[0].progress_delta == pytest.approx(0.3, abs=0.01)

    def test_new_goal_detection(self, parser):
        """解析新目标。"""
        result = parser.parse("我要准备期末高数考试")

        assert result.intent == ParseIntent.NEW_GOAL
        assert result.goal_data is not None
        assert result.goal_data["title"]

    def test_new_goal_extracts_chinese_deadline(self, parser):
        """“9月19日我要提交报告”应提取出 ISO 截止日期。"""
        result = parser.parse(
            "9月19日我要参加一个比赛，我要提交报告",
            now=datetime(2026, 9, 3, 9, 0, 0),
        )

        assert result.intent == ParseIntent.NEW_GOAL
        assert result.goal_data["deadline"] == "2026-09-19"
        assert result.goal_data["title"] == "参加一个比赛，提交报告"

    def test_explicit_goal_not_downgraded_to_event(self):
        """即使模型误判成事件，明确目标句也应保留为 new_goal。"""
        class WrongLlm:
            mock_mode = False

            def chat_json(self, messages, **kwargs):
                return {
                    "intent": "new_event",
                    "confidence": 0.9,
                    "data": {
                        "events": [{
                            "title": "参加比赛",
                            "event_type": "activity",
                        }]
                    },
                }

        result = InputParser(llm=WrongLlm()).parse(
            "9月19日我要参加一个比赛，我要提交报告",
            now=datetime(2026, 9, 3, 9, 0, 0),
        )

        assert result.intent == ParseIntent.NEW_GOAL
        assert result.goal_data["deadline"] == "2026-09-19"

    def test_illness_detection(self, parser):
        """解析生病事件。"""
        result = parser.parse("我今天感冒发烧了，没法学习")

        assert result.intent == ParseIntent.NEW_EVENT
        assert len(result.life_events) == 1
        assert "illness" in result.life_events[0].event_type

    def test_status_query_detection(self, parser):
        """解析状态查询。"""
        result = parser.parse("我现在进度怎么样了？")

        assert result.intent == ParseIntent.STATUS_QUERY

    def test_unknown_message(self, parser):
        """无法识别的消息。"""
        result = parser.parse("今天天气真好")

        assert result.intent == ParseIntent.UNKNOWN
        assert result.confidence < 0.5


class FakeChatLLM:
    """最小 LLM 替身，只验证聊天入口不被规则固定文案短路。"""

    mock_mode = False

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return {"content": "这是模型回复的真实内容"}

    def chat_json(self, messages, **kwargs):
        raise RuntimeError("not used")


# ==========================================
# 目标分解测试
# ==========================================

class TestGoalDecomposer:
    def test_exam_goal_decomposition(self, decomposer):
        """考试类目标的分解。"""
        goal = Goal(
            id="g1",
            profile_id="p1",
            title="高数期末复习",
            weight=0.9,
        )
        deadline = datetime(2024, 6, 15, 9, 0)

        tasks = decomposer.decompose(goal, deadline=deadline)

        # 考试类目标至少有 5 个任务
        assert len(tasks) >= 5

        # 所有任务都关联到该目标
        for t in tasks:
            assert t.goal_id == "g1"

        # 最后一个任务应该是硬截止
        assert tasks[-1].deadline_type == DeadlineType.HARD

        # 任务之间应该有依赖关系
        deps_count = sum(1 for t in tasks if t.dependencies)
        assert deps_count >= len(tasks) - 2

    def test_project_goal_decomposition(self, decomposer):
        """项目类目标的分解。"""
        goal = Goal(
            id="g1",
            profile_id="p1",
            title="软件工程大作业",
            weight=0.8,
        )

        tasks = decomposer.decompose(goal)

        assert len(tasks) >= 3
        assert tasks[-1].deadline_type == DeadlineType.HARD

    def test_generic_goal_decomposition(self, decomposer):
        """通用目标的分解。"""
        goal = Goal(
            id="g1",
            profile_id="p1",
            title="学英语",
            weight=0.5,
        )

        tasks = decomposer.decompose(goal)

        assert len(tasks) >= 2
        for t in tasks:
            assert t.goal_id == "g1"


# ==========================================
# 解释生成测试
# ==========================================

class TestPlanExplainer:
    def test_initial_plan_explanation(self, explainer):
        """首次计划的解释。"""
        from lifeops.scheduler import ScheduleResult
        from lifeops.models import TimeSlot

        result = ScheduleResult(
            success=True,
            time_slots=[
                TimeSlot(
                    task_id="t1",
                    start_time="2024-03-15T09:00:00",
                    end_time="2024-03-15T10:00:00",
                ),
                TimeSlot(
                    task_id="t2",
                    start_time="2024-03-15T10:10:00",
                    end_time="2024-03-15T11:10:00",
                ),
            ],
            sacrifice_list=[],
            unscheduled_tasks=[],
        )

        changelog = explainer.explain(
            old_snapshot=None,
            new_result=result,
            reason="新目标：考试复习",
        )

        assert changelog.trigger_reason == "initial"
        assert "新" in changelog.detail or "安排" in changelog.detail

    def test_sacrifice_explanation(self, explainer):
        """有牺牲时的解释。"""
        from lifeops.scheduler import ScheduleResult
        from lifeops.models import TimeSlot, SacrificeItem

        result = ScheduleResult(
            success=True,
            time_slots=[],
            sacrifice_list=[
                SacrificeItem(
                    task_id="t1",
                    task_title="拓展阅读",
                    reason="时间不足",
                    priority_before=0.2,
                ),
                SacrificeItem(
                    task_id="t2",
                    task_title="额外练习",
                    reason="时间不足",
                    priority_before=0.3,
                ),
            ],
            unscheduled_tasks=["t1", "t2"],
        )

        changelog = explainer.explain(
            old_snapshot=None,
            new_result=result,
            reason="时间紧张",
        )

        assert "搁置" in changelog.detail or "牺牲" in changelog.detail
        assert len(changelog.affected_tasks) > 0


# ==========================================
# Agent 主流程测试
# ==========================================

class TestLifeAgent:
    def test_new_goal_creates_plan(self, agent, now):
        """新目标 → 创建计划。"""
        result = agent.run("我要准备高数考试", now=now)

        assert result.changed
        assert result.snapshot is not None
        assert len(agent._tasks) >= 5
        assert len(agent._goals) == 1
        assert result.intent == ParseIntent.NEW_GOAL

    def test_status_query_no_change(self, agent, now):
        """状态查询不触发重规划。"""
        # 先创建一个目标
        agent.run("我要准备高数考试", now=now)

        # 状态查询
        result = agent.run("看看我的进度", now=now + timedelta(hours=1))

        assert not result.changed
        assert result.intent == ParseIntent.STATUS_QUERY
        assert "任务" in result.reply or "进度" in result.reply

    def test_start_task_only_changes_status(self, agent, now):
        """开始任务：只把状态切成 in_progress，进度仍为 0。"""
        agent.run("我要准备高数考试", now=now)
        task = agent._tasks[0]
        assert task.status == TaskStatus.PENDING
        assert task.progress == 0.0

        assert agent.start_task(task.id) is True
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.progress == 0.0

    def test_start_task_ignores_completed(self, agent, now):
        """已完成的任务不允许再“开始”。"""
        agent.run("我要准备高数考试", now=now)
        task = agent._tasks[0]
        agent.complete_task(task.id)

        assert agent.start_task(task.id) is False
        assert task.status == TaskStatus.COMPLETED

    def test_start_task_unknown_id(self, agent):
        """任务 id 不存在时安全返回 False。"""
        assert agent.start_task("not_exist") is False

    def test_progress_update_triggers_replan(self, agent, now):
        """大的进度更新触发重规划。"""
        # 先创建目标
        agent.run("我要准备高数考试", now=now)
        old_snapshot = agent.current_snapshot

        # 进度更新（30% >= 20% 阈值）
        result = agent.run("我已经完成了 30%", now=now + timedelta(days=1))

        assert result.changed
        assert result.intent == ParseIntent.PROGRESS_REPORT
        # 有新快照
        assert len(agent._snapshots) >= 2

    def test_small_progress_no_replan(self, agent, now):
        """小进度更新不触发重规划。"""
        agent.run("我要准备高数考试", now=now)
        old_count = len(agent._snapshots)

        # 5% 的进度更新 < 20% 阈值
        result = agent.run("我完成了 5%", now=now + timedelta(hours=2))

        assert not result.changed
        assert len(agent._snapshots) == old_count

    def test_snapshot_chain_maintained(self, agent, now):
        """快照链正确维护。"""
        agent.run("我要准备高数考试", now=now)
        assert len(agent._snapshots) == 1

        agent.run("我生病了", now=now + timedelta(days=1))
        assert len(agent._snapshots) == 2

        agent.run("我完成了 30%", now=now + timedelta(days=2))
        assert len(agent._snapshots) == 3

        # 快照按时间顺序
        for i in range(1, len(agent._snapshots)):
            prev = datetime.fromisoformat(agent._snapshots[i-1].created_at)
            curr = datetime.fromisoformat(agent._snapshots[i].created_at)
            assert curr > prev

    def test_illness_event_triggers_replan(self, agent, now):
        """生病事件触发重规划。"""
        agent.run("我要准备高数考试", now=now)
        old_count = len(agent._snapshots)

        result = agent.run("我今天感冒发烧了", now=now + timedelta(hours=4))

        assert result.changed
        assert result.intent == ParseIntent.NEW_EVENT
        assert len(agent._snapshots) == old_count + 1

    def test_task_progress_updated(self, agent, now):
        """进度汇报后任务进度确实更新了。"""
        agent.run("我要准备高数考试", now=now)

        # 找一个进行中的任务
        pending_tasks = [t for t in agent._tasks if t.status == TaskStatus.PENDING]
        assert len(pending_tasks) > 0
        first_task = pending_tasks[0]
        initial_progress = first_task.progress

        # 汇报进度
        agent.run(f"我{first_task.title[:4]}完成了 30%", now=now + timedelta(hours=2))

        # 任务进度应该更新了
        assert first_task.progress > initial_progress
        assert first_task.status in (TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED)

    def test_progress_report_is_current_unless_explicit_delta(self, agent, now):
        """“完成 30%”按当前总进度理解，只有“又/再”才累加。"""
        agent.run("我要准备高数考试", now=now)
        task = agent._tasks[0]

        agent.run("我已经完成了 30%", now=now + timedelta(hours=1))
        assert task.progress == pytest.approx(0.3)

        agent.run("现在完成了 60%", now=now + timedelta(hours=2))
        assert task.progress == pytest.approx(0.6)

        agent.run("又完成了 10%", now=now + timedelta(hours=3))
        assert task.progress == pytest.approx(0.7)

    def test_structured_goal_keeps_deadline_and_weight(self, agent, now):
        """结构化建目标应保留 deadline/weight，不经过规则解析丢失。"""
        deadline = (now + timedelta(days=14)).isoformat()
        result = agent.create_goal_structured(
            title="高数期末复习",
            description="系统复习",
            weight=0.9,
            deadline=deadline,
            now=now,
        )

        assert result.changed
        assert agent._goals[0].deadline == deadline
        assert agent._goals[0].weight == 0.9
        hard_tasks = [
            t for t in agent._tasks
            if t.deadline_type == DeadlineType.HARD
        ]
        assert hard_tasks
        assert all(t.deadline == deadline for t in hard_tasks)

    def test_unknown_question_uses_llm_instead_of_fixed_reply(self, agent, now):
        """普通闲聊不应再返回固定的“不需要调整计划”。"""
        fake = FakeChatLLM()
        agent.llm = fake
        agent.parser.llm = fake

        result = agent.run("你好，介绍一下你自己", now=now)

        assert result.reply == "这是模型回复的真实内容"
        assert fake.calls >= 1

    def test_remove_task_syncs_to_db(self, tmp_path, now):
        """手动删除任务时，内存与 SQLite 都要清掉。"""
        db = Database(str(tmp_path / "lifeos.db"))
        db.init_schema()
        profile = db.get_default_profile()

        agent = LifeAgent(profile=profile, db=db)
        agent.run("我要准备高数考试", now=now)
        target = agent._tasks[0]

        assert agent.remove_task(target.id)
        assert all(t.id != target.id for t in agent._tasks)
        assert all(
            t.id != target.id
            for t in db.list_tasks(profile_id=profile.id or "default")
        )

        # 重新加载后任务不会复活
        reloaded = LifeAgent(profile=db.get_default_profile(), db=db)
        assert all(t.id != target.id for t in reloaded._tasks)

    def test_plan_consistency_self_heals_after_orphan_delete(self, tmp_path, now):
        """删除任务但未触发重排时，ensure_plan_consistent 应校正快照。

        模拟历史遗留场景：任务已被删除、最新快照仍引用它 → 校正性重排应
        生成不含该任务的新快照，且重复调用不产生多余快照。
        """
        db = Database(str(tmp_path / "lifeos.db"))
        db.init_schema()
        profile = db.get_default_profile()
        agent = LifeAgent(profile=profile, db=db)
        agent.run("我要准备高数考试", now=now)

        target = agent._tasks[0]
        assert agent.current_snapshot is not None
        assert any(
            ts.task_id == target.id for ts in (agent.current_snapshot.time_slots or [])
        )

        # 模拟“删了任务但没重排”（旧路径/外部改动留下的不一致状态）
        assert agent.remove_task(target.id)

        assert not agent.plan_is_consistent()
        assert agent.ensure_plan_consistent(now=now)
        healed = agent.current_snapshot
        assert healed is not None
        assert not any(
            ts.task_id == target.id for ts in (healed.time_slots or [])
        )
        # 再次调用应为空操作，避免每次进入页面都重复生成快照
        assert agent.ensure_plan_consistent(now=now) is False

        # 重启后状态一致，不会再次校正
        reloaded = LifeAgent(profile=db.get_default_profile(), db=db)
        assert reloaded.plan_is_consistent()

    def test_commitment_update_and_remove(self, agent):
        """课表管理所需的承诺更新/删除接口。"""
        commitment = Commitment(
            profile_id="p1",
            title="高数",
            type=CommitmentType.CLASS,
            start_time="2026-09-07T08:00:00",
            end_time="2026-09-07T09:40:00",
            recurrence="weekly",
            description="A101",
        )
        agent.add_commitment(commitment)

        updated = agent.update_commitment(
            commitment.id,
            title="高等数学",
            start_time="2026-09-07T10:00:00",
            end_time="2026-09-07T11:40:00",
            description="A102",
        )
        assert updated
        stored = agent._commitments[0]
        assert stored.title == "高等数学"
        assert stored.start_time == "2026-09-07T10:00:00"
        assert stored.description == "A102"

        removed = agent.remove_commitment(commitment.id)
        assert removed
        assert agent._commitments == []


class TestPersistence:
    """SQLite 状态跨 Agent 实例恢复。"""

    def test_state_survives_new_agent_instance(self, tmp_path):
        db = Database(str(tmp_path / "lifeos.db"))
        db.init_schema()
        profile = db.get_default_profile()

        agent = LifeAgent(profile=profile, db=db)
        agent.run(
            "我要准备高数考试",
            now=datetime(2024, 3, 15, 8, 0, 0),
        )
        assert len(agent._goals) == 1
        assert len(agent._tasks) >= 5

        # 模拟应用重启：新的 Agent 应直接从数据库恢复
        reloaded = LifeAgent(
            profile=db.get_default_profile(),
            db=db,
        )
        assert len(reloaded._goals) == 1
        assert len(reloaded._tasks) == len(agent._tasks)
        assert len(reloaded._snapshots) == 1

        # 绝对进度也应持久化
        target_task = reloaded._tasks[0]
        reloaded.report_absolute_progress(
            task_id=target_task.id,
            progress=0.3,
            actual_minutes=45,
            note="持久化测试",
        )

        reloaded_again = LifeAgent(
            profile=db.get_default_profile(),
            db=db,
        )
        restored_task = reloaded_again.get_task(target_task.id)
        assert restored_task is not None
        assert restored_task.progress == pytest.approx(0.3)
        assert restored_task.actual_minutes >= 45
        assert len(db.list_snapshots(profile.id)) >= 2

        # 课表删除也要从 SQLite 移除，重启后不会复活
        course = Commitment(
            profile_id=profile.id,
            title="高数",
            type=CommitmentType.CLASS,
            start_time="2026-09-07T08:00:00",
            end_time="2026-09-07T09:40:00",
            recurrence="weekly",
        )
        reloaded_again.add_commitment(course)
        reloaded_again.persist_state()
        assert db.list_commitments(profile.id)
        reloaded_again.remove_commitment(course.id)
        assert not db.list_commitments(profile.id)


class FakeIntentLlm:
    """返回固定意图的 LLM 替身，用于验证确认流程。"""

    mock_mode = False

    def __init__(self, intent: str, data: dict):
        self.intent = intent
        self.data = data
        self.calls = 0

    def chat_json(self, messages, **kwargs):
        self.calls += 1
        return {"intent": self.intent, "confidence": 0.95, "data": self.data}

    def chat(self, messages, **kwargs):
        self.calls += 1
        return {"content": "已按你的要求处理。"}


def _attach_fake_llm(agent, intent: str, data: dict) -> FakeIntentLlm:
    fake = FakeIntentLlm(intent, data)
    agent.llm = fake
    agent.parser.llm = fake
    return fake


class TestCancelPlan:
    """通过对话取消已有计划（删除类操作必须先确认）。"""

    def _build_agent_with_goal(self, profile, now):
        agent = LifeAgent(profile=profile)
        agent.run("我要准备高数考试", now=now)
        return agent

    def test_cancel_goal_phrase_parsed(self, parser):
        """“取消高数复习计划”应识别为取消意图，而不是状态查询。"""
        result = parser.parse("取消高数复习计划", now=datetime(2026, 9, 4, 9, 0))

        assert result.intent == ParseIntent.CANCEL_PLAN
        assert result.cancel_data is not None
        assert result.cancel_data["target_type"] == "goal"
        assert result.cancel_data["keyword"] == "高数复习"

    def test_cancel_task_phrase_parsed(self, parser):
        """取消任务类语句解析。"""
        result = parser.parse("删除英语背单词任务")

        assert result.intent == ParseIntent.CANCEL_PLAN
        assert result.cancel_data["target_type"] == "task"
        assert result.cancel_data["keyword"] == "英语背单词"

    def test_cancel_all_phrase_parsed(self, parser):
        """清空所有计划。"""
        result = parser.parse("清空所有计划")

        assert result.intent == ParseIntent.CANCEL_PLAN
        assert result.cancel_data["target_type"] == "all"

    def test_cancel_requires_confirmation(self, profile, now):
        """取消是破坏性操作，先反问确认，不直接删除。"""
        agent = self._build_agent_with_goal(profile, now)
        goals_before = len(agent._goals)

        result = agent.run("取消备考计划", now=now + timedelta(hours=1))

        assert result.intent == ParseIntent.CANCEL_PLAN
        assert not result.changed
        assert len(agent._goals) == goals_before
        assert agent.pending_action is not None

    def test_cancel_executes_after_confirm(self, profile, now):
        """确认后目标与其任务一起被删除，并重新排期。"""
        agent = self._build_agent_with_goal(profile, now)
        tasks_before = len(agent._tasks)

        agent.run("取消备考计划", now=now + timedelta(hours=1))
        result = agent.run("可以", now=now + timedelta(hours=1, seconds=10))

        assert result.changed
        assert len(agent._goals) == 0
        assert len(agent._tasks) == 0
        assert tasks_before > 0
        assert "已取消" in result.reply

    def test_cancel_aborted_by_negative(self, profile, now):
        """回复“不用了”应保留原计划。"""
        agent = self._build_agent_with_goal(profile, now)

        agent.run("取消备考计划", now=now + timedelta(hours=1))
        result = agent.run("不用了", now=now + timedelta(hours=1, seconds=10))

        assert not result.changed
        assert len(agent._goals) == 1
        assert len(agent._tasks) >= 5
        assert agent.pending_action is None

    def test_cancel_unknown_target_hints_current_plan(self, profile, now):
        """取消不存在的计划时第一轮直接提示，不进入确认、不误删任何内容。"""
        agent = self._build_agent_with_goal(profile, now)

        result = agent.run("取消物理实验计划", now=now + timedelta(hours=1))

        assert result.intent == ParseIntent.CANCEL_PLAN
        assert not result.changed
        assert len(agent._goals) == 1
        assert "没找到" in result.reply
        assert agent.pending_action is None

    def test_cancel_task_only(self, profile, now):
        """按任务名取消单个任务。"""
        agent = self._build_agent_with_goal(profile, now)
        target = agent._tasks[0]

        agent.run(f"删除{target.title}任务", now=now + timedelta(hours=1))
        result = agent.run("可以", now=now + timedelta(hours=1, seconds=10))

        assert result.changed
        assert all(t.id != target.id for t in agent._tasks)
        assert len(agent._goals) == 1

    def test_cancel_all_clears_goals_and_tasks(self, profile, now):
        """清空所有计划。"""
        agent = self._build_agent_with_goal(profile, now)

        agent.run("清空所有计划", now=now + timedelta(hours=1))
        result = agent.run("可以", now=now + timedelta(hours=1, seconds=10))

        assert result.changed
        assert not agent._goals
        assert not agent._tasks

    def test_negative_activity_phrase_parsed_as_cancel(self, parser):
        """"八点不去打篮球了"是取消意图，绝不能当成新增事件。"""
        result = parser.parse(
            "八点不去打篮球了", now=datetime(2026, 9, 4, 20, 0)
        )

        assert result.intent == ParseIntent.CANCEL_PLAN
        assert result.cancel_data is not None
        assert "打篮球" in result.cancel_data["keyword"]
        assert result.intent != ParseIntent.NEW_EVENT

    def test_negative_activity_without_target_replies_hint(self, profile, now):
        """日程里没有"打篮球"时，提示没找到而不是新增事件。"""
        agent = LifeAgent(profile=profile)
        agent.run("我要准备高数考试", now=now)

        result = agent.run("八点不去打篮球了", now=now + timedelta(hours=2))

        assert result.intent == ParseIntent.CANCEL_PLAN
        assert not result.changed
        assert "没找到" in result.reply
        assert agent.pending_action is None


class TestConfirmationFlow:
    """"先确认再执行"对话流程（有 LLM 的真实对话场景）。"""

    EVENT_DATA = {
        "events": [{
            "title": "学习",
            "event_type": "other",
            "event_time": "2026-09-04 20:00",
            "end_time": None,
        }]
    }
    GOAL_DATA = {
        "title": "高数复习",
        "deadline": None,
        "description": "",
    }

    def test_new_event_confirms_then_executes_on_affirmative(self, profile, now):
        """事件先反问确认；回复「可以」后执行。"""
        agent = LifeAgent(profile=profile)
        _attach_fake_llm(agent, "new_event", self.EVENT_DATA)

        first = agent.run("八点去学习", now=now)
        assert not first.changed
        assert first.intent == ParseIntent.NEW_EVENT
        assert "学习" in first.reply
        assert "可以" in first.reply or "好的" in first.reply
        # 待确认状态已建立且持久化了原始消息
        assert agent.pending_action is not None
        assert agent.pending_action.message == "八点去学习"

        second = agent.run("可以", now=now + timedelta(seconds=10))
        assert second.changed
        assert second.intent == ParseIntent.NEW_EVENT
        assert agent.pending_action is None
        # 确认后执行并生成了新快照
        assert len(agent._snapshots) >= 1

    def test_affirmative_variants_recognized(self, profile, now):
        """常见肯定变体都能触发确认。"""
        for reply in ("OK", "ok", "好的", "行", "没问题", "可以啊", "嗯，可以", "就这么办"):
            agent = LifeAgent(profile=profile)
            _attach_fake_llm(agent, "new_event", self.EVENT_DATA)
            agent.run("八点去学习", now=now)
            result = agent.run(reply, now=now + timedelta(seconds=10))
            assert result.changed, f"肯定回复 {reply!r} 未被识别"
            assert agent.pending_action is None

    def test_negative_cancels_pending(self, profile, now):
        """否定回复应取消待确认，不执行任何变更。"""
        agent = LifeAgent(profile=profile)
        _attach_fake_llm(agent, "new_event", self.EVENT_DATA)
        agent.run("八点去学习", now=now)

        result = agent.run("不用了", now=now + timedelta(seconds=10))
        assert not result.changed
        assert "不安排" in result.reply
        assert agent.pending_action is None
        assert not agent._snapshots

    def test_confirm_pending_api(self, profile, now):
        """UI 按钮走 confirm_pending 显式接口。"""
        agent = LifeAgent(profile=profile)
        _attach_fake_llm(agent, "new_goal", self.GOAL_DATA)
        agent.run("帮我安排一下高数复习", now=now)

        # 取消
        result = agent.confirm_pending(confirmed=False, now=now)
        assert not result.changed
        assert len(agent._goals) == 0

        # 再次建立待确认后确认
        agent.run("帮我安排一下高数复习", now=now)
        result = agent.confirm_pending(confirmed=True, now=now)
        assert result.changed
        assert len(agent._goals) == 1
        assert agent.pending_action is None

    def test_pending_persists_across_agent_restart(self, tmp_path, now):
        """待确认状态持久化：进程重启后仍可确认执行。"""
        db = Database(str(tmp_path / "lifeos.db"))
        db.init_schema()
        profile = db.get_default_profile()

        agent1 = LifeAgent(profile=profile, db=db)
        _attach_fake_llm(agent1, "new_event", self.EVENT_DATA)
        agent1.run("八点去学习", now=now)
        assert agent1.pending_action is not None

        # 模拟重启：新的 Agent 应恢复 pending
        agent2 = LifeAgent(profile=db.get_default_profile(), db=db)
        _attach_fake_llm(agent2, "new_event", self.EVENT_DATA)
        assert agent2.pending_action is not None
        assert agent2.pending_action.message == "八点去学习"

        result = agent2.run("OK", now=now + timedelta(seconds=10))
        assert result.changed
        assert agent2.pending_action is None

        # 重启后 pending 不再残留
        agent3 = LifeAgent(profile=db.get_default_profile(), db=db)
        assert agent3.pending_action is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

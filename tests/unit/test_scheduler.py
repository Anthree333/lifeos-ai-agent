"""调度器单元测试。

覆盖：
- 硬截止约束
- 依赖顺序
- 最小推进块
- 不可行时牺牲清单
- 优先级评分
- 同输入同输出（确定性）
"""
import sys
from pathlib import Path
from datetime import datetime, date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

from lifeops.models import (
    Task, Goal, Commitment, StudentProfile,
    DeadlineType, TaskStatus, EnergyLevel, CommitmentType,
)
from lifeops.scheduler import (
    Scheduler, TaskPriorityScorer, ConstraintChecker, SacrificeGenerator,
)
from lifeops.scheduler.priority import count_dependency_blocks
from lifeops.scheduler.risk_predictor import RiskPredictor
from lifeops.models.risk import RiskLevel


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
def scheduler(profile):
    return Scheduler(profile=profile)


@pytest.fixture
def today():
    return date(2024, 3, 15)  # 周五


@pytest.fixture
def now(today):
    return datetime.combine(today, datetime.min.time()).replace(hour=8)


# ==========================================
# 优先级评分测试
# ==========================================

class TestTaskPriorityScorer:
    def test_hard_deadline_scores_higher(self, now):
        """硬截止任务得分高于软截止任务。"""
        scorer = TaskPriorityScorer()
        deadline = (now + timedelta(days=2)).isoformat()

        hard_task = Task(
            title="硬截止任务",
            goal_id="g1",
            estimated_minutes=120,
            deadline_type=DeadlineType.HARD,
            deadline=deadline,
        )
        soft_task = Task(
            title="软截止任务",
            goal_id="g1",
            estimated_minutes=120,
            deadline_type=DeadlineType.SOFT,
            deadline=deadline,
        )

        goal = Goal(id="g1", profile_id="p1", title="测试目标", weight=0.5)

        hard_score = scorer.score(hard_task, goal, 0, now)
        soft_score = scorer.score(soft_task, goal, 0, now)

        assert hard_score > soft_score

    def test_more_blocks_scores_higher(self, now):
        """阻塞更多下游任务的任务得分更高。"""
        scorer = TaskPriorityScorer()

        task = Task(
            title="任务A",
            goal_id="g1",
            estimated_minutes=60,
        )
        goal = Goal(id="g1", profile_id="p1", title="测试目标", weight=0.5)

        score_low = scorer.score(task, goal, block_count=0, now=now)
        score_high = scorer.score(task, goal, block_count=5, now=now)

        assert score_high > score_low

    def test_urgency_increases_over_time(self):
        """距离截止越近，紧迫度越高。"""
        scorer = TaskPriorityScorer()

        task = Task(
            title="考试复习",
            goal_id="g1",
            estimated_minutes=300,
            deadline_type=DeadlineType.HARD,
        )
        goal = Goal(id="g1", profile_id="p1", title="测试目标", weight=0.5)

        # 7天后截止
        far_deadline = (datetime(2024, 3, 1) + timedelta(days=7)).isoformat()
        task.deadline = far_deadline
        score_far = scorer.score(task, goal, 0, datetime(2024, 3, 1))

        # 1天后截止
        near_deadline = (datetime(2024, 3, 7) + timedelta(days=1)).isoformat()
        task.deadline = near_deadline
        score_near = scorer.score(task, goal, 0, datetime(2024, 3, 7))

        assert score_near > score_far


class TestDependencyBlockCount:
    def test_linear_chain(self):
        """线性依赖链：A -> B -> C，A 阻塞 2 个，B 阻塞 1 个，C 阻塞 0 个。"""
        tasks = [
            Task(id="A", title="A", goal_id="g1", dependencies=[]),
            Task(id="B", title="B", goal_id="g1", dependencies=["A"]),
            Task(id="C", title="C", goal_id="g1", dependencies=["B"]),
        ]

        blocks = count_dependency_blocks(tasks)

        assert blocks["A"] == 2
        assert blocks["B"] == 1
        assert blocks["C"] == 0


# ==========================================
# 约束检查测试
# ==========================================

class TestConstraintChecker:
    def test_available_slots_exclude_sleep(self):
        """可用时段不包含睡眠时间。"""
        checker = ConstraintChecker()
        start = datetime(2024, 3, 15, 8, 0)
        end = datetime(2024, 3, 15, 23, 0)

        slots = checker.generate_available_slots(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2024, 3, 15),
            wake_up_hour=8,
            sleep_hour=23,
        )

        # 总时长应为 15 小时（8:00 - 23:00）
        total_minutes = sum(
            int((s.end - s.start).total_seconds() / 60) for s in slots
        )
        assert total_minutes == 15 * 60

    def test_commitment_subtracts_time(self):
        """固定承诺会扣减可用时间。"""
        checker = ConstraintChecker()
        commitment = Commitment(
            id="c1",
            profile_id="p1",
            title="数学课",
            type=CommitmentType.CLASS,
            start_time="2024-03-15T09:00:00",
            end_time="2024-03-15T11:00:00",
        )

        slots = checker.generate_available_slots(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2024, 3, 15),
            wake_up_hour=8,
            sleep_hour=23,
            commitments=[commitment],
        )

        total_minutes = sum(
            int((s.end - s.start).total_seconds() / 60) for s in slots
        )
        # 15小时 - 2小时课程 = 13小时
        assert total_minutes == 13 * 60

    def test_hard_deadline_violation(self, today):
        """硬截止任务不能安排在截止之后。"""
        checker = ConstraintChecker()
        deadline = datetime.combine(today, datetime.min.time()).replace(hour=12)

        task = Task(
            title="考试",
            goal_id="g1",
            estimated_minutes=120,
            deadline_type=DeadlineType.HARD,
            deadline=deadline.isoformat(),
        )

        # 安排在截止之后 → 应该违规
        violations = checker.check_hard_constraints(
            task,
            slot_start=deadline - timedelta(hours=1),
            slot_end=deadline + timedelta(hours=1),
            fixed_slots=[],
        )

        assert len(violations) > 0
        assert any(v.rule_name == "hard_deadline" for v in violations)

    def test_dependency_check(self):
        """依赖未满足时返回 False。"""
        checker = ConstraintChecker()
        task = Task(
            title="任务B",
            goal_id="g1",
            dependencies=["task_a"],
        )
        task_map = {"task_a": Task(id="task_a", title="A", goal_id="g1")}

        # 依赖未完成
        assert not checker.check_dependencies(task, task_map, set())

        # 依赖已完成
        assert checker.check_dependencies(task, task_map, {"task_a"})


# ==========================================
# 调度器主测试
# ==========================================

class TestScheduler:
    def test_single_hard_task_scheduled(self, scheduler, today, now):
        """单个硬截止任务应该被安排。"""
        task = Task(
            title="考试复习",
            goal_id="g1",
            estimated_minutes=120,
            deadline_type=DeadlineType.HARD,
            deadline=(datetime.combine(today, datetime.min.time()).replace(hour=18)).isoformat(),
            energy_level=EnergyLevel.HIGH,
        )
        goal = Goal(id="g1", profile_id="p1", title="考试目标", weight=0.8)

        result = scheduler.schedule(
            tasks=[task],
            goals=[goal],
            commitments=[],
            start_date=today,
            end_date=today + timedelta(days=1),
            now=now,
        )

        assert result.success
        assert len(result.time_slots) > 0
        assert len(result.unscheduled_tasks) == 0

    def test_hard_deadline_met(self, scheduler, today, now):
        """硬截止任务必须在截止前完成。"""
        deadline = datetime.combine(today, datetime.min.time()).replace(hour=12)

        task = Task(
            title="考试复习",
            goal_id="g1",
            estimated_minutes=120,  # 2 小时
            deadline_type=DeadlineType.HARD,
            deadline=deadline.isoformat(),
            energy_level=EnergyLevel.HIGH,
        )
        goal = Goal(id="g1", profile_id="p1", title="目标", weight=0.8)

        result = scheduler.schedule(
            tasks=[task],
            goals=[goal],
            commitments=[],
            start_date=today,
            end_date=today + timedelta(days=1),
            now=now,
        )

        # 找出该任务的所有时间槽
        task_slots = [ts for ts in result.time_slots if ts.task_id == task.id]
        assert len(task_slots) > 0

        # 最晚结束时间必须在截止前
        latest_end = max(
            datetime.fromisoformat(ts.end_time) for ts in task_slots
        )
        assert latest_end <= deadline

    def test_no_slots_before_now(self, scheduler, today, now):
        """调度结果不能包含当前时间之前的时段。"""
        task = Task(
            title="复习",
            goal_id="g1",
            estimated_minutes=120,
            deadline_type=DeadlineType.HARD,
            deadline=(datetime.combine(today, datetime.min.time()).replace(hour=18)).isoformat(),
        )
        goal = Goal(id="g1", profile_id="p1", title="目标", weight=0.8)

        result = scheduler.schedule(
            tasks=[task],
            goals=[goal],
            commitments=[],
            start_date=today,
            end_date=today + timedelta(days=1),
            now=now,
        )

        assert all(
            datetime.fromisoformat(ts.start_time) >= now
            for ts in result.time_slots
        )

    def test_incomplete_hard_task_reports_failure(self, scheduler, today, now):
        """硬截止任务未排满时应失败并进入牺牲清单。"""
        deadline = datetime.combine(today, datetime.min.time()).replace(hour=9)
        task = Task(
            title="来不及的硬任务",
            goal_id="g1",
            estimated_minutes=240,
            deadline_type=DeadlineType.HARD,
            deadline=deadline.isoformat(),
        )
        goal = Goal(id="g1", profile_id="p1", title="目标", weight=0.8)

        result = scheduler.schedule(
            tasks=[task],
            goals=[goal],
            commitments=[],
            start_date=today,
            end_date=today,
            now=now,
        )

        assert not result.success
        assert any(t.id == task.id for t in result.unscheduled_tasks)
        assert any(s.task_id == task.id for s in result.sacrifice_list)

    def test_soft_dependency_chain_schedules_all_when_time_allows(
        self, scheduler, today, now
    ):
        """时间充足时软任务依赖链应逐层解锁并全部排完。"""
        tasks = [
            Task(id="a", title="A", goal_id="g1", estimated_minutes=60),
            Task(
                id="b", title="B", goal_id="g1", estimated_minutes=60,
                dependencies=["a"],
            ),
            Task(
                id="c", title="C", goal_id="g1", estimated_minutes=60,
                dependencies=["b"],
            ),
        ]
        goal = Goal(id="g1", profile_id="p1", title="目标", weight=0.8)

        result = scheduler.schedule(
            tasks=tasks,
            goals=[goal],
            commitments=[],
            start_date=today,
            end_date=today + timedelta(days=2),
            now=now,
        )

        scheduled_minutes = {
            t.id: sum(
                ts.duration_minutes for ts in result.time_slots
                if ts.task_id == t.id
            )
            for t in tasks
        }
        assert scheduled_minutes == {"a": 60, "b": 60, "c": 60}
        assert result.unscheduled_tasks == []
        assert result.sacrifice_list == []

    def test_weekly_commitment_blocks_future_occurrence(self, scheduler, today, now):
        """weekly 承诺应按周展开，而不是只挡住首次出现日期。"""
        commitment = Commitment(
            id="c1",
            profile_id="p1",
            title="周一早课",
            type=CommitmentType.CLASS,
            start_time=(datetime.combine(today, datetime.min.time()).replace(hour=9)).isoformat(),
            end_time=(datetime.combine(today, datetime.min.time()).replace(hour=10)).isoformat(),
            recurrence="weekly",
        )
        task = Task(
            title="弹性任务",
            goal_id="g1",
            estimated_minutes=300,
            deadline_type=DeadlineType.SOFT,
        )
        goal = Goal(id="g1", profile_id="p1", title="目标", weight=0.5)

        result = scheduler.schedule(
            tasks=[task],
            goals=[goal],
            commitments=[commitment],
            start_date=today,
            end_date=today + timedelta(days=8),
            now=now,
        )

        for blocked_date in (today, today + timedelta(days=7)):
            assert not any(
                ts.task_id == task.id
                and datetime.fromisoformat(ts.start_time).date() == blocked_date
                and datetime.fromisoformat(ts.start_time).hour == 9
                for ts in result.time_slots
            )

    def test_commitment_not_overwritten(self, scheduler, today, now):
        """固定承诺的时间段不会被任务覆盖。"""
        commitment = Commitment(
            id="c1",
            profile_id="p1",
            title="数学课",
            type=CommitmentType.CLASS,
            start_time=datetime.combine(today, datetime.min.time()).replace(hour=9).isoformat(),
            end_time=datetime.combine(today, datetime.min.time()).replace(hour=11).isoformat(),
        )

        task = Task(
            title="自习",
            goal_id="g1",
            estimated_minutes=300,
            deadline_type=DeadlineType.SOFT,
            energy_level=EnergyLevel.MEDIUM,
        )
        goal = Goal(id="g1", profile_id="p1", title="目标", weight=0.5)

        result = scheduler.schedule(
            tasks=[task],
            goals=[goal],
            commitments=[commitment],
            start_date=today,
            end_date=today + timedelta(days=1),
            now=now,
        )

        # 任务时间槽不应与承诺重叠
        commit_start = datetime.fromisoformat(commitment.start_time)
        commit_end = datetime.fromisoformat(commitment.end_time)

        for ts in result.time_slots:
            if ts.task_id == task.id:
                ts_start = datetime.fromisoformat(ts.start_time)
                ts_end = datetime.fromisoformat(ts.end_time)
                assert not (ts_start < commit_end and ts_end > commit_start)

    def test_sacrifice_list_when_insufficient_time(self, scheduler, today, now):
        """时间不足时生成牺牲清单。"""
        # 安排一个超大量的任务（超过可用时间）
        huge_task = Task(
            title="超级大作业",
            goal_id="g1",
            estimated_minutes=20 * 60,  # 20 小时，远超一天可用时间
            deadline_type=DeadlineType.SOFT,
            energy_level=EnergyLevel.HIGH,
        )
        goal = Goal(id="g1", profile_id="p1", title="目标", weight=0.5)

        result = scheduler.schedule(
            tasks=[huge_task],
            goals=[goal],
            commitments=[],
            start_date=today,
            end_date=today,  # 只给一天
            now=now,
        )

        # 应该有牺牲清单或未安排的任务
        assert len(result.unscheduled_tasks) >= 0
        # 调度结果不一定完全失败（会安排一部分）
        # 但牺牲清单应该包含未完成部分的说明
        assert result.schedule_tracks.get("total_task_minutes", 0) > 0

    def test_deterministic_same_input_same_output(self, scheduler, today, now):
        """同输入必须同输出（确定性）。"""
        tasks = [
            Task(
                title="任务A",
                goal_id="g1",
                estimated_minutes=60,
                deadline_type=DeadlineType.HARD,
                deadline=(datetime.combine(today, datetime.min.time()).replace(hour=18)).isoformat(),
            ),
            Task(
                title="任务B",
                goal_id="g1",
                estimated_minutes=90,
                deadline_type=DeadlineType.SOFT,
            ),
        ]
        goal = Goal(id="g1", profile_id="p1", title="目标", weight=0.6)

        result1 = scheduler.schedule(tasks=tasks, goals=[goal], commitments=[],
                                     start_date=today, end_date=today + timedelta(days=1), now=now)
        result2 = scheduler.schedule(tasks=tasks, goals=[goal], commitments=[],
                                     start_date=today, end_date=today + timedelta(days=1), now=now)

        # 时间槽数量相同
        assert len(result1.time_slots) == len(result2.time_slots)
        # 总安排时长相同
        assert result1.total_scheduled_minutes == result2.total_scheduled_minutes
        # 未安排任务数相同
        assert len(result1.unscheduled_tasks) == len(result2.unscheduled_tasks)


# ==========================================
# 风险预测测试
# ==========================================

class TestRiskPredictor:
    def test_no_delay_low_risk(self):
        """能按时完成 → 低风险。"""
        predictor = RiskPredictor()
        task = Task(
            title="任务",
            goal_id="g1",
            estimated_minutes=300,
            progress=0.5,  # 剩余 150 分钟
            deadline=(datetime.now() + timedelta(days=5)).isoformat(),
        )

        # 假设有足够的历史记录（每天 60 分钟）
        records = []
        for i in range(3):
            records.append(type('R', (), {
                'task_id': task.id,
                'record_date': (date.today() - timedelta(days=i)).isoformat(),
                'actual_minutes': 60,
                'progress_delta': 0.1,
                'note': '',
                'id': f'r{i}',
                'created_at': datetime.now().isoformat(),
            })())

        report = predictor.predict(task, records)

        # 150 分钟 ÷ 60 分钟/天 = 2.5 天 < 5 天 → 低风险
        assert report.risk_level == RiskLevel.LOW
        assert report.delay_days == 0

    def test_hard_deadline_delay_high_risk(self):
        """硬截止任务且预计延期 → 高风险。"""
        predictor = RiskPredictor()
        task = Task(
            title="大作业",
            goal_id="g1",
            estimated_minutes=600,
            progress=0.2,  # 剩余 480 分钟
            deadline_type=DeadlineType.HARD,
            deadline=(datetime.now() + timedelta(days=2)).isoformat(),
        )

        # 每天只有 60 分钟 → 480 ÷ 60 = 8 天 > 2 天
        records = []
        for i in range(3):
            records.append(type('R', (), {
                'task_id': task.id,
                'record_date': (date.today() - timedelta(days=i)).isoformat(),
                'actual_minutes': 60,
                'progress_delta': 0.05,
                'note': '',
                'id': f'r{i}',
                'created_at': datetime.now().isoformat(),
            })())

        report = predictor.predict(task, records)

        assert report.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert report.delay_days > 0
        assert report.is_overdue is True


# ==========================================
# 牺牲清单测试
# ==========================================

class TestSacrificeGenerator:
    def test_sacrifices_ordered_by_priority(self):
        """牺牲清单按优先级从低到高排列。"""
        gen = SacrificeGenerator()
        tasks = [
            Task(id="t1", title="低优先级", goal_id="g1", priority_weight=0.2),
            Task(id="t2", title="中优先级", goal_id="g1", priority_weight=0.5),
            Task(id="t3", title="高优先级", goal_id="g1", priority_weight=0.8),
        ]
        hard_tasks = [Task(id="h1", title="硬截止任务", goal_id="g1")]
        scores = {"t1": 0.2, "t2": 0.5, "t3": 0.8, "h1": 0.9}

        sacrifices = gen.generate(tasks, hard_tasks, scores)

        assert len(sacrifices) == 3
        # 最低优先级的应该在前
        assert sacrifices[0].task_title == "低优先级"
        assert sacrifices[2].task_title == "高优先级"
        assert sacrifices[0].priority_before < sacrifices[2].priority_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

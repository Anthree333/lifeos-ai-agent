"""目标分解器。

将用户的模糊目标（如"准备高数考试"）分解为具体任务列表。
LLM 负责分解，调度器负责排期，职责分离。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..llm.chat_model import ChatModel, ChatMessage, ChatRole
from ..models import Task, Goal, DeadlineType, EnergyLevel


SYSTEM_PROMPT = """你是 LifeOS 的目标分解器。你的任务是将用户的模糊目标分解为具体、可执行、可衡量的任务列表。

分解原则：
1. 每个任务时长在 30-120 分钟之间，避免过长或过短
2. 任务之间可以有依赖关系（先 A 后 B）
3. 区分硬截止（考试、提交日期）和软截止（自我设定）
4. 考虑前置知识/准备工作
5. 总工作量要合理，不要低估也不要高估
6. 任务标题要具体，"看教材第3章" 比 "复习" 好

请严格按以下 JSON 格式输出：
{
  "tasks": [
    {
      "title": "任务标题（具体）",
      "estimated_minutes": 60,
      "dependencies": ["任务A标题或索引"],
      "deadline_type": "hard | soft",
      "deadline": "YYYY-MM-DD HH:MM（硬截止必填，软截止可选）",
      "energy_level": "high | medium | low",
      "description": "简短说明这个任务做什么"
    }
  ],
  "total_estimated_minutes": 600,
  "breakdown_strategy": "简述分解思路"
}

常见任务类型参考：
- 信息收集：查资料、找教材、了解范围
- 知识学习：看书、看视频、记笔记
- 练习巩固：做题、刷题、写代码
- 复习总结：整理笔记、做思维导图
- 模拟测试：做真题、限时练习
- 考前准备：查漏补缺、调整状态
"""


class GoalDecomposer:
    """目标分解器。"""

    def __init__(self, llm: Optional[ChatModel] = None):
        self.llm = llm

    def decompose(
        self,
        goal: Goal,
        deadline: Optional[datetime] = None,
    ) -> List[Task]:
        """将目标分解为任务列表。

        Args:
            goal: 目标对象
            deadline: 截止时间（可选，从 goal 中推断）

        Returns:
            Task 列表（尚未持久化，id 为空）
        """
        if self.llm is None or self.llm.mock_mode:
            return self._rule_based_decompose(goal, deadline)

        deadline_str = deadline.isoformat() if deadline else "未指定"

        user_prompt = f"""
目标标题：{goal.title}
目标描述：{goal.description or '无'}
目标权重：{goal.weight}
截止时间：{deadline_str}

请将此目标分解为具体任务列表。
"""

        messages = [
            ChatMessage(ChatRole.SYSTEM, SYSTEM_PROMPT),
            ChatMessage(ChatRole.USER, user_prompt),
        ]

        try:
            response = self.llm.chat_json(messages, temperature=0.7)
            return self._build_tasks(goal, response, deadline)
        except Exception as e:
            return self._rule_based_decompose(goal, deadline)

    def _build_tasks(
        self,
        goal: Goal,
        response: Dict,
        deadline: Optional[datetime],
    ) -> List[Task]:
        """从 LLM 响应构建 Task 对象。"""
        task_dicts = response.get("tasks", [])
        tasks: List[Task] = []

        # 标题到索引的映射（用于解析依赖）
        title_to_idx: Dict[str, int] = {}

        for i, td in enumerate(task_dicts):
            title = td.get("title", f"任务{i+1}")
            title_to_idx[title] = i

            deadline_type = DeadlineType.HARD if td.get("deadline_type") == "hard" else DeadlineType.SOFT
            task_deadline = td.get("deadline") or (deadline.isoformat() if deadline else None)

            energy_str = td.get("energy_level", "medium")
            energy_level = EnergyLevel(energy_str) if energy_str in EnergyLevel else EnergyLevel.MEDIUM

            task = Task(
                title=title,
                goal_id=goal.id,
                estimated_minutes=td.get("estimated_minutes", 60),
                deadline_type=deadline_type,
                deadline=task_deadline,
                energy_level=energy_level,
                description=td.get("description", ""),
                priority_weight=goal.weight,
            )
            tasks.append(task)

        # 解析依赖（将标题/索引转换为 task.id 占位符，后续 Agent 会处理）
        for i, td in enumerate(task_dicts):
            deps = td.get("dependencies", [])
            dep_ids = []
            for dep in deps:
                if dep in title_to_idx:
                    # 用索引位置作为临时标识，Agent 会在分配真实 id 后替换
                    dep_ids.append(f"__idx_{title_to_idx[dep]}")
                elif dep.startswith("__idx_"):
                    dep_ids.append(dep)
            tasks[i].dependencies = dep_ids

        return tasks

    def _rule_based_decompose(
        self,
        goal: Goal,
        deadline: Optional[datetime],
    ) -> List[Task]:
        """基于规则的兜底分解（Mock 模式使用）。

        对于考试类目标，生成标准的五阶段分解。
        """
        title = goal.title.lower()
        tasks: List[Task] = []

        # 考试类目标的标准分解
        if any(kw in title for kw in ["考试", "期末", "期中", "考研", "复习"]):
            exam_tasks = [
                ("信息收集：了解考试范围和重点", 45, EnergyLevel.MEDIUM, DeadlineType.SOFT),
                ("第一轮复习：通读教材和笔记", 180, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("第二轮复习：重点章节深入理解", 180, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("做题练习：历年真题/课后习题", 240, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("第三轮复习：错题和薄弱点突破", 120, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("模拟测试：限时完整模拟", 120, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("考前整理：知识图谱和公式速记", 60, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
            for i, (t, mins, energy, dl_type) in enumerate(exam_tasks):
                task = Task(
                    title=t,
                    goal_id=goal.id,
                    estimated_minutes=mins,
                    deadline_type=dl_type,
                    deadline=deadline.isoformat() if deadline and dl_type == DeadlineType.HARD else None,
                    energy_level=energy,
                    description=f"考试复习第 {i+1} 阶段",
                    priority_weight=goal.weight,
                    dependencies=[f"__idx_{i-1}"] if i > 0 else [],
                )
                tasks.append(task)

        # 项目/大作业类目标的标准分解
        elif any(kw in title for kw in ["项目", "大作业", "作业", "报告", "论文"]):
            project_tasks = [
                ("需求分析与方案设计", 60, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("核心功能实现", 240, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("测试与调试", 120, EnergyLevel.MEDIUM, DeadlineType.SOFT),
                ("文档撰写与整理", 90, EnergyLevel.LOW, DeadlineType.SOFT),
                ("最终检查与提交", 30, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
            for i, (t, mins, energy, dl_type) in enumerate(project_tasks):
                task = Task(
                    title=t,
                    goal_id=goal.id,
                    estimated_minutes=mins,
                    deadline_type=dl_type,
                    deadline=deadline.isoformat() if deadline and dl_type == DeadlineType.HARD else None,
                    energy_level=energy,
                    description=f"项目阶段 {i+1}",
                    priority_weight=goal.weight,
                    dependencies=[f"__idx_{i-1}"] if i > 0 else [],
                )
                tasks.append(task)

        # 通用分解
        else:
            generic_tasks = [
                ("制定计划与收集资料", 60, EnergyLevel.MEDIUM),
                ("核心学习/实践", 300, EnergyLevel.HIGH),
                ("总结与巩固", 120, EnergyLevel.MEDIUM),
            ]
            for i, (t, mins, energy) in enumerate(generic_tasks):
                task = Task(
                    title=t,
                    goal_id=goal.id,
                    estimated_minutes=mins,
                    deadline_type=DeadlineType.SOFT,
                    energy_level=energy,
                    description=f"通用任务 {i+1}",
                    priority_weight=goal.weight,
                    dependencies=[f"__idx_{i-1}"] if i > 0 else [],
                )
                tasks.append(task)

        return tasks

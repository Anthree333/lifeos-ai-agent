"""目标分解器。

将用户的模糊目标（如"准备高数考试"）分解为具体任务列表。
LLM 负责分解，调度器负责排期，职责分离。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..llm.chat_model import ChatModel, ChatMessage, ChatRole
from ..models import (
    Task,
    Goal,
    DeadlineType,
    EnergyLevel,
    StudentProfile,
    ExecutionRecord,
)


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


PLANNER_PROMPT = """你是 LifeOS 的智能学习规划师。你的任务是将用户的学习目标分解为个性化、可执行的任务列表。

分解原则：
1. 每个任务时长在 30-120 分钟之间
2. 任务之间有依赖关系（先 A 后 B），支持并行依赖
3. 区分硬截止（考试/提交日期）和软截止（自我设定）
4. 根据剩余天数动态调整任务粒度和工作量
5. 考虑学生精力分布（高精力时段安排难点，低精力时段安排简单任务）
6. 任务标题要具体，"看教材第3章微积分" 比 "复习" 好

个性化考虑：
- 剩余天数充足（>14天）：粗粒度分解，每天投入适度
- 剩余天数适中（3-14天）：中粒度分解，聚焦核心内容
- 剩余天数紧张（<3天）：压缩为关键任务，优先硬截止

输出 JSON：
{
  "tasks": [
    {
      "title": "具体任务标题",
      "estimated_minutes": 60,
      "dependencies": ["任务标题或索引"],
      "deadline_type": "hard | soft",
      "deadline": "YYYY-MM-DD HH:MM（硬截止必填）",
      "energy_level": "high | medium | low",
      "description": "做什么"
    }
  ],
  "total_estimated_minutes": 600,
  "breakdown_strategy": "分解思路说明",
  "daily_target_minutes": 180,
  "milestone_tasks": [0, 3],
  "risk_notes": "风险预判"
}
"""


@dataclass
class DecomposeResult:
    """目标分解结果（增强版）。"""
    tasks: List[Task] = field(default_factory=list)
    total_estimated_minutes: int = 0
    breakdown_strategy: str = ""
    daily_target_minutes: int = 180      # 建议每天投入多少分钟
    milestone_tasks: List[int] = field(default_factory=list)  # 里程碑任务索引
    risk_notes: str = ""                 # 规划时的风险预判
    decision_trace: List[str] = field(default_factory=list)   # 决策链


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

    def decompose_with_context(
        self,
        goal: Goal,
        deadline: Optional[datetime] = None,
        profile: Optional[StudentProfile] = None,
        existing_tasks: Optional[List[Task]] = None,
        execution_records: Optional[List[ExecutionRecord]] = None,
        now: Optional[datetime] = None,
    ) -> DecomposeResult:
        """带上下文的智能分解。

        注入学生画像、剩余天数、已有任务负载、历史执行记录，
        生成个性化分解策略。
        """
        if now is None:
            now = datetime.now()

        if self.llm is None or self.llm.mock_mode:
            return self._rule_based_decompose_adaptive(goal, deadline, profile, now)

        # 构建个性化上下文
        days_remaining = None
        if deadline:
            days_remaining = (deadline - now).days

        # 计算已有负载
        existing_load = sum(t.remaining_minutes for t in (existing_tasks or []))

        # 分析历史完成率
        completion_rate = 0.8  # 默认
        if execution_records:
            completed = sum(1 for r in execution_records if r.progress_delta >= 1.0)
            total = len(execution_records)
            if total > 0:
                completion_rate = completed / total

        # 构建上下文 prompt
        user_prompt = self._build_context_prompt(
            goal, deadline, profile, days_remaining, existing_load, completion_rate
        )

        messages = [
            ChatMessage(ChatRole.SYSTEM, PLANNER_PROMPT),
            ChatMessage(ChatRole.USER, user_prompt),
        ]

        try:
            response = self.llm.chat_json(messages, temperature=0.7)
            tasks = self._build_tasks(goal, response, deadline)
            return DecomposeResult(
                tasks=tasks,
                total_estimated_minutes=response.get(
                    "total_estimated_minutes",
                    sum(t.estimated_minutes for t in tasks),
                ),
                breakdown_strategy=response.get("breakdown_strategy", ""),
                daily_target_minutes=response.get("daily_target_minutes", 180),
                milestone_tasks=response.get("milestone_tasks", []),
                risk_notes=response.get("risk_notes", ""),
                decision_trace=[
                    f"LLM分解: {len(tasks)}个任务, 日标"
                    f"{response.get('daily_target_minutes', 180)}分钟"
                ],
            )
        except Exception:
            result = self._rule_based_decompose_adaptive(goal, deadline, profile, now)
            result.decision_trace.append("LLM失败，回退规则分解")
            return result

    def _build_context_prompt(
        self,
        goal: Goal,
        deadline: Optional[datetime],
        profile: Optional[StudentProfile],
        days_remaining: Optional[int],
        existing_load: int,
        completion_rate: float,
    ) -> str:
        """构建个性化上下文提示。"""
        parts = [f"目标标题：{goal.title}"]
        parts.append(f"目标描述：{goal.description or '无'}")
        parts.append(f"目标权重：{goal.weight}")
        parts.append(f"截止时间：{deadline.isoformat() if deadline else '未指定'}")
        if days_remaining is not None:
            parts.append(f"剩余天数：{days_remaining}天")
        if profile:
            parts.append(f"每日高精力时段：{profile.daily_high_energy_hours}小时")
            parts.append(f"起床时间：{profile.wake_up_time.strftime('%H:%M')}")
        if existing_load:
            parts.append(f"已有任务负载：{existing_load}分钟")
        if completion_rate:
            parts.append(f"历史完成率：{completion_rate:.0%}")
        parts.append("\n请根据以上信息个性化分解目标。")
        return "\n".join(parts)

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

    # ==========================================
    # 自适应规则分解（带上下文）
    # ==========================================

    def _rule_based_decompose_adaptive(
        self,
        goal: Goal,
        deadline: Optional[datetime],
        profile: Optional[StudentProfile],
        now: datetime,
    ) -> DecomposeResult:
        """自适应规则分解。"""
        days_remaining = 7  # 默认
        if deadline:
            days_remaining = max(1, (deadline - now).days)

        title = goal.title.lower()

        if any(kw in title for kw in ["考试", "期末", "期中", "考研", "复习"]):
            tasks = self._exam_decompose_adaptive(goal, deadline, days_remaining)
        elif any(kw in title for kw in ["项目", "大作业", "作业", "报告", "论文"]):
            tasks = self._project_decompose_adaptive(goal, deadline, days_remaining)
        else:
            tasks = self._generic_decompose_adaptive(goal, deadline, days_remaining)

        daily_target = min(
            240,
            max(60, sum(t.estimated_minutes for t in tasks) // max(1, days_remaining)),
        )

        return DecomposeResult(
            tasks=tasks,
            total_estimated_minutes=sum(t.estimated_minutes for t in tasks),
            breakdown_strategy=f"自适应规则分解（{days_remaining}天）",
            daily_target_minutes=daily_target,
            milestone_tasks=[0, len(tasks) // 2] if len(tasks) >= 3 else [0],
            risk_notes="时间紧张" if days_remaining < 3 else "",
            decision_trace=[f"规则分解: {len(tasks)}个任务, {days_remaining}天"],
        )

    def _build_spec_tasks(
        self,
        goal: Goal,
        deadline: Optional[datetime],
        spec: List[tuple],
        stage_prefix: str,
    ) -> List[Task]:
        """根据 (标题, 时长, 精力, 截止类型) 规格列表构建任务。

        最后一个任务的截止类型应为 HARD，其余 SOFT；依赖链为前置索引。
        """
        tasks: List[Task] = []
        for i, (title, mins, energy, dl_type) in enumerate(spec):
            task = Task(
                title=title,
                goal_id=goal.id,
                estimated_minutes=mins,
                deadline_type=dl_type,
                deadline=deadline.isoformat() if deadline and dl_type == DeadlineType.HARD else None,
                energy_level=energy,
                description=f"{stage_prefix}第 {i+1} 阶段",
                priority_weight=goal.weight,
                dependencies=[f"__idx_{i-1}"] if i > 0 else [],
            )
            tasks.append(task)
        return tasks

    def _exam_decompose_adaptive(
        self,
        goal: Goal,
        deadline: Optional[datetime],
        days_remaining: int,
    ) -> List[Task]:
        """考试类自适应分解。

        >14天：7任务（信息收集→通读→重点→做题→错题→模拟→考前整理）
        3-14天：5任务（合并通读+重点，保留做题+模拟+考前整理）
        <3天：4任务（核心复习→做题→模拟→考前整理），每个时长拉长
        """
        if days_remaining > 14:
            spec = [
                ("信息收集：了解考试范围和重点", 45, EnergyLevel.MEDIUM, DeadlineType.SOFT),
                ("第一轮复习：通读教材和笔记", 180, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("第二轮复习：重点章节深入理解", 180, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("做题练习：历年真题和课后习题", 240, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("错题突破：薄弱点专项训练", 120, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("模拟测试：限时完整模拟", 120, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("考前整理：知识图谱和公式速记", 60, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
        elif days_remaining >= 3:
            spec = [
                ("信息收集：确定考试重点", 30, EnergyLevel.MEDIUM, DeadlineType.SOFT),
                ("通读重点：教材核心章节", 150, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("做题练习：真题与课后习题", 180, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("模拟测试：限时演练", 90, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("考前整理：公式与错题速记", 60, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
        else:
            # <3天，压缩为关键任务，每个时长拉长
            spec = [
                ("核心复习：高频考点速过", 120, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("做题冲刺：真题限时训练", 120, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("模拟测试：完整演练", 90, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("考前整理：必背公式速记", 60, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
        return self._build_spec_tasks(goal, deadline, spec, "考试复习")

    def _project_decompose_adaptive(
        self,
        goal: Goal,
        deadline: Optional[datetime],
        days_remaining: int,
    ) -> List[Task]:
        """项目类自适应分解。

        >14天：5任务（需求→实现→测试→文档→提交）
        3-14天：4任务（合并测试+文档）
        <3天：3任务（核心实现→测试→提交），压缩非核心
        """
        if days_remaining > 14:
            spec = [
                ("需求分析与方案设计", 60, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("核心功能实现", 240, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("测试与调试", 120, EnergyLevel.MEDIUM, DeadlineType.SOFT),
                ("文档撰写与整理", 90, EnergyLevel.LOW, DeadlineType.SOFT),
                ("最终检查与提交", 30, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
        elif days_remaining >= 3:
            spec = [
                ("需求分析与方案设计", 45, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("核心功能实现", 200, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("测试调试与文档整理", 120, EnergyLevel.MEDIUM, DeadlineType.SOFT),
                ("最终检查与提交", 30, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
        else:
            # <3天，压缩为关键任务
            spec = [
                ("核心功能实现", 180, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("测试调试", 90, EnergyLevel.MEDIUM, DeadlineType.SOFT),
                ("最终检查与提交", 30, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
        return self._build_spec_tasks(goal, deadline, spec, "项目阶段")

    def _generic_decompose_adaptive(
        self,
        goal: Goal,
        deadline: Optional[datetime],
        days_remaining: int,
    ) -> List[Task]:
        """通用类自适应分解。

        >14天：5任务（计划→资料→核心学习→练习→总结）
        3-14天：3任务（计划→核心学习→总结）
        <3天：2任务（核心学习→总结）
        """
        if days_remaining > 14:
            spec = [
                ("制定计划与收集资料", 45, EnergyLevel.MEDIUM, DeadlineType.SOFT),
                ("资料研读与笔记整理", 90, EnergyLevel.MEDIUM, DeadlineType.SOFT),
                ("核心学习与实践", 240, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("练习巩固", 120, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("总结与巩固", 60, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
        elif days_remaining >= 3:
            spec = [
                ("制定计划与收集资料", 45, EnergyLevel.MEDIUM, DeadlineType.SOFT),
                ("核心学习与实践", 240, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("总结与巩固", 60, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
        else:
            # <3天，压缩为关键任务
            spec = [
                ("核心学习与实践", 180, EnergyLevel.HIGH, DeadlineType.SOFT),
                ("总结与巩固", 60, EnergyLevel.MEDIUM, DeadlineType.HARD),
            ]
        return self._build_spec_tasks(goal, deadline, spec, "通用任务")

"""Agent 编排层。

Agent 每轮执行固定五步：
1. 解析用户输入为新状态/事件
2. 判断是否需要重规划
3. 分解目标或补充分解任务
4. 输出计划变更意图，交由确定性调度器执行
5. 生成中文解释，说明改了什么、为什么、牺牲了什么
"""

from .life_agent import LifeAgent, AgentResult
from .parser import InputParser
from .decomposer import GoalDecomposer
from .explainer import PlanExplainer

__all__ = [
    "LifeAgent",
    "AgentResult",
    "InputParser",
    "GoalDecomposer",
    "PlanExplainer",
]

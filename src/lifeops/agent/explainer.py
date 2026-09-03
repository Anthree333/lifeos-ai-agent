"""计划变更解释器。

对比两个 PlanSnapshot，生成自然语言解释，说明：
- 改了什么（新增/删除/调整时间的任务）
- 为什么（原因：进度偏差 / 新事件 / 新目标）
- 牺牲了什么（如果有）
- 下一步建议

不需要 LLM 也能生成基础解释（规则驱动），有 LLM 时更自然。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..llm.chat_model import ChatModel, ChatMessage, ChatRole
from ..models import TimeSlot, PlanSnapshot, SacrificeItem, ChangeLog, Task
from ..scheduler import ScheduleResult


@dataclass
class PlanDiff:
    """计划差异。"""
    added_tasks: List[str] = field(default_factory=list)       # 新增的任务标题
    removed_tasks: List[str] = field(default_factory=list)     # 被移除的任务标题
    moved_tasks: List[Tuple[str, str, str]] = field(default_factory=list)  # (标题, 原时间, 新时间)
    sacrificed: List[str] = field(default_factory=list)        # 被牺牲的任务标题
    new_risks: List[str] = field(default_factory=list)         # 新出现的风险
    total_minutes_change: int = 0                               # 总时长变化
    reason: str = ""                                            # 变更原因


SYSTEM_PROMPT = """你是 LifeOS 的计划解释器。你的任务是用友好、清晰的中文向用户解释计划变更。

解释风格：
- 口语化，但专业可信
- 先说最重要的变化（硬截止、高优先级）
- 先说原因，再说具体变化，最后说建议
- 如果有牺牲，明确说清楚牺牲了什么、为什么
- 控制在 150 字以内，简洁有力
- 用"你"称呼用户
- 不要罗列全部细节，挑最重要的说

输出格式：直接输出中文解释文本，不要 JSON，不要 Markdown 标题。
"""


class PlanExplainer:
    """计划变更解释器。"""

    def __init__(self, llm: Optional[ChatModel] = None):
        self.llm = llm

    def explain(
        self,
        old_snapshot: Optional[PlanSnapshot],
        new_result: ScheduleResult,
        reason: str = "",
        new_tasks: Optional[List[Task]] = None,
    ) -> ChangeLog:
        """生成计划变更解释。

        Args:
            old_snapshot: 旧计划快照（None 表示首次计划）
            new_result: 新调度结果
            reason: 变更原因（自然语言描述）
            new_tasks: 新增的任务列表

        Returns:
            ChangeLog 变更记录
        """
        diff = self._compute_diff(old_snapshot, new_result, new_tasks)
        diff.reason = reason

        if self.llm is None or self.llm.mock_mode:
            explanation = self._rule_based_explain(diff)
        else:
            explanation = self._llm_explain(diff)

        return ChangeLog(
            trigger_reason="replan" if old_snapshot else "initial",
            summary=explanation[:100],
            detail=explanation,
            reason=reason,
            affected_tasks=diff.added_tasks + diff.removed_tasks + [t[0] for t in diff.moved_tasks] + diff.sacrificed,
        )

    def _compute_diff(
        self,
        old_snapshot: Optional[PlanSnapshot],
        new_result: ScheduleResult,
        new_tasks: Optional[List[Task]],
    ) -> PlanDiff:
        """计算两个计划的差异。"""
        diff = PlanDiff()

        # 构建旧计划任务 -> 时间映射
        old_task_times: Dict[str, str] = {}
        if old_snapshot:
            for ts in old_snapshot.time_slots:
                if ts.task_id and ts.task_id not in old_task_times:
                    old_task_times[ts.task_id] = ts.start_time

        # 构建新计划任务 -> 时间映射
        new_task_times: Dict[str, str] = {}
        for ts in new_result.time_slots:
            if ts.task_id and ts.task_id not in new_task_times:
                new_task_times[ts.task_id] = ts.start_time

        # 新增任务（新计划有，旧计划无）
        new_task_ids = set(new_task_times.keys())
        old_task_ids = set(old_task_times.keys())

        for tid in new_task_ids - old_task_ids:
            diff.added_tasks.append(tid)

        # 移除的任务（旧计划有，新计划无）
        for tid in old_task_ids - new_task_ids:
            diff.removed_tasks.append(tid)

        # 时间变动的任务
        for tid in new_task_ids & old_task_ids:
            if old_task_times[tid] != new_task_times[tid]:
                diff.moved_tasks.append((tid, old_task_times[tid], new_task_times[tid]))

        # 牺牲清单
        for s in new_result.sacrifice_list:
            diff.sacrificed.append(s.task_title)

        # 总时长变化
        old_total = old_snapshot.total_scheduled_minutes if old_snapshot else 0
        diff.total_minutes_change = new_result.total_scheduled_minutes - old_total

        return diff

    def _rule_based_explain(self, diff: PlanDiff) -> str:
        """基于规则的兜底解释。"""
        lines = []

        # 原因
        if diff.reason:
            lines.append(f"【变更原因】{diff.reason}")

        # 首次计划
        if not diff.removed_tasks and not diff.moved_tasks and diff.added_tasks:
            lines.append(f"已为你生成新的学习计划，共安排 {len(diff.added_tasks)} 个任务。")
            return "\n".join(lines)

        # 新增任务
        if diff.added_tasks:
            lines.append(f"新增 {len(diff.added_tasks)} 个任务。")

        # 移除任务
        if diff.removed_tasks:
            lines.append(f"移除 {len(diff.removed_tasks)} 个任务。")

        # 时间调整
        if diff.moved_tasks:
            lines.append(f"调整了 {len(diff.moved_tasks)} 个任务的时间安排。")

        # 牺牲
        if diff.sacrificed:
            lines.append(f"由于时间紧张，暂时搁置了 {len(diff.sacrificed)} 个低优先级任务：{', '.join(diff.sacrificed[:3])}")
            if len(diff.sacrificed) > 3:
                lines[-1] += f" 等 {len(diff.sacrificed)} 个。"

        # 建议
        if diff.reason and "进度" in diff.reason:
            lines.append("建议优先保证硬截止任务的完成，软截止任务可以适当延后。")

        if not lines:
            lines.append("计划没有重大变更。")

        return "\n".join(lines)

    def _llm_explain(self, diff: PlanDiff) -> str:
        """用 LLM 生成更自然的解释。"""
        diff_desc = {
            "reason": diff.reason,
            "added_count": len(diff.added_tasks),
            "added_examples": diff.added_tasks[:3],
            "removed_count": len(diff.removed_tasks),
            "moved_count": len(diff.moved_tasks),
            "sacrificed": diff.sacrificed[:3],
            "total_minutes_change": diff.total_minutes_change,
        }

        import json
        user_prompt = f"计划变更详情：\n{json.dumps(diff_desc, ensure_ascii=False, indent=2)}\n\n请生成中文解释。"

        messages = [
            ChatMessage(ChatRole.SYSTEM, SYSTEM_PROMPT),
            ChatMessage(ChatRole.USER, user_prompt),
        ]

        try:
            response = self.llm.chat(messages, temperature=0.7)
            return response.content.strip()
        except Exception:
            return self._rule_based_explain(diff)

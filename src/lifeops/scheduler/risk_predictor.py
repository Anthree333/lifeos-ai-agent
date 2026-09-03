"""风险预测器。

公式：预计完成日 = 今天 + 剩余任务时长 ÷ 近 N 天平均完成速度

当预计完成日晚于截止日时，生成 RiskReport，
并给出「提前 / 降级 / 延期 / 求助」四类建议。
"""
from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

from ..models import Task, ExecutionRecord, DeadlineType
from ..models.risk import RiskReport, RiskLevel, RiskSuggestionType


class RiskPredictor:
    """风险预测器。"""

    def __init__(
        self,
        lookback_days: int = 3,
        min_records_for_estimate: int = 2,
    ):
        self.lookback_days = lookback_days
        self.min_records_for_estimate = min_records_for_estimate

    def predict(
        self,
        task: Task,
        execution_records: List[ExecutionRecord],
        now: Optional[datetime] = None,
    ) -> RiskReport:
        """预测单个任务的完成风险。

        Args:
            task: 任务对象
            execution_records: 该任务的执行记录列表
            now: 当前时间

        Returns:
            RiskReport 风险报告
        """
        if now is None:
            now = datetime.now()

        remaining_minutes = task.remaining_minutes

        # 计算近 N 天平均完成速度
        avg_daily_minutes = self._calc_avg_daily_minutes(
            execution_records, now
        )

        # 预计完成时间
        estimated_finish_date, delay_days = self._calc_estimated_finish(
            remaining_minutes, avg_daily_minutes, task.deadline, now
        )

        # 判断风险等级
        risk_level = self._calc_risk_level(
            delay_days, task.deadline_type
        )

        # 生成建议
        suggestion_type, suggestion_text = self._generate_suggestion(
            task, risk_level, delay_days, avg_daily_minutes, remaining_minutes
        )

        return RiskReport(
            task_id=task.id,
            task_title=task.title,
            risk_level=risk_level,
            estimated_finish_date=estimated_finish_date.isoformat() if estimated_finish_date else None,
            deadline=task.deadline,
            delay_days=round(delay_days, 1),
            remaining_minutes=remaining_minutes,
            avg_daily_minutes=round(avg_daily_minutes, 1),
            suggestion_type=suggestion_type,
            suggestion_text=suggestion_text,
        )

    def predict_all(
        self,
        tasks: List[Task],
        all_records: List[ExecutionRecord],
        now: Optional[datetime] = None,
    ) -> List[RiskReport]:
        """批量预测所有任务的风险。"""
        # 按任务分组执行记录
        records_by_task: Dict[str, List[ExecutionRecord]] = {}
        for r in all_records:
            records_by_task.setdefault(r.task_id, []).append(r)

        reports = []
        for task in tasks:
            if task.status == "completed" or task.status == "cancelled":
                continue
            if task.remaining_minutes <= 0:
                continue

            records = records_by_task.get(task.id, [])
            report = self.predict(task, records, now)
            reports.append(report)

        # 按风险等级排序（高风险在前）
        level_order = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 3,
        }
        reports.sort(key=lambda r: level_order.get(r.risk_level, 99))

        return reports

    # ==========================================
    # 内部计算方法
    # ==========================================

    def _calc_avg_daily_minutes(
        self,
        records: List[ExecutionRecord],
        now: datetime,
    ) -> float:
        """计算近 N 天的平均每日完成分钟数。

        如果记录不足，使用估算值（每天 60 分钟起步）。
        """
        if not records:
            return 60.0  # 默认估算：每天 1 小时

        # 筛选近 N 天的记录
        cutoff = now - timedelta(days=self.lookback_days)
        recent = [
            r for r in records
            if self._parse_date(r.record_date) >= cutoff.date()
        ]

        if not recent:
            # 没有近期数据，用历史平均
            total_minutes = sum(r.actual_minutes for r in records)
            days = max(len(records), 1)
            return total_minutes / days

        # 按天汇总
        daily_minutes: Dict[date, int] = {}
        for r in recent:
            d = self._parse_date(r.record_date)
            daily_minutes[d] = daily_minutes.get(d, 0) + r.actual_minutes

        total_minutes = sum(daily_minutes.values())
        days_count = max(len(daily_minutes), self.min_records_for_estimate)

        return total_minutes / days_count

    def _calc_estimated_finish(
        self,
        remaining_minutes: int,
        avg_daily_minutes: float,
        deadline: Optional[str],
        now: datetime,
    ) -> Tuple[Optional[date], float]:
        """计算预计完成日期和延期天数。

        Returns:
            (预计完成日期, 延期天数)
            延期天数 > 0 表示会延期
        """
        if avg_daily_minutes <= 0:
            avg_daily_minutes = 30.0  # 保底值

        days_needed = remaining_minutes / avg_daily_minutes
        estimated_date = now.date() + timedelta(days=days_needed)

        # 计算延期天数
        delay_days = 0.0
        if deadline:
            try:
                deadline_date = datetime.fromisoformat(deadline).date()
                delta = (estimated_date - deadline_date).total_seconds() / 86400.0
                delay_days = max(0.0, delta)
            except (ValueError, TypeError):
                pass

        return estimated_date, delay_days

    def _calc_risk_level(
        self,
        delay_days: float,
        deadline_type: DeadlineType,
    ) -> RiskLevel:
        """根据延期天数和截止类型判断风险等级。"""
        if delay_days <= 0:
            return RiskLevel.LOW

        if deadline_type == DeadlineType.HARD:
            if delay_days > 3:
                return RiskLevel.CRITICAL
            elif delay_days > 1:
                return RiskLevel.HIGH
            else:
                return RiskLevel.MEDIUM
        else:
            # 软截止
            if delay_days > 5:
                return RiskLevel.HIGH
            elif delay_days > 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

    def _generate_suggestion(
        self,
        task: Task,
        risk_level: RiskLevel,
        delay_days: float,
        avg_daily_minutes: float,
        remaining_minutes: int,
    ) -> Tuple[RiskSuggestionType, str]:
        """生成风险建议。"""
        if risk_level == RiskLevel.LOW:
            return (
                RiskSuggestionType.ADVANCE,
                "进度正常，按计划执行即可。",
            )

        if risk_level == RiskLevel.MEDIUM:
            extra_daily = remaining_minutes / max(1, delay_days) - avg_daily_minutes
            return (
                RiskSuggestionType.ADVANCE,
                f"建议每天多投入 {max(15, int(extra_daily))} 分钟，"
                f"或利用周末集中推进，可按时完成。",
            )

        if risk_level == RiskLevel.HIGH:
            if task.deadline_type == DeadlineType.HARD:
                return (
                    RiskSuggestionType.DOWNGRADE,
                    f"预计延期 {delay_days:.1f} 天。"
                    f"建议：1) 降低任务范围，优先完成核心部分；"
                    f"2) 每天增加 1-2 小时投入；"
                    f"3) 考虑寻求同学帮助。",
                )
            else:
                return (
                    RiskSuggestionType.DELAY,
                    f"预计延期 {delay_days:.1f} 天。"
                    f"该任务为软截止，建议推迟 deadline，优先保障硬截止任务。",
                )

        # CRITICAL
        return (
            RiskSuggestionType.DELEGATE,
            f"严重风险：预计延期 {delay_days:.1f} 天。"
            f"强烈建议：1) 立即寻求老师/助教帮助；"
            f"2) 大幅度简化任务目标；"
            f"3) 考虑申请延期。",
        )

    def _parse_date(self, date_str: str) -> date:
        """解析日期字符串。"""
        try:
            return date.fromisoformat(date_str)
        except (ValueError, TypeError):
            return date.today()

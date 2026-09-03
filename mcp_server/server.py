"""LifeOS MCP Server.

基于官方 MCP Python SDK (v2.x) 的 stdio 服务。
提供 LifeOS 核心能力的工具接口：目标管理、计划调度、进度汇报等。
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp.server.mcpserver import MCPServer

from lifeops.models import (
    StudentProfile, Goal, Task, Commitment,
    TaskStatus, DeadlineType, EnergyLevel, CommitmentType,
)
from lifeops.agent import LifeAgent
from lifeops.storage.database import get_db


# 创建 MCP 服务器实例
mcp = MCPServer("lifeos-tools")

# 全局 Agent 实例（懒加载）
_agent: Optional[LifeAgent] = None


def get_agent() -> LifeAgent:
    """获取或创建全局 Agent 实例。"""
    global _agent
    if _agent is None:
        default_db_path = str(
            Path(__file__).resolve().parent.parent / "data" / "lifeos.db"
        )
        db = get_db(os.environ.get("DB_PATH") or default_db_path)
        profile = db.get_default_profile()
        _agent = LifeAgent(profile=profile, db=db)
    return _agent


# ==========================================
# 核心工具：目标与计划
# ==========================================

@mcp.tool()
def create_goal(title: str, description: str = "", weight: float = 0.5,
                deadline: Optional[str] = None) -> dict:
    """创建新目标并自动生成学习计划。

    告诉 LifeOS 你的目标，它会自动分解任务并安排到日程中。

    Args:
        title: 目标标题（如"准备高数期末考试"）
        description: 目标详细描述（可选）
        weight: 目标权重 0-1，越高越优先（默认 0.5）
        deadline: 截止日期，ISO 格式（如"2024-06-15T09:00:00"），可选

    Returns:
        包含目标信息、任务列表和计划摘要的字典
    """
    agent = get_agent()

    # 结构化建目标，避免自然语言规则吞掉 deadline/weight
    result = agent.create_goal_structured(
        title=title,
        description=description,
        weight=weight,
        deadline=deadline,
    )

    # 格式化响应
    tasks_info = [
        {
            "id": t.id,
            "title": t.title,
            "estimated_minutes": t.estimated_minutes,
            "deadline_type": t.deadline_type.value if hasattr(t.deadline_type, 'value') else t.deadline_type,
            "progress": t.progress,
            "status": t.status.value if hasattr(t.status, 'value') else t.status,
        }
        for t in agent._tasks
    ]

    response = {
        "success": True,
        "goal_title": title,
        "tasks_count": len(agent._tasks),
        "tasks": tasks_info,
        "total_minutes": result.snapshot.total_scheduled_minutes if result.snapshot else 0,
        "sacrifice_count": len(result.snapshot.sacrifice_list) if result.snapshot else 0,
        "explanation": result.reply,
    }

    return response


@mcp.tool()
def report_progress(task_id: str = "", progress: float = 0.0,
                    actual_minutes: int = 0, note: str = "") -> dict:
    """汇报任务进度，触发计划重排（如果偏差足够大）。

    Args:
        task_id: 任务 ID（为空则自动匹配第一个进行中的任务）
        progress: 当前进度 0-1（如 0.3 表示 30%）
        actual_minutes: 本次实际花费的分钟数
        note: 进度备注

    Returns:
        包含更新后状态和计划变更的字典
    """
    agent = get_agent()

    result = agent.report_absolute_progress(
        task_id=task_id,
        progress=progress,
        actual_minutes=actual_minutes,
        note=note,
    )

    response = {
        "success": True,
        "plan_changed": result.changed,
        "explanation": result.reply,
        "total_minutes": result.snapshot.total_scheduled_minutes if result.snapshot else 0,
    }

    return response


@mcp.tool()
def get_current_plan(days: int = 3) -> dict:
    """获取当前学习计划。

    Args:
        days: 查看未来几天的计划（默认 3 天）

    Returns:
        包含计划快照和每日安排的字典
    """
    agent = get_agent()
    snapshot = agent.current_snapshot

    if snapshot is None:
        return {
            "success": False,
            "message": "还没有计划，先创建一个目标吧！",
            "tasks": [],
            "daily_schedule": {},
        }

    # 按天分组
    today = date.today()
    daily_schedule: Dict[str, list] = {}

    for i in range(days):
        day = today + timedelta(days=i)
        day_str = day.isoformat()
        day_slots = [
            ts for ts in snapshot.time_slots
            if ts.start_time.startswith(day_str)
        ]
        daily_schedule[day_str] = [
            {
                "start": ts.start_time,
                "end": ts.end_time,
                "task_id": ts.task_id,
                "title": ts.title,
                "source": ts.source.value if hasattr(ts.source, 'value') else ts.source,
                "duration_minutes": int(
                    (datetime.fromisoformat(ts.end_time)
                     - datetime.fromisoformat(ts.start_time)).total_seconds() / 60
                ),
            }
            for ts in day_slots
        ]

    response = {
        "success": True,
        "total_tasks": len(agent._tasks),
        "total_minutes": snapshot.total_scheduled_minutes,
        "sacrifice_list": [
            {"task_title": s.task_title, "reason": s.reason}
            for s in snapshot.sacrifice_list
        ],
        "daily_schedule": daily_schedule,
    }

    return response


@mcp.tool()
def get_task_list(status: str = "all") -> dict:
    """获取任务列表。

    Args:
        status: 任务状态过滤：all / pending / in_progress / completed

    Returns:
        任务列表字典
    """
    agent = get_agent()
    tasks = agent._tasks

    if status != "all":
        tasks = [t for t in tasks if (
            t.status.value if hasattr(t.status, 'value') else t.status
        ) == status]

    tasks_info = [
        {
            "id": t.id,
            "title": t.title,
            "goal_id": t.goal_id,
            "estimated_minutes": t.estimated_minutes,
            "progress": t.progress,
            "progress_percent": int(t.progress * 100),
            "status": t.status.value if hasattr(t.status, 'value') else t.status,
            "deadline_type": t.deadline_type.value if hasattr(t.deadline_type, 'value') else t.deadline_type,
            "deadline": t.deadline,
            "energy_level": t.energy_level.value if hasattr(t.energy_level, 'value') else t.energy_level,
        }
        for t in tasks
    ]

    return {
        "success": True,
        "count": len(tasks_info),
        "tasks": tasks_info,
    }


@mcp.tool()
def add_commitment(title: str, start_time: str, end_time: str,
                   commitment_type: str = "other",
                   recurrence: str = "none") -> dict:
    """添加固定承诺（课程、会议、活动等）。

    固定承诺的时间不会被任务占用。

    Args:
        title: 承诺标题（如"高等数学课"）
        start_time: 开始时间，ISO datetime 格式
        end_time: 结束时间，ISO datetime 格式
        commitment_type: 类型：class / meeting / activity / rest / other
        recurrence: 重复：none / daily / weekly / monthly

    Returns:
        添加结果
    """
    agent = get_agent()

    valid_types = {t.value for t in CommitmentType}
    if commitment_type not in valid_types:
        commitment_type = "other"
    if recurrence not in {"none", "daily", "weekly"}:
        recurrence = "none"

    commitment = Commitment(
        profile_id=agent.profile.id or "default",
        title=title,
        type=commitment_type,
        start_time=start_time,
        end_time=end_time,
        recurrence=recurrence,
    )
    agent.add_commitment(commitment)

    # 直接强制重排，不依赖自然语言意图
    result = agent.force_replan(reason=f"新增固定承诺：{title}")

    return {
        "success": True,
        "commitment_id": commitment.id,
        "message": f"已添加承诺：{title}",
        "plan_updated": result.changed,
        "total_minutes": result.snapshot.total_scheduled_minutes if result.snapshot else 0,
    }


@mcp.tool()
def chat(message: str) -> dict:
    """和 LifeOS 自然语言对话。

    你可以说任何话：设定目标、汇报进度、询问状态、分享生活事件...
    LifeOS 会自动理解并调整计划。

    Args:
        message: 你的自然语言消息

    Returns:
        LifeOS 的回复和相关信息
    """
    agent = get_agent()
    result = agent.run(message)

    response = {
        "reply": result.reply,
        "plan_changed": result.changed,
        "intent": result.intent.value if hasattr(result.intent, 'value') else result.intent,
        "has_snapshot": result.snapshot is not None,
    }

    if result.snapshot:
        response["total_minutes"] = result.snapshot.total_scheduled_minutes
        response["task_count"] = len(agent._tasks)
        response["sacrifice_count"] = len(result.snapshot.sacrifice_list)

    if result.change_log:
        response["change_reason"] = result.change_log.reason
        response["change_summary"] = result.change_log.summary

    return response


# ==========================================
# Echo 工具（兼容 D1）
# ==========================================

@mcp.tool()
def echo(message: str) -> str:
    """回显输入的消息，用于测试 MCP 通信是否正常。

    Args:
        message: 要回显的消息内容

    Returns:
        原样返回的消息，附带 "ECHO: " 前缀
    """
    return f"ECHO: {message}"


@mcp.tool()
def system_now() -> dict:
    """获取当前系统时间。

    Returns:
        包含当前时间和时区的字典
    """
    return {
        "current_time": datetime.now().isoformat(),
        "timezone": os.environ.get("APP_TIMEZONE", "Asia/Shanghai"),
    }


# ==========================================
# 资源
# ==========================================

@mcp.resource("lifeops://status")
def get_status() -> str:
    """LifeOS MCP 服务状态。"""
    agent = get_agent()
    task_count = len(agent._tasks)
    goal_count = len(agent._goals)
    snapshot_count = len(agent._snapshots)
    return (f"LifeOS MCP Server is running. "
            f"Goals: {goal_count}, Tasks: {task_count}, Snapshots: {snapshot_count}. "
            f"Tools: create_goal, report_progress, get_current_plan, get_task_list, "
            f"add_commitment, chat, echo, system_now")


# ==========================================
# 入口
# ==========================================

def main():
    """启动 MCP stdio 服务器。"""
    # 加载 .env
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    print("[lifeops-mcp] 服务器启动（stdio 模式）", file=sys.stderr)
    print(f"[lifeops-mcp] 可用工具：create_goal, report_progress, get_current_plan, "
          f"get_task_list, add_commitment, chat, echo, system_now",
          file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

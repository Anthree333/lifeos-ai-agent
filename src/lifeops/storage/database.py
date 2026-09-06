"""SQLite 数据库管理 + CRUD 封装。"""
from __future__ import annotations

import os
import sqlite3
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..models import (
    StudentProfile, Goal, Task, Commitment,
    TimeSlot, PlanSnapshot, ChangeLog, ChangeLogItem, SacrificeItem,
    ExecutionRecord, LifeEvent, RiskReport,
    TaskStatus, DeadlineType, EnergyLevel, CommitmentType,
    EventType, RiskLevel,
)


SCHEMA_SQL = """
-- 学生档案
CREATE TABLE IF NOT EXISTS profile (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '学生',
    daily_high_energy_hours INTEGER NOT NULL DEFAULT 4,
    wake_up_time TEXT NOT NULL DEFAULT '07:00',
    sleep_time TEXT NOT NULL DEFAULT '23:00',
    default_buffer_minutes INTEGER NOT NULL DEFAULT 10,
    min_progress_block_minutes INTEGER NOT NULL DEFAULT 20
);

-- 目标
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    weight REAL NOT NULL DEFAULT 0.5,
    deadline TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    FOREIGN KEY (profile_id) REFERENCES profile(id)
);

-- 任务
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    parent_id TEXT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    estimated_minutes INTEGER NOT NULL DEFAULT 60,
    actual_minutes INTEGER NOT NULL DEFAULT 0,
    energy_level TEXT NOT NULL DEFAULT 'medium',
    deadline_type TEXT NOT NULL DEFAULT 'soft',
    deadline TEXT,
    earliest_start_time TEXT,
    priority_weight REAL NOT NULL DEFAULT 0.5,
    progress REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending',
    dependencies TEXT NOT NULL DEFAULT '[]',
    order_index INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (goal_id) REFERENCES goals(id)
);

-- 固定承诺
CREATE TABLE IF NOT EXISTS commitments (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    title TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'other',
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    recurrence TEXT NOT NULL DEFAULT 'none',
    description TEXT,
    FOREIGN KEY (profile_id) REFERENCES profile(id)
);

-- 时间槽
CREATE TABLE IF NOT EXISTS time_slots (
    id TEXT PRIMARY KEY,
    snapshot_id TEXT,
    task_id TEXT,
    commitment_id TEXT,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    energy_level TEXT NOT NULL DEFAULT 'medium',
    is_fixed INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'task',
    title TEXT NOT NULL DEFAULT ''
);

-- 执行记录
CREATE TABLE IF NOT EXISTS executions (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    record_date TEXT NOT NULL,
    actual_minutes INTEGER NOT NULL DEFAULT 0,
    progress_delta REAL NOT NULL DEFAULT 0.0,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

-- 生活事件
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    event_type TEXT NOT NULL DEFAULT 'other',
    event_time TEXT,
    end_time TEXT,
    change_content TEXT NOT NULL DEFAULT '',
    source_image_hash TEXT,
    user_confirmed INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    raw_data TEXT NOT NULL DEFAULT '{}'
);

-- 计划快照
CREATE TABLE IF NOT EXISTS plan_snapshots (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    trigger_reason TEXT NOT NULL DEFAULT '',
    hard_deadline_count INTEGER NOT NULL DEFAULT 0,
    risk_count INTEGER NOT NULL DEFAULT 0,
    change_log_json TEXT NOT NULL DEFAULT '{}',
    schedule_tracks TEXT NOT NULL DEFAULT '{}',
    sacrifice_list_json TEXT NOT NULL DEFAULT '[]'
);

-- 风险报告
CREATE TABLE IF NOT EXISTS risk_reports (
    id TEXT PRIMARY KEY,
    snapshot_id TEXT,
    task_id TEXT NOT NULL,
    task_title TEXT NOT NULL DEFAULT '',
    risk_level TEXT NOT NULL DEFAULT 'low',
    estimated_finish_date TEXT,
    deadline TEXT,
    delay_days REAL NOT NULL DEFAULT 0.0,
    remaining_minutes INTEGER NOT NULL DEFAULT 0,
    avg_daily_minutes REAL NOT NULL DEFAULT 0.0,
    suggestion_type TEXT NOT NULL DEFAULT 'advance',
    suggestion_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

-- LLM 缓存
CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    prompt_preview TEXT NOT NULL DEFAULT '',
    response TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL
);

-- 视觉缓存
CREATE TABLE IF NOT EXISTS vision_cache (
    image_hash TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    result TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL
);

-- 工具调用记录
CREATE TABLE IF NOT EXISTS tool_call_records (
    id TEXT PRIMARY KEY,
    snapshot_id TEXT,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL DEFAULT '{}',
    result TEXT NOT NULL DEFAULT '{}',
    duration_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'success',
    error_message TEXT,
    created_at TEXT NOT NULL
);

-- 知识库文档
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL,
    last_accessed_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

-- 待确认动作（对话状态，进程重启后仍可恢复）
CREATE TABLE IF NOT EXISTS pending_actions (
    profile_id TEXT PRIMARY KEY,
    action_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 学习笔记
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_tasks_goal_id ON tasks(goal_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_time_slots_snapshot_id ON time_slots(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_executions_task_id ON executions(task_id);
CREATE INDEX IF NOT EXISTS idx_risk_reports_snapshot_id ON risk_reports(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_goals_profile_id ON goals(profile_id);
CREATE INDEX IF NOT EXISTS idx_commitments_profile_id ON commitments(profile_id);
CREATE INDEX IF NOT EXISTS idx_plan_snapshots_profile_id ON plan_snapshots(profile_id);
"""


class Database:
    """SQLite 数据库连接管理 + CRUD。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._connect()
        return self._conn

    def _connect(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def init_schema(self) -> None:
        """初始化数据库表结构。"""
        self.conn.executescript(SCHEMA_SQL)
        # 轻量列迁移：旧库可能缺少 earliest_start_time 列
        self._ensure_column("tasks", "earliest_start_time", "TEXT")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, col_type: str) -> None:
        """如果列不存在则添加（兼容旧数据库）。"""
        cols = {
            row[1]
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in cols:
            self.conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
            )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ==========================================
    # Profile
    # ==========================================

    def get_profile(self, profile_id: str) -> Optional[StudentProfile]:
        row = self.conn.execute(
            "SELECT * FROM profile WHERE id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_profile(row)

    def get_default_profile(self) -> StudentProfile:
        """获取默认档案（第一个），不存在则创建。"""
        row = self.conn.execute("SELECT * FROM profile LIMIT 1").fetchone()
        if row is None:
            profile = StudentProfile(id="default", name="默认学生")
            self.save_profile(profile)
            return profile
        return self._row_to_profile(row)

    def save_profile(self, profile: StudentProfile) -> None:
        now = datetime.now().isoformat()
        if not profile.id:
            profile.id = str(uuid.uuid4())

        self.conn.execute(
            """INSERT OR REPLACE INTO profile
               (id, name, daily_high_energy_hours, wake_up_time, sleep_time,
                default_buffer_minutes, min_progress_block_minutes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                profile.id,
                profile.name,
                profile.daily_high_energy_hours,
                profile.wake_up_time.isoformat(),
                profile.sleep_time.isoformat(),
                profile.default_buffer_minutes,
                profile.min_progress_block_minutes,
            ),
        )
        self.conn.commit()

    def _row_to_profile(self, row: sqlite3.Row) -> StudentProfile:
        from datetime import time as dt_time

        def _parse_time(val):
            if isinstance(val, str):
                parts = val.split(":")
                return dt_time(int(parts[0]), int(parts[1]))
            return val

        return StudentProfile(
            id=row["id"],
            name=row["name"],
            daily_high_energy_hours=row["daily_high_energy_hours"],
            wake_up_time=_parse_time(row["wake_up_time"]),
            sleep_time=_parse_time(row["sleep_time"]),
            default_buffer_minutes=row["default_buffer_minutes"],
            min_progress_block_minutes=row["min_progress_block_minutes"],
        )

    # ==========================================
    # Goal
    # ==========================================

    def list_goals(self, profile_id: str) -> List[Goal]:
        rows = self.conn.execute(
            "SELECT * FROM goals WHERE profile_id = ? ORDER BY created_at DESC",
            (profile_id,),
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        row = self.conn.execute(
            "SELECT * FROM goals WHERE id = ?", (goal_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_goal(row)

    def save_goal(self, goal: Goal) -> None:
        now = datetime.now().isoformat()
        if not goal.id:
            goal.id = str(uuid.uuid4())
        if not goal.created_at:
            goal.created_at = now

        self.conn.execute(
            """INSERT OR REPLACE INTO goals
               (id, profile_id, title, description, weight, deadline, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                goal.id,
                goal.profile_id,
                goal.title,
                goal.description,
                goal.weight,
                goal.deadline,
                goal.status.value if hasattr(goal.status, 'value') else goal.status,
                goal.created_at,
            ),
        )
        self.conn.commit()

    def delete_goal(self, goal_id: str) -> None:
        """删除目标（同时级联删除其下属任务及任务执行记录）。"""
        # 先找出该目标下所有任务 id
        rows = self.conn.execute(
            "SELECT id FROM tasks WHERE goal_id = ?", (goal_id,)
        ).fetchall()
        for row in rows:
            task_id = row["id"]
            # 删除该任务的执行记录（executions.task_id 引用 tasks.id）
            self.conn.execute(
                "DELETE FROM executions WHERE task_id = ?", (task_id,)
            )
        # 再删 tasks，最后删 goals
        self.conn.execute("DELETE FROM tasks WHERE goal_id = ?", (goal_id,))
        self.conn.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
        self.conn.commit()

    def _row_to_goal(self, row: sqlite3.Row) -> Goal:
        return Goal(
            id=row["id"],
            profile_id=row["profile_id"],
            title=row["title"],
            description=row["description"],
            weight=row["weight"],
            deadline=row["deadline"],
            status=row["status"],
            created_at=row["created_at"],
        )

    # ==========================================
    # Task
    # ==========================================

    def list_tasks(self, goal_id: Optional[str] = None,
                   profile_id: Optional[str] = None) -> List[Task]:
        if goal_id:
            rows = self.conn.execute(
                "SELECT * FROM tasks WHERE goal_id = ? ORDER BY order_index, created_at",
                (goal_id,),
            ).fetchall()
        elif profile_id:
            rows = self.conn.execute(
                """SELECT t.* FROM tasks t
                   JOIN goals g ON t.goal_id = g.id
                   WHERE g.profile_id = ?
                   ORDER BY t.order_index, t.created_at""",
                (profile_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM tasks ORDER BY order_index, created_at"
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def get_task(self, task_id: str) -> Optional[Task]:
        row = self.conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def save_task(self, task: Task) -> None:
        now = datetime.now().isoformat()
        if not task.id:
            task.id = str(uuid.uuid4())
        if not hasattr(task, 'created_at') or not task.created_at:
            task.created_at = now
        task.updated_at = now

        deps_json = json.dumps(task.dependencies, ensure_ascii=False)

        self.conn.execute(
            """INSERT OR REPLACE INTO tasks
               (id, goal_id, parent_id, title, description, estimated_minutes,
                actual_minutes, energy_level, deadline_type, deadline,
                earliest_start_time,
                priority_weight, progress, status, dependencies, order_index,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.id,
                task.goal_id,
                task.parent_id,
                task.title,
                task.description,
                task.estimated_minutes,
                task.actual_minutes if hasattr(task, 'actual_minutes') else 0,
                task.energy_level.value if hasattr(task.energy_level, 'value') else task.energy_level,
                task.deadline_type.value if hasattr(task.deadline_type, 'value') else task.deadline_type,
                task.deadline,
                getattr(task, 'earliest_start_time', None),
                task.priority_weight,
                task.progress,
                task.status.value if hasattr(task.status, 'value') else task.status,
                deps_json,
                task.order_index,
                task.created_at,
                task.updated_at,
            ),
        )
        self.conn.commit()

    def save_tasks(self, tasks: List[Task]) -> None:
        """批量保存任务。"""
        for task in tasks:
            self.save_task(task)

    def delete_task(self, task_id: str) -> bool:
        """删除单个任务（同时删除其执行记录），返回是否真的删掉了。"""
        # executions.task_id 引用 tasks.id，先清执行记录
        self.conn.execute("DELETE FROM executions WHERE task_id = ?", (task_id,))
        cursor = self.conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def update_task_progress(self, task_id: str, progress: float,
                             actual_minutes: int = 0) -> None:
        """更新任务进度。"""
        now = datetime.now().isoformat()
        status = "completed" if progress >= 1.0 else "in_progress" if progress > 0 else "pending"
        self.conn.execute(
            """UPDATE tasks SET progress = ?, status = ?,
               actual_minutes = actual_minutes + ?, updated_at = ?
               WHERE id = ?""",
            (progress, status, actual_minutes, now, task_id),
        )
        self.conn.commit()

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        deps = json.loads(row["dependencies"]) if row["dependencies"] else []
        return Task(
            id=row["id"],
            goal_id=row["goal_id"],
            parent_id=row["parent_id"],
            title=row["title"],
            description=row["description"],
            estimated_minutes=row["estimated_minutes"],
            actual_minutes=row["actual_minutes"] if "actual_minutes" in row.keys() else 0,
            energy_level=row["energy_level"],
            deadline_type=row["deadline_type"],
            deadline=row["deadline"],
            earliest_start_time=row["earliest_start_time"] if "earliest_start_time" in row.keys() else None,
            priority_weight=row["priority_weight"],
            progress=row["progress"],
            status=row["status"],
            dependencies=deps,
            order_index=row["order_index"],
            created_at=row["created_at"] if "created_at" in row.keys() else None,
            updated_at=row["updated_at"] if "updated_at" in row.keys() else None,
        )

    # ==========================================
    # Commitment
    # ==========================================

    def list_commitments(self, profile_id: str) -> List[Commitment]:
        rows = self.conn.execute(
            "SELECT * FROM commitments WHERE profile_id = ? ORDER BY start_time",
            (profile_id,),
        ).fetchall()
        return [self._row_to_commitment(r) for r in rows]

    def save_commitment(self, commitment: Commitment) -> None:
        if not commitment.id:
            commitment.id = str(uuid.uuid4())

        self.conn.execute(
            """INSERT OR REPLACE INTO commitments
               (id, profile_id, title, type, start_time, end_time,
                recurrence, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                commitment.id,
                commitment.profile_id,
                commitment.title,
                commitment.type.value if hasattr(commitment.type, 'value') else commitment.type,
                commitment.start_time,
                commitment.end_time,
                commitment.recurrence.value if hasattr(commitment.recurrence, 'value') else commitment.recurrence,
                commitment.description,
            ),
        )
        self.conn.commit()

    def delete_commitment(self, commitment_id: str) -> bool:
        """删除一个固定承诺/课程。"""
        cursor = self.conn.execute(
            "DELETE FROM commitments WHERE id = ?",
            (commitment_id,),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def _row_to_commitment(self, row: sqlite3.Row) -> Commitment:
        return Commitment(
            id=row["id"],
            profile_id=row["profile_id"],
            title=row["title"],
            type=row["type"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            recurrence=row["recurrence"],
            description=row["description"],
        )

    # ==========================================
    # Plan Snapshot
    # ==========================================

    def save_snapshot(self, snapshot: PlanSnapshot) -> None:
        """保存计划快照（含时间槽和牺牲清单）。"""
        if not snapshot.id:
            snapshot.id = str(uuid.uuid4())

        change_log_json = "{}"
        if snapshot.change_log:
            cl = snapshot.change_log
            change_log_json = json.dumps({
                "trigger_reason": cl.trigger_reason,
                "summary": cl.summary,
                "detail": cl.detail,
                "reason": cl.reason,
                "affected_tasks": cl.affected_tasks,
                "items": [
                    {
                        "task_id": it.task_id,
                        "task_title": it.task_title,
                        "action": it.action.value if hasattr(it.action, "value") else it.action,
                        "old_time": it.old_time,
                        "new_time": it.new_time,
                        "old_duration": it.old_duration,
                        "new_duration": it.new_duration,
                        "reason": it.reason,
                    }
                    for it in cl.items
                ] if cl.items else [],
                "sacrifice_list": [
                    {
                        "task_id": s.task_id,
                        "task_title": s.task_title,
                        "reason": s.reason,
                        "priority_before": s.priority_before,
                        "action": s.action,
                        "delayed_to": s.delayed_to,
                    }
                    for s in cl.sacrifice_list
                ] if cl.sacrifice_list else [],
            }, ensure_ascii=False)

        sacrifice_json = json.dumps([
            {"task_id": s.task_id, "task_title": s.task_title,
             "reason": s.reason, "priority_before": s.priority_before,
             "action": s.action, "delayed_to": s.delayed_to}
            for s in snapshot.sacrifice_list
        ], ensure_ascii=False)

        schedule_tracks = json.dumps(
            dict(snapshot.schedule_tracks), ensure_ascii=False
        ) if snapshot.schedule_tracks else "{}"

        self.conn.execute(
            """INSERT OR REPLACE INTO plan_snapshots
               (id, profile_id, version, created_at, trigger_reason,
                hard_deadline_count, risk_count, change_log_json,
                schedule_tracks, sacrifice_list_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot.id,
                snapshot.profile_id,
                snapshot.version,
                snapshot.created_at,
                snapshot.trigger_reason,
                snapshot.hard_deadline_count,
                snapshot.risk_count,
                change_log_json,
                schedule_tracks,
                sacrifice_json,
            ),
        )

        # 保存时间槽
        for ts in snapshot.time_slots:
            self.conn.execute(
                """INSERT OR REPLACE INTO time_slots
                   (id, snapshot_id, task_id, commitment_id, start_time,
                    end_time, energy_level, is_fixed, source, title)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts.id,
                    snapshot.id,
                    ts.task_id,
                    ts.commitment_id,
                    ts.start_time,
                    ts.end_time,
                    ts.energy_level.value if hasattr(ts.energy_level, 'value') else ts.energy_level,
                    1 if ts.is_fixed else 0,
                    ts.source.value if hasattr(ts.source, 'value') else ts.source,
                    ts.title,
                ),
            )

        self.conn.commit()

    def get_latest_snapshot(self, profile_id: str) -> Optional[PlanSnapshot]:
        row = self.conn.execute(
            """SELECT * FROM plan_snapshots
               WHERE profile_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (profile_id,),
        ).fetchone()
        if row is None:
            return None
        return self._load_snapshot_with_slots(row)

    def list_snapshots(self, profile_id: str, limit: int = 10) -> List[PlanSnapshot]:
        rows = self.conn.execute(
            """SELECT * FROM plan_snapshots
               WHERE profile_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (profile_id, limit),
        ).fetchall()
        return [self._load_snapshot_with_slots(r) for r in rows]

    def clear_all_snapshots(self, profile_id: Optional[str] = None) -> int:
        """删除所有历史计划快照（及其 time_slots / risk_reports），返回删除条数。

        若指定 profile_id 则只删该用户的；否则删全部。
        当前正在使用的快照不在快照表中删除后不会影响 agent.current_snapshot（内存对象），
        下次重排会重建。
        """
        if profile_id:
            ids = [r[0] for r in self.conn.execute(
                "SELECT id FROM plan_snapshots WHERE profile_id = ?", (profile_id,)
            )]
        else:
            ids = [r[0] for r in self.conn.execute("SELECT id FROM plan_snapshots")]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        # 先删依赖快照的从表，再删主表
        self.conn.execute(
            f"DELETE FROM time_slots WHERE snapshot_id IN ({placeholders})", ids
        )
        self.conn.execute(
            f"DELETE FROM risk_reports WHERE snapshot_id IN ({placeholders})", ids
        )
        if profile_id:
            self.conn.execute(
                "DELETE FROM plan_snapshots WHERE profile_id = ?", (profile_id,)
            )
        else:
            self.conn.execute("DELETE FROM plan_snapshots")
        self.conn.commit()
        return len(ids)

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除单条历史快照（含其 time_slots / risk_reports）。"""
        existed = self.conn.execute(
            "SELECT 1 FROM plan_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if not existed:
            return False
        self.conn.execute(
            "DELETE FROM time_slots WHERE snapshot_id = ?", (snapshot_id,)
        )
        self.conn.execute(
            "DELETE FROM risk_reports WHERE snapshot_id = ?", (snapshot_id,)
        )
        self.conn.execute(
            "DELETE FROM plan_snapshots WHERE id = ?", (snapshot_id,)
        )
        self.conn.commit()
        return True

    def _load_snapshot_with_slots(self, row: sqlite3.Row) -> PlanSnapshot:
        snapshot = PlanSnapshot(
            id=row["id"],
            profile_id=row["profile_id"],
            version=row["version"],
            created_at=row["created_at"],
            trigger_reason=row["trigger_reason"],
            hard_deadline_count=row["hard_deadline_count"],
            risk_count=row["risk_count"],
            schedule_tracks=json.loads(row["schedule_tracks"]) if row["schedule_tracks"] else {},
        )

        # 加载时间槽
        slot_rows = self.conn.execute(
            "SELECT * FROM time_slots WHERE snapshot_id = ? ORDER BY start_time",
            (snapshot.id,),
        ).fetchall()
        snapshot.time_slots = [self._row_to_timeslot(r) for r in slot_rows]

        # 加载牺牲清单
        sacrifice_data = json.loads(row["sacrifice_list_json"]) if row["sacrifice_list_json"] else []
        snapshot.sacrifice_list = [
            SacrificeItem(
                task_id=s["task_id"],
                task_title=s["task_title"],
                reason=s["reason"],
                priority_before=s["priority_before"],
                action=s.get("action", "delayed"),
                delayed_to=s.get("delayed_to"),
            )
            for s in sacrifice_data
        ]

        # 加载变更日志（兼容旧数据：缺 items / sacrifice_list 字段时留空）
        if row["change_log_json"]:
            cl_data = json.loads(row["change_log_json"])
            snapshot.change_log = ChangeLog(
                trigger_reason=cl_data.get("trigger_reason", ""),
                summary=cl_data.get("summary", ""),
                detail=cl_data.get("detail", ""),
                reason=cl_data.get("reason", ""),
                affected_tasks=cl_data.get("affected_tasks", []),
                items=[
                    ChangeLogItem(
                        task_id=it.get("task_id", ""),
                        task_title=it.get("task_title", ""),
                        action=it.get("action", "kept"),
                        old_time=it.get("old_time"),
                        new_time=it.get("new_time"),
                        old_duration=it.get("old_duration"),
                        new_duration=it.get("new_duration"),
                        reason=it.get("reason", ""),
                    )
                    for it in cl_data.get("items", [])
                ],
                sacrifice_list=[
                    SacrificeItem(
                        task_id=s.get("task_id", ""),
                        task_title=s.get("task_title", ""),
                        reason=s.get("reason", ""),
                        priority_before=s.get("priority_before", 0.0),
                        action=s.get("action", "delayed"),
                        delayed_to=s.get("delayed_to"),
                    )
                    for s in cl_data.get("sacrifice_list", [])
                ],
            )

        return snapshot

    def _row_to_timeslot(self, row: sqlite3.Row) -> TimeSlot:
        return TimeSlot(
            id=row["id"],
            task_id=row["task_id"],
            commitment_id=row["commitment_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            energy_level=row["energy_level"],
            is_fixed=bool(row["is_fixed"]),
            source=row["source"],
            title=row["title"],
        )

    # ==========================================
    # Execution Record
    # ==========================================

    def add_execution_record(self, record: ExecutionRecord) -> None:
        if not record.id:
            record.id = str(uuid.uuid4())
        if not record.created_at:
            record.created_at = datetime.now().isoformat()

        self.conn.execute(
            """INSERT INTO executions
               (id, task_id, record_date, actual_minutes, progress_delta, note, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.task_id,
                record.record_date,
                record.actual_minutes,
                record.progress_delta,
                record.note,
                record.created_at,
            ),
        )
        self.conn.commit()

    def list_executions(self, task_id: str) -> List[ExecutionRecord]:
        rows = self.conn.execute(
            "SELECT * FROM executions WHERE task_id = ? ORDER BY created_at DESC",
            (task_id,),
        ).fetchall()
        return [self._row_to_execution(r) for r in rows]

    def _row_to_execution(self, row: sqlite3.Row) -> ExecutionRecord:
        return ExecutionRecord(
            id=row["id"],
            task_id=row["task_id"],
            record_date=row["record_date"],
            actual_minutes=row["actual_minutes"],
            progress_delta=row["progress_delta"],
            note=row["note"],
            created_at=row["created_at"],
        )

    # ==========================================
    # Life Event
    # ==========================================

    def add_event(self, event: LifeEvent) -> None:
        if not event.id:
            event.id = str(uuid.uuid4())
        if not event.created_at:
            event.created_at = datetime.now().isoformat()

        self.conn.execute(
            """INSERT INTO events
               (id, title, event_type, event_time, end_time, change_content,
                source_image_hash, user_confirmed, confidence, created_at, raw_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.id,
                event.title,
                event.event_type.value if hasattr(event.event_type, 'value') else event.event_type,
                event.event_time,
                event.end_time,
                event.change_content,
                event.source_image_hash,
                1 if event.user_confirmed else 0,
                event.confidence,
                event.created_at,
                json.dumps(event.raw_data, ensure_ascii=False) if event.raw_data else "{}",
            ),
        )
        self.conn.commit()

    # ==========================================
    # Pending Action（待确认对话状态）
    # ==========================================

    def save_pending_action(self, profile_id: str, payload: Dict[str, Any]) -> None:
        """保存当前待确认动作（每个 profile 仅保留一条，覆盖旧值）。"""
        now = datetime.now().isoformat()
        action_json = json.dumps(payload, ensure_ascii=False)
        self.conn.execute(
            """INSERT OR REPLACE INTO pending_actions (profile_id, action_json, created_at)
               VALUES (?, ?, ?)""",
            (profile_id, action_json, now),
        )
        self.conn.commit()

    def load_pending_action(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """读取最近一条待确认动作，无则返回 None。"""
        row = self.conn.execute(
            "SELECT action_json FROM pending_actions WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["action_json"])
        except (json.JSONDecodeError, TypeError):
            return None

    def clear_pending_action(self, profile_id: str) -> None:
        """清除待确认动作。"""
        self.conn.execute(
            "DELETE FROM pending_actions WHERE profile_id = ?", (profile_id,)
        )
        self.conn.commit()

    # ==========================================
    # Knowledge Documents
    # ==========================================

    def add_knowledge_document(self, doc: Any, chunk_count: int) -> None:
        """插入知识库文档元数据。

        使用 duck typing：doc 需具有 id, title, file_path, file_type, file_size, metadata 属性。
        """
        now = datetime.now().isoformat()
        metadata_json = json.dumps(doc.metadata, ensure_ascii=False) if doc.metadata else "{}"

        self.conn.execute(
            """INSERT OR REPLACE INTO knowledge_documents
               (id, title, file_path, file_type, file_size, chunk_count,
                uploaded_at, last_accessed_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.id,
                doc.title,
                doc.file_path,
                doc.file_type,
                doc.file_size,
                chunk_count,
                now,
                now,
                metadata_json,
            ),
        )
        self.conn.commit()

    def delete_knowledge_document(self, doc_id: str) -> None:
        """删除知识库文档。"""
        self.conn.execute(
            "DELETE FROM knowledge_documents WHERE id = ?",
            (doc_id,),
        )
        self.conn.commit()

    def list_knowledge_documents(self) -> List[dict]:
        """列出所有知识库文档，返回 dict 列表。

        每个 dict 包含：id, title, file_type, file_size, chunk_count, uploaded_at
        """
        rows = self.conn.execute(
            """SELECT id, title, file_type, file_size, chunk_count, uploaded_at
               FROM knowledge_documents ORDER BY uploaded_at DESC"""
        ).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "file_type": row["file_type"],
                "file_size": row["file_size"],
                "chunk_count": row["chunk_count"],
                "uploaded_at": row["uploaded_at"],
            }
            for row in rows
        ]

    # ==========================================
    # Notes（学习笔记）
    # ==========================================

    def save_note(self, note_id: Optional[str] = None, title: str = "",
                  content: str = "", tags: Optional[List[str]] = None,
                  pinned: bool = False) -> str:
        """新增或整体覆盖一篇笔记，返回笔记 id。"""
        now = datetime.now().isoformat()
        nid = note_id or str(uuid.uuid4())
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        self.conn.execute(
            """INSERT INTO notes (id, title, content, tags, pinned, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, content=excluded.content,
                 tags=excluded.tags, pinned=excluded.pinned,
                 updated_at=excluded.updated_at""",
            (nid, title, content, tags_json, 1 if pinned else 0, now, now),
        )
        self.conn.commit()
        return nid

    def update_note(self, note_id: str, **fields: Any) -> bool:
        """局部更新笔记字段（title/content/tags/pinned），自动刷新 updated_at。

        返回是否真的有笔记被更新（笔记不存在时返回 False）。
        """
        allowed = {"title", "content", "tags", "pinned"}
        sets: List[str] = []
        vals: List[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "tags":
                value = json.dumps(list(value or []), ensure_ascii=False)
            elif key == "pinned":
                value = 1 if value else 0
            sets.append(f"{key}=?")
            vals.append(value)
        if not sets:
            return False
        sets.append("updated_at=?")
        vals.append(datetime.now().isoformat())
        vals.append(note_id)
        cur = self.conn.execute(
            f"UPDATE notes SET {', '.join(sets)} WHERE id=?", vals
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_note(self, note_id: str) -> bool:
        """删除一篇笔记，返回是否存在并被删除。"""
        cur = self.conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def list_notes(self) -> List[dict]:
        """列出全部笔记：置顶优先、最近更新在前。

        每个 dict 包含：id, title, content, tags(列表), pinned, created_at, updated_at
        """
        rows = self.conn.execute(
            """SELECT id, title, content, tags, pinned, created_at, updated_at
               FROM notes ORDER BY pinned DESC, updated_at DESC"""
        ).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "tags": json.loads(row["tags"] or "[]"),
                "pinned": bool(row["pinned"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    # ==========================================
    # LLM Cache
    # ==========================================

    def get_llm_cache(self, prompt_hash: str, model: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT response FROM llm_cache WHERE prompt_hash = ? AND model = ?",
            (prompt_hash, model),
        ).fetchone()
        if row is None:
            return None

        # 更新命中计数
        self.conn.execute(
            """UPDATE llm_cache SET hit_count = hit_count + 1,
               last_used_at = ? WHERE prompt_hash = ? AND model = ?""",
            (datetime.now().isoformat(), prompt_hash, model),
        )
        self.conn.commit()
        return row["response"]

    def set_llm_cache(self, prompt_hash: str, model: str,
                      prompt_preview: str, response: str) -> None:
        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO llm_cache
               (prompt_hash, model, prompt_preview, response, hit_count,
                created_at, last_used_at)
               VALUES (?, ?, ?, ?, 1, ?, ?)""",
            (prompt_hash, model, prompt_preview, response, now, now),
        )
        self.conn.commit()


# 全局实例
_db_instance: Optional[Database] = None


def get_db(db_path: Optional[str] = None) -> Database:
    """获取数据库单例（每次调用都会幂等执行建表语句）。

    代码热更新（服务器未重启）或磁盘上为旧库文件时，缓存连接里的 schema
    可能缺少新表；这里每次调用都重跑 init_schema 自动补齐，避免
    “no such table”一类的运行时错误。
    """
    global _db_instance
    if _db_instance is None:
        if db_path is None:
            db_path = os.environ.get("DB_PATH", "./data/lifeos.db")
        _db_instance = Database(db_path)
    # 幂等：全部为 CREATE TABLE/INDEX IF NOT EXISTS，重复执行无副作用、开销极小
    _db_instance.init_schema()
    return _db_instance

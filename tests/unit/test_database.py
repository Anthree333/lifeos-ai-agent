"""Database 通用数据表存储测试（学习笔记等）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lifeops.storage.database import Database


def _db(tmp_path) -> Database:
    db = Database(str(tmp_path / "lifeos.db"))
    db.init_schema()
    return db


class TestNotes:
    def test_save_list_update_delete(self, tmp_path):
        db = _db(tmp_path)
        nid = db.save_note(title="高数错题", content="泰勒公式展开", tags=["高数", "错题"])
        notes = db.list_notes()
        assert len(notes) == 1
        assert notes[0]["id"] == nid
        assert notes[0]["title"] == "高数错题"
        assert notes[0]["content"] == "泰勒公式展开"
        assert notes[0]["tags"] == ["高数", "错题"]
        assert notes[0]["pinned"] is False

        # 局部更新不影响其它字段
        assert db.update_note(nid, title="高数错题整理")
        row = db.list_notes()[0]
        assert row["title"] == "高数错题整理"
        assert row["content"] == "泰勒公式展开"
        assert row["tags"] == ["高数", "错题"]

        assert db.delete_note(nid)
        assert db.list_notes() == []

    def test_pinned_first_and_update_ordering(self, tmp_path):
        db = _db(tmp_path)
        db.save_note(title="A")
        db.save_note(title="B", pinned=True)
        c = db.save_note(title="C")
        # 置顶优先
        assert db.list_notes()[0]["title"] == "B"
        # 更新 C 后，C 应排到非置顶组第一位
        db.update_note(c, content="刚编辑过")
        titles = [n["title"] for n in db.list_notes()]
        assert titles == ["B", "C", "A"]

    def test_update_or_delete_missing_note(self, tmp_path):
        db = _db(tmp_path)
        assert db.update_note("missing", title="x") is False
        assert db.delete_note("missing") is False

    def test_tags_json_roundtrip_with_unicode(self, tmp_path):
        db = _db(tmp_path)
        nid = db.save_note(title="复习", tags=["英语 四级", "口语", "🎯"])
        assert db.list_notes()[0]["tags"] == ["英语 四级", "口语", "🎯"]

    def test_upsert_overwrites_same_id(self, tmp_path):
        db = _db(tmp_path)
        nid = db.save_note(title="初稿", content="第一版")
        db.save_note(note_id=nid, title="终稿", content="第二版")
        notes = db.list_notes()
        assert len(notes) == 1
        assert notes[0]["title"] == "终稿"


class TestSchemaSelfHeal:
    def test_rerun_schema_restores_missing_notes_table(self, tmp_path):
        """模拟旧库缺 notes 表：重跑 init_schema 后应能自动补齐（get_db 热更新场景）。"""
        db = _db(tmp_path)
        # 人为制造旧库状态：直接删掉 notes 表
        db.conn.execute("DROP TABLE notes")
        db.conn.commit()

        db.init_schema()  # 等价 get_db 每次调用时的幂等建表
        nid = db.save_note(title="迁移后写入", content="ok")
        assert db.list_notes()[0]["id"] == nid


class TestSnapshots:
    def test_snapshot_roundtrip_keeps_change_log_items(self, tmp_path):
        """快照入库→读回后，结构化变更明细（items/牺牲/时段）必须完整还原。"""
        import json
        from lifeops.models.schedule import (
            PlanSnapshot, ChangeLog, ChangeLogItem, SacrificeItem,
            TimeSlot, TimeSlotSource,
        )
        db = _db(tmp_path)
        snap = PlanSnapshot(
            profile_id="p1",
            version=3,
            trigger_reason="progress_report",
            hard_deadline_count=1,
            risk_count=2,
            time_slots=[TimeSlot(
                title="高数复习",
                start_time="2024-03-15T08:00:00",
                end_time="2024-03-15T09:30:00",
                source=TimeSlotSource.TASK,
            )],
        )
        snap.change_log = ChangeLog(
            trigger_reason="progress_report",
            summary="进度更新触发重排",
            detail="高数复习与英语口语冲突，顺延到明天",
            items=[ChangeLogItem(
                task_id="task_1", task_title="高数复习",
                action="moved",
                old_time="2024-03-15T08:00:00",
                new_time="2024-03-16T08:00:00",
                old_duration=90, new_duration=90,
                reason="时间冲突",
            )],
        )
        snap.sacrifice_list = [SacrificeItem(
            task_id="task_9", task_title="背单词",
            reason="让位给高数复习", priority_before=0.3,
            action="delayed", delayed_to="2024-03-16T08:00:00",
        )]
        db.save_snapshot(snap)

        loaded = db.list_snapshots("p1")[0]
        assert loaded.id == snap.id
        assert len(loaded.time_slots) == 1
        assert loaded.hard_deadline_count == 1
        assert loaded.risk_count == 2
        # change_log 结构与字段完整还原（否则重启后明细丢失）
        assert loaded.change_log is not None
        assert loaded.change_log.summary == "进度更新触发重排"
        assert len(loaded.change_log.items) == 1
        item = loaded.change_log.items[0]
        assert item.task_title == "高数复习"
        assert item.action.value == "moved"
        assert item.new_time == "2024-03-16T08:00:00"
        assert item.old_duration == 90
        assert loaded.sacrifice_list[0].task_title == "背单词"

    def test_snapshot_delete_single(self, tmp_path):
        """单条快照删除不影响其它快照。"""
        from lifeops.models.schedule import PlanSnapshot
        db = _db(tmp_path)
        a = PlanSnapshot(profile_id="p1", trigger_reason="new_goal")
        b = PlanSnapshot(profile_id="p1", trigger_reason="new_event")
        db.save_snapshot(a)
        db.save_snapshot(b)
        assert len(db.list_snapshots("p1")) == 2

        assert db.delete_snapshot(a.id) is True
        rest = db.list_snapshots("p1")
        assert len(rest) == 1
        assert rest[0].id == b.id
        # 已删除 / 不存在的 id
        assert db.delete_snapshot(a.id) is False
        assert db.delete_snapshot("missing") is False

    def test_snapshot_legacy_json_without_items(self, tmp_path):
        """旧版本只写 5 个文本字段的 change_log_json 仍能正常读取。"""
        import json
        from lifeops.models.schedule import PlanSnapshot, ChangeLog
        db = _db(tmp_path)
        snap = PlanSnapshot(profile_id="p1", trigger_reason="progress_report")
        snap.change_log = ChangeLog(
            trigger_reason="progress_report", summary="旧摘要", detail="旧详情"
        )
        db.save_snapshot(snap)
        db.conn.execute(
            "UPDATE plan_snapshots SET change_log_json = ? WHERE id = ?",
            (json.dumps({
                "trigger_reason": "progress_report",
                "summary": "旧摘要",
                "detail": "旧详情",
                "reason": "",
                "affected_tasks": [],
            }, ensure_ascii=False), snap.id),
        )
        db.conn.commit()

        loaded = db.list_snapshots("p1")[0]
        assert loaded.change_log is not None
        assert loaded.change_log.summary == "旧摘要"
        assert loaded.change_log.items == []
        assert loaded.change_log.sacrifice_list == []


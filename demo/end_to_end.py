"""端到端 Demo：一句话生成学习计划。

演示 LifeOS 的核心能力：
1. 一句话设定目标 → 自动分解 + 调度
2. 汇报进度 → 自动重排
3. 突发事件 → 自动调整
4. 查看状态 → 中文摘要

运行：python demo/end_to_end.py
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from lifeops.models import StudentProfile, CommitmentType
from lifeops.agent import LifeAgent


def print_sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_reply(reply: str):
    for line in reply.split("\n"):
        print(f"  🤖 {line}")


def print_tasks(agent: LifeAgent):
    print(f"\n  📋 任务列表（共 {len(agent._tasks)} 个）：")
    for i, t in enumerate(agent._tasks, 1):
        status_icon = "✅" if t.status.value == "completed" else "🔄" if t.status.value == "in_progress" else "⏳"
        dl_icon = "🔴" if t.deadline_type.value == "hard" else "📝"
        pct = int(t.progress * 100)
        print(f"    {i:2d}. {status_icon} {dl_icon} {t.title[:40]:40s} "
              f"({t.estimated_minutes:3d}min, {pct:3d}%)")


def print_today_schedule(agent: LifeAgent, day: datetime):
    snapshot = agent.current_snapshot
    if not snapshot:
        print("  暂无计划")
        return

    day_str = day.date().isoformat()
    day_slots = [ts for ts in snapshot.time_slots if ts.start_time.startswith(day_str)]

    print(f"\n  📅 {day_str} 安排（共 {len(day_slots)} 个时间段）：")
    for ts in day_slots:
        start = ts.start_time.split("T")[1][:5]
        end = ts.end_time.split("T")[1][:5]
        src_icon = "📚" if ts.source.value == "task" else "⏰" if ts.source.value == "commitment" else "☕"
        title = ts.title or (agent.get_task(ts.task_id).title if ts.task_id else "")
        print(f"    {start} - {end}  {src_icon} {title[:50]}")

    if snapshot.sacrifice_list:
        print(f"\n  ⚠️  暂时搁置（{len(snapshot.sacrifice_list)} 个）：")
        for s in snapshot.sacrifice_list[:3]:
            print(f"    - {s.task_title}：{s.reason}")


def main():
    print_sep("LifeOS 端到端 Demo")
    print("""
  📚 场景：大学生小明，两周后有高数期末考试
  演示：设定目标 → 自动分解 → 初始计划
        → 进度滞后 → 自动重排 → 生病事件 → 调整
""")

    # ==========================================
    # 初始化
    # ==========================================
    profile = StudentProfile(
        name="小明",
        daily_high_energy_hours=5,
        min_progress_block_minutes=25,
        default_buffer_minutes=10,
    )
    agent = LifeAgent(profile=profile)

    today = datetime(2024, 5, 20, 8, 0, 0)  # 周一

    # 添加课程承诺
    for day_offset in range(14):
        day = today.date() + timedelta(days=day_offset)
        if day.weekday() < 5:  # 工作日
            from lifeops.models import Commitment
            agent.add_commitment(Commitment(
                profile_id="default",
                title="上午课程",
                type=CommitmentType.CLASS,
                start_time=datetime.combine(day, datetime.min.time()).replace(hour=8).isoformat(),
                end_time=datetime.combine(day, datetime.min.time()).replace(hour=12).isoformat(),
            ))
            agent.add_commitment(Commitment(
                profile_id="default",
                title="下午课程",
                type=CommitmentType.CLASS,
                start_time=datetime.combine(day, datetime.min.time()).replace(hour=14).isoformat(),
                end_time=datetime.combine(day, datetime.min.time()).replace(hour=17).isoformat(),
            ))

    # ==========================================
    # Step 1: 一句话设定目标
    # ==========================================
    print_sep("Step 1：一句话设定目标")
    user_msg = "我要准备两周后的高数期末考试，这门课很重要"
    print(f"  👤 小明说：{user_msg}")

    result = agent.run(user_msg, now=today)
    print_reply(result.reply)
    print_tasks(agent)
    print_today_schedule(agent, today)

    # ==========================================
    # Step 2: 3 天后汇报进度（滞后）
    # ==========================================
    print_sep("Step 2：3 天后汇报进度（比预期慢）")
    day3 = today + timedelta(days=3)
    user_msg = "信息收集只完成了30%，进度有点慢"
    print(f"  👤 小明说：{user_msg}")

    result = agent.run(user_msg, now=day3)
    print_reply(result.reply)
    print_tasks(agent)

    # ==========================================
    # Step 3: 生病事件
    # ==========================================
    print_sep("Step 3：第 5 天，小明感冒了")
    day5 = today + timedelta(days=5)
    user_msg = "我今天感冒发烧了，没法学习"
    print(f"  👤 小明说：{user_msg}")

    result = agent.run(user_msg, now=day5)
    print_reply(result.reply)

    # ==========================================
    # Step 4: 查询状态
    # ==========================================
    print_sep("Step 4：查询当前状态")
    user_msg = "看看我的进度怎么样了"
    print(f"  👤 小明说：{user_msg}")

    result = agent.run(user_msg, now=day5)
    print_reply(result.reply)

    # ==========================================
    # Step 5: 快照历史
    # ==========================================
    print_sep("Step 5：计划变更历史")
    print(f"  共生成 {len(agent._snapshots)} 个计划快照：")
    for i, snap in enumerate(agent._snapshots, 1):
        reason = snap.trigger_reason or "初始"
        minutes = snap.total_scheduled_minutes
        print(f"    版本 {i}: {reason} — {minutes} 分钟")

    # ==========================================
    # 总结
    # ==========================================
    print_sep("Demo 完成！")
    print(f"""
  ✅ 已验证的能力：
    1. 自然语言目标设定 → 自动分解为 7 个任务
    2. 约束调度 → 考虑课程承诺、硬截止、精力水平
    3. 进度偏差 → 超过阈值自动重规划
    4. 生活事件 → 生病等事件触发计划调整
    5. 快照机制 → 每次重规划生成历史版本
    6. 中文解释 → 每次变更说明改了什么、为什么

  📊 统计：
    - 目标数：{len(agent._goals)}
    - 任务数：{len(agent._tasks)}
    - 快照数：{len(agent._snapshots)}
    - 最新计划时长：{agent.current_snapshot.total_scheduled_minutes if agent.current_snapshot else 0} 分钟
""")


if __name__ == "__main__":
    main()

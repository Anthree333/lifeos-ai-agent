"""视觉解析 Prompt 模板。

所有 Prompt 均要求视觉模型输出严格 JSON 格式，
便于后续结构化解析和入库。
"""

# 统一的 JSON 输出格式说明（供各 Prompt 复用）
_JSON_OUTPUT_FORMAT = r"""
请严格以 JSON 格式输出，不要有任何额外解释文字。JSON 结构如下：
{
  "type": "exam_schedule|course_table|homework|generic",
  "items": [
    {
      "item_type": "exam|course|homework|task|event|commitment",
      "title": "条目名称/标题",
      "date": "YYYY-MM-DD 或 null",
      "start_time": "HH:MM 或 null",
      "end_time": "HH:MM 或 null",
      "location": "地点或 null",
      "description": "详细描述或 null",
      "deadline_type": "hard|soft|none",
      "recurrence": "none|daily|weekly",
      "day_of_week": "0-6（周一=0）或 null",
      "priority": "1-10 的整数，优先级越高数字越大",
      "estimated_minutes": "预估所需分钟数或 null",
      "energy_level": "high|medium|low"
    }
  ],
  "summary": "一句话总结识别出的全部内容",
  "confidence": "0.0-1.0 的浮点数，表示整体识别置信度"
}

字段说明：
- type: 截图内容类型，自动判断
- item_type: 条目的具体类型
- date: 日期，格式为 YYYY-MM-DD，无法确定则为 null
- start_time / end_time: 起止时间，格式为 HH:MM（24 小时制），无法确定则为 null
- location: 地点，如教室、考场等，无法确定则为 null
- deadline_type: 截止类型，hard 为硬截止（必须按时完成），soft 为软截止（可灵活调整），none 为无截止
- recurrence: 重复模式，none 为不重复，daily 为每天，weekly 为每周
- day_of_week: 星期几，周一=0，周二=1，...，周日=6，仅当 recurrence=weekly 时有效
- priority: 优先级，1-10，默认 5
- estimated_minutes: 预估完成所需分钟数，仅对 homework/task 类有意义
- energy_level: 所需精力档位，high/medium/low
- summary: 对整张截图内容的一句话中文总结
- confidence: 识别置信度，0.0-1.0
"""


GENERIC_PARSE_PROMPT = f"""你是一个专业的截图内容解析助手。请仔细分析这张截图，
自动识别其内容类型（考试安排、课程表、作业清单、通知、日程或其他），
并将其中的关键信息提取为结构化数据。

{_JSON_OUTPUT_FORMAT}

注意事项：
1. 自动判断截图类型，填入 type 字段
2. 尽可能多地提取有价值的条目信息
3. 不确定的字段设为 null 或合理默认值
4. summary 用简洁的中文概括整张截图的内容
5. 所有文字均使用中文
"""


EXAM_SCHEDULE_PROMPT = f"""你是一个专业的考试安排解析助手。请仔细分析这张考试安排截图，
提取每一门考试的详细信息。

{_JSON_OUTPUT_FORMAT}

注意事项：
1. type 固定为 "exam_schedule"
2. 每个 item 的 item_type 固定为 "exam"
3. 考试科目填入 title 字段
4. 考试日期填入 date 字段（YYYY-MM-DD）
5. 考试开始时间填入 start_time，结束时间填入 end_time（HH:MM）
6. 考试地点/考场填入 location 字段
7. deadline_type 统一为 "hard"（考试为硬截止）
8. recurrence 统一为 "none"
9. priority 统一为 9（考试优先级最高）
10. energy_level 统一为 "high"
11. 如有多门考试，每个考试为一个 item
12. summary 用一句话概括考试安排（如"共 X 门考试，时间从 X 到 X"）
"""


COURSE_TABLE_PROMPT = f"""你是一个专业的课程表解析助手。请仔细分析这张课程表截图，
提取每一门课程的详细信息。

{_JSON_OUTPUT_FORMAT}

注意事项：
1. type 固定为 "course_table"
2. 每个 item 的 item_type 固定为 "course"
3. 课程名称填入 title 字段
4. 上课地点填入 location 字段
5. 课程时间：
   - 每周固定的课程，recurrence 设为 "weekly"，day_of_week 填入星期几（周一=0）
   - start_time 和 end_time 填入上课起止时间（HH:MM）
   - 如果知道具体开始日期，填入 date 字段，否则为 null
6. 周数信息（如"第1-8周"）写入 description 字段
7. deadline_type 统一为 "none"
8. priority 统一为 7（课程优先级较高）
9. energy_level 根据课程类型判断，一般为 "medium"
10. 每门课的每个上课时段为一个独立 item（如同一门课周一和周三各一节，则为两个 item）
11. summary 用一句话概括课程表（如"共 X 门课程，覆盖周一至周 X"）
"""


HOMEWORK_PROMPT = f"""你是一个专业的作业清单解析助手。请仔细分析这张作业清单截图，
提取每一项作业的详细信息。

{_JSON_OUTPUT_FORMAT}

注意事项：
1. type 固定为 "homework"
2. 每个 item 的 item_type 固定为 "homework"
3. 作业标题填入 title 字段
4. 科目/课程信息写入 description 字段开头
5. 截止日期填入 date 字段（YYYY-MM-DD）
6. 截止具体时间填入 end_time 字段（HH:MM），如"23:59"
7. deadline_type 默认为 "hard"，如果明确写了"建议"或"参考"则为 "soft"
8. recurrence 统一为 "none"
9. priority 根据截止时间远近和重要性判断，范围 5-9
10. estimated_minutes 根据作业量合理预估（如简单作业 30 分钟，大作业 240 分钟）
11. energy_level 根据作业难度判断
12. 作业要求/说明写入 description 字段
13. 如有多项作业，每个作业为一个 item
14. summary 用一句话概括作业清单（如"共 X 项作业，最近截止日期为 X"）
"""

# LifeOS 能力探测报告

**探测时间**：2026-09-02 15:07:43

## 1. 文本对话
✅ **状态**：ok
- 模型：`deepseek-chat`
- 响应时间：0ms
- 详情：返回 42 字符
- 示例输出：`时间管理就是合理分配精力，优先完成重要且紧急的任务，同时给自己留出缓冲时间应对意外。`

## 2. Function Calling
✅ **状态**：ok
- 模型：`deepseek-chat`
- 响应时间：0ms
- 详情：调用工具: add_task
- 示例输出：`{"title": "Python 大作业", "duration_minutes": 480, "deadline": "2024-03-20", "priority": "high"}`

## 3. 视觉模型
✅ **状态**：ok
- 模型：`qwen-vl-max`
- 响应时间：0ms
- 详情：成功输出结构化 JSON
- 示例输出：`{"events": [{"title": "高等数学期中考试", "date": "2024-03-20", "start_time": "09:00", "end_time": "11:00", "event_type": "exam"}, {"title": "Python 大作业截止", "`

## 4. Embedding 向量化
✅ **状态**：ok
- 模型：`text-embedding-v2`
- 响应时间：0ms
- 详情：维度=1536, 数量=3
- 示例输出：`[-0.0846, -0.0924, ...]`

## 5. LLM 缓存
✅ **状态**：ok
- 详情：缓存条目: 1, 总命中: 4

## 总结
✅ **核心能力全部通过**，可以进入 D2 开发。

## 配置信息
- LLM_BASE_URL: `https://api.deepseek.com/v1`
- LLM_MODEL: `deepseek-chat`
- LLM_VISION_MODEL: `qwen-vl-max`
- EMBED_MODEL: `text-embedding-v2`

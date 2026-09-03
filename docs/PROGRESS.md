# LifeOS 开发进度记录

> 每日结束花 10-15 分钟更新此文件，不把素材回忆留到最后。

## D1 · 能力探测（第 1 天）

**日期**：2026-09-02

### 完成内容

- [x] 项目骨架与目录结构创建
- [x] 领域模型定义（profile / goal / schedule / event / risk）
- [x] SQLite 数据库 schema 与 Database 类（13 张表）
- [x] LLM 统一客户端 ChatModel + 缓存 LLMCache
- [x] LLM Mock 模式（无需真实 API 即可开发）
- [x] 能力探测脚本 probe.py（文本/function calling/视觉/embedding/缓存）
- [x] MCP echo 服务器 + 连接验证脚本（适配 mcp 2.x）
- [x] 基础配置文件（requirements.txt, .env, pytest.ini, README.md）
- [x] docs 文档模板（CAPABILITY.md, PROGRESS.md, SCORING_EVIDENCE.md）

### 能力探测结果

详见 [CAPABILITY.md](./CAPABILITY.md)

**当前状态**：Mock 模式下 5 项能力全部通过，MCP 通信验证通过。
待填入真实 API 密钥后重新探测确认实际服务商能力。

### 阻塞点

无。真实 API 密钥待用户提供后填入 .env。

### 明日计划（D2）

- 调度器核心算法实现（约束贪心 + 优先级评分 + 牺牲清单）
- StudentEnv 仿真环境骨架
- 单元测试：调度器硬截止、依赖顺序、最小推进块、不可行牺牲清单

---

## D2 · 核心地基（第 2 天）

**日期**：2026-09-02

### 完成内容

- [x] 优先级评分器（priority.py）— 综合目标权重/剩余工作量/依赖阻塞/紧迫度
- [x] 约束检查器（constraints.py）— 可用时段生成、硬约束检查、精力匹配
- [x] 牺牲清单生成器（sacrifice.py）— 不可行时按优先级从低到高牺牲
- [x] 核心调度器（scheduler.py）— 四步调度算法：硬截止优先 → 最小推进块 → 剩余填充 → 缓冲
- [x] 风险预测器（risk_predictor.py）— 预计完成日公式 + 四级风险等级 + 四类建议
- [x] StudentEnv 仿真环境骨架（env.py）— 事件注入、按天推进、基础指标
- [x] 16 个单元测试全部通过

### 测试结果

```
16 passed in 0.15s
```

**覆盖的测试项**：
- 优先级评分：硬截止加分、阻塞数加分、紧迫度递增
- 依赖阻塞计数：线性链验证
- 约束检查：可用时段排除睡眠、承诺扣减时间、硬截止违规检测、依赖满足检查
- 调度器：单任务安排、硬截止满足、承诺不覆盖、时间不足牺牲、确定性
- 风险预测：无延期低风险、硬截止延期高风险
- 牺牲清单：按优先级从低到高排序

### 阻塞点

无

### 明日计划（D3）

- Agent 输入解析器（parse_user_message）
- Agent 目标分解器（decompose_goal）
- Agent 重规划逻辑 + 解释器
- PlanSnapshot 快照机制
- 跑通 "完成 30% 后重排" 场景

---

## D3-D4 · Agent 闭环（第 3-4 天）

**日期**：2026-09-02

### 完成内容

- [x] 输入解析器（parser.py）— 5 种意图识别 + 规则兜底 + LLM JSON mode
- [x] 目标分解器（decomposer.py）— 考试/项目/通用 3 类分解模板
- [x] 解释生成器（explainer.py）— PlanDiff 差异计算 + 规则/LLM 双模式中文解释
- [x] Agent 主类（life_agent.py）— 五步执行法 + 快照链 + 重规划阈值判断
- [x] PlanSnapshot 快照机制（字段扩展：sacrifice_list, time_slots, total_scheduled_minutes property）
- [x] ChangeLog 字段扩展（summary, detail, reason, affected_tasks）
- [x] Task 字段扩展（actual_minutes, created_at, updated_at）
- [x] 30% 进度偏差重排场景 — 7 项验证全部通过

### 五步执行法

```
用户输入
  ↓
① 解析输入 → ParseResult（5种意图：进度汇报/新目标/新事件/状态查询/未知）
  ↓
② 状态更新 → 应用进度/事件到内部状态（任务匹配、进度累加）
  ↓
③ 重规划判断 → 阈值：进度变化≥20% 或 新目标/新事件 或 首次规划
  ↓
④ 执行调度 → 调用确定性调度器（约束贪心 + 优先级评分 + 牺牲清单）
  ↓
⑤ 生成解释 → ChangeLog（中文说明：改了什么/为什么/牺牲了什么）
  ↓
AgentResult（回复 + 快照 + 变更记录）
```

### 测试结果

```
35 passed in 0.97s
```

**单元测试（33 个）**：
- 输入解析：进度汇报/新目标/生病事件/状态查询/无法识别
- 目标分解：考试类/项目类/通用类
- 解释生成：首次计划/牺牲清单解释
- Agent 主流程：新目标建计划/状态查询不重排/大进度触发重排/小进度不触发/快照链/生病事件触发/任务进度更新
- 调度器：16 个（D2 已覆盖）

**场景测试（2 个）**：
- 30% 进度偏差重排（7 项验证：快照链/硬截止/中文解释/变更原因/进度更新/计划变化/任务留存）
- 时间紧迫时生成牺牲清单

### 阻塞点

无

### 明日计划（D5-D6）

- 数据库持久化层完善（CRUD 封装）
- Streamlit UI 骨架
- MCP 工具注册（让 Agent 能调用调度器/数据库）
- 端到端 demo：用户说一句话 → Agent 解析 → 调度 → 展示计划

---

## D5-D6 · 全栈打通（第 5-6 天）

**日期**：2026-09-02

### 完成内容

- [x] 数据库 CRUD 完整封装（database.py，约 770 行）
  - Profile / Goal / Task / Commitment / PlanSnapshot / ExecutionRecord / LifeEvent / LLM Cache
  - 快照级联保存（时间槽 + 牺牲清单 + 变更日志 JSON）
  - 批量保存、进度更新、默认档案获取
- [x] Streamlit UI 骨架（ui/app.py，约 350 行）
  - 侧边栏：档案设置 + 目标列表 + 统计数据
  - 今日计划：按天展示时间槽（任务/承诺/缓冲不同颜色）
  - 任务列表：待开始/进行中/已完成三列看板
  - 历史快照：版本浏览 + 变更说明
  - 底部聊天：自然语言交互 + 快捷操作按钮
- [x] MCP 工具注册（mcp_server/server.py，8 个工具）
  - `create_goal` — 创建目标并生成计划
  - `report_progress` — 汇报进度触发重排
  - `get_current_plan` — 获取当前计划（按天分组）
  - `get_task_list` — 获取任务列表（按状态过滤）
  - `add_commitment` — 添加固定承诺并重排
  - `chat` — 自然语言对话（通用入口）
  - `echo` / `system_now` — 基础工具
- [x] 端到端 Demo（demo/end_to_end.py）
  - 一句话设定目标 → 自动分解 7 个任务
  - 进度滞后 → 触发重规划
  - 生病事件 → 计划调整
  - 状态查询 → 中文摘要
  - 快照历史 → 版本追溯

### 测试结果

```
35 passed in 0.97s
```

### 运行方式

**Streamlit UI**：
```bash
streamlit run ui/app.py
```

**MCP 服务器**：
```bash
python mcp_server/server.py
```

**端到端 Demo**：
```bash
python demo/end_to_end.py
```

### 架构总览

```
用户输入（自然语言）
    │
    ▼
┌─────────────────────────┐
│    Streamlit UI / MCP   │  ← 交互层
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   LifeAgent (五步执行法) │  ← 决策层
│  解析 → 更新 → 判断 →   │
│  调度 → 解释            │
└───────────┬─────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌──────────┐ ┌──────────┐
│ 调度器   │ │目标分解器│  ← 核心算法层
│ (确定性) │ │ (LLM+规则)│
└──────────┘ └──────────┘
     │
     ▼
┌─────────────────────────┐
│   SQLite Database       │  ← 持久化层
│  13 张表 + 完整 CRUD    │
└─────────────────────────┘
```

### 阻塞点

无

### 后续可优化方向

- LLM 真实 API 接入（当前 Mock 模式可用）
- 多模态截图解析（视觉事件提取）
- RAG 知识库（错题本、知识点检索）
- 多学生/多 profile 支持
- 移动端适配
- 数据可视化图表（进度曲线、时间分配饼图）

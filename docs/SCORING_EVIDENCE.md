# 评分项证据链

> 答辩时 30 秒内定位对应证据。每行含：评分项 / 代码位置 / 演示方式。

## 一、创新性与技术亮点

| 评分项 | 代码位置 | 演示方式 |
|---|---|---|
| 确定性调度（LLM 只做理解，排程用算法） | `src/lifeops/scheduler/scheduler.py:60-120` | 演示同一输入两次，计划完全一致 |
| 五步执行法闭环 | `src/lifeops/agent/life_agent.py:649-729` | 演示：建目标→生病→重排→解释 |
| 待确认机制（破坏性操作必确认） | `src/lifeops/agent/life_agent.py:690-711, 877-932` | 演示：说"取消计划"→反问→确认才删 |
| 快照链可追溯 | `src/lifeops/agent/life_agent.py:135-138, 556-565` | UI 历史快照面板，展示版本差异 |
| LLM 失败规则兜底 | `src/lifeops/agent/parser.py:198-214, 255-267` | Mock 模式无 API Key 也能跑全流程 |
| 解析结果持久化避免 LLM 漂移 | `src/lifeops/agent/life_agent.py:778-809, 906-907` | 确认后复用已保存结果，不二次调 LLM |

## 二、功能完整度

| 评分项 | 代码位置 | 演示方式 |
|---|---|---|
| 自然语言建目标+自动分解 | `src/lifeops/agent/parser.py`, `src/lifeops/agent/decomposer.py` | 说"我要准备高数考试"→分解7个任务 |
| 进度汇报触发重排 | `src/lifeops/agent/life_agent.py:624-632` | 说"完成了30%"→计划重排 |
| 生活事件（生病/改期） | `src/lifeops/agent/life_agent.py:950-952` | 说"我感冒了"→计划调整 |
| 课表/固定承诺管理 | `src/lifeops/agent/life_agent.py:233-275` | 添加课程→任务不占用上课时间 |
| 取消计划（目标/任务/课表） | `src/lifeops/agent/life_agent.py:320-449` | "删除英语任务"→确认后删除 |
| 状态查询 | `src/lifeops/agent/life_agent.py:719-720` | "看看进度"→中文摘要 |
| MCP 工具接口（8个工具） | `mcp_server/server.py:52-366` | MCP 客户端调用 create_goal |
| 多模态截图解析 | `src/lifeops/vision/screenshot_parser.py` | 上传课表截图→自动建课表 |
| RAG 知识库检索 | `src/lifeops/rag/knowledge_base.py` | 检索知识点 |

## 三、工程化与代码质量

| 评分项 | 代码位置 | 演示方式 |
|---|---|---|
| 分层架构（交互/决策/算法/持久化） | `src/lifeops/` 目录结构 | 架构图：`docs/PROGRESS.md:199-228` |
| SQLite 13 表完整 CRUD | `src/lifeops/storage/database.py` | 重启应用数据不丢失 |
| 跨进程状态恢复 | `src/lifeops/agent/life_agent.py:187-204` | 测试 `TestPersistence` |
| LLM 缓存（省 token） | `src/lifeops/llm/cache.py`, `src/lifeops/llm/chat_model.py:121-127` | 相同输入第二次秒回 |
| Mock 模式（无 API Key 可开发） | `src/lifeops/llm/chat_model.py:64, 146-153` | `LLM_MOCK_MODE=true` 跑全流程 |
| 日志系统 | `src/lifeops/llm/logger.py`（新增） | 运行时输出分级日志 |
| Docker 容器化 | `Dockerfile` | `docker build -t lifeos .` |
| 配置管理（.env） | `.env.example` | 复制后填入 API Key |

## 四、测试覆盖

| 评分项 | 代码位置 | 演示方式 |
|---|---|---|
| 单元测试 35+ | `tests/unit/` | `python -m pytest` 全绿 |
| 集成测试（30%延迟场景） | `tests/integration/test_30pct_delay.py` | 跑集成测试 |
| 边界测试（取消/确认/自愈） | `tests/unit/test_agent.py:635-866` | 取消流程、确认流程、快照自愈 |
| 测试覆盖率报告 | `pytest.ini`（已配 cov） | `pytest --cov` 输出覆盖率 |
| 红队测试（恶意/边界/超长） | `tests/unit/test_redteam.py`（新增） | 恶意输入不崩溃 |

## 五、可解释性与可观测

| 评分项 | 代码位置 | 演示方式 |
|---|---|---|
| 每次重排输出变更原因 | `src/lifeops/agent/explainer.py:55-87` | 重排后显示"改了什么/为什么/牺牲了什么" |
| 牺牲清单（时间不足时） | `src/lifeops/scheduler/sacrifice.py` | 任务过多时显示被搁置的任务 |
| 变更日志 ChangeLog | `src/lifeops/models/schedule.py` | 每次重排留痕 |
| 风险预测 | `src/lifeops/scheduler/risk_predictor.py` | 显示任务完成风险等级 |

## 六、演示脚本（端到端）

| 场景 | 步骤 | 预期效果 |
|---|---|---|
| 建目标 | "我要准备高数考试，6月15日考" | 自动分解7个任务，排出一周计划 |
| 进度更新 | "我完成了30%" | 触发重排，显示变更说明 |
| 突发事件 | "我今天感冒发烧了" | 计划调整，腾出休息时间 |
| 取消 | "取消备考计划"→"可以" | 目标+任务全删，重新排期 |
| 状态查询 | "看看我的进度" | 中文摘要+任务完成度 |
| 历史回溯 | 打开历史快照面板 | 展示每次重排的版本差异 |

## 七、运行命令速查

```bash
# 安装依赖
pip install -r requirements.txt

# 能力探测
python src/lifeops/probe.py

# 启动 UI
streamlit run ui/app.py

# 启动 MCP Server
python mcp_server/server.py

# 跑测试（含覆盖率）
python -m pytest

# 跑端到端 demo
python demo/end_to_end.py

# Docker 构建
docker build -t lifeos .
docker run -p 8501:8501 lifeos
```

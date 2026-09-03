# LifeOS — AI 大学生人生任务操作 Agent

> 面向大学生的 AI 人生任务操作系统：Plan → Execute → Observe → Replan 完整闭环

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API 密钥

# 3. 能力探测（首次运行必做）
python src/lifeops/probe.py

# 4. 启动应用
streamlit run ui/app.py

# 5. 跑端到端演示与测试
python demo/end_to_end.py
python -m pytest
```

## 项目结构

```
lifeos/
├── src/lifeops/        # 核心代码
│   ├── models/         # 领域模型
│   ├── scheduler/      # 确定性调度器
│   ├── agent/          # Agent 编排
│   ├── llm/            # LLM 客户端 + 缓存
│   ├── rag/            # 知识检索
│   ├── vision/         # 截图解析
│   ├── simulation/     # 仿真环境
│   └── storage/        # SQLite 持久化
├── mcp_server/         # MCP stdio 服务
├── app/                # Streamlit 前端
├── demo/               # 演示场景与 fixture
├── tests/              # 单元与集成测试
├── docs/               # 文档与进度记录
└── outputs/            # 运行输出
```

## 核心特性

- **确定性调度**：约束贪心算法，LLM 只负责理解与解释
- **完全可解释**：每次重规划输出原因标签、变更清单、牺牲清单
- **真实工具调用**：MCP 工具带可见副作用（.ics / .md / UI 条目）
- **多模态输入**：截图转结构化事件，支持 OCR 兜底
- **RAG 知识层**：课件检索 + 严格引用校验
- **仿真评测**：3 套标准场景 + StaticBaseline 对比

## 开发进度

详见 [docs/PROGRESS.md](docs/PROGRESS.md)

# LifeOS Dockerfile
# 大学生 AI 人生任务操作系统
FROM python:3.11-slim

WORKDIR /app

# 系统依赖（OCR 可选，精简安装）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 默认以 Mock 模式启动，便于无 API Key 时也能运行
ENV LLM_MOCK_MODE=true
ENV PYTHONUNBUFFERED=1

# Streamlit UI 端口
EXPOSE 8501

# 默认启动 Streamlit UI
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# 也可通过覆盖 CMD 启动 MCP server:
#   docker run lifeos python mcp_server/server.py

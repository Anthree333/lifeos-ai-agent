"""LifeOS Streamlit 应用 — 精致版 v2。

运行：streamlit run ui/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
from datetime import datetime, date, time as dt_time, timedelta
import math
import os
import tempfile

from lifeops.models import (
    StudentProfile, Goal, Task, Commitment,
    TaskStatus, DeadlineType, EnergyLevel, CommitmentType,
)
from lifeops.agent import LifeAgent
from lifeops.storage.database import get_db

# UI 启动时加载 .env（Streamlit 不会自动读取）
_env_warnings = []
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env_example_path = env_path.parent / ".env.example"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    elif env_example_path.exists():
        _env_warnings.append(
            "⚠️ 未找到 `.env` 配置文件，AI 对话功能不可用。"
            f" 请复制 `.env.example` 为 `.env` 并填入 API 密钥："
            "运行 `cp .env.example .env`（Windows 用 `copy .env.example .env`）。"
        )
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

# 可选模块（未安装也不影响核心功能）
try:
    from lifeops.vision import ScreenshotParser
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

try:
    from lifeops.rag import KnowledgeBase, Retriever
    HAS_RAG = True
except ImportError:
    HAS_RAG = False


# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="LifeOS · AI 学习管家",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "LifeOS — 你的 AI 学习管家",
    },
)


# ==========================================
# 主题样式
# ==========================================
st.markdown("""
<style>
/* ===== 基础重置 ===== */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    section.main > div {
        padding-top: 0;
    }

/* ===== 收起后的侧栏展开按钮 ===== */
    [data-testid="stExpandSidebarButton"] {
        position: fixed !important;
        top: 8px !important;
        left: 8px !important;
        z-index: 99999 !important;
        width: 38px !important;
        height: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12) !important;
        cursor: pointer !important;
    }
    body:has([data-testid="stExpandSidebarButton"])
        [data-testid="stHeader"] {
        display: block !important;
    }

/* ===== 字体 ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif;
    }

/* ===== 滚动条美化 ===== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

/* ===== 头部 ===== */
    /* 隐藏 Streamlit 默认顶栏，避免悬浮遮住问候语与统计卡片 */
    [data-testid="stHeader"] {
        display: none;
    }
    .app-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
    }
    .app-icon {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4);
        flex-shrink: 0;
        animation: pulse-glow 3s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4); }
        50% { box-shadow: 0 4px 20px rgba(102, 126, 234, 0.6); }
    }
    .app-title h1 {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #475569 0%, #1e293b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .app-title p {
        font-size: 0.82rem;
        color: #94a3b8;
        margin: 2px 0 0 0;
    }

/* ===== 统计卡片 ===== */
    .stat-card {
        background: white;
        border-radius: 14px;
        padding: 16px 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .stat-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        transform: translateY(-2px);
        border-color: #cbd5e1;
    }
    .stat-card .stat-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        flex-shrink: 0;
        transition: transform 0.25s ease;
    }
    .stat-card:hover .stat-icon {
        transform: scale(1.08);
    }
    .stat-card .stat-content {
        flex: 1;
        min-width: 0;
    }
    .stat-card .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
        font-weight: 500;
    }
    .stat-icon.blue { background: linear-gradient(135deg, #eff6ff, #dbeafe); color: #3b82f6; }
    .stat-icon.green { background: linear-gradient(135deg, #f0fdf4, #dcfce7); color: #22c55e; }
    .stat-icon.amber { background: linear-gradient(135deg, #fffbeb, #fef3c7); color: #f59e0b; }
    .stat-icon.purple { background: linear-gradient(135deg, #faf5ff, #f3e8ff); color: #a855f7; }
    .stat-icon.red { background: linear-gradient(135deg, #fef2f2, #fee2e2); color: #ef4444; }
    .stat-icon.cyan { background: linear-gradient(135deg, #ecfeff, #cffafe); color: #06b6d4; }

/* ===== 进度环 ===== */
    .progress-ring {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }
    .progress-ring svg {
        transform: rotate(-90deg);
        filter: drop-shadow(0 1px 2px rgba(0,0,0,0.05));
    }
    .progress-ring-text {
        position: absolute;
        font-weight: 700;
        color: #334155;
        letter-spacing: -0.5px;
    }

/* ===== 时间轴 ===== */
    .timeline-container {
        position: relative;
        padding-left: 8px;
    }
    .timeline-container::before {
        content: '';
        position: absolute;
        left: 103px;
        top: 0;
        bottom: 0;
        width: 2px;
        background: #f1f5f9;
        border-radius: 1px;
    }
    .timeline-item {
        display: flex;
        gap: 14px;
        margin-bottom: 8px;
        position: relative;
        animation: slideIn 0.3s ease;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-6px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .timeline-time {
        flex-shrink: 0;
        width: 90px;
        text-align: right;
        font-size: 0.78rem;
        color: #64748b;
        padding-top: 11px;
        font-weight: 500;
        font-variant-numeric: tabular-nums;
    }
    .timeline-dot {
        flex-shrink: 0;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-top: 12px;
        position: relative;
        z-index: 2;
        border: 2px solid white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
    }
    .timeline-card {
        flex: 1;
        background: white;
        border-radius: 10px;
        padding: 10px 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .timeline-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        transform: translateX(2px);
    }
    .timeline-card.task {
        border-left: 3px solid #6366f1;
    }
    .timeline-card.commitment {
        border-left: 3px solid #f59e0b;
        background: linear-gradient(135deg, #fffbeb, #fefce8);
    }
    .timeline-card.buffer {
        border-left: 3px solid #94a3b8;
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        opacity: 0.65;
    }
    .timeline-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: #1e293b;
        margin-bottom: 3px;
    }
    .timeline-meta {
        font-size: 0.72rem;
        color: #94a3b8;
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
    }
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .tag-hard { background: linear-gradient(135deg, #fee2e2, #fecaca); color: #dc2626; }
    .tag-soft { background: linear-gradient(135deg, #e0e7ff, #c7d2fe); color: #4f46e5; }
    .tag-high { background: linear-gradient(135deg, #dcfce7, #bbf7d0); color: #16a34a; }
    .tag-medium { background: linear-gradient(135deg, #fef3c7, #fde68a); color: #d97706; }
    .tag-low { background: linear-gradient(135deg, #f1f5f9, #e2e8f0); color: #64748b; }
    .tag-priority-high { background: linear-gradient(135deg, #fef2f2, #fee2e2); color: #dc2626; }
    .tag-priority-mid { background: linear-gradient(135deg, #fffbeb, #fef3c7); color: #d97706; }
    .tag-priority-low { background: linear-gradient(135deg, #f0fdf4, #dcfce7); color: #16a34a; }

/* ===== 任务卡片 ===== */
    .task-card {
        background: white;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
        position: relative;
        overflow: hidden;
    }
    .task-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transition: transform 0.3s ease;
        transform-origin: left;
    }
    .task-card:hover::before {
        transform: scaleX(1);
    }
    .task-card:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        transform: translateY(-2px);
        border-color: #cbd5e1;
    }
    .task-card.completed {
        opacity: 0.55;
        background: #f8fafc;
    }
    .task-card.completed::before {
        background: #22c55e;
        transform: scaleX(1);
    }
    .task-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 8px;
        gap: 8px;
    }
    .task-card-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: #1e293b;
        line-height: 1.4;
        flex: 1;
    }
    .task-card.completed .task-card-title {
        text-decoration: line-through;
        color: #94a3b8;
    }
    .task-card-badges {
        display: flex;
        gap: 4px;
        flex-shrink: 0;
        flex-wrap: wrap;
        justify-content: flex-end;
    }
    .task-progress-bar {
        height: 5px;
        background: #f1f5f9;
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 8px;
    }
    .task-progress-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .task-progress-fill.high { background: linear-gradient(90deg, #22c55e, #16a34a); }
    .task-progress-fill.mid { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .task-progress-fill.low { background: linear-gradient(90deg, #6366f1, #4f46e5); }
    .task-card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
        color: #94a3b8;
    }
    .kanban-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 2px solid #f1f5f9;
    }
    .kanban-header h3 {
        font-size: 0.9rem;
        font-weight: 600;
        color: #334155;
        margin: 0;
    }
    .kanban-count {
        background: #f1f5f9;
        color: #64748b;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        min-width: 24px;
        text-align: center;
    }

/* ===== 聊天界面 ===== */
    .chat-container {
        max-height: 420px;
        overflow-y: auto;
        padding: 12px 6px;
        margin-bottom: 10px;
    }
    .chat-message {
        display: flex;
        gap: 10px;
        margin-bottom: 16px;
        animation: fadeInUp 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .chat-message.user {
        flex-direction: row-reverse;
    }
    .chat-message .chat-bubble-wrap {
        display: flex;
        flex-direction: column;
        min-width: 0;
        max-width: 72%;
    }
    .chat-message.bot .chat-bubble-wrap {
        align-items: flex-start;
    }
    .chat-message.user .chat-bubble-wrap {
        align-items: flex-end;
    }
    .chat-avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .chat-avatar.bot {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .chat-avatar.user {
        background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
        color: #475569;
    }
    .chat-bubble {
        max-width: 100%;
        width: fit-content;
        padding: 10px 14px;
        border-radius: 18px;
        font-size: 0.88rem;
        line-height: 1.55;
        word-wrap: break-word;
        overflow-wrap: anywhere;
        word-break: break-word;
        white-space: pre-wrap;
        position: relative;
    }
    .chat-bubble.bot {
        background: white;
        border: 1px solid #e2e8f0;
        border-top-left-radius: 6px;
        color: #334155;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .chat-bubble.user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-top-right-radius: 6px;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    .chat-time {
        font-size: 0.68rem;
        color: #cbd5e1;
        margin-top: 4px;
        text-align: right;
    }
    .chat-message.user .chat-time {
        text-align: right;
    }
    .chat-message.bot .chat-time {
        text-align: left;
    }

/* ===== 打字指示器 ===== */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 12px 16px;
    }
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #94a3b8;
        border-radius: 50%;
        animation: typing-bounce 1.4s ease-in-out infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing-bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
        30% { transform: translateY(-6px); opacity: 1; }
    }

/* ===== 侧边栏 ===== */
    .sidebar-section {
        margin-bottom: 20px;
    }
    .sidebar-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 10px;
        padding-left: 2px;
    }
    [data-testid="stSidebar"] button[kind="secondary"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        font-weight: 600;
        color: #334155;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        border-color: #a5b4fc;
        color: #4f46e5;
        background: linear-gradient(135deg, #eef2ff, #ffffff);
    }
    .goal-item {
        background: white;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 6px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    .goal-item:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transform: translateX(2px);
    }
    .goal-item-title {
        font-weight: 600;
        font-size: 0.82rem;
        color: #1e293b;
        margin-bottom: 5px;
        line-height: 1.3;
    }
    .goal-item-progress {
        height: 3px;
        background: #e2e8f0;
        border-radius: 2px;
        overflow: hidden;
        margin-bottom: 5px;
    }
    .goal-item-fill {
        height: 100%;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        border-radius: 2px;
        transition: width 0.5s ease;
    }
    .goal-item-meta {
        font-size: 0.68rem;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
    }

/* ===== 牺牲清单 ===== */
    .sacrifice-card {
        background: linear-gradient(135deg, #fef2f2, #fff1f2);
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }
    .sacrifice-card:hover {
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.1);
    }
    .sacrifice-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #dc2626;
        margin-bottom: 2px;
    }
    .sacrifice-reason {
        font-size: 0.72rem;
        color: #f87171;
    }

/* ===== 空状态 ===== */
    .empty-state {
        text-align: center;
        padding: 48px 20px;
        color: #94a3b8;
    }
    .empty-state-icon {
        font-size: 44px;
        margin-bottom: 12px;
        opacity: 0.6;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }
    .empty-state-text {
        font-size: 0.92rem;
        color: #64748b;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .empty-state-sub {
        font-size: 0.78rem;
        color: #94a3b8;
    }

/* ===== 日期标签 ===== */
    .date-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 18px;
    }
    .date-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.3px;
    }
    .date-subtitle {
        font-size: 0.78rem;
        color: #64748b;
    }

/* ===== 快捷操作 ===== */
    .quick-action {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 14px;
    }
    .quick-chip {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.78rem;
        color: #475569;
        cursor: pointer;
        transition: all 0.2s ease;
        white-space: nowrap;
        font-weight: 500;
    }
    .quick-chip:hover {
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
        border-color: #c7d2fe;
        color: #4f46e5;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.15);
    }
    .quick-chip:active {
        transform: translateY(0);
    }

/* ===== 快照时间线 ===== */
    .snapshot-item {
        display: flex;
        gap: 12px;
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 8px;
        background: white;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .snapshot-item:hover {
        background: #fafafa;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transform: translateX(2px);
    }
    .snapshot-item::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, #667eea, #764ba2);
    }
    .snapshot-version {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    .snapshot-info {
        flex: 1;
    }
    .snapshot-reason {
        font-weight: 600;
        font-size: 0.88rem;
        color: #1e293b;
        margin-bottom: 3px;
    }
    .snapshot-meta {
        font-size: 0.72rem;
        color: #94a3b8;
    }

/* ===== 分布条 ===== */
    .dist-row {
        margin-bottom: 10px;
    }
    .dist-row:last-child {
        margin-bottom: 0;
    }
    .dist-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        margin-bottom: 4px;
    }
    .dist-label span:first-child {
        color: #64748b;
        font-weight: 500;
    }
    .dist-label span:last-child {
        font-weight: 600;
        color: #334155;
    }
    .dist-bar {
        height: 6px;
        background: #f1f5f9;
        border-radius: 3px;
        overflow: hidden;
    }
    .dist-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

/* ===== 摘要卡片 ===== */
    .summary-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
    }
    .summary-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .summary-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.5px;
    }

/* ===== 分隔线 ===== */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 16px 0;
    }

/* ===== Tab 样式微调 ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 10px 10px 0 0;
        padding: 0 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8fafc;
    }
    .stTabs [aria-selected="true"] {
        background: white;
    }

    /* 学习时间段 — 开始/结束输入框紧凑内边距 */
    .stTextInput input[data-testid="stTextInputField"][aria-label="开始"],
    .stTextInput input[data-testid="stTextInputField"][aria-label="结束"] {
        padding-top: 5px !important;
        padding-bottom: 5px !important;
    }
    /* ===== 侧栏吸顶实时时钟横幅（fixed 钉在侧栏顶部） ===== */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 96px;
    }
    section[data-testid="stSidebar"] {
        transform: translateZ(0);  /* 让侧栏成为 fixed 横幅的包含块 */
    }
    .live-clock-banner {
        position: fixed;
        top: 0;
        left: 0;
        z-index: 999;
        width: 100%;
        box-sizing: border-box;
        margin: 0;
        padding: 18px 20px 15px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 0 0 16px 16px;
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.28);
        color: #ffffff;
        text-align: center;
    }
    .live-clock-banner .clock-date {
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        opacity: 0.88;
    }
    .live-clock-banner .clock-time {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        line-height: 1.15;
        margin-top: 1px;
        font-variant-numeric: tabular-nums;
    }
    /* 侧栏 Logo：悬停反馈 + 透明覆盖按钮（点击回主页） */
    .home-logo-wrap {
        border-radius: 10px;
        transition: opacity 0.15s ease;
    }
    .home-logo-wrap:hover {
        opacity: 0.75;
    }
    div.st-key-nav_home {
        position: relative;
        margin-top: -72px;
        height: 72px;
        z-index: 20;
        margin-bottom: 0 !important;
    }
    div.st-key-nav_home button {
        width: 100%;
        height: 72px;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: transparent !important;
        padding: 0 !important;
        min-height: 0 !important;
        font-size: 0 !important;
    }
    div.st-key-nav_home button:hover,
    div.st-key-nav_home button:focus {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 工具函数
# ==========================================
def progress_ring(percent: int, size: int = 36, color: str = "#6366f1", bg_color: str = "#e2e8f0") -> str:
    """生成 SVG 进度环。"""
    radius = (size - 4) / 2
    circumference = 2 * math.pi * radius
    offset = circumference * (1 - percent / 100)
    font_size = max(10, int(size * 0.22))

    return f"""
    <div class="progress-ring" style="width:{size}px;height:{size}px;">
        <svg width="{size}" height="{size}">
            <defs>
                <linearGradient id="grad-{size}" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
                </linearGradient>
            </defs>
            <circle cx="{size/2}" cy="{size/2}" r="{radius}"
                    stroke="{bg_color}" stroke-width="3" fill="none"/>
            <circle cx="{size/2}" cy="{size/2}" r="{radius}"
                    stroke="url(#grad-{size})" stroke-width="3" fill="none"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"
                    stroke-linecap="round"/>
        </svg>
        <span class="progress-ring-text" style="font-size:{font_size}px;">{percent}%</span>
    </div>
    """


def energy_icon(level: str) -> str:
    """精力等级图标。"""
    if level == "high":
        return "⚡"
    elif level == "medium":
        return "🔋"
    else:
        return "🌙"


def status_color(status: str) -> str:
    """状态颜色。"""
    if status == "completed":
        return "#22c55e"
    elif status == "in_progress":
        return "#f59e0b"
    else:
        return "#94a3b8"


def priority_label(priority: int) -> tuple:
    """优先级标签文本和样式。"""
    if priority >= 8:
        return ("🔥 高优", "tag-priority-high")
    elif priority >= 5:
        return ("📌 中优", "tag-priority-mid")
    else:
        return ("🌱 低优", "tag-priority-low")


def time_to_str(t) -> str:
    """将 time 对象转为 HH:MM 字符串。"""
    if isinstance(t, dt_time):
        return t.strftime("%H:%M")
    return str(t)


def parse_time_str(s) -> dt_time:
    """将 HH:MM 字符串转为 time 对象。支持 24:00 → 00:00。"""
    if isinstance(s, dt_time):
        return s
    parts = str(s).strip().split(":")
    try:
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        if h == 24:
            h = 0
        return dt_time(h, m)
    except (ValueError, IndexError):
        return dt_time(7, 0)


def normalize_time_str(val: str) -> str:
    """补全简写时间：8 → 08:00, 17 → 17:00, 24 → 00:00（跨日）。"""
    s = str(val).strip()
    if not s:
        return s
    if ":" not in s:
        try:
            h = int(s)
            if 0 <= h <= 24:
                return f"{h % 24:02d}:00"
        except ValueError:
            pass
    else:
        parts = s.split(":")
        try:
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            if 0 <= h <= 24 and 0 <= m <= 59:
                if h == 24 and m != 0:
                    return s  # 非法，原样保留
                return f"{h % 24:02d}:{m:02d}"
        except ValueError:
            pass
    return s


def _normalize_study_slot() -> None:
    """st.text_input 的 on_change 回调：补全学习时间段 + 自动保存到档案。"""
    changed = False
    for key in list(st.session_state.keys()):
        if key.startswith("slot_start_") or key.startswith("slot_end_"):
            raw = st.session_state.get(key, "")
            norm = normalize_time_str(raw)
            if norm != raw:
                st.session_state[key] = norm
                changed = True

    # 同步 study_slots dict 里的值（render 循环里读的是它）
    for i, s in enumerate(st.session_state.get("study_slots", [])):
        start_key = f"slot_start_{i}"
        end_key = f"slot_end_{i}"
        if start_key in st.session_state:
            s["start"] = st.session_state[start_key]
        if end_key in st.session_state:
            s["end"] = st.session_state[end_key]

    _auto_save_profile()


def _auto_save_profile() -> None:
    """把 study_slots 自动写入 profile + 触发重规划。"""
    from __main__ import agent, db, profile  # type: ignore

    slots = st.session_state.get("study_slots", [])
    if not slots:
        return
    total = 0
    for slot in slots:
        try:
            sh, sm = map(int, slot["start"].split(":"))
            eh, em = map(int, slot["end"].split(":"))
            start_mins = sh * 60 + sm
            end_mins = (eh % 24) * 60 + em
            if eh == 24 and em == 0:
                end_mins = 24 * 60
            delta = end_mins - start_mins
            if delta < 0:
                delta += 24 * 60
            total += max(0, delta)
        except (ValueError, KeyError):
            pass

    try:
        profile.wake_up_time = parse_time_str(slots[0]["start"])
        profile.sleep_time = parse_time_str(slots[-1]["end"])
        profile.daily_high_energy_hours = max(1, int(total / 60))
        db.save_profile(profile)
        st.session_state.profile = profile
        if agent and getattr(agent, "_tasks", None):
            agent.force_replan("学习时间段已自动更新")
    except Exception:
        pass


WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


@st.fragment(run_every=timedelta(seconds=1))
def render_live_clock() -> None:
    """侧栏顶部实时日期时钟（吸顶横幅，每秒自动刷新）。"""
    now = datetime.now()
    st.markdown(f"""
    <div class="live-clock-banner">
        <div class="clock-date">
            {now.strftime('%Y年%m月%d日')} · {WEEKDAY_NAMES[now.weekday()]}
        </div>
        <div class="clock-time">{now.strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)


# 侧栏导航视图：view_key -> (图标, 名称, 占位提示)
NAV_VIEWS = {
    "knowledge": ("📚", "知识库", "上传课件与资料，AI 为你问答"),
    "notes": ("📝", "笔记", "记录学习笔记与灵感"),
    "schedule": ("📅", "日程", "管理你的日程安排"),
    "courses": ("📖", "课表", "查看与维护每周课表"),
}


def render_placeholder_view(view_key: str) -> None:
    """4 个新视图的空白占位界面（后续填充内容）。"""
    icon, label, sub = NAV_VIEWS.get(view_key, ("✨", "LifeOS", ""))
    st.markdown(f"""
    <div class="app-header">
        <div class="app-icon" style="font-size:1.6rem;">{icon}</div>
        <div class="app-title">
            <h1>{label}</h1>
            <p>{sub}</p>
        </div>
    </div>
    <div class="empty-state" style="margin-top:60px;">
        <div class="empty-state-icon" style="font-size:3.2rem;">{icon}</div>
        <div class="empty-state-text">{label}页面建设中</div>
        <div class="empty-state-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def next_weekday_date(weekday: int, time_value: dt_time) -> date:
    """返回未来（含今天）最近一次 weekday 的日期。"""
    today = date.today()
    days_ahead = (weekday - today.weekday()) % 7
    result = today + timedelta(days=days_ahead)
    if days_ahead == 0 and datetime.now().time() > time_value:
        result += timedelta(days=7)
    return result


def commitment_weekday(commitment) -> int:
    """从课程 start_time 推导星期索引。"""
    try:
        start = datetime.fromisoformat(commitment.start_time)
        return start.weekday()
    except (ValueError, TypeError):
        return 0


def commitment_time_label(commitment) -> str:
    """课程起止时间展示文本。"""
    try:
        start = datetime.fromisoformat(commitment.start_time)
        end = datetime.fromisoformat(commitment.end_time)
        return f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"
    except (ValueError, TypeError):
        return commitment.start_time


def commitment_is_class(commitment) -> bool:
    """判断固定承诺是否为课程。"""
    value = getattr(commitment.type, "value", commitment.type)
    return value == "class"


# ==========================================
# 初始化 Session State
# ==========================================
if "agent" not in st.session_state:
    base_dir = PROJECT_ROOT
    db_path = os.environ.get("DB_PATH") or str(base_dir / "data" / "lifeos.db")
    chroma_path = (
        os.environ.get("CHROMA_PATH")
        or str(base_dir / "data" / "knowledge" / "chroma")
    )
    db = get_db(db_path)
    profile = db.get_default_profile()

    # 初始化 LLM / 截图解析器 / 知识库（可选）
    from lifeops.llm import ChatModel
    try:
        llm = ChatModel.from_env()
    except Exception as _e:
        llm = None
        _msg = str(_e)
        if _env_warnings:
            _env_warnings.append("")  # 空行分隔
        _env_warnings.append(f"⚠️ LLM 初始化失败：{_msg}。请检查 `.env` 中的 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`。")

    agent = LifeAgent(profile=profile, db=db, llm=llm)

    if HAS_VISION:
        try:
            if llm is not None:
                parser = ScreenshotParser(llm=llm)
                agent.screenshot_parser = parser
        except Exception:
            pass

    if HAS_RAG:
        try:
            if llm is not None:
                kb = KnowledgeBase(persist_dir=chroma_path, llm=llm, db=db)
                retriever = Retriever(kb, llm=llm)
                agent.knowledge_base = kb
                agent.retriever = retriever
        except Exception:
            pass

    st.session_state.agent = agent
    st.session_state.profile = profile
    st.session_state.db = db
    st.session_state.chat_history = []
    st.session_state.current_date = date.today()
    st.session_state.active_tab = "dashboard"
    st.session_state.pending_screenshot = None  # 待确认的截图解析结果


agent: LifeAgent = st.session_state.agent
profile: StudentProfile = st.session_state.profile
db = st.session_state.db

# 主区视图：home=默认 LifeOS 界面，其余为知识库/笔记/日程/课表
# 支持通过 ?view=home 链接（侧栏 Logo）回到主页
if st.query_params.get("view") == "home":
    st.session_state.main_view = "home"
    st.query_params.pop("view", None)
main_view = st.session_state.get("main_view", "home")


# ==========================================
# 侧边栏
# ==========================================
with st.sidebar:
    # 实时日期时间
    render_live_clock()

    # Logo（视觉展示，下方透明按钮覆盖实现点击回主页）
    st.markdown(f"""
    <div class="home-logo-wrap">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:20px;">
            <div class="app-icon">✨</div>
            <div>
                <div style="font-weight:700; font-size:1.05rem; color:#0f172a; letter-spacing:-0.3px;">LifeOS</div>
                <div style="font-size:0.72rem; color:#94a3b8; margin-top:1px;">AI 学习管家</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # 透明覆盖按钮：点击回到 LifeOS 默认界面
    if st.button(" ", key="nav_home", use_container_width=True):
        st.session_state.main_view = "home"
        st.rerun()

    # 四个功能入口：切换主区视图（不跳转新页面，侧栏保持展示）
    nav_items = [
        ("📚", "知识库", "knowledge"),
        ("📝", "笔记", "notes"),
        ("📅", "日程", "schedule"),
        ("📖", "课表", "courses"),
    ]
    for i in range(0, len(nav_items), 2):
        col_left, col_right = st.columns(2)
        with col_left:
            icon, label, key = nav_items[i]
            if st.button(
                f"{icon} {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if main_view == key else "secondary",
            ):
                st.session_state.main_view = key
                st.rerun()
        with col_right:
            if i + 1 < len(nav_items):
                icon, label, key = nav_items[i + 1]
                if st.button(
                    f"{icon} {label}",
                    key=f"nav_{key}",
                    use_container_width=True,
                    type="primary" if main_view == key else "secondary",
                ):
                    st.session_state.main_view = key
                    st.rerun()

    # 档案设置
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">👤 学生档案</div>', unsafe_allow_html=True)

    # 初始化学习时间段
    if "study_slots" not in st.session_state:
        st.session_state.study_slots = [
            {"start": time_to_str(profile.wake_up_time), "end": time_to_str(profile.sleep_time)}
        ]

    # 选项 1：今日预计学习时间（只读，自动计算）— 占位，渲染后填入
    time_calc_placeholder = st.empty()

    # 选项 2：学习时间段（可添加/删除）
    st.markdown('<div style="font-size:0.78rem; font-weight:600; color:#475569; margin-bottom:6px;">学习时间段</div>', unsafe_allow_html=True)

    slots_to_remove = []
    for i, slot in enumerate(st.session_state.study_slots):
        c1, c2, c3 = st.columns([1, 1, 0.35])
        with c1:
            slot["start"] = st.text_input(
                "开始", value=slot["start"], key=f"slot_start_{i}",
                label_visibility="collapsed", placeholder="08:00",
                on_change=_normalize_study_slot,
            )
        with c2:
            slot["end"] = st.text_input(
                "结束", value=slot["end"], key=f"slot_end_{i}",
                label_visibility="collapsed", placeholder="10:00",
                on_change=_normalize_study_slot,
            )
        with c3:
            if len(st.session_state.study_slots) > 1:
                if st.button("✕", key=f"slot_del_{i}"):
                    slots_to_remove.append(i)

    if slots_to_remove:
        for i in reversed(slots_to_remove):
            st.session_state.study_slots.pop(i)
        _auto_save_profile()
        st.rerun()

    if st.button("➕ 添加时间段", use_container_width=True):
        st.session_state.study_slots.append({"start": "14:00", "end": "16:00"})
        st.rerun()

    # 渲染后再算总时长，读最新值
    total_study_minutes = 0
    for slot in st.session_state.study_slots:
        try:
            sh, sm = map(int, slot["start"].split(":"))
            eh, em = map(int, slot["end"].split(":"))
            start_mins = sh * 60 + sm
            end_mins = (eh % 24) * 60 + em
            if eh == 24 and em == 0:
                end_mins = 24 * 60
            delta = end_mins - start_mins
            if delta < 0:
                delta += 24 * 60
            total_study_minutes += max(0, delta)
        except (ValueError, KeyError):
            pass
    total_study_hours = total_study_minutes / 60

    time_calc_placeholder.markdown(f"""
    <div style="background:#f1f5f9; border-radius:8px; padding:10px 12px; margin-bottom:10px;">
        <div style="font-size:0.72rem; color:#64748b; margin-bottom:2px;">今日预计学习时间</div>
        <div style="font-size:1.3rem; font-weight:700; color:#6366f1;">{total_study_hours:.1f} 小时</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 目标列表
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🎯 我的目标</div>', unsafe_allow_html=True)

    goals = agent._goals
    if goals:
        for g in goals:
            tasks = agent.get_tasks_by_goal(g.id)
            done = sum(1 for t in tasks if t.status.value == "completed")
            total = len(tasks)
            pct = int(done / total * 100) if total > 0 else 0

            st.markdown(f"""
            <div class="goal-item">
                <div class="goal-item-title">{g.title}</div>
                <div class="goal-item-progress">
                    <div class="goal-item-fill" style="width:{pct}%"></div>
                </div>
                <div class="goal-item-meta">
                    <span>{done}/{total} 任务</span>
                    <span>权重 {g.weight}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:24px 12px; color:#94a3b8; font-size:0.82rem;">
            <div style="font-size:28px; margin-bottom:6px; opacity:0.5;">🎯</div>
            还没有目标<br>
            <span style="font-size:0.72rem; opacity:0.8;">在下方聊天框告诉 LifeOS 你的目标</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 快速统计
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 快速统计</div>', unsafe_allow_html=True)

    total_tasks = len(agent._tasks)
    done_tasks = sum(1 for t in agent._tasks if t.status.value == "completed")
    in_progress = sum(1 for t in agent._tasks if t.status.value == "in_progress")
    pending = sum(1 for t in agent._tasks if t.status.value == "pending")
    completion_rate = int(done_tasks / total_tasks * 100) if total_tasks > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">总任务</div>
            <div class="summary-value" style="color:#6366f1;">{total_tasks}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">完成率</div>
            <div class="summary-value" style="color:#22c55e;">{completion_rate}%</div>
        </div>
        """, unsafe_allow_html=True)

    if agent.current_snapshot:
        st.markdown(f"""
        <div class="summary-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.75rem; color:#64748b; font-weight:500;">计划详情</span>
                <span style="font-size:0.7rem; color:#94a3b8;">今日</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:4px;">
                <span style="color:#64748b;">⏱ 计划时长</span>
                <span style="font-weight:600; color:#0f172a;">{agent.current_snapshot.total_scheduled_minutes} 分钟</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                <span style="color:#64748b;">⚠️ 搁置任务</span>
                <span style="font-weight:600; color:#ef4444;">{len(agent.current_snapshot.sacrifice_list)} 个</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 主内容区 — 视图分支
# ==========================================
if main_view != "home":
    render_placeholder_view(main_view)
    st.stop()


# ==========================================
# 主内容区 — 头部（默认 LifeOS 界面）
# ==========================================

# 启动时的环境/配置警告提示
for _w in _env_warnings:
    if _w.strip():
        st.warning(_w, icon="⚠️")
if _env_warnings:
    st.info("配置好之后刷新页面即可生效。", icon="💡")
now_hour = datetime.now().hour
if now_hour < 6:
    greeting = "夜深了"
elif now_hour < 12:
    greeting = "早上好"
elif now_hour < 14:
    greeting = "中午好"
elif now_hour < 18:
    greeting = "下午好"
else:
    greeting = "晚上好"

today_str = date.today().strftime("%m月%d日")
weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
weekday = weekday_names[date.today().weekday()]

st.markdown(f"""
<div class="app-header">
    <div class="app-icon">✨</div>
    <div class="app-title">
        <h1>{greeting}，{profile.name} 👋</h1>
        <p>今天是 {today_str} · {weekday}，让我们一起高效学习</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 仪表盘统计卡
# ==========================================
snapshot = agent.current_snapshot
total_minutes = snapshot.total_scheduled_minutes if snapshot else 0
total_hours = round(total_minutes / 60, 1)
sacrifice_count = len(snapshot.sacrifice_list) if snapshot else 0
hard_tasks = sum(1 for t in agent._tasks if t.deadline_type.value == "hard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon blue">📝</div>
        <div class="stat-content">
            <div class="stat-value">{total_tasks}</div>
            <div class="stat-label">总任务数</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon green">✅</div>
        <div class="stat-content">
            <div class="stat-value">{completion_rate}%</div>
            <div class="stat-label">完成进度</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon purple">⏱️</div>
        <div class="stat-content">
            <div class="stat-value">{total_hours}h</div>
            <div class="stat-label">计划时长</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    if sacrifice_count > 0:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon red">⚠️</div>
            <div class="stat-content">
                <div class="stat-value">{sacrifice_count}</div>
                <div class="stat-label">搁置任务</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon amber">🎯</div>
            <div class="stat-content">
                <div class="stat-value">{hard_tasks}</div>
                <div class="stat-label">硬截止任务</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# Tab 切换
# ==========================================
tab_dashboard, tab_schedule, tab_plan, tab_tasks, tab_history = st.tabs([
    "  📊  概览  ",
    "  📖  课表  ",
    "  📅  今日计划  ",
    "  📋  任务看板  ",
    "  📜  历史快照  ",
])


# ==========================================
# Tab 1: 概览仪表盘
# ==========================================
with tab_dashboard:
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("#### 📅 今日安排")

        today = date.today()
        if snapshot and snapshot.time_slots:
            today_slots = [
                ts for ts in snapshot.time_slots
                if ts.start_time.startswith(str(today))
            ]

            if today_slots:
                st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
                for idx, ts in enumerate(today_slots[:8]):
                    start = ts.start_time.split("T")[1][:5]
                    end = ts.end_time.split("T")[1][:5]
                    duration = int((datetime.fromisoformat(ts.end_time)
                                  - datetime.fromisoformat(ts.start_time)).total_seconds() / 60)

                    src = ts.source.value if hasattr(ts.source, 'value') else ts.source
                    task = agent.get_task(ts.task_id) if ts.task_id else None
                    title = task.title if task else ts.title or "安排"
                    energy = task.energy_level.value if task and task.energy_level else "medium"

                    # 圆点颜色
                    if src == "commitment":
                        dot_color = "#f59e0b"
                        card_class = "commitment"
                    elif src == "buffer":
                        dot_color = "#94a3b8"
                        card_class = "buffer"
                    else:
                        dot_color = "#6366f1"
                        card_class = "task"

                    tags = []
                    if task and task.deadline_type.value == "hard":
                        tags.append('<span class="tag tag-hard">硬截止</span>')
                    if task:
                        tags.append(f'<span class="tag tag-{energy}">{energy_icon(energy)}</span>')

                    st.markdown(f"""
                    <div class="timeline-item" style="animation-delay: {idx * 0.05}s;">
                        <div class="timeline-time">{start}</div>
                        <div class="timeline-dot" style="background:{dot_color};"></div>
                        <div class="timeline-card {card_class}">
                            <div class="timeline-title">{title}</div>
                            <div class="timeline-meta">
                                <span>⏱ {duration} 分钟</span>
                                {" ".join(tags)}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if len(today_slots) > 8:
                    st.markdown(f"""
                    <div style="text-align:center; padding:10px 0 4px; color:#94a3b8; font-size:0.78rem;
                                cursor:pointer; transition: color 0.2s;" onmouseover="this.style.color='#64748b'">
                        还有 {len(today_slots) - 8} 个安排 → 切换到「今日计划」查看全部
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">🌅</div>
                    <div class="empty-state-text">今天还没有安排</div>
                    <div class="empty-state-sub">在下方聊天框设定你的第一个目标吧</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🎯</div>
                <div class="empty-state-text">还没有计划</div>
                <div class="empty-state-sub">告诉我你的目标，我来帮你安排</div>
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        # 风险提示（完成进度与任务分布已移至侧栏）
        if sacrifice_count > 0:
            st.markdown("#### ⚠️ 需要关注")
            for s in (snapshot.sacrifice_list if snapshot else [])[:3]:
                st.markdown(f"""
                <div class="sacrifice-card">
                    <div class="sacrifice-title">{s.task_title}</div>
                    <div class="sacrifice-reason">{s.reason}</div>
                </div>
                """, unsafe_allow_html=True)


# ==========================================
# Tab 2: 学生课表
# ==========================================
with tab_schedule:
    st.markdown("#### 学生课表")

    classes = [
        c for c in agent._commitments
        if commitment_is_class(c)
    ]
    edit_course_id = st.session_state.get("edit_course_id")
    editing_course = next(
        (c for c in classes if c.id == edit_course_id),
        None,
    )
    if edit_course_id and editing_course is None:
        st.session_state.pop("edit_course_id", None)

    # 修改已有课程
    if editing_course is not None:
        st.markdown(f"#### 修改课程：{editing_course.title}")
        with st.form("edit_course_form"):
            edit_title = st.text_input("课程名称", value=editing_course.title)
            edit_weekday = st.selectbox(
                "上课日",
                range(7),
                index=commitment_weekday(editing_course),
                format_func=lambda i: WEEKDAY_NAMES[i],
                key="edit_weekday_select",
            )
            edit_start = st.time_input(
                "开始时间",
                value=datetime.fromisoformat(editing_course.start_time).time(),
                step=1800,
            )
            edit_end = st.time_input(
                "结束时间",
                value=datetime.fromisoformat(editing_course.end_time).time(),
                step=1800,
            )
            edit_location = st.text_input(
                "地点",
                value=editing_course.description or "",
            )
            recurrence_value = getattr(
                editing_course.recurrence, "value", editing_course.recurrence
            )
            edit_recurrence = st.selectbox(
                "重复方式",
                ["每周", "仅一次"],
                index=0 if recurrence_value == "weekly" else 1,
            )
            save_edit = st.form_submit_button("保存修改", type="primary")

        if st.button("取消修改", key="cancel_edit_course"):
            st.session_state.pop("edit_course_id", None)
            st.rerun()

        if save_edit:
            if not edit_title.strip():
                st.error("课程名称不能为空")
            elif edit_end <= edit_start:
                st.error("结束时间必须晚于开始时间")
            else:
                anchor = next_weekday_date(edit_weekday, edit_start)
                agent.update_commitment(
                    editing_course.id,
                    title=edit_title.strip(),
                    start_time=(
                        f"{anchor.isoformat()}T{edit_start.strftime('%H:%M:%S')}"
                    ),
                    end_time=(
                        f"{anchor.isoformat()}T{edit_end.strftime('%H:%M:%S')}"
                    ),
                    recurrence="weekly" if edit_recurrence == "每周" else "none",
                    description=edit_location.strip(),
                )
                st.session_state.pop("edit_course_id", None)
                agent.force_replan("课表已更新")
                st.rerun()

    st.markdown("#### 新增课程")
    with st.form("add_course_form", clear_on_submit=True):
        add_title = st.text_input("课程名称")
        add_weekday = st.selectbox(
            "上课日",
            range(7),
            format_func=lambda i: WEEKDAY_NAMES[i],
            key="add_weekday_select",
        )
        add_start = st.time_input("开始时间", value=dt_time(8, 0), step=1800)
        add_end = st.time_input("结束时间", value=dt_time(9, 35), step=1800)
        add_location = st.text_input("地点")
        add_recurrence = st.selectbox(
            "重复方式",
            ["每周", "仅一次"],
        )
        submit_add = st.form_submit_button("加入课表", type="primary")

    if submit_add:
        if not add_title.strip():
            st.error("课程名称不能为空")
        elif add_end <= add_start:
            st.error("结束时间必须晚于开始时间")
        else:
            anchor = next_weekday_date(add_weekday, add_start)
            agent.add_commitment(Commitment(
                profile_id=profile.id or "default",
                title=add_title.strip(),
                type=CommitmentType.CLASS,
                start_time=f"{anchor.isoformat()}T{add_start.strftime('%H:%M:%S')}",
                end_time=f"{anchor.isoformat()}T{add_end.strftime('%H:%M:%S')}",
                recurrence="weekly" if add_recurrence == "每周" else "none",
                description=add_location.strip(),
            ))
            agent.force_replan("课表已更新")
            st.rerun()

    if classes:
        sorted_classes = sorted(
            classes,
            key=lambda c: (
                commitment_weekday(c),
                datetime.fromisoformat(c.start_time).time(),
            ),
        )
        st.markdown("#### 当前课程")
        for course in sorted_classes:
            col_name, col_weekday, col_time, col_loc, col_action = st.columns(
                [2.2, 1.0, 1.2, 1.4, 1.6]
            )
            with col_name:
                st.markdown(f"**{course.title}**")
            with col_weekday:
                st.markdown(WEEKDAY_NAMES[commitment_weekday(course)])
            with col_time:
                st.markdown(commitment_time_label(course))
            with col_loc:
                st.markdown(course.description or "未填地点")
            with col_action:
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("修改", key=f"edit_course_{course.id}"):
                        st.session_state["edit_course_id"] = course.id
                        st.rerun()
                with btn_col2:
                    if st.button("删除", key=f"delete_course_{course.id}"):
                        agent.remove_commitment(course.id)
                        agent.force_replan("课表已更新")
                        st.session_state.pop("edit_course_id", None)
                        st.rerun()
            st.divider()
    else:
        st.markdown("还没有课程，先在上方添加第一门课。")


# ==========================================
# Tab 3: 今日计划（时间轴）
# ==========================================
with tab_plan:
    col_date, _ = st.columns([1, 3])
    with col_date:
        selected_date = st.date_input(
            "选择日期",
            value=st.session_state.current_date,
            label_visibility="collapsed",
            key="plan_date_tab",
        )
    st.session_state.current_date = selected_date

    date_title = selected_date.strftime("%Y年%m月%d日")
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekday_names[selected_date.weekday()]

    st.markdown(f"""
    <div class="date-header">
        <div>
            <div class="date-title">{date_title} · {weekday}</div>
            <div class="date-subtitle">今日学习计划</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if snapshot and snapshot.time_slots:
        day_slots = [
            ts for ts in snapshot.time_slots
            if ts.start_time.startswith(str(selected_date))
        ]

        if day_slots:
            # 统计
            task_minutes = sum(
                int((datetime.fromisoformat(ts.end_time) - datetime.fromisoformat(ts.start_time)).total_seconds() / 60)
                for ts in day_slots
                if (ts.source.value if hasattr(ts.source, 'value') else ts.source) == "task"
            )
            commitment_count = sum(
                1 for ts in day_slots
                if (ts.source.value if hasattr(ts.source, 'value') else ts.source) == "commitment"
            )
            st.markdown(f"""
            <div style="display:flex; gap:10px; margin-bottom:20px; flex-wrap:wrap;">
                <div style="background:linear-gradient(135deg, #eff6ff, #dbeafe); padding:7px 14px; border-radius:20px; font-size:0.78rem; color:#1d4ed8; font-weight:500;">
                    📚 学习 {task_minutes} 分钟
                </div>
                <div style="background:linear-gradient(135deg, #f0fdf4, #dcfce7); padding:7px 14px; border-radius:20px; font-size:0.78rem; color:#16a34a; font-weight:500;">
                    📝 共 {len(day_slots)} 个安排
                </div>
                {f'<div style="background:linear-gradient(135deg, #fffbeb, #fef3c7); padding:7px 14px; border-radius:20px; font-size:0.78rem; color:#d97706; font-weight:500;">📌 {commitment_count} 个承诺</div>' if commitment_count > 0 else ''}
            </div>
            """, unsafe_allow_html=True)

            # 时间轴
            st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
            for idx, ts in enumerate(day_slots):
                start = ts.start_time.split("T")[1][:5]
                end = ts.end_time.split("T")[1][:5]
                duration = int((datetime.fromisoformat(ts.end_time)
                              - datetime.fromisoformat(ts.start_time)).total_seconds() / 60)

                src = ts.source.value if hasattr(ts.source, 'value') else ts.source
                task = agent.get_task(ts.task_id) if ts.task_id else None
                title = task.title if task else ts.title or "安排"
                energy = task.energy_level.value if task and task.energy_level else "medium"

                if src == "commitment":
                    dot_color = "#f59e0b"
                    card_class = "commitment"
                elif src == "buffer":
                    dot_color = "#94a3b8"
                    card_class = "buffer"
                    title = "☕ 休息缓冲"
                else:
                    dot_color = "#6366f1"
                    card_class = "task"

                tags = []
                if task and task.deadline_type.value == "hard":
                    tags.append('<span class="tag tag-hard">硬截止</span>')
                if task:
                    tags.append(f'<span class="tag tag-{energy}">{energy_icon(energy)}</span>')

                st.markdown(f"""
                <div class="timeline-item" style="animation-delay: {idx * 0.04}s;">
                    <div class="timeline-time">{start} - {end}</div>
                    <div class="timeline-dot" style="background:{dot_color};"></div>
                    <div class="timeline-card {card_class}">
                        <div class="timeline-title">{title}</div>
                        <div class="timeline-meta">
                            <span>⏱ {duration} 分钟</span>
                            {" ".join(tags)}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # 牺牲清单
            if snapshot.sacrifice_list:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### ⚠️ 暂时搁置的任务")
                for s in snapshot.sacrifice_list[:5]:
                    st.markdown(f"""
                    <div class="sacrifice-card">
                        <div class="sacrifice-title">{s.task_title}</div>
                        <div class="sacrifice-reason">{s.reason}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="empty-state">
                <div class="empty-state-icon">🌴</div>
                <div class="empty-state-text">{weekday}没有安排</div>
                <div class="empty-state-sub">是休息日吗？好好休息吧！</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🎯</div>
            <div class="empty-state-text">还没有计划</div>
            <div class="empty-state-sub">在下方聊天框告诉我你的目标吧</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# Tab 3: 任务看板
# ==========================================
with tab_tasks:
    if agent._tasks:
        pending_tasks = [t for t in agent._tasks if t.status.value == "pending"]
        in_progress_tasks = [t for t in agent._tasks if t.status.value == "in_progress"]
        completed_tasks = [t for t in agent._tasks if t.status.value == "completed"]

        # 按优先级排序
        pending_tasks.sort(key=lambda x: -x.priority)
        in_progress_tasks.sort(key=lambda x: -x.priority)

        col1, col2, col3 = st.columns(3)

        # 待开始
        with col1:
            st.markdown(f"""
            <div class="kanban-header">
                <h3>⏳ 待开始</h3>
                <span class="kanban-count">{len(pending_tasks)}</span>
            </div>
            """, unsafe_allow_html=True)

            for idx, t in enumerate(pending_tasks[:15]):
                hard_badge = '<span class="tag tag-hard">硬截止</span>' if t.deadline_type.value == "hard" else ""
                energy_badge = f'<span class="tag tag-{t.energy_level.value}">{energy_icon(t.energy_level.value)}</span>'
                pri_text, pri_class = priority_label(t.priority)
                pri_badge = f'<span class="tag {pri_class}">{pri_text}</span>'

                st.markdown(f"""
                <div class="task-card" style="animation-delay: {idx * 0.03}s;">
                    <div class="task-card-header">
                        <div class="task-card-title">{t.title}</div>
                        <div class="task-card-badges">
                            {pri_badge}
                        </div>
                    </div>
                    <div class="task-progress-bar">
                        <div class="task-progress-fill low" style="width:0%"></div>
                    </div>
                    <div class="task-card-footer">
                        <span>⏱ {t.estimated_minutes} 分钟</span>
                        <span style="display:flex; gap:4px;">{hard_badge} {energy_badge}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if len(pending_tasks) > 15:
                st.markdown(f"""
                <div style="text-align:center; padding:8px; color:#94a3b8; font-size:0.75rem;">
                    还有 {len(pending_tasks) - 15} 个...
                </div>
                """, unsafe_allow_html=True)

        # 进行中
        with col2:
            st.markdown(f"""
            <div class="kanban-header">
                <h3>🔄 进行中</h3>
                <span class="kanban-count">{len(in_progress_tasks)}</span>
            </div>
            """, unsafe_allow_html=True)

            for idx, t in enumerate(in_progress_tasks[:15]):
                pct = int(t.progress * 100)
                progress_class = "mid" if pct < 70 else "high"
                hard_badge = '<span class="tag tag-hard">硬截止</span>' if t.deadline_type.value == "hard" else ""
                energy_badge = f'<span class="tag tag-{t.energy_level.value}">{energy_icon(t.energy_level.value)}</span>'
                pri_text, pri_class = priority_label(t.priority)
                pri_badge = f'<span class="tag {pri_class}">{pri_text}</span>'

                st.markdown(f"""
                <div class="task-card" style="animation-delay: {idx * 0.03}s;">
                    <div class="task-card-header">
                        <div class="task-card-title">{t.title}</div>
                        <div class="task-card-badges">
                            {pri_badge}
                        </div>
                    </div>
                    <div class="task-progress-bar">
                        <div class="task-progress-fill {progress_class}" style="width:{pct}%"></div>
                    </div>
                    <div class="task-card-footer">
                        <span style="font-weight:600; color:#334155;">{pct}%</span>
                        <span style="display:flex; gap:4px;">{hard_badge} {energy_badge}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 已完成
        with col3:
            st.markdown(f"""
            <div class="kanban-header">
                <h3>✅ 已完成</h3>
                <span class="kanban-count">{len(completed_tasks)}</span>
            </div>
            """, unsafe_allow_html=True)

            for idx, t in enumerate(completed_tasks[:15]):
                st.markdown(f"""
                <div class="task-card completed" style="animation-delay: {idx * 0.03}s;">
                    <div class="task-card-header">
                        <div class="task-card-title">✅ {t.title}</div>
                    </div>
                    <div class="task-progress-bar">
                        <div class="task-progress-fill high" style="width:100%"></div>
                    </div>
                    <div class="task-card-footer">
                        <span>已完成</span>
                        <span style="color:#22c55e; font-weight:600;">100%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if len(completed_tasks) > 15:
                st.markdown(f"""
                <div style="text-align:center; padding:8px; color:#94a3b8; font-size:0.75rem;">
                    还有 {len(completed_tasks) - 15} 个...
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <div class="empty-state-text">还没有任务</div>
            <div class="empty-state-sub">开始设定你的第一个目标吧</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# Tab 4: 历史快照
# ==========================================
with tab_history:
    if len(agent._snapshots) > 0:
        st.markdown("#### 📜 计划变更历史")
        st.markdown("<p style='color:#64748b; font-size:0.82rem; margin-bottom:16px;'>每次计划调整都会生成一个快照，方便你追溯变化。</p>", unsafe_allow_html=True)

        for i, snap in enumerate(reversed(agent._snapshots)):
            version = len(agent._snapshots) - i
            reason = snap.trigger_reason or "初始计划"
            task_count = len(snap.time_slots)
            minutes = snap.total_scheduled_minutes
            created = snap.created_at

            # 格式化原因
            reason_map = {
                "initial": "🎯 初始计划",
                "progress_report": "📊 进度更新",
                "new_goal": "➕ 新增目标",
                "new_event": "📢 新事件",
                "status_query": "🔍 状态查询",
            }
            reason_display = reason_map.get(reason, f"🔄 {reason}")

            # 格式化时间
            try:
                created_dt = datetime.fromisoformat(created)
                created_display = created_dt.strftime("%m-%d %H:%M")
            except (ValueError, TypeError):
                created_display = created

            with st.container():
                st.markdown(f"""
                <div class="snapshot-item" style="animation-delay: {i * 0.05}s;">
                    <div class="snapshot-version">v{version}</div>
                    <div class="snapshot-info">
                        <div class="snapshot-reason">{reason_display}</div>
                        <div class="snapshot-meta">
                            {created_display} · {task_count} 个时间段 · {minutes} 分钟
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if snap.change_log and snap.change_log.detail:
                    with st.expander("查看变更详情"):
                        st.markdown(snap.change_log.detail)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📜</div>
            <div class="empty-state-text">还没有历史快照</div>
            <div class="empty-state-sub">制定计划后，每次调整都会记录在这里</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 聊天界面
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 💬 对话")

# 聊天消息
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding:40px 10px; color:#94a3b8;">
            <div style="font-size:40px; margin-bottom:10px; opacity:0.6;">💬</div>
            <div style="font-size:0.9rem; font-weight:500; color:#64748b;">开始和 LifeOS 对话吧</div>
            <div style="font-size:0.75rem; margin-top:4px;">试试说"我要准备期末考试"或"看看我的进度"</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg_idx, msg in enumerate(st.session_state.chat_history):
            time_str = datetime.now().strftime("%H:%M")
            msg_type = msg.get("type", "text")

            if msg["role"] == "user":
                if msg_type == "image":
                    # 图片消息
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div class="chat-avatar user">👤</div>
                        <div class="chat-bubble-wrap">
                            <div class="chat-bubble user" style="padding:6px;">
                                📷 {msg['content'].replace('[截图] ', '')}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message user">
                        <div class="chat-avatar user">👤</div>
                        <div class="chat-bubble-wrap">
                            <div class="chat-bubble user">{msg['content']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                content = msg['content'].replace('\n', '<br>')
                references = msg.get("references", [])

                # 引用展示
                ref_html = ""
                if references:
                    ref_items = []
                    for ref in references:
                        title = ref.get("document_title", ref.get("document_id", ""))
                        score = ref.get("score", 0)
                        score_pct = int(score * 100)
                        ref_items.append(f"""
                        <div style="padding:6px 10px; background:#f8fafc; border-radius:6px; margin-top:4px; border-left:2px solid #6366f1;">
                            <div style="font-size:0.72rem; font-weight:600; color:#4f46e5;">
                                [{ref.get('index', i)}] {title} · {score_pct}% 相关
                            </div>
                            <div style="font-size:0.7rem; color:#64748b; margin-top:2px; line-height:1.4; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;">
                                {ref.get("content", "")[:150]}...
                            </div>
                        </div>
                        """)
                    ref_html = f"""
                    <div style="margin-top:8px; padding-top:8px; border-top:1px solid #e2e8f0;">
                        <div style="font-size:0.7rem; color:#94a3b8; margin-bottom:4px;">📚 参考资料</div>
                        {''.join(ref_items)}
                    </div>
                    """

                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="chat-avatar bot">✨</div>
                    <div class="chat-bubble-wrap">
                        <div class="chat-bubble bot">{content}{ref_html}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if msg_type == "screenshot_result":
                    item_count = len(msg.get("parsed_items", []))
                    if st.button(
                        f"确认并导入这 {item_count} 条",
                        key=f"confirm_screenshot_{msg_idx}",
                        type="secondary",
                    ):
                        pending = st.session_state.get("pending_screenshot")
                        if pending and pending.parsed_screenshot:
                            import_result = agent.apply_parsed_items(
                                pending.parsed_screenshot.items
                            )
                            st.session_state.chat_history.append({
                                "role": "bot",
                                "content": import_result.reply,
                            })
                            st.session_state.pending_screenshot = None
                            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# 输入框
user_input = st.chat_input("告诉我你的目标、进度，或者问我计划安排...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("思考中..."):
        # 使用带知识库的对话
        if hasattr(agent, 'chat_with_knowledge') and getattr(agent, 'retriever', None):
            result = agent.chat_with_knowledge(user_input, use_rag=True)
        else:
            result = agent.run(user_input)

    # 构建 bot 消息
    bot_msg = {"role": "bot", "content": result.reply}
    # 带上 RAG 引用
    if hasattr(result, 'rag_references') and result.rag_references:
        bot_msg["references"] = result.rag_references
    st.session_state.chat_history.append(bot_msg)
    st.rerun()

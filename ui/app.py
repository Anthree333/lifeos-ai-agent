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
import re
import html
import json
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
        display: flex !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
        overflow: visible !important;
        z-index: 99999 !important;
    }
    body:has([data-testid="stExpandSidebarButton"])
        [data-testid="stHeader"] > *:not([data-testid="stExpandSidebarButton"]) {
        visibility: hidden !important;
    }
    /* 强制侧栏折叠/展开按钮始终可见 */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] * {
        visibility: visible !important;
        opacity: 1 !important;
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
        padding: 10px 12px;
        margin-bottom: 8px;
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
        margin-bottom: 6px;
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
    .snap-chip {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 999px;
        border: 1px solid;
        font-size: 0.68rem;
        font-weight: 600;
        line-height: 1.7;
        margin-right: 4px;
    }
    .snap-summary {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 6px;
        background: #f8fafc;
        border-radius: 8px;
        padding: 6px 10px;
    }
    .snap-diff-row {
        display: flex;
        align-items: baseline;
        gap: 8px;
        padding: 5px 0;
        border-bottom: 1px dashed #f1f5f9;
        font-size: 0.8rem;
    }
    .snap-diff-row:last-child {
        border-bottom: none;
    }
    .snap-diff-title {
        font-weight: 600;
        color: #334155;
    }
    .snap-diff-extra {
        color: #94a3b8;
        font-size: 0.74rem;
    }
    .snap-slot-row {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 4px 0;
        font-size: 0.78rem;
        color: #475569;
        border-bottom: 1px dashed #f8fafc;
    }
    .snap-slot-min {
        color: #94a3b8;
        font-size: 0.72rem;
        margin-left: auto;
    }
    .snap-section-label {
        font-size: 0.76rem;
        color: #94a3b8;
        font-weight: 600;
        margin: 6px 0 4px;
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
    /* ===== 侧栏实时时钟（iframe 内自走，见 render_live_clock） ===== */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 12px;
    }
    section[data-testid="stSidebar"] iframe {
        display: block;
        background: transparent;
        border: none;
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


# ==========================================
# 手动删除：安排（时间块）与任务
# ==========================================

def _slot_source(ts) -> str:
    """时间块的来源字符串。"""
    return ts.source.value if hasattr(ts.source, "value") else str(ts.source)


def _slot_is_removable(ts) -> bool:
    """该时间块是否可以被手动删除（缓冲/休息不支持）。"""
    src = _slot_source(ts)
    return bool((src == "task" and ts.task_id) or (src == "commitment" and ts.commitment_id))


def _unscheduled_active_tasks(agent, snapshot) -> list:
    """获取未被安排进快照、但状态为待开始/进行中的任务。

    这些任务在任务看板中可见，但调度器未给它们分配时间槽
    （例如截止时间已过、被牺牲等），需在今日计划中补显。

    同时包含刚被标记完成的任务（session_state 中记录），
    以便显示撤销入口，避免误点完成后无法回退。
    """
    scheduled_task_ids = set()
    if snapshot and snapshot.time_slots:
        for ts in snapshot.time_slots:
            if ts.task_id:
                scheduled_task_ids.add(ts.task_id)

    # 刚被标记完成、需要显示撤销按钮的任务 ID
    just_done_ids = {
        tid for tid, done in st.session_state.items()
        if tid.startswith("unsched_done_") and done
    }

    result = []
    for t in agent._tasks:
        if t.id in scheduled_task_ids:
            continue
        if t.status.value in ("pending", "in_progress"):
            result.append(t)
        elif f"unsched_done_{t.id}" in just_done_ids:
            result.append(t)
    return result


def _task_deadline_date(task):
    """提取任务截止日期（date 对象），无截止日期返回 None。"""
    if not task.deadline:
        return None
    try:
        return datetime.fromisoformat(task.deadline).date()
    except (ValueError, TypeError):
        return None


def _render_unscheduled_task_row(agent, task, key_prefix: str, compact: bool = False):
    """渲染一条带完成按钮的待安排任务，支持误点后撤销。

    完成后任务不会立即从列表消失，而是变为已完成样式并显示「撤销」按钮，
    点击撤销调用 uncomplete_task 恢复为待开始状态。
    """
    # 该任务是否刚被标记完成（用于显示撤销入口）
    done_key = f"unsched_done_{task.id}"
    is_done = task.status.value == "completed" or st.session_state.get(done_key, False)

    deadline_str = ""
    if task.deadline:
        try:
            dl = datetime.fromisoformat(task.deadline)
            deadline_str = dl.strftime("%H:%M") if compact else dl.strftime("%m-%d %H:%M")
        except (ValueError, TypeError):
            pass
    hard_badge = " 🔴硬截止" if task.deadline_type.value == "hard" else ""
    energy = task.energy_level.value if task.energy_level else "medium"

    if is_done:
        # 已完成样式：绿色、删除线、撤销按钮
        if compact:
            c_card, c_btn = st.columns([5, 1])
            with c_card:
                st.markdown(f"""
                <div style="border-left:3px solid rgba(22,163,74,0.45);padding:6px 10px;margin:4px 0;border-radius:0 6px 6px 0;background:#f0fdf4;">
                    <span style="font-weight:600;font-size:0.85rem;text-decoration:line-through;color:#86efac;">{task.title}</span>
                    <span style="font-size:0.7rem;color:#16a34a;margin-left:6px;font-weight:600;">✓ 已完成</span>
                </div>
                """, unsafe_allow_html=True)
            with c_btn:
                if st.button("↩", key=f"{key_prefix}_undo_{task.id}",
                             help="撤销完成", use_container_width=True):
                    agent.uncomplete_task(task.id)
                    st.session_state[done_key] = False
                    st.rerun()
        else:
            c_card, c_btn = st.columns([5, 1])
            with c_card:
                st.markdown(f"""
                <div style="border-left:3px solid rgba(22,163,74,0.45);padding:8px 12px;margin-bottom:6px;border-radius:0 8px 8px 0;background:#f0fdf4;">
                    <div style="font-weight:600;font-size:0.9rem;text-decoration:line-through;color:#86efac;">{task.title}</div>
                    <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">
                        ⏱ {task.remaining_minutes} 分钟{hard_badge}
                        {f' · 截止 {deadline_str}' if deadline_str else ''}
                    </div>
                    <div style="color:#16a34a;font-size:0.72rem;font-weight:600;margin-top:2px;">✓ 已完成（点击右侧撤销可恢复）</div>
                </div>
                """, unsafe_allow_html=True)
            with c_btn:
                if st.button("↩ 撤销", key=f"{key_prefix}_undo_{task.id}",
                             help="撤销完成，恢复为待开始", use_container_width=True):
                    agent.uncomplete_task(task.id)
                    st.session_state[done_key] = False
                    st.rerun()
        return

    # 未完成样式：橙色、完成按钮
    if compact:
        c_card, c_btn = st.columns([5, 1])
        with c_card:
            st.markdown(f"""
            <div style="border-left:3px solid #f59e0b;padding:6px 10px;margin:4px 0;border-radius:0 6px 6px 0;background:#fffbeb;">
                <span style="font-weight:600;font-size:0.85rem;">{task.title}</span>
                <span style="font-size:0.72rem;color:#92400e;margin-left:6px;">
                    ⏱{task.remaining_minutes}分{f' · 截止{deadline_str}' if deadline_str else ''}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with c_btn:
            if st.button("☐", key=f"{key_prefix}_{task.id}",
                         help=f"标记完成：{task.title}", use_container_width=True):
                agent.complete_task(task.id)
                st.session_state[done_key] = True
                st.rerun()
    else:
        c_card, c_btn = st.columns([5, 1])
        with c_card:
            st.markdown(f"""
            <div style="border-left:3px solid #f59e0b;padding:8px 12px;margin-bottom:6px;border-radius:0 8px 8px 0;background:#fffbeb;">
                <div style="font-weight:600;font-size:0.9rem;">{task.title}</div>
                <div style="font-size:0.72rem;color:#92400e;margin-top:2px;">
                    ⏱ {task.remaining_minutes} 分钟{hard_badge}
                    {f' · {energy_icon(energy)} {energy}' if energy else ''}
                    {f' · 截止 {deadline_str}' if deadline_str else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_btn:
            if st.button("☐ 完成", key=f"{key_prefix}_{task.id}",
                         help=f"标记完成：{task.title}", use_container_width=True):
                agent.complete_task(task.id)
                st.session_state[done_key] = True
                st.rerun()


def _delete_slot(agent, ts) -> tuple:
    """删除一个安排：任务块 → 删任务；固定安排 → 删承诺。返回 (是否成功, 名称)。"""
    src = _slot_source(ts)
    if src == "task" and ts.task_id:
        task = agent.get_task(ts.task_id)
        label = task.title if task else (ts.title or "安排")
        return agent.remove_task(ts.task_id), label
    if src == "commitment" and ts.commitment_id:
        return agent.remove_commitment(ts.commitment_id), (ts.title or "固定安排")
    return False, (ts.title or "安排")


def _delete_task(agent, task) -> tuple:
    """删除单个任务，返回 (是否成功, 名称)。"""
    return agent.remove_task(task.id), task.title


def _render_delete_control(key_base: str, help_text: str) -> bool:
    """带二次确认的删除按钮，返回 True 表示用户已确认删除。"""
    confirm_key = f"{key_base}_confirm"
    if st.session_state.get(confirm_key):
        c_yes, c_no = st.columns(2)
        if c_yes.button("✔", key=f"{key_base}_yes", help="确认删除",
                        use_container_width=True):
            st.session_state.pop(confirm_key, None)
            return True
        if c_no.button("✘", key=f"{key_base}_no", help="取消删除",
                       use_container_width=True):
            st.session_state.pop(confirm_key, None)
            st.rerun()
        return False

    if st.button("🗑", key=key_base, help=help_text, use_container_width=True):
        st.session_state[confirm_key] = True
        st.rerun()
    return False


def _apply_delete(agent, ok: bool, label: str) -> None:
    """删除后统一处理：重排、写提示。"""
    if ok:
        agent.force_replan(reason=f"手动删除：{label}")
        st.session_state["delete_notice"] = f"已删除「{label}」，计划已重新安排"
    else:
        st.session_state["delete_notice"] = f"删除失败：「{label}」可能已经不存在"


def _render_delete_notice() -> None:
    """渲染删除 / 开始等操作的结果提示（显示一次即清除）。"""
    notice = st.session_state.pop("delete_notice", None)
    if notice is None:
        notice = st.session_state.pop("action_notice", None)
    if notice:
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;color:#15803d;
                    border-radius:10px;padding:8px 12px;font-size:0.82rem;margin-bottom:12px;">
            ✅ {notice}
        </div>
        """, unsafe_allow_html=True)


WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def render_live_clock() -> None:
    """侧栏顶部实时日期时钟：iframe 内前端每秒自走。

    不用 st.fragment(run_every=1)：高频 fragment 重跑会干扰主区文本输入的
    提交（输入丢失），也让 AppTest 永远等不到静止态；iframe 内脚本可靠执行、
    零服务端重跑。
    """
    weekdays_js = json.dumps(WEEKDAY_NAMES, ensure_ascii=False)
    st.html(f"""
    <div class="live-clock-banner">
        <div class="clock-date" id="live-clock-date"></div>
        <div class="clock-time" id="live-clock-time"></div>
    </div>
    <style>
        html, body {{
            background: transparent !important; margin: 0; padding: 0;
            overflow: hidden;
            font-family: 'Microsoft YaHei', 'PingFang SC', system-ui, sans-serif;
        }}
        .live-clock-banner {{
            box-sizing: border-box; width: 100%;
            padding: 12px 18px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 0 0 16px 16px;
            box-shadow: 0 8px 22px rgba(99, 102, 241, 0.28);
            color: #ffffff; text-align: center;
        }}
        .live-clock-banner .clock-date {{
            font-size: 0.75rem; font-weight: 500; letter-spacing: 0.5px; opacity: 0.88;
        }}
        .live-clock-banner .clock-time {{
            font-size: 1.75rem; font-weight: 700; letter-spacing: 1.5px;
            line-height: 1.15; margin-top: 1px; font-variant-numeric: tabular-nums;
        }}
    </style>
    <script>
    (function() {{
        const weekdays = {weekdays_js};
        function pad(n) {{ return String(n).padStart(2, '0'); }}
        function updateClock() {{
            const d = new Date();
            const dateEl = document.getElementById('live-clock-date');
            const timeEl = document.getElementById('live-clock-time');
            if (dateEl) dateEl.textContent = d.getFullYear() + '年' + pad(d.getMonth() + 1) + '月' + pad(d.getDate()) + '日 · ' + weekdays[(d.getDay() + 6) % 7];
            if (timeEl) timeEl.textContent = pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
        }}
        updateClock();
        setInterval(updateClock, 1000);
    }})();
    </script>
    """, unsafe_allow_javascript=True)


# 侧栏导航视图：view_key -> (图标, 名称, 占位提示)
NAV_VIEWS = {
    "knowledge": ("📚", "知识库", "上传课件与资料，AI 为你问答"),
    "notes": ("📝", "笔记", "记录学习笔记与灵感"),
    "schedule": ("📅", "日程", "管理你的日程安排"),
    "courses": ("📖", "课表", "查看与维护每周课表"),
}


def render_placeholder_view(view_key: str) -> None:
    """侧栏导航视图：courses 渲染真实课表，其余为占位界面。"""
    icon, label, sub = NAV_VIEWS.get(view_key, ("✨", "LifeOS", ""))

    # 课表视图：渲染完整的课程管理界面
    if view_key == "courses":
        render_courses_view()
        return

    # 笔记视图：渲染真实的笔记管理界面
    if view_key == "notes":
        render_notes_view()
        return

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


# ==========================================
# 笔记视图（📝 Notes）
# ==========================================

_NOTES_CSS = """
<style>
/* ---- 笔记视图（仿 ima）：左时间分组列表 + 右文档式编辑器 ---- */
.nt-all { font-size: 16px; font-weight: 700; color: #0f172a; padding: 4px 0 6px; }
.nt-group {
    font-size: 12px; color: #94a3b8; font-weight: 600;
    margin: 14px 0 2px; letter-spacing: 0.3px;
}
/* 列表行：幽灵按钮当标题行，meta 行显示日期与摘要 */
/* 主区列表行：幽灵按钮（仅笔记视图注入时生效，侧栏不受影响） */
[data-testid="stMain"] button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    text-align: left !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #0f172a !important;
    padding: 6px 10px !important;
    border-radius: 8px !important;
}
[data-testid="stMain"] button[kind="secondary"]:hover { background: #f1f5f9 !important; }
[data-testid="stMain"] button[kind="primary"] {
    background: #e8ebfd !important;
    color: #4f46e5 !important;
    box-shadow: none !important;
    border: none !important;
}
.nt-row-meta {
    font-size: 12px; color: #94a3b8; padding: 0 10px 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
/* 右侧文档式编辑器（按 aria-label 定位，无需容器包裹） */
input[aria-label="笔记标题"] {
    font-size: 21px !important; font-weight: 700 !important;
    border: none !important; box-shadow: none !important;
    padding-left: 2px !important; color: #0f172a !important;
}
textarea[aria-label="正文"] {
    font-size: 15px !important; line-height: 1.75 !important;
    border: none !important; box-shadow: none !important;
    background: transparent !important; padding-left: 2px !important;
}
input[aria-label="笔记标签"] { font-size: 13px !important; }
.nt-wc { text-align: right; font-size: 12px; color: #94a3b8; padding: 4px 2px 0; }
.nt-saved { font-size: 12px; color: #10b981; text-align: right; padding: 8px 0 0; }
.nt-empty { text-align: center; color: #94a3b8; padding: 30px 10px; font-size: 13px; line-height: 1.8; }
.nt-hint {
    border: 1.5px dashed #cbd5e1;
    border-radius: 16px;
    padding: 60px 20px;
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
}
.nt-hint b { color: #6366f1; }
.nt-hint .big { font-size: 42px; margin-bottom: 10px; }
</style>
"""


def _nt_notes() -> list:
    """笔记列表缓存（懒加载，编辑后即时刷新）。"""
    db = get_db()
    if "nt_notes" not in st.session_state:
        st.session_state["nt_notes"] = db.list_notes()
    return st.session_state["nt_notes"]


def _nt_reload() -> list:
    """重新从数据库读取笔记列表。"""
    db = get_db()
    st.session_state["nt_notes"] = db.list_notes()
    return st.session_state["nt_notes"]


def _nt_parse_tags(raw: str) -> list:
    """把标签输入切成列表（兼容空格 / 中英文逗号 / 顿号 / 分号）。"""
    out, seen = [], set()
    for part in re.split(r"[\s，,、;；/]+", raw or ""):
        p = part.strip().strip("#")
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _nt_fmt(iso_str: str) -> str:
    """把 ISO 时间显示为「刚刚 / 今天 / 昨天 / 9月2日」。"""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        return iso_str[:16]
    now = datetime.now()
    if (now - dt).total_seconds() < 3600:
        return "刚刚"
    if dt.date() == now.date():
        return "今天"
    if dt.date() == (now.date() - timedelta(days=1)):
        return "昨天"
    if dt.year == now.year:
        return f"{dt.month}月{dt.day}日"
    return f"{dt.year}-{dt.month}-{dt.day}"


def _nt_marks() -> dict:
    """每篇笔记「最近一次已写入数据库的内容」，用于判断是否有未落盘改动。"""
    return st.session_state.setdefault("nt_last_vals", {})


def _nt_sync_from_ui(nid: str | None = None) -> None:
    """把编辑器当前输入同步进数据库（有变化才写）。

    作为输入框 on_change 回调，也在每次 rerun 顶部兜底调用，
    保证切走视图/刷新前不会丢最后未触发的改动。
    """
    db = get_db()
    nid = nid or st.session_state.get("nt_open_id")
    if not nid:
        return
    t = st.session_state.get(f"nt_t_{nid}")
    c = st.session_state.get(f"nt_c_{nid}")
    g_raw = st.session_state.get(f"nt_g_{nid}")
    if t is None and c is None and g_raw is None:
        return
    cur = {"t": t or "", "c": c or "", "g": _nt_parse_tags(g_raw or "")}
    if _nt_marks().get(nid) == cur:
        return
    if db.update_note(nid, title=cur["t"], content=cur["c"], tags=cur["g"]):
        _nt_marks()[nid] = cur
        _nt_reload()


def _nt_open(nid: str) -> None:
    """打开一篇笔记编辑；顺带清理此前遗留的全空笔记（未填写就被点开别处）。"""
    db = get_db()
    old = st.session_state.get("nt_open_id")
    if old and old != nid:
        for n in _nt_notes():
            if (n["id"] == old and not n["title"].strip()
                    and not n["content"].strip() and not n["tags"]):
                db.delete_note(old)
                _nt_marks().pop(old, None)
                _nt_reload()
                break
    st.session_state["nt_open_id"] = nid
    st.rerun()


def render_notes_view() -> None:
    """笔记视图（仿 ima 布局）：左侧按时间分组列表 + 右侧文档式编辑器，自动保存。"""
    st.markdown(_NOTES_CSS, unsafe_allow_html=True)
    db = get_db()
    notes = _nt_notes()

    def _find(nid):
        for n in notes:
            if n["id"] == nid:
                return n
        return None

    # 打开态校验：目标若已被删除（例如其它窗口），回退
    open_id = st.session_state.get("nt_open_id")
    if open_id and _find(open_id) is None:
        st.session_state.pop("nt_open_id", None)
        open_id = None
        notes = _nt_notes()
    # 无打开笔记时默认打开最近一篇
    if not open_id and notes:
        open_id = notes[0]["id"]
        st.session_state["nt_open_id"] = open_id

    left, right = st.columns([0.8, 1.6], gap="medium")

    # ---------------- 左栏：全部 + 搜索 / 新建 / 时间分组列表 ----------------
    with left:
        head = st.columns([1, 1.15])
        with head[0]:
            st.markdown('<div class="nt-all">全部</div>', unsafe_allow_html=True)
        with head[1]:
            if st.button("＋ 新建", type="primary", use_container_width=True,
                         key="nt_new"):
                nid = db.save_note(title="", content="", tags=[])
                _nt_reload()
                st.session_state["nt_open_id"] = nid
                st.rerun()

        kw = st.text_input("搜索", placeholder="🔍  搜索笔记",
                           label_visibility="collapsed", key="nt_search")
        kw = (kw or "").strip().lower()

        # 无笔记时自动建一篇空笔记（仿 ima 进入即为未命名笔记）
        if not notes:
            nid = db.save_note(title="", content="", tags=[])
            _nt_reload()
            st.session_state["nt_open_id"] = nid
            notes = _nt_notes()
            open_id = nid

        filtered = [
            n for n in notes
            if not kw or kw in (
                f"{n.get('title', '')} {n.get('content', '')} "
                f"{' '.join(n.get('tags', []))}".lower()
            )
        ]

        if not filtered and notes:
            st.markdown('<div class="nt-empty">没有匹配的笔记</div>',
                        unsafe_allow_html=True)

        # 按更新时间分组：今天 / 昨天 / 更早（组内置顶优先）
        now_dt = datetime.now()
        groups = {"今天": [], "昨天": [], "更早": []}
        for n in filtered:
            try:
                dt_ = datetime.fromisoformat(n.get("updated_at", ""))
            except ValueError:
                dt_ = None
            days = (now_dt.date() - dt_.date()).days if dt_ else 9999
            key = "今天" if days == 0 else ("昨天" if days == 1 else "更早")
            groups[key].append(n)

        for gname, items in groups.items():
            if not items:
                continue
            items = sorted(items, key=lambda x: not x.get("pinned"))
            st.markdown(f'<div class="nt-group">{gname}</div>', unsafe_allow_html=True)
            for n in items:
                active = n["id"] == open_id
                title = (n.get("title") or "").strip() or "未命名笔记"
                pin = "📌 " if n.get("pinned") else ""
                preview = re.sub(r"\s+", " ", (n.get("content") or "").strip())[:60]
                if st.button(f"{pin}{title}", key=f"nt_row_{n['id']}",
                             use_container_width=True,
                             type="primary" if active else "secondary"):
                    _nt_open(n["id"])
                st.markdown(
                    f'<div class="nt-row-meta">{_nt_fmt(n.get("updated_at", ""))}'
                    f' · {html.escape(preview) if preview else "无附加文本"}</div>',
                    unsafe_allow_html=True)
    # ---------------- 右栏：文档式编辑器 ----------------
    with right:
        note = _find(open_id) if open_id else None
        if note is None:
            if notes:
                st.markdown('<div class="nt-hint">👈 从左侧选择一篇笔记开始编辑</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="nt-hint">
                    <div class="big">📝</div>
                    从左侧选择一篇笔记开始编辑
                </div>""", unsafe_allow_html=True)
        else:
            nid = note["id"]
            # 初始化「已入库」标记，随后同步一次（切回视图可能带未落盘输入）
            if nid not in _nt_marks():
                _nt_marks()[nid] = {
                    "t": note.get("title", ""),
                    "c": note.get("content", ""),
                    "g": list(note.get("tags", [])),
                }
            _nt_sync_from_ui(nid)

            st.text_input("笔记标题", placeholder="请输入标题",
                          key=f"nt_t_{nid}", on_change=_nt_sync_from_ui,
                          value=note.get("title", ""))
            st.text_area("正文", placeholder="请输入正文，「Ctrl+/」快速呼出ima帮助写作",
                         key=f"nt_c_{nid}", on_change=_nt_sync_from_ui,
                         value=note.get("content", ""), height=560,
                         label_visibility="collapsed")

            content_now = st.session_state.get(f"nt_c_{nid}")
            if content_now is None:
                content_now = note.get("content", "")
            wc = len(re.sub(r"\s", "", content_now))

            # 底部：左字数统计 / 中删除 / 右保存
            foot_l, foot_m, foot_r = st.columns([4, 1, 1])
            with foot_l:
                st.markdown(f'<div class="nt-wc">{wc} 个字</div>', unsafe_allow_html=True)
            with foot_m:
                if _render_delete_control(f"nt_del_{nid}", "删除这篇笔记"):
                    db.delete_note(nid)
                    _nt_reload()
                    _nt_marks().pop(nid, None)
                    st.session_state.pop("nt_open_id", None)
                    st.rerun()
            with foot_r:
                if st.button("💾 保存", type="primary", use_container_width=True,
                             key=f"nt_save_{nid}"):
                    _nt_sync_from_ui(nid)
                    # 保存后新建一篇空笔记并打开，编辑器恢复默认状态
                    new_id = db.save_note(title="", content="", tags=[])
                    _nt_reload()
                    _nt_marks().pop(new_id, None)
                    st.session_state["nt_open_id"] = new_id
                    st.rerun()



def render_courses_view() -> None:
    """课表管理视图：原生 Streamlit 实现，自适应铺满主区。"""
    agent: LifeAgent = st.session_state.agent

    # 注入 CSS：课表视图铺满主区 + 网格样式
    st.markdown("""
    <style>
    /* 课表视图：移除主区宽度限制，铺满整个右侧 */
    [data-testid="stMainBlockContainer"] > div:first-child {
        max-width: 100% !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-top: 0.4rem !important;
    }
    /* 网格间距 */
    .tt-grid .stHorizontalBlock { gap: 6px !important; }
    .tt-grid .stColumn { gap: 6px; }
    /* 单元格 button：统一方块格子样式 */
    .tt-grid .stButton > button {
        width: 100%;
        min-height: 84px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        color: #1e293b;
        font-weight: 600;
        font-size: 14px;
        padding: 6px;
        line-height: 1.3;
        transition: all .12s;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .tt-grid .stButton > button:hover {
        background: #eef2ff;
        border-color: #c7d2fe;
        color: #4f46e5;
    }
    .tt-time-label { font-size: 13px; font-weight: 600; color: #475569; text-align: center; padding: 8px 2px; }
    .tt-day-header { text-align: center; font-weight: 700; color: #1e293b; font-size: 15px; padding: 8px 0; }
    </style>
    """, unsafe_allow_html=True)

    # 时段配置
    if "tt_slots" not in st.session_state:
        st.session_state["tt_slots"] = [
            "08:00-09:35", "09:50-11:25", "13:30-15:05", "15:20-16:55", "18:30-20:05"
        ]
    slots = st.session_state["tt_slots"]

    # 课程数据：初始化时从 agent 读取
    if "tt_classes" not in st.session_state:
        st.session_state["tt_classes"] = _load_classes_from_agent(agent, slots)
    classes = st.session_state["tt_classes"]

    # 状态
    if "tt_form" not in st.session_state:
        st.session_state["tt_form"] = None
    if "tt_show_slots" not in st.session_state:
        st.session_state["tt_show_slots"] = False

    # 顶部工具栏
    tb = st.columns([1, 1, 1, 1])
    tb[0].markdown("### 📖 课表")
    tb[1].markdown(f"<div class='tt-time-label'>{len(classes)} 门课程</div>", unsafe_allow_html=True)
    if tb[2].button("➕ 新增", use_container_width=True, key="tt_add"):
        st.session_state["tt_form"] = {"mode": "add", "id": None}
        st.rerun()
    if tb[3].button("⚙️ 时段", use_container_width=True, key="tt_slots_toggle"):
        st.session_state["tt_show_slots"] = not st.session_state["tt_show_slots"]
        st.rerun()

    # 时段设置
    if st.session_state["tt_show_slots"]:
        _render_slots_editor(slots)
        st.divider()

    # 表单
    if st.session_state["tt_form"] is not None:
        _render_course_form(slots, classes, agent)
        st.divider()

    # 课表网格
    st.markdown('<div class="tt-grid">', unsafe_allow_html=True)
    days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    header = st.columns([1] + [1] * 7)
    header[0].markdown("")
    for i, d in enumerate(days):
        header[i + 1].markdown(f"<div class='tt-day-header'>{d}</div>", unsafe_allow_html=True)

    for slot_idx, slot_label in enumerate(slots):
        cols = st.columns([1] + [1] * 7)
        cols[0].markdown(f"<div class='tt-time-label'>{slot_label}</div>", unsafe_allow_html=True)
        for day_idx in range(7):  # 0=周一..6=周日
            our_day = (day_idx + 1) % 7  # 转成我们的索引: 0=周日,1=周一..6=周六
            with cols[day_idx + 1]:
                cell_classes = [c for c in classes
                               if c["day"] == our_day
                               and c["startSlot"] <= slot_idx <= c["endSlot"]]
                if cell_classes:
                    c = cell_classes[0]
                    is_start = c["startSlot"] == slot_idx
                    label = c["name"] if is_start else "↑"
                    if is_start and c.get("location"):
                        label = f"{c['name']} · {c['location']}"
                    if st.button(label, key=f"tt_cell_{our_day}_{slot_idx}_{c['id']}",
                                 use_container_width=True):
                        st.session_state["tt_form"] = {"mode": "edit", "id": c["id"]}
                        st.rerun()
                else:
                    if st.button(" ", key=f"tt_empty_{our_day}_{slot_idx}",
                                 use_container_width=True):
                        st.session_state["tt_form"] = {
                            "mode": "add", "id": None,
                            "prefill_day": our_day, "prefill_slot": slot_idx
                        }
                        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def _parse_slot_times(slot_label):
    """从时段标签解析起止时间。"""
    import re
    m = re.match(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", slot_label or "")
    if not m:
        return dt_time(8, 0), dt_time(9, 35)
    h1, m1, h2, m2 = map(int, m.groups())
    return dt_time(h1, m1), dt_time(h2, m2)


def _time_to_slot(slots, t, is_end=False):
    """把时间匹配到时段索引。"""
    for i, slot in enumerate(slots):
        s, e = _parse_slot_times(slot)
        if is_end:
            if t <= e:
                return i
        else:
            if t >= s:
                return i
    return 0


def _load_classes_from_agent(agent, slots):
    """从 agent 的 commitments 读取课程并转为课表结构。"""
    classes = []
    for c in agent._commitments:
        if not commitment_is_class(c):
            continue
        wd = commitment_weekday(c)  # 0=周一..6=周日
        day = (wd + 1) % 7  # 转成我们的索引: 0=周日,1=周一..6=周六
        try:
            start_t = datetime.fromisoformat(c.start_time).time()
            end_t = datetime.fromisoformat(c.end_time).time()
        except (ValueError, TypeError):
            continue
        classes.append({
            "id": c.id,
            "name": c.title,
            "day": day,
            "startSlot": _time_to_slot(slots, start_t),
            "endSlot": _time_to_slot(slots, end_t, is_end=True),
            "location": c.description or "",
            "teacher": "",
            "color": "#4f46e5",
            "note": "",
        })
    return classes


def _sync_class_to_agent(cls, agent, slots):
    """把课表课程同步到 agent 的 commitments（持久化到数据库）。"""
    from lifeops.models import RecurrenceType
    our_day = cls["day"]
    py_weekday = (our_day - 1) % 7
    s_time, _ = _parse_slot_times(slots[cls["startSlot"]])
    _, e_time = _parse_slot_times(slots[cls["endSlot"]])
    start_date = next_weekday_date(py_weekday, s_time)
    start_dt = datetime.combine(start_date, s_time)
    end_dt = datetime.combine(start_date, e_time)

    existing = next((c for c in agent._commitments if c.id == cls["id"]), None)
    if existing:
        agent.update_commitment(
            cls["id"],
            title=cls["name"],
            start_time=start_dt.isoformat(),
            end_time=end_dt.isoformat(),
            description=cls.get("location", ""),
        )
    else:
        commit = Commitment(
            id=cls["id"],
            profile_id=agent.profile.id,
            title=cls["name"],
            type=CommitmentType.CLASS,
            start_time=start_dt.isoformat(),
            end_time=end_dt.isoformat(),
            recurrence=RecurrenceType.WEEKLY,
            description=cls.get("location", ""),
        )
        agent.add_commitment(commit)
    if agent.db is not None:
        agent.persist_state()


def _remove_class_from_agent(cls_id, agent):
    """从 agent 删除课程。"""
    agent.remove_commitment(cls_id)
    if agent.db is not None:
        agent.persist_state()


def _render_slots_editor(slots):
    """渲染时间段设置。"""
    st.markdown("#### ⚙️ 时间段设置")
    st.caption("按行定义节次，修改后点其他按钮课表自动更新。")
    for i in range(len(slots)):
        cols = st.columns([5, 1])
        new_val = cols[0].text_input(f"第{i+1}节", value=slots[i], label_visibility="collapsed")
        if new_val != slots[i]:
            slots[i] = new_val
        if cols[1].button("🗑", key=f"tt_del_slot_{i}", use_container_width=True):
            if len(slots) > 1:
                slots.pop(i)
                st.rerun()
    cols = st.columns([5, 1])
    new_slot = cols[0].text_input("新增时段", placeholder="例：08:00-09:35", key="tt_new_slot_input", label_visibility="collapsed")
    if cols[1].button("➕", key="tt_add_slot", use_container_width=True) and new_slot.strip():
        slots.append(new_slot.strip())
        st.rerun()
    cols = st.columns([1, 1])
    if cols[0].button("恢复默认", key="tt_reset_slots", use_container_width=True):
        st.session_state["tt_slots"] = ["08:00-09:35", "09:50-11:25", "13:30-15:05", "15:20-16:55", "18:30-20:05"]
        st.rerun()
    if cols[1].button("完成", key="tt_close_slots", use_container_width=True):
        st.session_state["tt_show_slots"] = False
        st.rerun()


def _render_course_form(slots, classes, agent):
    """渲染新增/编辑课程表单。"""
    form = st.session_state["tt_form"]
    is_edit = form["mode"] == "edit"
    editing_id = form.get("id")
    cls = next((c for c in classes if c["id"] == editing_id), None) if is_edit else None

    st.markdown("#### " + ("编辑课程" if is_edit else "新增课程"))
    with st.form("tt_course_form"):
        name = st.text_input("课程名", value=cls["name"] if cls else "", placeholder="例如：高等数学")
        cols = st.columns([1, 1, 1])
        day_options = [0, 1, 2, 3, 4, 5, 6]
        day_labels = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"]
        day_val = cls["day"] if cls else form.get("prefill_day", 1)
        day = cols[0].selectbox("星期", options=day_options,
                               format_func=lambda x: day_labels[x],
                               index=day_options.index(day_val) if day_val in day_options else 1)
        start_val = cls["startSlot"] if cls else form.get("prefill_slot", 0)
        start_slot = cols[1].selectbox("起始", options=list(range(len(slots))),
                                     format_func=lambda x: f"{x+1}-{slots[x]}",
                                     index=min(start_val, len(slots) - 1))
        end_val = cls["endSlot"] if cls else form.get("prefill_slot", 0)
        end_slot = cols[2].selectbox("结束", options=list(range(len(slots))),
                                   format_func=lambda x: f"{x+1}-{slots[x]}",
                                   index=min(max(end_val, start_slot), len(slots) - 1))
        cols = st.columns([1, 1])
        location = cols[0].text_input("教室", value=cls["location"] if cls else "", placeholder="A101")
        teacher = cols[1].text_input("老师", value=cls["teacher"] if cls else "", placeholder="张老师")
        note = st.text_input("备注", value=cls["note"] if cls else "", placeholder="单双周等")
        cols = st.columns([1, 1, 2])
        submit = cols[0].form_submit_button("保存", use_container_width=True)
        cancel = cols[1].form_submit_button("取消", use_container_width=True)
        delete = cols[2].form_submit_button("删除", use_container_width=True) if is_edit else False

    if submit:
        if not name.strip():
            st.error("请输入课程名")
        else:
            import uuid
            new_cls = {
                "id": editing_id if is_edit else f"tt_{uuid.uuid4().hex[:8]}",
                "name": name.strip(),
                "day": day,
                "startSlot": start_slot,
                "endSlot": max(end_slot, start_slot),
                "location": location.strip(),
                "teacher": teacher.strip(),
                "note": note.strip(),
            }
            if is_edit:
                for i, c in enumerate(classes):
                    if c["id"] == editing_id:
                        classes[i] = new_cls
                        break
            else:
                classes.append(new_cls)
            try:
                _sync_class_to_agent(new_cls, agent, slots)
            except Exception as e:
                st.warning(f"同步失败：{e}")
            st.session_state["tt_form"] = None
            st.rerun()
    if cancel:
        st.session_state["tt_form"] = None
        st.rerun()
    if is_edit and delete:
        classes[:] = [c for c in classes if c["id"] != editing_id]
        try:
            _remove_class_from_agent(editing_id, agent)
        except Exception as e:
            st.warning(f"删除失败：{e}")
        st.session_state["tt_form"] = None
        st.rerun()



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
# 兼容热更新：Streamlit 只重跑 app.py、不重载已导入模块，若进程未重启且
# session 中保留了旧版 LifeAgent（缺少 ensure_plan_consistent），
# 直接调用会 AttributeError，故此处检测到旧实例时重建一次。
if "agent" not in st.session_state or not hasattr(
    st.session_state.agent, "ensure_plan_consistent"
):
    base_dir = PROJECT_ROOT
    db_path = os.environ.get("DB_PATH") or str(base_dir / "data" / "lifeos.db")
    # 写回环境变量：代码热更新导致模块重载时，get_db() 无参调用仍能拿到正确路径
    os.environ["DB_PATH"] = db_path
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
    st.session_state.chat_history = []
    st.session_state.current_date = date.today()
    st.session_state.active_tab = "dashboard"
    st.session_state.pending_screenshot = None  # 待确认的截图解析结果


agent: LifeAgent = st.session_state.agent
profile: StudentProfile = st.session_state.profile
# 用 get_db() 而非缓存实例：代码热更新后单例会随模块重载重建，
# 旧实例可能缺少新增方法（如 list_notes），缓存它会导致 AttributeError。
db = get_db()

# 幂等补齐缺失的表：旧库文件或代码热更新（进程未重启）后，缓存连接里可能
# 没有 notes 等新表；此处每次 rerun 重跑建表语句，自动补齐、无副作用。
db.init_schema()

# 笔记兜底同步：每次 rerun 把编辑器中未触发 on_change 的输入写回数据库，
# 保证切走视图 / 刷新页面时不会丢失最后键入的内容。
_nt_sync_from_ui()

# 主区视图：home=默认 LifeOS 界面，其余为知识库/笔记/日程/课表
# 通过 query_params 传 view，避免 fragment 干扰 session_state
_view_param = st.query_params.get("view", "")
if _view_param:
    st.session_state.main_view = _view_param
    st.query_params.pop("view", None)
main_view = st.session_state.get("main_view", "home")

# 数据自愈：若最新快照仍包含已删除任务/承诺的安排（历史遗留或某删除路径
# 未触发重排所致），立即按当前数据校正性重排一次并写回，保证侧栏「计划详情」
# 与主区计划始终显示最新状态。正常状态下为轻量比对、零副作用。
try:
    agent.ensure_plan_consistent()
except Exception:
    pass


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
        st.query_params["view"] = "home"
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
                st.query_params["view"] = key
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
                    st.query_params["view"] = key
                    st.rerun()


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

            # 目标卡片：标题行带删除按钮
            _hc1, _hc2 = st.columns([8, 1])
            with _hc1:
                st.markdown(f"**{g.title}**")
            with _hc2:
                if st.button("✕", key=f"goal_del_{g.id}", help="删除该目标及其任务"):
                    if agent.remove_goal(g.id):
                        # 删除目标后重新排期，避免计划与侧栏残留该目标的安排
                        agent.force_replan(reason=f"手动删除目标：{g.title}")
                    st.rerun()

            st.markdown(f"""
            <div class="goal-item">
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

    # 计划详情：仅在确有“可执行任务”内容时展示，避免残留/空计划显示 0 分钟
    _snap = agent.current_snapshot
    _has_plan_content = bool(
        _snap
        and (
            agent._tasks
            or _snap.sacrifice_list
            or any(
                (ts.source.value if hasattr(ts.source, "value") else ts.source) == "task"
                for ts in (_snap.time_slots or [])
            )
        )
    )
    if _has_plan_content:
        st.markdown(f"""
        <div class="summary-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.75rem; color:#64748b; font-weight:500;">计划详情</span>
                <span style="font-size:0.7rem; color:#94a3b8;">今日</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:4px;">
                <span style="color:#64748b;">⏱ 计划时长</span>
                <span style="font-weight:600; color:#0f172a;">{_snap.total_scheduled_minutes} 分钟</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                <span style="color:#64748b;">⚠️ 搁置任务</span>
                <span style="font-weight:600; color:#ef4444;">{len(_snap.sacrifice_list)} 个</span>
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
tab_dashboard, tab_plan, tab_tasks, tab_history = st.tabs([
    "  📊  概览  ",
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
                    # 完成状态：与「今日计划」共用同一 session_state key
                    done_key = f"plan_done_{ts.id or ts.task_id or idx}"
                    show_done = bool(
                        (task and task.status.value == "completed")
                        or st.session_state.get(done_key)
                    )

                    # 圆点颜色（已完成统一绿色）
                    if show_done:
                        dot_color = "#16a34a"
                        card_class = "task"
                    elif src == "commitment":
                        dot_color = "#f59e0b"
                        card_class = "commitment"
                    elif src == "buffer":
                        dot_color = "#94a3b8"
                        card_class = "buffer"
                    else:
                        dot_color = "#6366f1"
                        card_class = "task"

                    if show_done:
                        st.markdown(
                            '<div class="timeline-item" style="animation-delay:'
                            f' {idx * 0.05}s;">'
                            f'<div class="timeline-time">{start}</div>'
                            '<div class="timeline-dot" style="background:#16a34a;"></div>'
                            '<div class="timeline-card" style="background:#f0fdf4;border-left:3px solid rgba(22,163,74,0.45);">'
                            f'<div class="timeline-title" style="text-decoration:line-through;color:#86efac;">{title}</div>'
                            '<div class="timeline-meta">⏱ '
                            f'{duration} 分钟 · <span style="color:#16a34a;font-weight:600">✓ 已完成</span>'
                            '</div></div></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="timeline-item" style="animation-delay:'
                            f' {idx * 0.05}s;">'
                            f'<div class="timeline-time">{start}</div>'
                            f'<div class="timeline-dot" style="background:{dot_color};"></div>'
                            f'<div class="timeline-card {card_class}">'
                            f'<div class="timeline-title">{title}</div>'
                            f'<div class="timeline-meta">⏱ {duration} 分钟'
                            f'{" · 🔴硬截止" if task and task.deadline_type.value == "hard" else ""}'
                            f'{" · " + energy_icon(energy) if task else ""}'
                            '</div></div></div>',
                            unsafe_allow_html=True,
                        )

                if len(today_slots) > 8:
                    st.markdown(f"""
                    <div style="text-align:center; padding:10px 0 4px; color:#94a3b8; font-size:0.78rem;
                                cursor:pointer; transition: color 0.2s;" onmouseover="this.style.color='#64748b'">
                        还有 {len(today_slots) - 8} 个安排 → 切换到「今日计划」查看全部
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # 今日截止但未安排的活跃任务（日期对得上则补显到概览）
                _unsched_today = [
                    t for t in _unscheduled_active_tasks(agent, snapshot)
                    if _task_deadline_date(t) == today
                ]
                if _unsched_today:
                    st.markdown(
                        "<div style='margin-top:10px; font-size:0.8rem; color:#d97706; font-weight:600;'>"
                        "📌 今日截止 · 待安排</div>",
                        unsafe_allow_html=True,
                    )
                    for t in _unsched_today:
                        _render_unscheduled_task_row(agent, t, "dash_today", compact=True)
            else:
                # 今天没有已安排时间槽，检查是否有今日截止的未安排任务
                _unsched_today = [
                    t for t in _unscheduled_active_tasks(agent, snapshot)
                    if _task_deadline_date(t) == today
                ]
                if _unsched_today:
                    st.markdown(
                        "<div style='font-size:0.8rem; color:#d97706; font-weight:600; margin-bottom:8px;'>"
                        "📌 今日截止 · 待安排</div>",
                        unsafe_allow_html=True,
                    )
                    for t in _unsched_today:
                        _render_unscheduled_task_row(agent, t, "dash_empty", compact=True)
                else:
                    st.markdown("""
                    <div class="empty-state">
                        <div class="empty-state-icon">🌅</div>
                        <div class="empty-state-text">今天还没有安排</div>
                        <div class="empty-state-sub">在下方聊天框设定你的第一个目标吧</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # 没有快照时，检查是否有今日截止的未安排任务
            _unsched_today = [
                t for t in _unscheduled_active_tasks(agent, snapshot)
                if _task_deadline_date(t) == today
            ]
            if _unsched_today:
                st.markdown(
                    "<div style='font-size:0.8rem; color:#d97706; font-weight:600; margin-bottom:8px;'>"
                    "📌 今日截止 · 待安排</div>",
                    unsafe_allow_html=True,
                )
                for t in _unsched_today:
                    _render_unscheduled_task_row(agent, t, "dash_nosnap", compact=True)
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
# Tab 2: 今日计划（时间轴）
# ==========================================
with tab_plan:
    _render_delete_notice()
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

            # 时间轴（纯 Streamlit 原生行布局，不混用 HTML columns）
            for idx, ts in enumerate(day_slots):
                start = ts.start_time.split("T")[1][:5]
                end = ts.end_time.split("T")[1][:5]
                duration = int((datetime.fromisoformat(ts.end_time)
                              - datetime.fromisoformat(ts.start_time)).total_seconds() / 60)

                src = ts.source.value if hasattr(ts.source, 'value') else ts.source
                task = agent.get_task(ts.task_id) if ts.task_id else None
                title = task.title if task else ts.title or "安排"
                energy = task.energy_level.value if task and task.energy_level else "medium"
                # 完成状态：task.status 优先，session_state 兜底
                is_done = bool(task and task.status.value == "completed")
                done_key = f"plan_done_{ts.id or ts.task_id or idx}"
                is_done_local = done_key in st.session_state and st.session_state[done_key]
                show_done = is_done or is_done_local

                if src == "commitment":
                    dot_color = "#f59e0b"
                    card_border = "rgba(245,158,11,0.35)"
                elif src == "buffer":
                    dot_color = "#94a3b8"
                    card_border = "rgba(148,163,184,0.35)"
                    title = "☕ 休息缓冲"
                else:
                    dot_color = "#6366f1"
                    card_border = "rgba(99,102,241,0.35)"

                # 标签
                tag_strs = []
                if task and task.deadline_type.value == "hard":
                    tag_strs.append("🔴 硬截止")
                if task:
                    tag_strs.append(f"{energy_icon(energy)} {energy}")
                tag_line = " · ".join(tag_strs)

                # 行布局：时间 | 圆点+卡片 | 完成 | 删除
                _c_time, _c_dot, _c_card, _c_done, _c_del = st.columns(
                    [1.2, 0.15, 5.5, 1.3, 1.1]
                )
                with _c_time:
                    st.markdown(f"<div style='color:#64748b;font-size:0.78rem;text-align:center;padding-top:6px;white-space:nowrap'>{start}<br>—<br>{end}</div>", unsafe_allow_html=True)
                with _c_dot:
                    st.markdown(f"<div style='width:10px;height:10px;border-radius:50%;background:{dot_color};margin-top:14px;margin-left:4px;'></div>", unsafe_allow_html=True)
                with _c_card:
                    if show_done:
                        st.markdown(f"""
                        <div style="border-left:3px solid rgba(22,163,74,0.45);padding:6px 12px;margin-bottom:4px;border-radius:0 8px 8px 0;background:#f0fdf4;">
                            <div style="font-weight:600;text-decoration:line-through;color:#86efac;font-size:0.9rem;">{title}</div>
                            <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">⏱ {duration} 分钟{f' · {tag_line}' if tag_line else ''}</div>
                            <div style="color:#16a34a;font-size:0.72rem;font-weight:600;margin-top:2px;">✓ 已完成（再点一次可取消）</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="border-left:3px solid {card_border};padding:6px 12px;margin-bottom:4px;border-radius:0 8px 8px 0;background:{'#eef2ff' if src=='task' else '#fff7ed' if src=='commitment' else '#f8fafc'};">
                            <div style="font-weight:600;font-size:0.9rem;">{title}</div>
                            <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">⏱ {duration} 分钟{f' · {tag_line}' if tag_line else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)
                with _c_done:
                    if src == "task":
                        if show_done:
                            if st.button("✓", key=f"{done_key}_btn", help="点击取消完成",
                                         use_container_width=True):
                                st.session_state[done_key] = False
                                if task is not None:
                                    agent.uncomplete_task(task.id)
                                st.rerun()
                        else:
                            if st.button("☐", key=f"{done_key}_btn",
                                         help="标记完成", use_container_width=True):
                                st.session_state[done_key] = True
                                if task is not None:
                                    agent.complete_task(task.id)
                                st.rerun()
                    else:
                        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

                with _c_del:
                    if _slot_is_removable(ts):
                        _slot_key = (
                            f"plan_del_{selected_date}_{idx}_"
                            f"{ts.id or ts.task_id or ts.commitment_id}"
                        )
                        if _render_delete_control(_slot_key, f"删除：{title}"):
                            _ok, _label = _delete_slot(agent, ts)
                            _apply_delete(agent, _ok, _label)
                            st.rerun()
                    else:
                        st.markdown(
                            "<div style='height:40px'></div>", unsafe_allow_html=True
                        )

            # 待安排的活跃任务（任务看板中待开始/进行中，但未被调度器安排）
            unscheduled = _unscheduled_active_tasks(agent, snapshot)
            if unscheduled:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 📌 待安排任务")
                st.markdown(
                    "<p style='color:#64748b; font-size:0.8rem; margin-bottom:10px;'>"
                    "以下任务尚未排入时间计划，请尽快安排或调整截止时间。</p>",
                    unsafe_allow_html=True,
                )
                for t in unscheduled:
                    _render_unscheduled_task_row(agent, t, "plan_after")

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
            # 当天无已安排时间槽，但可能有未安排的活跃任务
            unscheduled = _unscheduled_active_tasks(agent, snapshot)
            if unscheduled:
                st.markdown("#### 📌 待安排任务")
                st.markdown(
                    "<p style='color:#64748b; font-size:0.8rem; margin-bottom:10px;'>"
                    "以下任务尚未排入时间计划，请尽快安排或调整截止时间。</p>",
                    unsafe_allow_html=True,
                )
                for t in unscheduled:
                    _render_unscheduled_task_row(agent, t, "plan_empty")
            else:
                st.markdown(f"""
                <div class="empty-state">
                    <div class="empty-state-icon">🌴</div>
                    <div class="empty-state-text">{weekday}没有安排</div>
                    <div class="empty-state-sub">是休息日吗？好好休息吧！</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        # 没有快照时，仍展示未安排的活跃任务
        unscheduled = _unscheduled_active_tasks(agent, snapshot)
        if unscheduled:
            st.markdown("#### 📌 待安排任务")
            st.markdown(
                "<p style='color:#64748b; font-size:0.8rem; margin-bottom:10px;'>"
                "以下任务尚未排入时间计划，请尽快安排或调整截止时间。</p>",
                unsafe_allow_html=True,
            )
            for t in unscheduled:
                _render_unscheduled_task_row(agent, t, "plan_nosnap")
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
    _render_delete_notice()
    if agent._tasks:
        pending_tasks = [t for t in agent._tasks if t.status.value == "pending"]
        in_progress_tasks = [t for t in agent._tasks if t.status.value == "in_progress"]
        completed_tasks = [t for t in agent._tasks if t.status.value == "completed"]

        # 按优先级排序
        pending_tasks.sort(key=lambda x: -x.priority)
        in_progress_tasks.sort(key=lambda x: -x.priority)

        col1, col2, col3 = st.columns(3, gap="small")

        # 待开始
        with col1:
            st.markdown(f"""
            <div class="kanban-header">
                <h3>⏳ 待开始</h3>
                <span class="kanban-count">{len(pending_tasks)}</span>
            </div>
            """, unsafe_allow_html=True)

            for idx, t in enumerate(pending_tasks[:10]):
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

                # 卡片操作区：开始 + 删除，紧跟对应卡片（key 用任务 id，不用 idx）
                _, _kb_start, _kb_del = st.columns([1.6, 1.0, 1.4])
                with _kb_start:
                    if st.button("▶", key=f"kb_start_{t.id}",
                                 help=f"开始任务：{t.title}",
                                 use_container_width=True):
                        # 只切状态为「进行中」，进度保持 0，等真正汇报时再更新
                        if agent.start_task(t.id):
                            st.session_state["action_notice"] = (
                                f"已开始「{t.title}」，已移入「进行中」"
                            )
                            st.rerun()
                with _kb_del:
                    if _render_delete_control(
                        f"kb_pending_{t.id}", f"删除任务：{t.title}"
                    ):
                        _ok, _label = _delete_task(agent, t)
                        _apply_delete(agent, _ok, _label)
                        st.rerun()
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            if len(pending_tasks) > 10:
                st.markdown(f"""
                <div style="text-align:center; padding:8px; color:#94a3b8; font-size:0.75rem;">
                    还有 {len(pending_tasks) - 10} 个...
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

            for idx, t in enumerate(in_progress_tasks[:10]):
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

                # 操作按钮：退回待开始 + 删除
                _, _kb_back, _kb_del = st.columns([2.0, 1.4, 1.4])
                with _kb_back:
                    if st.button("↩ 待开始", key=f"kb_back_doing_{t.id}",
                                 help="退回待开始状态", use_container_width=True):
                        agent.reset_task_to_pending(t.id)
                        st.session_state["action_notice"] = f"已将「{t.title}」退回待开始"
                        st.rerun()
                with _kb_del:
                    if _render_delete_control(
                        f"kb_doing_{t.id}", f"删除任务：{t.title}"
                    ):
                        _ok, _label = _delete_task(agent, t)
                        _apply_delete(agent, _ok, _label)
                        st.rerun()
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            if len(in_progress_tasks) > 10:
                st.markdown(f"""
                <div style="text-align:center; padding:8px; color:#94a3b8; font-size:0.75rem;">
                    还有 {len(in_progress_tasks) - 10} 个...
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

            for idx, t in enumerate(completed_tasks[:10]):
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

                # 操作按钮：退回进行中 + 删除
                _, _kb_back, _kb_del = st.columns([2.0, 1.4, 1.4])
                with _kb_back:
                    if st.button("↩ 进行中", key=f"kb_back_done_{t.id}",
                                 help="退回进行中状态", use_container_width=True):
                        agent.uncomplete_task(t.id)
                        st.session_state["action_notice"] = f"已将「{t.title}」退回进行中"
                        st.rerun()
                with _kb_del:
                    if _render_delete_control(
                        f"kb_done_{t.id}", f"删除任务：{t.title}"
                    ):
                        _ok, _label = _delete_task(agent, t)
                        _apply_delete(agent, _ok, _label)
                        st.rerun()
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            if len(completed_tasks) > 10:
                st.markdown(f"""
                <div style="text-align:center; padding:8px; color:#94a3b8; font-size:0.75rem;">
                    还有 {len(completed_tasks) - 10} 个...
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

_SNAP_REASON_META = {
    "initial": ("🎯 初始计划", "#6366f1"),
    "progress_report": ("📊 进度更新", "#0ea5e9"),
    "new_goal": ("➕ 新增目标", "#10b981"),
    "new_event": ("📢 新事件", "#f59e0b"),
    "cancel_plan": ("✂️ 取消计划", "#f43f5e"),
    "status_query": ("🔍 状态查询", "#94a3b8"),
}

_SNAP_ACTION_META = {
    "added": ("➕", "新增", "#10b981"),
    "moved": ("↔️", "调整时间", "#0ea5e9"),
    "removed": ("➖", "移除", "#f43f5e"),
    "shortened": ("⏬", "缩短", "#f59e0b"),
    "extended": ("⏫", "延长", "#f97316"),
    "kept": ("•", "保留", "#94a3b8"),
}


def _snap_dt(iso: str):
    """ISO 时间 → datetime；解析失败返回 None。"""
    try:
        return datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return None


def _snap_reason_meta(reason: str):
    """触发原因 → (文案, 主题色)。"""
    r = reason or "initial"
    return _SNAP_REASON_META.get(r, (f"🔄 {r}", "#64748b"))


def _snap_items_html(snap) -> str:
    """结构化变更明细 HTML。"""
    cl = snap.change_log
    if not cl or not cl.items:
        return ""
    rows = []
    for it in cl.items:
        action = it.action.value if hasattr(it.action, "value") else it.action
        icon, label, color = _SNAP_ACTION_META.get(action, ("•", "变更", "#94a3b8"))
        t0, t1 = _snap_dt(it.old_time or ""), _snap_dt(it.new_time or "")
        extra = []
        if t0 and t1:
            extra.append(f"{t0:%H:%M} → {t1:%H:%M}")
        elif t0:
            extra.append(f"原 {t0:%m-%d %H:%M}")
        elif t1:
            extra.append(f"{t1:%m-%d %H:%M} 起")
        if (it.old_duration is not None and it.new_duration is not None
                and it.old_duration != it.new_duration):
            extra.append(f"时长 {it.old_duration}′ → {it.new_duration}′")
        elif it.new_duration and action == "added":
            extra.append(f"{it.new_duration} 分钟")
        if it.reason:
            extra.append(it.reason)
        rows.append(
            '<div class="snap-diff-row">'
            f'<span class="snap-chip" style="color:{color};border-color:{color}55;'
            f'background:{color}14;">{icon} {label}</span>'
            f'<span class="snap-diff-title">{it.task_title or it.task_id}</span>'
            f'<span class="snap-diff-extra">{(" · ".join(extra)) if extra else ""}</span>'
            '</div>'
        )
    return "".join(rows)


def _snap_sacrifice_html(snap) -> str:
    """牺牲清单 HTML（快照级 + 变更日志内嵌，按 task_id 去重）。"""
    items, seen = [], set()
    for s in list(snap.sacrifice_list or []):
        if not s.task_id or s.task_id in seen:
            continue
        seen.add(s.task_id)
        items.append(s)
    if snap.change_log and snap.change_log.sacrifice_list:
        for s in snap.change_log.sacrifice_list:
            if not s.task_id or s.task_id in seen:
                continue
            seen.add(s.task_id)
            items.append(s)
    if not items:
        return ""
    rows = []
    for s in items:
        tag = "延期" if s.action == "delayed" else ("放弃" if s.action == "dropped" else str(s.action))
        dest = ""
        if s.delayed_to:
            dd = _snap_dt(s.delayed_to)
            dest = f" → {dd:%m-%d}" if dd else f" → {s.delayed_to}"
        rows.append(
            '<div class="snap-diff-row">'
            '<span class="snap-chip" style="color:#f59e0b;border-color:#f59e0b55;'
            f'background:#f59e0b14;">⏳ {tag}</span>'
            f'<span class="snap-diff-title">{s.task_title}</span>'
            f'<span class="snap-diff-extra">{s.reason}{dest}</span>'
            '</div>'
        )
    return "".join(rows)


def _snap_slots_html(snap) -> str:
    """该次快照的时间安排 HTML。"""
    slots = snap.time_slots or []
    if not slots:
        return ""
    src_icons = {"task": "📌", "commitment": "🔁", "buffer": "🧩", "rest": "☕"}
    rows = []
    for ts in slots:
        t0, t1 = _snap_dt(ts.start_time), _snap_dt(ts.end_time)
        if not (t0 and t1):
            continue
        src = ts.source.value if hasattr(ts.source, "value") else ts.source
        icon = src_icons.get(src, "•")
        name = ts.title or ""
        rows.append(
            f'<div class="snap-slot-row">{icon} {t0:%H:%M}–{t1:%H:%M} '
            f'<span style="font-weight:600;color:#334155;">{name}</span>'
            f'<span class="snap-slot-min">{ts.duration_minutes} 分钟</span></div>'
        )
    return "".join(rows)


with tab_history:
    snaps_desc = list(reversed(agent._snapshots))
    if snaps_desc:
        head_cols = st.columns([4.6, 1.2])
        with head_cols[0]:
            st.markdown("#### 📜 计划变更历史")
            st.markdown(
                "<p style='color:#64748b; font-size:0.82rem; margin-bottom:12px;'>"
                "每次计划调整都会生成一个快照，可展开查看逐条变更、牺牲项与当时的计划安排。</p>",
                unsafe_allow_html=True,
            )
        with head_cols[1]:
            if "clear_snap_confirm" not in st.session_state:
                if st.button("🗑 清空全部", key="clear_snap_btn"):
                    st.session_state["clear_snap_confirm"] = True
                    st.rerun()
            else:
                st.caption("⚠ 无法恢复")
                cy, cn = st.columns(2)
                if cy.button("✔", type="primary", use_container_width=True, key="clear_snap_yes"):
                    n = db.clear_all_snapshots(profile_id=agent.profile.id) if db is not None else 0
                    agent._snapshots = []
                    st.session_state.pop("clear_snap_confirm", None)
                    st.toast(f"已清空 {n} 条历史快照", icon="✅")
                    st.rerun()
                if cn.button("✘", use_container_width=True, key="clear_snap_no"):
                    st.session_state.pop("clear_snap_confirm", None)
                    st.rerun()

        # 逐条展示（最新在前）
        for idx, snap in enumerate(snaps_desc):
            ver = len(snaps_desc) - idx
            created_dt = _snap_dt(snap.created_at)
            created_display = created_dt.strftime("%m-%d %H:%M") if created_dt else snap.created_at
            reason_label, reason_color = _snap_reason_meta(snap.trigger_reason)
            n_slots = len(snap.time_slots or [])
            minutes = snap.total_scheduled_minutes
            cl = snap.change_log
            summary = (cl.summary or "") if cl else ""
            n_items = len(cl.items) if (cl and cl.items) else 0

            badges = ""
            if snap.hard_deadline_count:
                badges += ('<span class="snap-chip" style="color:#ef4444;border-color:#ef444455;'
                           f'background:#ef444414;">硬截止 {snap.hard_deadline_count}</span>')
            if snap.risk_count:
                badges += ('<span class="snap-chip" style="color:#f59e0b;border-color:#f59e0b55;'
                           f'background:#f59e0b14;">风险 {snap.risk_count}</span>')

            st.markdown(f"""
            <div class="snapshot-item">
                <div class="snapshot-version" style="background:linear-gradient(135deg, {reason_color}, #94a3b8); box-shadow:0 2px 8px {reason_color}44;">{ver}</div>
                <div class="snapshot-info">
                    <div class="snapshot-reason">{reason_label}
                        <span style="color:#cbd5e1;font-weight:400;font-size:0.75rem;">· {created_display}</span>
                    </div>
                    <div class="snapshot-meta">{n_slots} 个时段 · {minutes} 分钟　{badges}</div>
                    {('<div class="snap-summary">' + summary + '</div>') if summary else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

            slots_html = _snap_slots_html(snap)
            sac_html = _snap_sacrifice_html(snap)
            detail_text = cl.detail if (cl and cl.detail) else ""

            if n_items or sac_html or slots_html or detail_text:
                with st.expander(f"查看明细 · {n_items} 项变更 / {n_slots} 段安排"):
                    if n_items:
                        st.markdown('<div class="snap-section-label">📋 变更明细</div>'
                                    + _snap_items_html(snap), unsafe_allow_html=True)
                    if sac_html:
                        st.markdown('<div class="snap-section-label">⏳ 牺牲清单</div>'
                                    + sac_html, unsafe_allow_html=True)
                    if slots_html:
                        st.markdown('<div class="snap-section-label">🕐 该次计划安排</div>'
                                    + slots_html, unsafe_allow_html=True)
                    if detail_text:
                        st.caption("说明：" + detail_text)

            # 单条删除（二次确认）
            del_cols = st.columns([5.2, 1.0])
            with del_cols[0]:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            with del_cols[1]:
                _dkey = f"snap_del_{snap.id}"
                _ckey = f"{_dkey}_confirm"
                if st.session_state.get(_ckey):
                    dy, dn = st.columns(2)
                    if dy.button("✔", key=f"{_dkey}_yes", help="确认删除该快照",
                                 use_container_width=True):
                        if db is not None:
                            db.delete_snapshot(snap.id)
                        agent._snapshots = [s for s in agent._snapshots if s.id != snap.id]
                        st.session_state.pop(_ckey, None)
                        st.toast(f"已删除快照 v{ver}", icon="🗑")
                        st.rerun()
                    if dn.button("✘", key=f"{_dkey}_no", help="取消",
                                 use_container_width=True):
                        st.session_state.pop(_ckey, None)
                        st.rerun()
                else:
                    if st.button("🗑 删除", key=f"{_dkey}_btn",
                                 help=f"删除快照 v{ver}", use_container_width=True):
                        st.session_state[_ckey] = True
                        st.rerun()
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

# 从 chat_history 提取最近对话（作为 Agent 的"记忆"上下文）
def _chat_history_for_agent():
    """取最近 8 轮对话，供 Agent 解析/闲聊时理解语境。"""
    out = []
    for msg in st.session_state.chat_history[-8:]:
        role = msg.get("role")
        content = str(msg.get("content") or "").strip()
        if role not in ("user", "bot") or not content:
            continue
        out.append({"role": role, "content": content[:400]})
    return out


# ============ 待确认动作：快捷按钮 ============
_pending = getattr(agent, "_pending_action", None)
if _pending is not None:
    _is_cancel = getattr(_pending, "intent", None) == "cancel_plan"
    _tip = (
        "🕓 有一笔「取消计划」待确认，确认后会删除对应安排并重新排期。"
        if _is_cancel
        else "🕓 还有一笔安排待确认，也可以直接输入「好的/可以」或「不用了」。"
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;'
        f'background:#fefce8;border:1px solid #fde68a;border-radius:10px;'
        f'padding:8px 12px;margin:4px 0 8px;">'
        f'<span style="font-size:0.85rem;color:#854d0e;">{_tip}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _c1, _c2, _c3 = st.columns([1, 1, 3])
    if _c1.button(
        "✅ 确认取消" if _is_cancel else "✅ 确认，加入日程",
        key="pending_confirm_yes",
        type="primary", use_container_width=True,
    ):
        with st.spinner("正在安排..."):
            _res = agent.confirm_pending(confirmed=True)
        st.session_state.chat_history.append(
            {"role": "bot", "content": _res.reply}
        )
        st.rerun()
    if _c2.button(
        "❌ 不用了，取消", key="pending_confirm_no",
        use_container_width=True,
    ):
        _res = agent.confirm_pending(confirmed=False)
        st.session_state.chat_history.append(
            {"role": "bot", "content": _res.reply}
        )
        st.rerun()

# 输入框
user_input = st.chat_input("告诉我你的目标、进度，或者问我计划安排...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("思考中..."):
        _history = _chat_history_for_agent()
        # 使用带知识库的对话
        if hasattr(agent, 'chat_with_knowledge') and getattr(agent, 'retriever', None):
            try:
                result = agent.chat_with_knowledge(
                    user_input, use_rag=True, history=_history
                )
            except TypeError:
                # 兼容旧版 Agent（未支持 history 参数）：重启 Streamlit 可启用对话记忆
                result = agent.chat_with_knowledge(user_input, use_rag=True)
        else:
            try:
                result = agent.run(user_input, history=_history)
            except TypeError:
                result = agent.run(user_input)

    # 构建 bot 消息
    bot_msg = {"role": "bot", "content": result.reply}
    # 带上 RAG 引用
    if hasattr(result, 'rag_references') and result.rag_references:
        bot_msg["references"] = result.rag_references
    st.session_state.chat_history.append(bot_msg)
    st.rerun()

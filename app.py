from __future__ import annotations

import html
import json
import os
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from risk_engine import (
    NUMERIC_COLUMNS,
    build_summary_text,
    generate_risks,
    load_health_file,
    merge_manual_inputs,
    simulate_what_if,
    summarize_timeseries,
)

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


st.set_page_config(page_title="CareNav", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        html, body, .stApp {
            background: #060b16;
            color: #e5e7eb;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="collapsedControl"] {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 999px;
            color: #f8fafc;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.35);
        }
        [data-testid="collapsedControl"]:hover {
            background: rgba(255,255,255,0.14);
            border-color: rgba(255,255,255,0.28);
        }
        [data-testid="collapsedControl"] svg {
            fill: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background: #0b1324;
            border-right: 1px solid rgba(148, 163, 184, 0.16);
        }
        [data-testid="stSidebar"] * {
            color: #dbe3f0;
        }
        .main {
            background:
                radial-gradient(circle at 8% 4%, rgba(59, 130, 246, 0.22), transparent 44%),
                radial-gradient(circle at 92% 8%, rgba(14, 165, 233, 0.20), transparent 42%),
                radial-gradient(circle at 75% 85%, rgba(16, 185, 129, 0.14), transparent 35%),
                linear-gradient(#060b16, #0a1120);
        }
        .carenav-hero {
            background:
                linear-gradient(180deg, rgba(15,23,42,0.86), rgba(15,23,42,0.74));
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            color: #e2e8f0;
            border-radius: 22px;
            padding: 20px 22px;
            margin-bottom: 8px;
            border: 1px solid rgba(148, 163, 184, 0.20);
            box-shadow: 0 10px 36px rgba(2, 6, 23, 0.45);
            position: relative;
            overflow: hidden;
            min-height: 152px;
        }
        .carenav-hero::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: linear-gradient(90deg, #2563eb, #0ea5e9, #10b981);
        }
        .carenav-hero h1 {
            margin: 0;
            font-size: 2.0rem;
            font-weight: 680;
            letter-spacing: -0.02em;
        }
        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.10);
            border: 1px solid rgba(56, 189, 248, 0.18);
            color: #bae6fd;
            font-size: 0.82rem;
            font-weight: 650;
            letter-spacing: 0.01em;
            position: relative;
            z-index: 2;
        }
        .hero-mark {
            position: absolute;
            right: 22px;
            top: 22px;
            width: 58px;
            height: 58px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.24), rgba(16, 185, 129, 0.18));
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #e0f2fe;
            font-size: 1.9rem;
            font-weight: 700;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.36);
            z-index: 2;
        }
        .carenav-hero p {
            margin: 7px 0 0 0;
            color: #cbd5e1;
            font-size: 0.96rem;
            max-width: 48rem;
        }
        .mini-card {
            background: rgba(15,23,42,0.78);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 16px;
            padding: 13px 14px;
            box-shadow: 0 6px 20px rgba(2, 6, 23, 0.38);
            position: relative;
            overflow: hidden;
        }
        .mini-card::before {
            content: "";
            position: absolute;
            left: 0; top: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #cbd5e1, #e2e8f0);
        }
        .mini-card.blue::before {
            background: linear-gradient(90deg, #2563eb, #38bdf8);
        }
        .mini-card.red::before {
            background: linear-gradient(90deg, #ef4444, #fb7185);
        }
        .mini-card.green::before {
            background: linear-gradient(90deg, #10b981, #34d399);
        }
        .mini-card.yellow::before {
            background: linear-gradient(90deg, #f59e0b, #fcd34d);
        }
        .mini-card.purple::before {
            background: linear-gradient(90deg, #4f46e5, #60a5fa);
        }
        .mini-label {
            color: #94a3b8;
            font-size: 0.8rem;
            margin-bottom: 2px;
        }
        .mini-value {
            color: #e2e8f0;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .mini-sub {
            color: #9aa6bb;
            font-size: 0.78rem;
            margin-top: 3px;
        }
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid transparent;
        }
        .pill-high { background:#fee2e2; color:#991b1b; border-color:#fecaca; }
        .pill-mod { background:#fef3c7; color:#92400e; border-color:#fde68a; }
        .pill-low { background:#dcfce7; color:#166534; border-color:#bbf7d0; }
        .section-title {
            margin-top: 0.25rem;
            margin-bottom: 0.35rem;
            font-weight: 650;
            color: #e5e7eb;
        }
        .section-header-card {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            margin-top: 16px;
            margin-bottom: 8px;
        }
        .section-icon-badge {
            width: 28px;
            height: 28px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 700;
            line-height: 1;
            margin-top: 1px;
        }
        .section-header-title {
            color: #f8fafc;
            font-size: 1.03rem;
            font-weight: 680;
            line-height: 1.25;
        }
        .section-header-sub {
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 1px;
        }
        .section-blue .section-icon-badge {
            color: #dbeafe;
            background: rgba(59, 130, 246, 0.16);
            border: 1px solid rgba(59, 130, 246, 0.30);
        }
        .section-green .section-icon-badge {
            color: #d1fae5;
            background: rgba(16, 185, 129, 0.16);
            border: 1px solid rgba(16, 185, 129, 0.28);
        }
        .section-cyan .section-icon-badge {
            color: #cffafe;
            background: rgba(14, 165, 233, 0.14);
            border: 1px solid rgba(14, 165, 233, 0.28);
        }
        .section-purple .section-icon-badge {
            color: #ede9fe;
            background: rgba(99, 102, 241, 0.18);
            border: 1px solid rgba(129, 140, 248, 0.30);
        }
        .section-divider {
            height: 1px;
            margin: 14px 0 10px 0;
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.00), rgba(59, 130, 246, 0.40), rgba(16, 185, 129, 0.00));
        }
        .subtle-note {
            color: #94a3b8;
            font-size: 0.82rem;
        }
        .csv-preview-card {
            background: rgba(15,23,42,0.70);
            border: 1px solid rgba(59, 130, 246, 0.22);
            border-radius: 16px;
            padding: 10px 12px;
            margin-top: 6px;
            margin-bottom: 10px;
            box-shadow: 0 6px 18px rgba(2, 6, 23, 0.34);
        }
        .soft-panel {
            background: rgba(15,23,42,0.62);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 12px 14px;
            box-shadow: 0 6px 20px rgba(2, 6, 23, 0.28);
        }
        .ai-search-title {
            color: #dbeafe;
            font-weight: 650;
            margin-bottom: 0.15rem;
        }
        .stButton button[kind="secondary"] {
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15,23,42,0.78);
            color: #dbeafe;
            min-height: 46px;
            box-shadow: 0 8px 20px rgba(2, 6, 23, 0.18);
        }
        .stButton button[kind="secondary"]:hover {
            border-color: rgba(96, 165, 250, 0.28);
            background: rgba(15,23,42,0.92);
            color: #f8fafc;
        }
        .stButton button[kind="primary"] {
            box-shadow: 0 12px 24px rgba(14, 165, 233, 0.20);
        }
        .status-panel {
            background: linear-gradient(180deg, rgba(15,23,42,0.90), rgba(15,23,42,0.74));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            padding: 16px 18px;
            margin-top: 8px;
            margin-bottom: 18px;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.34);
            position: relative;
            overflow: hidden;
        }
        .status-panel::before {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            top: 0;
            height: 4px;
        }
        .status-panel.status-green {
            border-color: rgba(16, 185, 129, 0.28);
            box-shadow: 0 12px 34px rgba(3, 105, 92, 0.18);
            background:
                radial-gradient(circle at 0% 0%, rgba(16, 185, 129, 0.14), transparent 32%),
                linear-gradient(180deg, rgba(15,23,42,0.92), rgba(15,23,42,0.76));
        }
        .status-panel.status-green::before {
            background: linear-gradient(90deg, #10b981, #34d399);
        }
        .status-panel.status-yellow {
            border-color: rgba(245, 158, 11, 0.26);
            box-shadow: 0 12px 34px rgba(120, 53, 15, 0.18);
            background:
                radial-gradient(circle at 0% 0%, rgba(245, 158, 11, 0.14), transparent 32%),
                linear-gradient(180deg, rgba(15,23,42,0.92), rgba(15,23,42,0.76));
        }
        .status-panel.status-yellow::before {
            background: linear-gradient(90deg, #f59e0b, #fcd34d);
        }
        .status-panel.status-red {
            border-color: rgba(239, 68, 68, 0.26);
            box-shadow: 0 12px 34px rgba(127, 29, 29, 0.20);
            background:
                radial-gradient(circle at 0% 0%, rgba(239, 68, 68, 0.13), transparent 32%),
                linear-gradient(180deg, rgba(15,23,42,0.92), rgba(15,23,42,0.76));
        }
        .status-panel.status-red::before {
            background: linear-gradient(90deg, #ef4444, #fb7185);
        }
        .status-row {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }
        .status-badge {
            min-width: 98px;
            padding: 9px 14px;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-align: center;
            font-size: 0.95rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 10px 22px rgba(2, 6, 23, 0.28);
        }
        .status-green {
            background: linear-gradient(135deg, rgba(6, 95, 70, 0.85), rgba(16, 185, 129, 0.32));
            color: #ecfdf5;
            border: 1px solid rgba(16, 185, 129, 0.46);
        }
        .status-yellow {
            background: linear-gradient(135deg, rgba(146, 64, 14, 0.82), rgba(245, 158, 11, 0.30));
            color: #fffbeb;
            border: 1px solid rgba(245, 158, 11, 0.42);
        }
        .status-red {
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.82), rgba(239, 68, 68, 0.28));
            color: #fff1f2;
            border: 1px solid rgba(239, 68, 68, 0.42);
        }
        .status-title {
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: 680;
            margin: 0;
        }
        .status-caption {
            color: #a5b4cc;
            font-size: 0.86rem;
            margin-top: 2px;
        }
        .focus-item {
            margin-top: 10px;
        }
        .focus-label {
            color: #8fb6ff;
            font-size: 0.78rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 3px;
        }
        .focus-value {
            color: #e2e8f0;
            font-size: 0.98rem;
        }
        .results-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-top: 8px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        .results-title {
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: 680;
        }
        .results-subtle {
            color: #94a3b8;
            font-size: 0.84rem;
        }
        .info-chip {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.10);
            border: 1px solid rgba(59, 130, 246, 0.18);
            color: #bfdbfe;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .legend-wrap {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }
        .legend-chip {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 5px 10px;
            border-radius: 999px;
            background: rgba(15,23,42,0.64);
            border: 1px solid rgba(148, 163, 184, 0.14);
            color: #dbe3f0;
            font-size: 0.78rem;
        }
        .legend-swatch {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
        }
        .stats-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }
        .stats-pill {
            background: rgba(15,23,42,0.60);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 16px;
            padding: 11px 13px;
            box-shadow: 0 8px 20px rgba(2, 6, 23, 0.22);
            position: relative;
            overflow: hidden;
        }
        .stats-pill::before {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            top: 0;
            height: 2px;
            background: linear-gradient(90deg, #2563eb, #38bdf8);
        }
        .stats-pill.purple::before {
            background: linear-gradient(90deg, #4f46e5, #60a5fa);
        }
        .stats-pill.green::before {
            background: linear-gradient(90deg, #10b981, #34d399);
        }
        .stats-pill.yellow::before {
            background: linear-gradient(90deg, #f59e0b, #fcd34d);
        }
        .stats-label {
            color: #94a3b8;
            font-size: 0.74rem;
            margin-bottom: 3px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 650;
        }
        .stats-value {
            color: #f8fafc;
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .stats-sub {
            color: #9aa6bb;
            font-size: 0.76rem;
            margin-top: 2px;
        }
        .action-card {
            background: linear-gradient(180deg, rgba(15,23,42,0.84), rgba(15,23,42,0.70));
            border: 1px solid rgba(59, 130, 246, 0.16);
            border-radius: 18px;
            padding: 14px 15px;
            min-height: 138px;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.28);
            position: relative;
            overflow: hidden;
        }
        .action-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: linear-gradient(90deg, #38bdf8, #2563eb);
        }
        .action-step {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.18);
            border: 1px solid rgba(59, 130, 246, 0.26);
            color: #dbeafe;
            font-size: 0.9rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .action-title {
            color: #f8fafc;
            font-size: 0.84rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .action-body {
            color: #dbe3f0;
            font-size: 0.95rem;
            line-height: 1.45;
        }
        .sim-panel {
            background:
                radial-gradient(circle at 100% 0%, rgba(16, 185, 129, 0.10), transparent 34%),
                linear-gradient(180deg, rgba(15,23,42,0.86), rgba(15,23,42,0.74));
            border: 1px solid rgba(52, 211, 153, 0.16);
            border-radius: 16px;
            padding: 12px 14px;
            margin-bottom: 10px;
            box-shadow: 0 10px 22px rgba(2, 6, 23, 0.18);
        }
        .sim-title {
            color: #d1fae5;
            font-weight: 680;
            margin-bottom: 3px;
        }
        .sim-subtle {
            color: #94a3b8;
            font-size: 0.82rem;
        }
        .sim-range {
            color: #93c5fd;
            font-size: 0.78rem;
            margin-top: 4px;
        }
        .snapshot-panel {
            background: rgba(15,23,42,0.52);
            border: 1px solid rgba(148, 163, 184, 0.10);
            border-radius: 16px;
            padding: 12px 14px;
            margin-top: 4px;
        }
        .ask-banner {
            background:
                radial-gradient(circle at 100% 0%, rgba(56, 189, 248, 0.12), transparent 34%),
                linear-gradient(180deg, rgba(15,23,42,0.84), rgba(15,23,42,0.72));
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-left: 3px solid rgba(56, 189, 248, 0.78);
            border-radius: 16px;
            padding: 12px 14px;
            margin-bottom: 10px;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.20);
        }
        .ask-answer-label {
            color: #93c5fd;
            font-size: 0.76rem;
            font-weight: 650;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .quick-prompt-note {
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 8px;
            margin-bottom: 6px;
        }
        [data-testid="stExpander"] {
            background: rgba(15,23,42,0.46);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 14px;
            margin-top: 12px;
        }
        [data-testid="stExpander"] > details summary p {
            color: #e2e8f0;
            font-weight: 620;
        }
        @media (max-width: 900px) {
            .main .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
                padding-top: 0.8rem;
            }
            .carenav-hero {
                padding: 14px 14px;
                min-height: 126px;
                border-radius: 18px;
            }
            .carenav-hero h1 { font-size: 1.62rem; }
            .carenav-hero p {
                font-size: 0.88rem;
                line-height: 1.35;
                max-width: none;
            }
            .hero-kicker {
                font-size: 0.74rem;
                padding: 5px 10px;
                margin-bottom: 8px;
            }
            .hero-mark {
                width: 44px;
                height: 44px;
                right: 14px;
                top: 14px;
                font-size: 1.35rem;
                border-radius: 13px;
            }
            .results-header {
                margin-top: 6px;
                margin-bottom: 8px;
                gap: 8px;
            }
            .results-title { font-size: 1.0rem; }
            .results-subtle { font-size: 0.78rem; }
            .info-chip {
                font-size: 0.73rem;
                padding: 5px 9px;
            }
            .status-panel {
                padding: 12px 12px;
                border-radius: 16px;
                margin-bottom: 14px;
            }
            .status-title { font-size: 1.02rem; }
            .status-caption { font-size: 0.8rem; }
            .focus-label { font-size: 0.72rem; }
            .focus-value { font-size: 0.92rem; }
            .section-header-card {
                margin-top: 12px;
                margin-bottom: 7px;
                gap: 8px;
            }
            .section-icon-badge {
                width: 24px;
                height: 24px;
                border-radius: 8px;
                font-size: 0.78rem;
            }
            .section-header-title { font-size: 0.96rem; }
            .section-header-sub { font-size: 0.76rem; }
            .section-divider { margin: 12px 0 8px 0; }
            .stats-strip {
                grid-template-columns: 1fr;
                gap: 8px;
                margin-bottom: 14px;
            }
            .stats-pill {
                padding: 9px 10px;
                border-radius: 12px;
            }
            .stats-label { font-size: 0.68rem; }
            .stats-value { font-size: 0.95rem; }
            .stats-sub { font-size: 0.72rem; }
            .action-card {
                min-height: 104px;
                padding: 10px 11px;
                border-radius: 14px;
            }
            .action-step {
                width: 28px;
                height: 28px;
                font-size: 0.76rem;
                margin-bottom: 7px;
            }
            .action-title {
                font-size: 0.74rem;
                margin-bottom: 4px;
            }
            .action-body {
                font-size: 0.86rem;
                line-height: 1.36;
            }
            .sim-panel {
                padding: 10px 11px;
                border-radius: 14px;
                margin-bottom: 8px;
            }
            .sim-title { font-size: 0.92rem; }
            .sim-subtle { font-size: 0.76rem; }
            .sim-range { font-size: 0.72rem; }
            .ask-banner {
                padding: 10px 11px;
                border-radius: 14px;
                margin-bottom: 8px;
            }
            .quick-prompt-note {
                font-size: 0.74rem;
                margin-top: 6px;
                margin-bottom: 5px;
            }
            .snapshot-panel {
                padding: 9px 10px;
                border-radius: 12px;
            }
            .legend-wrap {
                justify-content: flex-start;
                gap: 6px;
            }
            .legend-chip {
                font-size: 0.72rem;
                padding: 4px 8px;
            }
            [data-testid="stExpander"] {
                margin-top: 10px;
                border-radius: 12px;
            }
            .stButton button[kind="primary"] {
                min-height: 42px;
                font-size: 0.93rem;
            }
            .stButton button[kind="secondary"] {
                min-height: 40px;
                font-size: 0.84rem;
                padding-top: 0.32rem;
                padding-bottom: 0.32rem;
            }
            .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb=\"select\"] > div {
                font-size: 16px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sample_csv() -> pd.DataFrame:
    rows = [
        ["2026-02-12", 58, 48, 7.8, 8200, 98, 72.2, 122, 78, 94, 98.2],
        ["2026-02-13", 59, 46, 7.1, 7600, 98, 72.3, 124, 79, 96, 98.4],
        ["2026-02-14", 60, 44, 6.6, 6300, 97, 72.3, 126, 80, 98, 98.7],
        ["2026-02-15", 62, 42, 6.0, 5400, 97, 72.6, 130, 83, 102, 99.1],
        ["2026-02-16", 64, 39, 5.8, 4100, 96, 73.0, 134, 86, 108, 99.4],
        ["2026-02-17", 66, 35, 5.4, 3000, 95, 73.5, 138, 88, 116, 99.8],
        ["2026-02-18", 69, 31, 5.2, 2200, 94, 74.2, 142, 92, 124, 100.6],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "resting_hr",
            "hrv",
            "sleep_hours",
            "steps",
            "spo2",
            "weight_kg",
            "systolic_bp",
            "diastolic_bp",
            "glucose_fasting",
            "temperature_f",
        ],
    )


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _parse_optional_float(value: str) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_optional_int(value: str) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _build_ai_payload(
    profile: dict[str, Any],
    manual_inputs: dict[str, Any],
    summary: dict[str, Any],
    risks: list[Any],
    positives: list[str],
    gaps: list[str],
    alerts: list[str],
    symptoms: list[str],
    conditions: list[str],
    main_concern: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in [
        "resting_hr",
        "hrv",
        "sleep_hours",
        "steps",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "glucose_fasting",
        "weight_kg",
        "temperature_f",
    ]:
        item = summary.get("features", {}).get(key, {})
        if item:
            metrics[key] = {
                "latest": item.get("latest"),
                "avg7": item.get("avg7"),
                "avg30": item.get("avg30"),
                "avg90": item.get("avg90"),
                "trend_delta": item.get("trend_delta"),
                "delta_vs_30": item.get("delta_vs_30"),
                "delta_vs_90": item.get("delta_vs_90"),
                "zscore_30": item.get("zscore_30"),
                "sudden_shift_30": item.get("sudden_shift_30"),
                "count": item.get("count"),
            }
        elif manual_inputs.get(key) is not None:
            metrics[key] = {"latest": manual_inputs.get(key)}

    return {
        "app": "CareNav",
        "main_concern": main_concern or None,
        "profile": {
            "age": profile.get("age"),
            "sex": profile.get("sex"),
            "bmi": round(profile["bmi"], 1) if profile.get("bmi") is not None else None,
            "smoking_status": profile.get("smoking_status"),
        },
        "known_conditions": conditions,
        "current_symptoms": symptoms,
        "trend_summary": {
            "n_days": summary.get("n_days"),
            "freshness_days": summary.get("freshness_days"),
            "metrics": metrics,
        },
        "recovery_signal": summary.get("recovery"),
        "predicted_risks": [
            {
                "risk_domain": r.name,
                "probability_percent": round(r.probability, 1),
                "severity": r.severity,
                "confidence": round(r.confidence, 3),
                "trend": r.trend,
                "drivers": r.why[:4],
                "protective_factors": r.protective_factors[:3],
                "next_steps": r.next_steps[:3],
            }
            for r in risks
        ],
        "positive_behaviors": positives[:4],
        "data_gaps": gaps[:5],
        "urgent_alerts": alerts[:4],
    }


def _extract_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    try:
        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        if chunks:
            return "\n\n".join(chunks)
    except Exception:
        pass
    return ""


def _generate_ai_narrative(
    api_key: str,
    model: str,
    payload: dict[str, Any],
) -> str:
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    if OpenAI is None:
        raise RuntimeError("The `openai` package is not installed. Install dependencies from requirements.txt.")

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are CareNav's healthcare education assistant. "
        "Write a short plain-language explanation of only the top predicted risk from the provided CareNav results. "
        "Keep it very brief and easy for non-clinical users. "
        "Format exactly: 1 short paragraph, then 2 bullet points. "
        "Bullet 1: why this risk may be elevated for this user. "
        "Bullet 2: one practical next measurement/action. "
        "If urgent alerts exist, add one urgent warning line first. "
        "Do not diagnose. Do not list all risks. Do not invent data."
    )
    user_prompt = (
        "Use the structured risk output as the source of truth. Do not invent measurements.\n\n"
        + json.dumps(payload, indent=2)
    )

    try:
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = _extract_response_text(response)
            if text:
                return text

        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = chat.choices[0].message.content if chat.choices else ""
        return (content or "").strip()
    except Exception as exc:  # pragma: no cover - network/runtime path
        raise RuntimeError(str(exc)) from exc


def _fallback_analysis(risks: list[Any], positives: list[str], gaps: list[str], alerts: list[str]) -> str:
    if not risks:
        return "Not enough data yet to generate meaningful risk analysis."
    top = risks[0]
    lines: list[str] = []
    if alerts:
        lines.append(f"**Urgent:** {alerts[0]}")
    lines.append(
        f"Top current risk is **{top.name} ({top.probability:.0f}%)**. "
        "This is a directional estimate based on your latest available trends."
    )
    driver = top.why[0] if top.why else "Limited data is lowering confidence."
    lines.append(f"- Why this may be elevated: {driver}")
    if top.next_steps:
        lines.append(f"- Next best action: {top.next_steps[0]}")
    elif gaps:
        lines.append(f"- Next best action: {gaps[0]}")
    elif positives:
        lines.append(f"- Keep doing: {positives[0]}")
    return "\n".join(lines)


def _extract_actions_from_text(text: str) -> list[str]:
    if not text:
        return []
    actions: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        elif len(line) > 2 and line[0].isdigit() and line[1] in {".", ")"}:
            line = line[2:].strip()
        if not line:
            continue
        actions.append(line)
    deduped: list[str] = []
    for item in actions:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _generate_ai_mitigation_actions(
    api_key: str,
    model: str,
    payload: dict[str, Any],
) -> list[str]:
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    if OpenAI is None:
        raise RuntimeError("The `openai` package is not installed. Install dependencies from requirements.txt.")

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a healthcare education assistant. "
        "Generate a short list of practical mitigation actions for the top predicted risk only. "
        "Use only the provided data. "
        "Output 3 to 5 bullets, no intro text. "
        "Prioritize behavior and monitoring actions that reduce risk impact. "
        "Avoid generic advice and avoid asking for extra data collection unless essential. "
        "Do not diagnose."
    )
    user_prompt = "Return mitigation actions for the top risk from this structured payload:\n\n" + json.dumps(payload, indent=2)

    try:
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = _extract_response_text(response)
            actions = _extract_actions_from_text(text)
            if actions:
                return actions

        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = chat.choices[0].message.content if chat.choices else ""
        actions = _extract_actions_from_text(content or "")
        if actions:
            return actions
    except Exception as exc:  # pragma: no cover - network/runtime path
        raise RuntimeError(str(exc)) from exc
    return []


def _generate_ai_search_answer(
    api_key: str,
    model: str,
    payload: dict[str, Any],
    question: str,
) -> str:
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    if OpenAI is None:
        raise RuntimeError("The `openai` package is not installed. Install dependencies from requirements.txt.")
    cleaned_question = _clean_text(question)
    if not cleaned_question:
        raise ValueError("Question is empty.")

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are CareNav's interactive health education assistant. "
        "Answer the user's question using the CareNav structured data first, then general health education context if helpful. "
        "Keep the answer concise, practical, and easy to understand. "
        "Do not diagnose, do not claim certainty, and do not invent measurements or history. "
        "If the question goes beyond the available data, say that clearly. "
        "Format: one short paragraph, then up to 3 bullets if useful."
    )
    user_prompt = (
        "User question:\n"
        + cleaned_question
        + "\n\nCareNav structured data:\n"
        + json.dumps(payload, indent=2)
    )

    try:
        if hasattr(client, "responses"):
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = _extract_response_text(response)
            if text:
                return text

        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = chat.choices[0].message.content if chat.choices else ""
        return (content or "").strip()
    except Exception as exc:  # pragma: no cover - network/runtime path
        raise RuntimeError(str(exc)) from exc


def _normalize_no_selection(values: list[str], none_label: str) -> list[str]:
    if not values:
        return []
    if none_label in values:
        return []
    return [v for v in values if v != none_label]


def _prime_ai_search(question: str, auto_run: bool = False) -> None:
    st.session_state["ai_search_question"] = question
    if auto_run:
        st.session_state["carenav_ai_search_auto_run"] = True


def _build_trend_view(
    ts_df: pd.DataFrame,
    selected_cols: list[str],
    view_mode: str,
) -> pd.DataFrame:
    if ts_df is None or ts_df.empty or not selected_cols:
        return pd.DataFrame()
    work = ts_df[["date", *selected_cols]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date").set_index("date")
    if work.empty:
        return pd.DataFrame()

    if view_mode == "Day":
        return work.tail(90)
    if view_mode == "Week":
        return work.resample("W-MON").mean().dropna(how="all")
    if view_mode == "Month":
        return work.resample("MS").mean().dropna(how="all")
    return work


def _build_priority_actions(risks: list[Any], alerts: list[str]) -> list[str]:
    actions: list[str] = []
    if alerts:
        actions.append("Urgent symptom or reading detected: seek immediate medical evaluation if the alert is accurate/current.")
    if risks:
        top = risks[0]
        top_name = top.name.lower()
        if "sleep" in top_name or "recovery" in top_name:
            actions.extend(
                [
                    "Set a fixed sleep-wake window for the next 7 days, including weekends.",
                    "Reduce late caffeine/alcohol and heavy evening meals to improve overnight recovery.",
                    "Use a low-intensity recovery day after nights with poor sleep.",
                ]
            )
        elif "metabolic" in top_name or "diabetes" in top_name or "glucose" in top_name:
            actions.extend(
                [
                    "Prioritize lower-glycemic meals and pair carbs with protein/fiber.",
                    "Add a 10-20 minute walk after major meals.",
                    "Aim for consistent sleep and meal timing to improve glucose stability.",
                ]
            )
        elif "respiratory" in top_name or "infectious" in top_name:
            actions.extend(
                [
                    "Scale down intensity and prioritize rest/hydration until vitals normalize.",
                    "Track symptom progression daily and escalate care quickly if breathing worsens.",
                    "Avoid high-exertion sessions while symptoms are active.",
                ]
            )
        else:
            actions.extend(
                [
                    "Keep moderate daily movement and avoid sudden spikes in exertion.",
                    "Reduce sodium-heavy meals and alcohol for the next few days.",
                    "Prioritize sleep consistency and stress management to lower physiologic strain.",
                ]
            )

    deduped: list[str] = []
    for item in actions:
        if item not in deduped:
            deduped.append(item)
        if len(deduped) >= 5:
            break
    return deduped[:5]


def _severity_class(severity: str) -> str:
    s = (severity or "").lower()
    if s == "high":
        return "pill-high"
    if s == "moderate":
        return "pill-mod"
    return "pill-low"


def _card(title: str, value: str, subtitle: str = "", tone: str = "") -> None:
    tone_class = f" {tone}" if tone else ""
    st.markdown(
        f"""
        <div class="mini-card{tone_class}">
          <div class="mini-label">{title}</div>
          <div class="mini-value">{value}</div>
          <div class="mini-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_action_cards(actions: list[str]) -> None:
    shown = actions[:3]
    if not shown:
        return
    cols = st.columns(len(shown), gap="medium")
    titles = ["Do First", "Then", "Keep Going"]
    for idx, (col, action) in enumerate(zip(cols, shown), start=1):
        with col:
            title = titles[idx - 1] if idx - 1 < len(titles) else f"Step {idx}"
            st.markdown(
                f"""
                <div class="action-card">
                  <div class="action-step">{idx}</div>
                  <div class="action-title">{title}</div>
                  <div class="action-body">{html.escape(action)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _section_header(title: str, subtitle: str = "", icon: str = "◈", tone: str = "blue") -> None:
    st.markdown(
        f"""
        <div class="section-header-card section-{tone}">
          <div class="section-icon-badge">{html.escape(icon)}</div>
          <div>
            <div class="section-header-title">{html.escape(title)}</div>
            <div class="section-header-sub">{html.escape(subtitle)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _simulator_action_catalog() -> list[dict[str, Any]]:
    return [
        {
            "label": "Sleep +45 min/night",
            "description": "Improve sleep duration and consistency for the next 7 days.",
            "adjustments": {"sleep_hours": 0.75, "hrv": 3.0, "resting_hr": -2.0},
        },
        {
            "label": "Walk +2,000 steps/day",
            "description": "Add one or two extra walks daily to raise baseline movement.",
            "adjustments": {"steps": 2000.0, "resting_hr": -1.0, "glucose_fasting": -4.0},
        },
        {
            "label": "Recovery week",
            "description": "Lower training strain while prioritizing sleep and hydration.",
            "adjustments": {"sleep_hours": 0.5, "hrv": 5.0, "resting_hr": -3.0, "temperature_f": -0.2},
        },
        {
            "label": "Post-meal 15 min walks",
            "description": "Short walks after meals to improve metabolic stability.",
            "adjustments": {"steps": 1200.0, "glucose_fasting": -6.0, "resting_hr": -0.8},
        },
        {
            "label": "No late alcohol/caffeine",
            "description": "Avoid late stimulants and alcohol to improve overnight recovery.",
            "adjustments": {"sleep_hours": 0.4, "hrv": 2.0, "resting_hr": -1.0},
        },
    ]


def _available_sim_actions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = _simulator_action_catalog()
    features = summary.get("features", {})
    available_metrics = {k for k, v in features.items() if v.get("latest") is not None}
    if not available_metrics:
        return catalog[:3]

    scored: list[tuple[int, dict[str, Any]]] = []
    for action in catalog:
        overlap = len(available_metrics & set(action["adjustments"].keys()))
        scored.append((overlap, action))
    scored.sort(key=lambda x: (x[0], len(x[1]["adjustments"])), reverse=True)
    filtered = [action for overlap, action in scored if overlap > 0]
    return filtered[:5] if filtered else catalog[:3]


def _fmt_signed(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}{suffix}"


def _fmt_delta_range(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "Range unavailable"
    lo = value * 0.7
    hi = value * 1.3
    lo_v = min(lo, hi)
    hi_v = max(lo, hi)
    lo_sign = "+" if lo_v > 0 else ""
    hi_sign = "+" if hi_v > 0 else ""
    return f"Expected range: {lo_sign}{lo_v:.1f}{suffix} to {hi_sign}{hi_v:.1f}{suffix}"


def _simulate_week_impact(result: dict[str, Any], adjustments: dict[str, float]) -> dict[str, Any]:
    summary = result.get("summary", {})
    risks = result.get("risks", [])
    manual_inputs = result.get("manual_inputs", {})
    profile = result.get("profile", {})
    symptoms = result.get("symptoms", [])
    conditions = result.get("conditions", [])

    sim_risks, sim_summary = simulate_what_if(
        summary=summary,
        manual_inputs=manual_inputs,
        profile=profile,
        symptoms=symptoms,
        conditions=conditions,
        adjustments=adjustments,
    )

    top_now = risks[0] if risks else None
    sim_map = {r.name: r for r in sim_risks}
    top_after = sim_map.get(top_now.name) if top_now else (sim_risks[0] if sim_risks else None)
    top_delta = None
    if top_now and top_after:
        top_delta = float(top_after.probability - top_now.probability)

    base_load = sum(r.probability for r in risks[:3]) / max(1, len(risks[:3]))
    sim_load = sum(r.probability for r in sim_risks[:3]) / max(1, len(sim_risks[:3]))
    load_delta = float(sim_load - base_load) if sim_risks else None

    base_recovery = summary.get("recovery", {}).get("score")
    sim_recovery = sim_summary.get("recovery", {}).get("score")
    recovery_delta = None
    if base_recovery is not None and sim_recovery is not None:
        recovery_delta = float(sim_recovery) - float(base_recovery)

    feature_changes: list[tuple[str, float]] = []
    feature_map = summary.get("features", {})
    sim_feature_map = sim_summary.get("features", {})
    for metric in adjustments:
        before = feature_map.get(metric, {}).get("latest")
        after = sim_feature_map.get(metric, {}).get("latest")
        if before is None or after is None:
            continue
        delta = float(after) - float(before)
        if abs(delta) < 1e-6:
            continue
        feature_changes.append((metric, delta))
    feature_changes = sorted(feature_changes, key=lambda x: abs(x[1]), reverse=True)[:3]

    impact_score = 50.0
    if top_delta is not None:
        impact_score += max(0.0, -top_delta * 2.8)
        impact_score -= max(0.0, top_delta * 2.2)
    if load_delta is not None:
        impact_score += max(0.0, -load_delta * 2.0)
        impact_score -= max(0.0, load_delta * 1.8)
    if recovery_delta is not None:
        impact_score += max(0.0, recovery_delta * 2.2)
        impact_score -= max(0.0, -recovery_delta * 1.8)
    impact_score = max(0.0, min(100.0, impact_score))
    if impact_score >= 75:
        impact_band = "High projected impact"
    elif impact_score >= 55:
        impact_band = "Moderate projected impact"
    else:
        impact_band = "Low projected impact"

    return {
        "sim_risks": sim_risks,
        "sim_summary": sim_summary,
        "top_now": top_now,
        "top_after": top_after,
        "top_delta": top_delta,
        "load_delta": load_delta,
        "recovery_delta": recovery_delta,
        "feature_changes": feature_changes,
        "impact_score": impact_score,
        "impact_band": impact_band,
    }


def _status_from_risk(risks: list[Any], alerts: list[str], confidence: float) -> dict[str, str]:
    if alerts:
        return {
            "color": "red",
            "label": "Red",
            "title": "Needs attention now",
            "caption": "Current readings or symptoms suggest you should review this promptly.",
        }
    if not risks:
        return {
            "color": "green",
            "label": "Green",
            "title": "Baseline looks stable",
            "caption": "No strong early-warning signal yet. More data improves confidence.",
        }

    top_prob = float(risks[0].probability or 0)
    if top_prob >= 65:
        return {
            "color": "red",
            "label": "Red",
            "title": "Elevated early-warning signal",
            "caption": "The current pattern suggests higher short-term risk than usual.",
        }
    if top_prob >= 35 or confidence < 0.45:
        return {
            "color": "yellow",
            "label": "Yellow",
            "title": "Watch closely this week",
            "caption": "There is a moderate signal worth tracking before it worsens.",
        }
    return {
        "color": "green",
        "label": "Green",
        "title": "On track for now",
        "caption": "Current data points to a lower near-term risk signal.",
    }


def _get_api_key() -> str:
    def _clean(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().strip('"').strip("'")
        return text

    # 1) Environment variables
    for env_name in ["OPENAI_API_KEY", "OPENAI_KEY"]:
        env_val = _clean(os.getenv(env_name, ""))
        if env_val:
            return env_val

    # 2) Streamlit secrets (supports multiple common layouts)
    try:
        # Flat keys
        for key in ["OPENAI_API_KEY", "openai_api_key", "OPENAI_KEY", "openai_key"]:
            val = _clean(st.secrets.get(key, ""))
            if val:
                return val

        # Nested blocks, e.g. [openai] api_key="..."
        for section in ["openai", "OPENAI"]:
            block = st.secrets.get(section, {})
            if hasattr(block, "get"):
                for key in ["api_key", "API_KEY", "openai_api_key", "OPENAI_API_KEY"]:
                    val = _clean(block.get(key, ""))
                    if val:
                        return val
    except Exception:
        return ""

    return ""


def _format_prefill_value(metric: str, value: float | None) -> str:
    if value is None:
        return ""
    if metric in {"steps", "resting_hr", "hrv", "spo2"}:
        return str(int(round(value)))
    return f"{float(value):.1f}"


def _preview_upload(uploaded_file: Any) -> tuple[pd.DataFrame, dict[str, Any] | None, str | None]:
    if uploaded_file is None:
        return pd.DataFrame(), None, None
    try:
        ts = load_health_file(uploaded_file, filename=getattr(uploaded_file, "name", None))
        summary = summarize_timeseries(ts)
        return ts, summary, None
    except Exception as exc:
        return pd.DataFrame(), None, str(exc)


def _prefill_form_from_summary(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return []
    features = summary.get("features", {})
    widget_map = {
        "weight_kg": "weight_text",
        "resting_hr": "resting_hr_text",
        "hrv": "hrv_text",
        "spo2": "spo2_text",
        "sleep_hours": "sleep_hours_text",
        "steps": "steps_text",
        "temperature_f": "temp_f_text",
    }
    filled: list[str] = []
    for metric, widget_key in widget_map.items():
        latest = features.get(metric, {}).get("latest")
        if latest is None:
            continue
        current = st.session_state.get(widget_key, "")
        if str(current).strip():
            continue
        st.session_state[widget_key] = _format_prefill_value(metric, float(latest))
        filled.append(metric)
    return filled


def _recommended_check_in(risks: list[Any], confidence: float, alerts: list[str]) -> str:
    if alerts:
        return "Now / urgent care"
    top_prob = risks[0].probability if risks else 0
    if top_prob >= 70:
        return "Re-check in 24h"
    if top_prob >= 45 or confidence < 0.45:
        return "Re-check in 2-3 days"
    return "Weekly check-in"


def _data_coverage(summary: dict[str, Any]) -> tuple[int, int]:
    features = summary.get("features", {})
    present = sum(1 for c in NUMERIC_COLUMNS if c in features and features[c].get("latest") is not None)
    return present, len(NUMERIC_COLUMNS)


def _run_analysis(
    profile: dict[str, Any],
    manual_inputs: dict[str, Any],
    symptoms: list[str],
    conditions: list[str],
    uploaded_file: Any,
    ai_enabled: bool,
    ai_model: str,
    main_concern: str,
) -> dict[str, Any]:
    ts_df = pd.DataFrame()
    upload_error = None
    if uploaded_file is not None:
        try:
            ts_df = load_health_file(uploaded_file, filename=getattr(uploaded_file, "name", None))
        except Exception as exc:
            upload_error = str(exc)

    summary = summarize_timeseries(ts_df)
    summary = merge_manual_inputs(summary, manual_inputs)
    risks, positives, gaps, alerts = generate_risks(summary, manual_inputs, profile, symptoms, conditions)
    confidence = risks[0].confidence if risks else 0.0

    ai_text = None
    ai_error = None
    ai_actions: list[str] = []
    ai_actions_error = None
    payload: dict[str, Any] = _build_ai_payload(
        profile,
        manual_inputs,
        summary,
        risks,
        positives,
        gaps,
        alerts,
        symptoms,
        conditions,
        main_concern,
    )
    if ai_enabled:
        api_key = _get_api_key()
        if api_key:
            try:
                ai_text = _generate_ai_narrative(api_key=api_key, model=ai_model, payload=payload)
            except Exception as exc:
                ai_error = str(exc)
            try:
                ai_actions = _generate_ai_mitigation_actions(api_key=api_key, model=ai_model, payload=payload)
            except Exception as exc:
                ai_actions_error = str(exc)
        else:
            ai_error = "No OpenAI API key found. Set OPENAI_API_KEY or .streamlit/secrets.toml."

    if not ai_text:
        ai_text = _fallback_analysis(risks, positives, gaps, alerts)

    summary_text = build_summary_text(profile, symptoms, conditions, summary, risks, positives, gaps, alerts)
    return {
        "ts_df": ts_df,
        "upload_error": upload_error,
        "summary": summary,
        "profile": profile,
        "manual_inputs": manual_inputs,
        "symptoms": symptoms,
        "conditions": conditions,
        "risks": risks,
        "positives": positives,
        "gaps": gaps,
        "alerts": alerts,
        "confidence": confidence,
        "ai_text": ai_text,
        "ai_error": ai_error,
        "ai_actions": ai_actions,
        "ai_actions_error": ai_actions_error,
        "summary_text": summary_text,
        "main_concern": main_concern,
        "ai_model": ai_model,
        "ai_payload": payload,
    }


def main() -> None:
    _inject_styles()

    st.markdown(
        """
        <div class="carenav-hero">
          <div class="hero-mark">✦</div>
          <div class="hero-kicker">✦ Early Warning Health Signal</div>
          <h1>CareNav</h1>
          <p>CareNav helps people catch health shifts earlier by turning wearable trends and current readings into one clear signal, one clear reason, and one clear next move before issues escalate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### AI Settings")
        ai_enabled = st.toggle("Use ChatGPT analysis", value=True)
        model_preset = st.selectbox(
            "Model",
            ["gpt-5.2", "gpt-5.2-chat-latest", "gpt-5.2-pro", "gpt-5", "gpt-5-mini", "gpt-5.3-codex", "gpt-4.1-mini", "gpt-4.1"],
            index=0,
            help="Pick a model your account can access.",
            key="model_select_sidebar",
        )
        custom_model = st.text_input("Custom model", placeholder="e.g. gpt-5.2", key="custom_model_sidebar")
        ai_model = _clean_text(custom_model) or model_preset
        detected_key = _get_api_key()
        if detected_key:
            st.caption("OpenAI key detected.")
        else:
            st.caption("Set `OPENAI_API_KEY` or `.streamlit/secrets.toml` to enable ChatGPT analysis.")
            with st.expander("API key setup example"):
                st.code('OPENAI_API_KEY = "sk-..."', language="toml")
        st.markdown("### AI Output")
        st.caption("CareNav uses ChatGPT for a short explanation and mitigation guidance after the risk engine scores the data.")

    st.markdown("### Inputs")
    st.caption("Start with what you know. Add wearable data if available.")

    input_col, wearable_col = st.columns([1.55, 0.95], gap="large")
    with wearable_col:
        st.markdown("### Wearable Data")
        uploaded_file = st.file_uploader("Upload CSV or Apple Health XML", type=["csv", "xml"])
        st.download_button(
            "Sample CSV template",
            data=_sample_csv().to_csv(index=False).encode("utf-8"),
            file_name="carenav_sample_wearable_vitals.csv",
            mime="text/csv",
        )
        with st.expander("Supported columns"):
            st.code(", ".join(["date", *NUMERIC_COLUMNS]), language="text")
            st.caption("CareNav auto-maps many Apple Health and WHOOP-style CSV formats, and also accepts Apple Health `export.xml` directly.")

    upload_preview_df, upload_preview_summary, upload_preview_error = _preview_upload(uploaded_file)
    prefilled_metrics = _prefill_form_from_summary(upload_preview_summary)

    with wearable_col:
        if upload_preview_error:
            st.warning(f"File preview could not be parsed yet: {upload_preview_error}")
        elif upload_preview_summary and prefilled_metrics:
            pretty = ", ".join(m.replace("_", " ") for m in prefilled_metrics[:6])
            extra = "" if len(prefilled_metrics) <= 6 else f" +{len(prefilled_metrics) - 6} more"
            st.markdown(
                f'<div class="subtle-note">File parsed and prefilled blank fields from latest readings: {pretty}{extra}.</div>',
                unsafe_allow_html=True,
            )
        if upload_preview_summary and upload_preview_summary.get("features"):
            preview_rows = []
            feature_order = ["resting_hr", "hrv", "sleep_hours", "steps", "spo2", "weight_kg", "temperature_f"]
            for metric in feature_order:
                item = upload_preview_summary.get("features", {}).get(metric)
                if not item or item.get("latest") is None:
                    continue
                preview_rows.append(
                    {
                        "Metric": metric.replace("_", " "),
                        "Latest": round(float(item["latest"]), 2),
                        "7d avg": None if item.get("avg7") is None else round(float(item["avg7"]), 2),
                        "Records": item.get("count"),
                    }
                )
            if preview_rows:
                st.markdown('<div class="csv-preview-card">', unsafe_allow_html=True)
                st.markdown("**Detected from CSV**")
                st.dataframe(
                    pd.DataFrame(preview_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=min(300, 45 + len(preview_rows) * 35),
                )
                st.markdown("</div>", unsafe_allow_html=True)

    with input_col:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        form = st.form("carenav_form", clear_on_submit=False)
        with form:
            p1, p2, p3 = st.columns(3)
            with p1:
                age_text = st.text_input("Age", placeholder="e.g. 42", key="age_text")
                height_text = st.text_input("Height cm", placeholder="e.g. 170", key="height_text")
                sex = st.selectbox("Sex", ["Female", "Male", "Other / Prefer not to say"], index=None, key="sex_select")
            with p2:
                weight_text = st.text_input("Weight kg", placeholder="e.g. 72.5", key="weight_text")
                smoking_status = st.selectbox("Smoking status", ["Never", "Former", "Current"], index=None, key="smoking_status_select")
                main_concern_choices = st.multiselect(
                    "Main concerns today",
                    [
                        "No concerns",
                        "High blood pressure",
                        "Poor sleep / recovery",
                        "Fatigue / low energy",
                        "Stress / burnout",
                        "High resting heart rate",
                        "Low HRV",
                        "Blood sugar concerns",
                        "Breathing / oxygen concerns",
                        "Weight gain / fluid retention",
                        "Palpitations",
                    ],
                    default=[],
                    placeholder="Choose any that apply",
                    key="main_concern_choices",
                )
                main_concern_other = st.text_input("Other concern", placeholder="If not listed", key="main_concern_other")
            with p3:
                conditions = st.multiselect(
                    "Known conditions",
                    [
                        "Hypertension",
                        "Prediabetes",
                        "Diabetes",
                        "Heart failure",
                        "Coronary artery disease",
                        "Asthma",
                        "COPD",
                        "Kidney disease",
                        "Sleep apnea",
                    ],
                    default=[],
                    placeholder="Select any that apply",
                    key="conditions_select",
                )
                symptoms = st.multiselect(
                    "Current symptoms",
                    [
                        "No symptoms",
                        "Chest pain",
                        "Shortness of breath",
                        "Cough",
                        "Fever",
                        "Fatigue",
                        "Poor sleep",
                        "Palpitations",
                        "Leg swelling",
                        "Dizziness",
                        "Confusion",
                    ],
                    default=[],
                    placeholder="Select any that apply",
                    key="symptoms_select",
                )

            st.markdown("#### Current Measurements")
            m1, m2, m3 = st.columns(3)
            with m1:
                resting_hr_text = st.text_input("Resting HR", placeholder="e.g. 62", key="resting_hr_text")
                hrv_text = st.text_input("HRV (ms)", placeholder="e.g. 40", key="hrv_text")
                spo2_text = st.text_input("SpO2 (%)", placeholder="e.g. 98", key="spo2_text")
            with m2:
                sleep_hours_text = st.text_input("Sleep hours", placeholder="e.g. 7.2", key="sleep_hours_text")
                steps_text = st.text_input("Steps", placeholder="e.g. 6500", key="steps_text")
                temp_f_text = st.text_input("Temperature F", placeholder="e.g. 98.6", key="temp_f_text")
            with m3:
                st.info(
                    "Fastest value: add resting HR, sleep, steps, and HRV. Uploads can fill blanks automatically from wearable data."
                )

            submitted = st.form_submit_button("Run AI-powered Analysis", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        age = _parse_optional_int(age_text)
        height_cm = _parse_optional_float(height_text)
        weight_kg = _parse_optional_float(weight_text)
        bmi = None
        if height_cm and weight_kg and height_cm > 0:
            bmi = weight_kg / ((height_cm / 100) ** 2)

        normalized_concerns = _normalize_no_selection(main_concern_choices, "No concerns")
        normalized_symptoms = _normalize_no_selection(symptoms, "No symptoms")

        concern_parts = [c for c in normalized_concerns if c]
        if _clean_text(main_concern_other) and "No concerns" not in main_concern_choices:
            concern_parts.append(_clean_text(main_concern_other))
        main_concern = ", ".join(concern_parts)

        profile = {
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "smoking_status": smoking_status or "Unknown",
        }
        manual_inputs = {
            "resting_hr": _parse_optional_float(resting_hr_text),
            "hrv": _parse_optional_float(hrv_text),
            "spo2": _parse_optional_float(spo2_text),
            "sleep_hours": _parse_optional_float(sleep_hours_text),
            "steps": _parse_optional_float(steps_text),
            "temperature_f": _parse_optional_float(temp_f_text),
            "weight_kg": weight_kg,
        }
        st.session_state["carenav_result"] = _run_analysis(
            profile=profile,
            manual_inputs=manual_inputs,
            symptoms=normalized_symptoms,
            conditions=conditions,
            uploaded_file=uploaded_file,
            ai_enabled=ai_enabled,
            ai_model=ai_model,
            main_concern=main_concern,
        )

    result = st.session_state.get("carenav_result")
    if not result:
        st.info("Add any values you know above and click `Run AI-powered Analysis` to generate an early-warning summary.")
        return

    summary = result["summary"]
    risks = result["risks"]
    alerts = result["alerts"]
    positives = result["positives"]
    gaps = result["gaps"]
    ts_df = result["ts_df"]
    upload_error = result["upload_error"]
    confidence = float(result["confidence"] or 0)
    ai_text = result["ai_text"]
    ai_error = result["ai_error"]
    ai_actions = result.get("ai_actions", [])
    ai_actions_error = result.get("ai_actions_error")
    summary_text = result["summary_text"]
    ai_payload = result.get("ai_payload", {})

    if upload_error:
        st.warning(f"Upload could not be parsed: {upload_error}")

    st.markdown(
        f"""
        <div class="results-header">
          <div>
            <div class="results-title">Your Weekly Health Signal</div>
            <div class="results-subtle">This view is built for early warning and next steps, not diagnosis.</div>
          </div>
          <div class="info-chip">✦ {summary.get("n_days", 0)} tracked days · {len(ts_df) if ts_df is not None else 0} records reviewed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if alerts:
        st.error("Escalation signal: " + " | ".join(alerts[:2]))

    top_risk = risks[0] if risks else None
    top_prob = f"{top_risk.probability:.0f}%" if top_risk else "N/A"
    check_in = _recommended_check_in(risks, confidence, alerts)
    present_metrics, total_metrics = _data_coverage(summary)
    primary_actions = ai_actions if ai_actions else _build_priority_actions(risks, alerts)
    status = _status_from_risk(risks, alerts, confidence)
    main_watch = top_risk.name if top_risk else "No strong signal detected"
    main_action = primary_actions[0] if primary_actions else "Keep healthy routines steady and check in again after new readings."
    main_positive = positives[0] if positives else "No clear protective signal yet."

    st.markdown(
        f"""
        <div class="status-panel status-{status["color"]}">
          <div class="status-row">
            <div class="status-badge status-{status["color"]}">{status["label"]} Signal</div>
            <div>
              <div class="status-title">{status["title"]}</div>
              <div class="status-caption">{status["caption"]}</div>
            </div>
          </div>
          <div class="focus-item">
            <div class="focus-label">Main thing to watch this week</div>
            <div class="focus-value">{main_watch} ({top_prob})</div>
          </div>
          <div class="focus-item">
            <div class="focus-label">Best next move</div>
            <div class="focus-value">{main_action}</div>
          </div>
          <div class="focus-item">
            <div class="focus-label">Positive behaviour to keep</div>
            <div class="focus-value">{main_positive}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    window_label = f"{summary.get('n_days', 0)} tracked days"
    freshness = summary.get("freshness_days")
    freshness_text = "Current" if freshness in (None, 0) else f"{freshness} day(s) old"
    recovery = summary.get("recovery") or {}
    recovery_score = recovery.get("score")
    recovery_label = str(recovery.get("label", "Watch"))
    recovery_tone = "green" if recovery_label == "Recovered" else "yellow" if recovery_label == "Watch" else "red"
    third_tone = recovery_tone if recovery_score is not None else status["color"]
    third_label = "Recovery score" if recovery_score is not None else "Data window"
    third_value = f"{float(recovery_score):.0f}/100" if recovery_score is not None else window_label
    third_sub = recovery_label if recovery_score is not None else freshness_text
    st.markdown(
        f"""
        <div class="stats-strip">
          <div class="stats-pill purple">
            <div class="stats-label">Confidence</div>
            <div class="stats-value">{confidence:.0%}</div>
            <div class="stats-sub">Data coverage {present_metrics}/{total_metrics}</div>
          </div>
          <div class="stats-pill">
            <div class="stats-label">Next check-in</div>
            <div class="stats-value">{html.escape(check_in)}</div>
            <div class="stats-sub">Suggested cadence</div>
          </div>
          <div class="stats-pill {third_tone}">
            <div class="stats-label">{html.escape(third_label)}</div>
            <div class="stats-value">{html.escape(third_value)}</div>
            <div class="stats-sub">{html.escape(third_sub)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _section_header(
        title="Top Risk Explanation",
        subtitle="Plain-language view of your highest current risk signal.",
        icon="◉",
        tone="blue",
    )
    if ai_error and _get_api_key():
        st.warning(f"ChatGPT analysis failed, showing local summary instead. Error: {ai_error}")
    elif ai_error and "No OpenAI API key" in ai_error:
        st.info("ChatGPT key not configured. Showing local summary.")
    st.markdown(ai_text)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    _section_header(
        title="What To Do Next",
        subtitle="Focused action plan for the next 7 days.",
        icon="▣",
        tone="cyan",
    )
    if ai_actions_error and _get_api_key():
        st.caption("AI action guidance is unavailable right now. Showing local fallback suggestions.")
    if primary_actions:
        _render_action_cards(primary_actions)
    else:
        _render_action_cards(["Keep healthy routines consistent and rerun after new readings are available."])

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    _section_header(
        title="1-Week Impact Simulator",
        subtitle="Estimate how one positive habit may shift risk and recovery.",
        icon="◆",
        tone="green",
    )
    st.markdown(
        """
        <div class="sim-panel">
          <div class="sim-title">Simulate one positive action for the next 7 days</div>
          <div class="sim-subtle">Pick one habit and preview how much it may reduce risk and improve recovery. This is an estimate, not a diagnosis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sim_actions = _available_sim_actions(summary)
    sim_labels = [a["label"] for a in sim_actions]
    sim_selected_label = st.selectbox(
        "Action to simulate",
        sim_labels,
        index=0 if sim_labels else None,
        key="sim_action_select",
    )
    selected_action = next((a for a in sim_actions if a["label"] == sim_selected_label), None)
    if selected_action:
        st.caption(selected_action["description"])
        sim_outcome = _simulate_week_impact(result, selected_action["adjustments"])

        sim_top_delta = sim_outcome.get("top_delta")
        sim_recovery_delta = sim_outcome.get("recovery_delta")
        sim_load_delta = sim_outcome.get("load_delta")
        sim_score = float(sim_outcome.get("impact_score", 0.0))
        sim_band = str(sim_outcome.get("impact_band", "Projected impact"))

        s1, s2, s3 = st.columns(3)
        with s1:
            delta_display = _fmt_signed(-sim_top_delta if sim_top_delta is not None else None, " pts")
            st.metric("Top risk reduction", delta_display)
            st.markdown(f'<div class="sim-range">{_fmt_delta_range(-sim_top_delta if sim_top_delta is not None else None, " pts")}</div>', unsafe_allow_html=True)
        with s2:
            st.metric("Recovery score change", _fmt_signed(sim_recovery_delta, " pts"))
            st.markdown(f'<div class="sim-range">{_fmt_delta_range(sim_recovery_delta, " pts")}</div>', unsafe_allow_html=True)
        with s3:
            delta_display = _fmt_signed(-sim_load_delta if sim_load_delta is not None else None, " pts")
            st.metric("Overall risk load", delta_display)
            st.markdown(f'<div class="sim-range">{_fmt_delta_range(-sim_load_delta if sim_load_delta is not None else None, " pts")}</div>', unsafe_allow_html=True)

        st.progress(sim_score / 100.0, text=f"Impact score: {sim_score:.0f}/100 · {sim_band}")

        feature_changes = sim_outcome.get("feature_changes", [])
        if feature_changes:
            lines = []
            for metric, delta in feature_changes:
                unit = " h" if metric == "sleep_hours" else ""
                lines.append(f"- **{metric.replace('_', ' ')}:** {_fmt_signed(delta, unit)}")
            st.markdown("Likely metric shifts")
            st.markdown("\n".join(lines))

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    _section_header(
        title="Ask CareNav AI",
        subtitle="Interactive Q&A based on your results and trends.",
        icon="✦",
        tone="purple",
    )
    with st.container(border=True):
        st.markdown(
            """
            <div class="ask-banner">
              <div class="ai-search-title">AI Follow-up</div>
              <div class="results-subtle">Use this for quick follow-up questions about your results, trends, and practical next steps.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        q1, q2 = st.columns([1.35, 0.65])
        with q1:
            ai_question = st.text_input(
                "Your question",
                placeholder="e.g. Why is recovery risk high for me right now?",
                key="ai_search_question",
                label_visibility="collapsed",
            )
        with q2:
            ask_clicked = st.button("✦ Ask AI", use_container_width=True, type="primary")

        st.markdown('<div class="quick-prompt-note">Suggested questions</div>', unsafe_allow_html=True)
        quick_q_cols = st.columns(3)
        quick_questions = [
            "What does my top risk mean?",
            "What can I do this week to reduce this risk?",
            "Which trend matters most right now?",
        ]
        for col, text in zip(quick_q_cols, quick_questions):
            with col:
                st.button(
                    text,
                    key=f"quick_q_{text}",
                    use_container_width=True,
                    on_click=_prime_ai_search,
                    args=(text, True),
                )

    ask_clicked = ask_clicked or bool(st.session_state.pop("carenav_ai_search_auto_run", False))

    if ask_clicked:
        question_to_run = _clean_text(st.session_state.get("ai_search_question", ai_question))
        if not question_to_run:
            st.info("Enter a question first.")
        elif not ai_enabled:
            st.info("Enable ChatGPT analysis in the sidebar to use AI search.")
        else:
            api_key = _get_api_key()
            if not api_key:
                st.warning("No OpenAI API key detected. Add it in Streamlit secrets or env vars to use AI search.")
            else:
                with st.spinner("Asking CareNav AI..."):
                    try:
                        answer = _generate_ai_search_answer(
                            api_key=api_key,
                            model=result.get("ai_model", ai_model),
                            payload=ai_payload,
                            question=question_to_run,
                        )
                        st.session_state["carenav_ai_search_answer"] = answer
                        st.session_state["carenav_ai_search_question_last"] = question_to_run
                    except Exception as exc:
                        st.error(f"AI search failed: {exc}")

    search_answer = st.session_state.get("carenav_ai_search_answer")
    search_question_last = st.session_state.get("carenav_ai_search_question_last")
    if search_answer:
        with st.container(border=True):
            st.markdown('<div class="ask-answer-label">AI response</div>', unsafe_allow_html=True)
            if search_question_last:
                st.caption(f"Question: {search_question_last}")
            st.markdown(search_answer)

    with st.expander("◌ Signals Snapshot"):
        chips = "".join(
            [
                f'<span class="pill {_severity_class(r.severity)}">{html.escape(r.name)}: {r.probability:.0f}%</span>'
                for r in risks[:3]
            ]
        )
        st.markdown(f'<div class="snapshot-panel">{chips}</div>', unsafe_allow_html=True)

    with st.expander("◌ More Detail"):
        for r in risks[:3]:
            st.markdown(f"#### {r.name}")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Probability", f"{r.probability:.0f}%")
            d2.metric("Severity", r.severity)
            d3.metric("Trend", r.trend)
            d4.metric("Confidence", f"{r.confidence:.0%}")
            if r.why:
                st.write("Why")
                for line in r.why:
                    st.write(f"- {line}")
            if r.protective_factors:
                st.write("Protective factors")
                for line in r.protective_factors:
                    st.write(f"- {line}")
            if r.next_steps:
                st.write("Next monitoring/actions")
                for line in r.next_steps[:3]:
                    st.write(f"- {line}")
            st.divider()

    with st.expander("◌ Trend Data (Uploaded File)"):
        if ts_df is None or ts_df.empty:
            st.caption("No file uploaded. Analysis used manual inputs only.")
        else:
            st.write(
                f"Loaded {len(ts_df)} rows from {summary.get('date_min').date()} to {summary.get('date_max').date()}."
            )
            present_cols = [c for c in NUMERIC_COLUMNS if c in ts_df.columns and ts_df[c].notna().any()]
            trend_granularity = st.radio(
                "View",
                ["Day", "Week", "Month"],
                horizontal=True,
                key="trend_granularity",
            )
            selected_cols = st.multiselect(
                "Metrics to chart",
                present_cols,
                default=present_cols[:3],
                placeholder="Choose metrics",
                key="trend_chart_metrics",
            )
            if selected_cols:
                chart_df = _build_trend_view(ts_df, selected_cols, trend_granularity)
                if chart_df.empty:
                    st.caption("No chartable points for this view.")
                else:
                    metric_palette = {
                        "resting_hr": "#38bdf8",
                        "hrv": "#3b82f6",
                        "sleep_hours": "#fca5a5",
                        "steps": "#34d399",
                        "spo2": "#22d3ee",
                        "weight_kg": "#c084fc",
                        "systolic_bp": "#f59e0b",
                        "diastolic_bp": "#fb7185",
                        "glucose_fasting": "#f97316",
                        "temperature_f": "#f43f5e",
                    }
                    legend_domain = list(selected_cols)
                    legend_range = [metric_palette.get(metric, "#94a3b8") for metric in legend_domain]
                    legend_html = "".join(
                        [
                            (
                                '<span class="legend-chip">'
                                f'<span class="legend-swatch" style="background:{metric_palette.get(metric, "#94a3b8")}"></span>'
                                f'{metric.replace("_", " ")}'
                                "</span>"
                            )
                            for metric in selected_cols
                        ]
                    )
                    st.markdown(f'<div class="legend-wrap">{legend_html}</div>', unsafe_allow_html=True)
                    chart_long = (
                        chart_df.reset_index()
                        .melt(id_vars="date", var_name="Metric", value_name="Value")
                        .dropna(subset=["Value"])
                    )
                    line = (
                        alt.Chart(chart_long)
                        .mark_line()
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y("Value:Q", title="Value"),
                            color=alt.Color(
                                "Metric:N",
                                legend=None,
                                scale=alt.Scale(
                                    domain=legend_domain,
                                    range=legend_range,
                                ),
                            ),
                            tooltip=["date:T", "Metric:N", "Value:Q"],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(line, use_container_width=True)
                    if trend_granularity == "Day":
                        st.caption("Daily values (last 90 days).")
                    elif trend_granularity == "Week":
                        st.caption("Weekly averages.")
                    else:
                        st.caption("Monthly averages.")
            st.dataframe(ts_df.tail(14), use_container_width=True, hide_index=True)

    with st.expander("◌ Care Team Summary / Export"):
        st.text_area("Summary text", value=summary_text, height=320)
        st.download_button(
            "Download summary (.txt)",
            data=summary_text.encode("utf-8"),
            file_name="carenav_summary.txt",
            mime="text/plain",
        )
        if ai_text:
            st.download_button(
                "Download AI analysis (.md)",
                data=ai_text.encode("utf-8"),
                file_name="carenav_ai_analysis.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()

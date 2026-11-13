"""
EviGuard AI — Enterprise SaaS AI Proctoring & Cyber-Command Platform
Pristine Live-Testing Production Architecture (Zero Mock Data / Fresh Initialization).

Features:
1. Dynamic State & Cache Reset:
   - Programmatic cache purging (`@st.cache_data.clear()` / `@st.cache_resource.clear()`).
   - Clean session state initialization with empty data structures.
   - 1-Click System Data Reset & Storage Purge button.
2. Zero Mock/Hardcoded Placeholders:
   - Dynamically generated session timestamps.
   - Onboarding candidate registration flow for real-world testing.
3. High-Contrast Midnight Indigo Aesthetic & Modular 4-View Architecture:
   - Executive Overview & Metrics (Clean zero counters on fresh session).
   - Real-Time Vision & AI Proctoring (Zero-latency live OpenCV stream).
   - Threat Telemetry & Behavioral Analytics (Dynamic timeline & anomaly breakdown).
   - Audit Dossier & Neural Settings (Evidence review, XAI attributions, PDF/CSV export).
"""

import sys
import os

# Add parent directory to sys.path and register packages at the very top
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import backend
import backend.detection
import backend.db
import backend.explainability
import backend.pose
import backend.reporting
import backend.scoring
import backend.tracking

from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import math
import shutil
import socket
import threading
import time
from typing import Dict, Any, List, Optional
import warnings
import streamlit.components.v1 as components

# Suppress background telemetry & deprecation warnings
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from backend.db.models import DatabaseManager, ExamSession, Incident, RiskMetricLog
from backend.detection.base import DetectionResult
from backend.explainability.reason_generator import ReasonGenerator
from backend.pipeline import EviGuardPipeline, PipelineOutput
from backend.reporting.report_generator import generate_candidate_pdf_report, generate_candidate_csv_report


# ==============================================================================
# 1. STREAMLIT SESSION STATE & SYSTEM INITIALIZATION
# ==============================================================================
if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = True
    st.session_state.incident_logs = []
    st.session_state.flagged_events = []
    st.session_state.evidence_snapshots = []
    st.session_state.audit_history = []
    st.session_state.telemetry_data = []
    st.session_state.active_session_id = None
    st.session_state.live_stream_active = False


# ==============================================================================
# 2. PAGE SETUP & MIDNIGHT INDIGO HIGH-CONTRAST STYLESHEET
# ==============================================================================
st.set_page_config(
    page_title="EviGuard AI — Enterprise Proctoring Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:ital,wght@0,400;0,600;0,700;0,800;1,400&family=Outfit:wght@600;700;800;900&display=swap');

    /* =========================================================================
       1. AMBIENT TOP LIGHT STRIP & EDITORIAL CANVAS
       ========================================================================= */
    @keyframes ambientGlowShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Animated gradient light bar along top edge */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2563EB, #38BDF8, #FF4D00, #2563EB);
        background-size: 300% 100%;
        animation: ambientGlowShift 8s ease infinite;
        z-index: 999999;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #F6F7F9 !important;
        background-image: 
            radial-gradient(circle at 14% 12%, rgba(37, 99, 235, 0.035) 0%, transparent 45%),
            radial-gradient(circle at 86% 16%, rgba(255, 77, 0, 0.025) 0%, transparent 40%),
            radial-gradient(circle at 75% 85%, rgba(4, 120, 87, 0.025) 0%, transparent 45%),
            linear-gradient(rgba(15, 23, 42, 0.018) 1px, transparent 1px),
            linear-gradient(90deg, rgba(15, 23, 42, 0.018) 1px, transparent 1px) !important;
        background-size: 100% 100%, 100% 100%, 100% 100%, 36px 36px, 36px 36px !important;
        background-attachment: fixed !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0F172A !important;
    }

    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }

    /* Remove default Streamlit top bar / header clutter */
    header[data-testid="stHeader"] {
        height: 0px !important;
        min-height: 0px !important;
        padding: 0 !important;
        background: transparent !important;
        visibility: hidden !important;
    }

    /* Screen-fitting compact container padding */
    .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        max-width: 98% !important;
        margin: 0 auto !important;
    }

    /* =========================================================================
       2. TAB NAVIGATION & ANIMATED SECTION SWITCHING
       ========================================================================= */
    @keyframes sectionReveal {
        0% {
            opacity: 0;
            transform: translateY(8px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    div[data-baseweb="tab-panel"],
    div[role="tabpanel"],
    div[data-testid="stVerticalBlock"] > div[data-baseweb="tab-panel"] {
        animation: sectionReveal 0.38s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: #FFFFFF !important;
        padding: 6px 10px !important;
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04) !important;
        margin-bottom: 16px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 10px !important;
        color: #475569 !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 8px 18px !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border: 1px solid transparent !important;
        position: relative !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        background: #F8FAFC !important;
    }

    .stTabs [aria-selected="true"] {
        background: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        color: #1D4ED8 !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.10) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: linear-gradient(90deg, #2563EB, #FF4D00) !important;
        box-shadow: 0 0 10px rgba(255, 77, 0, 0.4) !important;
        height: 3px !important;
        border-radius: 9999px !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        background-color: transparent !important;
    }

    /* =========================================================================
       3. EDITORIAL CARDS & INTERACTIVE HOVER LIGHTING
       ========================================================================= */
    div[data-testid="stVerticalBlockBorderWrapper"],
    .enterprise-kpi-card,
    .telemetry-stat-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04) !important;
        position: relative !important;
        overflow: hidden !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
        margin-bottom: 12px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover,
    .enterprise-kpi-card:hover,
    .telemetry-stat-card:hover {
        border-color: rgba(37, 99, 235, 0.35) !important;
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.12), 0 0 0 1px rgba(37, 99, 235, 0.35) !important;
        transform: translateY(-2px) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]::before,
    .enterprise-kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(37, 99, 235, 0.5), transparent);
    }

    /* =========================================================================
       4. TYPOGRAPHY & HEADINGS
       ========================================================================= */
    h1, .page-title {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: #0F172A !important;
        margin-bottom: 4px !important;
    }

    h2, h3, .section-header {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 4px !important;
    }

    p, span, li, .body-text {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 450 !important;
        color: #334155 !important;
        line-height: 1.55 !important;
    }

    .eyebrow-label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #2563EB !important;
    }

    .kpi-value-text {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
    }

    /* KPI Card Inner Elements */
    .kpi-card-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .kpi-icon-badge {
        font-size: 1.15rem;
        filter: drop-shadow(0 2px 4px rgba(37, 99, 235, 0.2));
    }
    .kpi-footer-status {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 6px;
    }
    .kpi-status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }

    /* =========================================================================
       5. COMPACT TOP EXECUTIVE NAVBAR
       ========================================================================= */
    .saas-top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 10px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        transition: all 0.3s ease;
    }
    .saas-top-navbar:hover {
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
    }

    .saas-brand-title {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .saas-logo-icon {
        font-size: 1.6rem;
        filter: drop-shadow(0 2px 6px rgba(37, 99, 235, 0.25));
    }

    .saas-brand-text {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
        font-size: 19px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0F172A;
    }

    .saas-brand-badge {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        font-family: 'JetBrains Mono', monospace;
    }

    .saas-candidate-chip {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 6px 14px;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        font-size: 0.88rem;
    }
    .saas-chip-label {
        color: #64748B;
        font-weight: 600;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        font-family: 'JetBrains Mono', monospace;
    }
    .saas-chip-name {
        font-weight: 700;
        color: #0F172A;
    }
    .saas-chip-id {
        color: #2563EB;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }

    /* =========================================================================
       6. STATUS BADGES & ACCENTS
       ========================================================================= */
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; transform: scale(1); filter: drop-shadow(0 0 6px rgba(4, 120, 87, 0.6)); }
        50% { opacity: 0.65; transform: scale(0.96); filter: drop-shadow(0 0 2px rgba(4, 120, 87, 0.2)); }
    }

    .badge-luxury-ready, .pill-safe {
        background: #ECFDF5 !important;
        color: #047857 !important;
        border: 1px solid #A7F3D0 !important;
        padding: 4px 12px !important;
        border-radius: 9999px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        box-shadow: 0 1px 4px rgba(4, 120, 87, 0.08) !important;
    }
    .badge-luxury-ready::before, .pill-safe::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #047857;
        box-shadow: 0 0 6px rgba(4, 120, 87, 0.6);
        animation: pulse-glow 2s infinite ease-in-out;
    }

    .pill-alert {
        background: #FEF2F2 !important;
        color: #B91C1C !important;
        border: 1px solid #FECACA !important;
        padding: 4px 10px !important;
        border-radius: 9999px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 5px !important;
        box-shadow: 0 1px 4px rgba(185, 28, 28, 0.08) !important;
    }

    .pill-warn {
        background: #FFFBEB !important;
        color: #B45309 !important;
        border: 1px solid #FDE68A !important;
        padding: 4px 10px !important;
        border-radius: 9999px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 5px !important;
        box-shadow: 0 1px 4px rgba(180, 83, 9, 0.08) !important;
    }

    /* Video HUD Topbar */
    .video-hud-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 8px 14px;
        margin-bottom: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
    }
    .hud-rec-indicator {
        color: #DC2626;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    .hud-spec-tag {
        color: #64748B;
        font-weight: 600;
    }
    .hud-encryption-tag {
        color: #2563EB;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    /* =========================================================================
       7. BUTTONS & INPUT CONTROLS
       ========================================================================= */
    /* Primary Action Buttons */
    button[kind="primary"], 
    button[data-testid="baseButton-primary"], 
    .stFormSubmitButton > button,
    div[data-testid="stDownloadButton"] > button {
        background: #2563EB !important;
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 10px 22px !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100% !important;
    }
    button[kind="primary"]:hover, 
    button[data-testid="baseButton-primary"]:hover, 
    .stFormSubmitButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: #1D4ED8 !important;
        background-color: #1D4ED8 !important;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.35) !important;
        transform: translateY(-2px) !important;
    }

    /* Secondary / Standard Action Buttons */
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"],
    div[data-testid="stButton"] > button:not([kind="primary"]) {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
    }
    div[data-testid="stButton"] > button[data-testid="baseButton-secondary"]:hover,
    div[data-testid="stButton"] > button:not([kind="primary"]):hover {
        background: #F8FAFC !important;
        color: #0F172A !important;
        border-color: #2563EB !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12) !important;
        transform: translateY(-1px) !important;
    }

    /* Destructive Action Container (Reset All Exam Data button) */
    .danger-action-container div[data-testid="stButton"] > button,
    .danger-action-container button {
        background: #FEF2F2 !important;
        background-color: #FEF2F2 !important;
        border: 1px solid #FECACA !important;
        color: #B91C1C !important;
        border-radius: 10px !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 10px 18px !important;
        box-shadow: 0 2px 8px rgba(185, 28, 28, 0.06) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100% !important;
    }
    .danger-action-container div[data-testid="stButton"] > button:hover,
    .danger-action-container button:hover {
        background: #DC2626 !important;
        background-color: #DC2626 !important;
        border-color: #DC2626 !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.35) !important;
        transform: translateY(-2px) !important;
    }

    /* Form Inputs, Selectboxes & Textareas */
    .stTextInput>div>div>input, 
    div[data-baseweb="select"] > div,
    select, textarea {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 8px 14px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
    }
    .stTextInput>div>div>input:focus, 
    div[data-baseweb="select"] > div:focus-within,
    select:focus, textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
    }

    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08) !important;
    }
    li[data-baseweb="menu-item"] {
        color: #334155 !important;
        background-color: transparent !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
    }
    li[data-baseweb="menu-item"]:hover, li[aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
    }

    /* Multiselect Tags */
    span[data-baseweb="tag"] {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 6px !important;
        color: #1D4ED8 !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #2563EB !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 8px rgba(37, 99, 235, 0.5) !important;
    }
    div[data-testid="stSliderTrack"] > div:first-child {
        background-color: #2563EB !important;
    }
    div[data-testid="stSliderTrack"] {
        background-color: #E2E8F0 !important;
    }
    .stSlider label p, .stSelectbox label p, .stTextInput label p, .stMultiSelect label p {
        color: #334155 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    }

    /* Exam Rule Cards Styling */
    .exam-rules-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-top: 6px;
    }
    .exam-rule-card {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 16px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .exam-rule-card:hover {
        background: #FFFFFF;
        border-color: #CBD5E1;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        transform: translateY(-2px);
    }
    .exam-rule-icon {
        width: 44px;
        height: 44px;
        min-width: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        border: 1px solid;
    }
    .exam-rule-body {
        flex: 1;
    }
    .exam-rule-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 5px;
    }
    .exam-rule-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .exam-rule-desc {
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.55;
        margin: 0;
        font-weight: 450;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .rule-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 9999px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .rule-badge-danger {
        background: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    .rule-badge-warning {
        background: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .rule-badge-info {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .rule-badge-success {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }

    /* Telemetry HUD Ribbons */
    .telemetry-stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 12px 16px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .telemetry-stat-card:hover {
        border-color: rgba(37, 99, 235, 0.35);
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.12), 0 0 0 1px rgba(37, 99, 235, 0.35);
        transform: translateY(-2px);
    }
    .telemetry-stat-label {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .telemetry-stat-value {
        font-size: 1.45rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        line-height: 1.2;
        color: #0F172A;
    }

    /* Video & Chart Containers */
    [data-testid="stImage"] img, video, .js-plotly-plot {
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04) !important;
    }

    /* Telemetry Row Layout */
    .enterprise-telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-bottom: 1px solid #F1F5F9;
        border-radius: 10px;
        margin-bottom: 6px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .enterprise-telemetry-row:hover {
        background: #F8FAFC;
        border-color: #CBD5E1;
        transform: translateY(-1px);
    }

    /* Dossier Meta Grid */
    .dossier-meta-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 6px;
        margin-bottom: 10px;
    }
    .dossier-meta-item {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 6px 12px;
    }
    .dossier-meta-label {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        color: #64748B;
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .dossier-meta-val {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0F172A;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #F6F7F9;
    }
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2563EB;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. BACKEND CONNECTIONS & PIPELINE INITIALIZATION
# ==============================================================================
@st.cache_resource
def get_db_manager():
    return DatabaseManager.get_instance("sqlite:///data/eviguard.db")

@st.cache_resource
def get_pipeline():
    p = EviGuardPipeline("config.yaml")
    p.inference_stride = 1
    return p

db_manager = get_db_manager()
pipeline = get_pipeline()


# ==============================================================================
# 4. HIGH-SPEED THREADED OPENCV CAMERA WORKER
# ==============================================================================
class ThreadedCamera:
    """Zero-latency threaded hardware camera capture worker with DirectShow and native worker initialization."""

    def __init__(self, src: int = 0, width: int = 640, height: int = 480):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.is_connected = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        return self

    def _capture_worker(self):
        # On Windows, DirectShow is the standard reliable backend for webcams
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, None] if os.name == 'nt' else [cv2.CAP_ANY, None]
        indices_to_try = [self.src, 0, 1] if self.src not in (0, 1) else [self.src, 1 if self.src == 0 else 0]

        # Retry loop to connect if device was temporarily busy or releasing
        while self.running and not self.is_connected:
            for b in backends:
                for idx in indices_to_try:
                    if not self.running:
                        break
                    try:
                        cap = cv2.VideoCapture(idx, b) if b is not None else cv2.VideoCapture(idx)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            # DirectShow hardware camera requires a brief warm-up read
                            for _ in range(6):
                                ret, test_frame = cap.read()
                                if ret and test_frame is not None and test_frame.size > 0:
                                    if test_frame.shape[1] != self.width or test_frame.shape[0] != self.height:
                                        test_frame = cv2.resize(test_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                                    self.cap = cap
                                    self.is_connected = True
                                    with self.lock:
                                        self.frame = test_frame
                                    break
                                time.sleep(0.04)
                        if self.is_connected:
                            break
                        cap.release()
                    except Exception:
                        pass
                if self.is_connected:
                    break
            if not self.is_connected and self.running:
                time.sleep(0.25)

        # Steady grabbing loop
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                try:
                    grabbed = self.cap.grab()
                    if grabbed:
                        ret, frame = self.cap.retrieve()
                        if ret and frame is not None and frame.size > 0:
                            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                            with self.lock:
                                self.frame = frame
                    else:
                        time.sleep(0.01)
                except Exception:
                    time.sleep(0.01)
            else:
                time.sleep(0.02)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive() and self.thread != threading.current_thread():
            try:
                self.thread.join(timeout=0.3)
            except Exception:
                pass
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.is_connected = False


# ==============================================================================
# 4B. SINGLE-PIPELINE BACKGROUND MJPEG STREAMER (HIGH-FPS PERSISTENT VIEWPORT)
# ==============================================================================
class SinglePipelineStreamer:
    """
    Singleton background MJPEG streaming server on 127.0.0.1:8502:
    - Maintains single shared ThreadedCamera and EviGuardPipeline instance.
    - Dedicated background thread continuously pulls freshest frame, runs pipeline.process_frame,
      preserves all full HUD annotations (bounding boxes, red alert badges, face mesh, gaze vectors),
      and encodes JPEG once into shared_state for all clients.
    - Serves /stream for 30 FPS video and /telemetry for JSON telemetry readouts.
    - Zero st.image() bottleneck; native HTML <img src="http://127.0.0.1:8502/stream" /> renders at hardware 30+ FPS.
    """
    def __init__(self, pipeline: EviGuardPipeline, host: str = "127.0.0.1", port: int = 8502):
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self.camera: Optional[ThreadedCamera] = None
        self.camera_src: int = -1
        self.is_active = False
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.session_id: str = "DEFAULT_SESSION"
        self.candidate_name: str = "Candidate"

        # Shared atomic state
        self.shared_state: Dict[str, Any] = {
            "latest_output": None,
            "latest_annotated": None,
            "latest_jpeg": None,
            "frame_id": 0,
            "frame_seq": 0,
        }

        # Standby placeholder JPEG matching #090A0F dark viewport background
        standby_img = np.zeros((480, 640, 3), dtype=np.uint8)
        standby_img[:] = (9, 10, 15)
        cv2.putText(standby_img, "EviGuard AI - Vision Active", (160, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 225, 235), 2)
        cv2.putText(standby_img, "Connecting to camera sensor...", (180, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 135, 150), 1)
        _, standby_buf = cv2.imencode('.jpg', standby_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        self.standby_jpeg = standby_buf.tobytes()

        # Thread A: Dedicated AI inference worker
        self.ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
        self.ai_thread.start()

        # Thread B: Dedicated 30 FPS annotated frame stream encoder
        self.stream_thread = threading.Thread(target=self._stream_encoder_worker, daemon=True)
        self.stream_thread.start()

        # Start HTTP server
        self.server = None
        self._start_http_server()

    def _ai_worker(self):
        """Thread A: Dedicated AI inference worker - continuously runs pipeline without blocking video streamer."""
        self.pipeline.inference_stride = 1
        while True:
            try:
                if not self.is_active or self.camera is None or not self.camera.is_connected:
                    time.sleep(0.04)
                    continue

                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                output = self.pipeline.process_frame(
                    frame=frame,
                    session_id=self.session_id,
                    candidate_name=self.candidate_name
                )
                with self.lock:
                    self.shared_state["latest_output"] = output
                    self.shared_state["latest_annotated"] = output.annotated_frame
            except Exception:
                pass

            time.sleep(0.01)

    def _stream_encoder_worker(self):
        """Thread B: Dedicated 30 FPS Stream Encoder Worker - runs at steady ~30 FPS, never waits on AI inference."""
        while True:
            try:
                if not self.is_active or self.camera is None or not self.camera.is_connected:
                    time.sleep(0.04)
                    continue

                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                with self.lock:
                    output = self.shared_state.get("latest_output")
                    latest_annotated = self.shared_state.get("latest_annotated")

                display_frame = None
                if output is not None and getattr(output, "annotated_frame", None) is not None:
                    try:
                        person_count = int(output.pose_gaze.face_count if output.pose_gaze.face_detected else len(output.detections))
                        display_frame = self.pipeline._render_hud(
                            frame=frame.copy(),
                            detections=output.detections,
                            pose_gaze=output.pose_gaze,
                            risk=output.risk,
                            person_count=person_count
                        )
                    except Exception:
                        display_frame = latest_annotated

                if display_frame is None:
                    display_frame = latest_annotated if latest_annotated is not None else frame

                ret, buf = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret and buf is not None:
                    jpeg_bytes = buf.tobytes()
                    with self.lock:
                        self.shared_state["latest_jpeg"] = jpeg_bytes
                        self.shared_state["frame_id"] += 1
                        self.shared_state["frame_seq"] = self.shared_state["frame_id"]
                        self.cond.notify_all()
            except Exception:
                pass

            time.sleep(0.03) # Steady ~30 FPS pacing

    def _start_http_server(self):
        streamer_ref = self

        class StreamHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ('/stream', '/video_feed'):
                    try:
                        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except Exception:
                        pass

                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

                    last_seq = -1
                    try:
                        while True:
                            with streamer_ref.lock:
                                current_seq = streamer_ref.shared_state["frame_seq"]
                                if current_seq == last_seq:
                                    streamer_ref.cond.wait(timeout=0.035)
                                    current_seq = streamer_ref.shared_state["frame_seq"]

                                if not streamer_ref.is_active or streamer_ref.camera is None or not streamer_ref.camera.is_connected:
                                    jpg = streamer_ref.standby_jpeg
                                    last_seq = -1
                                else:
                                    jpg = streamer_ref.shared_state.get("latest_jpeg")
                                    if jpg is None:
                                        jpg = streamer_ref.standby_jpeg
                                    else:
                                        last_seq = current_seq

                            part = (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(jpg)).encode('ascii') + b"\r\n\r\n"
                                + jpg +
                                b"\r\n"
                            )
                            self.wfile.write(part)
                            time.sleep(0.008)
                    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                        pass
                    except Exception:
                        pass

                elif self.path in ('/telemetry', '/api/telemetry'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    try:
                        out = streamer_ref.get_latest_output()
                        if out is not None:
                            risk_score = float(out.risk.smoothed_score)
                            risk_level = str(out.risk.risk_level)
                            active_violations = list(out.risk.active_violations)
                            person_count = int(out.pose_gaze.face_count if out.pose_gaze.face_detected else len(out.detections))
                            yaw_val = float(out.pose_gaze.yaw)
                            pitch_val = float(out.pose_gaze.pitch)
                            gaze_status = str(out.pose_gaze.gaze_direction) if out.pose_gaze.face_detected else "Candidate Absent"
                            is_flagged = bool(out.risk.is_incident_triggered or len(active_violations) > 0 or risk_score >= 70.0)
                            data = {
                                "active": True,
                                "risk_score": risk_score,
                                "risk_level": risk_level,
                                "active_violations": active_violations,
                                "person_count": person_count,
                                "yaw": round(yaw_val, 1),
                                "pitch": round(pitch_val, 1),
                                "gaze_status": gaze_status,
                                "is_flagged": is_flagged
                            }
                        else:
                            data = {
                                "active": streamer_ref.is_active,
                                "risk_score": 0.0,
                                "risk_level": "LOW",
                                "active_violations": [],
                                "person_count": 1,
                                "yaw": 0.0,
                                "pitch": 0.0,
                                "gaze_status": "Direct (Screen)",
                                "is_flagged": False
                            }
                        self.wfile.write(json.dumps(data).encode('utf-8'))
                    except Exception:
                        self.wfile.write(b'{"active":false}')
                else:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b"EviGuard Single-Pipeline Streamer Active")

            def log_message(self, format, *args):
                return

        class ReusableServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        for attempt in range(5):
            try:
                self.server = ReusableServer((self.host, self.port), StreamHandler)
                server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                server_thread.start()
                break
            except Exception:
                time.sleep(0.5)

    def set_active(self, active: bool, cam_src: int = 0, session_id: str = "DEFAULT_SESSION", candidate_name: str = "Candidate"):
        with self.lock:
            self.session_id = session_id
            self.candidate_name = candidate_name
            self.is_active = active
            if active:
                if self.camera is None or self.camera_src != cam_src or not self.camera.running or not self.camera.is_connected:
                    if self.camera is not None:
                        self.camera.stop()
                    self.camera_src = cam_src
                    self.camera = ThreadedCamera(src=cam_src, width=640, height=480).start()
            else:
                if self.camera is not None:
                    self.camera.stop()
                    self.camera = None
                    self.camera_src = -1
                self.shared_state["latest_output"] = None
                self.shared_state["latest_annotated"] = None
                self.shared_state["latest_jpeg"] = None
                self.shared_state["frame_id"] = 0
                self.shared_state["frame_seq"] = 0

    def get_latest_output(self) -> Optional[PipelineOutput]:
        with self.lock:
            return self.shared_state.get("latest_output")


@st.cache_resource(show_spinner=False)
def get_streamer(_pipeline: EviGuardPipeline) -> SinglePipelineStreamer:
    return SinglePipelineStreamer(_pipeline, host="127.0.0.1", port=8502)

streamer = get_streamer(pipeline)


# ==============================================================================
# 5. GLOWING CIRCULAR THREAT ARC DIAL
# ==============================================================================
def get_threat_meter_html(risk_score: float, risk_level: str) -> str:
    """Renders a crisp editorial SVG circular threat dial with subtle ambient backlight and sharp typography."""
    if risk_score >= 70.0 or risk_level == "CRITICAL":
        stroke_gradient_start = "#DC2626"
        stroke_gradient_end = "#EF4444"
        glow_color = "rgba(220, 38, 38, 0.35)"
        glow_subtle = "rgba(220, 38, 38, 0.08)"
        bg_color = "#FEF2F2"
        border_color = "#FECACA"
        badge_text = "CRITICAL THREAT"
        badge_color = "#B91C1C"
        sub_text = "Violation Active"
        display_val = f"{risk_score:.0f}"
        pct = min(100.0, max(0.0, risk_score))
        dash_offset = 235.6 - (pct / 100.0) * 235.6
    elif risk_score >= 30.0 or risk_level in ("SUSPICIOUS", "MEDIUM", "MONITOR"):
        stroke_gradient_start = "#D97706"
        stroke_gradient_end = "#F59E0B"
        glow_color = "rgba(217, 119, 6, 0.30)"
        glow_subtle = "rgba(217, 119, 6, 0.08)"
        bg_color = "#FFFBEB"
        border_color = "#FDE68A"
        badge_text = "ELEVATED RISK"
        badge_color = "#B45309"
        sub_text = "Sensor Deviation"
        display_val = f"{risk_score:.0f}"
        pct = min(100.0, max(0.0, risk_score))
        dash_offset = 235.6 - (pct / 100.0) * 235.6
    else:
        stroke_gradient_start = "#047857"
        stroke_gradient_end = "#10B981"
        glow_color = "rgba(4, 120, 87, 0.25)"
        glow_subtle = "rgba(4, 120, 87, 0.08)"
        bg_color = "#ECFDF5"
        border_color = "#A7F3D0"
        badge_text = "100% OPTIMAL INTEGRITY"
        badge_color = "#047857"
        sub_text = "Compliant Session"
        integrity_pct = max(0.0, min(100.0, 100.0 - risk_score))
        display_val = "100%" if risk_score <= 2.0 else f"{integrity_pct:.0f}%"
        dash_offset = 0.0

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 6px 0 12px 0;">
        <div style="position: relative; width: 210px; height: 125px; display: flex; justify-content: center; align-items: flex-end;">
            <!-- Ambient Radial Glow Behind Gauge -->
            <div style="position: absolute; width: 160px; height: 160px; top: -20px; border-radius: 50%; background: radial-gradient(circle, {glow_subtle} 0%, transparent 70%); filter: blur(12px); pointer-events: none;"></div>
            <svg width="210" height="210" viewBox="0 0 220 220" style="position: absolute; top: -45px; transform: rotate(180deg); z-index: 2;">
                <defs>
                    <linearGradient id="meterGrad_{display_val.replace('%', '')}" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="{stroke_gradient_start}" />
                        <stop offset="100%" stop-color="{stroke_gradient_end}" />
                    </linearGradient>
                </defs>
                <!-- Crisp Editorial Track -->
                <circle cx="110" cy="110" r="75" fill="none" stroke="#E2E8F0" stroke-width="14" stroke-dasharray="235.6 235.6" stroke-dashoffset="0" />
                <!-- Active Luminous Progress Arc -->
                <circle cx="110" cy="110" r="75" fill="none" stroke="url(#meterGrad_{display_val.replace('%', '')})" stroke-width="14" 
                    stroke-dasharray="235.6 235.6" stroke-dashoffset="{dash_offset}" 
                    stroke-linecap="round" 
                    style="transition: stroke-dashoffset 0.25s ease; filter: drop-shadow(0 2px 8px {glow_color});" />
            </svg>
            <div style="text-align: center; z-index: 5; margin-bottom: 2px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 34px; font-weight: 700; color: #0F172A; line-height: 1; letter-spacing: -0.02em;">{display_val}</div>
                <div style="display: inline-block; background: {bg_color}; border: 1px solid {border_color}; border-radius: 9999px; padding: 3px 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; color: {badge_color}; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 6px;">{badge_text}</div>
                <div style="font-family: 'Plus Jakarta Sans', 'Inter', sans-serif; font-size: 0.85rem; font-weight: 500; color: #475569; margin-top: 3px;">{sub_text}</div>
            </div>
        </div>
    </div>
    """

# ==============================================================================
# 6. SAAS TOP APPLICATION HEADER & SESSION MANAGEMENT
# ==============================================================================
all_sessions = db_manager.get_all_sessions()
session_ids = [s["session_id"] for s in all_sessions]

if not session_ids:
    init_session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db_manager.create_session(init_session_id, "CAND-001", "Live Candidate", "Real-Time AI Proctoring Assessment")
    st.session_state.active_session_id = init_session_id
    all_sessions = db_manager.get_all_sessions()
    session_ids = [s["session_id"] for s in all_sessions]

if st.session_state.active_session_id not in session_ids:
    st.session_state.active_session_id = session_ids[0] if session_ids else None

current_session = db_manager.get_session_by_id(st.session_state.active_session_id) if st.session_state.active_session_id else None

session_id = current_session.get("session_id") if current_session else "SESSION_LIVE_DEFAULT"
candidate_name = current_session.get("candidate_name") if current_session else "Live Candidate"
candidate_id = current_session.get("candidate_id") if current_session else "CAND-001"
exam_title = current_session.get("exam_title") if current_session else "Real-Time AI Proctoring Assessment"
incidents = db_manager.get_session_incidents(session_id) if current_session else []
metrics = db_manager.get_session_metrics(session_id, limit=500) if current_session else []

# Sync dynamic state model into st.session_state
st.session_state.incidents = incidents
st.session_state.total_flags = len(incidents)
st.session_state.confirmed_count = sum(1 for i in incidents if i.get("proctor_verdict") == "CONFIRMED")
st.session_state.pending_count = sum(1 for i in incidents if i.get("proctor_verdict") == "PENDING")

metric_risk_scores = [float(m.get("risk_score", 0.0)) for m in metrics] if metrics else []
session_peak_risk = float(current_session.get("peak_risk_score", 0.0)) if current_session else 0.0
st.session_state.peak_risk = max(metric_risk_scores + [session_peak_risk, 0.0])

# Dynamic Overall Integrity Index Calculation
if current_session:
    base_integrity = 100.0
    confirmed_penalties = st.session_state.confirmed_count * 10.0
    pending_penalties = sum(3.0 if i.get("severity") in ("CRITICAL", "HIGH") else 1.0 for i in incidents if i.get("proctor_verdict") == "PENDING")
    calculated_integrity = max(0.0, min(100.0, base_integrity - confirmed_penalties - pending_penalties))
    st.session_state.integrity_score = calculated_integrity
else:
    st.session_state.integrity_score = 100.0

# Top SaaS Navigation Bar with System Reset Action
st.markdown(f"""
<div class="saas-top-navbar">
    <div class="saas-brand-title">
        <span class="saas-logo-icon">🛡️</span>
        <div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="saas-brand-text">EviGuard AI</span>
                <span class="saas-brand-badge">SMART EXAM PROCTOR</span>
            </div>
            <div style="font-size: 11px; color: #475569; font-weight: 500;">Real-Time AI Cheat Detection & Integrity Monitoring</div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 12px;">
        <div class="saas-candidate-chip">
            <span class="saas-chip-label">STUDENT:</span>
            <span class="saas-chip-name">{candidate_name}</span>
            <span class="saas-chip-id">({candidate_id})</span>
        </div>
        <div class="badge-luxury-ready">
            AI MONITORING ACTIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 7. FULL-WIDTH MODULAR TABBED SaaS ARCHITECTURE
# ==============================================================================
tab_overview, tab_vision, tab_telemetry, tab_audit = st.tabs([
    "👤 1. Student & Exam Info",
    "📹 2. Live Proctoring Feed",
    "📊 3. Threat Analytics & Charts",
    "📑 4. Audit Logs & Reports"
])


# ------------------------------------------------------------------------------
# MODULE 1: EXECUTIVE OVERVIEW & REGISTRATION
# ------------------------------------------------------------------------------
with tab_overview:
    confirmed_count = st.session_state.confirmed_count
    total_flags = st.session_state.total_flags
    integrity_score = st.session_state.integrity_score
    peak_risk = st.session_state.peak_risk
    score_color = "#047857" if integrity_score >= 80 else ("#D97706" if integrity_score >= 50 else "#DC2626")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="kpi-card-header">
                <span class="kpi-icon-badge">🛡️</span>
                <div class="eyebrow-label">Academic Integrity</div>
            </div>
            <div class="kpi-value-text" style="color: {score_color}; ">{integrity_score:.1f}%</div>
            <div class="kpi-footer-status">
                <span class="kpi-status-dot" style="background: {score_color}; box-shadow: 0 0 8px {score_color};"></span>
                <span style="color: #475569; font-size: 11.5px; font-weight: 600;">Status:</span>
                <b style="color: {score_color}; font-size: 11.5px;">{'CLEAN (No Cheating)' if integrity_score >= 80 else 'FLAGGED FOR REVIEW'}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="kpi-card-header">
                <span class="kpi-icon-badge">🚨</span>
                <div class="eyebrow-label">Suspicious Alerts</div>
            </div>
            <div class="kpi-value-text" style="color: {'#047857' if total_flags == 0 else '#DC2626'}; ">{total_flags}</div>
            <div class="kpi-footer-status">
                <span style="color: #475569; font-size: 11.5px; font-weight: 600;">{confirmed_count} Confirmed Violations</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="kpi-card-header">
                <span class="kpi-icon-badge">📈</span>
                <div class="eyebrow-label">Highest Threat Score</div>
            </div>
            <div class="kpi-value-text" style="color: {'#047857' if peak_risk < 30 else ('#D97706' if peak_risk < 70 else '#DC2626')}; ">{peak_risk:.1f}<span style="font-size: 16px; color: #475569;">/100</span></div>
            <div class="kpi-footer-status">
                <span style="color: #475569; font-size: 11.5px; font-weight: 600;">Peak Anomaly Score</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="kpi-card-header">
                <span class="kpi-icon-badge">⚖️</span>
                <div class="eyebrow-label">Verified Malpractice</div>
            </div>
            <div class="kpi-value-text" style="color: {'#047857' if confirmed_count == 0 else '#DC2626'}; ">{confirmed_count}</div>
            <div class="kpi-footer-status">
                <span style="color: #475569; font-size: 11.5px; font-weight: 600;">Proctor Confirmed Cheating</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    col_dash1, col_dash2 = st.columns([1, 1], gap="medium")

    with col_dash1:
        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 class="section-header" style="margin: 0;">👤 Student Registration & Exam Setup</h3>
                <span class="pill-safe">● SETUP</span>
            </div>
            """, unsafe_allow_html=True)

            if not current_session:
                st.info("No active session initialized. Fill in the student details below to start.")

            with st.form("onboarding_form"):
                c_name_in = st.text_input("Candidate Full Name", placeholder="e.g. John Doe")
                c_id_in = st.text_input("Candidate / Student ID", placeholder="e.g. CAND-2026-001")
                c_exam_in = st.text_input("Assessment Title", placeholder="e.g. Midterm Machine Learning Exam")

                launch_btn = st.form_submit_button("🚀 Start Proctoring Session", type="primary", width="stretch")
                if launch_btn:
                    cand_name = c_name_in.strip() if c_name_in.strip() else "Live Candidate"
                    cand_id = c_id_in.strip() if c_id_in.strip() else f"ID-{datetime.now().strftime('%H%M%S')}"
                    exam_name = c_exam_in.strip() if c_exam_in.strip() else "Real-World AI Proctoring Assessment"
                    new_session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    db_manager.create_session(new_session_id, cand_id, cand_name, exam_name)
                    st.session_state.active_session_id = new_session_id
                    st.success("Assessment session activated successfully!")
                    st.rerun()

            if session_ids:
                st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                st.markdown("<span class='eyebrow-label'>Switch Active Session</span>", unsafe_allow_html=True)
                switch_selected = st.selectbox(
                    "Session Select Box",
                    session_ids,
                    index=session_ids.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_ids else 0,
                    label_visibility="collapsed",
                    key="switch_existing_session_select"
                )
                if switch_selected != st.session_state.active_session_id:
                    st.session_state.active_session_id = switch_selected
                    st.rerun()

    with col_dash2:
        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 class="section-header" style="margin: 0;">📋 Exam Rules & Integrity Guidelines</h3>
                <span class="pill-safe">● COMPLIANCE POLICY</span>
            </div>
            <div class="exam-rules-container">
                <div class="exam-rule-card">
                    <div class="exam-rule-icon" style="background: #FEF2F2; border-color: #FECACA; color: #DC2626;">
                        📱
                    </div>
                    <div class="exam-rule-body">
                        <div class="exam-rule-header">
                            <span class="exam-rule-title">No Unauthorized Devices & Study Materials</span>
                            <span class="rule-badge rule-badge-danger">PROHIBITED</span>
                        </div>
                        <p class="exam-rule-desc">
                            Mobile phones, smartwatches, physical cheat sheets, notebooks, books, and secondary displays are strictly prohibited on your desk.
                        </p>
                    </div>
                </div>
                <div class="exam-rule-card">
                    <div class="exam-rule-icon" style="background: #FFFBEB; border-color: #FDE68A; color: #D97706;">
                        👀
                    </div>
                    <div class="exam-rule-body">
                        <div class="exam-rule-header">
                            <span class="exam-rule-title">Continuous Screen & Desk Focus</span>
                            <span class="rule-badge rule-badge-warning">GAZE MONITORED</span>
                        </div>
                        <p class="exam-rule-desc">
                            Keep head and eyes facing the assessment screen. Repeatedly looking sideways, turning around, or ducking triggers immediate suspicion.
                        </p>
                    </div>
                </div>
                <div class="exam-rule-card">
                    <div class="exam-rule-icon" style="background: #EFF6FF; border-color: #BFDBFE; color: #2563EB;">
                        👤
                    </div>
                    <div class="exam-rule-body">
                        <div class="exam-rule-header">
                            <span class="exam-rule-title">Single Candidate Environment</span>
                            <span class="rule-badge rule-badge-info">SOLO TEST-TAKER</span>
                        </div>
                        <p class="exam-rule-desc">
                            Only the registered student is permitted in the camera view. Multiple faces, passing bystanders, or outside voices cause instant flags.
                        </p>
                    </div>
                </div>
                <div class="exam-rule-card" style="margin-bottom: 0;">
                    <div class="exam-rule-icon" style="background: #ECFDF5; border-color: #A7F3D0; color: #047857;">
                        🛡️
                    </div>
                    <div class="exam-rule-body">
                        <div class="exam-rule-header">
                            <span class="exam-rule-title">Autonomous Forensic Evidence Logging</span>
                            <span class="rule-badge rule-badge-success">AUTO-RECORDED</span>
                        </div>
                        <p class="exam-rule-desc">
                            Any detected violation automatically triggers an encrypted photo snapshot with timestamped telemetry saved to the proctor audit dossier.
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MODULE 2: REAL-TIME VISION & AI PROCTORING
# ------------------------------------------------------------------------------
with tab_vision:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
        <div>
            <h3 class="page-title" style="margin: 0;">📹 High-Definition Real-Time Vision Feed</h3>
            <p class="body-text" style="margin-top: 4px; margin-bottom: 0;">Zero-latency hardware vision stream with neural HUD overlays, 3D head pose vectors, and violation alerts.</p>
        </div>
        <span class="pill-safe">● ZERO-LATENCY • 60 FPS • NATIVE</span>
    </div>
    """, unsafe_allow_html=True)

    vision_banner_holder = st.empty()

    col_v_left, col_v_right = st.columns([14, 6], gap="medium")

    with col_v_left:
        with st.container(border=True):
            st.markdown("""
            <div class="video-hud-topbar">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="hud-rec-indicator">● LIVE SENSOR FEED</span>
                    <span class="hud-spec-tag">• 30 FPS • REAL-TIME AI</span>
                </div>
                <div class="hud-encryption-tag">🔒 LIVE PROCTOR FEED</div>
            </div>
            """, unsafe_allow_html=True)
            t_c1, t_c2 = st.columns([3, 2])
            start_stream = t_c1.toggle("▶ Enable Live AI Vision Stream", value=True, key="vision_tab_stream_toggle")
            cam_src = t_c2.selectbox("Select Camera Device Index", [0, 1, 2], index=0, key="vision_cam_src_select")
            video_placeholder = st.empty()

    with col_v_right:
        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 class="section-header" style="margin: 0;">⚡ Live Sensor Telemetry</h3>
                <span class="eyebrow-label" style="color: #2563EB;">LIVE READOUTS</span>
            </div>
            """, unsafe_allow_html=True)
            telemetry_placeholder = st.empty()

    if start_stream:
        streamer.set_active(True, cam_src=cam_src, session_id=session_id, candidate_name=candidate_name)

        vision_banner_holder.markdown("""
        <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 14px 20px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px; box-shadow: 0 4px 16px rgba(4, 120, 87, 0.08);">
            <span style="font-size: 1.3rem;">✅</span>
            <span style="color: #047857; font-weight: 700; font-size: 14px; font-family: 'Inter', sans-serif;">LIVE SENSOR ACTIVE: Multi-modal AI neural proctoring pipeline engaged</span>
        </div>
        """, unsafe_allow_html=True)

        with video_placeholder:
            components.html(
                """
                <div style="width: 100%; border-radius: 8px; overflow: hidden; background: #090A0F;">
                    <img src="http://127.0.0.1:8502/stream" style="width:100%; height:520px; object-fit:contain; border-radius:8px; display:block; background:#090A0F;" />
                </div>
                """,
                height=540,
                scrolling=False
            )

        with telemetry_placeholder:
            components.html(
                """
                <!DOCTYPE html>
                <html>
                <head>
                <meta charset="utf-8">
                <style>
                    body {
                        margin: 0;
                        padding: 0;
                        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                        background: transparent;
                        overflow: hidden;
                    }
                    .dial-box {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        padding: 2px 0 8px 0;
                    }
                    .dial-card {
                        width: 100%;
                        background: #ECFDF5;
                        border: 1px solid #A7F3D0;
                        border-radius: 12px;
                        padding: 10px 14px;
                        box-sizing: border-box;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        box-shadow: 0 4px 14px rgba(4, 120, 87, 0.08);
                        transition: all 0.3s ease;
                    }
                    .dial-val {
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 1.85rem;
                        font-weight: 800;
                        color: #0F172A;
                        line-height: 1.1;
                    }
                    .dial-badge {
                        font-size: 0.72rem;
                        font-weight: 700;
                        padding: 3px 8px;
                        border-radius: 9999px;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        margin-top: 4px;
                    }
                    .dial-sub {
                        font-size: 0.75rem;
                        color: #64748B;
                        margin-top: 2px;
                        font-weight: 500;
                    }
                    .telemetry-row {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 9px 12px;
                        background: #F8FAFC;
                        border: 1px solid #E2E8F0;
                        border-radius: 8px;
                        margin-bottom: 7px;
                        font-size: 13.5px;
                    }
                    .telemetry-label {
                        color: #475569;
                        font-weight: 600;
                    }
                    .telemetry-val {
                        font-family: 'JetBrains Mono', monospace;
                        font-weight: 700;
                    }
                    .alert-banner {
                        border-radius: 10px;
                        padding: 8px 12px;
                        margin-bottom: 10px;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        font-size: 12.5px;
                        font-weight: 700;
                        box-sizing: border-box;
                        transition: all 0.3s ease;
                    }
                </style>
                </head>
                <body>
                    <div id="banner" class="alert-banner" style="background:#ECFDF5; border:1px solid #A7F3D0; color:#047857;">
                        <span>✅</span>
                        <span id="banner-text">COMPLIANCE VERIFIED</span>
                    </div>

                    <div class="dial-box">
                        <div id="dial-card" class="dial-card">
                            <svg width="150" height="85" viewBox="0 0 190 115" style="overflow: visible;">
                                <path d="M 25 105 A 70 70 0 0 1 165 105" fill="none" stroke="#E2E8F0" stroke-width="12" stroke-linecap="round" />
                                <path id="dial-arc" d="M 25 105 A 70 70 0 0 1 165 105" fill="none" stroke="#10B981" stroke-width="12" stroke-linecap="round" stroke-dasharray="235.6" stroke-dashoffset="0" style="transition: stroke-dashoffset 0.35s ease, stroke 0.35s ease;" />
                            </svg>
                            <div id="dial-num" class="dial-val" style="margin-top: -30px;">100%</div>
                            <div id="dial-badge" class="dial-badge" style="background:#D1FAE5; color:#047857;">100% OPTIMAL INTEGRITY</div>
                            <div id="dial-sub" class="dial-sub">Compliant Session</div>
                        </div>
                    </div>

                    <div style="margin-top: 4px;">
                        <div class="telemetry-row">
                            <span class="telemetry-label">👥 Candidate Count</span>
                            <span id="tel-count" class="telemetry-val" style="color:#047857;">1 Detected</span>
                        </div>
                        <div class="telemetry-row">
                            <span class="telemetry-label">🔄 Head Yaw (L/R)</span>
                            <span id="tel-yaw" class="telemetry-val" style="color:#0F172A;">+0.0°</span>
                        </div>
                        <div class="telemetry-row">
                            <span class="telemetry-label">📐 Head Pitch (U/D)</span>
                            <span id="tel-pitch" class="telemetry-val" style="color:#0F172A;">+0.0°</span>
                        </div>
                        <div class="telemetry-row">
                            <span class="telemetry-label">👀 Gaze Tracker</span>
                            <span id="tel-gaze" class="telemetry-val" style="color:#047857;">Direct (Screen)</span>
                        </div>
                    </div>

                    <script>
                    function updateTelemetry() {
                        fetch('http://127.0.0.1:8502/telemetry')
                            .then(r => r.json())
                            .then(data => {
                                if (!data) return;
                                const score = Number(data.risk_score || 0);
                                const level = String(data.risk_level || 'LOW');
                                const violations = data.active_violations || [];
                                const count = data.person_count !== undefined ? data.person_count : 1;
                                const yaw = Number(data.yaw || 0);
                                const pitch = Number(data.pitch || 0);
                                const gaze = String(data.gaze_status || 'Direct (Screen)');
                                const flagged = Boolean(data.is_flagged || violations.length > 0 || score >= 70);

                                const banner = document.getElementById('banner');
                                const bannerText = document.getElementById('banner-text');
                                if (flagged || violations.length > 0) {
                                    banner.style.background = '#FEF2F2';
                                    banner.style.borderColor = '#FECACA';
                                    banner.style.color = '#B91C1C';
                                    banner.firstElementChild.innerText = '🚨';
                                    const hasUnauth = violations.some(v => /unauth|phone|book|notes|chit|paper/i.test(v));
                                    if (hasUnauth) {
                                        bannerText.innerText = 'ALERT: UNAUTHORIZED MATERIAL';
                                    } else {
                                        const cleanV = violations.map(v => v.replace(/^(VIOLATION|ALERT):\\s*/i, ''));
                                        bannerText.innerText = 'ALERT: ' + (cleanV.join(' • ') || 'ELEVATED RISK');
                                    }
                                } else {
                                    banner.style.background = '#ECFDF5';
                                    banner.style.borderColor = '#A7F3D0';
                                    banner.style.color = '#047857';
                                    banner.firstElementChild.innerText = '✅';
                                    bannerText.innerText = 'COMPLIANCE VERIFIED: In Tolerance';
                                }

                                const card = document.getElementById('dial-card');
                                const arc = document.getElementById('dial-arc');
                                const num = document.getElementById('dial-num');
                                const badge = document.getElementById('dial-badge');
                                const sub = document.getElementById('dial-sub');

                                if (score >= 70 || level === 'CRITICAL') {
                                    card.style.background = '#FEF2F2';
                                    card.style.borderColor = '#FECACA';
                                    arc.setAttribute('stroke', '#EF4444');
                                    const pct = Math.min(100, Math.max(0, score));
                                    const offset = 235.6 - (pct / 100.0) * 235.6;
                                    arc.setAttribute('stroke-dashoffset', offset);
                                    num.innerText = Math.round(score);
                                    badge.style.background = '#FEE2E2';
                                    badge.style.color = '#B91C1C';
                                    badge.innerText = 'CRITICAL THREAT';
                                    sub.innerText = 'Violation Active';
                                } else if (score >= 30 || level === 'SUSPICIOUS' || level === 'MEDIUM') {
                                    card.style.background = '#FFFBEB';
                                    card.style.borderColor = '#FDE68A';
                                    arc.setAttribute('stroke', '#F59E0B');
                                    const pct = Math.min(100, Math.max(0, score));
                                    const offset = 235.6 - (pct / 100.0) * 235.6;
                                    arc.setAttribute('stroke-dashoffset', offset);
                                    num.innerText = Math.round(score);
                                    badge.style.background = '#FEF3C7';
                                    badge.style.color = '#B45309';
                                    badge.innerText = 'ELEVATED RISK';
                                    sub.innerText = 'Sensor Deviation';
                                } else {
                                    card.style.background = '#ECFDF5';
                                    card.style.borderColor = '#A7F3D0';
                                    arc.setAttribute('stroke', '#10B981');
                                    arc.setAttribute('stroke-dashoffset', '0');
                                    num.innerText = score <= 2 ? '100%' : Math.round(100 - score) + '%';
                                    badge.style.background = '#D1FAE5';
                                    badge.style.color = '#047857';
                                    badge.innerText = '100% OPTIMAL INTEGRITY';
                                    sub.innerText = 'Compliant Session';
                                }

                                const elCount = document.getElementById('tel-count');
                                elCount.innerText = count + ' Detected';
                                elCount.style.color = (count === 1) ? '#047857' : '#DC2626';

                                const elYaw = document.getElementById('tel-yaw');
                                elYaw.innerText = (yaw >= 0 ? '+' : '') + yaw.toFixed(1) + '°';

                                const elPitch = document.getElementById('tel-pitch');
                                elPitch.innerText = (pitch >= 0 ? '+' : '') + pitch.toFixed(1) + '°';

                                const elGaze = document.getElementById('tel-gaze');
                                elGaze.innerText = gaze;
                                elGaze.style.color = (/look|away|absent/i.test(gaze)) ? '#DC2626' : '#047857';
                            })
                            .catch(e => {});
                    }
                    setInterval(updateTelemetry, 250);
                    updateTelemetry();
                    </script>
                </body>
                </html>
                """,
                height=480,
                scrolling=False
            )
    else:
        streamer.set_active(False)
        vision_banner_holder.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 20px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);">
            <span style="font-size: 1.3rem;">🛡️</span>
            <span style="color: #2563EB; font-weight: 700; font-size: 0.92rem; font-family: 'Plus Jakarta Sans', sans-serif;">VISION SENSOR ON STANDBY: Ready for live proctoring stream</span>
        </div>
        """, unsafe_allow_html=True)

        with video_placeholder:
            st.markdown("""
            <div style="background: #FFFFFF; border: 2px dashed #CBD5E1; border-radius: 12px; height: 500px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #475569; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);">
                <span style="font-size: 3rem; margin-bottom: 10px; filter: drop-shadow(0 2px 8px rgba(37, 99, 235, 0.25));">📷</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 6px; font-family: 'Plus Jakarta Sans', sans-serif;">Camera Feed on Standby</div>
                <div style="font-size: 0.9rem; color: #475569; font-family: 'Plus Jakarta Sans', sans-serif;">Toggle switch above to activate real-time AI vision proctoring.</div>
            </div>
            """, unsafe_allow_html=True)

        with telemetry_placeholder:
            st.markdown(get_threat_meter_html(0.0, "LOW"), unsafe_allow_html=True)
            st.markdown("""
            <div style="margin-top: 4px;">
                <div class="enterprise-telemetry-row">
                    <span class="body-text" style="color: #475569; font-size: 13.5px; font-weight: 600;">👥 Candidate Count</span>
                    <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #047857;">1 Detected</span>
                </div>
                <div class="enterprise-telemetry-row">
                    <span class="body-text" style="color: #475569; font-size: 13.5px; font-weight: 600;">🔄 Head Yaw (L/R)</span>
                    <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #0F172A;">+0.0°</span>
                </div>
                <div class="enterprise-telemetry-row">
                    <span class="body-text" style="color: #475569; font-size: 13.5px; font-weight: 600;">📐 Head Pitch (U/D)</span>
                    <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #0F172A;">+0.0°</span>
                </div>
                <div class="enterprise-telemetry-row">
                    <span class="body-text" style="color: #475569; font-size: 13.5px; font-weight: 600;">👀 Gaze Tracker</span>
                    <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #047857;">Direct (Screen)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MODULE 3: THREAT TELEMETRY & BEHAVIORAL ANALYTICS
# ------------------------------------------------------------------------------
with tab_telemetry:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <h3 class="page-title" style="margin: 0;">📊 Exam Threat Analytics & Behavioral Charts</h3>
        <p class="body-text" style="margin-top: 2px; margin-bottom: 0;">Multi-dimensional integrity metrics: live threat timeline, session risk exposure distribution, and violation breakdown.</p>
    </div>
    """, unsafe_allow_html=True)

    # Compute high-level telemetry stats
    total_frames = len(metrics) if metrics else 0
    clean_frames = len([m for m in metrics if m.get("risk_score", 0) < 30]) if metrics else 0
    clean_pct = round((clean_frames / total_frames * 100), 1) if total_frames > 0 else 100.0
    warn_frames = len([m for m in metrics if 30 <= m.get("risk_score", 0) < 70]) if metrics else 0
    crit_frames = len([m for m in metrics if m.get("risk_score", 0) >= 70]) if metrics else 0

    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 10px;">
        <div class="telemetry-stat-card">
            <span class="telemetry-stat-label">TOTAL FRAMES MONITORED</span>
            <div class="telemetry-stat-value" style="color: #2563EB;">{total_frames} <span style="font-size: 12px; color: #475569;">frames</span></div>
        </div>
        <div class="telemetry-stat-card">
            <span class="telemetry-stat-label">CLEAN EXAM DURATION</span>
            <div class="telemetry-stat-value" style="color: #047857;">{clean_pct:.1f}% <span style="font-size: 12px; color: #475569;">normal (&lt;30)</span></div>
        </div>
        <div class="telemetry-stat-card">
            <span class="telemetry-stat-label">SUSPICIOUS WARNINGS</span>
            <div class="telemetry-stat-value" style="color: #D97706;">{warn_frames} <span style="font-size: 12px; color: #475569;">frames (30-69)</span></div>
        </div>
        <div class="telemetry-stat-card">
            <span class="telemetry-stat-label">CRITICAL CHEAT FRAMES</span>
            <div class="telemetry-stat-value" style="color: {'#047857' if crit_frames == 0 else '#DC2626'};">{crit_frames} <span style="font-size: 12px; color: #475569;">frames (≥70)</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2, col_t3 = st.columns([5, 3.5, 3.5], gap="small")

    with col_t1:
        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                <div>
                    <h3 class="section-header" style="margin: 0;">📈 Threat Score Timeline</h3>
                    <p style="margin: 2px 0 0 0; font-size: 11px; color: #475569;">Real-time risk curve (0-100). Red = Critical (70), Yellow = Warning (30).</p>
                </div>
                <span class="eyebrow-label">LIVE TIMELINE</span>
            </div>
            """, unsafe_allow_html=True)

            if metrics:
                df = pd.DataFrame(metrics)
                fig_line = px.line(
                    df,
                    x=df.index,
                    y="risk_score",
                    labels={"x": "Exam Frames", "risk_score": "Threat Level"},
                    title=None
                )
                fig_line.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Critical (70)", annotation_position="top left")
                fig_line.add_hline(y=30, line_dash="dot", line_color="#F59E0B", annotation_text="Warning (30)", annotation_position="top left")
                fig_line.update_traces(line_color="#2563EB", line_width=2.5)
                fig_line.update_layout(
                    height=230,
                    margin=dict(l=10, r=10, t=10, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#334155", family="Plus Jakarta Sans"),
                    xaxis=dict(gridcolor="#E2E8F0"),
                    yaxis=dict(gridcolor="#E2E8F0", range=[0, 100])
                )
                st.plotly_chart(fig_line, key="telemetry_timeline_chart", width="stretch")
            else:
                st.info("No continuous frame telemetry recorded yet for this session.")

    with col_t2:
        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                <div>
                    <h3 class="section-header" style="margin: 0;">🚦 Risk Level Exposure</h3>
                    <p style="margin: 2px 0 0 0; font-size: 11px; color: #475569;">% of exam time in Normal vs Warning vs Critical.</p>
                </div>
                <span class="eyebrow-label">DISTRIBUTION</span>
            </div>
            """, unsafe_allow_html=True)

            if metrics:
                df = pd.DataFrame(metrics)
                tot_f = len(df)
                norm_f = len(df[df["risk_score"] < 30])
                warn_f = len(df[(df["risk_score"] >= 30) & (df["risk_score"] < 70)])
                crit_f = len(df[df["risk_score"] >= 70])

                n_pct = round((norm_f / tot_f) * 100, 1) if tot_f > 0 else 0
                w_pct = round((warn_f / tot_f) * 100, 1) if tot_f > 0 else 0
                c_pct = round((crit_f / tot_f) * 100, 1) if tot_f > 0 else 0

                risk_summary_df = pd.DataFrame({
                    "Category": ["Normal (<30)", "Warning (30-69)", "Critical (≥70)"],
                    "Percentage": [n_pct, w_pct, c_pct],
                    "Frames": [norm_f, warn_f, crit_f],
                })

                fig_risk = px.bar(
                    risk_summary_df,
                    x="Percentage",
                    y="Category",
                    orientation="h",
                    text=risk_summary_df["Percentage"].apply(lambda p: f"{p}%"),
                    color="Category",
                    color_discrete_map={
                        "Normal (<30)": "#047857",
                        "Warning (30-69)": "#D97706",
                        "Critical (≥70)": "#DC2626"
                    }
                )
                fig_risk.update_layout(
                    height=230,
                    margin=dict(l=10, r=20, t=10, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    font=dict(color="#334155", family="Plus Jakarta Sans"),
                    xaxis=dict(gridcolor="#E2E8F0", range=[0, 105], ticksuffix="%"),
                    yaxis=dict(gridcolor="#E2E8F0", autorange="reversed")
                )
                fig_risk.update_traces(
                    textposition="outside",
                    textfont=dict(size=11, color="#0F172A", family="JetBrains Mono"),
                    marker=dict(line=dict(width=1, color="rgba(255,255,255,0.2)"))
                )
                st.plotly_chart(fig_risk, key="telemetry_risk_exposure_chart", width="stretch")
            else:
                st.info("No risk distribution data available yet.")

    with col_t3:
        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                <div>
                    <h3 class="section-header" style="margin: 0;">🍩 Violations Breakdown</h3>
                    <p style="margin: 2px 0 0 0; font-size: 11px; color: #475569;">Proportion of detected infractions by violation type.</p>
                </div>
                <span class="eyebrow-label">TAXONOMY</span>
            </div>
            """, unsafe_allow_html=True)

            if incidents:
                v_types = [i["violation_type"] for i in incidents]
                v_df = pd.Series(v_types).value_counts().reset_index()
                v_df.columns = ["Violation Type", "Count"]
                fig_pie = px.pie(v_df, values="Count", names="Violation Type", hole=0.55, color_discrete_sequence=["#2563EB", "#FF4D00", "#047857", "#D97706", "#7C3AED", "#06B6D4"])
                fig_pie.update_layout(
                    height=230,
                    margin=dict(l=10, r=10, t=10, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", 
                    font=dict(color="#0F172A", family="Plus Jakarta Sans")
                )
                st.plotly_chart(fig_pie, key="telemetry_pie_chart", width="stretch")
            else:
                st.info("No violations or anomalies recorded for this candidate.")


# ------------------------------------------------------------------------------
# MODULE 4: AUDIT DOSSIER & SYSTEM RESET
# ------------------------------------------------------------------------------
with tab_audit:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <h3 class="page-title" style="margin: 0;">📑 Incident History & Evidence Reports</h3>
        <p class="body-text" style="margin-top: 2px; margin-bottom: 0;">Review flagged cheating snapshots, download official PDF/CSV certificates, and manage exam data.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([13, 7], gap="medium")

    with col_a1:
        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 class="section-header" style="margin: 0;">📋 Flagged Incident Evidence Log</h3>
                <span class="eyebrow-label">INCIDENT TIMELINE</span>
            </div>
            """, unsafe_allow_html=True)

            if not incidents:
                st.markdown("""
                <div style="text-align: center; padding: 36px 20px; background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 14px; margin: 8px 0; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);">
                    <span style="font-size: 2.5rem; filter: drop-shadow(0 2px 6px rgba(4, 120, 87, 0.2));">🛡️</span>
                    <h4 style="color: #0F172A; font-weight: 700; font-size: 1.05rem; margin-top: 10px; margin-bottom: 4px; font-family: 'Plus Jakarta Sans', sans-serif;">Zero Cheating Incidents Recorded</h4>
                    <p style="font-size: 0.9rem; color: #475569; margin: 0; font-family: 'Plus Jakarta Sans', sans-serif;">Candidate has maintained full academic integrity throughout this session.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                f1, f2 = st.columns(2)
                sev_filter = f1.multiselect("Filter by Severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM"], key="audit_sev_filt")
                ver_filter = f2.multiselect("Filter by Verdict", ["PENDING", "CONFIRMED", "FALSE_POSITIVE", "DISMISSED"], default=["PENDING", "CONFIRMED", "FALSE_POSITIVE"], key="audit_ver_filt")

                filtered_incidents = [
                    inc for inc in incidents
                    if (not sev_filter or inc["severity"] in sev_filter)
                    and (not ver_filter or inc["proctor_verdict"] in ver_filter)
                ]

                st.markdown(f"<div style='font-size: 12.5px; color: #475569; font-weight: 600; margin-bottom: 12px;'>Displaying <b>{len(filtered_incidents)}</b> flagged incident(s):</div>", unsafe_allow_html=True)

                for inc in filtered_incidents:
                    sev_cls = "pill-alert" if inc["severity"] in ("CRITICAL", "HIGH") else "pill-warn"
                    
                    with st.expander(
                        f"🚨 #{inc['id']} — [{inc['severity']}] {inc['violation_type']} ({inc['timestamp']})",
                        expanded=True
                    ):
                        i_col1, i_col2 = st.columns([3, 2], gap="medium")
                        
                        with i_col1:
                            st.markdown(f"""
                            <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px;">
                                <span class="{sev_cls}">{inc['severity']}</span>
                                <span style="font-size: 13.5px; font-weight: 700; color: #0F172A;">{inc['reason_summary']}</span>
                            </div>
                            <p class="body-text" style="font-size: 13px; margin-bottom: 8px;">{inc['reason_narrative']}</p>
                            """, unsafe_allow_html=True)

                            st.markdown(f"""
                            <div class="dossier-meta-grid">
                                <div class="dossier-meta-item">
                                    <span class="dossier-meta-label">Incident ID</span>
                                    <span class="dossier-meta-val">#{inc['id']}</span>
                                </div>
                                <div class="dossier-meta-item">
                                    <span class="dossier-meta-label">Frame Index</span>
                                    <span class="dossier-meta-val">#{inc['frame_index']}</span>
                                </div>
                                <div class="dossier-meta-item">
                                    <span class="dossier-meta-label">Timestamp</span>
                                    <span class="dossier-meta-val">{inc['timestamp']}</span>
                                </div>
                                <div class="dossier-meta-item">
                                    <span class="dossier-meta-label">Threat Score</span>
                                    <span class="dossier-meta-val" style="color: {'#DC2626' if inc['risk_score'] >= 70 else '#D97706'};">{inc['risk_score']:.1f}/100</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            if inc.get("evidence_snapshot_path") and os.path.exists(inc["evidence_snapshot_path"]):
                                st.image(inc["evidence_snapshot_path"], caption=f"Forensic Snapshot - Frame #{inc['frame_index']}", width="stretch")
                            elif inc.get("evidence_clip_path") and os.path.exists(inc["evidence_clip_path"]):
                                st.video(inc["evidence_clip_path"])

                        with i_col2:
                            st.markdown("<span class='eyebrow-label'>AI Detection Attribution</span>", unsafe_allow_html=True)
                            details = inc.get("details", {})
                            attribution = details.get("factor_attribution", {})
                            if attribution:
                                attr_df = pd.DataFrame([{"Factor": k.replace("_", " ").title(), "Weight %": v} for k, v in attribution.items()])
                                fig_b = px.bar(attr_df, x="Weight %", y="Factor", orientation='h', color="Weight %", color_continuous_scale="Reds")
                                fig_b.update_layout(
                                    height=140, 
                                    margin=dict(l=5, r=5, t=10, b=5), 
                                    paper_bgcolor="rgba(0,0,0,0)", 
                                    plot_bgcolor="rgba(0,0,0,0)", 
                                    font=dict(color="#0F172A", family="Plus Jakarta Sans")
                                )
                                st.plotly_chart(fig_b, key=f"saas_xai_{inc['id']}", width="stretch")

                            if details.get("recommended_action"):
                                st.info(f"**Suggested Action**: {details['recommended_action']}")

                            st.markdown("<div style='margin-top: 8px;'><span class='eyebrow-label'>Examiner Decision</span></div>", unsafe_allow_html=True)
                            v1, v2, v3 = st.columns(3)
                            if v1.button("✅ Confirm", key=f"saas_conf_{inc['id']}", width="stretch"):
                                db_manager.update_incident_verdict(inc['id'], "CONFIRMED")
                                st.success("Confirmed violation.")
                                st.rerun()
                            if v2.button("⚠️ False Pos", key=f"saas_fp_{inc['id']}", width="stretch"):
                                db_manager.update_incident_verdict(inc['id'], "FALSE_POSITIVE")
                                st.warning("Marked as False Positive.")
                                st.rerun()
                            if v3.button("❌ Dismiss", key=f"saas_dsm_{inc['id']}", width="stretch"):
                                db_manager.update_incident_verdict(inc['id'], "DISMISSED")
                                st.info("Dismissed.")
                                st.rerun()

                            p_notes = st.text_input("Examiner Notes", value=inc.get("proctor_notes") or "", key=f"saas_notes_{inc['id']}")
                            if st.button("💾 Save Notes", key=f"saas_savenotes_{inc['id']}", width="stretch"):
                                db_manager.update_incident_verdict(inc['id'], inc['proctor_verdict'], p_notes)
                                st.success("Notes saved.")

    with col_a2:
        if current_session:
            with st.container(border=True):
                st.markdown("""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h3 class="section-header" style="margin: 0;">📄 Official Exam Reports</h3>
                    <span class="pill-safe">● 1-CLICK EXPORT</span>
                </div>
                <p class="body-text" style="font-size: 0.9rem; color: #475569; margin-bottom: 12px;">Generate certified integrity PDF certificates and detailed CSV logs.</p>
                """, unsafe_allow_html=True)

                try:
                    cand_pdf_data = generate_candidate_pdf_report(session_id, db_manager)
                    st.download_button(
                        label="📄 Download Official PDF Report",
                        data=cand_pdf_data,
                        file_name=f"EviGuard_Report_{session_id}.pdf",
                        mime="application/pdf",
                        type="primary",
                        width="stretch",
                        key="saas_dl_pdf"
                    )
                except Exception as e:
                    st.error(f"Error compiling PDF: {e}")

                st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

                try:
                    cand_csv_data = generate_candidate_csv_report(session_id, db_manager)
                    st.download_button(
                        label="📊 Export CSV Audit Log",
                        data=cand_csv_data,
                        file_name=f"EviGuard_Audit_{session_id}.csv",
                        mime="text/csv",
                        type="primary",
                        width="stretch",
                        key="saas_dl_csv"
                    )
                except Exception as e:
                    st.error(f"Error compiling CSV: {e}")

        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 class="section-header" style="margin: 0; color: #B91C1C;">🧹 Clear Data & Reset</h3>
                <span class="pill-alert">MAINTENANCE</span>
            </div>
            <p class="body-text" style="font-size: 0.9rem; color: #475569; margin-bottom: 12px;">Clear session logs, temporary snapshots, and reset database to start fresh.</p>
            """, unsafe_allow_html=True)

            st.markdown('<div class="danger-action-container">', unsafe_allow_html=True)
            purge_clicked = st.button("⚠️ Reset All Exam Data", width="stretch", key="btn_system_purge")
            st.markdown('</div>', unsafe_allow_html=True)
            if purge_clicked:
                st.cache_data.clear()
                st.cache_resource.clear()
                db_manager.purge_all_data()
                st.session_state.clear()
                st.session_state.system_initialized = True
                st.session_state.incident_logs = []
                st.session_state.flagged_events = []
                st.session_state.evidence_snapshots = []
                st.session_state.audit_history = []
                st.session_state.telemetry_data = []
                st.session_state.active_session_id = None
                st.success("✅ System reset completed!")
                st.rerun()

        with st.container(border=True):
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 class="section-header" style="margin: 0;">⚙️ Detection Settings</h3>
                <span class="eyebrow-label">CALIBRATION</span>
            </div>
            """, unsafe_allow_html=True)

            with st.form("saas_settings_form"):
                c_yolo = st.slider("Object Detection Sensitivity", 0.10, 0.90, 0.22, 0.02)
                c_yaw = st.slider("Head Movement Tolerance (°)", 5.0, 45.0, 16.0, 1.0)
                c_pitch = st.slider("Gaze Pitch Tolerance (°)", 5.0, 45.0, 14.0, 1.0)
                c_phone_w = st.slider("Phone Penalty Weight", 10.0, 100.0, 95.0, 5.0)

                if st.form_submit_button("💾 Save Settings", type="primary", width="stretch"):
                    updated_cfg = {
                        "system": {"app_name": "EviGuard AI", "version": "2.5.0", "inference_stride": 3},
                        "detection": {"confidence_threshold": c_yolo, "phone_confidence_threshold": c_yolo, "person_confidence_threshold": 0.35, "book_confidence_threshold": 0.22, "enable_paper_heuristic": False, "imgsz": 416},
                        "tracking": {"person_conf_threshold": 0.35, "person_nms_iou": 0.45},
                        "pose_gaze": {"head_pose": {"yaw_limit_left": -c_yaw, "yaw_limit_right": c_yaw, "pitch_limit_down": c_pitch}, "face_absence": {"absence_frames_threshold": 15}},
                        "risk_engine": {"weights": {"cell_phone": c_phone_w, "multiple_persons": 90.0, "face_absent": 85.0, "head_pose_deviation": 35.0, "gaze_deviation": 35.0}}
                    }
                    with open("config.yaml", "w") as f:
                        yaml.dump(updated_cfg, f)
                    st.success("Settings saved successfully!")

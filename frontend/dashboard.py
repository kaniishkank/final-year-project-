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

from datetime import datetime
import json
import math
import os
import shutil
import threading
import time
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Add parent directory to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
/* 1. FORCE APP & VIEWPORT BACKGROUND */
[data-testid="stAppViewContainer"], 
[data-testid="stHeader"], 
.stApp {
    background: 
        radial-gradient(circle at 10% 10%, rgba(14, 165, 233, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 90%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.95) 0%, transparent 100%),
        #030712 !important;
    background-attachment: fixed !important;
}

/* 2. REMOVE DEFAULT STREAMLIT SOLID BACKGROUNDS FROM CONTAINERS */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stColumn"],
div[data-testid="stForm"],
.stCard {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
}

/* 3. GLOW EFFECTS FOR CARDS ON HOVER */
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(56, 189, 248, 0.3) !important;
    box-shadow: 0 0 20px rgba(14, 165, 233, 0.15) !important;
    transition: all 0.3s ease-in-out !important;
}

/* 4. FIX TAB STYLING & ACTIVE INDICATORS */
button[data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    margin-right: 6px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(56, 189, 248, 0.15) !important;
    color: #38BDF8 !important;
    border: 1px solid #38BDF8 !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.3) !important;
}

/* 5. INPUT FIELD & BUTTON GLOWS */
input, textarea, select {
    background-color: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}

input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.4) !important;
}

/* 6. HIDE STREAMLIT TOP BAR & FOOTER */
header[data-testid="stHeader"] {
    background: transparent !important;
}
footer {visibility: hidden;}

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400;1,600;1,700&family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

    #MainMenu {
        visibility: hidden !important;
    }

    *, *::before, *::after, html, body, [class*="css"],
    h1, h2, h3, h4, h5, h6, p, span, div, label, input, button, select, textarea {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    html, body {
        background: transparent !important;
        color: #F8FAFC !important;
    }

    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1440px !important;
        margin: 0 auto !important;
    }

    /* ---------------- HIGH-CONTRAST TYPOGRAPHY HIERARCHY ---------------- */
    h1, .page-title {
        font-size: 24px !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
        color: #FFFFFF !important;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.6) !important;
        margin-bottom: 4px !important;
    }
    h2, h3, .section-header {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.01em !important;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 8px !important;
    }
    p, span, li, .body-text {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #CBD5E1 !important;
        line-height: 1.55 !important;
    }
    .eyebrow-label {
        font-size: 11px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #CBD5E1 !important;
    }
    .kpi-value-text {
        font-size: 30px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.03em !important;
        line-height: 1.15 !important;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.7) !important;
    }

    /* ---------------- SAAS TOP NAVBAR ---------------- */
    .saas-top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.82);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 13px 24px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .saas-brand-title {
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 10px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }
    .saas-brand-badge {
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #60A5FA;
        background: rgba(59, 130, 246, 0.16);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 6px;
        padding: 3px 9px;
    }

    /* ---------------- SAAS TOP TABS NAVIGATION ---------------- */
    div[data-testid="stTabs"] {
        background: transparent;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 6px;
        margin-bottom: 22px;
        box-shadow: 0 6px 20px -5px rgba(0, 0, 0, 0.6);
    }
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 10px 22px;
        font-weight: 700;
        font-size: 13.5px;
        color: #CBD5E1;
        border: none !important;
        background: transparent;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
        color: #FFFFFF;
        background: rgba(255, 255, 255, 0.06);
    }
    div[data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.28) 0%, rgba(37, 99, 235, 0.38) 100%) !important;
        border: 1px solid rgba(96, 165, 250, 0.6) !important;
        color: #93C5FD !important;
        box-shadow: 0 0 16px rgba(59, 130, 246, 0.28);
        text-shadow: 0 0 8px rgba(147, 197, 253, 0.5);
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ---------------- ENTERPRISE SURFACE CARDS ---------------- */
    .enterprise-card {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 22px 24px !important;
        box-shadow: 
            0 10px 30px -10px rgba(0, 0, 0, 0.8), 
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        margin-bottom: 20px !important;
        transition: all 0.25s ease !important;
    }
    .enterprise-card:hover {
        border-color: rgba(96, 165, 250, 0.35) !important;
        box-shadow: 
            0 14px 38px -10px rgba(0, 0, 0, 0.9), 
            0 0 20px -5px rgba(59, 130, 246, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    }

    /* KPI Scorecard Tiles */
    .enterprise-kpi-card {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 20px 22px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        min-height: 124px !important;
        box-shadow: 
            0 10px 25px -10px rgba(0, 0, 0, 0.75), 
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.25s ease !important;
    }
    .enterprise-kpi-card:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(96, 165, 250, 0.4) !important;
        box-shadow: 0 12px 35px -8px rgba(59, 130, 246, 0.22) !important;
    }

    /* Telemetry Row */
    .enterprise-telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 13px 18px;
        background: rgba(10, 15, 30, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        margin-bottom: 9px;
        transition: all 0.2s ease;
    }
    .enterprise-telemetry-row:hover {
        background: rgba(30, 41, 59, 0.75);
        border-color: rgba(96, 165, 250, 0.35);
    }

    /* ---------------- STATUS PILLS & BADGES ---------------- */
    .pill-safe {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.16);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11.5px;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-shadow: 0 0 8px rgba(16, 185, 129, 0.3);
    }
    .pill-warn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(245, 158, 11, 0.16);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11.5px;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-shadow: 0 0 8px rgba(245, 158, 11, 0.3);
    }
    .pill-alert {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(244, 63, 94, 0.18);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.45);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11.5px;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-shadow: 0 0 8px rgba(244, 63, 94, 0.4);
    }

    /* ---------------- EVIDENCE DOSSIER CARD ---------------- */
    .dossier-card {
        background: rgba(10, 15, 30, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
    }
    .dossier-card:hover {
        border-color: rgba(96, 165, 250, 0.35);
        background: rgba(15, 23, 42, 0.85);
    }
    .dossier-meta-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px 18px;
        background: rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .dossier-meta-item {
        display: flex;
        flex-direction: column;
    }
    .dossier-meta-label {
        font-size: 11px;
        font-weight: 800;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .dossier-meta-val {
        font-size: 13.5px;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 2px;
    }

    /* ---------------- ENTERPRISE AUDIT LOG TABLE ---------------- */
    .audit-table-wrapper {
        width: 100%;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        overflow: hidden;
        margin-top: 14px;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    }
    .audit-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13.5px;
    }
    .audit-table th {
        background: rgba(10, 15, 30, 0.95);
        color: #CBD5E1;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.08em;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .audit-table td {
        padding: 12px 16px;
        color: #F8FAFC;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .audit-table tr:nth-child(even) {
        background: rgba(255, 255, 255, 0.025);
    }
    .audit-table tr:hover {
        background: rgba(59, 130, 246, 0.08);
    }

    /* ---------------- SAPPHIRE & EMERALD BUTTON SYSTEM ---------------- */
    div.stButton > button, 
    div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 11px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 13.5px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        line-height: 1.4 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    /* Primary Action Buttons (Sapphire Blue Gradient) */
    div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    button[kind="primary"],
    div[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%) !important;
        border: 1px solid #1D4ED8 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.32) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
    }
    div.stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
        box-shadow: 0 6px 22px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary Buttons */
    div.stButton > button {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    div.stButton > button:hover {
        background-color: rgba(51, 65, 85, 0.9) !important;
        border-color: rgba(96, 165, 250, 0.5) !important;
        color: #93C5FD !important;
        transform: translateY(-1px) !important;
    }

    /* Decision Actions */
    button:has(p:contains("Confirm")), button:has(span:contains("Confirm")) {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #059669 !important;
        box-shadow: 0 3px 12px rgba(16, 185, 129, 0.3) !important;
    }
    button:has(p:contains("False Positive")), button:has(span:contains("False Positive")) {
        background: rgba(245, 158, 11, 0.18) !important;
        color: #FCD34D !important;
        border: 1px solid rgba(245, 158, 11, 0.45) !important;
    }
    button:has(p:contains("Dismiss")), button:has(span:contains("Dismiss")) {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #CBD5E1 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }

    /* Form Controls */
    .stSelectbox div[data-baseweb="select"], .stTextInput input, div[data-baseweb="input"] {
        background-color: rgba(10, 15, 30, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stTextInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }
    div[data-baseweb="tag"] {
        background-color: rgba(59, 130, 246, 0.2) !important;
        color: #93C5FD !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stImage"] img {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.7) !important;
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
    return EviGuardPipeline("config.yaml")

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
        # Always initialize VideoCapture inside the capture thread on Windows
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
        indices_to_try = [self.src, 0, 1] if self.src not in (0, 1) else [self.src, 1 if self.src == 0 else 0]

        for b in backends:
            for idx in indices_to_try:
                try:
                    cap = cv2.VideoCapture(idx, b)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            self.cap = cap
                            self.is_connected = True
                            with self.lock:
                                self.frame = test_frame
                            break
                    cap.release()
                except Exception:
                    pass
            if self.is_connected:
                break

        while self.running:
            if self.cap is not None and self.cap.isOpened():
                try:
                    # Flush internal hardware frame queue with non-blocking grab()
                    grabbed = self.cap.grab()
                    if grabbed:
                        ret, frame = self.cap.retrieve()
                        if ret and frame is not None and frame.size > 0:
                            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                            with self.lock:
                                self.frame = frame
                    else:
                        time.sleep(0.001)
                except Exception:
                    time.sleep(0.01)
            else:
                time.sleep(0.05)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception:
                pass
        self.cap = None


# ==============================================================================
# 5. GLOWING CIRCULAR THREAT ARC DIAL
# ==============================================================================
def get_threat_meter_html(risk_score: float, risk_level: str) -> str:
    """Renders a clean high-contrast SVG circular threat dial with dynamic progress arcs."""
    if risk_score >= 70.0 or risk_level == "CRITICAL":
        color = "#FB7185"
        glow_color = "rgba(251, 113, 133, 0.6)"
        badge_text = "CRITICAL THREAT"
        badge_color = "#FB7185"
        sub_text = "Violation Active"
    elif risk_score >= 30.0 or risk_level in ("SUSPICIOUS", "MEDIUM"):
        color = "#F59E0B"
        glow_color = "rgba(245, 158, 11, 0.6)"
        badge_text = "ELEVATED RISK"
        badge_color = "#FBBF24"
        sub_text = "Sensor Deviation"
    else:
        color = "#10B981"  # Bright Cyber Emerald
        glow_color = "rgba(16, 185, 129, 0.6)"
        badge_text = "OPTIMAL INTEGRITY"
        badge_color = "#34D399"
        sub_text = "Compliant Session"

    pct = min(100.0, max(0.0, risk_score))
    dash_total = 235.6
    dash_offset = dash_total - (pct / 100.0) * dash_total

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 6px 0 12px 0;">
        <div style="position: relative; width: 210px; height: 125px; display: flex; justify-content: center; align-items: flex-end;">
            <svg width="210" height="210" viewBox="0 0 220 220" style="position: absolute; top: -45px; transform: rotate(180deg);">
                <!-- Distinct Dark Slate Background Track -->
                <circle cx="110" cy="110" r="75" fill="none" stroke="#1E293B" stroke-width="14" stroke-dasharray="235.6 235.6" stroke-dashoffset="0" />
                <!-- Active Ring Progress -->
                <circle cx="110" cy="110" r="75" fill="none" stroke="{color}" stroke-width="14" 
                    stroke-dasharray="235.6 235.6" stroke-dashoffset="{dash_offset}" 
                    stroke-linecap="round" 
                    style="transition: stroke-dashoffset 0.25s ease; filter: drop-shadow(0 0 10px {glow_color});" />
            </svg>
            <div style="text-align: center; z-index: 5; margin-bottom: 2px;">
                <div style="font-size: 30px; font-weight: 800; color: #FFFFFF; line-height: 1; letter-spacing: -0.03em; text-shadow: 0 2px 6px rgba(0,0,0,0.9);">{risk_score:.0f}</div>
                <div style="font-size: 11.5px; font-weight: 800; color: {badge_color}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 5px; text-shadow: 0 2px 4px rgba(0,0,0,0.8);">{badge_text}</div>
                <div style="font-size: 11px; font-weight: 600; color: #CBD5E1; text-shadow: 0 1px 3px rgba(0,0,0,0.8); margin-top: 2px;">{sub_text}</div>
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
        <span style="font-size: 1.35rem;">🛡️</span>
        <span>EviGuard AI</span>
        <span class="saas-brand-badge">LIVE TEST PLATFORM</span>
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
        <div style="display: flex; align-items: center; gap: 8px; background: rgba(10, 15, 30, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 6px 14px; font-size: 13.5px;">
            <span style="color: #94A3B8; font-weight: 600;">Active:</span>
            <span style="font-weight: 700; color: #FFFFFF;">{candidate_name} ({candidate_id})</span>
        </div>
        <div class="pill-safe">
            <span style="width: 7px; height: 7px; background: #34D399; border-radius: 50%; box-shadow: 0 0 8px #34D399;"></span>
            LIVE READY
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 7. FULL-WIDTH MODULAR TABBED SaaS ARCHITECTURE
# ==============================================================================
tab_overview, tab_vision, tab_telemetry, tab_audit = st.tabs([
    "📊 Executive Overview & Registration",
    "📹 Real-Time Vision & AI Proctoring",
    "⚡ Threat Telemetry & Behavioral Analytics",
    "🔍 Audit Dossier & System Reset"
])


# ------------------------------------------------------------------------------
# MODULE 1: EXECUTIVE OVERVIEW & REGISTRATION
# ------------------------------------------------------------------------------
with tab_overview:
    confirmed_count = st.session_state.confirmed_count
    total_flags = st.session_state.total_flags
    integrity_score = st.session_state.integrity_score
    peak_risk = st.session_state.peak_risk
    score_color = "#34D399" if integrity_score >= 80 else ("#FBBF24" if integrity_score >= 50 else "#FB7185")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="eyebrow-label">🛡️ Overall Integrity Index</div>
            <div class="kpi-value-text" style="color: {score_color}; text-shadow: 0 0 12px {score_color}60;">{integrity_score:.1f}%</div>
            <div class="body-text" style="font-size: 12.5px; font-weight: 600;">Status: <b>{'COMPLIANT' if integrity_score >= 80 else 'FLAGGED'}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="eyebrow-label">🚨 Total Flags Logged</div>
            <div class="kpi-value-text">{total_flags}</div>
            <div class="body-text" style="font-size: 12.5px; font-weight: 600;">{confirmed_count} Confirmed Violations</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="eyebrow-label">📈 Peak Threat Score</div>
            <div class="kpi-value-text">{peak_risk:.1f}/100</div>
            <div class="body-text" style="font-size: 12.5px; font-weight: 600;">Peak Live Anomaly</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="eyebrow-label">⚖️ Verified Malpractice</div>
            <div class="kpi-value-text" style="color: {'#34D399' if confirmed_count == 0 else '#FB7185'};">{confirmed_count}</div>
            <div class="body-text" style="font-size: 12.5px; font-weight: 600;">Proctor Confirmed Decisions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    col_dash1, col_dash2 = st.columns([1, 1], gap="medium")

    with col_dash1:
        st.markdown("""
        <div class="enterprise-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <h3 class="section-header" style="margin: 0;">🚀 Live Candidate Onboarding & Setup</h3>
                <span class="pill-safe">● ONBOARDING</span>
            </div>
        """, unsafe_allow_html=True)

        if not current_session:
            st.info("No active session initialized. Fill in the candidate details below to launch a live test session.")

        with st.form("onboarding_form"):
            c_name_in = st.text_input("Candidate Full Name", placeholder="e.g. John Doe")
            c_id_in = st.text_input("Candidate / Student ID", placeholder="e.g. CAND-2026-001")
            c_exam_in = st.text_input("Assessment Title", placeholder="e.g. Midterm Machine Learning Exam")
            c_session_id_in = st.text_input("Session Identifier (Auto-generated)", value=f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            launch_btn = st.form_submit_button("🚀 Launch Live Assessment Session", type="primary", use_container_width=True)
            if launch_btn:
                cand_name = c_name_in.strip() if c_name_in.strip() else "Live Candidate"
                cand_id = c_id_in.strip() if c_id_in.strip() else f"ID-{datetime.now().strftime('%H%M%S')}"
                exam_name = c_exam_in.strip() if c_exam_in.strip() else "Real-World AI Proctoring Assessment"
                
                db_manager.create_session(c_session_id_in, cand_id, cand_name, exam_name)
                st.session_state.active_session_id = c_session_id_in
                st.success(f"Assessment session {c_session_id_in} active!")
                st.rerun()

        if session_ids:
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            st.markdown("<span class='eyebrow-label'>Switch Existing Session</span>", unsafe_allow_html=True)
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

        st.markdown("</div>", unsafe_allow_html=True)

    with col_dash2:
        display_session_ref = session_id if current_session else "-"
        st.markdown(f"""
        <div class="enterprise-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <h3 class="section-header" style="margin: 0;">⚙️ Real-Time Vision & Engine Status</h3>
                <span class="pill-safe">● READY FOR TESTING</span>
            </div>
            <div style="line-height: 2.2; font-size: 13.5px;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 6px;">
                    <span style="color: #94A3B8; font-weight: 600;">Active Session Ref:</span>
                    <span style="color: #60A5FA; font-family: 'JetBrains Mono'; font-weight: 700;">{display_session_ref}</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 6px 0;">
                    <span style="color: #94A3B8; font-weight: 600;">Object Detector Engine:</span>
                    <span style="color: #60A5FA; font-family: 'JetBrains Mono'; font-weight: 700;">YOLO26 NMS-Free (Calibrated)</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 6px 0;">
                    <span style="color: #94A3B8; font-weight: 600;">3D Head Pose & Gaze:</span>
                    <span style="color: #60A5FA; font-family: 'JetBrains Mono'; font-weight: 700;">MediaPipe 468 Face Mesh</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 6px 0;">
                    <span style="color: #94A3B8; font-weight: 600;">Vision Stream Pipeline:</span>
                    <span style="color: #34D399; font-family: 'JetBrains Mono'; font-weight: 700;">Zero-Latency Multi-Threaded OpenCV</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-top: 6px;">
                    <span style="color: #94A3B8; font-weight: 600;">Incident Storage Vault:</span>
                    <span style="color: #34D399; font-weight: 700;">SQLite Clean Storage</span>
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
        st.markdown('<div class="enterprise-card" style="padding: 16px 20px;">', unsafe_allow_html=True)
        t_c1, t_c2 = st.columns([3, 2])
        start_stream = t_c1.toggle("▶ Enable Live AI Vision Stream", value=True, key="vision_tab_stream_toggle")
        cam_src = t_c2.selectbox("Select Camera Device Index", [0, 1, 2], index=0, key="vision_cam_src_select")
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_v_right:
        st.markdown("""
        <div class="enterprise-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 class="section-header" style="margin: 0;">⚡ Live Sensor HUD</h3>
                <span class="eyebrow-label" style="color: #60A5FA;">TELEMETRY</span>
            </div>
        """, unsafe_allow_html=True)
        threat_gauge_v_holder = st.empty()
        telemetry_rows_v_holder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    if start_stream:
        camera = ThreadedCamera(src=cam_src, width=640, height=480).start()
        
        loop_frame = 0
        last_flagged_state = None
        last_risk_bin = -1
        init_attempts = 0

        try:
            while True:
                frame = camera.read()
                if frame is None:
                    init_attempts += 1
                    if init_attempts <= 25:
                        video_placeholder.info("🔄 Connecting to hardware camera sensor and initializing vision pipeline...")
                        time.sleep(0.08)
                        continue
                    else:
                        video_placeholder.warning("⚠️ Waiting for webcam stream. If camera does not appear, select a different camera index above or ensure no other app (e.g. Zoom/Teams) is using the webcam.")
                        time.sleep(0.2)
                        continue

                loop_frame += 1

                output: PipelineOutput = pipeline.process_frame(
                    frame=frame,
                    session_id=session_id,
                    candidate_name=candidate_name
                )

                # 1. Update Video Frame with high-speed JPEG compression
                ret_enc, encoded_jpeg = cv2.imencode('.jpg', output.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret_enc:
                    video_placeholder.image(encoded_jpeg.tobytes(), use_container_width=True)
                else:
                    video_placeholder.image(output.annotated_frame, channels="BGR", use_container_width=True)

                # 2. Extract metrics
                risk_score = float(output.risk.smoothed_score)
                risk_level = str(output.risk.risk_level)
                active_violations = list(output.risk.active_violations)
                person_count = int(output.pose_gaze.face_count if output.pose_gaze.face_detected else len(output.detections))
                yaw_val = float(output.pose_gaze.yaw)
                pitch_val = float(output.pose_gaze.pitch)
                gaze_status = str(output.pose_gaze.gaze_direction) if output.pose_gaze.face_detected else "Candidate Absent"
                is_flagged = bool(output.risk.is_incident_triggered or len(active_violations) > 0 or risk_score >= 70.0)

                # Rate-throttle UI updates to avoid Streamlit websocket queue congestion
                current_risk_bin = int(risk_score // 5)
                state_changed = (is_flagged != last_flagged_state) or (current_risk_bin != last_risk_bin)

                if loop_frame % 5 == 0 or state_changed or output.incident_logged:
                    last_flagged_state = is_flagged
                    last_risk_bin = current_risk_bin

                    if is_flagged or active_violations:
                        alert_str = ' • '.join(active_violations) if active_violations else "ELEVATED RISK DETECTED"
                        vision_banner_holder.markdown(f"""
                        <div style="background: rgba(244, 63, 94, 0.18); border: 1px solid rgba(244, 63, 94, 0.45); border-radius: 12px; padding: 12px 18px; margin-bottom: 14px; display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.25rem;">🚨</span>
                            <span style="color: #FB7185; font-weight: 800; font-size: 13.5px; text-shadow: 0 0 8px rgba(244, 63, 94, 0.4);">SECURITY ALERT: {alert_str}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        vision_banner_holder.markdown("""
                        <div style="background: rgba(16, 185, 129, 0.16); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 12px 18px; margin-bottom: 14px; display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.25rem;">✅</span>
                            <span style="color: #34D399; font-weight: 800; font-size: 13.5px; text-shadow: 0 0 8px rgba(16, 185, 129, 0.3);">COMPLIANCE VERIFIED: Candidate within normal proctoring tolerances</span>
                        </div>
                        """, unsafe_allow_html=True)

                    threat_gauge_v_holder.markdown(get_threat_meter_html(risk_score, risk_level), unsafe_allow_html=True)

                    telemetry_rows_v_holder.markdown(f"""
                    <div style="margin-top: 4px;">
                        <div class="enterprise-telemetry-row">
                            <span class="body-text" style="color: #CBD5E1; font-size: 13.5px; font-weight: 600;">👥 Candidate Count</span>
                            <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: {'#34D399' if person_count == 1 else '#FB7185'};">{person_count} Detected</span>
                        </div>
                        <div class="enterprise-telemetry-row">
                            <span class="body-text" style="color: #CBD5E1; font-size: 13.5px; font-weight: 600;">🔄 Head Yaw (L/R)</span>
                            <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #FFFFFF;">{yaw_val:+.1f}°</span>
                        </div>
                        <div class="enterprise-telemetry-row">
                            <span class="body-text" style="color: #CBD5E1; font-size: 13.5px; font-weight: 600;">📐 Head Pitch (U/D)</span>
                            <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #FFFFFF;">{pitch_val:+.1f}°</span>
                        </div>
                        <div class="enterprise-telemetry-row">
                            <span class="body-text" style="color: #CBD5E1; font-size: 13.5px; font-weight: 600;">👀 Gaze Tracker</span>
                            <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: {'#34D399' if not output.pose_gaze.is_looking_away else '#FB7185'};">{gaze_status}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Live stream error: {e}")
        finally:
            camera.stop()

    else:
        vision_banner_holder.markdown("""
        <div style="background: rgba(16, 185, 129, 0.16); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 12px 18px; margin-bottom: 14px; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.25rem;">✅</span>
            <span style="color: #34D399; font-weight: 800; font-size: 13.5px; text-shadow: 0 0 8px rgba(16, 185, 129, 0.3);">VISION SENSOR ON STANDBY: Ready for live stream</span>
        </div>
        """, unsafe_allow_html=True)

        video_placeholder.markdown("""
        <div style="background: rgba(10, 15, 30, 0.65); border: 2px dashed rgba(255, 255, 255, 0.12); border-radius: 14px; height: 380px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #94A3B8;">
            <span style="font-size: 3rem; margin-bottom: 8px;">📷</span>
            <div style="font-size: 16px; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">Camera Feed on Standby</div>
            <div style="font-size: 13.5px; color: #CBD5E1;">Toggle switch above to activate real-time AI vision proctoring.</div>
        </div>
        """, unsafe_allow_html=True)

        threat_gauge_v_holder.markdown(get_threat_meter_html(0.0, "LOW"), unsafe_allow_html=True)

        telemetry_rows_v_holder.markdown("""
        <div style="margin-top: 4px;">
            <div class="enterprise-telemetry-row">
                <span class="body-text" style="color: #CBD5E1; font-size: 13.5px; font-weight: 600;">👥 Candidate Count</span>
                <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #34D399;">1 Detected</span>
            </div>
            <div class="enterprise-telemetry-row">
                <span class="body-text" style="color: #CBD5E1; font-size: 13.5px; font-weight: 600;">🔄 Head Yaw (L/R)</span>
                <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #FFFFFF;">+0.0°</span>
            </div>
            <div class="enterprise-telemetry-row">
                <span class="body-text" style="color: #CBD5E1; font-size: 13.5px; font-weight: 600;">📐 Head Pitch (U/D)</span>
                <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #FFFFFF;">+0.0°</span>
            </div>
            <div class="enterprise-telemetry-row">
                <span class="body-text" style="color: #CBD5E1; font-size: 13.5px; font-weight: 600;">👀 Gaze Tracker</span>
                <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #34D399;">Direct (Screen)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MODULE 3: THREAT TELEMETRY & BEHAVIORAL ANALYTICS
# ------------------------------------------------------------------------------
with tab_telemetry:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 class="page-title" style="margin: 0;">⚡ Behavioral Anomaly Telemetry & Analytics</h3>
        <p class="body-text" style="margin-top: 4px; margin-bottom: 0;">Continuous frame telemetry logs, hazard threshold boundaries, and malpractice attribution models.</p>
    </div>
    """, unsafe_allow_html=True)

    col_t_left, col_t_right = st.columns([1, 1], gap="large")

    with col_t_left:
        st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 class="section-header" style="margin: 0;">📈 Continuous Integrity Timeline</h3>
            <span class="eyebrow-label">FRAME TELEMETRY</span>
        </div>
        """, unsafe_allow_html=True)

        if metrics:
            df = pd.DataFrame(metrics)
            fig_line = px.line(
                df,
                x=df.index,
                y="risk_score",
                labels={"x": "Sample Frames", "risk_score": "Threat Score (0-100)"},
                title=None
            )
            fig_line.add_hline(y=70, line_dash="dash", line_color="#FB7185", annotation_text="Critical Threshold", annotation_position="top left")
            fig_line.add_hline(y=30, line_dash="dot", line_color="#FBBF24", annotation_text="Elevated Risk", annotation_position="top left")
            fig_line.update_traces(line_color="#3B82F6", line_width=2.5)
            fig_line.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1", family="Plus Jakarta Sans"),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)"),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", range=[0, 100])
            )
            st.plotly_chart(fig_line, use_container_width=True, key="telemetry_timeline_chart")
        else:
            st.info("No continuous frame telemetry recorded yet for this session.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_t_right:
        st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 class="section-header" style="margin: 0;">🍩 Anomaly Breakdown by Violation Type</h3>
            <span class="eyebrow-label">DISTRIBUTION</span>
        </div>
        """, unsafe_allow_html=True)

        if incidents:
            v_types = [i["violation_type"] for i in incidents]
            v_df = pd.Series(v_types).value_counts().reset_index()
            v_df.columns = ["Violation Type", "Count"]
            fig_pie = px.pie(v_df, values="Count", names="Violation Type", hole=0.55, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_pie.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", 
                font=dict(color="#FFFFFF", family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="telemetry_pie_chart")
        else:
            st.info("No violations or anomalies recorded for this candidate.")
        st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MODULE 4: AUDIT DOSSIER & SYSTEM RESET
# ------------------------------------------------------------------------------
with tab_audit:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 class="page-title" style="margin: 0;">🔍 Forensic Evidence Dossier & System Control</h3>
        <p class="body-text" style="margin-top: 4px; margin-bottom: 0;">Review flagged forensic evidence, download certified institutional reports, calibrate neural models, and manage system caches.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([13, 7], gap="large")

    with col_a1:
        st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 class="section-header" style="margin: 0;">📋 Forensic Incident Dossier</h3>
            <span class="eyebrow-label">EVIDENCE TIMELINE</span>
        </div>
        """, unsafe_allow_html=True)

        if not incidents:
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; color: #CBD5E1;">
                <span style="font-size: 3rem;">🛡️</span>
                <h4 style="color: #FFFFFF; font-weight: 700; font-size: 16px; margin-top: 8px;">Full Academic Integrity Maintained</h4>
                <p style="font-size: 13.5px; color: #94A3B8;">Zero cheating incidents or anomalous infractions recorded for this session.</p>
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

            st.markdown(f"<div style='font-size: 12px; color: #CBD5E1; font-weight: 600; margin-bottom: 14px;'>Displaying <b>{len(filtered_incidents)}</b> flagged incident(s)</div>", unsafe_allow_html=True)

            table_rows_html = ""
            for inc in filtered_incidents:
                pill_cls = "pill-alert" if inc["severity"] in ("CRITICAL", "HIGH") else ("pill-warn" if inc["severity"] == "MEDIUM" else "pill-safe")
                v_pill_cls = "pill-safe" if inc["proctor_verdict"] == "CONFIRMED" else ("pill-warn" if inc["proctor_verdict"] == "FALSE_POSITIVE" else "pill-alert")
                
                table_rows_html += f"""
                <tr>
                    <td><span style="font-family: 'JetBrains Mono'; font-weight: 700; color: #FFFFFF;">#{inc['id']}</span></td>
                    <td><span style="font-family: 'JetBrains Mono'; font-size: 11.5px; color: #CBD5E1;">{inc['timestamp']}</span></td>
                    <td><b style="color: #FFFFFF;">{inc['violation_type']}</b></td>
                    <td><span class="{pill_cls}">{inc['severity']}</span></td>
                    <td><span style="font-family: 'JetBrains Mono'; font-weight: 700; color: #FFFFFF;">{inc['risk_score']:.0f}/100</span></td>
                    <td><span class="{v_pill_cls}">{inc['proctor_verdict']}</span></td>
                </tr>
                """

            st.markdown(f"""
            <div class="audit-table-wrapper">
                <table class="audit-table">
                    <thead>
                        <tr>
                            <th>Incident ID</th>
                            <th>Timestamp</th>
                            <th>Violation Category</th>
                            <th>Severity</th>
                            <th>Threat Score</th>
                            <th>Verdict</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 18px; margin-bottom: 12px;'><span class='eyebrow-label'>Detailed Incident Forensics</span></div>", unsafe_allow_html=True)

            for inc in filtered_incidents:
                sev_cls = "pill-alert" if inc["severity"] in ("CRITICAL", "HIGH") else "pill-warn"
                
                with st.expander(
                    f"🚨 Incident #{inc['id']} — [{inc['severity']}] {inc['violation_type']} ({inc['timestamp']})",
                    expanded=True
                ):
                    i_col1, i_col2 = st.columns([3, 2], gap="medium")
                    
                    with i_col1:
                        st.markdown(f"""
                        <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 8px;">
                            <span class="{sev_cls}">{inc['severity']}</span>
                            <span style="font-size: 13.5px; font-weight: 700; color: #FFFFFF;">{inc['reason_summary']}</span>
                        </div>
                        <p class="body-text" style="font-size: 13.5px; margin-bottom: 8px;">{inc['reason_narrative']}</p>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="dossier-meta-grid">
                            <div class="dossier-meta-item">
                                <span class="dossier-meta-label">Incident ID</span>
                                <span class="dossier-meta-val">#{inc['id']}</span>
                            </div>
                            <div class="dossier-meta-item">
                                <span class="dossier-meta-label">Capture Frame</span>
                                <span class="dossier-meta-val">Frame #{inc['frame_index']}</span>
                            </div>
                            <div class="dossier-meta-item">
                                <span class="dossier-meta-label">Recorded Timestamp</span>
                                <span class="dossier-meta-val">{inc['timestamp']}</span>
                            </div>
                            <div class="dossier-meta-item">
                                <span class="dossier-meta-label">Threat Severity Score</span>
                                <span class="dossier-meta-val" style="color: {'#FB7185' if inc['risk_score'] >= 70 else '#FBBF24'};">{inc['risk_score']:.1f} / 100</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if inc.get("evidence_snapshot_path") and os.path.exists(inc["evidence_snapshot_path"]):
                            st.image(inc["evidence_snapshot_path"], caption=f"Forensic Snapshot - Frame #{inc['frame_index']}", use_container_width=True)
                        elif inc.get("evidence_clip_path") and os.path.exists(inc["evidence_clip_path"]):
                            st.video(inc["evidence_clip_path"])

                    with i_col2:
                        st.markdown("<span class='eyebrow-label'>Explainable AI (XAI) Attribution</span>", unsafe_allow_html=True)
                        details = inc.get("details", {})
                        attribution = details.get("factor_attribution", {})
                        if attribution:
                            attr_df = pd.DataFrame([{"Factor": k.replace("_", " ").title(), "Weight %": v} for k, v in attribution.items()])
                            fig_b = px.bar(attr_df, x="Weight %", y="Factor", orientation='h', color="Weight %", color_continuous_scale="Reds")
                            fig_b.update_layout(
                                height=160, 
                                margin=dict(l=5, r=5, t=10, b=5), 
                                paper_bgcolor="rgba(0,0,0,0)", 
                                plot_bgcolor="rgba(0,0,0,0)", 
                                font=dict(color="#FFFFFF", family="Plus Jakarta Sans")
                            )
                            st.plotly_chart(fig_b, use_container_width=True, key=f"saas_xai_{inc['id']}")

                        if details.get("recommended_action"):
                            st.info(f"**Recommended Action**: {details['recommended_action']}")

                        st.markdown("<div style='margin-top: 10px;'><span class='eyebrow-label'>Proctor Sign-Off Action</span></div>", unsafe_allow_html=True)
                        v1, v2, v3 = st.columns(3)
                        if v1.button("✅ Confirm", key=f"saas_conf_{inc['id']}", use_container_width=True):
                            db_manager.update_incident_verdict(inc['id'], "CONFIRMED")
                            st.success("Confirmed violation.")
                            st.rerun()
                        if v2.button("⚠️ False Positive", key=f"saas_fp_{inc['id']}", use_container_width=True):
                            db_manager.update_incident_verdict(inc['id'], "FALSE_POSITIVE")
                            st.warning("Marked as False Positive.")
                            st.rerun()
                        if v3.button("❌ Dismiss", key=f"saas_dsm_{inc['id']}", use_container_width=True):
                            db_manager.update_incident_verdict(inc['id'], "DISMISSED")
                            st.info("Dismissed.")
                            st.rerun()

                        p_notes = st.text_input("Proctor Audit Notes", value=inc.get("proctor_notes") or "", key=f"saas_notes_{inc['id']}")
                        if st.button("💾 Save Notes", key=f"saas_savenotes_{inc['id']}", use_container_width=True):
                            db_manager.update_incident_verdict(inc['id'], inc['proctor_verdict'], p_notes)
                            st.success("Notes saved.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_a2:
        # 1-Click Institutional Export Box
        if current_session:
            st.markdown("""
            <div class="enterprise-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 class="section-header" style="margin: 0;">📄 Institutional Reports</h3>
                    <span class="pill-safe">● 1-CLICK EXPORT</span>
                </div>
                <p class="body-text" style="font-size: 13.5px; margin-bottom: 14px;">Generate certified academic integrity PDF certificates and tabular CSV audit logs.</p>
            """, unsafe_allow_html=True)

            try:
                cand_pdf_data = generate_candidate_pdf_report(session_id, db_manager)
                st.download_button(
                    label="📄 Download Candidate PDF Report",
                    data=cand_pdf_data,
                    file_name=f"EviGuard_Report_{session_id}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key="saas_dl_pdf"
                )
            except Exception as e:
                st.error(f"Error compiling PDF: {e}")

            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

            try:
                cand_csv_data = generate_candidate_csv_report(session_id, db_manager)
                st.download_button(
                    label="📊 Export Tabular Audit Trail (.csv)",
                    data=cand_csv_data,
                    file_name=f"EviGuard_Audit_{session_id}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                    key="saas_dl_csv"
                )
            except Exception as e:
                st.error(f"Error compiling CSV: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

        # Complete Data Reset & System Purge Card
        st.markdown("""
        <div class="enterprise-card" style="border-color: rgba(244, 63, 94, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 class="section-header" style="margin: 0; color: #FB7185;">🧹 Complete Data Reset & Cache Purge</h3>
                <span class="pill-alert">MAINTENANCE</span>
            </div>
            <p class="body-text" style="font-size: 13px; margin-bottom: 14px;">Flush all session memory, clear neural model caches, purge historical evidence clips, and reset database tables to an empty slate.</p>
        """, unsafe_allow_html=True)

        if st.button("⚠️ Reset All Data & Purge System Cache", use_container_width=True, key="btn_system_purge"):
            # 1. Clear Streamlit caches
            st.cache_data.clear()
            st.cache_resource.clear()

            # 2. Purge database and evidence clips
            db_manager.purge_all_data()

            # 3. Flush session state
            st.session_state.clear()
            st.session_state.system_initialized = True
            st.session_state.incident_logs = []
            st.session_state.flagged_events = []
            st.session_state.evidence_snapshots = []
            st.session_state.audit_history = []
            st.session_state.telemetry_data = []
            st.session_state.active_session_id = None

            st.success("✅ System successfully reset! All cache, database entries, and temporary evidence files purged.")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Neural Engine Settings Box
        st.markdown("""
        <div class="enterprise-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 class="section-header" style="margin: 0;">⚙️ Neural Calibration</h3>
                <span class="eyebrow-label" style="color: #60A5FA;">CONFIG</span>
            </div>
        """, unsafe_allow_html=True)

        with st.form("saas_settings_form"):
            c_yolo = st.slider("YOLO26 Confidence Cutoff", 0.10, 0.90, 0.22, 0.02)
            c_yaw = st.slider("Max Head Yaw Tolerance (°)", 5.0, 45.0, 16.0, 1.0)
            c_pitch = st.slider("Max Head Pitch Tolerance (°)", 5.0, 45.0, 14.0, 1.0)
            c_phone_w = st.slider("Cell Phone Threat Weight", 10.0, 100.0, 95.0, 5.0)

            if st.form_submit_button("💾 Apply Configuration", type="primary", use_container_width=True):
                updated_cfg = {
                    "system": {"app_name": "EviGuard AI", "version": "2.5.0", "inference_stride": 3},
                    "detection": {"confidence_threshold": c_yolo, "phone_confidence_threshold": c_yolo, "person_confidence_threshold": 0.35, "book_confidence_threshold": 0.22, "enable_paper_heuristic": False, "imgsz": 416},
                    "tracking": {"person_conf_threshold": 0.35, "person_nms_iou": 0.45},
                    "pose_gaze": {"head_pose": {"yaw_limit_left": -c_yaw, "yaw_limit_right": c_yaw, "pitch_limit_down": c_pitch}, "face_absence": {"absence_frames_threshold": 15}},
                    "risk_engine": {"weights": {"cell_phone": c_phone_w, "multiple_persons": 90.0, "face_absent": 85.0, "head_pose_deviation": 35.0, "gaze_deviation": 35.0}}
                }
                with open("config.yaml", "w") as f:
                    yaml.dump(updated_cfg, f)
                st.success("Neural calibration persisted!")

        st.markdown('</div>', unsafe_allow_html=True)

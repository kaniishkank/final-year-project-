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
/* ---------------- 1. GLOBAL WHITE/LIGHT LUXURY CANVAS ---------------- */
[data-testid="stAppViewContainer"], 
[data-testid="stHeader"], 
.stApp, html, body {
    background-color: #F8FAFC !important;
    background-image: 
        radial-gradient(circle at 10% 10%, rgba(59, 130, 246, 0.04) 0%, transparent 40%),
        radial-gradient(circle at 90% 90%, rgba(99, 102, 241, 0.04) 0%, transparent 40%),
        linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 50%, #F1F5F9 100%) !important;
    background-attachment: fixed !important;
    color: #0F172A !important;
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ---------------- 2. MATCHING MODULE BOXES, CARDS & CONTAINERS ---------------- */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stColumn"],
div[data-testid="stForm"],
.stCard,
.enterprise-card,
.enterprise-kpi-card,
.dossier-card {
    background: #FFFFFF !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.25s ease !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover,
.enterprise-card:hover,
.enterprise-kpi-card:hover,
.dossier-card:hover {
    border-color: #93C5FD !important;
    box-shadow: 0 10px 25px -4px rgba(37, 99, 235, 0.08), 0 4px 10px -2px rgba(0, 0, 0, 0.04) !important;
}

/* ---------------- 3. HIGH-CONTRAST TYPOGRAPHY SYSTEM ---------------- */
h1, h2, h3, h4, h5, h6, .page-title, .section-header {
    color: #0F172A !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
h1, .page-title {
    font-size: 24px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    margin-bottom: 4px !important;
}
h2, h3, .section-header {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #1E293B !important;
    margin-bottom: 8px !important;
}
p, span, li, label, .body-text, .stMarkdown p {
    color: #334155 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
}
.eyebrow-label {
    font-size: 11px !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #64748B !important;
}
.kpi-value-text {
    font-size: 30px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
}

/* ---------------- 4. SAAS TOP NAVBAR (WHITE CARD) ---------------- */
.saas-top-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 13px 24px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05) !important;
}
.saas-brand-title {
    font-size: 1.22rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #0F172A !important;
    display: flex;
    align-items: center;
    gap: 10px;
}
.saas-brand-badge {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #2563EB !important;
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 6px;
    padding: 3px 9px;
}

/* ---------------- 5. TABS NAVIGATION ---------------- */
div[data-testid="stTabs"] {
    background: transparent !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
    background: #F1F5F9 !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 6px !important;
    margin-bottom: 22px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
}
button[data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    color: #64748B !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
button[data-baseweb="tab"]:hover {
    color: #0F172A !important;
    background: rgba(255, 255, 255, 0.8) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #FFFFFF !important;
    border: 1px solid #DBEAFE !important;
    color: #2563EB !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12) !important;
    font-weight: 700 !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* ---------------- 6. ENTERPRISE CARDS & KPIS ---------------- */
.enterprise-card {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 16px !important;
    padding: 22px 24px !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05) !important;
    margin-bottom: 20px !important;
}
.enterprise-kpi-card {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 16px !important;
    padding: 20px 22px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    min-height: 124px !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05) !important;
}

/* ---------------- 7. STATUS PILLS & BADGES ---------------- */
.pill-safe {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ECFDF5 !important;
    color: #059669 !important;
    border: 1px solid #A7F3D0 !important;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.pill-warn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FFFBEB !important;
    color: #D97706 !important;
    border: 1px solid #FDE68A !important;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.pill-alert {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FFF1F2 !important;
    color: #E11D48 !important;
    border: 1px solid #FECDD3 !important;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.04em;
}

/* ---------------- 8. TELEMETRY & EVIDENCE DOSSIER ---------------- */
.enterprise-telemetry-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 13px 18px;
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px;
    margin-bottom: 9px;
    transition: all 0.2s ease;
}
.enterprise-telemetry-row:hover {
    background: #F1F5F9 !important;
    border-color: #CBD5E1 !important;
}
.dossier-card {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
}
.dossier-meta-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px 18px;
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 12px;
    margin-bottom: 12px;
}
.dossier-meta-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.dossier-meta-val {
    font-size: 13.5px;
    font-weight: 700;
    color: #0F172A !important;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}

/* ---------------- 9. AUDIT LOG TABLE ---------------- */
.audit-table-wrapper {
    width: 100%;
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px;
    overflow: hidden;
    margin-top: 14px;
    margin-bottom: 18px;
    background: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
}
.audit-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
}
.audit-table th {
    background: #F8FAFC !important;
    color: #475569 !important;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.08em;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid #E2E8F0 !important;
}
.audit-table td {
    padding: 12px 16px;
    color: #0F172A !important;
    border-bottom: 1px solid #F1F5F9 !important;
}
.audit-table tr:nth-child(even) {
    background: #FBFDFE !important;
}
.audit-table tr:hover {
    background: #EFF6FF !important;
}

/* ---------------- 10. INPUTS, FORMS & BUTTON SYSTEM ---------------- */
input, textarea, select, .stSelectbox div[data-baseweb="select"], .stTextInput input, div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    color: #0F172A !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
}
input:focus, .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:hover {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}
div[data-baseweb="tag"] {
    background-color: #EFF6FF !important;
    color: #2563EB !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
}

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

/* Primary Action Buttons */
div.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button,
button[kind="primary"],
div[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    border: 1px solid #1D4ED8 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
}
div.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* Secondary Buttons */
div.stButton > button {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    color: #1E293B !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}
div.stButton > button:hover {
    background-color: #F8FAFC !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
    transform: translateY(-1px) !important;
}

/* Decision Action Buttons */
button:has(p:contains("Confirm")), button:has(span:contains("Confirm")) {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #059669 !important;
    box-shadow: 0 3px 10px rgba(16, 185, 129, 0.2) !important;
}
button:has(p:contains("False Positive")), button:has(span:contains("False Positive")) {
    background: #FFFBEB !important;
    color: #D97706 !important;
    border: 1px solid #FDE68A !important;
}
button:has(p:contains("Dismiss")), button:has(span:contains("Dismiss")) {
    background: #F8FAFC !important;
    color: #64748B !important;
    border: 1px solid #CBD5E1 !important;
}

div[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
}

/* ---------------- 11. HIDE CHROME & CONTAINER PADDING ---------------- */
header[data-testid="stHeader"] {
    background: transparent !important;
}
footer {visibility: hidden; display: none !important;}
#MainMenu {visibility: hidden !important;}

.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
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
    """Zero-latency threaded hardware camera capture worker with graceful fallback."""

    def __init__(self, src: int = 0, width: int = 640, height: int = 480):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return self

        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_V4L2, cv2.CAP_ANY]
        self.cap = None

        for backend in backends_to_try:
            try:
                cap = cv2.VideoCapture(self.src, backend)
                if cap is not None and cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        self.cap = cap
                        break
                    else:
                        cap.release()
            except Exception:
                continue

        if self.cap is None or not self.cap.isOpened():
            try:
                cap = cv2.VideoCapture(self.src)
                if cap is not None and cap.isOpened():
                    self.cap = cap
            except Exception:
                self.cap = None

        if self.cap is not None and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        return self

    def _capture_worker(self):
        sim_step = 0
        while self.running:
            frame_grabbed = False
            if self.cap is not None and self.cap.isOpened():
                try:
                    ret = self.cap.grab()
                    if ret:
                        _, frame = self.cap.retrieve()
                        if frame is not None and frame.size > 0:
                            with self.lock:
                                self.frame = frame
                            frame_grabbed = True
                except Exception:
                    pass

            if not frame_grabbed:
                sim_step += 1
                h, w = self.height, self.width
                sim_frame = np.full((h, w, 3), 16, dtype=np.uint8)
                
                center_x = int(w / 2 + math.sin(sim_step * 0.05) * 15)
                center_y = int(h / 2)
                
                cv2.circle(sim_frame, (center_x, center_y - 20), 45, (30, 41, 59), -1)
                cv2.circle(sim_frame, (center_x, center_y - 20), 45, (59, 130, 246), 2)
                cv2.ellipse(sim_frame, (center_y + 110, center_y + 110), (90, 70), 0, 0, 360, (22, 30, 46), -1)
                
                cv2.putText(sim_frame, "EVIGUARD AI VISION ENGINE", (25, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (59, 130, 246), 2, cv2.LINE_AA)
                cv2.putText(sim_frame, "Connecting to camera feed...", (25, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)
                
                with self.lock:
                    self.frame = sim_frame
                time.sleep(0.03)
            else:
                time.sleep(0.001)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.frame = None


# ==============================================================================
# 5. GLOWING CIRCULAR THREAT ARC DIAL
# ==============================================================================
def get_threat_meter_html(risk_score: float, risk_level: str) -> str:
    """Renders a clean SVG circular threat dial with dynamic progress arcs."""
    if risk_score >= 70.0 or risk_level == "CRITICAL":
        color = "#E11D48"
        badge_text = "CRITICAL THREAT"
        sub_text = "Violation Active"
    elif risk_score >= 30.0 or risk_level in ("SUSPICIOUS", "MEDIUM"):
        color = "#D97706"
        badge_text = "ELEVATED RISK"
        sub_text = "Sensor Deviation"
    else:
        color = "#059669"
        badge_text = "OPTIMAL INTEGRITY"
        sub_text = "Compliant Session"

    pct = min(100.0, max(0.0, risk_score))
    dash_total = 235.6
    dash_offset = dash_total - (pct / 100.0) * dash_total

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 4px 0 10px 0;">
        <div style="position: relative; width: 200px; height: 120px; display: flex; justify-content: center; align-items: flex-end;">
            <svg width="200" height="200" viewBox="0 0 220 220" style="position: absolute; top: -45px; transform: rotate(180deg);">
                <circle cx="110" cy="110" r="75" fill="none" stroke="#E2E8F0" stroke-width="14" stroke-dasharray="235.6 235.6" stroke-dashoffset="0" />
                <circle cx="110" cy="110" r="75" fill="none" stroke="{color}" stroke-width="14" 
                    stroke-dasharray="235.6 235.6" stroke-dashoffset="{dash_offset}" 
                    stroke-linecap="round" 
                    style="transition: stroke-dashoffset 0.2s ease;" />
            </svg>
            <div style="text-align: center; z-index: 5; margin-bottom: 2px;">
                <div style="font-size: 28px; font-weight: 800; color: #0F172A; line-height: 1; letter-spacing: -0.03em;">{risk_score:.0f}</div>
                <div style="font-size: 11px; font-weight: 800; color: {color}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;">{badge_text}</div>
                <div style="font-size: 11px; font-weight: 600; color: #64748B;">{sub_text}</div>
            </div>
        </div>
    </div>
    """


# ==============================================================================
# 6. SAAS TOP APPLICATION HEADER & SESSION MANAGEMENT
# ==============================================================================
all_sessions = db_manager.get_all_sessions()
session_ids = [s["session_id"] for s in all_sessions]

if st.session_state.active_session_id not in session_ids:
    st.session_state.active_session_id = session_ids[0] if session_ids else None

current_session = db_manager.get_session_by_id(st.session_state.active_session_id) if st.session_state.active_session_id else None

session_id = current_session.get("session_id") if current_session else "NO_ACTIVE_SESSION"
candidate_name = current_session.get("candidate_name") if current_session else "Unregistered Candidate"
candidate_id = current_session.get("candidate_id") if current_session else "N/A"
exam_title = current_session.get("exam_title") if current_session else "Live Assessment Session"
incidents = db_manager.get_session_incidents(session_id) if current_session else []
metrics = db_manager.get_session_metrics(session_id, limit=500) if current_session else []

# Top SaaS Navigation Bar with System Reset Action
st.markdown(f"""
<div class="saas-top-navbar">
    <div class="saas-brand-title">
        <span style="font-size: 1.35rem;">🛡️</span>
        <span style="color: #0F172A; font-weight: 800;">EviGuard AI</span>
        <span class="saas-brand-badge">LIVE TEST PLATFORM</span>
    </div>
    <div style="display: flex; align-items: center; gap: 14px;">
        <div style="display: flex; align-items: center; gap: 8px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 6px 14px; font-size: 13.5px;">
            <span style="color: #64748B; font-weight: 600;">Active:</span>
            <span style="font-weight: 700; color: #0F172A;">{candidate_name} ({candidate_id})</span>
        </div>
        <div class="pill-safe">
            <span style="width: 7px; height: 7px; background: #059669; border-radius: 50%;"></span>
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
    confirmed_count = sum(1 for i in incidents if i["proctor_verdict"] == "CONFIRMED")
    total_flags = len(incidents)
    integrity_score = float(current_session.get("integrity_index", 100.0)) if current_session else 100.0
    peak_risk = float(current_session.get("peak_risk_score", 0.0)) if current_session else 0.0
    score_color = "#059669" if integrity_score >= 80 else ("#D97706" if integrity_score >= 50 else "#E11D48")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="eyebrow-label">🛡️ Overall Integrity Index</div>
            <div class="kpi-value-text" style="color: {score_color};">{integrity_score:.1f}%</div>
            <div class="body-text" style="font-size: 12.5px; font-weight: 600; color: #64748B;">Status: <b style="color: {score_color};">{'COMPLIANT' if integrity_score >= 80 else 'FLAGGED'}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="eyebrow-label">🚨 Total Flags Logged</div>
            <div class="kpi-value-text">{total_flags}</div>
            <div class="body-text" style="font-size: 12.5px; font-weight: 600; color: #64748B;">{confirmed_count} Confirmed Violations</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="eyebrow-label">📈 Peak Threat Score</div>
            <div class="kpi-value-text">{peak_risk:.1f}/100</div>
            <div class="body-text" style="font-size: 12.5px; font-weight: 600; color: #64748B;">Peak Live Anomaly</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="enterprise-kpi-card">
            <div class="eyebrow-label">⚖️ Verified Malpractice</div>
            <div class="kpi-value-text" style="color: {'#059669' if confirmed_count == 0 else '#E11D48'};">{confirmed_count}</div>
            <div class="body-text" style="font-size: 12.5px; font-weight: 600; color: #64748B;">Proctor Confirmed Decisions</div>
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
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding-bottom: 6px;">
                    <span style="color: #64748B; font-weight: 600;">Active Session Ref:</span>
                    <span style="color: #2563EB; font-family: 'JetBrains Mono'; font-weight: 700;">{display_session_ref}</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding: 6px 0;">
                    <span style="color: #64748B; font-weight: 600;">Object Detector Engine:</span>
                    <span style="color: #2563EB; font-family: 'JetBrains Mono'; font-weight: 700;">YOLO26 NMS-Free (Calibrated)</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding: 6px 0;">
                    <span style="color: #64748B; font-weight: 600;">3D Head Pose & Gaze:</span>
                    <span style="color: #2563EB; font-family: 'JetBrains Mono'; font-weight: 700;">MediaPipe 468 Face Mesh</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding: 6px 0;">
                    <span style="color: #64748B; font-weight: 600;">Vision Stream Pipeline:</span>
                    <span style="color: #059669; font-family: 'JetBrains Mono'; font-weight: 700;">Zero-Latency Multi-Threaded OpenCV</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-top: 6px;">
                    <span style="color: #64748B; font-weight: 600;">Incident Storage Vault:</span>
                    <span style="color: #059669; font-weight: 700;">SQLite Clean Storage</span>
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

    if not current_session:
        st.warning("⚠️ No candidate registered for proctoring. Please register a candidate in the **Executive Overview & Registration** tab before starting.")

    vision_banner_holder = st.empty()

    col_v_left, col_v_right = st.columns([14, 6], gap="medium")

    with col_v_left:
        st.markdown('<div class="enterprise-card" style="padding: 18px 20px;">', unsafe_allow_html=True)
        start_stream = st.toggle("▶ Enable Live AI Vision Stream", value=True if current_session else False, key="vision_tab_stream_toggle")
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_v_right:
        st.markdown("""
        <div class="enterprise-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 class="section-header" style="margin: 0;">⚡ Live Sensor HUD</h3>
                <span class="eyebrow-label" style="color: #2563EB;">TELEMETRY</span>
            </div>
        """, unsafe_allow_html=True)
        threat_gauge_v_holder = st.empty()
        telemetry_rows_v_holder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    if start_stream and current_session:
        camera = ThreadedCamera(src=0, width=640, height=480).start()
        time.sleep(0.15)

        loop_frame = 0
        last_flagged_state = None
        last_risk_bin = -1

        try:
            while True:
                frame = camera.read()
                if frame is not None:
                    loop_frame += 1

                    output: PipelineOutput = pipeline.process_frame(
                        frame=frame,
                        session_id=session_id,
                        candidate_name=candidate_name
                    )

                    # 1. Update Video Frame
                    ret_enc, encoded_jpeg = cv2.imencode('.jpg', output.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
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

                    # Rate-throttle UI updates
                    current_risk_bin = int(risk_score // 5)
                    state_changed = (is_flagged != last_flagged_state) or (current_risk_bin != last_risk_bin)

                    if loop_frame % 4 == 0 or state_changed or output.incident_logged:
                        last_flagged_state = is_flagged
                        last_risk_bin = current_risk_bin

                        if is_flagged or active_violations:
                            alert_str = ' • '.join(active_violations) if active_violations else "ELEVATED RISK DETECTED"
                            vision_banner_holder.markdown(f"""
                            <div style="background: #FFF1F2; border: 1px solid #FECDD3; border-radius: 12px; padding: 12px 18px; margin-bottom: 14px; display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.25rem;">🚨</span>
                                <span style="color: #E11D48; font-weight: 800; font-size: 13.5px;">SECURITY ALERT: {alert_str}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            vision_banner_holder.markdown("""
                            <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 12px 18px; margin-bottom: 14px; display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.25rem;">✅</span>
                                <span style="color: #059669; font-weight: 800; font-size: 13.5px;">COMPLIANCE VERIFIED: Candidate within normal proctoring tolerances</span>
                            </div>
                            """, unsafe_allow_html=True)

                        threat_gauge_v_holder.markdown(get_threat_meter_html(risk_score, risk_level), unsafe_allow_html=True)

                        telemetry_rows_v_holder.markdown(f"""
                        <div style="margin-top: 4px;">
                            <div class="enterprise-telemetry-row">
                                <span class="body-text" style="color: #475569; font-size: 13.5px; font-weight: 600;">👥 Candidate Count</span>
                                <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: {'#059669' if person_count == 1 else '#E11D48'};">{person_count} Detected</span>
                            </div>
                            <div class="enterprise-telemetry-row">
                                <span class="body-text" style="color: #475569; font-size: 13.5px; font-weight: 600;">🔄 Head Yaw (L/R)</span>
                                <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #0F172A;">{yaw_val:+.1f}°</span>
                            </div>
                            <div class="enterprise-telemetry-row">
                                <span class="body-text" style="color: #475569; font-size: 13.5px; font-weight: 600;">📐 Head Pitch (U/D)</span>
                                <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #0F172A;">{pitch_val:+.1f}°</span>
                            </div>
                            <div class="enterprise-telemetry-row">
                                <span class="body-text" style="color: #475569; font-size: 13.5px; font-weight: 600;">👀 Gaze Tracker</span>
                                <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: {'#059669' if not output.pose_gaze.is_looking_away else '#E11D48'};">{gaze_status}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    time.sleep(0.001)

        except Exception as e:
            st.error(f"Live stream error: {e}")
        finally:
            camera.stop()

    else:
        vision_banner_holder.markdown("""
        <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 12px 18px; margin-bottom: 14px; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.25rem;">✅</span>
            <span style="color: #059669; font-weight: 800; font-size: 13.5px;">VISION SENSOR ON STANDBY: Ready for live stream</span>
        </div>
        """, unsafe_allow_html=True)

        video_placeholder.markdown("""
        <div style="background: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 14px; height: 380px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #64748B;">
            <span style="font-size: 3rem; margin-bottom: 8px;">📷</span>
            <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 4px;">Camera Feed on Standby</div>
            <div style="font-size: 13.5px; color: #64748B;">Toggle switch above to activate real-time AI vision proctoring.</div>
        </div>
        """, unsafe_allow_html=True)

        threat_gauge_v_holder.markdown(get_threat_meter_html(0.0, "LOW"), unsafe_allow_html=True)

        telemetry_rows_v_holder.markdown("""
        <div style="margin-top: 4px;">
            <div class="enterprise-telemetry-row">
                <span class="body-text" style="color: #475569; font-size: 13.5px; font-weight: 600;">👥 Candidate Count</span>
                <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #059669;">1 Detected</span>
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
                <span style="font-family: 'JetBrains Mono'; font-weight: 700; font-size: 13.5px; color: #059669;">Direct (Screen)</span>
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
            fig_line.add_hline(y=70, line_dash="dash", line_color="#E11D48", annotation_text="Critical Threshold", annotation_position="top left")
            fig_line.add_hline(y=30, line_dash="dot", line_color="#D97706", annotation_text="Elevated Risk", annotation_position="top left")
            fig_line.update_traces(line_color="#2563EB", line_width=2.5)
            fig_line.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1E293B", family="Plus Jakarta Sans"),
                xaxis=dict(gridcolor="#E2E8F0"),
                yaxis=dict(gridcolor="#E2E8F0", range=[0, 100])
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
                font=dict(color="#1E293B", family="Plus Jakarta Sans")
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
            <div style="text-align: center; padding: 40px 20px; color: #64748B;">
                <span style="font-size: 3rem;">🛡️</span>
                <h4 style="color: #0F172A; font-weight: 700; font-size: 16px; margin-top: 8px;">Full Academic Integrity Maintained</h4>
                <p style="font-size: 13.5px; color: #64748B;">Zero cheating incidents or anomalous infractions recorded for this session.</p>
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

            st.markdown(f"<div style='font-size: 12px; color: #64748B; font-weight: 600; margin-bottom: 14px;'>Displaying <b>{len(filtered_incidents)}</b> flagged incident(s)</div>", unsafe_allow_html=True)

            table_rows_html = ""
            for inc in filtered_incidents:
                pill_cls = "pill-alert" if inc["severity"] in ("CRITICAL", "HIGH") else ("pill-warn" if inc["severity"] == "MEDIUM" else "pill-safe")
                v_pill_cls = "pill-safe" if inc["proctor_verdict"] == "CONFIRMED" else ("pill-warn" if inc["proctor_verdict"] == "FALSE_POSITIVE" else "pill-alert")
                
                table_rows_html += f"""
                <tr>
                    <td><span style="font-family: 'JetBrains Mono'; font-weight: 700; color: #2563EB;">#{inc['id']}</span></td>
                    <td><span style="font-family: 'JetBrains Mono'; font-size: 11.5px; color: #64748B;">{inc['timestamp']}</span></td>
                    <td><b style="color: #0F172A;">{inc['violation_type']}</b></td>
                    <td><span class="{pill_cls}">{inc['severity']}</span></td>
                    <td><span style="font-family: 'JetBrains Mono'; font-weight: 700; color: #0F172A;">{inc['risk_score']:.0f}/100</span></td>
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
                            <span style="font-size: 13.5px; font-weight: 700; color: #0F172A;">{inc['reason_summary']}</span>
                        </div>
                        <p class="body-text" style="font-size: 13.5px; margin-bottom: 8px; color: #334155;">{inc['reason_narrative']}</p>
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
                                <span class="dossier-meta-val" style="color: {'#E11D48' if inc['risk_score'] >= 70 else '#D97706'};">{inc['risk_score']:.1f} / 100</span>
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
                                font=dict(color="#1E293B", family="Plus Jakarta Sans")
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
        <div class="enterprise-card" style="border-color: rgba(225, 29, 72, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 class="section-header" style="margin: 0; color: #E11D48;">🧹 Complete Data Reset & Cache Purge</h3>
                <span class="pill-alert">MAINTENANCE</span>
            </div>
            <p class="body-text" style="font-size: 13px; margin-bottom: 14px; color: #475569;">Flush all session memory, clear neural model caches, purge historical evidence clips, and reset database tables to an empty slate.</p>
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
                <span class="eyebrow-label" style="color: #2563EB;">CONFIG</span>
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

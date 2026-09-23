"""
EviGuard AI Proctoring Studio — Executive AI Command Suite
Ultra-Luxury Enterprise UI (Apple / Linear / Stripe / Palantir aesthetic).
Featuring:
- High-Contrast White Titanium Canvas with Frosted Glass Cards
- Unified Luxury Color Palette (Royal Indigo, Cobalt, Integrity Emerald, Amber, Crimson)
- Zero-Latency Multi-Threaded OpenCV Video Streaming Engine
- Real-Time 3D Neural Telemetry Cockpit & SVG Circular Threat Arc
- Explainable AI (XAI) Forensic Evidence Dossier & 1-Click Institutional Audit Reports
"""

from datetime import datetime
import json
import math
import os
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
# 1. PAGE SETUP & LUXURY EXECUTIVE STYLESHEET
# ==============================================================================
st.set_page_config(
    page_title="EviGuard AI — Executive Proctoring Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,600&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ---------------- GLOBAL RESET & TYPOGRAPHY ---------------- */
    *, *::before, *::after, html, body, [class*="css"], .stApp, 
    h1, h2, h3, h4, h5, h6, p, span, div, label, input, button, select, textarea {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Luxury Titanium Canvas Background */
    html, body, [class*="css"], .stApp {
        background-color: #F8FAFC !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.04) 0px, transparent 45%),
            radial-gradient(at 100% 0%, rgba(37, 99, 235, 0.04) 0px, transparent 45%),
            radial-gradient(at 50% 100%, rgba(241, 245, 249, 0.8) 0px, transparent 100%) !important;
        background-attachment: fixed !important;
        color: #0F172A !important;
    }

    /* Streamlit Spacing & Layout */
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 98% !important;
    }
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.88) !important;
        backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }

    /* ---------------- EXECUTIVE CARD SURFACES ---------------- */
    .luxury-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 24px 26px;
        box-shadow: 
            0 10px 25px -5px rgba(15, 23, 42, 0.04), 
            0 8px 10px -6px rgba(15, 23, 42, 0.02),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        margin-bottom: 20px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .luxury-card:hover {
        box-shadow: 
            0 16px 36px -6px rgba(15, 23, 42, 0.08), 
            0 10px 16px -4px rgba(15, 23, 42, 0.03);
        border-color: #CBD5E1;
    }

    .luxury-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 16px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
    }

    /* ---------------- TOP KPI TILES ---------------- */
    .kpi-grid-tile {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 118px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .kpi-grid-tile:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -4px rgba(79, 70, 229, 0.12);
        border-color: #818CF8;
    }
    .kpi-title {
        font-size: 0.76rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        color: #64748B !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .kpi-metric {
        font-size: 1.70rem !important;
        font-weight: 900 !important;
        color: #0F172A !important;
        letter-spacing: -0.04em !important;
        line-height: 1.15 !important;
        margin-top: 6px;
    }
    .kpi-subtitle {
        font-size: 0.78rem !important;
        color: #64748B !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        margin-top: 6px;
    }

    /* ---------------- TELEMETRY LIST ROWS ---------------- */
    .telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 18px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .telemetry-row:hover {
        background: #F1F5F9;
        border-color: #CBD5E1;
        transform: translateX(2px);
    }
    .telemetry-label {
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .telemetry-val {
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        color: #0F172A !important;
    }

    /* ---------------- STATUS BADGES & PILLS ---------------- */
    .badge-pill-safe {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-pill-warn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-pill-alert {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #FFF1F2;
        color: #E11D48;
        border: 1px solid #FECDD3;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    /* ---------------- BUTTON COLOR SYSTEM ---------------- */
    div.stButton > button, 
    div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 12px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        padding: 11px 22px !important;
        line-height: 1.4 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    /* Primary Action Buttons (Executive Royal Gradient) */
    div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    button[kind="primary"],
    div[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%) !important;
        border: 1px solid #4338CA !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.28) !important;
    }
    div.stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    button[kind="primary"]:hover,
    div[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #4338CA 0%, #1D4ED8 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.38) !important;
        color: #FFFFFF !important;
    }
    div.stDownloadButton > button:active,
    div[data-testid="stFormSubmitButton"] > button:active,
    button[kind="primary"]:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 6px rgba(79, 70, 229, 0.20) !important;
    }

    /* Secondary / Standard Outline Buttons */
    div.stButton > button {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        color: #0F172A !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
    }
    div.stButton > button:hover {
        background-color: #F8FAFC !important;
        border-color: #4F46E5 !important;
        color: #4F46E5 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.12) !important;
    }

    /* Incident Decision Buttons */
    /* Confirm Violation Button (Emerald Gradient) */
    button:has(p:contains("Confirm")), 
    button:has(span:contains("Confirm")) {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #059669 !important;
        box-shadow: 0 3px 10px rgba(16, 185, 129, 0.25) !important;
    }
    button:has(p:contains("Confirm")) p,
    button:has(span:contains("Confirm")) span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    button:has(p:contains("Confirm")):hover, 
    button:has(span:contains("Confirm")):hover {
        background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(16, 185, 129, 0.35) !important;
        color: #FFFFFF !important;
    }

    /* False Positive Button (Warm Amber) */
    button:has(p:contains("False Positive")), 
    button:has(span:contains("False Positive")) {
        background: #FFFBEB !important;
        color: #B45309 !important;
        border: 1.5px solid #FCD34D !important;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.10) !important;
    }
    button:has(p:contains("False Positive")) p,
    button:has(span:contains("False Positive")) span {
        color: #B45309 !important;
        font-weight: 700 !important;
    }
    button:has(p:contains("False Positive")):hover, 
    button:has(span:contains("False Positive")):hover {
        background: #FEF3C7 !important;
        border-color: #F59E0B !important;
        color: #92400E !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(245, 158, 11, 0.18) !important;
    }

    /* Dismiss Button (Slate Tint) */
    button:has(p:contains("Dismiss")), 
    button:has(span:contains("Dismiss")) {
        background: #F8FAFC !important;
        color: #475569 !important;
        border: 1.5px solid #CBD5E1 !important;
    }
    button:has(p:contains("Dismiss")) p,
    button:has(span:contains("Dismiss")) span {
        color: #475569 !important;
        font-weight: 700 !important;
    }
    button:has(p:contains("Dismiss")):hover, 
    button:has(span:contains("Dismiss")):hover {
        background: #F1F5F9 !important;
        border-color: #94A3B8 !important;
        color: #1E293B !important;
        transform: translateY(-1px) !important;
    }

    /* ---------------- SIDEBAR CONTROLS ---------------- */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 2px 0 16px rgba(0, 0, 0, 0.02) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 10px !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #EEF2FF !important;
        border-color: #C7D2FE !important;
        transform: translateX(3px) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
        border-color: #4F46E5 !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span,
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label span {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
    }

    /* Form Controls & Tags */
    .stSelectbox div[data-baseweb="select"], .stTextInput input, div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        font-size: 0.90rem !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stTextInput input:focus, div[data-baseweb="input"]:focus-within {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
    }
    div[data-baseweb="tag"] {
        background-color: #EEF2FF !important;
        color: #4F46E5 !important;
        border: 1px solid #C7D2FE !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
    }

    /* Video Frame */
    div[data-testid="stImage"] img {
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03) !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 700 !important;
        font-size: 0.94rem !important;
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. BACKEND CONNECTIONS & PIPELINE INITIALIZATION
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
# 3. HIGH-SPEED THREADED OPENCV CAMERA WORKER
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
                sim_frame = np.full((h, w, 3), 245, dtype=np.uint8)
                
                center_x = int(w / 2 + math.sin(sim_step * 0.05) * 15)
                center_y = int(h / 2)
                
                cv2.circle(sim_frame, (center_x, center_y - 20), 45, (200, 210, 225), -1)
                cv2.circle(sim_frame, (center_x, center_y - 20), 45, (79, 70, 229), 2)
                cv2.ellipse(sim_frame, (center_x, center_y + 110), (90, 70), 0, 0, 360, (215, 225, 240), -1)
                
                cv2.putText(sim_frame, "EVIGUARD AI VISION ENGINE", (25, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (79, 70, 229), 2, cv2.LINE_AA)
                cv2.putText(sim_frame, "Connecting to camera feed...", (25, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 116, 139), 1, cv2.LINE_AA)
                
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
# 4. LUXURY SVG CIRCULAR THREAT ARC DIAL
# ==============================================================================
def get_threat_meter_html(risk_score: float, risk_level: str) -> str:
    """Renders an executive SVG circular threat dial with dynamic gradient needles."""
    if risk_score >= 70.0 or risk_level == "CRITICAL":
        color = "#E11D48"
        badge_text = "CRITICAL RISK"
        sub_text = "Action Required"
    elif risk_score >= 30.0 or risk_level in ("SUSPICIOUS", "MEDIUM"):
        color = "#D97706"
        badge_text = "ELEVATED RISK"
        sub_text = "Monitoring Closely"
    else:
        color = "#059669"
        badge_text = "OPTIMAL INTEGRITY"
        sub_text = "Compliant Session"

    pct = min(100.0, max(0.0, risk_score))
    dash_total = 235.6
    dash_offset = dash_total - (pct / 100.0) * dash_total

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 6px 0 14px 0;">
        <div style="position: relative; width: 220px; height: 130px; display: flex; justify-content: center; align-items: flex-end;">
            <svg width="220" height="220" viewBox="0 0 220 220" style="position: absolute; top: -45px; transform: rotate(180deg);">
                <!-- Background Rail Track -->
                <circle cx="110" cy="110" r="75" fill="none" stroke="#E2E8F0" stroke-width="16" stroke-dasharray="235.6 235.6" stroke-dashoffset="0" />
                <!-- Dynamic Gradient Threat Needle -->
                <circle cx="110" cy="110" r="75" fill="none" stroke="{color}" stroke-width="16" 
                    stroke-dasharray="235.6 235.6" stroke-dashoffset="{dash_offset}" 
                    stroke-linecap="round" style="transition: stroke-dashoffset 0.2s cubic-bezier(0.4, 0, 0.2, 1);" />
            </svg>
            <div style="text-align: center; z-index: 5; margin-bottom: 4px;">
                <div style="font-size: 2.6rem; font-weight: 900; color: #0F172A; line-height: 1; letter-spacing: -0.04em;">{risk_score:.0f}</div>
                <div style="font-size: 0.72rem; font-weight: 800; color: {color}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;">{badge_text}</div>
                <div style="font-size: 0.68rem; font-weight: 600; color: #64748B;">{sub_text}</div>
            </div>
        </div>
    </div>
    """


# ==============================================================================
# 5. SIDEBAR NAVIGATION & ASSESSMENT INITIALIZATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-top: 4px; margin-bottom: 8px;">
        <div style="background: linear-gradient(135deg, #4F46E5, #2563EB); padding: 10px; border-radius: 12px; box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25);">
            <span style="font-size: 1.4rem; color: #FFFFFF;">🛡️</span>
        </div>
        <div>
            <div style="font-size: 1.30rem; font-weight: 900; color: #0F172A; letter-spacing: -0.03em;">EviGuard AI</div>
            <div style="font-size: 0.72rem; font-weight: 800; color: #4F46E5; text-transform: uppercase; letter-spacing: 0.07em;">Enterprise Suite</div>
        </div>
    </div>
    <div style="margin-top: 6px; margin-bottom: 18px;">
        <span style="background: #EEF2FF; border: 1px solid #C7D2FE; color: #4F46E5; border-radius: 6px; padding: 4px 10px; font-size: 0.74rem; font-weight: 700; font-family: 'JetBrains Mono';">● AI VISION PIPELINE ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

    menu_option = st.radio(
        "Navigation Modules",
        ["📹 Real-Time Proctoring", "🔍 Incident Vault & Dossier", "📊 Executive Audit & Analytics", "⚙️ Neural Tuning & Settings"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 16px; margin-bottom: 16px; border-top: 1px solid #E2E8F0;'></div>", unsafe_allow_html=True)

    all_sessions = db_manager.get_all_sessions()
    session_ids = [s["session_id"] for s in all_sessions]

    if "active_session_id" not in st.session_state:
        if session_ids:
            st.session_state.active_session_id = session_ids[0]
        else:
            default_id = f"EXAM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            db_manager.create_session(default_id, "STD-101", "Alex Johnson", "CS401: Advanced AI Exam")
            st.session_state.active_session_id = default_id

    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: #64748B;">Active Candidate Session</span>
        <span class="badge-pill-safe" style="padding: 2px 8px; font-size: 0.70rem;">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    selected_session = st.selectbox(
        "Session Select",
        session_ids if session_ids else [st.session_state.active_session_id],
        index=0 if not session_ids else (session_ids.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_ids else 0),
        label_visibility="collapsed"
    )
    st.session_state.active_session_id = selected_session

    with st.expander("➕ Initialize New Assessment"):
        new_s_id = st.text_input("Session ID", f"EXAM_{datetime.now().strftime('%H%M%S')}")
        new_c_id = st.text_input("Candidate ID", "STD-102")
        new_c_name = st.text_input("Candidate Name", "Jane Doe")
        new_exam = st.text_input("Exam Name", "Final Engineering Assessment")
        if st.button("🚀 Start Assessment", key="btn_init_exam", type="primary", use_container_width=True):
            db_manager.create_session(new_s_id, new_c_id, new_c_name, new_exam)
            st.session_state.active_session_id = new_s_id
            st.success(f"Session {new_s_id} activated!")
            st.rerun()

    st.markdown("<div style='margin-top: 16px; margin-bottom: 16px; border-top: 1px solid #E2E8F0;'></div>", unsafe_allow_html=True)
    
    # System Telemetry Diagnostics Box
    st.markdown("""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; font-size: 0.76rem; line-height: 1.7; color: #64748B;">
        <div style="font-weight: 800; color: #0F172A; text-transform: uppercase; font-size: 0.72rem; margin-bottom: 6px;">Neural Engine Status</div>
        <div style="display: flex; justify-content: space-between;">
            <span>Object Detector</span>
            <span style="color: #4F46E5; font-family: 'JetBrains Mono'; font-weight: 700;">YOLO26 NMS-Free</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Gaze / 3D Pose</span>
            <span style="color: #4F46E5; font-family: 'JetBrains Mono'; font-weight: 700;">MediaPipe Mesh</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Streaming Engine</span>
            <span style="color: #059669; font-family: 'JetBrains Mono'; font-weight: 700;">Threaded OpenCV</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Database Vault</span>
            <span style="color: #059669; font-weight: 800;">● Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 6. TAB 1: REAL-TIME PROCTORING COCKPIT
# ==============================================================================
if menu_option == "📹 Real-Time Proctoring":
    current_session = db_manager.get_session_by_id(st.session_state.active_session_id) or {
        "session_id": st.session_state.active_session_id,
        "candidate_id": "STD-101",
        "candidate_name": "Alex Johnson",
        "exam_title": "CS401: Advanced AI Exam",
        "integrity_index": 100.0,
        "status": "ACTIVE"
    }

    session_id = current_session.get("session_id", "default_session")
    candidate_name = current_session.get("candidate_name", "Alex Johnson")
    candidate_id = current_session.get("candidate_id", "STD-101")
    exam_title = current_session.get("exam_title", "CS401: Advanced AI Exam")

    # Top KPI Metrics Placeholder
    kpi_placeholder = st.empty()

    st.markdown("<div style='margin-bottom: 18px;'></div>", unsafe_allow_html=True)

    # 65/35 Split Layout
    col_left, col_right = st.columns([13, 7], gap="medium")

    with col_left:
        st.markdown("""
        <div class="luxury-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.15rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">Live Video Stream & AI HUD</span>
                    <span class="badge-pill-safe">● ZERO-LATENCY</span>
                </div>
                <span style="background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 6px; padding: 4px 10px; font-size: 0.74rem; font-family: 'JetBrains Mono'; font-weight: 600; color: #475569;">640x480 • 30 FPS Native</span>
            </div>
        """, unsafe_allow_html=True)

        start_stream = st.toggle("▶ Enable Live AI Vision Stream", value=True, key="live_stream_toggle")
        video_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="luxury-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 1.15rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">Threat Telemetry & Sensors</span>
                <span style="color: #64748B; font-size: 0.76rem; font-weight: 700;">TELEMETRY</span>
            </div>
        """, unsafe_allow_html=True)
        threat_banner_placeholder = st.empty()
        threat_gauge_placeholder = st.empty()
        telemetry_rows_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    if start_stream:
        camera = ThreadedCamera(src=0, width=640, height=480).start()
        time.sleep(0.15)

        loop_frame = 0
        last_flagged_state = None
        last_risk_bin = -1
        cached_total_incidents = 0
        cached_confirmed_incidents = 0
        cached_integrity_pct = 100.0

        try:
            while True:
                frame = camera.read()
                if frame is not None:
                    loop_frame += 1

                    # Process frame through EviGuard AI Neural Pipeline
                    output: PipelineOutput = pipeline.process_frame(
                        frame=frame,
                        session_id=session_id,
                        candidate_name=candidate_name
                    )

                    # 1. Update Video Stream HUD
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

                        # 3. Update Threat Banner
                        if is_flagged or active_violations:
                            alert_str = ' • '.join(active_violations) if active_violations else "ELEVATED RISK DETECTED"
                            threat_banner_placeholder.markdown(f"""
                            <div style="background: #FFF1F2; border: 1px solid #FECDD3; border-radius: 12px; padding: 12px 16px; margin-bottom: 12px; display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.25rem;">🚨</span>
                                <span style="color: #E11D48; font-weight: 800; font-size: 0.88rem;">SECURITY ALERT: {alert_str}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            threat_banner_placeholder.markdown("""
                            <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 12px 16px; margin-bottom: 12px; display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.25rem;">✅</span>
                                <span style="color: #059669; font-weight: 800; font-size: 0.88rem;">COMPLIANCE VERIFIED: Candidate within normal limits</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # 4. Update SVG Threat Meter
                        threat_gauge_placeholder.markdown(get_threat_meter_html(risk_score, risk_level), unsafe_allow_html=True)

                        # 5. Update Sensor Telemetry Rows
                        telemetry_rows_placeholder.markdown(f"""
                        <div style="margin-top: 4px; margin-bottom: 10px;">
                            <div class="telemetry-row">
                                <span class="telemetry-label">👥 Candidate Tracking</span>
                                <span class="telemetry-val" style="color: {'#059669' if person_count == 1 else '#E11D48'};">{person_count} Detected</span>
                            </div>
                            <div class="telemetry-row">
                                <span class="telemetry-label">🔄 Head Pose Yaw (L/R)</span>
                                <span class="telemetry-val">{yaw_val:+.1f}°</span>
                            </div>
                            <div class="telemetry-row">
                                <span class="telemetry-label">📐 Head Pose Pitch (U/D)</span>
                                <span class="telemetry-val">{pitch_val:+.1f}°</span>
                            </div>
                            <div class="telemetry-row">
                                <span class="telemetry-label">👀 Gaze Orientation</span>
                                <span class="telemetry-val" style="color: {'#059669' if not output.pose_gaze.is_looking_away else '#E11D48'};">{gaze_status}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Periodically refresh KPI scorecards
                    if loop_frame % 16 == 0 or output.incident_logged or loop_frame == 1:
                        incidents = db_manager.get_session_incidents(session_id)
                        cached_total_incidents = len(incidents)
                        cached_confirmed_incidents = sum(1 for i in incidents if i.get("proctor_verdict") == "CONFIRMED")
                        cached_integrity_pct = max(0.0, 100.0 - (cached_confirmed_incidents * 5.0) - (cached_total_incidents * 1.5))
                        score_color = "#059669" if cached_integrity_pct >= 80 else ("#D97706" if cached_integrity_pct >= 50 else "#E11D48")

                        if is_flagged or risk_score >= 70.0:
                            badge_html = '<span class="badge-pill-alert">● CRITICAL THREAT</span>'
                        elif risk_score >= 30.0:
                            badge_html = '<span class="badge-pill-warn">● ELEVATED</span>'
                        else:
                            badge_html = '<span class="badge-pill-safe">● ALL CLEAR</span>'

                        with kpi_placeholder.container():
                            k_col1, k_col2, k_col3, k_col4 = st.columns(4)
                            with k_col1:
                                st.markdown(f"""
                                <div class="kpi-grid-tile">
                                    <div class="kpi-title">👤 Candidate Profile</div>
                                    <div class="kpi-metric">{candidate_name}</div>
                                    <div class="kpi-subtitle">ID: <b>{candidate_id}</b></div>
                                </div>
                                """, unsafe_allow_html=True)
                            with k_col2:
                                st.markdown(f"""
                                <div class="kpi-grid-tile">
                                    <div class="kpi-title">📚 Assessment Module</div>
                                    <div class="kpi-metric" style="font-size: 1.30rem;">{exam_title}</div>
                                    <div class="kpi-subtitle">Ref: <code>{session_id}</code></div>
                                </div>
                                """, unsafe_allow_html=True)
                            with k_col3:
                                st.markdown(f"""
                                <div class="kpi-grid-tile">
                                    <div class="kpi-title">🛡️ Integrity Index</div>
                                    <div class="kpi-metric" style="color: {score_color};">{cached_integrity_pct:.1f}%</div>
                                    <div class="kpi-subtitle">Flags: {cached_total_incidents} ({cached_confirmed_incidents} Confirmed)</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with k_col4:
                                st.markdown(f"""
                                <div class="kpi-grid-tile">
                                    <div class="kpi-title">🚦 Defense Status</div>
                                    <div style="margin-top: 6px;">{badge_html}</div>
                                    <div class="kpi-subtitle">Stream: 30 FPS Native</div>
                                </div>
                                """, unsafe_allow_html=True)

                    time.sleep(0.001)

        except Exception as e:
            st.error(f"Live stream error: {e}")
        finally:
            camera.stop()

    else:
        # Standby UI when stream is toggled off
        with video_placeholder:
            st.markdown("""
            <div style="background: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 14px; height: 380px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #64748B;">
                <span style="font-size: 3rem; margin-bottom: 8px;">📷</span>
                <div style="font-size: 1.2rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;">Camera Feed on Standby</div>
                <div style="font-size: 0.88rem; color: #64748B;">Toggle above to <b>▶ Enable Live AI Vision Stream</b> to activate real-time proctoring.</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_placeholder.container():
            k_col1, k_col2, k_col3, k_col4 = st.columns(4)
            with k_col1:
                st.markdown(f"""
                <div class="kpi-grid-tile">
                    <div class="kpi-title">👤 Candidate Profile</div>
                    <div class="kpi-metric">{candidate_name}</div>
                    <div class="kpi-subtitle">ID: <b>{candidate_id}</b></div>
                </div>
                """, unsafe_allow_html=True)
            with k_col2:
                st.markdown(f"""
                <div class="kpi-grid-tile">
                    <div class="kpi-title">📚 Assessment Module</div>
                    <div class="kpi-metric" style="font-size: 1.30rem;">{exam_title}</div>
                    <div class="kpi-subtitle">Ref: <code>{session_id}</code></div>
                </div>
                """, unsafe_allow_html=True)
            with k_col3:
                st.markdown("""
                <div class="kpi-grid-tile">
                    <div class="kpi-title">🛡️ Integrity Index</div>
                    <div class="kpi-metric" style="color: #059669;">100.0%</div>
                    <div class="kpi-subtitle">Flags: 0 (0 Confirmed)</div>
                </div>
                """, unsafe_allow_html=True)
            with k_col4:
                st.markdown("""
                <div class="kpi-grid-tile">
                    <div class="kpi-title">🚦 Defense Status</div>
                    <div style="margin-top: 6px;"><span class="badge-pill-safe">● ALL CLEAR</span></div>
                    <div class="kpi-subtitle">Stream: Standby</div>
                </div>
                """, unsafe_allow_html=True)

        threat_banner_placeholder.markdown("""
        <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 12px 16px; margin-bottom: 12px; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.25rem;">✅</span>
            <span style="color: #059669; font-weight: 800; font-size: 0.88rem;">COMPLIANCE VERIFIED: Candidate within normal limits</span>
        </div>
        """, unsafe_allow_html=True)

        threat_gauge_placeholder.markdown(get_threat_meter_html(0.0, "LOW"), unsafe_allow_html=True)

        telemetry_rows_placeholder.markdown("""
        <div style="margin-top: 4px; margin-bottom: 10px;">
            <div class="telemetry-row">
                <span class="telemetry-label">👥 Candidate Tracking</span>
                <span class="telemetry-val" style="color: #059669;">1 Detected</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">🔄 Head Pose Yaw (L/R)</span>
                <span class="telemetry-val">+0.0°</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">📐 Head Pose Pitch (U/D)</span>
                <span class="telemetry-val">+0.0°</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">👀 Gaze Orientation</span>
                <span class="telemetry-val" style="color: #059669;">Direct (Screen)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 7. TAB 2: INCIDENT VAULT & EVIDENCE REVIEW
# ==============================================================================
elif menu_option == "🔍 Incident Vault & Dossier":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <h2 style="font-weight: 900; color: #0F172A; letter-spacing: -0.03em; margin-bottom: 4px;">🔍 Incident Vault & Forensic Evidence Review</h2>
        <p style="color: #64748B; font-size: 0.94rem; margin: 0;">Review flagged security violations with automated video clips, snapshots, and Explainable AI (XAI) factor attributions.</p>
    </div>
    """, unsafe_allow_html=True)

    incidents = db_manager.get_session_incidents(st.session_state.active_session_id)

    if not incidents:
        st.markdown("""
        <div class="luxury-card" style="text-align: center; padding: 50px 20px;">
            <span style="font-size: 3.5rem;">🛡️</span>
            <h3 style="color: #0F172A; font-weight: 800; font-size: 1.35rem; margin-top: 12px;">No Suspicious Incidents Recorded</h3>
            <p style="color: #64748B; font-size: 0.90rem;">The candidate maintained full compliance throughout this assessment session.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        f_col1, f_col2 = st.columns([2, 2])
        severity_filter = f_col1.multiselect(
            "Filter by Severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM"]
        )
        verdict_filter = f_col2.multiselect(
            "Filter by Verdict",
            ["PENDING", "CONFIRMED", "FALSE_POSITIVE", "DISMISSED"],
            default=["PENDING", "CONFIRMED", "FALSE_POSITIVE"]
        )

        filtered_incidents = [
            inc for inc in incidents
            if (not severity_filter or inc["severity"] in severity_filter)
            and (not verdict_filter or inc["proctor_verdict"] in verdict_filter)
        ]

        st.markdown(f"<div style='font-size: 0.88rem; color: #64748B; margin-bottom: 14px;'>Displaying <b>{len(filtered_incidents)}</b> flagged incident(s)</div>", unsafe_allow_html=True)

        for inc in filtered_incidents:
            with st.expander(
                f"🚨 Incident #{inc['id']} | [{inc['severity']}] {inc['violation_type']} at {inc['timestamp']} (Risk: {inc['risk_score']}/100) — Verdict: {inc['proctor_verdict']}",
                expanded=True
            ):
                inc_col1, inc_col2 = st.columns([3, 2], gap="large")

                with inc_col1:
                    st.markdown(f"#### 📄 {inc['reason_summary']}")
                    st.write(inc['reason_narrative'])

                    if inc.get("evidence_snapshot_path") and os.path.exists(inc["evidence_snapshot_path"]):
                        st.image(inc["evidence_snapshot_path"], caption=f"Snapshot - Frame #{inc['frame_index']}", use_container_width=True)
                    elif inc.get("evidence_clip_path") and os.path.exists(inc["evidence_clip_path"]):
                        st.video(inc["evidence_clip_path"])
                    else:
                        st.caption("📸 Snapshot / Clip recorded in evidence archive.")

                with inc_col2:
                    st.markdown("#### 🧠 Explainable AI (XAI) Attribution")
                    details = inc.get("details", {})
                    attribution = details.get("factor_attribution", {})

                    if attribution:
                        attr_df = pd.DataFrame([
                            {"Factor": k.replace("_", " ").title(), "Weight %": v}
                            for k, v in attribution.items()
                        ])
                        fig_bar = px.bar(
                            attr_df,
                            x="Weight %",
                            y="Factor",
                            orientation='h',
                            title="Threat Contribution Breakdown",
                            color="Weight %",
                            color_continuous_scale="Reds"
                        )
                        fig_bar.update_layout(
                            height=200,
                            margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#334155", family="Plus Jakarta Sans")
                        )
                        st.plotly_chart(fig_bar, use_container_width=True, key=f"xai_chart_{inc['id']}")

                    if details.get("recommended_action"):
                        st.info(f"**Recommended Action**: {details['recommended_action']}")

                    st.markdown("---")
                    st.markdown("##### ⚖️ Proctor Decision")
                    v_col1, v_col2, v_col3 = st.columns(3)
                    
                    if v_col1.button("✅ Confirm", key=f"conf_{inc['id']}", use_container_width=True):
                        db_manager.update_incident_verdict(inc['id'], "CONFIRMED")
                        st.success("Incident confirmed as cheating violation.")
                        st.rerun()

                    if v_col2.button("⚠️ False Positive", key=f"fp_{inc['id']}", use_container_width=True):
                        db_manager.update_incident_verdict(inc['id'], "FALSE_POSITIVE")
                        st.warning("Incident marked as False Positive.")
                        st.rerun()

                    if v_col3.button("❌ Dismiss", key=f"dsm_{inc['id']}", use_container_width=True):
                        db_manager.update_incident_verdict(inc['id'], "DISMISSED")
                        st.info("Incident dismissed.")
                        st.rerun()

                    notes = st.text_input("Proctor Notes", value=inc.get("proctor_notes") or "", key=f"notes_{inc['id']}")
                    if st.button("💾 Save Notes", key=f"save_notes_{inc['id']}", use_container_width=True):
                        db_manager.update_incident_verdict(inc['id'], inc['proctor_verdict'], notes)
                        st.success("Notes saved.")


# ==============================================================================
# 8. TAB 3: EXECUTIVE AUDIT & ANALYTICS
# ==============================================================================
elif menu_option == "📊 Executive Audit & Analytics":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <h2 style="font-weight: 900; color: #0F172A; letter-spacing: -0.03em; margin-bottom: 4px;">📊 Institutional Analytics & Session Audit</h2>
        <p style="color: #64748B; font-size: 0.94rem; margin: 0;">Candidate malpractice breakdown, continuous timeline telemetry, and formal academic reports.</p>
    </div>
    """, unsafe_allow_html=True)
    
    current_session = db_manager.get_session_by_id(st.session_state.active_session_id)
    incidents = db_manager.get_session_incidents(st.session_state.active_session_id)
    metrics = db_manager.get_session_metrics(st.session_state.active_session_id, limit=500)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Integrity Index", f"{current_session.get('integrity_index', 100.0):.1f}%")
    with kpi2:
        st.metric("Total Incidents Flagged", len(incidents))
    with kpi3:
        st.metric("Peak Risk Score", f"{current_session.get('peak_risk_score', 0.0):.1f}/100")
    with kpi4:
        confirmed_count = sum(1 for i in incidents if i["proctor_verdict"] == "CONFIRMED")
        st.metric("Confirmed Violations", confirmed_count)

    st.markdown("---")

    col_c1, col_c2 = st.columns(2, gap="large")

    with col_c1:
        st.subheader("Violation Distribution by Type")
        if incidents:
            v_types = [i["violation_type"] for i in incidents]
            v_df = pd.Series(v_types).value_counts().reset_index()
            v_df.columns = ["Violation Type", "Count"]
            fig_pie = px.pie(v_df, values="Count", names="Violation Type", hole=0.45, color_discrete_sequence=px.colors.sequential.Teal)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#334155", family="Plus Jakarta Sans"))
            st.plotly_chart(fig_pie, use_container_width=True, key="analytics_pie")
        else:
            st.info("No violations recorded for this candidate.")

    with col_c2:
        st.subheader("Session Integrity Timeline")
        if metrics:
            df = pd.DataFrame(metrics)
            fig_line = px.line(
                df,
                x=df.index,
                y="risk_score",
                labels={"x": "Frames", "risk_score": "Risk Index"},
                title="Session Continuous Integrity Timeline"
            )
            fig_line.add_hline(y=70, line_dash="dash", line_color="#EF4444")
            fig_line.add_hline(y=30, line_dash="dot", line_color="#F59E0B")
            fig_line.update_traces(line_color="#4F46E5", line_width=2.5)
            fig_line.update_layout(
                height=185,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748B", family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig_line, use_container_width=True, key="analytics_timeline")
        else:
            st.info("No telemetry logs recorded.")

    st.markdown("---")
    st.subheader("📄 Export Formal Integrity Reports")
    st.caption("Generate audit-grade institutional candidate malpractice reports and tabular incident logs.")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        try:
            cand_pdf_data = generate_candidate_pdf_report(current_session.get("session_id"), db_manager)
            st.download_button(
                label="📄 Download Candidate Integrity Report (.pdf)",
                data=cand_pdf_data,
                file_name=f"EviGuard_Integrity_Report_{current_session.get('session_id')}.pdf",
                mime="application/pdf",
                type="primary",
                help="Download formal academic integrity report with candidate identity, incident breakdown, and proctor sign-off.",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error compiling PDF report: {e}")

    with col_dl2:
        try:
            cand_csv_data = generate_candidate_csv_report(current_session.get("session_id"), db_manager)
            st.download_button(
                label="📊 Export Incident Audit Trail (.csv)",
                data=cand_csv_data,
                file_name=f"EviGuard_Audit_Trail_{current_session.get('session_id')}.csv",
                mime="text/csv",
                type="primary",
                help="Export complete tabular incident logs and AI confidence metrics for university archives.",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error compiling CSV export: {e}")


# ==============================================================================
# 9. TAB 4: NEURAL TUNING & SENSITIVITY SETTINGS
# ==============================================================================
elif menu_option == "⚙️ Neural Tuning & Settings":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <h2 style="font-weight: 900; color: #0F172A; letter-spacing: -0.03em; margin-bottom: 4px;">⚙️ Neural Model Tuning & Sensitivity Calibration</h2>
        <p style="color: #64748B; font-size: 0.94rem; margin: 0;">Customize neural model thresholds, head pose tolerances, and risk penalty weights.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("settings_form"):
        st.subheader("1. Object Detection & Tracking Parameters")
        c1, c2 = st.columns(2)
        conf_thresh = c1.slider("YOLO26 Confidence Baseline", 0.10, 0.90, 0.22, 0.02)
        person_conf = c2.slider("Person Detection Confidence Cutoff", 0.20, 0.90, 0.35, 0.05)

        st.subheader("2. Head Pose & Gaze Limits (Degrees)")
        g1, g2, g3 = st.columns(3)
        yaw_limit = g1.slider("Max Yaw Angle (Turn Left/Right)", 5.0, 45.0, 16.0, 1.0)
        pitch_limit = g2.slider("Max Pitch Angle (Looking Down)", 5.0, 45.0, 14.0, 1.0)
        absence_timeout = g3.slider("Candidate Absence Timeout (Frames)", 5, 120, 15, 5)

        st.subheader("3. Risk Engine Factor Weights")
        r1, r2, r3, r4 = st.columns(4)
        w_phone = r1.slider("Cell Phone Weight", 10.0, 100.0, 95.0, 5.0)
        w_multi = r2.slider("Multiple Persons Weight", 10.0, 100.0, 90.0, 5.0)
        w_absent = r3.slider("Face Absent Weight", 10.0, 100.0, 85.0, 5.0)
        w_gaze = r4.slider("Gaze Deviation Weight", 5.0, 100.0, 35.0, 5.0)

        submitted = st.form_submit_button("💾 Save & Apply Configuration", type="primary", use_container_width=True)
        if submitted:
            updated_cfg = {
                "system": {"app_name": "EviGuard AI", "version": "2.0.0", "inference_stride": 3},
                "detection": {
                    "confidence_threshold": conf_thresh,
                    "phone_confidence_threshold": conf_thresh,
                    "person_confidence_threshold": person_conf,
                    "book_confidence_threshold": 0.22,
                    "enable_paper_heuristic": False,
                    "imgsz": 416
                },
                "tracking": {"person_conf_threshold": person_conf, "person_nms_iou": 0.45},
                "pose_gaze": {
                    "head_pose": {"yaw_limit_left": -yaw_limit, "yaw_limit_right": yaw_limit, "pitch_limit_down": pitch_limit},
                    "face_absence": {"absence_frames_threshold": absence_timeout}
                },
                "risk_engine": {
                    "weights": {
                        "cell_phone": w_phone,
                        "multiple_persons": w_multi,
                        "face_absent": w_absent,
                        "head_pose_deviation": w_gaze,
                        "gaze_deviation": w_gaze
                    }
                }
            }
            with open("config.yaml", "w") as f:
                yaml.dump(updated_cfg, f)
            st.success("Configuration updated and persisted successfully!")

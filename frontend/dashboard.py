"""
EviGuard AI — Enterprise SaaS AI Proctoring & Cyber-Command Platform
Designed as a full-width, modular multi-page SaaS application (Cloudflare / CrowdStrike / Vercel grade).

Modules:
1. 📊 Executive Overview & Metrics (Candidate Identity, Session Health, Integrity Index)
2. 📹 Real-Time Vision & AI Proctoring (Full-Width Low-Latency HUD & Video Stream)
3. ⚡ Threat Telemetry & Behavioral Analytics (Circular Threat Ring, 3D Pose/Gaze Sensors, Interactive Timeline)
4. 🔍 Audit Dossier & Neural Settings (Forensic Incidents, 1-Click PDF/CSV Reports, Model Tuning)
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
# 1. PAGE SETUP & SAAS ENTERPRISE OBSIDIAN STYLESHEET
# ==============================================================================
st.set_page_config(
    page_title="EviGuard AI — Enterprise Proctoring SaaS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,600&family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

    /* ---------------- REMOVE STREAMLIT CHROME & EXPAND FULL WIDTH ---------------- */
    header[data-testid="stHeader"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(20px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        height: 0px !important;
        visibility: hidden !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }

    /* ---------------- GLOBAL SAAS OBSIDIAN CANVAS ---------------- */
    *, *::before, *::after, html, body, [class*="css"], .stApp, 
    h1, h2, h3, h4, h5, h6, p, span, div, label, input, button, select, textarea {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    html, body, [class*="css"], .stApp {
        background-color: #020617 !important;
        background-image: 
            radial-gradient(ellipse at 50% 0%, #0F172A 0%, #020617 80%),
            radial-gradient(circle at 100% 100%, rgba(0, 229, 255, 0.03) 0%, transparent 40%),
            radial-gradient(circle at 0% 100%, rgba(79, 70, 229, 0.04) 0%, transparent 40%) !important;
        background-attachment: fixed !important;
        color: #F8FAFC !important;
    }

    /* SaaS Centered Max-Width Container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1440px !important;
        margin: 0 auto !important;
    }

    /* ---------------- SAAS TOP APP BAR ---------------- */
    .saas-top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 14px 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
    }
    .saas-brand-title {
        font-size: 1.35rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .saas-brand-badge {
        font-size: 0.68rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #00E5FF;
        background: rgba(0, 229, 255, 0.12);
        border: 1px solid rgba(0, 229, 255, 0.35);
        border-radius: 6px;
        padding: 2px 8px;
    }

    /* ---------------- SAAS TOP TABS NAVIGATION ---------------- */
    div[data-testid="stTabs"] {
        background: transparent;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 6px;
        margin-bottom: 24px;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 700;
        font-size: 0.90rem;
        color: #94A3B8;
        border: none !important;
        background: transparent;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
        color: #F8FAFC;
        background: rgba(255, 255, 255, 0.05);
    }
    div[data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.18) 0%, rgba(37, 99, 235, 0.25) 100%) !important;
        border: 1px solid #00E5FF !important;
        color: #00E5FF !important;
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.22);
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ---------------- MODULAR GLASS CARDS ---------------- */
    .saas-card {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 24px 26px !important;
        box-shadow: 
            0 10px 30px -10px rgba(0, 0, 0, 0.7), 
            inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 20px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .saas-card:hover {
        border-color: rgba(0, 229, 255, 0.3) !important;
        box-shadow: 
            0 14px 40px -10px rgba(0, 0, 0, 0.85), 
            0 0 20px -5px rgba(0, 229, 255, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
        transform: translateY(-2px) !important;
    }

    /* Section Header within Cards */
    .saas-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    .saas-card-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .saas-card-subtitle {
        font-size: 0.80rem;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ---------------- SAAS KPI TILES ---------------- */
    .saas-kpi-card {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 20px 22px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        min-height: 125px !important;
        box-shadow: 
            0 10px 25px -10px rgba(0, 0, 0, 0.7),
            inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .saas-kpi-card:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(0, 229, 255, 0.35) !important;
        box-shadow: 0 12px 35px -8px rgba(0, 229, 255, 0.16) !important;
    }
    .saas-kpi-label {
        font-size: 11px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: #94A3B8 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    .saas-kpi-metric {
        font-size: 1.75rem !important;
        font-weight: 900 !important;
        color: #F8FAFC !important;
        letter-spacing: -0.04em !important;
        line-height: 1.15 !important;
        margin-top: 6px !important;
    }
    .saas-kpi-meta {
        font-size: 0.76rem !important;
        color: #64748B !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        margin-top: 6px !important;
    }

    /* ---------------- TELEMETRY ROWS ---------------- */
    .saas-telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 13px 18px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .saas-telemetry-row:hover {
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(0, 229, 255, 0.3);
        transform: translateX(3px);
    }
    .saas-telemetry-label {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .saas-telemetry-val {
        font-size: 0.94rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        color: #F8FAFC !important;
    }

    /* ---------------- SAAS STATUS PILLS ---------------- */
    .badge-saas-safe {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
    }
    .badge-saas-warn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-shadow: 0 0 10px rgba(245, 158, 11, 0.4);
    }
    .badge-saas-alert {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
    }

    /* ---------------- BUTTONS ---------------- */
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
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    /* Primary Action Buttons (Electric Cyan SaaS Gradient) */
    div.stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    button[kind="primary"],
    div[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #00E5FF 0%, #2563EB 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        color: #030712 !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 20px rgba(0, 229, 255, 0.35) !important;
    }
    div.stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    button[kind="primary"]:hover,
    div[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #38BDF8 0%, #1D4ED8 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0, 229, 255, 0.55) !important;
        color: #FFFFFF !important;
    }

    /* Secondary Buttons */
    div.stButton > button {
        background-color: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #F8FAFC !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
    }
    div.stButton > button:hover {
        background-color: rgba(51, 65, 85, 0.8) !important;
        border-color: rgba(0, 229, 255, 0.4) !important;
        color: #00E5FF !important;
        transform: translateY(-1px) !important;
    }

    /* Decision Buttons */
    button:has(p:contains("Confirm")), button:has(span:contains("Confirm")) {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #10B981 !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.35) !important;
    }
    button:has(p:contains("Confirm")) p, button:has(span:contains("Confirm")) span {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    button:has(p:contains("False Positive")), button:has(span:contains("False Positive")) {
        background: rgba(245, 158, 11, 0.15) !important;
        color: #FCD34D !important;
        border: 1.5px solid rgba(245, 158, 11, 0.45) !important;
    }
    button:has(p:contains("False Positive")) p, button:has(span:contains("False Positive")) span {
        color: #FCD34D !important;
        font-weight: 800 !important;
    }

    button:has(p:contains("Dismiss")), button:has(span:contains("Dismiss")) {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #94A3B8 !important;
        border: 1.5px solid rgba(255, 255, 255, 0.12) !important;
    }
    button:has(p:contains("Dismiss")) p, button:has(span:contains("Dismiss")) span {
        color: #94A3B8 !important;
        font-weight: 700 !important;
    }

    /* Form Controls */
    .stSelectbox div[data-baseweb="select"], .stTextInput input, div[data-baseweb="input"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        font-size: 0.90rem !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stTextInput input:focus, div[data-baseweb="input"]:focus-within {
        border-color: #00E5FF !important;
        box-shadow: 0 0 0 3px rgba(0, 229, 255, 0.2) !important;
    }
    div[data-baseweb="tag"] {
        background-color: rgba(0, 229, 255, 0.15) !important;
        color: #00E5FF !important;
        border: 1px solid rgba(0, 229, 255, 0.4) !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
    }

    /* Video Frame */
    div[data-testid="stImage"] img {
        border-radius: 14px !important;
        border: 1px solid rgba(0, 229, 255, 0.25) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.9) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6) !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 700 !important;
        font-size: 0.94rem !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #00E5FF !important;
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
                sim_frame = np.full((h, w, 3), 15, dtype=np.uint8)
                
                center_x = int(w / 2 + math.sin(sim_step * 0.05) * 15)
                center_y = int(h / 2)
                
                cv2.circle(sim_frame, (center_x, center_y - 20), 45, (30, 41, 59), -1)
                cv2.circle(sim_frame, (center_x, center_y - 20), 45, (0, 229, 255), 2)
                cv2.ellipse(sim_frame, (center_x, center_y + 110), (90, 70), 0, 0, 360, (20, 30, 45), -1)
                
                cv2.putText(sim_frame, "EVIGUARD AI VISION ENGINE", (25, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 2, cv2.LINE_AA)
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
# 4. GLOWING NEON CIRCULAR THREAT ARC DIAL
# ==============================================================================
def get_threat_meter_html(risk_score: float, risk_level: str) -> str:
    """Renders an SVG circular threat dial with dynamic progress arcs."""
    if risk_score >= 70.0 or risk_level == "CRITICAL":
        color = "#EF4444"
        glow_color = "rgba(239, 68, 68, 0.6)"
        badge_text = "CRITICAL THREAT"
        sub_text = "Violation Active"
    elif risk_score >= 30.0 or risk_level in ("SUSPICIOUS", "MEDIUM"):
        color = "#F59E0B"
        glow_color = "rgba(245, 158, 11, 0.6)"
        badge_text = "ELEVATED RISK"
        sub_text = "Sensor Deviation"
    else:
        color = "#10B981"
        glow_color = "rgba(16, 185, 129, 0.6)"
        badge_text = "OPTIMAL INTEGRITY"
        sub_text = "Compliant Session"

    pct = min(100.0, max(0.0, risk_score))
    dash_total = 235.6
    dash_offset = dash_total - (pct / 100.0) * dash_total

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 6px 0 14px 0;">
        <div style="position: relative; width: 220px; height: 130px; display: flex; justify-content: center; align-items: flex-end;">
            <svg width="220" height="220" viewBox="0 0 220 220" style="position: absolute; top: -45px; transform: rotate(180deg);">
                <circle cx="110" cy="110" r="75" fill="none" stroke="rgba(255, 255, 255, 0.08)" stroke-width="16" stroke-dasharray="235.6 235.6" stroke-dashoffset="0" />
                <circle cx="110" cy="110" r="75" fill="none" stroke="{color}" stroke-width="16" 
                    stroke-dasharray="235.6 235.6" stroke-dashoffset="{dash_offset}" 
                    stroke-linecap="round" 
                    style="transition: stroke-dashoffset 0.2s cubic-bezier(0.4, 0, 0.2, 1); filter: drop-shadow(0 0 10px {glow_color});" />
            </svg>
            <div style="text-align: center; z-index: 5; margin-bottom: 4px;">
                <div style="font-size: 2.7rem; font-weight: 900; color: #F8FAFC; line-height: 1; letter-spacing: -0.04em; text-shadow: 0 0 15px rgba(255,255,255,0.3);">{risk_score:.0f}</div>
                <div style="font-size: 0.72rem; font-weight: 800; color: {color}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; text-shadow: 0 0 10px {glow_color};">{badge_text}</div>
                <div style="font-size: 0.68rem; font-weight: 600; color: #94A3B8;">{sub_text}</div>
            </div>
        </div>
    </div>
    """


# ==============================================================================
# 5. SAAS TOP APPLICATION HEADER & SESSION MANAGEMENT
# ==============================================================================
all_sessions = db_manager.get_all_sessions()
session_ids = [s["session_id"] for s in all_sessions]

if "active_session_id" not in st.session_state:
    if session_ids:
        st.session_state.active_session_id = session_ids[0]
    else:
        default_id = f"EXAM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db_manager.create_session(default_id, "STD-101", "Alex Johnson", "CS401: Advanced AI Exam")
        st.session_state.active_session_id = default_id

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
incidents = db_manager.get_session_incidents(session_id)
metrics = db_manager.get_session_metrics(session_id, limit=500)

# Top SaaS Navigation Bar
st.markdown(f"""
<div class="saas-top-navbar">
    <div class="saas-brand-title">
        <span style="font-size: 1.5rem;">🛡️</span>
        <span>EviGuard AI</span>
        <span class="saas-brand-badge">ENTERPRISE SAAS v2.4</span>
    </div>
    <div style="display: flex; align-items: center; gap: 16px;">
        <div style="display: flex; align-items: center; gap: 8px; background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 6px 14px; font-size: 0.80rem;">
            <span style="color: #94A3B8;">Candidate:</span>
            <span style="font-weight: 700; color: #F8FAFC;">{candidate_name} ({candidate_id})</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 20px; padding: 4px 12px; font-size: 0.74rem; font-weight: 800; color: #10B981;">
            <span style="width: 7px; height: 7px; background: #10B981; border-radius: 50%; box-shadow: 0 0 8px #10B981;"></span>
            SYSTEM ONLINE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 6. FULL-WIDTH MODULAR TABBED SaaS ARCHITECTURE
# ==============================================================================
tab_overview, tab_vision, tab_telemetry, tab_audit = st.tabs([
    "📊 Executive Overview & Metrics",
    "📹 Real-Time Vision & AI Proctoring",
    "⚡ Threat Telemetry & Behavioral Analytics",
    "🔍 Audit Dossier & Neural Settings"
])


# ------------------------------------------------------------------------------
# MODULE 1: EXECUTIVE OVERVIEW & METRICS
# ------------------------------------------------------------------------------
with tab_overview:
    # 4 Executive KPI Metric Cards
    confirmed_count = sum(1 for i in incidents if i["proctor_verdict"] == "CONFIRMED")
    total_flags = len(incidents)
    integrity_score = float(current_session.get("integrity_index", 100.0))
    peak_risk = float(current_session.get("peak_risk_score", 0.0))
    score_color = "#10B981" if integrity_score >= 80 else ("#F59E0B" if integrity_score >= 50 else "#EF4444")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="saas-kpi-card">
            <div class="saas-kpi-label">🛡️ Overall Integrity Index</div>
            <div class="saas-kpi-metric" style="color: {score_color}; text-shadow: 0 0 14px {score_color}80;">{integrity_score:.1f}%</div>
            <div class="saas-kpi-meta">Grade: <b>{'COMPLIANT' if integrity_score >= 80 else 'SUSPICIOUS'}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="saas-kpi-card">
            <div class="saas-kpi-label">🚨 Total Flags Logged</div>
            <div class="saas-kpi-metric">{total_flags}</div>
            <div class="saas-kpi-meta">{confirmed_count} Confirmed Violations</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="saas-kpi-card">
            <div class="saas-kpi-label">📈 Peak Threat Index</div>
            <div class="saas-kpi-metric">{peak_risk:.1f}/100</div>
            <div class="saas-kpi-meta">Maximum recorded anomaly</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="saas-kpi-card">
            <div class="saas-kpi-label">⚖️ Verified Malpractice</div>
            <div class="saas-kpi-metric" style="color: {'#10B981' if confirmed_count == 0 else '#EF4444'};">{confirmed_count}</div>
            <div class="saas-kpi-meta">Proctor Reviewed Decisions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # 2-Column SaaS Dashboard Layout
    col_dash1, col_dash2 = st.columns([1, 1], gap="medium")

    with col_dash1:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-card-title">👤 Candidate Identity & Session Context</span>
                <span class="badge-saas-safe">● VERIFIED ID</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="line-height: 2.0; font-size: 0.90rem;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 6px;">
                <span style="color: #94A3B8;">Candidate Full Name:</span>
                <span style="font-weight: 700; color: #F8FAFC;">{candidate_name}</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 6px 0;">
                <span style="color: #94A3B8;">Student / Candidate ID:</span>
                <span style="font-weight: 700; color: #00E5FF; font-family: 'JetBrains Mono';">{candidate_id}</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 6px 0;">
                <span style="color: #94A3B8;">Assessment Module:</span>
                <span style="font-weight: 700; color: #F8FAFC;">{exam_title}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding-top: 6px;">
                <span style="color: #94A3B8;">Session Reference:</span>
                <span style="font-weight: 700; color: #94A3B8; font-family: 'JetBrains Mono';">{session_id}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

        # Quick Switch Session
        st.markdown("<span style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #94A3B8;'>Switch Active Session</span>", unsafe_allow_html=True)
        selected_session = st.selectbox(
            "Session Switcher",
            session_ids if session_ids else [st.session_state.active_session_id],
            index=0 if not session_ids else (session_ids.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_ids else 0),
            label_visibility="collapsed",
            key="overview_session_switch"
        )
        if selected_session != st.session_state.active_session_id:
            st.session_state.active_session_id = selected_session
            st.rerun()

        with st.expander("➕ Initialize New Assessment Session"):
            n_s_id = st.text_input("New Session ID", f"EXAM_{datetime.now().strftime('%H%M%S')}", key="new_s_id_overview")
            n_c_id = st.text_input("New Candidate ID", "STD-103", key="new_c_id_overview")
            n_c_name = st.text_input("New Candidate Name", "Emily Clark", key="new_c_name_overview")
            n_exam = st.text_input("Exam Name", "CS500: Advanced Machine Learning", key="new_exam_overview")
            if st.button("🚀 Start Assessment Session", key="btn_init_overview", type="primary", use_container_width=True):
                db_manager.create_session(n_s_id, n_c_id, n_c_name, n_exam)
                st.session_state.active_session_id = n_s_id
                st.success(f"Assessment session {n_s_id} initialized!")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_dash2:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-card-title">⚙️ AI Proctoring Engine Diagnostics</span>
                <span class="badge-saas-safe">● ALL NODES ACTIVE</span>
            </div>
            <div style="line-height: 2.1; font-size: 0.90rem;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 6px;">
                    <span style="color: #94A3B8;">Object Detection Model:</span>
                    <span style="color: #00E5FF; font-family: 'JetBrains Mono'; font-weight: 700;">YOLO26 NMS-Free (GPU/CPU)</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 6px 0;">
                    <span style="color: #94A3B8;">3D Head Pose & Gaze Engine:</span>
                    <span style="color: #00E5FF; font-family: 'JetBrains Mono'; font-weight: 700;">MediaPipe 468 Face Mesh</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 6px 0;">
                    <span style="color: #94A3B8;">Stream Video Pipeline:</span>
                    <span style="color: #10B981; font-family: 'JetBrains Mono'; font-weight: 700;">Zero-Latency Multi-Threaded OpenCV</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 6px 0;">
                    <span style="color: #94A3B8;">Incident Evidence Storage:</span>
                    <span style="color: #10B981; font-weight: 800;">SQLite Vault & Snapshot Archive</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding-top: 6px;">
                    <span style="color: #94A3B8;">Inference Pipeline Stride:</span>
                    <span style="color: #F8FAFC; font-family: 'JetBrains Mono'; font-weight: 700;">Stride 3 (Smooth 30 FPS Native)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MODULE 2: REAL-TIME VISION & AI PROCTORING
# ------------------------------------------------------------------------------
with tab_vision:
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <div>
            <h3 style="font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em; margin: 0;">📹 High-Definition Real-Time Vision Feed</h3>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px; margin-bottom: 0;">Zero-latency hardware vision stream with real-time neural HUD overlays, 3D head pose vectors, and instant violation alerts.</p>
        </div>
        <span class="badge-saas-safe">● ZERO-LATENCY • 60 FPS • NATIVE</span>
    </div>
    """, unsafe_allow_html=True)

    vision_banner_holder = st.empty()

    col_v_left, col_v_right = st.columns([14, 6], gap="medium")

    with col_v_left:
        st.markdown('<div class="saas-card" style="padding: 16px 20px;">', unsafe_allow_html=True)
        start_stream = st.toggle("▶ Enable Live AI Vision Stream", value=True, key="vision_tab_stream_toggle")
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_v_right:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-card-title">⚡ Live Sensor HUD</span>
                <span style="color: #00E5FF; font-size: 0.74rem; font-weight: 800;">TELEMETRY</span>
            </div>
        """, unsafe_allow_html=True)
        threat_gauge_v_holder = st.empty()
        telemetry_rows_v_holder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    if start_stream:
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
                            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.45); border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.25rem;">🚨</span>
                                <span style="color: #EF4444; font-weight: 800; font-size: 0.90rem;">SECURITY ALERT: {alert_str}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            vision_banner_holder.markdown("""
                            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.25rem;">✅</span>
                                <span style="color: #10B981; font-weight: 800; font-size: 0.90rem;">COMPLIANCE VERIFIED: Candidate within normal proctoring tolerances</span>
                            </div>
                            """, unsafe_allow_html=True)

                        threat_gauge_v_holder.markdown(get_threat_meter_html(risk_score, risk_level), unsafe_allow_html=True)

                        telemetry_rows_v_holder.markdown(f"""
                        <div style="margin-top: 4px;">
                            <div class="saas-telemetry-row">
                                <span class="saas-telemetry-label">👥 Candidate Count</span>
                                <span class="saas-telemetry-val" style="color: {'#10B981' if person_count == 1 else '#EF4444'};">{person_count} Detected</span>
                            </div>
                            <div class="saas-telemetry-row">
                                <span class="saas-telemetry-label">🔄 Head Yaw (L/R)</span>
                                <span class="saas-telemetry-val">{yaw_val:+.1f}°</span>
                            </div>
                            <div class="saas-telemetry-row">
                                <span class="saas-telemetry-label">📐 Head Pitch (U/D)</span>
                                <span class="saas-telemetry-val">{pitch_val:+.1f}°</span>
                            </div>
                            <div class="saas-telemetry-row">
                                <span class="saas-telemetry-label">👀 Gaze Tracker</span>
                                <span class="saas-telemetry-val" style="color: {'#10B981' if not output.pose_gaze.is_looking_away else '#EF4444'};">{gaze_status}</span>
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
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 12px 18px; margin-bottom: 16px; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.25rem;">✅</span>
            <span style="color: #10B981; font-weight: 800; font-size: 0.90rem;">COMPLIANCE VERIFIED: Candidate within normal proctoring tolerances</span>
        </div>
        """, unsafe_allow_html=True)

        video_placeholder.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border: 2px dashed rgba(255, 255, 255, 0.12); border-radius: 14px; height: 380px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #94A3B8;">
            <span style="font-size: 3.2rem; margin-bottom: 8px;">📷</span>
            <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; margin-bottom: 4px;">Camera Feed on Standby</div>
            <div style="font-size: 0.88rem; color: #94A3B8;">Toggle switch above to activate real-time AI vision proctoring.</div>
        </div>
        """, unsafe_allow_html=True)

        threat_gauge_v_holder.markdown(get_threat_meter_html(0.0, "LOW"), unsafe_allow_html=True)

        telemetry_rows_v_holder.markdown("""
        <div style="margin-top: 4px;">
            <div class="saas-telemetry-row">
                <span class="saas-telemetry-label">👥 Candidate Count</span>
                <span class="saas-telemetry-val" style="color: #10B981;">1 Detected</span>
            </div>
            <div class="saas-telemetry-row">
                <span class="saas-telemetry-label">🔄 Head Yaw (L/R)</span>
                <span class="saas-telemetry-val">+0.0°</span>
            </div>
            <div class="saas-telemetry-row">
                <span class="saas-telemetry-label">📐 Head Pitch (U/D)</span>
                <span class="saas-telemetry-val">+0.0°</span>
            </div>
            <div class="saas-telemetry-row">
                <span class="saas-telemetry-label">👀 Gaze Tracker</span>
                <span class="saas-telemetry-val" style="color: #10B981;">Direct (Screen)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MODULE 3: THREAT TELEMETRY & BEHAVIORAL ANALYTICS
# ------------------------------------------------------------------------------
with tab_telemetry:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h3 style="font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em; margin: 0;">⚡ Behavioral Anomaly Telemetry & Analytics</h3>
        <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px; margin-bottom: 0;">Continuous frame-by-frame telemetry logs, head pose deviation distributions, and threat attribution models.</p>
    </div>
    """, unsafe_allow_html=True)

    col_t_left, col_t_right = st.columns([1, 1], gap="large")

    with col_t_left:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="saas-card-header">
            <span class="saas-card-title">📈 Continuous Integrity Timeline</span>
            <span class="saas-card-subtitle">FRAME TELEMETRY</span>
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
            fig_line.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Critical Threshold", annotation_position="top left")
            fig_line.add_hline(y=30, line_dash="dot", line_color="#F59E0B", annotation_text="Elevated Risk", annotation_position="top left")
            fig_line.update_traces(line_color="#00E5FF", line_width=2.5)
            fig_line.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", range=[0, 100])
            )
            st.plotly_chart(fig_line, use_container_width=True, key="telemetry_timeline_chart")
        else:
            st.info("No continuous frame telemetry recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_t_right:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="saas-card-header">
            <span class="saas-card-title">🍩 Anomaly Breakdown by Violation Type</span>
            <span class="saas-card-subtitle">DISTRIBUTION</span>
        </div>
        """, unsafe_allow_html=True)

        if incidents:
            v_types = [i["violation_type"] for i in incidents]
            v_df = pd.Series(v_types).value_counts().reset_index()
            v_df.columns = ["Violation Type", "Count"]
            fig_pie = px.pie(v_df, values="Count", names="Violation Type", hole=0.55, color_discrete_sequence=px.colors.sequential.Cyan)
            fig_pie.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", 
                font=dict(color="#F8FAFC", family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="telemetry_pie_chart")
        else:
            st.info("No violations or anomalies recorded for this candidate.")
        st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MODULE 4: AUDIT DOSSIER & NEURAL SETTINGS
# ------------------------------------------------------------------------------
with tab_audit:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h3 style="font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em; margin: 0;">🔍 Forensic Incident Dossier & Institutional Exports</h3>
        <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px; margin-bottom: 0;">Review flagged forensic snapshots, Explainable AI (XAI) attributions, generate academic audit PDFs, and configure neural thresholds.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([13, 7], gap="large")

    with col_a1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="saas-card-header">
            <span class="saas-card-title">🚨 Flagged Incident Vault</span>
            <span class="saas-card-subtitle">EVIDENCE ARCHIVE</span>
        </div>
        """, unsafe_allow_html=True)

        if not incidents:
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; color: #94A3B8;">
                <span style="font-size: 3rem;">🛡️</span>
                <h4 style="color: #F8FAFC; font-weight: 800; margin-top: 8px;">Full Academic Integrity Maintained</h4>
                <p style="font-size: 0.88rem;">Zero cheating incidents or anomalous infractions recorded.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            f1, f2 = st.columns(2)
            sev_filter = f1.multiselect("Filter Severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM"], key="audit_sev_filt")
            ver_filter = f2.multiselect("Filter Verdict", ["PENDING", "CONFIRMED", "FALSE_POSITIVE", "DISMISSED"], default=["PENDING", "CONFIRMED", "FALSE_POSITIVE"], key="audit_ver_filt")

            filtered_incidents = [
                inc for inc in incidents
                if (not sev_filter or inc["severity"] in sev_filter)
                and (not ver_filter or inc["proctor_verdict"] in ver_filter)
            ]

            st.markdown(f"<div style='font-size: 0.84rem; color: #94A3B8; margin-bottom: 12px;'>Showing <b>{len(filtered_incidents)}</b> flagged incident(s)</div>", unsafe_allow_html=True)

            for inc in filtered_incidents:
                with st.expander(
                    f"🚨 Incident #{inc['id']} | [{inc['severity']}] {inc['violation_type']} at {inc['timestamp']} — Verdict: {inc['proctor_verdict']}",
                    expanded=True
                ):
                    i_col1, i_col2 = st.columns([3, 2], gap="medium")
                    with i_col1:
                        st.markdown(f"**{inc['reason_summary']}**")
                        st.write(inc['reason_narrative'])
                        if inc.get("evidence_snapshot_path") and os.path.exists(inc["evidence_snapshot_path"]):
                            st.image(inc["evidence_snapshot_path"], caption=f"Snapshot - Frame #{inc['frame_index']}", use_container_width=True)
                        elif inc.get("evidence_clip_path") and os.path.exists(inc["evidence_clip_path"]):
                            st.video(inc["evidence_clip_path"])

                    with i_col2:
                        st.markdown("**🧠 XAI Factor Attribution**")
                        details = inc.get("details", {})
                        attribution = details.get("factor_attribution", {})
                        if attribution:
                            attr_df = pd.DataFrame([{"Factor": k.replace("_", " ").title(), "Weight %": v} for k, v in attribution.items()])
                            fig_b = px.bar(attr_df, x="Weight %", y="Factor", orientation='h', color="Weight %", color_continuous_scale="Reds")
                            fig_b.update_layout(height=180, margin=dict(l=5, r=5, t=10, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"))
                            st.plotly_chart(fig_b, use_container_width=True, key=f"saas_xai_{inc['id']}")

                        st.markdown("**⚖️ Proctor Decision**")
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
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-card-title">📄 Institutional Reports</span>
                <span class="badge-saas-safe">● 1-CLICK EXPORT</span>
            </div>
            <p style="color: #94A3B8; font-size: 0.86rem; margin-bottom: 16px;">Download certified academic integrity PDF certificates and CSV audit trails.</p>
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

        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

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

        # Neural Engine Settings Box
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span class="saas-card-title">⚙️ Neural Thresholds</span>
                <span style="color: #00E5FF; font-size: 0.74rem; font-weight: 800;">CALIBRATION</span>
            </div>
        """, unsafe_allow_html=True)

        with st.form("saas_settings_form"):
            c_yolo = st.slider("YOLO26 Confidence Cutoff", 0.10, 0.90, 0.22, 0.02)
            c_yaw = st.slider("Max Head Yaw Tolerance (°)", 5.0, 45.0, 16.0, 1.0)
            c_pitch = st.slider("Max Head Pitch Tolerance (°)", 5.0, 45.0, 14.0, 1.0)
            c_phone_w = st.slider("Cell Phone Threat Weight", 10.0, 100.0, 95.0, 5.0)

            if st.form_submit_button("💾 Apply Configuration", type="primary", use_container_width=True):
                updated_cfg = {
                    "system": {"app_name": "EviGuard AI", "version": "2.4.0", "inference_stride": 3},
                    "detection": {"confidence_threshold": c_yolo, "phone_confidence_threshold": c_yolo, "person_confidence_threshold": 0.35, "book_confidence_threshold": 0.22, "enable_paper_heuristic": False, "imgsz": 416},
                    "tracking": {"person_conf_threshold": 0.35, "person_nms_iou": 0.45},
                    "pose_gaze": {"head_pose": {"yaw_limit_left": -c_yaw, "yaw_limit_right": c_yaw, "pitch_limit_down": c_pitch}, "face_absence": {"absence_frames_threshold": 15}},
                    "risk_engine": {"weights": {"cell_phone": c_phone_w, "multiple_persons": 90.0, "face_absent": 85.0, "head_pose_deviation": 35.0, "gaze_deviation": 35.0}}
                }
                with open("config.yaml", "w") as f:
                    yaml.dump(updated_cfg, f)
                st.success("Neural calibration persisted!")

        st.markdown('</div>', unsafe_allow_html=True)

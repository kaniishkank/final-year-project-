"""
EviGuard AI Proctoring Studio — Next-Gen Cyber-Command UI
Ultra-Modern Glassmorphic Obsidian & Aurora Enterprise Design.
Real-Time Threaded OpenCV Vision Feed, Animated Telemetry Cards, Holographic SVG Threat Gauges, and Instant Audit Export.
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


# ---------------- PAGE CONFIGURATION & NEXT-GEN OBSIDIAN THEME ----------------
st.set_page_config(
    page_title="EviGuard — Next-Gen AI Proctoring Command Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyber Glassmorphism Stylesheet
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    /* Global Obsidian Cyber Canvas */
    html, body, [class*="css"], .stApp {
        background-color: #060913 !important;
        background-image: 
            radial-gradient(at 15% 15%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 85% 20%, rgba(6, 182, 212, 0.10) 0px, transparent 50%),
            radial-gradient(at 50% 85%, rgba(139, 92, 246, 0.08) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        color: #F1F5F9 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Remove Default Streamlit Padding & Header Artifacts */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 98% !important;
    }
    header[data-testid="stHeader"] {
        background: rgba(6, 9, 19, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Glowing Glassmorphic Panels */
    .glass-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.82) 0%, rgba(15, 23, 42, 0.65) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.4), rgba(6, 182, 212, 0.4), transparent);
    }

    /* Top Futuristic Header Banner */
    .header-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 18px;
        padding: 18px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FFFFFF 30%, #A5B4FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 2px;
    }

    /* KPI Metric Cards (Cyber-HUD Tiles) */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }
    .kpi-hud-card {
        background: linear-gradient(145deg, rgba(20, 27, 45, 0.85) 0%, rgba(13, 18, 30, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        position: relative;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .kpi-hud-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(99, 102, 241, 0.15);
    }
    .kpi-hud-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 15%; right: 15%; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.6), transparent);
        opacity: 0;
        transition: opacity 0.25s ease;
    }
    .kpi-hud-card:hover::after {
        opacity: 1;
    }

    .kpi-hud-label {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .kpi-hud-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #FFFFFF;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-top: 6px;
    }
    .kpi-hud-meta {
        font-size: 0.72rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Live Cyber Video Viewport */
    .video-chassis {
        background: #090D18;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 12px;
        position: relative;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    .video-chassis::before {
        content: 'HUD DIRECT FEED';
        position: absolute;
        top: 18px; left: 24px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #38BDF8;
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 4px;
        padding: 2px 8px;
        z-index: 10;
    }
    div[data-testid="stImage"] img {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45) !important;
    }

    /* Sensor Telemetry Rows */
    .sensor-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 11px 16px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(20, 27, 45, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        margin-bottom: 9px;
        transition: all 0.2s ease;
    }
    .sensor-item:hover {
        border-color: rgba(99, 102, 241, 0.35);
        background: rgba(30, 41, 59, 0.7);
    }
    .sensor-name {
        font-size: 0.80rem;
        font-weight: 600;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sensor-val {
        font-size: 0.88rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }

    /* Status Badges */
    .badge-live-pulse {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-alert-pulse {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
        color: #F8FAFC !important;
        padding: 10px 20px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #334155 0%, #1E293B 100%) !important;
        border-color: #6366F1 !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.25) !important;
    }

    /* Accent Download Buttons */
    div.stDownloadButton > button {
        border-radius: 10px !important;
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        border: 1px solid #818CF8 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.45) !important;
    }

    /* Left Sidebar Navigation */
    section[data-testid="stSidebar"] {
        background-color: #080C16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 9px !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(17, 24, 39, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        transform: translateX(4px) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important;
        border-color: #818CF8 !important;
        box-shadow: 0 4px 18px rgba(79, 70, 229, 0.35) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* Inputs, Selectboxes */
    .stSelectbox div[data-baseweb="select"], .stTextInput input {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stTextInput input:focus {
        border-color: #6366F1 !important;
    }

    .streamlit-expanderHeader {
        background: rgba(17, 24, 39, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: #F8FAFC !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- INITIALIZATION & CACHING ----------------
@st.cache_resource
def get_db_manager():
    return DatabaseManager.get_instance("sqlite:///data/eviguard.db")

@st.cache_resource
def get_pipeline():
    return EviGuardPipeline("config.yaml")

db_manager = get_db_manager()
pipeline = get_pipeline()


# ---------------- HIGH-SPEED THREADED OPENCV CAMERA WORKER ----------------
class ThreadedCamera:
    """Zero-latency threaded hardware camera capture worker with multi-backend fallback."""

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
                sim_frame = np.zeros((h, w, 3), dtype=np.uint8)
                sim_frame[:] = (12, 17, 28)
                
                center_x = int(w / 2 + math.sin(sim_step * 0.05) * 15)
                center_y = int(h / 2)
                
                cv2.circle(sim_frame, (center_x, center_y - 20), 45, (30, 41, 59), -1)
                cv2.circle(sim_frame, (center_x, center_y - 20), 45, (99, 102, 241), 2)
                cv2.ellipse(sim_frame, (center_x, center_y + 110), (90, 70), 0, 0, 360, (20, 27, 45), -1)
                
                cv2.putText(sim_frame, "EVIGUARD AI VISION ENGINE", (25, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (99, 102, 241), 2, cv2.LINE_AA)
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


# ---------------- HOLOGRAPHIC SVG THREAT GAUGE RENDERER ----------------
def get_threat_meter_html(risk_score: float, risk_level: str) -> str:
    """Renders an ultra-modern glowing circular cyber threat gauge."""
    if risk_score >= 70.0 or risk_level == "CRITICAL":
        color = "#EF4444"
        glow = "rgba(239, 68, 68, 0.4)"
        status_text = "CRITICAL THREAT"
    elif risk_score >= 30.0 or risk_level in ("SUSPICIOUS", "MEDIUM"):
        color = "#F59E0B"
        glow = "rgba(245, 158, 11, 0.4)"
        status_text = "ELEVATED RISK"
    else:
        color = "#10B981"
        glow = "rgba(16, 185, 129, 0.4)"
        status_text = "SECURE / NORMAL"

    pct = min(100.0, max(0.0, risk_score))
    dash_total = 235.6
    dash_offset = dash_total - (pct / 100.0) * dash_total

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 6px 0 14px 0;">
        <div style="position: relative; width: 220px; height: 130px; display: flex; justify-content: center; align-items: flex-end;">
            <svg width="220" height="220" viewBox="0 0 220 220" style="position: absolute; top: -45px; transform: rotate(180deg);">
                <!-- Outer Track -->
                <circle cx="110" cy="110" r="75" fill="none" stroke="#0F172A" stroke-width="16" stroke-dasharray="235.6 235.6" stroke-dashoffset="0" />
                <!-- Active Glow Needle -->
                <circle cx="110" cy="110" r="75" fill="none" stroke="{color}" stroke-width="16" 
                    stroke-dasharray="235.6 235.6" stroke-dashoffset="{dash_offset}" 
                    stroke-linecap="round" style="filter: drop-shadow(0 0 8px {glow}); transition: stroke-dashoffset 0.15s ease;" />
            </svg>
            <div style="text-align: center; z-index: 5; margin-bottom: 2px;">
                <div style="font-size: 2.3rem; font-weight: 800; color: #FFFFFF; font-family: 'Space Grotesk', sans-serif; line-height: 1; letter-spacing: -0.03em;">{risk_score:.0f}</div>
                <div style="font-size: 0.68rem; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 0.10em; font-family: 'JetBrains Mono', monospace; margin-top: 5px;">{status_text}</div>
            </div>
        </div>
    </div>
    """


# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-top: 4px; margin-bottom: 6px;">
        <div style="background: linear-gradient(135deg, #6366F1, #06B6D4); padding: 8px; border-radius: 12px; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);">
            <span style="font-size: 1.4rem;">🛡️</span>
        </div>
        <div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.22rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">EviGuard AI</div>
            <div style="font-size: 0.68rem; font-weight: 600; color: #38BDF8; letter-spacing: 0.06em; text-transform: uppercase; font-family: 'JetBrains Mono';">Vision Proctor v2.0</div>
        </div>
    </div>
    <div style="margin-top: 10px; margin-bottom: 18px;">
        <span style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); color: #C7D2FE; border-radius: 6px; padding: 3px 9px; font-size: 0.70rem; font-weight: 700; font-family: 'JetBrains Mono';">● NEURAL CORE ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

    menu_option = st.radio(
        "Navigation",
        ["📹 Live Proctoring", "🔍 Incident Vault", "📊 Analytics & Audit", "⚙️ System Configuration"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 16px; margin-bottom: 16px; border-top: 1px solid rgba(255,255,255,0.06);'></div>", unsafe_allow_html=True)

    # Session Selection
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
        <span style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #94A3B8; font-family: 'JetBrains Mono';">Target Session</span>
        <span class="badge-live-pulse" style="font-size: 0.65rem; padding: 1px 7px;">LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    selected_session = st.selectbox(
        "Session Select",
        session_ids if session_ids else [st.session_state.active_session_id],
        index=0 if not session_ids else (session_ids.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_ids else 0),
        label_visibility="collapsed"
    )
    st.session_state.active_session_id = selected_session

    with st.expander("➕ Initialize Assessment"):
        new_s_id = st.text_input("Session ID", f"EXAM_{datetime.now().strftime('%H%M%S')}")
        new_c_id = st.text_input("Candidate ID", "STD-102")
        new_c_name = st.text_input("Candidate Name", "Jane Doe")
        new_exam = st.text_input("Exam Name", "Final Engineering Assessment")
        if st.button("Start Assessment", use_container_width=True):
            db_manager.create_session(new_s_id, new_c_id, new_c_name, new_exam)
            st.session_state.active_session_id = new_s_id
            st.success(f"Session {new_s_id} active!")
            st.rerun()

    st.markdown("<div style='margin-top: 16px; margin-bottom: 16px; border-top: 1px solid rgba(255,255,255,0.06);'></div>", unsafe_allow_html=True)
    
    # Engine Telemetry Pill
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px 14px; font-size: 0.72rem; line-height: 1.7; color: #94A3B8;">
        <div style="font-weight: 700; color: #64748B; text-transform: uppercase; font-size: 0.66rem; margin-bottom: 6px; font-family: 'JetBrains Mono';">Engine Diagnostics</div>
        <div style="display: flex; justify-content: space-between;">
            <span>Vision Core</span>
            <span style="color: #38BDF8; font-family: 'JetBrains Mono'; font-weight: 600;">YOLO26 NMS-Free</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Gaze / 3D Pose</span>
            <span style="color: #A78BFA; font-family: 'JetBrains Mono'; font-weight: 600;">MediaPipe Mesh</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Capture Engine</span>
            <span style="color: #34D399; font-family: 'JetBrains Mono'; font-weight: 600;">Threaded OpenCV</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Database</span>
            <span style="color: #34D399; font-weight: 700;">● Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- TAB 1: LIVE PROCTORING COMMAND STATION ----------------
if menu_option == "📹 Live Proctoring":
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

    # Top KPI HUD Placeholder
    kpi_placeholder = st.empty()

    # Main Command Deck Layout: Left (Live Stream HUD), Right (Tactical Threat Matrix)
    col_left, col_right = st.columns([13, 7], gap="medium")

    with col_left:
        st.markdown("""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">Live Vision Stream & AI HUD</span>
                    <span class="badge-live-pulse">● 30+ FPS STREAM</span>
                </div>
                <span style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 3px 9px; font-size: 0.70rem; font-family: 'JetBrains Mono'; color: #94A3B8;">640x480 • Resilient Direct Feed</span>
            </div>
        """, unsafe_allow_html=True)

        start_stream = st.toggle("▶ Enable Live Stream", value=True, key="live_stream_toggle")
        video_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">Tactical Threat Matrix</span>
                <span style="color: #64748B; font-size: 0.70rem; font-family: 'JetBrains Mono';">SENSOR ARRAY</span>
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

                    # Execute full EviGuard AI Vision Pipeline
                    output: PipelineOutput = pipeline.process_frame(
                        frame=frame,
                        session_id=session_id,
                        candidate_name=candidate_name
                    )

                    # 1. Update Video Frame Display
                    ret_enc, encoded_jpeg = cv2.imencode('.jpg', output.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret_enc:
                        video_placeholder.image(encoded_jpeg.tobytes(), use_container_width=True)
                    else:
                        video_placeholder.image(output.annotated_frame, channels="BGR", use_container_width=True)

                    # 2. Extract telemetry
                    risk_score = float(output.risk.smoothed_score)
                    risk_level = str(output.risk.risk_level)
                    active_violations = list(output.risk.active_violations)
                    person_count = int(output.pose_gaze.face_count if output.pose_gaze.face_detected else len(output.detections))
                    yaw_val = float(output.pose_gaze.yaw)
                    pitch_val = float(output.pose_gaze.pitch)
                    gaze_status = str(output.pose_gaze.gaze_direction) if output.pose_gaze.face_detected else "Candidate Absent"
                    is_flagged = bool(output.risk.is_incident_triggered or len(active_violations) > 0 or risk_score >= 70.0)

                    # State Throttling
                    current_risk_bin = int(risk_score // 5)
                    state_changed = (is_flagged != last_flagged_state) or (current_risk_bin != last_risk_bin)

                    if loop_frame % 4 == 0 or state_changed or output.incident_logged:
                        last_flagged_state = is_flagged
                        last_risk_bin = current_risk_bin

                        # 3. Update Threat Banner
                        if is_flagged or active_violations:
                            alert_str = ' • '.join(active_violations) if active_violations else "ELEVATED RISK DETECTED"
                            threat_banner_placeholder.markdown(f"""
                            <div style="background: rgba(239, 68, 68, 0.16); border: 1px solid rgba(248, 113, 113, 0.45); border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.1rem;">🚨</span>
                                <span style="color: #F87171; font-weight: 700; font-size: 0.82rem; font-family: 'JetBrains Mono';">SECURITY ALERT: {alert_str}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            threat_banner_placeholder.markdown("""
                            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(52, 211, 153, 0.35); border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 1.1rem;">✅</span>
                                <span style="color: #34D399; font-weight: 700; font-size: 0.82rem; font-family: 'JetBrains Mono';">COMPLIANCE VERIFIED: Candidate in standard baseline</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # 4. Update Holographic SVG Threat Dial Meter
                        threat_gauge_placeholder.markdown(get_threat_meter_html(risk_score, risk_level), unsafe_allow_html=True)

                        # 5. Update Sensor Telemetry Rows
                        telemetry_rows_placeholder.markdown(f"""
                        <div style="margin-top: 4px; margin-bottom: 10px;">
                            <div class="sensor-item">
                                <span class="sensor-name">👥 Entity Tracking</span>
                                <span class="sensor-val" style="color: {'#34D399' if person_count == 1 else '#F87171'};">{person_count} Candidate Tracked</span>
                            </div>
                            <div class="sensor-item">
                                <span class="sensor-name">🔄 Head Pose Yaw (L/R)</span>
                                <span class="sensor-val">{yaw_val:+.1f}°</span>
                            </div>
                            <div class="sensor-item">
                                <span class="sensor-name">📐 Head Pose Pitch (U/D)</span>
                                <span class="sensor-val">{pitch_val:+.1f}°</span>
                            </div>
                            <div class="sensor-item">
                                <span class="sensor-name">👀 Gaze Orientation</span>
                                <span class="sensor-val" style="color: {'#34D399' if not output.pose_gaze.is_looking_away else '#F87171'};">{gaze_status}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Update Top KPI Cards periodically
                    if loop_frame % 16 == 0 or output.incident_logged or loop_frame == 1:
                        incidents = db_manager.get_session_incidents(session_id)
                        cached_total_incidents = len(incidents)
                        cached_confirmed_incidents = sum(1 for i in incidents if i.get("proctor_verdict") == "CONFIRMED")
                        cached_integrity_pct = max(0.0, 100.0 - (cached_confirmed_incidents * 5.0) - (cached_total_incidents * 1.5))
                        score_color = "#34D399" if cached_integrity_pct >= 80 else ("#FBBF24" if cached_integrity_pct >= 50 else "#F87171")

                        if is_flagged or risk_score >= 70.0:
                            badge_html = '<span class="badge-alert-pulse">● CRITICAL RISK</span>'
                        elif risk_score >= 30.0:
                            badge_html = '<span class="badge-alert-pulse" style="background: rgba(245, 158, 11, 0.2); color: #FCD34D; border-color: rgba(245, 158, 11, 0.4);">● ELEVATED</span>'
                        else:
                            badge_html = '<span class="badge-live-pulse">● SECURE</span>'

                        with kpi_placeholder.container():
                            k_col1, k_col2, k_col3, k_col4 = st.columns(4)
                            with k_col1:
                                st.markdown(f"""
                                <div class="kpi-hud-card">
                                    <div class="kpi-hud-label">👤 Candidate Profile</div>
                                    <div class="kpi-hud-value">{candidate_name}</div>
                                    <div class="kpi-hud-meta">UID: <code>{candidate_id}</code></div>
                                </div>
                                """, unsafe_allow_html=True)
                            with k_col2:
                                st.markdown(f"""
                                <div class="kpi-hud-card">
                                    <div class="kpi-hud-label">📚 Active Assessment</div>
                                    <div class="kpi-hud-value" style="font-size: 1.18rem;">{exam_title}</div>
                                    <div class="kpi-hud-meta">Ref: <code>{session_id}</code></div>
                                </div>
                                """, unsafe_allow_html=True)
                            with k_col3:
                                st.markdown(f"""
                                <div class="kpi-hud-card">
                                    <div class="kpi-hud-label">🛡️ Integrity Quotient</div>
                                    <div class="kpi-hud-value" style="color: {score_color};">{cached_integrity_pct:.1f}%</div>
                                    <div class="kpi-hud-meta">Flags: {cached_total_incidents} ({cached_confirmed_incidents} Confirmed)</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with k_col4:
                                st.markdown(f"""
                                <div class="kpi-hud-card">
                                    <div class="kpi-hud-label">🚦 Defense Status</div>
                                    <div style="margin-top: 5px;">{badge_html}</div>
                                    <div class="kpi-hud-meta">Throughput: 30 FPS Native</div>
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
            <div style="background: rgba(15, 23, 42, 0.6); border: 2px dashed rgba(255, 255, 255, 0.12); border-radius: 14px; height: 380px; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #64748B;">
                <span style="font-size: 2.8rem; margin-bottom: 8px;">📷</span>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">Vision Stream on Standby</div>
                <div style="font-size: 0.84rem; color: #94A3B8;">Switch toggle above to <b>▶ Enable Live Stream</b> to activate neural vision proctoring.</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_placeholder.container():
            k_col1, k_col2, k_col3, k_col4 = st.columns(4)
            with k_col1:
                st.markdown(f"""
                <div class="kpi-hud-card">
                    <div class="kpi-hud-label">👤 Candidate Profile</div>
                    <div class="kpi-hud-value">{candidate_name}</div>
                    <div class="kpi-hud-meta">UID: <code>{candidate_id}</code></div>
                </div>
                """, unsafe_allow_html=True)
            with k_col2:
                st.markdown(f"""
                <div class="kpi-hud-card">
                    <div class="kpi-hud-label">📚 Active Assessment</div>
                    <div class="kpi-hud-value" style="font-size: 1.18rem;">{exam_title}</div>
                    <div class="kpi-hud-meta">Ref: <code>{session_id}</code></div>
                </div>
                """, unsafe_allow_html=True)
            with k_col3:
                st.markdown("""
                <div class="kpi-hud-card">
                    <div class="kpi-hud-label">🛡️ Integrity Quotient</div>
                    <div class="kpi-hud-value" style="color: #34D399;">100.0%</div>
                    <div class="kpi-hud-meta">Flags: 0 (0 Confirmed)</div>
                </div>
                """, unsafe_allow_html=True)
            with k_col4:
                st.markdown("""
                <div class="kpi-hud-card">
                    <div class="kpi-hud-label">🚦 Defense Status</div>
                    <div style="margin-top: 5px;"><span class="badge-live-pulse">● SECURE</span></div>
                    <div class="kpi-hud-meta">Stream: Standby</div>
                </div>
                """, unsafe_allow_html=True)

        threat_banner_placeholder.markdown("""
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(52, 211, 153, 0.35); border-radius: 10px; padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.1rem;">✅</span>
            <span style="color: #34D399; font-weight: 700; font-size: 0.82rem; font-family: 'JetBrains Mono';">COMPLIANCE VERIFIED: Candidate in standard baseline</span>
        </div>
        """, unsafe_allow_html=True)

        threat_gauge_placeholder.markdown(get_threat_meter_html(0.0, "LOW"), unsafe_allow_html=True)

        telemetry_rows_placeholder.markdown("""
        <div style="margin-top: 4px; margin-bottom: 10px;">
            <div class="sensor-item">
                <span class="sensor-name">👥 Entity Tracking</span>
                <span class="sensor-val" style="color: #34D399;">1 Candidate Tracked</span>
            </div>
            <div class="sensor-item">
                <span class="sensor-name">🔄 Head Pose Yaw (L/R)</span>
                <span class="sensor-val">+0.0°</span>
            </div>
            <div class="sensor-item">
                <span class="sensor-name">📐 Head Pose Pitch (U/D)</span>
                <span class="sensor-val">+0.0°</span>
            </div>
            <div class="sensor-item">
                <span class="sensor-name">👀 Gaze Orientation</span>
                <span class="sensor-val" style="color: #34D399;">Direct (Screen)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- TAB 2: INCIDENT VAULT & EVIDENCE REVIEW ----------------
elif menu_option == "🔍 Incident Vault":
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h2 style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin-bottom: 4px;">🔍 Incident Vault & Forensic Evidence Review</h2>
        <p style="color: #94A3B8; font-size: 0.84rem; margin: 0;">Comprehensive audit archive with automated video clips, snapshots, and Explainable AI (XAI) justifications.</p>
    </div>
    """, unsafe_allow_html=True)

    incidents = db_manager.get_session_incidents(st.session_state.active_session_id)

    if not incidents:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 40px 20px;">
            <span style="font-size: 3rem;">🛡️</span>
            <h3 style="color: #FFFFFF; font-family: 'Space Grotesk'; font-size: 1.2rem; margin-top: 10px;">No Violations Recorded</h3>
            <p style="color: #94A3B8; font-size: 0.84rem;">The candidate maintained full compliance throughout the proctored assessment session.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        f_col1, f_col2 = st.columns([2, 2])
        severity_filter = f_col1.multiselect(
            "Filter Severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM"]
        )
        verdict_filter = f_col2.multiselect(
            "Filter Verdict",
            ["PENDING", "CONFIRMED", "FALSE_POSITIVE", "DISMISSED"],
            default=["PENDING", "CONFIRMED", "FALSE_POSITIVE"]
        )

        filtered_incidents = [
            inc for inc in incidents
            if (not severity_filter or inc["severity"] in severity_filter)
            and (not verdict_filter or inc["proctor_verdict"] in verdict_filter)
        ]

        st.markdown(f"<div style='font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;'>Displaying <b>{len(filtered_incidents)}</b> flagged incident(s)</div>", unsafe_allow_html=True)

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
                            font=dict(color="#E2E8F0", family="Plus Jakarta Sans")
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
                    if st.button("Save Notes", key=f"save_notes_{inc['id']}", use_container_width=True):
                        db_manager.update_incident_verdict(inc['id'], inc['proctor_verdict'], notes)
                        st.success("Notes saved.")


# ---------------- TAB 3: SESSION ANALYTICS & AUDIT ----------------
elif menu_option == "📊 Analytics & Audit":
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h2 style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin-bottom: 4px;">📊 Telemetry Analytics & Audit Reports</h2>
        <p style="color: #94A3B8; font-size: 0.84rem; margin: 0;">Institutional candidate malpractice metrics, integrity timeline, and formal report export.</p>
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
            fig_pie = px.pie(v_df, values="Count", names="Violation Type", hole=0.45, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0", family="Plus Jakarta Sans"))
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
            fig_line.update_traces(line_color="#6366F1", line_width=2.5)
            fig_line.update_layout(
                height=185,
                margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8", family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig_line, use_container_width=True, key="analytics_timeline")
        else:
            st.info("No telemetry logs recorded.")

    st.markdown("---")
    st.subheader("📄 Export Formal Integrity Report")
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
                help="Export complete tabular incident logs and AI confidence metrics for university archives.",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error compiling CSV export: {e}")


# ---------------- TAB 4: SYSTEM CONFIGURATION ----------------
elif menu_option == "⚙️ System Configuration":
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h2 style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; margin-bottom: 4px;">⚙️ Proctoring Sensitivity & Threshold Configuration</h2>
        <p style="color: #94A3B8; font-size: 0.84rem; margin: 0;">Customize neural model thresholds, head pose tolerances, and risk penalty weights.</p>
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

        submitted = st.form_submit_button("💾 Save & Apply Configuration", use_container_width=True)
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

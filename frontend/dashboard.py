"""
EviGuard AI Proctoring Dashboard
Deep Obsidian & Cyan Cyber-Shield Redesign.
Features WebRTC hardware-decoupled streaming, glassmorphism UI, JetBrains/Plus Jakarta Sans typography, and advanced interactive sidebar navigation.
"""

from datetime import datetime
import json
import os
import threading
import time
from typing import Dict, Any, List, Optional
import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import yaml

# Add parent directory to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.db.models import DatabaseManager, ExamSession, Incident, RiskMetricLog
from backend.detection.base import DetectionResult
from backend.explainability.reason_generator import ReasonGenerator
from backend.pipeline import EviGuardPipeline, PipelineOutput


# ---------------- PAGE CONFIGURATION & CYBER-SHIELD THEME ----------------
st.set_page_config(
    page_title="EviGuard - AI Proctoring & Cyber-Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS Injection (Deep Obsidian & Cyan Cyber-Shield with Interactive Sidebar)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Radiant Cyber-Mesh Aurora Theme */
    html, body, [class*="css"], .stApp {
        background: 
            radial-gradient(circle at 15% 15%, rgba(14, 165, 233, 0.20) 0%, transparent 45%),
            radial-gradient(circle at 85% 20%, rgba(99, 102, 241, 0.25) 0%, transparent 50%),
            radial-gradient(circle at 50% 85%, rgba(16, 185, 129, 0.15) 0%, transparent 55%),
            radial-gradient(circle at 80% 80%, rgba(56, 189, 248, 0.18) 0%, transparent 50%),
            linear-gradient(145deg, #090e1a 0%, #0c152a 45%, #050813 100%) !important;
        background-attachment: fixed !important;
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Glassmorphism Containers */
    .cyber-panel {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.82), rgba(11, 18, 33, 0.78)) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 12px 36px 0 rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        margin-bottom: 20px;
    }

    /* Top 4 KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(11, 18, 33, 0.80)) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.20);
        border-top: 3px solid #00f2fe;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card:hover {
        border-color: rgba(0, 242, 254, 0.6);
        transform: translateY(-4px);
        box-shadow: 0 16px 40px 0 rgba(0, 242, 254, 0.22);
    }
    .kpi-card.kpi-exam { border-top-color: #818cf8; }
    .kpi-card.kpi-integrity { border-top-color: #10b981; }
    .kpi-card.kpi-status { border-top-color: #f59e0b; }

    .kpi-title {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.2;
        margin-top: 4px;
        letter-spacing: -0.02em;
    }
    .kpi-subtext {
        font-size: 0.76rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 6px;
    }

    /* Dynamic Integrity Colors */
    .score-green {
        color: #10B981 !important;
        text-shadow: 0 0 18px rgba(16, 185, 129, 0.4);
    }
    .score-yellow {
        color: #F59E0B !important;
        text-shadow: 0 0 18px rgba(245, 158, 11, 0.4);
    }
    .score-red {
        color: #EF4444 !important;
        text-shadow: 0 0 18px rgba(239, 68, 68, 0.5);
    }

    /* Modern Telemetry Grid (Non-clipping values) */
    .telemetry-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .telemetry-item {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 12px;
        padding: 12px 14px;
        text-align: center;
    }
    .telemetry-label {
        font-size: 0.70rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .telemetry-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #00f2fe;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
        white-space: nowrap;
    }

    /* Status Badges */
    .status-badge-compliant {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
    }
    .status-badge-violation {
        display: inline-flex;
        align-items: center;
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.45);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        animation: pulseAlert 2s infinite;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
    }

    @keyframes pulseAlert {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(0.98); }
    }

    /* Button System (Pill Shaped with Cyber Glow) */
    div.stButton > button {
        border-radius: 50px !important;
        font-weight: 700 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95)) !important;
        color: #F9FAFB !important;
        padding: 10px 26px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.82rem !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.45) !important;
        border-color: #00f2fe !important;
        color: #00f2fe !important;
    }

    /* Primary / Download Buttons */
    div.stDownloadButton > button {
        border-radius: 50px !important;
        background: linear-gradient(135deg, #059669, #10b981) !important;
        border: 1px solid rgba(16, 185, 129, 0.5) !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.35) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 10px 26px !important;
        transition: all 0.3s ease !important;
    }
    div.stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 24px rgba(16, 185, 129, 0.6) !important;
    }

    /* Sidebar Glassmorphism & Custom Branding */
    section[data-testid="stSidebar"] {
        background: rgba(8, 12, 20, 0.92) !important;
        backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.14) !important;
    }
    
    .cyber-shield-badge {
        background: rgba(15, 23, 42, 0.85);
        border: 2px solid #38bdf8;
        border-radius: 14px;
        padding: 8px;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        animation: badgePulse 3s infinite ease-in-out;
    }
    @keyframes badgePulse {
        0%, 100% { box-shadow: 0 0 16px rgba(56, 189, 248, 0.35); border-color: #38bdf8; }
        50% { box-shadow: 0 0 24px rgba(0, 242, 254, 0.6); border-color: #00f2fe; }
    }
    .cyber-brand-title {
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .cyber-brand-sub {
        font-size: 0.68rem;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.08em;
    }
    .cyber-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.28);
        color: #38bdf8;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.70rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Sidebar Navigation Tile Pills */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(56, 189, 248, 0.14) !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        margin-bottom: 0px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        transform: translateX(4px) !important;
        border-color: rgba(56, 189, 248, 0.45) !important;
        background: rgba(30, 41, 59, 0.75) !important;
        box-shadow: 0 4px 16px rgba(0, 242, 254, 0.15) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(99, 102, 241, 0.18)) !important;
        border: 1px solid #00f2fe !important;
        box-shadow: 0 0 18px rgba(0, 242, 254, 0.28) !important;
    }

    /* Session Selector Frosted Card */
    .sidebar-session-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px;
        margin-top: 10px;
        margin-bottom: 14px;
    }
    .session-live-pill {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.68rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* System Health Widget */
    .system-health-card {
        background: rgba(10, 15, 29, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 20px;
    }
    .health-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.74rem;
        margin-bottom: 6px;
        color: #94A3B8;
    }
    .health-val {
        color: #F1F5F9;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.72rem;
    }
    .pulse-dot-green {
        display: inline-block;
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 5px;
        box-shadow: 0 0 8px #10B981;
        animation: dotPulse 2s infinite;
    }
    @keyframes dotPulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.6; }
    }

    /* Inputs, Selectboxes, Sliders */
    .stSelectbox div[data-baseweb="select"], .stTextInput input {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 10px !important;
        color: #F1F5F9 !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stTextInput input:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 12px rgba(0, 242, 254, 0.25) !important;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(56, 189, 248, 0.12) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
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


# ---------------- WEBRTC VIDEO PROCESSOR ----------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class ProctorVideoProcessor(VideoProcessorBase):
    """Asynchronous WebRTC video processor running EviGuard AI pipeline in real-time."""

    def __init__(self):
        self.pipeline = get_pipeline()
        self.session_id = "default_session"
        self.candidate_name = "Alex Johnson"
        self.latest_output: Optional[PipelineOutput] = None
        self.lock = threading.Lock()

    def set_session_info(self, session_id: str, candidate_name: str):
        self.session_id = session_id
        self.candidate_name = candidate_name

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Processes each incoming WebRTC video frame through the AI proctoring pipeline."""
        img = frame.to_ndarray(format="bgr24")
        if img is None or img.size == 0:
            return frame

        # Downscale to 640x480 for fast real-time inference
        if img.shape[1] != 640 or img.shape[0] != 480:
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Process through EviGuardPipeline (detection, tracking, pose/gaze, risk scoring, HUD)
        output: PipelineOutput = self.pipeline.process_frame(
            frame=img,
            session_id=self.session_id,
            candidate_name=self.candidate_name
        )

        with self.lock:
            self.latest_output = output

        # Return the annotated frame with HUD back to browser video player
        return av.VideoFrame.from_ndarray(output.annotated_frame, format="bgr24")


# ---------------- HELPER PLOT FUNCTIONS ----------------
def create_gauge_chart(score: float, risk_level: str) -> go.Figure:
    """Renders a modern dark semi-circular Plotly risk gauge with cyan/emerald theme."""
    color_map = {
        "LOW": "#10B981",
        "MEDIUM": "#F59E0B",
        "HIGH": "#EF4444"
    }
    bar_color = color_map.get(risk_level, "#10B981")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Score: {risk_level}", 'font': {'size': 15, 'color': '#CBD5E1', 'family': 'Plus Jakarta Sans'}},
        number={'font': {'size': 32, 'color': '#FFFFFF', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': bar_color, 'thickness': 0.35},
            'bgcolor': "rgba(15, 23, 42, 0.6)",
            'borderwidth': 1,
            'bordercolor': "rgba(56, 189, 248, 0.15)",
            'steps': [
                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.12)"},
                {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.12)"},
                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.18)"},
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 3},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=195, margin=dict(l=15, r=15, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def create_timeline_chart(metrics: List[Dict[str, Any]]) -> go.Figure:
    """Renders dynamic risk evolution line chart with glowing cyan styling."""
    if not metrics:
        fig = go.Figure()
        fig.update_layout(
            title={'text': "Awaiting Telemetry Stream...", 'font': {'color': '#64748B', 'size': 13}},
            height=195,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    df = pd.DataFrame(metrics)
    fig = px.line(
        df,
        x=df.index,
        y="risk_score",
        labels={"x": "Time (Frames)", "risk_score": "Risk Index"},
        title="Continuous Session Risk Timeline"
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Critical (70+)", annotation_font_color="#EF4444")
    fig.add_hline(y=30, line_dash="dot", line_color="#F59E0B", annotation_text="Medium (30+)", annotation_font_color="#F59E0B")
    fig.update_traces(line_color="#00f2fe", line_width=2.5)
    fig.update_layout(
        height=195,
        margin=dict(l=15, r=15, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)", range=[0, 100])
    )
    return fig


# ---------------- SIDEBAR CONTROLS & CYBER-SHIELD NAVIGATION ----------------
with st.sidebar:
    # Branding & Header Section
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-top: 4px; margin-bottom: 6px;">
        <div class="cyber-shield-badge">
            <img src="https://img.icons8.com/fluency/96/shield.png" width="34" height="34" style="display: block;" />
        </div>
        <div>
            <div class="cyber-brand-title">EviGuard AI</div>
            <div class="cyber-brand-sub">ENTERPRISE PROCTOR</div>
        </div>
    </div>
    <div style="margin-top: 6px; margin-bottom: 18px;">
        <span class="cyber-tag">v1.0 • Phase 1 Enterprise Edition</span>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Tiles
    menu_option = st.radio(
        "Navigation Modules",
        ["📹 Live Proctoring", "🔍 Incident Vault", "📊 Analytics & Reports", "⚙️ Settings & Sensitivity"],
        index=0,
        label_visibility="collapsed"
    )

    # Session Selector Card
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
    <div class="sidebar-session-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.74rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">Active Session</span>
            <span class="session-live-pill">● LIVE EXAM</span>
        </div>
    """, unsafe_allow_html=True)

    selected_session = st.selectbox(
        "Active Session Select",
        session_ids if session_ids else [st.session_state.active_session_id],
        index=0 if not session_ids else (session_ids.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_ids else 0),
        label_visibility="collapsed"
    )
    st.session_state.active_session_id = selected_session

    with st.expander("➕ Start New Exam Session"):
        new_s_id = st.text_input("Session ID", f"EXAM_{datetime.now().strftime('%H%M%S')}")
        new_c_id = st.text_input("Candidate ID", "STD-102")
        new_c_name = st.text_input("Candidate Name", "Jane Doe")
        new_exam = st.text_input("Exam Name", "Final Engineering Assessment")
        if st.button("Initialize Session", width="stretch"):
            db_manager.create_session(new_s_id, new_c_id, new_c_name, new_exam)
            st.session_state.active_session_id = new_s_id
            st.success(f"Session {new_s_id} active!")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Bottom System Health Widget
    st.markdown("""
    <div class="system-health-card">
        <div style="font-size: 0.70rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.06em;">System Status & Telemetry</div>
        <div class="health-row">
            <span>AI Engine</span>
            <span class="health-val">YOLOv8 + MediaPipe</span>
        </div>
        <div class="health-row">
            <span>FPS / Stream</span>
            <span class="health-val" style="color: #00f2fe;">30 FPS • WebRTC</span>
        </div>
        <div class="health-row" style="margin-bottom: 0px;">
            <span>Database</span>
            <span class="health-val"><span class="pulse-dot-green"></span>Connected</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- TAB 1: LIVE PROCTORING ----------------
if menu_option == "📹 Live Proctoring":
    current_session = db_manager.get_session_by_id(st.session_state.active_session_id) or {
        "session_id": st.session_state.active_session_id,
        "candidate_id": "STD-101",
        "candidate_name": "Alex Johnson",
        "exam_title": "CS401: Advanced AI Exam",
        "integrity_index": 100.0,
        "status": "ACTIVE"
    }

    # Retrieve session stats
    integrity_score = current_session.get('integrity_index', 100.0)
    score_class = "score-green" if integrity_score >= 80 else ("score-yellow" if integrity_score >= 50 else "score-red")
    
    total_inc = current_session.get('total_incidents', 0)
    is_compliant = (total_inc == 0)

    # Top 4 Sleek Glassmorphism KPI Cards with Glowing Top Accents
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">👤 Candidate Profile</div>
            <div class="kpi-value">{current_session.get('candidate_name', 'Alex Johnson')}</div>
            <div class="kpi-subtext">ID: {current_session.get('candidate_id', 'STD-101')}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card kpi-exam">
            <div class="kpi-title">📚 Exam Session</div>
            <div class="kpi-value" style="font-size: 1.15rem;">{current_session.get('exam_title', 'AI Assessment')}</div>
            <div class="kpi-subtext">Ref: <code>{current_session.get('session_id')}</code></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card kpi-integrity">
            <div class="kpi-title">🛡️ Live Integrity Score</div>
            <div class="kpi-value {score_class}">{integrity_score:.1f}%</div>
            <div class="kpi-subtext">Flagged Incidents: {total_inc}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        badge_html = '<span class="status-badge-compliant">● COMPLIANT</span>' if is_compliant else '<span class="status-badge-violation">● VIOLATION FLAGGED</span>'
        st.markdown(f"""
        <div class="kpi-card kpi-status">
            <div class="kpi-title">🚦 Security Status</div>
            <div style="margin-top: 4px;">{badge_html}</div>
            <div class="kpi-subtext">Engine: WebRTC Live</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # Main Grid Layout: Left 2/3 (Live Stream) and Right 1/3 (Threat Analytics)
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown("""
        <div class="cyber-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">📹 Real-Time Live Feed & AI HUD</span>
                <span style="font-size: 0.78rem; color: #00f2fe; font-family: 'JetBrains Mono'; font-weight: 600;">● ZERO-LATENCY STREAM</span>
            </div>
        """, unsafe_allow_html=True)

        webrtc_ctx = webrtc_streamer(
            key="eviguard-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=ProctorVideoProcessor,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False
            },
            async_processing=True
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.set_session_info(
                session_id=current_session.get("session_id", "default_session"),
                candidate_name=current_session.get("candidate_name", "Alex Johnson")
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="cyber-panel">
            <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">⚡ Real-Time Threat Analysis</div>
        """, unsafe_allow_html=True)
        
        # Read latest risk metrics from WebRTC processor
        latest_risk_score = 0.0
        latest_risk_level = "LOW"
        active_violations = []
        person_count = 1
        yaw_val, pitch_val = 0.0, 0.0

        if 'webrtc_ctx' in locals() and webrtc_ctx.video_processor:
            with webrtc_ctx.video_processor.lock:
                out = webrtc_ctx.video_processor.latest_output
                if out:
                    latest_risk_score = out.risk.smoothed_score
                    latest_risk_level = out.risk.risk_level
                    active_violations = out.risk.active_violations
                    person_count = out.pose_gaze.face_count if out.pose_gaze.face_detected else len(out.detections)
                    yaw_val, pitch_val = out.pose_gaze.yaw, out.pose_gaze.pitch

        # Threat Alert Banner
        if active_violations:
            st.error(f"🚨 **ACTIVE ALERT**: {' • '.join(active_violations)}")
        else:
            st.success("✅ **Integrity Compliant**: Candidate within normal bounds.")

        # Plotly Risk Meter Gauge
        gauge_fig = create_gauge_chart(latest_risk_score, latest_risk_level)
        st.plotly_chart(gauge_fig, width="stretch", key="webrtc_live_gauge")

        # Non-clipping Custom Telemetry Grid
        st.markdown(f"""
        <div class="telemetry-grid">
            <div class="telemetry-item">
                <div class="telemetry-label">👥 Persons</div>
                <div class="telemetry-val">{person_count}</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">🔄 Yaw (L/R)</div>
                <div class="telemetry-val">{yaw_val:+.1f}°</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-label">📐 Pitch (U/D)</div>
                <div class="telemetry-val">{pitch_val:+.1f}°</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Live Session Timeline Chart
        recent_metrics = db_manager.get_session_metrics(st.session_state.active_session_id, limit=50)
        timeline_fig = create_timeline_chart(recent_metrics)
        st.plotly_chart(timeline_fig, width="stretch", key="webrtc_timeline")

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------- TAB 2: INCIDENT VAULT & EVIDENCE REVIEW ----------------
elif menu_option == "🔍 Incident Vault":
    st.markdown("## 🔍 Security Incident Vault & Evidence Review")
    st.caption("Review flagged violations with automated video clips, snapshots, and explainable AI justifications.")

    incidents = db_manager.get_session_incidents(st.session_state.active_session_id)

    if not incidents:
        st.info("No suspicious incidents recorded for this session yet. All clear! 🛡️")
    else:
        # Filter Bar
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

        st.markdown(f"Displaying **{len(filtered_incidents)}** incident(s)")

        for inc in filtered_incidents:
            with st.expander(
                f"🚨 Incident #{inc['id']} | [{inc['severity']}] {inc['violation_type']} at {inc['timestamp']} (Risk: {inc['risk_score']}/100) — Verdict: {inc['proctor_verdict']}",
                expanded=True
            ):
                inc_col1, inc_col2 = st.columns([3, 2], gap="medium")

                with inc_col1:
                    st.markdown(f"#### 📄 {inc['reason_summary']}")
                    st.write(inc['reason_narrative'])

                    # Evidence Media Display
                    if inc.get("evidence_snapshot_path") and os.path.exists(inc["evidence_snapshot_path"]):
                        st.image(inc["evidence_snapshot_path"], caption=f"Violation Snapshot - Frame #{inc['frame_index']}", width="stretch")
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
                            title="Threat Factor Contribution",
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
                        st.plotly_chart(fig_bar, width="stretch", key=f"xai_chart_{inc['id']}")

                    if details.get("recommended_action"):
                        st.info(f"**Recommended Action**: {details['recommended_action']}")

                    # Proctor Verification Action Form
                    st.markdown("---")
                    st.markdown("##### ⚖️ Proctor Decision")
                    v_col1, v_col2, v_col3 = st.columns(3)
                    
                    if v_col1.button("✅ Confirm Violation", key=f"conf_{inc['id']}"):
                        db_manager.update_incident_verdict(inc['id'], "CONFIRMED")
                        st.success("Incident confirmed as cheating violation.")
                        st.rerun()

                    if v_col2.button("⚠️ False Positive", key=f"fp_{inc['id']}"):
                        db_manager.update_incident_verdict(inc['id'], "FALSE_POSITIVE")
                        st.warning("Incident marked as False Positive.")
                        st.rerun()

                    if v_col3.button("❌ Dismiss", key=f"dsm_{inc['id']}"):
                        db_manager.update_incident_verdict(inc['id'], "DISMISSED")
                        st.info("Incident dismissed.")
                        st.rerun()

                    notes = st.text_input("Proctor Audit Notes", value=inc.get("proctor_notes") or "", key=f"notes_{inc['id']}")
                    if st.button("Save Notes", key=f"save_notes_{inc['id']}"):
                        db_manager.update_incident_verdict(inc['id'], inc['proctor_verdict'], notes)
                        st.success("Notes saved.")


# ---------------- TAB 3: SESSION ANALYTICS & REPORTS ----------------
elif menu_option == "📊 Analytics & Reports":
    st.markdown("## 📊 Proctoring Analytics & Session Audit")
    
    current_session = db_manager.get_session_by_id(st.session_state.active_session_id)
    all_sessions = db_manager.get_all_sessions()
    incidents = db_manager.get_session_incidents(st.session_state.active_session_id)
    metrics = db_manager.get_session_metrics(st.session_state.active_session_id, limit=500)

    # Top KPI Cards
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
            st.plotly_chart(fig_pie, width="stretch")
        else:
            st.info("No violations recorded for this candidate.")

    with col_c2:
        st.subheader("Session Integrity Timeline")
        if metrics:
            fig_line = create_timeline_chart(metrics)
            st.plotly_chart(fig_line, width="stretch")
        else:
            st.info("No telemetry logs recorded.")

    # Formal Report Export
    st.markdown("---")
    st.subheader("📄 Export Formal Integrity Report")
    
    report_data = {
        "Session ID": current_session.get("session_id"),
        "Candidate ID": current_session.get("candidate_id"),
        "Candidate Name": current_session.get("candidate_name"),
        "Exam Title": current_session.get("exam_title"),
        "Integrity Index": f"{current_session.get('integrity_index', 100.0):.1f}%",
        "Total Incidents": len(incidents),
        "Peak Risk Score": current_session.get("peak_risk_score", 0.0),
        "Status": current_session.get("status")
    }

    report_md = f"""# EviGuard Academic Integrity Proctoring Report
**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Candidate & Exam Details
- **Candidate Name**: {report_data['Candidate Name']} ({report_data['Candidate ID']})
- **Exam Title**: {report_data['Exam Title']}
- **Session Reference**: `{report_data['Session ID']}`
- **Overall Integrity Index**: **{report_data['Integrity Index']}**
- **Total Flagged Incidents**: {report_data['Total Incidents']}
- **Peak Risk Recorded**: {report_data['Peak Risk Score']}/100

## Incident Summary Table
| Incident ID | Timestamp | Violation Type | Severity | Risk Score | Proctor Verdict | Reason |
|---|---|---|---|---|---|---|
"""
    for inc in incidents:
        report_md += f"| #{inc['id']} | {inc['timestamp']} | {inc['violation_type']} | {inc['severity']} | {inc['risk_score']} | {inc['proctor_verdict']} | {inc['reason_summary']} |\n"

    st.download_button(
        label="📥 Download Integrity Report (Markdown)",
        data=report_md,
        file_name=f"EviGuard_Report_{current_session.get('session_id')}.md",
        mime="text/markdown"
    )


# ---------------- TAB 4: SETTINGS & SENSITIVITY ----------------
elif menu_option == "⚙️ Settings & Sensitivity":
    st.markdown("## ⚙️ Proctoring Sensitivity & Threshold Configuration")
    st.caption("Customize model confidence, gaze tolerance limits, and risk weights dynamically.")

    with st.form("settings_form"):
        st.subheader("1. Object Detection & Tracking Parameters")
        c1, c2 = st.columns(2)
        conf_thresh = c1.slider("YOLOv8 Confidence Threshold", 0.1, 0.9, 0.45, 0.05)
        person_conf = c2.slider("Person Detection Confidence Cutoff", 0.3, 0.9, 0.55, 0.05)

        st.subheader("2. Head Pose & Gaze Limits (Degrees)")
        g1, g2, g3 = st.columns(3)
        yaw_limit = g1.slider("Max Yaw Angle (Turn Left/Right)", 10.0, 60.0, 25.0, 5.0)
        pitch_limit = g2.slider("Max Pitch Angle (Looking Down)", 10.0, 60.0, 20.0, 5.0)
        absence_timeout = g3.slider("Candidate Absence Timeout (Frames)", 10, 150, 30, 5)

        st.subheader("3. Risk Engine Factor Weights")
        r1, r2, r3, r4 = st.columns(4)
        w_phone = r1.slider("Cell Phone Weight", 10.0, 100.0, 50.0, 5.0)
        w_multi = r2.slider("Multiple Persons Weight", 10.0, 100.0, 45.0, 5.0)
        w_absent = r3.slider("Face Absent Weight", 10.0, 100.0, 40.0, 5.0)
        w_gaze = r4.slider("Gaze Deviation Weight", 5.0, 60.0, 25.0, 5.0)

        submitted = st.form_submit_button("💾 Save & Apply Configuration")
        if submitted:
            # Update config file
            updated_cfg = {
                "system": {"app_name": "EviGuard AI", "version": "1.0.0", "inference_stride": 3},
                "detection": {"confidence_threshold": conf_thresh, "imgsz": 320},
                "tracking": {"person_conf_threshold": person_conf, "person_nms_iou": 0.45},
                "pose_gaze": {
                    "head_pose": {"yaw_limit_left": -yaw_limit, "yaw_limit_right": yaw_limit, "pitch_limit_down": -pitch_limit},
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

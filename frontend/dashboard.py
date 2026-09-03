"""
EviGuard AI Proctoring Dashboard
Ultra-Refined Midnight Slate Enterprise UI with Direct VideoProcessor Attribute Polling.
Features real-time 300ms Streamlit fragment polling directly from the VideoProcessor instance for zero-lag gauge, degree metrics, and defense status updates.
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
from backend.pipeline import EviGuardPipeline, PipelineOutput, FreshFrameReader


# ---------------- PAGE CONFIGURATION & REFINED ENTERPRISE THEME ----------------
st.set_page_config(
    page_title="EviGuard - AI Proctoring & Evidence Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast, Cohesive Enterprise Stylesheet
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Midnight Slate Canvas */
    html, body, [class*="css"], .stApp {
        background-color: #0B0F19 !important;
        background-image: radial-gradient(circle at 50% 0%, rgba(79, 70, 229, 0.12) 0%, transparent 60%) !important;
        background-attachment: fixed !important;
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    header[data-testid="stHeader"] {
        background: rgba(11, 15, 25, 0.9) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* Uniform Slate Enterprise Cards */
    .slate-panel {
        background: #151C2C;
        border: 1px solid #283347;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        margin-bottom: 18px;
    }

    /* Top 4 KPI Metric Cards (Uniform Cohesive Styling) */
    .kpi-tile-pro {
        background: #151C2C;
        border: 1px solid #283347;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 110px;
        height: 100%;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .kpi-tile-pro:hover {
        border-color: #4F46E5;
        transform: translateY(-2px);
    }
    .kpi-label-pro {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-value-pro {
        font-size: 1.35rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-top: 4px;
    }
    .kpi-meta-pro {
        font-size: 0.74rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 6px;
    }

    /* Status Score Colors */
    .score-green {
        color: #10B981 !important;
    }
    .score-yellow {
        color: #F59E0B !important;
    }
    .score-red {
        color: #EF4444 !important;
    }

    /* Clean Rounded Status Badges */
    .badge-status-safe {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-status-alert {
        display: inline-flex;
        align-items: center;
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.45);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    /* High-Contrast Telemetry Checklist Rows */
    .telemetry-row-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: #0E1422;
        border: 1px solid #232D3F;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .telemetry-item-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94A3B8;
    }
    .telemetry-item-value {
        font-size: 0.88rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }

    /* WebRTC Video Player Containers */
    div[data-testid="stWebRtc"] video {
        border-radius: 10px !important;
        border: 1px solid #283347 !important;
    }
    div[data-testid="stWebRtc"] button {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 9px 20px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    }

    /* Modern Buttons */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        border: 1px solid #283347 !important;
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        padding: 9px 18px !important;
        transition: all 0.2s ease !important;
        font-size: 0.84rem !important;
    }
    div.stButton > button:hover {
        background-color: #334155 !important;
        border-color: #4F46E5 !important;
        color: #FFFFFF !important;
    }

    /* Primary Accent Download Button */
    div.stDownloadButton > button {
        border-radius: 8px !important;
        background-color: #4F46E5 !important;
        border: 1px solid #6366F1 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #4338CA !important;
    }

    /* Left Sidebar Navigation & Contrast Fix */
    section[data-testid="stSidebar"] {
        background-color: #0E1322 !important;
        border-right: 1px solid #232D3F !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: #151C2C !important;
        border: 1px solid #283347 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin-bottom: 0px !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #1E293B !important;
        border-color: #4F46E5 !important;
        transform: translateX(3px) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #4F46E5, #4338CA) !important;
        border-color: #6366F1 !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label span,
    section[data-testid="stSidebar"] div[role="radiogroup"] label div {
        color: #FFFFFF !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: inline-block !important;
    }

    /* Inputs, Selectboxes */
    .stSelectbox div[data-baseweb="select"], .stTextInput input {
        background-color: #151C2C !important;
        border: 1px solid #283347 !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stTextInput input:focus {
        border-color: #4F46E5 !important;
    }

    .streamlit-expanderHeader {
        background-color: #151C2C !important;
        border: 1px solid #283347 !important;
        border-radius: 10px !important;
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


# ---------------- WEBRTC VIDEO PROCESSOR WITH DIRECT TELEMETRY ATTRIBUTES ----------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}]}
)


class ProctorVideoProcessor(VideoProcessorBase):
    """Asynchronous WebRTC video processor with direct telemetry attribute storage for Streamlit polling."""

    def __init__(self):
        self.pipeline = get_pipeline()
        self.session_id = "default_session"
        self.candidate_name = "Alex Johnson"
        self.latest_output: Optional[PipelineOutput] = None
        self.lock = threading.Lock()

        # Direct Instance Telemetry Attributes for Streamlit Polling
        self.risk_score: float = 0.0
        self.risk_level: str = "LOW"
        self.person_count: int = 1
        self.yaw: float = 0.0
        self.pitch: float = 0.0
        self.roll: float = 0.0
        self.gaze: str = "Direct (Screen)"
        self.flagged: bool = False
        self.active_violations: List[str] = []
        self.phone_detected: bool = False

    def set_session_info(self, session_id: str, candidate_name: str):
        self.session_id = session_id
        self.candidate_name = candidate_name

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if img is None or img.size == 0:
            return frame

        if img.shape[1] != 640 or img.shape[0] != 480:
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Run frame through EviGuard AI Pipeline
        output: PipelineOutput = self.pipeline.process_frame(
            frame=img,
            session_id=self.session_id,
            candidate_name=self.candidate_name
        )

        with self.lock:
            self.latest_output = output
            self.risk_score = float(output.risk.smoothed_score)
            self.risk_level = output.risk.risk_level
            self.person_count = int(output.pose_gaze.face_count if output.pose_gaze.face_detected else len(output.detections))
            self.yaw = float(output.pose_gaze.yaw)
            self.pitch = float(output.pose_gaze.pitch)
            self.roll = float(output.pose_gaze.roll)
            self.gaze = output.pose_gaze.gaze_direction if output.pose_gaze.face_detected else "No Face"
            self.active_violations = list(output.risk.active_violations)
            self.flagged = bool(output.risk.is_incident_triggered or len(output.risk.active_violations) > 0 or output.risk.smoothed_score >= 70.0)
            self.phone_detected = "PHONE_DETECTED" in output.risk.active_violations

        return av.VideoFrame.from_ndarray(output.annotated_frame, format="bgr24")


# ---------------- HELPER PLOT FUNCTIONS ----------------
def create_gauge_chart(score: float, risk_level: str) -> go.Figure:
    """Renders a clean semi-circular Plotly risk meter gauge."""
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
        title={'text': f"Threat Score: {risk_level}", 'font': {'size': 14, 'color': '#94A3B8', 'family': 'Plus Jakarta Sans'}},
        number={'font': {'size': 30, 'color': '#FFFFFF', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': bar_color, 'thickness': 0.35},
            'bgcolor': "#0E1422",
            'borderwidth': 1,
            'bordercolor': "#283347",
            'steps': [
                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.12)"},
                {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.12)"},
                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.18)"},
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 2},
                'thickness': 0.7,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=185, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def create_timeline_chart(metrics: List[Dict[str, Any]]) -> go.Figure:
    """Renders dynamic risk evolution line chart with indigo styling."""
    if not metrics:
        fig = go.Figure()
        fig.update_layout(
            title={'text': "Awaiting Session Telemetry...", 'font': {'color': '#64748B', 'size': 12}},
            height=185,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    df = pd.DataFrame(metrics)
    fig = px.line(
        df,
        x=df.index,
        y="risk_score",
        labels={"x": "Frames", "risk_score": "Risk Index"},
        title="Session Continuous Integrity Timeline"
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Critical (70+)", annotation_font_color="#EF4444")
    fig.add_hline(y=30, line_dash="dot", line_color="#F59E0B", annotation_text="Medium (30+)", annotation_font_color="#F59E0B")
    fig.update_traces(line_color="#6366F1", line_width=2.5)
    fig.update_layout(
        height=185,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", range=[0, 100])
    )
    return fig


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-top: 4px; margin-bottom: 4px;">
        <span style="font-size: 1.6rem;">🛡️</span>
        <div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">EviGuard AI</div>
            <div style="font-size: 0.70rem; font-weight: 600; color: #64748B; letter-spacing: 0.05em; text-transform: uppercase;">Proctoring Studio</div>
        </div>
    </div>
    <div style="margin-top: 6px; margin-bottom: 18px;">
        <span style="background: rgba(79, 70, 229, 0.15); border: 1px solid rgba(79, 70, 229, 0.4); color: #A5B4FC; border-radius: 6px; padding: 3px 8px; font-size: 0.72rem; font-weight: 600; font-family: 'JetBrains Mono';">v1.0 • Enterprise Edition</span>
    </div>
    """, unsafe_allow_html=True)

    menu_option = st.radio(
        "Navigation Modules",
        ["📹 Live Proctoring", "🔍 Incident Vault", "📊 Analytics & Reports", "⚙️ Settings & Sensitivity"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 14px; margin-bottom: 14px; border-top: 1px solid #232D3F;'></div>", unsafe_allow_html=True)

    # Session Selector Container
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
        <span style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #94A3B8;">Active Assessment</span>
        <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: #34D399; border-radius: 12px; padding: 2px 8px; font-size: 0.68rem; font-weight: 700; font-family: 'JetBrains Mono';">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    selected_session = st.selectbox(
        "Session Select",
        session_ids if session_ids else [st.session_state.active_session_id],
        index=0 if not session_ids else (session_ids.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_ids else 0),
        label_visibility="collapsed"
    )
    st.session_state.active_session_id = selected_session

    with st.expander("➕ Initialize New Exam"):
        new_s_id = st.text_input("Session ID", f"EXAM_{datetime.now().strftime('%H%M%S')}")
        new_c_id = st.text_input("Candidate ID", "STD-102")
        new_c_name = st.text_input("Candidate Name", "Jane Doe")
        new_exam = st.text_input("Exam Name", "Final Engineering Assessment")
        if st.button("Start Assessment", width="stretch"):
            db_manager.create_session(new_s_id, new_c_id, new_c_name, new_exam)
            st.session_state.active_session_id = new_s_id
            st.success(f"Session {new_s_id} active!")
            st.rerun()

    st.markdown("<div style='margin-top: 14px; margin-bottom: 14px; border-top: 1px solid #232D3F;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #151C2C; border: 1px solid #283347; border-radius: 10px; padding: 12px 14px; font-size: 0.72rem; line-height: 1.6; color: #94A3B8;">
        <div style="font-weight: 700; color: #64748B; text-transform: uppercase; font-size: 0.68rem; margin-bottom: 4px;">System Diagnostics</div>
        <div style="display: flex; justify-content: space-between;">
            <span>AI Detector</span>
            <span style="color: #F8FAFC; font-family: 'JetBrains Mono'; font-weight: 600;">YOLOv8</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Gaze / 3D Pose</span>
            <span style="color: #F8FAFC; font-family: 'JetBrains Mono'; font-weight: 600;">MediaPipe</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>WebRTC Stream</span>
            <span style="color: #6366F1; font-family: 'JetBrains Mono'; font-weight: 600;">30 FPS Active</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Database</span>
            <span style="color: #10B981; font-weight: 700;">● Connected</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- STREAMLIT FRAGMENTS: DIRECT PROCESSOR POLLING ----------------
@st.fragment(run_every="300ms")
def render_live_top_kpis(webrtc_ctx, session_id: str, candidate_name: str, candidate_id: str, exam_title: str):
    """Polls the VideoProcessor directly every 300ms to update top KPI metrics."""
    risk_score = 0.0
    is_flagged = False

    if webrtc_ctx and webrtc_ctx.video_processor:
        vp = webrtc_ctx.video_processor
        with vp.lock:
            risk_score = float(vp.risk_score)
            is_flagged = bool(vp.flagged)

    incidents = db_manager.get_session_incidents(session_id)
    total_incidents = len(incidents)
    confirmed_incidents = sum(1 for i in incidents if i.get("proctor_verdict") == "CONFIRMED")
    
    integrity_pct = max(0.0, 100.0 - (confirmed_incidents * 5.0) - (total_incidents * 1.5))
    score_color_cls = "score-green" if integrity_pct >= 80 else ("score-yellow" if integrity_pct >= 50 else "score-red")

    if is_flagged or risk_score >= 70.0:
        badge_html = '<span class="badge-status-alert">● SECURITY FLAGGED</span>'
    elif risk_score >= 30.0:
        badge_html = '<span class="badge-status-alert" style="background: rgba(245, 158, 11, 0.2); color: #FCD34D; border-color: rgba(245, 158, 11, 0.4);">● ELEVATED RISK</span>'
    else:
        badge_html = '<span class="badge-status-safe">● ALL CLEAR</span>'

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-tile-pro">
            <div class="kpi-label-pro">👤 Candidate Identity</div>
            <div class="kpi-value-pro">{candidate_name}</div>
            <div class="kpi-meta-pro">ID: {candidate_id}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-tile-pro">
            <div class="kpi-label-pro">📚 Active Assessment</div>
            <div class="kpi-value-pro" style="font-size: 1.15rem;">{exam_title}</div>
            <div class="kpi-meta-pro">Ref: <code>{session_id}</code></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-tile-pro">
            <div class="kpi-label-pro">🛡️ Integrity Quotient</div>
            <div class="kpi-value-pro {score_color_cls}">{integrity_pct:.1f}%</div>
            <div class="kpi-meta-pro">Flags: {total_incidents} ({confirmed_incidents} Confirmed)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-tile-pro">
            <div class="kpi-label-pro">🚦 Defense Status</div>
            <div style="margin-top: 4px;">{badge_html}</div>
            <div class="kpi-meta-pro">Stream: WebRTC Live</div>
        </div>
        """, unsafe_allow_html=True)


@st.fragment(run_every="300ms")
def render_live_threat_panel(webrtc_ctx, session_id: str):
    """Polls the VideoProcessor directly every 300ms to update Plotly gauge and telemetry items."""
    risk_score = 0.0
    risk_level = "LOW"
    active_violations = []
    person_count = 1
    yaw_val = 0.0
    pitch_val = 0.0
    gaze_status = "Direct (Screen)"
    is_flagged = False

    if webrtc_ctx and webrtc_ctx.video_processor:
        vp = webrtc_ctx.video_processor
        with vp.lock:
            risk_score = float(vp.risk_score)
            risk_level = str(vp.risk_level)
            active_violations = list(vp.active_violations)
            person_count = int(vp.person_count)
            yaw_val = float(vp.yaw)
            pitch_val = float(vp.pitch)
            gaze_status = str(vp.gaze)
            is_flagged = bool(vp.flagged)

    st.markdown("""
    <div class="slate-panel">
        <div style="font-size: 1.0rem; font-weight: 700; color: #FFFFFF; margin-bottom: 12px;">Threat Analysis & Telemetry</div>
    """, unsafe_allow_html=True)

    # Dynamic Threat Alert Banner
    if is_flagged or active_violations:
        alert_str = ' • '.join(active_violations) if active_violations else "ELEVATED RISK DETECTED"
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.18); border: 1px solid rgba(248, 113, 113, 0.4); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.1rem;">🚨</span>
            <span style="color: #F87171; font-weight: 700; font-size: 0.82rem;">SECURITY ALERT: {alert_str}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.1rem;">✅</span>
            <span style="color: #34D399; font-weight: 700; font-size: 0.82rem;">COMPLIANCE VERIFIED: Candidate within normal limits</span>
        </div>
        """, unsafe_allow_html=True)

    # Semi-Circular Plotly Risk Meter Gauge directly bound to live risk score
    gauge_fig = create_gauge_chart(risk_score, risk_level)
    st.plotly_chart(gauge_fig, width="stretch", key="webrtc_live_gauge")

    # Live Numerical Telemetry Checklist Rows
    st.markdown(f"""
    <div style="margin-top: 4px; margin-bottom: 12px;">
        <div class="telemetry-row-item">
            <span class="telemetry-item-name">👥 Person Tracking</span>
            <span class="telemetry-item-value">{person_count} Detected</span>
        </div>
        <div class="telemetry-row-item">
            <span class="telemetry-item-name">🔄 Head Pose Yaw (L/R)</span>
            <span class="telemetry-item-value">{yaw_val:+.1f}°</span>
        </div>
        <div class="telemetry-row-item">
            <span class="telemetry-item-name">📐 Head Pose Pitch (U/D)</span>
            <span class="telemetry-item-value">{pitch_val:+.1f}°</span>
        </div>
        <div class="telemetry-row-item">
            <span class="telemetry-item-name">👀 Gaze Orientation</span>
            <span class="telemetry-item-value">{gaze_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Continuous Session Integrity Timeline Chart
    recent_metrics = db_manager.get_session_metrics(session_id, limit=50)
    timeline_fig = create_timeline_chart(recent_metrics)
    st.plotly_chart(timeline_fig, width="stretch", key="webrtc_timeline")

    st.markdown("</div>", unsafe_allow_html=True)


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

    # Layout: Top KPIs container Placeholder
    kpi_placeholder = st.container()

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Main Grid Layout: Left 65% (Live WebRTC Stream), Right 35% (Threat Matrix)
    col_left, col_right = st.columns([13, 7], gap="medium")

    with col_left:
        st.markdown("""
        <div class="slate-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.0rem; font-weight: 700; color: #FFFFFF;">Live Video Stream & AI HUD</span>
                    <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); color: #34D399; border-radius: 12px; padding: 2px 8px; font-size: 0.72rem; font-weight: 700;">● ZERO-LATENCY</span>
                </div>
                <span style="background: #0E1422; border: 1px solid #283347; border-radius: 6px; padding: 2px 8px; font-size: 0.72rem; font-family: 'JetBrains Mono'; color: #94A3B8;">640x480 • 30 FPS</span>
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

    with kpi_placeholder:
        # Render Top 4 KPI Cards polled directly from webrtc_ctx
        render_live_top_kpis(
            webrtc_ctx=webrtc_ctx,
            session_id=current_session.get("session_id", "default_session"),
            candidate_name=current_session.get("candidate_name", "Alex Johnson"),
            candidate_id=current_session.get("candidate_id", "STD-101"),
            exam_title=current_session.get("exam_title", "CS401: Advanced AI Exam")
        )

    with col_right:
        # Render Live Telemetry and Gauge Panel polled directly from webrtc_ctx
        render_live_threat_panel(
            webrtc_ctx=webrtc_ctx,
            session_id=current_session.get("session_id", "default_session")
        )


# ---------------- TAB 2: INCIDENT VAULT & EVIDENCE REVIEW ----------------
elif menu_option == "🔍 Incident Vault":
    st.markdown("## 🔍 Security Incident Vault & Evidence Review")
    st.caption("Review flagged violations with automated video clips, snapshots, and explainable AI justifications.")

    incidents = db_manager.get_session_incidents(st.session_state.active_session_id)

    if not incidents:
        st.info("No suspicious incidents recorded for this session yet. All clear! 🛡️")
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

        st.markdown(f"Displaying **{len(filtered_incidents)}** incident(s)")

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
                        st.image(inc["evidence_snapshot_path"], caption=f"Snapshot - Frame #{inc['frame_index']}", width="stretch")
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
                        st.plotly_chart(fig_bar, width="stretch", key=f"xai_chart_{inc['id']}")

                    if details.get("recommended_action"):
                        st.info(f"**Recommended Action**: {details['recommended_action']}")

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

                    notes = st.text_input("Proctor Notes", value=inc.get("proctor_notes") or "", key=f"notes_{inc['id']}")
                    if st.button("Save Notes", key=f"save_notes_{inc['id']}"):
                        db_manager.update_incident_verdict(inc['id'], inc['proctor_verdict'], notes)
                        st.success("Notes saved.")


# ---------------- TAB 3: SESSION ANALYTICS & REPORTS ----------------
elif menu_option == "📊 Analytics & Reports":
    st.markdown("## 📊 Proctoring Analytics & Session Audit")
    
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
        conf_thresh = c1.slider("YOLOv8 Confidence Threshold", 0.10, 0.90, 0.32, 0.02)
        person_conf = c2.slider("Person Detection Confidence Cutoff", 0.20, 0.90, 0.50, 0.05)

        st.subheader("2. Head Pose & Gaze Limits (Degrees)")
        g1, g2, g3 = st.columns(3)
        yaw_limit = g1.slider("Max Yaw Angle (Turn Left/Right)", 5.0, 45.0, 16.0, 1.0)
        pitch_limit = g2.slider("Max Pitch Angle (Looking Down)", 5.0, 45.0, 14.0, 1.0)
        absence_timeout = g3.slider("Candidate Absence Timeout (Frames)", 5, 120, 15, 5)

        st.subheader("3. Risk Engine Factor Weights")
        r1, r2, r3, r4 = st.columns(4)
        w_phone = r1.slider("Cell Phone Weight", 10.0, 100.0, 85.0, 5.0)
        w_multi = r2.slider("Multiple Persons Weight", 10.0, 100.0, 80.0, 5.0)
        w_absent = r3.slider("Face Absent Weight", 10.0, 100.0, 75.0, 5.0)
        w_gaze = r4.slider("Gaze Deviation Weight", 5.0, 100.0, 45.0, 5.0)

        submitted = st.form_submit_button("💾 Save Configuration")
        if submitted:
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

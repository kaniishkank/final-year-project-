"""
EviGuard AI Proctoring Dashboard
Stripe / Linear-Grade Refined Clean Light UI.
Features WebRTC live proctoring, pure white card architecture, high-contrast typography, and unified telemetry analysis.
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


# ---------------- PAGE CONFIGURATION & STRIPE/LINEAR CLEAN LIGHT THEME ----------------
st.set_page_config(
    page_title="EviGuard - AI Proctoring & Evidence Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Refined Custom CSS for Stripe/Linear Clean Light Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Off-White Canvas Background & Clean Header */
    html, body, [class*="css"], .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    header[data-testid="stHeader"] {
        background-color: #F8FAFC !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }

    /* Structured Pure White Cards */
    .white-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }

    /* Top 4 KPI Metric Cards (Equal Height & Clean Alignment) */
    .kpi-card-white {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 105px;
        height: 100%;
        transition: border-color 0.15s ease;
    }
    .kpi-card-white:hover {
        border-color: #CBD5E1;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }
    .kpi-value {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
        line-height: 1.2;
    }
    .kpi-meta {
        font-size: 12px;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Soft Integrity Scores */
    .score-high-safe {
        color: #10B981 !important;
    }
    .score-mid-warn {
        color: #F59E0B !important;
    }
    .score-low-crit {
        color: #EF4444 !important;
    }

    /* Soft Pastel Status Pills */
    .badge-compliant-pill {
        display: inline-flex;
        align-items: center;
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 700;
    }
    .badge-flagged-pill {
        display: inline-flex;
        align-items: center;
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 700;
    }
    .badge-id-pill {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* Video Player & Streaming Badges */
    .live-stream-badge {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
    }
    .stream-fps-badge {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* WebRTC Stream Styling */
    div[data-testid="stWebRtc"] video {
        border-radius: 10px !important;
        border: 1px solid #CBD5E1 !important;
    }
    div[data-testid="stWebRtc"] button {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 8px 18px !important;
        box-shadow: 0 1px 3px rgba(79, 70, 229, 0.3) !important;
    }
    div[data-testid="stWebRtc"] button:hover {
        background-color: #4338CA !important;
    }

    /* Compliance Banners */
    .compliance-banner-safe {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
        font-weight: 600;
        font-size: 13px;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
    }
    .compliance-banner-alert {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        font-weight: 600;
        font-size: 13px;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
    }

    /* Refined Telemetry Rows */
    .telemetry-row-clean {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #F1F5F9;
    }
    .telemetry-key {
        font-size: 13px;
        font-weight: 500;
        color: #64748B;
    }
    .telemetry-val {
        font-size: 13px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #0F172A;
    }

    /* Refined Action Buttons */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        padding: 8px 16px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease !important;
        font-size: 13px !important;
    }
    div.stButton > button:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        color: #4F46E5 !important;
    }

    /* Primary Accent Download Button */
    div.stDownloadButton > button {
        border-radius: 8px !important;
        background-color: #4F46E5 !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        padding: 9px 20px !important;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.2) !important;
        transition: background-color 0.15s ease !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #4338CA !important;
    }

    /* Crisp White Sidebar & Contrast Fix */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 0px !important;
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #EEF2FF !important;
        border: 1px solid #C7D2FE !important;
        color: #4338CA !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label div:first-child {
        display: none !important;
    }

    /* Inputs, Selectboxes */
    .stSelectbox div[data-baseweb="select"], .stTextInput input {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stTextInput input:focus {
        border-color: #4F46E5 !important;
    }

    /* Accordions */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #0F172A !important;
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
    """Asynchronous WebRTC video processor running EviGuard AI pipeline."""

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
        img = frame.to_ndarray(format="bgr24")
        if img is None or img.size == 0:
            return frame

        # Resize for smooth 30 FPS inference
        if img.shape[1] != 640 or img.shape[0] != 480:
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Process through EviGuardPipeline
        output: PipelineOutput = self.pipeline.process_frame(
            frame=img,
            session_id=self.session_id,
            candidate_name=self.candidate_name
        )

        with self.lock:
            self.latest_output = output

        return av.VideoFrame.from_ndarray(output.annotated_frame, format="bgr24")


# ---------------- HELPER PLOT FUNCTIONS ----------------
def create_gauge_chart(score: float, risk_level: str) -> go.Figure:
    """Renders a clean, minimalist Plotly risk meter gauge for the light theme."""
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
        title={'text': f"Risk Score: {risk_level}", 'font': {'size': 13, 'color': '#64748B', 'family': 'Plus Jakarta Sans'}},
        number={'font': {'size': 28, 'color': '#0F172A', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "#F8FAFC",
            'borderwidth': 1,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.12)"},
                {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.12)"},
                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.12)"},
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 2},
                'thickness': 0.7,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=175, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def create_timeline_chart(metrics: List[Dict[str, Any]]) -> go.Figure:
    """Renders dynamic risk evolution line chart with clean light styling."""
    if not metrics:
        fig = go.Figure()
        fig.update_layout(
            title={'text': "Awaiting Session Telemetry...", 'font': {'color': '#94A3B8', 'size': 12}},
            height=175,
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
        title="Session Risk Index Timeline"
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Alert 70+", annotation_font_color="#EF4444")
    fig.add_hline(y=30, line_dash="dot", line_color="#F59E0B", annotation_text="Warn 30+", annotation_font_color="#F59E0B")
    fig.update_traces(line_color="#4F46E5", line_width=2)
    fig.update_layout(
        height=175,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748B", family="Plus Jakarta Sans"),
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9", range=[0, 100])
    )
    return fig


# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-top: 2px; margin-bottom: 2px;">
        <span style="font-size: 1.35rem;">🛡️</span>
        <span style="font-size: 1.15rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">EviGuard</span>
    </div>
    <div style="font-size: 11px; color: #64748B; margin-bottom: 14px;">Automated Exam Proctoring Suite</div>
    """, unsafe_allow_html=True)

    menu_option = st.radio(
        "Navigation Menu",
        ["📹 Live Proctoring", "🔍 Incident Vault", "📊 Analytics & Reports", "⚙️ Settings & Sensitivity"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

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
    <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #64748B; margin-bottom: 6px;">Active Exam Session</div>
    """, unsafe_allow_html=True)

    selected_session = st.selectbox(
        "Session Select",
        session_ids if session_ids else [st.session_state.active_session_id],
        index=0 if not session_ids else (session_ids.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_ids else 0),
        label_visibility="collapsed"
    )
    st.session_state.active_session_id = selected_session

    with st.expander("➕ Create New Session"):
        new_s_id = st.text_input("Session ID", f"EXAM_{datetime.now().strftime('%H%M%S')}")
        new_c_id = st.text_input("Candidate ID", "STD-102")
        new_c_name = st.text_input("Candidate Name", "Jane Doe")
        new_exam = st.text_input("Exam Name", "Final Engineering Assessment")
        if st.button("Start Session", width="stretch"):
            db_manager.create_session(new_s_id, new_c_id, new_c_name, new_exam)
            st.session_state.active_session_id = new_s_id
            st.success(f"Session {new_s_id} active!")
            st.rerun()

    # Footer System Status
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 11px; color: #64748B; line-height: 1.6;">
        <div>Engine: <span style="font-weight: 600; color: #0F172A;">YOLOv8 + MediaPipe</span></div>
        <div>Stream: <span style="font-weight: 600; color: #0F172A;">WebRTC Live (30 FPS)</span></div>
        <div>Database: <span style="color: #15803D; font-weight: 700;">● Connected</span></div>
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
    score_color_cls = "score-high-safe" if integrity_score >= 80 else ("score-mid-warn" if integrity_score >= 50 else "score-low-crit")
    total_inc = current_session.get('total_incidents', 0)
    is_compliant = (total_inc == 0)

    # Top 4 Clean White KPI Cards (Equal Height Alignment)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card-white">
            <div class="kpi-label">👤 Candidate Profile</div>
            <div class="kpi-value">{current_session.get('candidate_name', 'Alex Johnson')}</div>
            <div class="kpi-meta"><span class="badge-id-pill">ID: {current_session.get('candidate_id', 'STD-101')}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card-white">
            <div class="kpi-label">📚 Subject & Exam</div>
            <div class="kpi-value" style="font-size: 15px;">{current_session.get('exam_title', 'AI Assessment')}</div>
            <div class="kpi-meta">Ref: <code>{current_session.get('session_id')}</code></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card-white">
            <div class="kpi-label">🛡️ Live Integrity Score</div>
            <div class="kpi-value {score_color_cls}">{integrity_score:.1f}%</div>
            <div class="kpi-meta">Flagged Incidents: {total_inc}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        badge_html = '<span class="badge-compliant-pill">● COMPLIANT</span>' if is_compliant else '<span class="badge-flagged-pill">● FLAGGED</span>'
        st.markdown(f"""
        <div class="kpi-card-white">
            <div class="kpi-label">🚦 Session Status</div>
            <div style="margin-top: 4px;">{badge_html}</div>
            <div class="kpi-meta">WebRTC Feed Active</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Main 2-Column Grid: Left 65% (Video Frame), Right 35% (Threat Analysis)
    col_left, col_right = st.columns([13, 7], gap="medium")

    with col_left:
        st.markdown("""
        <div class="white-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 15px; font-weight: 700; color: #0F172A;">Live Video Stream & AI HUD</span>
                    <span class="live-stream-badge">● LIVE STREAMING</span>
                </div>
                <span class="stream-fps-badge">640x480 • 30 FPS</span>
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
        # Read latest risk metrics from WebRTC processor
        latest_risk_score = 0.0
        latest_risk_level = "LOW"
        active_violations = []
        person_count = 1
        yaw_val, pitch_val = 0.0, 0.0
        gaze_status = "Direct (Screen)"

        if 'webrtc_ctx' in locals() and webrtc_ctx.video_processor:
            with webrtc_ctx.video_processor.lock:
                out = webrtc_ctx.video_processor.latest_output
                if out:
                    latest_risk_score = out.risk.smoothed_score
                    latest_risk_level = out.risk.risk_level
                    active_violations = out.risk.active_violations
                    person_count = out.pose_gaze.face_count if out.pose_gaze.face_detected else len(out.detections)
                    yaw_val, pitch_val = out.pose_gaze.yaw, out.pose_gaze.pitch
                    gaze_status = out.pose_gaze.gaze_direction if out.pose_gaze.face_detected else "No Face"

        st.markdown("""
        <div class="white-card">
            <div style="font-size: 15px; font-weight: 700; color: #0F172A; margin-bottom: 12px;">Threat Analysis & Telemetry</div>
        """, unsafe_allow_html=True)

        # Compliance Banner
        if active_violations:
            st.markdown(f"""
            <div class="compliance-banner-alert">
                <span style="font-size: 1.1rem; margin-right: 8px;">🚨</span>
                <span><strong>VIOLATION FLAGGED</strong>: {' • '.join(active_violations)}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="compliance-banner-safe">
                <span style="font-size: 1.1rem; margin-right: 8px;">✅</span>
                <span><strong>COMPLIANT</strong>: Candidate within normal bounds</span>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Minimalist Risk Gauge
        gauge_fig = create_gauge_chart(latest_risk_score, latest_risk_level)
        st.plotly_chart(gauge_fig, width="stretch", key="webrtc_live_gauge")

        # Refined Itemized Telemetry Rows with border-bottom
        st.markdown(f"""
        <div style="margin-top: 4px; margin-bottom: 12px;">
            <div class="telemetry-row-clean">
                <span class="telemetry-key">Person Count</span>
                <span class="telemetry-val">{person_count}</span>
            </div>
            <div class="telemetry-row-clean">
                <span class="telemetry-key">Head Pose Yaw (L/R)</span>
                <span class="telemetry-val">{yaw_val:+.1f}°</span>
            </div>
            <div class="telemetry-row-clean">
                <span class="telemetry-key">Head Pose Pitch (U/D)</span>
                <span class="telemetry-val">{pitch_val:+.1f}°</span>
            </div>
            <div class="telemetry-row-clean" style="border-bottom: none;">
                <span class="telemetry-key">Gaze Direction</span>
                <span class="telemetry-val">{gaze_status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Session Timeline Chart
        recent_metrics = db_manager.get_session_metrics(st.session_state.active_session_id, limit=50)
        timeline_fig = create_timeline_chart(recent_metrics)
        st.plotly_chart(timeline_fig, width="stretch", key="webrtc_timeline")

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------- TAB 2: INCIDENT VAULT & EVIDENCE REVIEW ----------------
elif menu_option == "🔍 Incident Vault":
    st.markdown("### 🔍 Security Incident Vault & Evidence Review")
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
                    st.markdown(f"#### {inc['reason_summary']}")
                    st.write(inc['reason_narrative'])

                    # Evidence Media Display
                    if inc.get("evidence_snapshot_path") and os.path.exists(inc["evidence_snapshot_path"]):
                        st.image(inc["evidence_snapshot_path"], caption=f"Snapshot - Frame #{inc['frame_index']}", width="stretch")
                    elif inc.get("evidence_clip_path") and os.path.exists(inc["evidence_clip_path"]):
                        st.video(inc["evidence_clip_path"])
                    else:
                        st.caption("📸 Snapshot / Clip recorded in evidence archive.")

                with inc_col2:
                    st.markdown("#### 🧠 Explainable AI Attribution")
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
                            height=190,
                            margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#475569", family="Plus Jakarta Sans")
                        )
                        st.plotly_chart(fig_bar, width="stretch", key=f"xai_chart_{inc['id']}")

                    if details.get("recommended_action"):
                        st.info(f"**Recommended Action**: {details['recommended_action']}")

                    # Proctor Decision Controls
                    st.markdown("---")
                    st.markdown("##### Proctor Decision")
                    v_col1, v_col2, v_col3 = st.columns(3)
                    
                    if v_col1.button("✅ Confirm", key=f"conf_{inc['id']}"):
                        db_manager.update_incident_verdict(inc['id'], "CONFIRMED")
                        st.success("Incident confirmed.")
                        st.rerun()

                    if v_col2.button("⚠️ False Positive", key=f"fp_{inc['id']}"):
                        db_manager.update_incident_verdict(inc['id'], "FALSE_POSITIVE")
                        st.warning("Marked as False Positive.")
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
    st.markdown("### 📊 Proctoring Analytics & Session Audit")
    
    current_session = db_manager.get_session_by_id(st.session_state.active_session_id)
    incidents = db_manager.get_session_incidents(st.session_state.active_session_id)
    metrics = db_manager.get_session_metrics(st.session_state.active_session_id, limit=500)

    # Top KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Integrity Index", f"{current_session.get('integrity_index', 100.0):.1f}%")
    with kpi2:
        st.metric("Total Incidents", len(incidents))
    with kpi3:
        st.metric("Peak Risk Score", f"{current_session.get('peak_risk_score', 0.0):.1f}/100")
    with kpi4:
        confirmed_count = sum(1 for i in incidents if i["proctor_verdict"] == "CONFIRMED")
        st.metric("Confirmed Violations", confirmed_count)

    st.markdown("---")

    col_c1, col_c2 = st.columns(2, gap="medium")

    with col_c1:
        st.subheader("Violation Distribution by Type")
        if incidents:
            v_types = [i["violation_type"] for i in incidents]
            v_df = pd.Series(v_types).value_counts().reset_index()
            v_df.columns = ["Violation Type", "Count"]
            fig_pie = px.pie(v_df, values="Count", names="Violation Type", hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#0F172A", family="Plus Jakarta Sans"))
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
    st.subheader("📄 Export Integrity Report")
    
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
    st.markdown("### ⚙️ Proctoring Sensitivity & Threshold Configuration")
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

        submitted = st.form_submit_button("💾 Save Configuration")
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

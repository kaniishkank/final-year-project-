"""
EviGuard AI Proctoring Dashboard
Minimalist Slate & Soft-Zinc Professional Theme.
Features WebRTC live video streaming, flat card architecture, muted enterprise color palette, and streamlined XAI audit controls.
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


# ---------------- PAGE CONFIGURATION & MINIMALIST SLATE THEME ----------------
st.set_page_config(
    page_title="EviGuard - AI Proctoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Minimalist Slate & Soft-Zinc Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Matte Dark Slate Background */
    html, body, [class*="css"], .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Flat Clean Container Cards */
    .slate-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    /* Top 4 KPI Metric Tiles */
    .kpi-tile {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 18px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 105px;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }
    .kpi-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
        line-height: 1.2;
    }
    .kpi-meta {
        font-size: 0.75rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 6px;
    }

    /* Soft Integrity Scores */
    .score-high-safe {
        color: #34D399 !important;
    }
    .score-mid-warn {
        color: #FBBF24 !important;
    }
    .score-low-crit {
        color: #F87171 !important;
    }

    /* Muted Status Badges */
    .badge-safe {
        display: inline-flex;
        align-items: center;
        background-color: #064E3B;
        color: #6EE7B7;
        border: 1px solid #047857;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-alert {
        display: inline-flex;
        align-items: center;
        background-color: #881337;
        color: #FDA4AF;
        border: 1px solid #BE123C;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-id {
        background-color: #334155;
        color: #94A3B8;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Itemized Telemetry Checklist */
    .telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .telemetry-name {
        font-size: 0.82rem;
        font-weight: 500;
        color: #94A3B8;
    }
    .telemetry-value {
        font-size: 0.88rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        color: #F8FAFC;
    }

    /* Minimalist Buttons */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        border: 1px solid #334155 !important;
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        padding: 8px 18px !important;
        transition: background-color 0.2s ease, border-color 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #334155 !important;
        border-color: #475569 !important;
        color: #FFFFFF !important;
    }

    /* Primary Accent Download Button */
    div.stDownloadButton > button {
        border-radius: 8px !important;
        background-color: #4F46E5 !important;
        border: 1px solid #4338CA !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #4338CA !important;
    }

    /* Minimalist Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0B1120 !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        color: #94A3B8 !important;
        transition: background-color 0.15s ease !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-left: 3px solid #6366F1 !important;
        color: #F8FAFC !important;
    }

    /* Inputs, Selects */
    .stSelectbox div[data-baseweb="select"], .stTextInput input {
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover, .stTextInput input:focus {
        border-color: #6366F1 !important;
    }

    /* Accordion Expanders */
    .streamlit-expanderHeader {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
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
    """Renders a clean, minimalist Plotly risk meter gauge."""
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
        title={'text': f"Risk Score: {risk_level}", 'font': {'size': 14, 'color': '#94A3B8', 'family': 'Inter'}},
        number={'font': {'size': 30, 'color': '#F8FAFC', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "#0F172A",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.1)"},
                {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.1)"},
                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.15)"},
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 2},
                'thickness': 0.7,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def create_timeline_chart(metrics: List[Dict[str, Any]]) -> go.Figure:
    """Renders dynamic risk evolution line chart with clean slate styling."""
    if not metrics:
        fig = go.Figure()
        fig.update_layout(
            title={'text': "Awaiting Session Telemetry...", 'font': {'color': '#64748B', 'size': 12}},
            height=180,
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
    fig.update_traces(line_color="#6366F1", line_width=2)
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Inter"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", range=[0, 100])
    )
    return fig


# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
        <span style="font-size: 1.4rem;">🛡️</span>
        <span style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">EviGuard</span>
    </div>
    <div style="font-size: 0.72rem; color: #64748B; margin-bottom: 18px;">Automated Exam Proctoring Suite</div>
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
    <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #94A3B8; margin-bottom: 6px;">Active Exam Session</div>
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
    <div style="font-size: 0.70rem; color: #64748B; line-height: 1.5;">
        <div>Engine: <span style="color: #94A3B8;">YOLOv8 + MediaPipe</span></div>
        <div>Stream: <span style="color: #94A3B8;">WebRTC Live (30 FPS)</span></div>
        <div>Database: <span style="color: #34D399;">● Connected</span></div>
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

    # Top 4 Uniform Flat KPI Tiles
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Candidate</div>
            <div class="kpi-title">{current_session.get('candidate_name', 'Alex Johnson')}</div>
            <div class="kpi-meta"><span class="badge-id">{current_session.get('candidate_id', 'STD-101')}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Subject & Course</div>
            <div class="kpi-title" style="font-size: 1.05rem;">{current_session.get('exam_title', 'AI Assessment')}</div>
            <div class="kpi-meta">Ref: {current_session.get('session_id')}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Live Integrity Score</div>
            <div class="kpi-title {score_color_cls}">{integrity_score:.1f}%</div>
            <div class="kpi-meta">Incidents: {total_inc}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        badge_html = '<span class="badge-safe">● COMPLIANT</span>' if is_compliant else '<span class="badge-alert">● FLAGGED</span>'
        st.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-label">Session Status</div>
            <div style="margin-top: 4px;">{badge_html}</div>
            <div class="kpi-meta">WebRTC Feed Active</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Main 2-Column Grid: Left 65% (Video Frame), Right 35% (Threat Analysis)
    col_left, col_right = st.columns([13, 7], gap="medium")

    with col_left:
        st.markdown("""
        <div class="slate-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC;">Live Video Stream & AI HUD</span>
                <span style="font-size: 0.72rem; color: #64748B; font-family: 'JetBrains Mono';">640x480 • 30 FPS</span>
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
        <div class="slate-card">
            <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-bottom: 10px;">Threat Analysis & Telemetry</div>
        """, unsafe_allow_html=True)

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

        # Alert Banner
        if active_violations:
            st.error(f"🚨 **VIOLATION**: {' • '.join(active_violations)}")
        else:
            st.success("✅ **STATUS SAFE**: Candidate within parameters.")

        # Plotly Minimalist Risk Gauge
        gauge_fig = create_gauge_chart(latest_risk_score, latest_risk_level)
        st.plotly_chart(gauge_fig, width="stretch", key="webrtc_live_gauge")

        # Itemized Telemetry Checklist
        st.markdown(f"""
        <div class="telemetry-row">
            <span class="telemetry-name">👥 Person Count</span>
            <span class="telemetry-value">{person_count}</span>
        </div>
        <div class="telemetry-row">
            <span class="telemetry-name">🔄 Head Pose Yaw (L/R)</span>
            <span class="telemetry-value">{yaw_val:+.1f}°</span>
        </div>
        <div class="telemetry-row">
            <span class="telemetry-name">📐 Head Pose Pitch (U/D)</span>
            <span class="telemetry-value">{pitch_val:+.1f}°</span>
        </div>
        <div class="telemetry-row">
            <span class="telemetry-name">👀 Gaze Tracking</span>
            <span class="telemetry-value">{gaze_status}</span>
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
                            font=dict(color="#94A3B8", family="Inter")
                        )
                        st.plotly_chart(fig_bar, width="stretch", key=f"xai_chart_{inc['id']}")

                    if details.get("recommended_action"):
                        st.info(f"**Recommended Action**: {details['recommended_action']}")

                    # Proctor Verification Action Form
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
            fig_pie = px.pie(v_df, values="Count", names="Violation Type", hole=0.4, color_discrete_sequence=px.colors.sequential.Darkmint)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC", family="Inter"))
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

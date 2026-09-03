"""
EviGuard AI Proctoring Dashboard
Interactive web interface for real-time exam monitoring, evidence clip playback, explainability review, and session auditing.
Features threaded background video capture and in-place DOM updates to eliminate scroll-jumping and lag.
"""

from datetime import datetime
import json
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
from backend.detection.factory import DetectorFactory
from backend.explainability.reason_generator import ReasonGenerator
from backend.pipeline import EviGuardPipeline, PipelineOutput
from backend.pose.pose_gaze import PoseGazeEstimator
from backend.scoring.risk_engine import RiskEngine


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="EviGuard - AI Proctoring & Evidence Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .badge-critical {
        background-color: #EF4444;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #F97316;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #EAB308;
        color: black;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #10B981;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- BUFFERLESS REAL-TIME CAMERA CAPTURE ----------------
class FreshFrameReader:
    """Bufferless real-time video capture reader.
    Continuously runs cap.grab() in a dedicated background thread so only
    the single latest real-time frame is stored and retrieved, eliminating video latency.
    """
    
    def __init__(self, src: int = 0, width: int = 640, height: int = 480):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()

    def start(self, src: int = 0):
        if self.running and self.src == src and self.cap is not None and self.cap.isOpened():
            return
        self.stop()
        self.src = src
        self.running = True
        try:
            # On Windows, try DirectShow for immediate hardware connection
            if sys.platform == "win32":
                self.cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(int(src))
            else:
                self.cap = cv2.VideoCapture(int(src))

            if self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            self.cap = None

        self.thread = threading.Thread(target=self._grab_worker, daemon=True)
        self.thread.start()

    def _grab_worker(self):
        """Continuously pulls latest frames to flush hardware buffer in real-time."""
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                grabbed = self.cap.grab()
                if grabbed:
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None:
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                        with self.lock:
                            self.latest_frame = frame
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.02)

    def read(self) -> Optional[np.ndarray]:
        """Always retrieves single freshest frame and drops all older queued frames."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.5)
            self.thread = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        with self.lock:
            self.latest_frame = None


# ---------------- INITIALIZATION & CACHING ----------------
@st.cache_resource
def get_db_manager():
    return DatabaseManager.get_instance("sqlite:///data/eviguard.db")

@st.cache_resource
def get_pipeline():
    return EviGuardPipeline("config.yaml")

@st.cache_resource
def get_camera():
    return FreshFrameReader(src=0, width=640, height=480)


db_manager = get_db_manager()
pipeline = get_pipeline()
camera = get_camera()


# ---------------- HELPER PLOT FUNCTIONS ----------------
def create_gauge_chart(score: float, risk_level: str) -> go.Figure:
    """Renders a modern semi-circular Plotly risk gauge."""
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
        title={'text': f"Risk Score: {risk_level}", 'font': {'size': 18, 'color': '#1E293B'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "#F1F5F9",
            'borderwidth': 2,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.15)"},
                {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.15)"},
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 4},
                'thickness': 0.8,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=35, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def create_timeline_chart(metrics: List[Dict[str, Any]]) -> go.Figure:
    """Renders dynamic risk evolution line chart."""
    if not metrics:
        fig = go.Figure()
        fig.update_layout(title="No telemetry data recorded yet", height=230)
        return fig

    df = pd.DataFrame(metrics)
    fig = px.line(
        df,
        x=df.index,
        y="risk_score",
        labels={"x": "Time / Frames", "risk_score": "Risk Index (0-100)"},
        title="Continuous Session Risk Timeline"
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Critical Alert (70+)")
    fig.add_hline(y=30, line_dash="dot", line_color="orange", annotation_text="Medium Risk (30+)")
    fig.update_traces(line_color="#4F46E5", line_width=2.5)
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=35, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("EviGuard AI")
    st.markdown("**Intelligent Exam Proctoring & Evidence Verification**")
    st.markdown("---")

    menu_option = st.radio(
        "Navigation",
        ["📹 Live Proctoring", "🔍 Incident Vault", "📊 Analytics & Reports", "⚙️ Settings & Sensitivity"],
        index=0
    )

    st.markdown("---")
    st.subheader("Session Management")

    # Session Selector or Creator
    all_sessions = db_manager.get_all_sessions()
    session_ids = [s["session_id"] for s in all_sessions]

    if "active_session_id" not in st.session_state:
        if session_ids:
            st.session_state.active_session_id = session_ids[0]
        else:
            default_id = f"EXAM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            db_manager.create_session(default_id, "STD-101", "Alex Johnson", "CS401: Advanced AI Exam")
            st.session_state.active_session_id = default_id

    selected_session = st.selectbox(
        "Active Session",
        session_ids if session_ids else [st.session_state.active_session_id],
        index=0 if not session_ids else (session_ids.index(st.session_state.active_session_id) if st.session_state.active_session_id in session_ids else 0)
    )
    st.session_state.active_session_id = selected_session

    with st.expander("➕ Create New Exam Session"):
        new_s_id = st.text_input("Session ID", f"EXAM_{datetime.now().strftime('%H%M%S')}")
        new_c_id = st.text_input("Candidate ID", "STD-102")
        new_c_name = st.text_input("Candidate Name", "Jane Doe")
        new_exam = st.text_input("Exam Name", "Final Engineering Assessment")
        if st.button("Start New Session", width="stretch"):
            db_manager.create_session(new_s_id, new_c_id, new_c_name, new_exam)
            st.session_state.active_session_id = new_s_id
            st.success(f"Session {new_s_id} created!")
            st.rerun()

    st.markdown("---")
    # Clean Stream Toggle in Sidebar
    stream_master_toggle = st.toggle("📹 Camera Streaming Active", value=True, help="Toggle camera stream on/off to inspect charts or logs without live stream re-renders")
    st.caption("EviGuard v1.0.0 | Final Year Project")


# ---------------- TAB 1: LIVE PROCTORING ----------------
if menu_option == "📹 Live Proctoring":
    current_session = db_manager.get_session_by_id(st.session_state.active_session_id) or {
        "session_id": st.session_state.active_session_id,
        "candidate_id": "STD-101",
        "candidate_name": "Alex Johnson",
        "exam_title": "CS401: Advanced AI Exam",
        "integrity_index": 100.0
    }

    # Top Status Banner
    col_t1, col_t2, col_t3, col_t4 = st.columns([3, 2, 2, 2])
    with col_t1:
        st.markdown(f"### 🎯 Candidate: **{current_session.get('candidate_name', 'Alex Johnson')}**")
        st.caption(f"Session: `{current_session.get('session_id')}` | Exam: {current_session.get('exam_title')}")
    with col_t2:
        st.metric("Integrity Index", f"{current_session.get('integrity_index', 100.0):.1f}%", delta=None)
    with col_t3:
        st.metric("Total Flagged Incidents", current_session.get('total_incidents', 0))
    with col_t4:
        st.metric("Status", current_session.get('status', 'ACTIVE'))

    st.markdown("---")

    # Source Selection
    col_src1, col_src2 = st.columns([2, 2])
    video_source_mode = col_src1.radio(
        "Video Stream Source",
        ["📹 Live Physical Webcam (Default: Device 0)", "🧪 Synthetic Avatar Simulator (Demo Mode)"],
        index=0,
        horizontal=True
    )
    cam_index = 0
    if "Webcam" in video_source_mode:
        cam_index = col_src2.number_input("Webcam Device Index", min_value=0, max_value=4, value=0, step=1)

    # Main Live Layout
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Live Video Feed & AI HUD")
        video_placeholder = st.empty()

        sim_phone, sim_multi, sim_lookaway, sim_absent = False, False, False, False
        if "Synthetic" in video_source_mode:
            st.markdown("##### 🧪 Interactive Violation Simulator (Live Demo Mode)")
            sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
            sim_phone = sim_col1.button("📱 Phone Detected")
            sim_multi = sim_col2.button("👥 Multiple People")
            sim_lookaway = sim_col3.button("👀 Looking Away")
            sim_absent = sim_col4.button("🚪 Candidate Absent")

    with col_right:
        st.subheader("Live Threat Analysis")
        alert_placeholder = st.empty()
        gauge_placeholder = st.empty()
        telemetry_placeholder = st.empty()
        timeline_placeholder = st.empty()

    # In-place Dedicated Streaming Loop (No st.rerun() to prevent scroll jumping!)
    if stream_master_toggle:
        if "Webcam" in video_source_mode:
            camera.start(int(cam_index))
        else:
            camera.stop()

        frame_counter = 0
        
        # Initial chart placeholders
        recent_metrics = db_manager.get_session_metrics(st.session_state.active_session_id, limit=50)
        timeline_fig = create_timeline_chart(recent_metrics)
        timeline_placeholder.plotly_chart(timeline_fig, width="stretch", key="init_timeline")

        # Smooth In-Place Update Loop
        while stream_master_toggle:
            frame = None
            if "Webcam" in video_source_mode:
                frame = camera.read()
                if frame is None:
                    time.sleep(0.04)
                    continue
            else:
                # Synthetic Avatar Mode
                frame_h, frame_w = 480, 640
                frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                frame[:] = (35, 30, 40)
                cv2.rectangle(frame, (100, 320), (540, 480), (70, 65, 80), -1)

                injected_dets: List[DetectionResult] = []
                if sim_absent:
                    pass
                else:
                    if sim_lookaway:
                        cv2.circle(frame, (300, 200), 75, (200, 180, 160), -1)
                        cv2.circle(frame, (270, 195), 10, (50, 40, 30), -1)
                        cv2.ellipse(frame, (300, 380), (140, 120), 0, 0, 180, (120, 90, 80), -1)
                    else:
                        cv2.circle(frame, (320, 200), 75, (220, 190, 170), -1)
                        cv2.circle(frame, (295, 195), 10, (50, 40, 30), -1)
                        cv2.circle(frame, (345, 195), 10, (50, 40, 30), -1)
                        cv2.ellipse(frame, (320, 380), (140, 120), 0, 0, 180, (120, 90, 80), -1)

                    injected_dets.append(
                        DetectionResult(box=[180.0, 120.0, 460.0, 460.0], confidence=0.95, class_id=0, class_name="person")
                    )

                if sim_phone:
                    cv2.rectangle(frame, (420, 340), (490, 440), (20, 20, 20), -1)
                    cv2.rectangle(frame, (425, 345), (485, 435), (200, 240, 255), -1)
                    injected_dets.append(
                        DetectionResult(box=[420.0, 340.0, 490.0, 440.0], confidence=0.92, class_id=67, class_name="cell phone")
                    )

                if sim_multi:
                    cv2.circle(frame, (120, 170), 50, (180, 150, 140), -1)
                    cv2.ellipse(frame, (120, 300), (90, 80), 0, 0, 180, (80, 100, 120), -1)
                    injected_dets.append(
                        DetectionResult(box=[40.0, 100.0, 200.0, 380.0], confidence=0.89, class_id=0, class_name="person")
                    )

                if hasattr(pipeline.detector, "set_injected_detections"):
                    pipeline.detector.set_injected_detections(injected_dets)

            if "Webcam" in video_source_mode and hasattr(pipeline.detector, "set_injected_detections"):
                pipeline.detector.set_injected_detections([])

            # Process frame through pipeline
            output: PipelineOutput = pipeline.process_frame(
                frame=frame,
                session_id=st.session_state.active_session_id,
                candidate_name=current_session.get("candidate_name", "Alex Johnson")
            )

            # Update video frame in-place with JPEG compression (fast & lightweight)
            _, encoded_jpeg = cv2.imencode('.jpg', output.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            video_placeholder.image(encoded_jpeg.tobytes(), output_format="JPEG", width="stretch")

            # Update alert badge in-place
            with alert_placeholder.container():
                if output.risk.active_violations:
                    st.error(f"🚨 **ACTIVE ALERT**: {' • '.join(output.risk.active_violations)}")
                    if output.incident_logged:
                        st.warning(f"**Audit Justification**: {output.incident_logged.get('reason_summary')}")
                else:
                    st.success("✅ **Status Normal**: Candidate within compliance limits.")

            # Update telemetry in-place
            with telemetry_placeholder.container():
                t_col1, t_col2, t_col3 = st.columns(3)
                t_col1.metric("Persons Tracked", output.pose_gaze.face_count if output.pose_gaze.face_detected else len(output.detections))
                t_col2.metric("Head Yaw (L/R)", f"{output.pose_gaze.yaw:+.1f}°")
                t_col3.metric("Head Pitch (U/D)", f"{output.pose_gaze.pitch:+.1f}°")

            # Throttled chart rendering every 10 frames (~2-3 times/sec) to keep UI ultra responsive
            if frame_counter % 10 == 0:
                gauge_fig = create_gauge_chart(output.risk.smoothed_score, output.risk.risk_level)
                gauge_placeholder.plotly_chart(gauge_fig, width="stretch", key=f"live_gauge_{frame_counter % 50}")

            if frame_counter % 30 == 0:
                recent_metrics = db_manager.get_session_metrics(st.session_state.active_session_id, limit=50)
                timeline_fig = create_timeline_chart(recent_metrics)
                timeline_placeholder.plotly_chart(timeline_fig, width="stretch", key=f"live_timeline_{frame_counter % 50}")

            frame_counter += 1
            time.sleep(0.03) # Cap at smooth ~25-30 FPS without CPU hogging
    else:
        camera.stop()
        st.info("⏸️ Video stream paused. Turn on '📹 Camera Streaming Active' in the sidebar to resume live monitoring.")


# ---------------- TAB 2: INCIDENT VAULT & EVIDENCE REVIEW ----------------
elif menu_option == "🔍 Incident Vault":
    camera.stop()
    st.markdown("## 🔍 Security Incident Vault & Evidence Review")
    st.markdown("Review flagged violations with automated video clips, snapshots, and explainable AI justifications.")

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
            severity_class = f"badge-{inc['severity'].lower()}"
            with st.expander(
                f"🚨 Incident #{inc['id']} | [{inc['severity']}] {inc['violation_type']} at {inc['timestamp']} (Risk: {inc['risk_score']}/100) - Verdict: {inc['proctor_verdict']}",
                expanded=True
            ):
                inc_col1, inc_col2 = st.columns([3, 2])

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
                            title="Threat Contribution Breakdown",
                            color="Weight %",
                            color_continuous_scale="Reds"
                        )
                        fig_bar.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_bar, width="stretch", key=f"xai_chart_{inc['id']}")

                    if details.get("recommended_action"):
                        st.info(f"**Recommended Action**: {details['recommended_action']}")

                    # Proctor Verification Action
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
    camera.stop()
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

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("Violation Distribution by Type")
        if incidents:
            v_types = [i["violation_type"] for i in incidents]
            v_df = pd.Series(v_types).value_counts().reset_index()
            v_df.columns = ["Violation Type", "Count"]
            fig_pie = px.pie(v_df, values="Count", names="Violation Type", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
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
        label="📥 Download Integrity Report (Markdown/Text)",
        data=report_md,
        file_name=f"EviGuard_Report_{current_session.get('session_id')}.md",
        mime="text/markdown"
    )


# ---------------- TAB 4: SETTINGS & SENSITIVITY ----------------
elif menu_option == "⚙️ Settings & Sensitivity":
    camera.stop()
    st.markdown("## ⚙️ Proctoring Sensitivity & Threshold Configuration")
    st.markdown("Customize model confidence, gaze tolerance limits, and risk weights dynamically.")

    with st.form("settings_form"):
        st.subheader("1. Object Detection Parameters")
        c1, c2 = st.columns(2)
        conf_thresh = c1.slider("YOLOv8 Confidence Threshold", 0.1, 0.9, 0.45, 0.05)
        iou_thresh = c2.slider("IoU NMS Threshold", 0.1, 0.9, 0.45, 0.05)

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

        submitted = st.form_submit_state = st.form_submit_button("💾 Save & Apply Configuration")
        if submitted:
            # Update config file
            updated_cfg = {
                "system": {"app_name": "EviGuard AI", "version": "1.0.0", "inference_stride": 3},
                "detection": {"confidence_threshold": conf_thresh, "iou_threshold": iou_thresh, "imgsz": 320},
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

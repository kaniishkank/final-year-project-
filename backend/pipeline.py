"""
EviGuard Main Processing Pipeline
Integrates Detection, Tracking, Pose/Gaze Estimation, Risk Scoring, XAI, Evidence Recording, and DB logging.
Optimized for zero-lag instant direct triggers, multi-student fair time-slicing, and rapid snapshot archival.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import os
import sys
import threading
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np
import yaml

from .db.models import DatabaseManager, Incident
from .detection.base import DetectionResult
from .detection.factory import DetectorFactory
from .explainability.reason_generator import ReasonGenerator, IncidentExplanation
from .pose.pose_gaze import PoseGazeEstimator, PoseGazeResult
from .scoring.risk_engine import RiskEngine, RiskAssessment
from .tracking.tracker import PersonTracker


@dataclass
class PipelineOutput:
    """Consolidated output from a single frame processing pass."""
    annotated_frame: np.ndarray
    raw_frame: np.ndarray
    detections: List[DetectionResult]
    pose_gaze: PoseGazeResult
    risk: RiskAssessment
    incident_logged: Optional[Dict[str, Any]]
    frame_index: int
    fps: float


class FreshFrameReader:
    """Bufferless real-time video capture reader.
    Runs a dedicated background thread calling cap.grab() to always maintain ONLY
    the single latest camera frame, preventing OpenCV buffer lag.
    """

    def __init__(self, src: Any = 0, width: int = 640, height: int = 480):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()

    def start(self, src: Optional[Any] = None):
        if src is not None:
            self.src = src
        if self.running and self.cap is not None and self.cap.isOpened():
            return
        self.stop()
        self.running = True
        
        try:
            camera_idx = int(self.src) if isinstance(self.src, (int, str)) and str(self.src).isdigit() else self.src
            if isinstance(camera_idx, int) and sys.platform == "win32":
                self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(camera_idx)
            else:
                self.cap = cv2.VideoCapture(camera_idx)

            if self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            self.cap = None

        self.thread = threading.Thread(target=self._grab_worker, daemon=True)
        self.thread.start()

    def _grab_worker(self):
        """Continuously grabs newest frames from camera to flush hardware buffer."""
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
        """Returns the single freshest frame, dropping all stale buffered frames."""
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


class EviGuardPipeline:
    """Master pipeline orchestrating all EviGuard AI components."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Initialize Modules
        det_cfg = self.config.get("detection", {})
        model_type = det_cfg.get("model_type", "yolov8")
        self.detector = DetectorFactory.create_detector(model_type, det_cfg)
        
        self.tracker = PersonTracker(self.config.get("tracking", {}))
        self.pose_gaze = PoseGazeEstimator(self.config.get("pose_gaze", {}))
        self.risk_engine = RiskEngine(self.config.get("risk_engine", {}))
        self.reason_gen = ReasonGenerator(self.config.get("explainability", {}))
        
        db_url = self.config.get("database", {}).get("db_url", "sqlite:///data/eviguard.db")
        self.db = DatabaseManager.get_instance(db_url)

        # Evidence Recorder Settings
        rec_cfg = self.config.get("evidence_recorder", {})
        self.evidence_dir = rec_cfg.get("output_dir", "data/evidence_clips")
        os.makedirs(self.evidence_dir, exist_ok=True)
        self.pre_roll_sec = rec_cfg.get("pre_roll_seconds", 3)
        self.post_roll_sec = rec_cfg.get("post_roll_seconds", 3)
        self.fps_target = self.config.get("video", {}).get("fps", 30)
        self.buffer_size = int((self.pre_roll_sec + self.post_roll_sec + 1) * self.fps_target)
        
        # Circular Rolling Frame Buffer
        self.frame_buffer = deque(maxlen=self.buffer_size)

        # Performance & State Tracking
        self.frame_index = 0
        self.last_time = time.time()
        self.current_fps = 30.0
        
        # Multi-Student Round-Robin State
        self.active_student_ids: List[int] = [1]
        self.student_round_robin_idx = 0
        
        # Temporal Inference Stride for High FPS
        self.inference_stride = self.config.get("system", {}).get("inference_stride", 3)
        self.target_w = self.config.get("video", {}).get("width", 640)
        self.target_h = self.config.get("video", {}).get("height", 480)
        self.last_tracked_detections: List[DetectionResult] = []
        self.last_pose_gaze_result: Optional[PoseGazeResult] = None
        self.last_person_count: int = 1

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads configuration from YAML file or provides robust defaults."""
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}

    def process_frame(
        self,
        frame: np.ndarray,
        session_id: str = "default_session",
        candidate_name: str = "Candidate"
    ) -> PipelineOutput:
        """Executes full analysis pipeline with zero-lag triggers and multi-student fair time-slicing."""
        self.frame_index += 1
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt)

        # 1. Downscale frame to target resolution if needed
        h, w = frame.shape[:2]
        if w != self.target_w or h != self.target_h:
            frame = cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        raw_frame = frame.copy()
        
        # 2. Store frame in circular rolling buffer
        self.frame_buffer.append((frame.copy(), self.frame_index, now))

        # 3. Object Detection & Pose/Gaze with Multi-Student Time-Slicing
        is_inference_frame = (self.frame_index % self.inference_stride == 0) or (self.last_pose_gaze_result is None)

        if is_inference_frame:
            detections = self.detector.detect(frame)
            tracked_detections = self.tracker.update(detections)
            person_count = self.tracker.get_person_count()

            # Multi-student tracking
            person_dets = [d for d in tracked_detections if d.class_name == "person"]
            if person_dets:
                self.active_student_ids = [d.track_id for d in person_dets if d.track_id is not None]
                if not self.active_student_ids:
                    self.active_student_ids = [1]
            else:
                self.active_student_ids = [1]

            # Fair round-robin selection for cropped FaceMesh calculation
            if self.student_round_robin_idx >= len(self.active_student_ids):
                self.student_round_robin_idx = 0
            current_target_id = self.active_student_ids[self.student_round_robin_idx]
            self.student_round_robin_idx = (self.student_round_robin_idx + 1) % len(self.active_student_ids)

            # Find bounding box for the current target student
            target_bbox = None
            for d in person_dets:
                if d.track_id == current_target_id:
                    target_bbox = d.box
                    break
            if target_bbox is None and person_dets:
                target_bbox = person_dets[0].box

            # Run FaceMesh on cropped candidate box for speed
            pose_gaze_result = self.pose_gaze.estimate(frame, bbox=target_bbox)

            # Cache results for intervening frames
            self.last_tracked_detections = tracked_detections
            self.last_pose_gaze_result = pose_gaze_result
            self.last_person_count = person_count
        else:
            tracked_detections = self.last_tracked_detections
            pose_gaze_result = self.last_pose_gaze_result or self.pose_gaze.estimate(frame)
            person_count = self.last_person_count

        # 4. Independent Risk Assessment per Student
        primary_student_id = self.active_student_ids[0] if self.active_student_ids else 1
        risk_result = self.risk_engine.evaluate(
            detections=tracked_detections,
            pose_gaze=pose_gaze_result,
            person_count=person_count,
            student_id=primary_student_id
        )

        # 5. Database Telemetry Logging (throttled every 3 frames)
        if self.frame_index % 3 == 0:
            try:
                self.db.log_metric(
                    session_id=session_id,
                    frame_index=self.frame_index,
                    risk_score=risk_result.smoothed_score,
                    person_count=person_count,
                    phone_detected="PHONE_DETECTED" in risk_result.active_violations,
                    yaw=pose_gaze_result.yaw,
                    pitch=pose_gaze_result.pitch,
                    active_violations=risk_result.active_violations
                )
            except Exception:
                pass

        # 6. Incident Handling & Evidence Recording (Zero-Lag Immediate Snapshot + Async Clip)
        logged_incident_dict = None
        if risk_result.is_incident_triggered:
            explanation = self.reason_gen.generate_explanation(
                risk=risk_result,
                detections=tracked_detections,
                pose_gaze=pose_gaze_result,
                candidate_name=candidate_name,
                timestamp_str=datetime.now().strftime("%H:%M:%S")
            )

            # Write Evidence Snapshot IMMEDIATELY to disk on first alert frame
            snapshot_filename = f"{session_id}_f{self.frame_index}_{int(now)}.jpg"
            snapshot_path = os.path.join(self.evidence_dir, snapshot_filename)
            try:
                cv2.imwrite(snapshot_path, frame)
            except Exception:
                snapshot_path = None

            # Asynchronously encode the pre/post-roll MP4 clip
            clip_filename = f"{session_id}_f{self.frame_index}_{int(now)}.mp4"
            clip_path = os.path.join(self.evidence_dir, clip_filename)
            self._save_evidence_clip_async(clip_path)

            details = {
                "active_violations": risk_result.active_violations,
                "violation_factors": risk_result.violation_factors,
                "factor_attribution": explanation.factor_attribution,
                "pose_gaze": {
                    "yaw": pose_gaze_result.yaw,
                    "pitch": pose_gaze_result.pitch,
                    "roll": pose_gaze_result.roll,
                    "gaze_direction": pose_gaze_result.gaze_direction
                },
                "detections": [d.to_dict() for d in tracked_detections],
                "recommended_action": explanation.recommended_action
            }

            try:
                primary_v = risk_result.primary_violation or "SUSPICIOUS_BEHAVIOR"
                inc_obj = self.db.log_incident(
                    session_id=session_id,
                    frame_index=self.frame_index,
                    violation_type=primary_v.upper(),
                    severity=explanation.severity,
                    risk_score=risk_result.smoothed_score,
                    confidence=explanation.confidence_score,
                    reason_summary=explanation.summary_headline,
                    reason_narrative=explanation.narrative_report,
                    evidence_clip_path=clip_path,
                    evidence_snapshot_path=snapshot_path,
                    details=details
                )
                logged_incident_dict = inc_obj.to_dict()
            except Exception:
                logged_incident_dict = {
                    "violation_type": risk_result.primary_violation or "SECURITY_ALERT",
                    "severity": explanation.severity,
                    "risk_score": risk_result.smoothed_score,
                    "reason_summary": explanation.summary_headline,
                    "reason_narrative": explanation.narrative_report,
                    "evidence_clip_path": clip_path,
                    "evidence_snapshot_path": snapshot_path,
                }

        # 7. Render Visual Annotations and HUD Overlay
        annotated_frame = self._render_hud(
            frame=frame.copy(),
            detections=tracked_detections,
            pose_gaze=pose_gaze_result,
            risk=risk_result,
            person_count=person_count
        )

        return PipelineOutput(
            annotated_frame=annotated_frame,
            raw_frame=raw_frame,
            detections=tracked_detections,
            pose_gaze=pose_gaze_result,
            risk=risk_result,
            incident_logged=logged_incident_dict,
            frame_index=self.frame_index,
            fps=self.current_fps
        )

    def _save_evidence_clip_async(self, output_path: str):
        """Saves current rolling buffer frames into an MP4 clip in a background thread."""
        frames_snapshot = [f[0].copy() for f in list(self.frame_buffer)]
        if not frames_snapshot:
            return

        def _worker(frames: List[np.ndarray], path: str, fps: float):
            try:
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, max(10.0, fps), (w, h))
                for f in frames:
                    writer.write(f)
                writer.release()
            except Exception:
                pass

        threading.Thread(target=_worker, args=(frames_snapshot, output_path, self.current_fps), daemon=True).start()

    def _render_hud(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        pose_gaze: PoseGazeResult,
        risk: RiskAssessment,
        person_count: int
    ) -> np.ndarray:
        """Draws sleek HUD overlays, bounding boxes, gaze vectors, and status cards."""
        h, w = frame.shape[:2]

        # 1. Draw Object Detection Bounding Boxes
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.box]
            cls_name = det.class_name.lower()

            if "phone" in cls_name:
                color = (0, 0, 255)
                label = f"ALERT: Phone ({det.confidence*100:.0f}%)"
            elif "book" in cls_name:
                color = (0, 165, 255)
                label = f"ALERT: Book ({det.confidence*100:.0f}%)"
            elif "person" in cls_name:
                if det.track_id == 1:
                    color = (255, 180, 0) # Cyan for primary candidate
                    label = f"Student #1 (Candidate) ({det.confidence*100:.0f}%)"
                else:
                    color = (0, 0, 255) # Red for unauthorized second person
                    label = f"ALERT: Secondary Person #{det.track_id} ({det.confidence*100:.0f}%)"
            else:
                color = (0, 255, 0)
                label = f"{det.class_name} ({det.confidence*100:.0f}%)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + lw + 8, max(0, y1)), color, -1)
            cv2.putText(frame, label, (x1 + 4, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw 3D Gaze / Head Pose Vector
        if pose_gaze.face_detected and pose_gaze.landmarks_2d:
            nose_pt = pose_gaze.landmarks_2d[0]
            p1 = (int(nose_pt[0]), int(nose_pt[1]))

            if pose_gaze.nose_projection_2d:
                p2 = (int(pose_gaze.nose_projection_2d[0]), int(pose_gaze.nose_projection_2d[1]))
                arrow_color = (0, 0, 255) if pose_gaze.is_looking_away else (0, 255, 0)
                cv2.arrowedLine(frame, p1, p2, arrow_color, 2, tipLength=0.25)

            for pt in pose_gaze.landmarks_2d:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

        # 3. Render Top HUD Banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        if risk.risk_level == "HIGH":
            risk_color = (0, 0, 255)
        elif risk.risk_level == "MEDIUM":
            risk_color = (0, 165, 255)
        else:
            risk_color = (0, 255, 100)

        cv2.putText(frame, "EVIGUARD PROCTOR", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        risk_text = f"RISK: {risk.smoothed_score:.0f}/100 [{risk.risk_level}]"
        cv2.putText(frame, risk_text, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2, cv2.LINE_AA)

        pose_text = f"Yaw:{pose_gaze.yaw:+.1f} Pitch:{pose_gaze.pitch:+.1f}" if pose_gaze.face_detected else "NO FACE"
        cv2.putText(frame, pose_text, (w - 240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        gaze_str = f"Gaze: {pose_gaze.gaze_direction}" if pose_gaze.face_detected else "CANDIDATE ABSENT"
        gaze_col = (0, 0, 255) if (pose_gaze.is_looking_away or pose_gaze.gaze_direction != "CENTER (FOCUSED)" or not pose_gaze.face_detected) else (0, 255, 120)
        cv2.putText(frame, gaze_str, (w - 280, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gaze_col, 2 if gaze_col == (0, 0, 255) else 1, cv2.LINE_AA)

        # 4. Critical Warning Strip if Active Violations
        if risk.active_violations:
            banner_overlay = frame.copy()
            cv2.rectangle(banner_overlay, (0, h - 35), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(banner_overlay, 0.8, frame, 0.2, 0, frame)
            
            warning_text = f"VIOLATION DETECTED: {' | '.join(risk.active_violations)}"
            cv2.putText(frame, warning_text, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

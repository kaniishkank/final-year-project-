"""
EviGuard Telemetry & Probabilistic Integrity Quotient Aggregator
Maintains sliding-window temporal frame counters for anomaly duration verification,
calculates probabilistic cheat likelihoods, and computes the real-time Integrity Quotient.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from config import CONFIG
from detector import DetectionBundle, DetectedObject, HandSignallingResult


@dataclass
class AnomalyEvent:
    """Individual flagged anomaly record with probabilistic likelihood and severity."""
    event_id: int
    timestamp: float
    timestamp_str: str
    anomaly_type: str
    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    cheat_probability: float  # [0.0 - 1.0]
    integrity_penalty: float
    description: str
    trigger_evidence_dump: bool = False


@dataclass
class TelemetryState:
    """Consolidated state emitted on every processed video frame."""
    frame_index: int
    timestamp: float
    integrity_quotient: float  # 0.0 - 100.0
    threat_level: str          # 'NORMAL', 'ELEVATED', 'CRITICAL'
    active_violations: List[str]
    active_cheat_probabilities: Dict[str, float]
    total_incidents: int
    trigger_mp4_dump: bool
    mp4_dump_reason: str
    fps: float


class TelemetryEngine:
    """Real-time sliding-window anomaly tracker and Integrity Quotient aggregator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or CONFIG
        self.mod_a = self.config.get("module_a_devices", {})
        self.mod_b = self.config.get("module_b_signalling", {})
        self.mod_c = self.config.get("module_c_pose_intrusions", {})

        # Sliding-Window Temporal Frame Counters
        self.pose_deviation_counter: int = 0
        self.finger_signalling_counter: int = 0
        self.absence_counter: int = 0
        self.paper_counter: int = 0

        # Persistent Incident Records & Penalties
        self.incidents: List[AnomalyEvent] = []
        self.total_penalties: float = 0.0
        self.cooldown_tracker: Dict[str, float] = {}
        self.cooldown_period_s: float = float(self.config.get("video", {}).get("cooldown_seconds", 3.0))

        # Performance Tracking
        self.frame_count: int = 0
        self.start_time: float = time.time()
        self.fps: float = 30.0
        self._last_fps_calc_time: float = time.time()
        self._fps_frames: int = 0

    def _is_on_cooldown(self, violation_key: str, now: float) -> bool:
        """Checks if a violation type is currently within cooldown to avoid duplicate penalties."""
        last_t = self.cooldown_tracker.get(violation_key, 0.0)
        return (now - last_t) < self.cooldown_period_s

    def _record_cooldown(self, violation_key: str, now: float):
        self.cooldown_tracker[violation_key] = now

    def update(self, bundle: DetectionBundle) -> TelemetryState:
        """Evaluates detection bundle, updates temporal counters, and aggregates integrity quotient."""
        self.frame_count += 1
        self._fps_frames += 1
        now = time.time()

        # Update FPS calculation every 1 second
        if (now - self._last_fps_calc_time) >= 1.0:
            self.fps = round(self._fps_frames / (now - self._last_fps_calc_time), 1)
            self._fps_frames = 0
            self._last_fps_calc_time = now

        active_violations: List[str] = []
        active_probabilities: Dict[str, float] = {}
        trigger_mp4_dump = False
        mp4_dump_reason = ""

        # ---------------- MODULE A: PHYSICAL MATERIALS & DEVICES ----------------
        for obj in bundle.objects:
            if obj.is_malpractice:
                v_type = obj.malpractice_type
                active_violations.append(v_type)

                # Linear probability scaling: 90% - 98% based on detector confidence
                conf = max(0.40, min(1.0, obj.confidence))
                prob = 0.90 + (conf - 0.40) * (0.08 / 0.60)
                prob = round(min(0.98, prob), 3)
                active_probabilities[v_type] = prob

                if not self._is_on_cooldown(v_type, now):
                    self._record_cooldown(v_type, now)
                    penalty = float(self.mod_a.get("integrity_penalty", 30.0))
                    self.total_penalties += penalty

                    event = AnomalyEvent(
                        event_id=len(self.incidents) + 1,
                        timestamp=now,
                        timestamp_str=time.strftime("%H:%M:%S", time.localtime(now)),
                        anomaly_type=v_type,
                        severity="CRITICAL",
                        cheat_probability=prob,
                        integrity_penalty=penalty,
                        description=f"{obj.class_name.upper()} detected with {prob*100:.1f}% malpractice likelihood.",
                        trigger_evidence_dump=True
                    )
                    self.incidents.append(event)
                    trigger_mp4_dump = True
                    mp4_dump_reason = event.description

        # ---------------- MODULE B: MCQ HAND & FINGER SIGNALLING ----------------
        signalling_in_frame = any(h.is_signalling_malpractice for h in bundle.hand_signalling)
        if signalling_in_frame:
            self.finger_signalling_counter += 1
            threshold_frames = int(self.mod_b.get("signalling_hold_frames", 30))
            if self.finger_signalling_counter >= threshold_frames:
                v_type = "SUSPICIOUS_SIGNALLING"
                active_violations.append(v_type)
                prob = 0.58  # 50% - 60% probability
                active_probabilities[v_type] = prob

                if not self._is_on_cooldown(v_type, now):
                    self._record_cooldown(v_type, now)
                    penalty = float(self.mod_b.get("integrity_penalty", 10.0))
                    self.total_penalties += penalty

                    # Extract fingers label
                    sig_label = next((h.gesture_label for h in bundle.hand_signalling if h.is_signalling_malpractice), "Hand Signalling")
                    event = AnomalyEvent(
                        event_id=len(self.incidents) + 1,
                        timestamp=now,
                        timestamp_str=time.strftime("%H:%M:%S", time.localtime(now)),
                        anomaly_type=v_type,
                        severity="WARNING",
                        cheat_probability=prob,
                        integrity_penalty=penalty,
                        description=f"{sig_label} held for {self.finger_signalling_counter} frames (MCQ Answer Signalling).",
                        trigger_evidence_dump=False
                    )
                    self.incidents.append(event)
        else:
            self.finger_signalling_counter = max(0, self.finger_signalling_counter - 2)

        # ---------------- MODULE C: HEAD POSE DEVIATIONS ----------------
        if bundle.head_pose.is_deviated:
            self.pose_deviation_counter += 1
            pose_thresh_frames = int(self.mod_c.get("pose_sustained_frames", 45))
            if self.pose_deviation_counter >= pose_thresh_frames:
                v_type = "PROLONGED_GAZE_MALPRACTICE"
                active_violations.append(f"GAZE: {bundle.head_pose.gaze_direction}")
                prob = 0.88  # 85% - 90% probability
                active_probabilities[v_type] = prob

                if not self._is_on_cooldown(v_type, now):
                    self._record_cooldown(v_type, now)
                    penalty = float(self.mod_c.get("pose_integrity_penalty", 20.0))
                    self.total_penalties += penalty

                    event = AnomalyEvent(
                        event_id=len(self.incidents) + 1,
                        timestamp=now,
                        timestamp_str=time.strftime("%H:%M:%S", time.localtime(now)),
                        anomaly_type=v_type,
                        severity="CRITICAL",
                        cheat_probability=prob,
                        integrity_penalty=penalty,
                        description=f"Sustained lookaway ({bundle.head_pose.gaze_direction}) for {self.pose_deviation_counter} frames.",
                        trigger_evidence_dump=True
                    )
                    self.incidents.append(event)
                    trigger_mp4_dump = True
                    mp4_dump_reason = event.description
        else:
            self.pose_deviation_counter = max(0, self.pose_deviation_counter - 3)

        # ---------------- MODULE C: CANDIDATE ABSENCE ----------------
        if not bundle.face_detected and bundle.person_count == 0:
            self.absence_counter += 1
            absence_thresh = int(self.mod_c.get("absence_frames_threshold", 90))
            if self.absence_counter >= absence_thresh:
                v_type = "SEAT_ABANDONMENT"
                active_violations.append("CANDIDATE ABSENT")
                prob = 0.91  # 85% - 92% probability
                active_probabilities[v_type] = prob

                if not self._is_on_cooldown(v_type, now):
                    self._record_cooldown(v_type, now)
                    penalty = float(self.mod_c.get("absence_integrity_penalty", 20.0))
                    self.total_penalties += penalty

                    event = AnomalyEvent(
                        event_id=len(self.incidents) + 1,
                        timestamp=now,
                        timestamp_str=time.strftime("%H:%M:%S", time.localtime(now)),
                        anomaly_type=v_type,
                        severity="CRITICAL",
                        cheat_probability=prob,
                        integrity_penalty=penalty,
                        description=f"Candidate absent from camera view for {self.absence_counter} frames (~{self.absence_counter/30:.1f}s).",
                        trigger_evidence_dump=True
                    )
                    self.incidents.append(event)
                    trigger_mp4_dump = True
                    mp4_dump_reason = event.description
        else:
            self.absence_counter = 0

        # ---------------- MODULE C: SECONDARY INTRUDER ----------------
        if bundle.person_count > 1 or bundle.face_count > 1:
            v_type = "SECONDARY_PERSON"
            active_violations.append("SECONDARY INTRUDER")
            prob = 0.94
            active_probabilities[v_type] = prob

            if not self._is_on_cooldown(v_type, now):
                self._record_cooldown(v_type, now)
                penalty = float(self.mod_c.get("intruder_integrity_penalty", 30.0))
                self.total_penalties += penalty

                event = AnomalyEvent(
                    event_id=len(self.incidents) + 1,
                    timestamp=now,
                    timestamp_str=time.strftime("%H:%M:%S", time.localtime(now)),
                    anomaly_type=v_type,
                    severity="CRITICAL",
                    cheat_probability=prob,
                    integrity_penalty=penalty,
                    description=f"Multiple individuals detected in exam workspace ({bundle.person_count} persons, {bundle.face_count} faces).",
                    trigger_evidence_dump=True
                )
                self.incidents.append(event)
                trigger_mp4_dump = True
                mp4_dump_reason = event.description

        # Calculate Final Integrity Quotient
        base_iq = float(self.config.get("integrity_quotient", {}).get("base_score", 100.0))
        current_iq = max(0.0, base_iq - self.total_penalties)

        # Threat Level
        if any(self.incidents[-1].severity == "CRITICAL" for _ in [1] if self.incidents and (now - self.incidents[-1].timestamp) < 4.0) or current_iq < 60.0:
            threat_level = "CRITICAL"
        elif active_violations or current_iq < 85.0:
            threat_level = "ELEVATED"
        else:
            threat_level = "NORMAL"

        return TelemetryState(
            frame_index=self.frame_count,
            timestamp=now,
            integrity_quotient=round(current_iq, 1),
            threat_level=threat_level,
            active_violations=active_violations,
            active_cheat_probabilities=active_probabilities,
            total_incidents=len(self.incidents),
            trigger_mp4_dump=trigger_mp4_dump,
            mp4_dump_reason=mp4_dump_reason,
            fps=self.fps
        )

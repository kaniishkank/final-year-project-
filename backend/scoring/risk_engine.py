"""
Dynamic Risk Scoring Engine Module
Aggregates detection, tracking, pose/gaze, and hand/finger signalling signals to compute real-time suspiciousness index.
Configured for Strict Zero-Tolerance High-Security Proctoring with direct triggers for cell phones,
multiple persons, unauthorized paper/notes, candidate absence, prolonged gaze malpractice, and finger signalling.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, Any, List, Optional
from ..detection.base import DetectionResult
from ..pose.pose_gaze import PoseGazeResult


@dataclass
class RiskAssessment:
    """Comprehensive risk evaluation for the current frame."""
    raw_score: float
    smoothed_score: float
    risk_level: str # LOW, MEDIUM, HIGH
    active_violations: List[str]
    violation_factors: Dict[str, float] # Factor weight contributions
    is_incident_triggered: bool
    primary_violation: Optional[str]
    student_id: int = 1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_score": round(self.raw_score, 1),
            "smoothed_score": round(self.smoothed_score, 1),
            "risk_level": self.risk_level,
            "active_violations": self.active_violations,
            "violation_factors": {k: round(v, 1) for k, v in self.violation_factors.items()},
            "is_incident_triggered": self.is_incident_triggered,
            "primary_violation": self.primary_violation,
            "student_id": self.student_id,
            "timestamp": self.timestamp,
        }


class RiskEngine:
    """Computes dynamic risk score based on multi-modal proctoring inputs with instant critical triggers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Strict Zero-Tolerance Risk Weights
        weights_cfg = self.config.get("weights", {})
        self.w_phone = float(weights_cfg.get("cell_phone", 85.0)) # Instant critical alert
        self.w_multiple_persons = float(weights_cfg.get("multiple_persons", 80.0)) # Instant critical intruder alert
        self.w_face_absent = float(weights_cfg.get("face_absent", 75.0)) # Immediate seat abandonment
        self.w_hand_signalling = float(weights_cfg.get("hand_signalling", 75.0)) # Finger/hand gesture cheating
        self.w_gaze = float(weights_cfg.get("gaze_deviation", 45.0)) # Gaze accumulation (+45.0)
        self.w_head_pose = float(weights_cfg.get("head_pose_deviation", 45.0))
        self.w_suspicious_object = float(weights_cfg.get("suspicious_object", 40.0))
        self.w_prolonged_gaze = float(weights_cfg.get("prolonged_gaze_malpractice", 85.0))

        # Fast Temporal Parameters (~0.8s responsive window)
        self.decay_rate = float(self.config.get("decay_rate", 0.88))
        self.accumulation_rate = float(self.config.get("accumulation_rate", 0.55))

        # Thresholds
        thresholds_cfg = self.config.get("thresholds", {})
        self.low_max = float(thresholds_cfg.get("low_max", 30.0))
        self.medium_max = float(thresholds_cfg.get("medium_max", 70.0))
        self.high_threshold = float(thresholds_cfg.get("high_threshold", 70.0))

        # Multi-Student Independent Risk Histories
        self.student_histories: Dict[int, float] = {}
        self.student_last_incidents: Dict[int, float] = {}
        self.current_smoothed_score: float = 0.0
        self.last_incident_time: float = 0.0
        self.incident_cooldown_seconds: float = 3.0

    def evaluate(
        self,
        detections: List[DetectionResult],
        pose_gaze: PoseGazeResult,
        person_count: int,
        student_id: int = 1
    ) -> RiskAssessment:
        """Evaluates proctoring signals with instant triggers for critical threats, hand gestures, and prolonged gaze."""
        active_violations: List[str] = []
        factors: Dict[str, float] = {}
        is_critical_direct_trigger = False

        # 1. Instant Trigger: Unauthorized mobile phones (confidence >= 0.50)
        phone_detections = [
            d for d in detections 
            if d.class_name in ("cell phone", "phone") and d.confidence >= 0.50
        ]
        if phone_detections:
            active_violations.append("PHONE_DETECTED")
            factors["cell_phone"] = self.w_phone
            is_critical_direct_trigger = True

        # 2. Strict Real Secondary Person Validation (require genuine secondary presence)
        if person_count > 1 or pose_gaze.face_count > 1:
            active_violations.append("MULTIPLE_PERSONS")
            factors["multiple_persons"] = self.w_multiple_persons
            is_critical_direct_trigger = True

        # 3. Unauthorized Paper / Study Notes / Book Detection (confidence >= 0.40)
        paper_detections = [
            d for d in detections 
            if d.class_name in ("book", "notes", "unauthorized paper/notes", "paper") and d.confidence >= 0.40
        ]
        if paper_detections:
            active_violations.append("UNAUTHORIZED PAPER/NOTES")
            factors["suspicious_object"] = self.w_suspicious_object + 20.0
            if any(d.confidence >= 0.60 for d in paper_detections):
                is_critical_direct_trigger = True

        # 4. Candidate absence (missing face)
        if pose_gaze.is_absent or (person_count == 0 and not pose_gaze.face_detected):
            active_violations.append("FACE_ABSENT")
            factors["face_absent"] = self.w_face_absent
            if pose_gaze.absence_frames >= 15:
                is_critical_direct_trigger = True

        # 5. Hand / Finger Signalling Malpractice (e.g. signaling 1-4 fingers to communicate answers)
        if getattr(pose_gaze, "hand_signalling", False) or getattr(pose_gaze, "hand_gesture_label", ""):
            label = getattr(pose_gaze, "hand_gesture_label", "FINGER SIGNALLING") or "FINGER SIGNALLING"
            active_violations.append(f"FLAG: {label}")
            factors["hand_signalling"] = self.w_hand_signalling
            is_critical_direct_trigger = True

        # 6. Prolonged Gaze Malpractice (continuous sustained deviation > 2.0s / ~45-60 frames)
        if pose_gaze.face_detected and (
            pose_gaze.is_prolonged_lookaway or 
            getattr(pose_gaze, "gaze_violation_frames", 0) >= 45 or
            getattr(pose_gaze, "gaze_violation_seconds", 0.0) >= 2.0
        ):
            direction = pose_gaze.gaze_direction
            seconds = max(2.0, round(getattr(pose_gaze, "gaze_violation_seconds", 2.0), 1))
            malpractice_label = f"CRITICAL_MALPRACTICE: Sustained Gaze Deviation ({direction}) for {seconds}s"
            active_violations.append(malpractice_label)
            factors["prolonged_gaze_malpractice"] = self.w_prolonged_gaze
            is_critical_direct_trigger = True

        # 7. Standard 4-Way Gaze & Head Pose Deviations (LEFT, RIGHT, DOWN, UP)
        elif pose_gaze.face_detected and (pose_gaze.is_looking_away or ("CENTER" not in pose_gaze.gaze_direction.upper())):
            gaze_dir = pose_gaze.gaze_direction.upper()
            if "LOOKING LEFT" in gaze_dir or "LEFT" in gaze_dir:
                active_violations.append("HEAD_TURN (LEFT)")
                factors["head_pose_deviation"] = self.w_head_pose
            elif "LOOKING RIGHT" in gaze_dir or "RIGHT" in gaze_dir:
                active_violations.append("HEAD_TURN (RIGHT)")
                factors["head_pose_deviation"] = self.w_head_pose
            elif "LOOKING DOWN" in gaze_dir or "DOWN" in gaze_dir:
                active_violations.append("GAZE_DOWN (DESK/LAP)")
                factors["gaze_deviation"] = self.w_gaze + 5.0
            elif "LOOKING UP" in gaze_dir or "UP" in gaze_dir:
                active_violations.append("GAZE_AWAY (UP)")
                factors["gaze_deviation"] = self.w_gaze
            elif "HEAD TILTED" in gaze_dir or "TILT" in gaze_dir:
                active_violations.append("HEAD_TILTED")
                factors["head_pose_deviation"] = self.w_head_pose * 0.8
            else:
                active_violations.append("GAZE_DEVIATION")
                factors["gaze_deviation"] = self.w_gaze

        # Compute raw instantaneous score
        raw_score = min(100.0, float(sum(factors.values())))

        # Retrieve student specific temporal history
        student_smoothed = self.student_histories.get(student_id, 0.0)

        # Apply Instant Bypass for Critical Violations or Fast Smoothing
        if is_critical_direct_trigger and raw_score >= 70.0:
            student_smoothed = max(student_smoothed, raw_score)
            if factors.get("hand_signalling"):
                student_smoothed = max(student_smoothed, 85.0)
        else:
            if raw_score > student_smoothed:
                student_smoothed = (
                    (1.0 - self.accumulation_rate) * student_smoothed + 
                    self.accumulation_rate * raw_score
                )
            else:
                student_smoothed = student_smoothed * self.decay_rate
                if student_smoothed < 1.0:
                    student_smoothed = 0.0

        self.student_histories[student_id] = student_smoothed
        self.current_smoothed_score = student_smoothed
        smoothed_score = round(student_smoothed, 1)

        # Classify Risk Level
        if smoothed_score <= self.low_max:
            risk_level = "LOW"
        elif smoothed_score <= self.medium_max:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Determine Immediate Incident Trigger
        now = time.time()
        last_incident = self.student_last_incidents.get(student_id, 0.0)
        is_incident_triggered = False
        primary_violation = None

        if active_violations:
            primary_violation = max(factors.keys(), key=lambda k: factors[k], default=active_violations[0])
            
            should_trigger = is_critical_direct_trigger or (smoothed_score >= self.high_threshold)
            
            if should_trigger and (now - last_incident) >= self.incident_cooldown_seconds:
                is_incident_triggered = True
                self.student_last_incidents[student_id] = now
                self.last_incident_time = now

        return RiskAssessment(
            raw_score=raw_score,
            smoothed_score=smoothed_score,
            risk_level=risk_level,
            active_violations=active_violations,
            violation_factors=factors,
            is_incident_triggered=is_incident_triggered,
            primary_violation=primary_violation,
            student_id=student_id,
            timestamp=now
        )

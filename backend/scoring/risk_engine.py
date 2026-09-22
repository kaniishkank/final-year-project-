"""
Dynamic Risk & Threat Scoring Engine Module
Computes real-time threat scores and risk levels across standardized bands:
- Phone detected -> CRITICAL, score >= 90
- Notes detected -> CRITICAL, score >= 90
- Student writing (looking down) -> NORMAL, score <= 10
- Hand raised + signalling -> SUSPICIOUS, 60 <= score <= 70
- Gaze diversion -> MONITOR, 30 <= score <= 40
- No violations -> NORMAL, score = 0
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
    risk_level: str  # NORMAL, MONITOR, SUSPICIOUS, CRITICAL
    active_violations: List[str]
    violation_factors: Dict[str, float]
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
    """Computes dynamic threat scores and classifies incidents into NORMAL, MONITOR, SUSPICIOUS, and CRITICAL."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Calibrated Threat Weights
        weights_cfg = self.config.get("weights", {})
        self.w_phone = float(weights_cfg.get("cell_phone", 95.0))             # >= 90 CRITICAL
        self.w_notes = float(weights_cfg.get("unauthorized_notes", 92.0))      # >= 90 CRITICAL
        self.w_multiple_persons = float(weights_cfg.get("multiple_persons", 90.0))
        self.w_face_absent = float(weights_cfg.get("face_absent", 85.0))
        self.w_hand_signalling = float(weights_cfg.get("hand_signalling", 65.0)) # 60-70 SUSPICIOUS
        self.w_gaze = float(weights_cfg.get("gaze_deviation", 35.0))           # 30-40 MONITOR
        self.w_head_pose = float(weights_cfg.get("head_pose_deviation", 35.0))

        # Temporal Parameters
        self.decay_rate = float(self.config.get("decay_rate", 0.88))
        self.accumulation_rate = float(self.config.get("accumulation_rate", 0.55))

        # Thresholds
        thresholds_cfg = self.config.get("thresholds", {})
        self.normal_max = float(thresholds_cfg.get("normal_max", 25.0))
        self.monitor_max = float(thresholds_cfg.get("monitor_max", 50.0))
        self.suspicious_max = float(thresholds_cfg.get("suspicious_max", 75.0))
        self.critical_min = float(thresholds_cfg.get("critical_min", 75.0))

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
        """Evaluates proctoring signals against the threat scoring rubric."""
        active_violations: List[str] = []
        factors: Dict[str, float] = {}
        is_critical_direct_trigger = False

        # 1. Phone Detection (score >= 90 -> CRITICAL)
        phone_detections = [
            d for d in detections 
            if d.class_name in ("cell phone", "phone") and d.confidence >= 0.35
        ]
        if phone_detections:
            active_violations.append("PHONE_DETECTED")
            factors["cell_phone"] = self.w_phone
            is_critical_direct_trigger = True

        # 2. Unauthorized Notes / Book Detection (score >= 90 -> CRITICAL)
        notes_detections = [
            d for d in detections 
            if d.class_name in ("book", "notes", "unauthorized paper/notes", "paper") and d.confidence >= 0.35
        ]
        if notes_detections:
            active_violations.append("UNAUTHORIZED_NOTES")
            factors["unauthorized_notes"] = self.w_notes
            is_critical_direct_trigger = True

        # 3. Secondary Person Detection (Intruder -> CRITICAL)
        if person_count > 1 or getattr(pose_gaze, "face_count", 1) > 1:
            active_violations.append("MULTIPLE_PERSONS")
            factors["multiple_persons"] = self.w_multiple_persons
            is_critical_direct_trigger = True

        # 4. Candidate Absence
        if pose_gaze.is_absent or (person_count == 0 and not pose_gaze.face_detected):
            active_violations.append("FACE_ABSENT")
            factors["face_absent"] = self.w_face_absent
            if getattr(pose_gaze, "absence_frames", 0) >= 15:
                is_critical_direct_trigger = True

        # 5. Hand Raised + Signalling (60 <= score <= 70 -> SUSPICIOUS)
        if getattr(pose_gaze, "hand_signalling", False) or getattr(pose_gaze, "hand_gesture_label", ""):
            label = getattr(pose_gaze, "hand_gesture_label", "FINGER SIGNALLING") or "FINGER SIGNALLING"
            active_violations.append(f"FLAG: {label}")
            factors["hand_signalling"] = self.w_hand_signalling

        # 6. Prolonged Gaze Malpractice (continuous sustained deviation > 2.0s / ~45 frames)
        if pose_gaze.face_detected and not is_critical_direct_trigger and (
            pose_gaze.is_prolonged_lookaway or 
            getattr(pose_gaze, "gaze_violation_frames", 0) >= 45 or
            getattr(pose_gaze, "gaze_violation_seconds", 0.0) >= 2.0
        ):
            direction = pose_gaze.gaze_direction or "LOOKING AWAY"
            seconds = max(2.0, round(getattr(pose_gaze, "gaze_violation_seconds", 2.0), 1))
            malpractice_label = f"CRITICAL_MALPRACTICE: Sustained Gaze Deviation ({direction}) for {seconds}s"
            active_violations.append(malpractice_label)
            factors["prolonged_gaze_malpractice"] = float(self.config.get("weights", {}).get("prolonged_gaze_malpractice", 85.0))
            is_critical_direct_trigger = True

        # 7. Pose & Gaze Classification:
        # Check if looking down: Is student legitimately writing on desk/paper?
        elif pose_gaze.face_detected and not is_critical_direct_trigger and not factors.get("hand_signalling"):
            gaze_dir = (pose_gaze.gaze_direction or "CENTER (FOCUSED)").upper()
            
            if "LOOKING DOWN" in gaze_dir or pose_gaze.pitch > 12.0:
                # Student writing legitimately: Score <= 10, Normal
                factors["student_writing"] = 0.0  # Safe normal behavior
            elif pose_gaze.is_looking_away or ("CENTER" not in gaze_dir):
                # Lateral gaze diversion (looking left, looking right, looking up)
                if "LEFT" in gaze_dir:
                    active_violations.append("HEAD_TURN (LEFT)")
                elif "RIGHT" in gaze_dir:
                    active_violations.append("HEAD_TURN (RIGHT)")
                elif "UP" in gaze_dir:
                    active_violations.append("GAZE_AWAY (UP)")
                else:
                    active_violations.append("GAZE_DEVIATION")
                factors["gaze_deviation"] = self.w_gaze  # 30-40 MONITOR range

        # Compute raw instantaneous score
        raw_score = min(100.0, float(sum(factors.values())))

        # Retrieve student specific temporal history
        student_smoothed = self.student_histories.get(student_id, 0.0)

        # Apply Instant Trigger for High Severity or Smooth Normal Transition
        if is_critical_direct_trigger:
            student_smoothed = max(student_smoothed, raw_score)
        elif factors.get("hand_signalling"):
            student_smoothed = self.w_hand_signalling
        elif factors.get("gaze_deviation"):
            student_smoothed = self.w_gaze
        elif not factors or factors.get("student_writing", -1) == 0.0:
            student_smoothed = 0.0
        else:
            if raw_score > student_smoothed:
                student_smoothed = (1.0 - self.accumulation_rate) * student_smoothed + self.accumulation_rate * raw_score
            else:
                student_smoothed = student_smoothed * self.decay_rate
                if student_smoothed < 1.0:
                    student_smoothed = 0.0

        self.student_histories[student_id] = student_smoothed
        self.current_smoothed_score = student_smoothed
        smoothed_score = round(student_smoothed, 1)

        # Map to Threat Level: NORMAL (<=25), MONITOR (26-50), SUSPICIOUS (51-75), CRITICAL (>=76)
        if smoothed_score <= self.normal_max:
            risk_level = "NORMAL"
        elif smoothed_score <= self.monitor_max:
            risk_level = "MONITOR"
        elif smoothed_score <= self.suspicious_max:
            risk_level = "SUSPICIOUS"
        else:
            risk_level = "CRITICAL"

        # Incident Trigger evaluation
        now = time.time()
        last_incident = self.student_last_incidents.get(student_id, 0.0)
        is_incident_triggered = False
        primary_violation = None

        if active_violations:
            primary_violation = max(factors.keys(), key=lambda k: factors[k], default=active_violations[0])
            should_trigger = is_critical_direct_trigger or (smoothed_score >= self.critical_min)
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

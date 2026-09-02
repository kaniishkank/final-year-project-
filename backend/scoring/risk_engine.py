"""
Dynamic Risk Scoring Engine Module
Aggregates detection, tracking, and pose/gaze signals to compute real-time suspiciousness index.
Implements temporal smoothing (EMA) and incident trigger logic.
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
            "timestamp": self.timestamp,
        }


class RiskEngine:
    """Computes dynamic risk score based on multi-modal proctoring inputs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Risk Weights
        weights_cfg = self.config.get("weights", {})
        self.w_phone = weights_cfg.get("cell_phone", 50.0)
        self.w_multiple_persons = weights_cfg.get("multiple_persons", 45.0)
        self.w_face_absent = weights_cfg.get("face_absent", 40.0)
        self.w_head_pose = weights_cfg.get("head_pose_deviation", 25.0)
        self.w_gaze = weights_cfg.get("gaze_deviation", 20.0)
        self.w_suspicious_object = weights_cfg.get("suspicious_object", 30.0)

        # Temporal Parameters
        self.decay_rate = self.config.get("decay_rate", 0.92)
        self.accumulation_rate = self.config.get("accumulation_rate", 0.30)

        # Thresholds
        thresholds_cfg = self.config.get("thresholds", {})
        self.low_max = thresholds_cfg.get("low_max", 30.0)
        self.medium_max = thresholds_cfg.get("medium_max", 70.0)
        self.high_threshold = thresholds_cfg.get("high_threshold", 70.0)

        # Internal State
        self.current_smoothed_score: float = 0.0
        self.last_incident_time: float = 0.0
        self.incident_cooldown_seconds: float = 4.0 # Minimum delay between triggering new incident records

    def evaluate(
        self,
        detections: List[DetectionResult],
        pose_gaze: PoseGazeResult,
        person_count: int
    ) -> RiskAssessment:
        """Evaluates all current signals and computes instantaneous and temporal risk.
        
        Args:
            detections: List of active object detections.
            pose_gaze: Result from pose and gaze estimator.
            person_count: Number of tracked persons.
            
        Returns:
            RiskAssessment object with risk scores and violation breakdown.
        """
        active_violations: List[str] = []
        factors: Dict[str, float] = {}

        # 1. Check unauthorized electronic devices (cell phones, laptops)
        phone_detections = [d for d in detections if d.class_name in ("cell phone", "phone")]
        if phone_detections:
            active_violations.append("PHONE_DETECTED")
            factors["cell_phone"] = self.w_phone

        # 2. Check unauthorized study materials (books, notes)
        book_detections = [d for d in detections if d.class_name in ("book", "notes")]
        if book_detections:
            active_violations.append("SUSPICIOUS_OBJECT")
            factors["suspicious_object"] = self.w_suspicious_object

        # 3. Check multiple persons
        if person_count > 1 or pose_gaze.face_count > 1:
            active_violations.append("MULTIPLE_PERSONS")
            factors["multiple_persons"] = self.w_multiple_persons

        # 4. Check candidate absence
        if pose_gaze.is_absent or (person_count == 0 and not pose_gaze.face_detected):
            active_violations.append("FACE_ABSENT")
            factors["face_absent"] = self.w_face_absent

        # 5. Check gaze and head pose deviations
        elif pose_gaze.face_detected and pose_gaze.is_looking_away:
            if "LOOKING_DOWN" in pose_gaze.gaze_direction:
                active_violations.append("GAZE_DOWN (DESK)")
                factors["gaze_deviation"] = self.w_gaze + 5.0 # Extra penalty for looking down
            elif "LOOKING_LEFT" in pose_gaze.gaze_direction or "LOOKING_RIGHT" in pose_gaze.gaze_direction:
                active_violations.append("GAZE_AWAY (SIDE)")
                factors["head_pose_deviation"] = self.w_head_pose

        # Compute raw instantaneous score (clamped to 100.0)
        raw_score = min(100.0, float(sum(factors.values())))

        # Compute smoothed temporal score
        if raw_score > self.current_smoothed_score:
            # Escalation
            self.current_smoothed_score = (
                (1.0 - self.accumulation_rate) * self.current_smoothed_score + 
                self.accumulation_rate * raw_score
            )
        else:
            # Decay towards zero
            self.current_smoothed_score = self.current_smoothed_score * self.decay_rate
            if self.current_smoothed_score < 1.0:
                self.current_smoothed_score = 0.0

        smoothed_score = round(self.current_smoothed_score, 1)

        # Classify Risk Level
        if smoothed_score <= self.low_max:
            risk_level = "LOW"
        elif smoothed_score <= self.medium_max:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Determine Incident Trigger
        now = time.time()
        is_incident_triggered = False
        primary_violation = None

        if active_violations:
            # Sort violations by severity/factor weight
            primary_violation = max(factors.keys(), key=lambda k: factors[k], default=active_violations[0])
            
            # Critical triggers (phone or multiple persons) or high score
            is_critical = ("cell_phone" in factors or "multiple_persons" in factors or smoothed_score >= self.high_threshold)
            
            if is_critical and (now - self.last_incident_time) >= self.incident_cooldown_seconds:
                is_incident_triggered = True
                self.last_incident_time = now

        return RiskAssessment(
            raw_score=raw_score,
            smoothed_score=smoothed_score,
            risk_level=risk_level,
            active_violations=active_violations,
            violation_factors=factors,
            is_incident_triggered=is_incident_triggered,
            primary_violation=primary_violation,
            timestamp=now
        )

    def reset(self):
        """Resets engine state."""
        self.current_smoothed_score = 0.0
        self.last_incident_time = 0.0

"""
Explainability & Natural Language Reason Generator Module
Transforms multi-modal model signals into human-interpretable incident explanations and audit justifications.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from ..detection.base import DetectionResult
from ..pose.pose_gaze import PoseGazeResult
from ..scoring.risk_engine import RiskAssessment


@dataclass
class IncidentExplanation:
    """Structured and natural language audit explanation for an incident."""
    summary_headline: str
    narrative_report: str
    severity: str
    factor_attribution: Dict[str, float]
    confidence_score: float
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_headline": self.summary_headline,
            "narrative_report": self.narrative_report,
            "severity": self.severity,
            "factor_attribution": {k: round(v, 2) for k, v in self.factor_attribution.items()},
            "confidence_score": round(self.confidence_score, 2),
            "recommended_action": self.recommended_action,
        }


class ReasonGenerator:
    """Generates explainable AI justifications for proctors and academic integrity auditors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def generate_explanation(
        self,
        risk: RiskAssessment,
        detections: List[DetectionResult],
        pose_gaze: PoseGazeResult,
        candidate_name: str = "Candidate",
        timestamp_str: Optional[str] = None
    ) -> IncidentExplanation:
        """Constructs audit-grade natural language explanation and factor attributions.
        
        Args:
            risk: Risk evaluation from RiskEngine.
            detections: List of detected objects.
            pose_gaze: Pose & Gaze state.
            candidate_name: Name of the student.
            timestamp_str: Formatted time string.
            
        Returns:
            IncidentExplanation object.
        """
        if not timestamp_str:
            timestamp_str = datetime.now().strftime("%H:%M:%S")

        factors = risk.violation_factors
        total_factor_weight = sum(factors.values()) if factors else 1.0
        
        # Calculate percentage attribution
        attribution = {
            k: (v / total_factor_weight) * 100.0 for k, v in factors.items()
        }

        # Build headline and narrative
        headlines: List[str] = []
        sentences: List[str] = []
        actions: List[str] = []
        confidences: List[float] = []

        # 1. Device Detections (Phone)
        phones = [d for d in detections if d.class_name in ("cell phone", "phone")]
        if phones:
            p_conf = max(p.confidence for p in phones)
            confidences.append(p_conf)
            headlines.append("Unauthorized Electronic Device (Phone)")
            sentences.append(
                f"An unauthorized mobile device was detected in {candidate_name}'s workspace "
                f"with {p_conf * 100:.1f}% model confidence."
            )
            actions.append("Issue immediate warning; request student to place phone out of reach.")

        # 2. Paper & Study Material Detections
        papers = [d for d in detections if d.class_name in ("book", "notes", "unauthorized paper/notes", "paper")]
        if papers:
            b_conf = max(b.confidence for b in papers)
            confidences.append(b_conf)
            headlines.append("Unauthorized Paper / Cheat Notes Detected")
            sentences.append(
                f"Physical notes or unauthorized paper material was detected on the desk surface "
                f"(confidence: {b_conf * 100:.1f}%)."
            )
            actions.append("Request a clear 360-degree camera pan of desk surface.")

        # 3. Multiple Persons
        if "multiple_persons" in factors or pose_gaze.face_count > 1:
            headlines.append("Multiple Persons in Frame")
            sentences.append(
                f"Multiple individuals ({pose_gaze.face_count} faces tracked) were detected in the exam environment."
            )
            actions.append("Verify room occupancy and confirm only the registered candidate is present.")

        # 4. Absence
        if "face_absent" in factors or pose_gaze.is_absent:
            headlines.append("Candidate Absence / Obscured Face")
            sentences.append(
                f"{candidate_name}'s face was absent or fully obscured from camera view "
                f"for over {pose_gaze.absence_frames} consecutive frames."
            )
            actions.append("Check if candidate left the workstation or if camera connection was blocked.")

        # 5. Prolonged Gaze Malpractice (>2.0s)
        if "prolonged_gaze_malpractice" in factors:
            sec_val = max(2.0, round(getattr(pose_gaze, "gaze_violation_seconds", 2.0), 1))
            headlines.append(f"Malpractice: Sustained Gaze/Eye Deviation ({pose_gaze.gaze_direction}) for {sec_val}s")
            sentences.append(
                f"Candidate maintained prolonged gaze deviation ({pose_gaze.gaze_direction}) "
                f"continuously for {sec_val}s without refocusing on the exam display."
            )
            actions.append("Flag for academic review; instruct student to look directly at the screen.")

        # 6. Transient Gaze / Head Pose Deviations
        elif "head_pose_deviation" in factors or "gaze_deviation" in factors:
            if "LOOKING DOWN" in pose_gaze.gaze_direction:
                headlines.append("Sustained Downward Gaze (Desk/Lap)")
                sentences.append(
                    f"Candidate exhibited downward head pitch ({pose_gaze.pitch:.1f}°) "
                    f"indicating attention focused below the screen."
                )
            elif "LOOKING LEFT" in pose_gaze.gaze_direction or "LOOKING RIGHT" in pose_gaze.gaze_direction:
                direction = "left" if "LOOKING LEFT" in pose_gaze.gaze_direction else "right"
                headlines.append(f"Gaze Deviation (Turned {direction.capitalize()})")
                sentences.append(
                    f"Candidate turned head significantly to the {direction} (Yaw: {pose_gaze.yaw:.1f}°) "
                    f"away from the primary display."
                )
            if not actions:
                actions.append("Monitor gaze timeline; issue verbal prompt to look directly at the screen.")

        # Fallback if empty
        if not headlines:
            headlines.append("Suspicious Behavioral Anomaly")
            sentences.append(f"Elevated risk score ({risk.smoothed_score}/100) recorded at {timestamp_str}.")
            actions.append("Review video playback for context.")

        summary_headline = " + ".join(headlines[:2])
        narrative_report = (
            f"At {timestamp_str}, an integrity alert was registered for {candidate_name} "
            f"with an aggregate risk score of {risk.smoothed_score}/100. " + " ".join(sentences)
        )
        recommended_action = " | ".join(actions[:2])
        overall_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.85

        # Determine severity level
        if "cell_phone" in factors or "multiple_persons" in factors or "prolonged_gaze_malpractice" in factors or risk.smoothed_score >= 80.0:
            severity = "CRITICAL"
        elif risk.smoothed_score >= 60.0:
            severity = "HIGH"
        elif risk.smoothed_score >= 30.0:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        return IncidentExplanation(
            summary_headline=summary_headline,
            narrative_report=narrative_report,
            severity=severity,
            factor_attribution=attribution,
            confidence_score=overall_confidence,
            recommended_action=recommended_action
        )

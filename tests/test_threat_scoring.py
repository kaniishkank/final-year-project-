"""
Unit Tests for Threat Scoring Engine
Verifies all 6 threat conditions and scoring bounds:
- Phone detected -> CRITICAL, score >= 90
- Notes detected -> CRITICAL, score >= 90
- Student writing (looking down) -> NORMAL, score <= 10
- Hand raised + signalling -> SUSPICIOUS, 60 <= score <= 70
- Gaze diversion 15s -> MONITOR, 30 <= score <= 40
- No violations -> NORMAL, score = 0
"""

import pytest
from backend.detection.base import DetectionResult
from backend.pose.pose_gaze import PoseGazeResult
from backend.scoring.risk_engine import RiskEngine


@pytest.fixture
def risk_engine():
    """Initializes standard RiskEngine with calibrated thresholds."""
    return RiskEngine()


def test_phone_detected_threat(risk_engine):
    """Verify that cell phone detection triggers CRITICAL threat level with score >= 90."""
    phone_det = [
        DetectionResult(
            box=[120.0, 150.0, 190.0, 290.0],
            confidence=0.88,
            class_id=67,
            class_name="cell phone"
        )
    ]
    pose = PoseGazeResult(
        face_detected=True,
        face_count=1,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        gaze_direction="CENTER (FOCUSED)",
        is_looking_away=False,
        is_absent=False,
        absence_frames=0
    )

    assessment = risk_engine.evaluate(phone_det, pose, person_count=1)
    
    assert assessment.risk_level == "CRITICAL"
    assert assessment.smoothed_score >= 90.0
    assert "PHONE_DETECTED" in assessment.active_violations
    assert assessment.is_incident_triggered is True


def test_notes_detected_threat(risk_engine):
    """Verify that unauthorized notes/books trigger CRITICAL threat level with score >= 90."""
    notes_det = [
        DetectionResult(
            box=[200.0, 300.0, 350.0, 420.0],
            confidence=0.85,
            class_id=73,
            class_name="unauthorized paper/notes"
        )
    ]
    pose = PoseGazeResult(
        face_detected=True,
        face_count=1,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        gaze_direction="CENTER (FOCUSED)",
        is_looking_away=False,
        is_absent=False,
        absence_frames=0
    )

    assessment = risk_engine.evaluate(notes_det, pose, person_count=1)
    
    assert assessment.risk_level == "CRITICAL"
    assert assessment.smoothed_score >= 90.0
    assert "UNAUTHORIZED_NOTES" in assessment.active_violations
    assert assessment.is_incident_triggered is True


def test_student_writing_looking_down_threat(risk_engine):
    """Verify that student writing (looking down at desk paper without unauthorized materials) is NORMAL with score <= 10."""
    pose = PoseGazeResult(
        face_detected=True,
        face_count=1,
        yaw=0.0,
        pitch=18.0,  # Looking down at exam paper
        roll=0.0,
        gaze_direction="LOOKING DOWN",
        is_looking_away=False,
        is_absent=False,
        absence_frames=0
    )

    assessment = risk_engine.evaluate([], pose, person_count=1)
    
    assert assessment.risk_level == "NORMAL"
    assert assessment.smoothed_score <= 10.0
    assert assessment.is_incident_triggered is False


def test_hand_raised_signalling_threat(risk_engine):
    """Verify that hand raised + finger signalling triggers SUSPICIOUS threat level with 60 <= score <= 70."""
    pose = PoseGazeResult(
        face_detected=True,
        face_count=1,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        gaze_direction="CENTER (FOCUSED)",
        is_looking_away=False,
        is_absent=False,
        absence_frames=0,
        hand_signalling=True,
        extended_fingers=2,
        hand_gesture_label="FINGER SIGNALLING (2 Extended Fingers)"
    )

    assessment = risk_engine.evaluate([], pose, person_count=1)
    
    assert assessment.risk_level == "SUSPICIOUS"
    assert 60.0 <= assessment.smoothed_score <= 70.0
    assert any("FLAG" in v or "SIGNALLING" in v for v in assessment.active_violations)


def test_gaze_diversion_threat(risk_engine):
    """Verify that lateral gaze diversion triggers MONITOR threat level with 30 <= score <= 40."""
    pose = PoseGazeResult(
        face_detected=True,
        face_count=1,
        yaw=22.0,  # Turned head to the right
        pitch=0.0,
        roll=0.0,
        gaze_direction="LOOKING RIGHT",
        is_looking_away=True,
        is_absent=False,
        absence_frames=0
    )

    assessment = risk_engine.evaluate([], pose, person_count=1)
    
    assert assessment.risk_level == "MONITOR"
    assert 30.0 <= assessment.smoothed_score <= 40.0
    assert any("RIGHT" in v or "GAZE" in v for v in assessment.active_violations)


def test_no_violations_normal_threat(risk_engine):
    """Verify that when no violations are present, threat level is NORMAL with score = 0."""
    pose = PoseGazeResult(
        face_detected=True,
        face_count=1,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        gaze_direction="CENTER (FOCUSED)",
        is_looking_away=False,
        is_absent=False,
        absence_frames=0
    )

    assessment = risk_engine.evaluate([], pose, person_count=1)
    
    assert assessment.risk_level == "NORMAL"
    assert assessment.smoothed_score == 0.0
    assert len(assessment.active_violations) == 0
    assert assessment.is_incident_triggered is False

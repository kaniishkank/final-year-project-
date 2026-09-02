"""
Unit and Integration Tests for EviGuard AI Proctoring System
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from backend.db.models import DatabaseManager, get_engine_and_session_factory
from backend.detection.base import DetectionResult
from backend.detection.factory import DetectorFactory
from backend.detection.yolov8_detector import MockDetector
from backend.explainability.reason_generator import ReasonGenerator
from backend.pipeline import EviGuardPipeline
from backend.pose.pose_gaze import PoseGazeEstimator, PoseGazeResult
from backend.scoring.risk_engine import RiskEngine
from backend.tracking.tracker import PersonTracker


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_eviguard.db")
    db_url = f"sqlite:///{db_path}"
    db = DatabaseManager(db_url)
    yield db
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_detection_result_properties():
    det = DetectionResult(
        box=[100.0, 150.0, 300.0, 450.0],
        confidence=0.92,
        class_id=0,
        class_name="person"
    )
    assert det.width == 200.0
    assert det.height == 300.0
    assert det.center == [200.0, 300.0]
    assert det.area == 60000.0
    d_dict = det.to_dict()
    assert d_dict["class_name"] == "person"
    assert d_dict["confidence"] == 0.92


def test_detector_factory():
    mock_det = DetectorFactory.create_detector("mock")
    assert isinstance(mock_det, MockDetector)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = mock_det.detect(dummy_frame)
    assert len(results) >= 1
    assert results[0].class_name == "person"


def test_person_tracker():
    tracker = PersonTracker({"max_disappeared_frames": 5, "iou_distance_threshold": 0.3})
    
    # Frame 1: Candidate appears
    det1 = [DetectionResult(box=[100.0, 100.0, 200.0, 200.0], confidence=0.9, class_id=0, class_name="person")]
    tracked1 = tracker.update(det1)
    assert len(tracked1) == 1
    assert tracked1[0].track_id == 1
    assert tracker.get_person_count() == 1

    # Frame 2: Candidate slightly moved
    det2 = [DetectionResult(box=[105.0, 102.0, 205.0, 202.0], confidence=0.91, class_id=0, class_name="person")]
    tracked2 = tracker.update(det2)
    assert len(tracked2) == 1
    assert tracked2[0].track_id == 1

    # Frame 3: Second person enters
    det3 = [
        DetectionResult(box=[105.0, 102.0, 205.0, 202.0], confidence=0.91, class_id=0, class_name="person"),
        DetectionResult(box=[400.0, 100.0, 500.0, 200.0], confidence=0.88, class_id=0, class_name="person")
    ]
    tracked3 = tracker.update(det3)
    assert len(tracked3) == 2
    assert tracker.get_person_count() == 2


def test_pose_gaze_estimator():
    estimator = PoseGazeEstimator({
        "head_pose": {"yaw_limit_left": -25.0, "yaw_limit_right": 25.0, "pitch_limit_down": -20.0},
        "face_absence": {"absence_frames_threshold": 2}
    })

    # Empty frame should report absent
    res_absent1 = estimator.estimate(None)
    assert res_absent1.face_detected is False

    res_absent2 = estimator.estimate(None)
    assert res_absent2.is_absent is True

    # Test classification logic
    direction, looking_away = estimator._classify_gaze(yaw=0.0, pitch=0.0, roll=0.0)
    assert "CENTER" in direction
    assert looking_away is False

    dir_down, look_down = estimator._classify_gaze(yaw=0.0, pitch=-30.0, roll=0.0)
    assert "LOOKING_DOWN" in dir_down
    assert look_down is True


def test_risk_engine():
    engine = RiskEngine({
        "weights": {"cell_phone": 50.0, "multiple_persons": 45.0, "face_absent": 40.0},
        "thresholds": {"low_max": 30.0, "medium_max": 70.0, "high_threshold": 70.0}
    })

    # 1. Normal state
    dummy_pose = PoseGazeResult(
        face_detected=True, face_count=1, yaw=0.0, pitch=0.0, roll=0.0,
        gaze_direction="CENTER", is_looking_away=False, is_absent=False, absence_frames=0
    )
    assessment1 = engine.evaluate([], dummy_pose, person_count=1)
    assert assessment1.risk_level == "LOW"
    assert len(assessment1.active_violations) == 0

    # 2. Violation: Phone detected
    phone_det = [DetectionResult(box=[10.0, 10.0, 50.0, 50.0], confidence=0.9, class_id=67, class_name="cell phone")]
    assessment2 = engine.evaluate(phone_det, dummy_pose, person_count=1)
    assert "PHONE_DETECTED" in assessment2.active_violations
    assert assessment2.raw_score >= 50.0
    assert assessment2.is_incident_triggered is True


def test_reason_generator():
    generator = ReasonGenerator()
    engine = RiskEngine()
    phone_det = [DetectionResult(box=[10.0, 10.0, 50.0, 50.0], confidence=0.92, class_id=67, class_name="cell phone")]
    dummy_pose = PoseGazeResult(
        face_detected=True, face_count=1, yaw=0.0, pitch=-25.0, roll=0.0,
        gaze_direction="LOOKING_DOWN (DESK/PHONE)", is_looking_away=True, is_absent=False, absence_frames=0
    )
    risk = engine.evaluate(phone_det, dummy_pose, person_count=1)

    explanation = generator.generate_explanation(risk, phone_det, dummy_pose, candidate_name="Test Student")
    assert "Phone" in explanation.summary_headline or "Unauthorized" in explanation.summary_headline
    assert "Test Student" in explanation.narrative_report
    assert explanation.severity in ("CRITICAL", "HIGH")
    assert len(explanation.factor_attribution) > 0


def test_database_manager(temp_db):
    db = temp_db
    session = db.create_session("S101", "C001", "Alice Smith", "Math Exam")
    assert session.session_id == "S101"

    # Log metric
    db.log_metric("S101", 1, 15.0, 1, False, 2.0, -1.0, [])
    metrics = db.get_session_metrics("S101")
    assert len(metrics) == 1
    assert metrics[0]["risk_score"] == 15.0

    # Log incident
    inc = db.log_incident(
        session_id="S101",
        frame_index=1,
        violation_type="PHONE_DETECTED",
        severity="HIGH",
        risk_score=75.0,
        confidence=0.94,
        reason_summary="Phone detected in workspace",
        reason_narrative="A phone was detected with 94% confidence.",
        evidence_snapshot_path=None
    )
    assert inc.id is not None
    incidents = db.get_session_incidents("S101")
    assert len(incidents) == 1
    assert incidents[0]["violation_type"] == "PHONE_DETECTED"

    # Update verdict
    updated = db.update_incident_verdict(inc.id, "CONFIRMED", "Confirmed on video.")
    assert updated is True
    incidents_after = db.get_session_incidents("S101")
    assert incidents_after[0]["proctor_verdict"] == "CONFIRMED"
    assert incidents_after[0]["proctor_notes"] == "Confirmed on video."

    # End session
    ended = db.end_session("S101")
    assert ended.status == "COMPLETED"
    assert ended.total_incidents == 1


def test_pipeline_end_to_end():
    pipeline = EviGuardPipeline("config.yaml")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50) # Neutral background

    output = pipeline.process_frame(frame, session_id="TEST_SESSION", candidate_name="Tester")
    assert output.annotated_frame is not None
    assert output.annotated_frame.shape == (480, 640, 3)
    assert output.risk is not None
    assert output.frame_index >= 1

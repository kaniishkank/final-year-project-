"""
Comprehensive Audit & Verification Test Suite for EviGuard AI
Tests false-positive heuristic filtering, paper sheet detection, NMS person counting,
strict secondary person distance validation (>150px), hand/finger signalling malpractice,
symmetric 4-way gaze thresholding, prolonged gaze malpractice timer,
candidate absence timeout, composite risk weights, and evidence recording.
"""

import os
import time
import numpy as np
import pytest
import cv2
from backend.detection.base import DetectionResult
from backend.detection.yolov8_detector import YOLOv8Detector
from backend.tracking.tracker import PersonTracker
from backend.pose.pose_gaze import PoseGazeEstimator, PoseGazeResult, HandSignallingDetector
from backend.scoring.risk_engine import RiskEngine
from backend.db.models import DatabaseManager
from backend.explainability.reason_generator import ReasonGenerator
from backend.pipeline import EviGuardPipeline


def test_phone_heuristic_and_area_filtering():
    """Verify that small cylindrical objects (pens, inhalers, lip balm) are rejected while valid smartphones are detected."""
    detector = YOLOv8Detector({
        "phone_confidence_threshold": 0.55,
        "person_confidence_threshold": 0.50,
        "phone_min_area": 2800.0,
        "phone_min_w": 35.0,
        "phone_min_h": 45.0,
        "phone_min_aspect_ratio": 1.35,
        "phone_max_aspect_ratio": 2.65
    })

    # 1. Valid Smartphone (Portrait, 60x120 px, Aspect Ratio 2.0, Area 7200 px^2, Conf 0.85) -> ACCEPT
    valid_phone_box = [100.0, 100.0, 160.0, 220.0]
    assert detector._is_valid_phone_geometry(valid_phone_box) is True

    # 2. Valid Smartphone (Landscape, 140x70 px, Aspect Ratio 2.0, Area 9800 px^2, Conf 0.85) -> ACCEPT
    valid_phone_landscape = [100.0, 100.0, 240.0, 170.0]
    assert detector._is_valid_phone_geometry(valid_phone_landscape) is True

    # 3. False Positive: Inhaler / Lip Balm (Small cylindrical stub: 20x35 px, Area 700 px^2) -> REJECT (Too small)
    small_inhaler_box = [100.0, 100.0, 120.0, 135.0]
    assert detector._is_valid_phone_geometry(small_inhaler_box) is False

    # 4. False Positive: Pen / Pencil (Narrow thin cylinder: 12x140 px, Aspect Ratio 11.66) -> REJECT (Too thin / aspect ratio mismatch)
    pen_box = [100.0, 100.0, 112.0, 240.0]
    assert detector._is_valid_phone_geometry(pen_box) is False

    # 5. False Positive: Square object (45x45 px, Aspect Ratio 1.0) -> REJECT (Too square for phone)
    square_box = [100.0, 100.0, 145.0, 145.0]
    assert detector._is_valid_phone_geometry(square_box) is False


def test_white_paper_sheet_heuristic_detector():
    """Verify that a high-contrast rectangular white paper sheet is flagged as unauthorized paper/notes."""
    detector = YOLOv8Detector({"enable_paper_heuristic": True})

    # Create synthetic frame with a dark background and a white rectangular paper sheet (100x140 px, area 14000 px^2)
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (300, 340), (245, 245, 245), -1)

    paper_dets = detector._detect_white_paper_sheets(frame)
    assert len(paper_dets) >= 1
    assert paper_dets[0].class_name == "unauthorized paper/notes"
    assert paper_dets[0].confidence >= 0.80


def test_person_tracker_iou_nms_and_distance_validation():
    """Verify that IoU NMS and >150px distance validation merge posture/arm shifts and isolate true secondary persons."""
    tracker = PersonTracker({
        "person_conf_threshold": 0.50,
        "person_nms_iou": 0.45,
        "min_person_area": 6000.0,
        "min_person_distance": 150.0
    })

    # Scenario A: Candidate extends arm/shifts posture -> Close proximity boxes (centroid distance = 40px < 150px)
    candidate_box1 = DetectionResult(box=[100.0, 50.0, 300.0, 450.0], confidence=0.92, class_id=0, class_name="person")
    arm_shift_box = DetectionResult(box=[140.0, 60.0, 320.0, 460.0], confidence=0.88, class_id=0, class_name="person")

    tracked_a = tracker.update([candidate_box1, arm_shift_box])
    # Must merge into exactly 1 candidate
    assert len(tracked_a) == 1
    assert tracker.get_person_count() == 1

    # Scenario B: Genuine secondary person appears in background/beside desk (centroid distance = 350px > 150px)
    tracker_b = PersonTracker({"min_person_distance": 150.0})
    student1 = DetectionResult(box=[50.0, 50.0, 200.0, 450.0], confidence=0.92, class_id=0, class_name="person")
    intruder = DetectionResult(box=[400.0, 50.0, 550.0, 450.0], confidence=0.89, class_id=0, class_name="person")

    tracked_b = tracker_b.update([student1, intruder])
    assert len(tracked_b) == 2
    assert tracker_b.get_person_count() == 2


def test_hand_finger_signalling_malpractice():
    """Verify that sustained hand/finger signalling triggers malpractice alerts and elevates risk score to 85.0."""
    engine = RiskEngine({
        "weights": {"hand_signalling": 75.0},
        "thresholds": {"high_threshold": 70.0}
    })

    dummy_signalling_pose = PoseGazeResult(
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
        hand_gesture_label="FINGER SIGNALLING (2 Extended Fingers)",
        hand_boxes=[[150.0, 200.0, 250.0, 350.0]]
    )

    assessment = engine.evaluate([], dummy_signalling_pose, person_count=1)
    assert any("FLAG" in v or "SIGNALLING" in v for v in assessment.active_violations)
    assert assessment.raw_score >= 75.0
    assert assessment.smoothed_score >= 85.0
    assert assessment.is_incident_triggered is True

    # Explainability justification
    reason_gen = ReasonGenerator()
    explanation = reason_gen.generate_explanation(assessment, [], dummy_signalling_pose, candidate_name="Bob")
    assert "Signalling" in explanation.summary_headline
    assert explanation.severity == "CRITICAL"


def test_symmetric_4way_head_pose_and_gaze():
    """Verify that directional deviations in all 4 axes (LEFT, RIGHT, DOWN, UP) trigger gaze violations symmetrically."""
    estimator = PoseGazeEstimator({
        "head_pose": {
            "max_yaw_angle": 16.0,
            "max_pitch_angle": 14.0
        }
    })

    # 1. Looking Straight (Center)
    dir_c, away_c = estimator._classify_gaze(yaw=0.0, pitch=0.0, roll=0.0)
    assert dir_c == "CENTER (FOCUSED)"
    assert away_c is False

    # 2. Looking Left (yaw < -16.0)
    dir_l, away_l = estimator._classify_gaze(yaw=-22.0, pitch=2.0, roll=0.0)
    assert dir_l == "LOOKING LEFT"
    assert away_l is True

    # 3. Looking Right (yaw > +16.0)
    dir_r, away_r = estimator._classify_gaze(yaw=+24.0, pitch=2.0, roll=0.0)
    assert dir_r == "LOOKING RIGHT"
    assert away_r is True

    # 4. Looking Down at Desk/Phone (pitch > +14.0)
    dir_d, away_d = estimator._classify_gaze(yaw=0.0, pitch=+18.0, roll=0.0)
    assert dir_d == "LOOKING DOWN"
    assert away_d is True

    # 5. Looking Up at Ceiling (pitch < -14.0)
    dir_u, away_u = estimator._classify_gaze(yaw=0.0, pitch=-19.0, roll=0.0)
    assert dir_u == "LOOKING UP"
    assert away_u is True


def test_prolonged_gaze_malpractice_timer():
    """Verify continuous lookaway for >2.0s triggers CRITICAL_MALPRACTICE and boosts Risk Score to 85.0."""
    engine = RiskEngine({
        "weights": {"prolonged_gaze_malpractice": 85.0},
        "thresholds": {"high_threshold": 70.0}
    })

    dummy_prolonged_pose = PoseGazeResult(
        face_detected=True,
        face_count=1,
        yaw=0.0,
        pitch=20.0,
        roll=0.0,
        gaze_direction="LOOKING DOWN",
        is_looking_away=True,
        is_absent=False,
        absence_frames=0,
        gaze_violation_frames=46,
        gaze_violation_seconds=2.1,
        is_prolonged_lookaway=True
    )

    assessment = engine.evaluate([], dummy_prolonged_pose, person_count=1)
    assert any("CRITICAL_MALPRACTICE" in v for v in assessment.active_violations)
    assert assessment.raw_score >= 85.0
    assert assessment.smoothed_score >= 85.0
    assert assessment.is_incident_triggered is True


def test_candidate_absence_timeout():
    """Verify candidate absence only triggers after 15 continuous missing frames."""
    estimator = PoseGazeEstimator({
        "face_absence": {"absence_frames_threshold": 15}
    })

    for _ in range(14):
        res = estimator.estimate(None)
        assert res.face_detected is False
        assert res.is_absent is False

    res15 = estimator.estimate(None)
    assert res15.face_detected is False
    assert res15.is_absent is True
    assert res15.absence_frames == 15


def test_composite_risk_weights_and_instant_triggers():
    """Verify strict weights (+85 phone, +80 multi-person, +75 absence, +45 gaze) and instant alerting."""
    cfg = {
        "weights": {
            "cell_phone": 85.0,
            "multiple_persons": 80.0,
            "face_absent": 75.0,
            "gaze_deviation": 45.0,
            "head_pose_deviation": 45.0
        },
        "thresholds": {"low_max": 30.0, "medium_max": 70.0, "high_threshold": 70.0}
    }

    dummy_center_pose = PoseGazeResult(
        face_detected=True, face_count=1, yaw=0.0, pitch=0.0, roll=0.0,
        gaze_direction="CENTER (FOCUSED)", is_looking_away=False, is_absent=False, absence_frames=0
    )

    # 1. Phone Detection (+85 weight -> Instant Score >= 85, Triggered = True)
    engine_phone = RiskEngine(cfg)
    phone_det = [DetectionResult(box=[100.0, 100.0, 160.0, 220.0], confidence=0.88, class_id=67, class_name="cell phone")]
    eval_phone = engine_phone.evaluate(phone_det, dummy_center_pose, person_count=1)
    assert "PHONE_DETECTED" in eval_phone.active_violations
    assert eval_phone.raw_score >= 85.0
    assert eval_phone.is_incident_triggered is True

    # 2. Multi-Person Intruder (+80 weight -> Instant Score >= 80, Triggered = True)
    engine_multi = RiskEngine(cfg)
    eval_multi = engine_multi.evaluate([], dummy_center_pose, person_count=2)
    assert "MULTIPLE_PERSONS" in eval_multi.active_violations
    assert eval_multi.raw_score >= 80.0
    assert eval_multi.is_incident_triggered is True

    # 3. Candidate Absence (+75 weight -> Instant Score >= 75, Triggered = True)
    engine_absence = RiskEngine(cfg)
    absent_pose = PoseGazeResult(
        face_detected=False, face_count=0, yaw=0.0, pitch=0.0, roll=0.0,
        gaze_direction="NO FACE DETECTED", is_looking_away=True, is_absent=True, absence_frames=16
    )
    eval_absence = engine_absence.evaluate([], absent_pose, person_count=0)
    assert "FACE_ABSENT" in eval_absence.active_violations
    assert eval_absence.raw_score >= 75.0
    assert eval_absence.is_incident_triggered is True

    # 4. Gaze Deviation (+45 weight)
    engine_gaze = RiskEngine(cfg)
    gaze_pose = PoseGazeResult(
        face_detected=True, face_count=1, yaw=-20.0, pitch=0.0, roll=0.0,
        gaze_direction="LOOKING LEFT", is_looking_away=True, is_absent=False, absence_frames=0
    )
    eval_gaze = engine_gaze.evaluate([], gaze_pose, person_count=1)
    assert "HEAD_TURN (LEFT)" in eval_gaze.active_violations
    assert eval_gaze.raw_score == 45.0


def test_evidence_clip_and_db_logging(tmp_path):
    """Verify evidence clip folder creation and incident database persistence on risk >= 70."""
    db_path = f"sqlite:///{tmp_path}/test_audit.db"
    db_mgr = DatabaseManager(db_path)
    db_mgr.create_session("SESSION_AUDIT_1", "STD-AUDIT", "Audit Student", "Audit Exam")

    evidence_dir = tmp_path / "evidence_clips"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    inc = db_mgr.log_incident(
        session_id="SESSION_AUDIT_1",
        frame_index=10,
        violation_type="PHONE_DETECTED",
        severity="CRITICAL",
        risk_score=85.0,
        confidence=0.92,
        reason_summary="Unauthorized mobile phone detected",
        reason_narrative="A cell phone was identified in candidate hand.",
        evidence_clip_path=str(evidence_dir / "test_clip.mp4"),
        evidence_snapshot_path=str(evidence_dir / "test_snap.jpg"),
        details={"active_violations": ["PHONE_DETECTED"]}
    )

    assert inc.id is not None
    assert inc.violation_type == "PHONE_DETECTED"
    assert inc.severity == "CRITICAL"

    incidents = db_mgr.get_session_incidents("SESSION_AUDIT_1")
    assert len(incidents) == 1
    assert incidents[0]["risk_score"] == 85.0

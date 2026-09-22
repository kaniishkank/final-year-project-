"""
Unit & Integration Tests for EviGuard Proctored Inference Engine Directive
Tests config.py, head_pose.py, detector.py, telemetry.py, and main.py.
"""

import numpy as np
import pytest

from config import CONFIG, EviGuardConfig
from head_pose import HeadPoseEstimator, HeadPoseResult, CANONICAL_3D_POINTS, LANDMARK_INDICES
from detector import ProctoredInferenceEngine, DetectedObject, HandSignallingResult, DetectionBundle
from telemetry import TelemetryEngine, TelemetryState, AnomalyEvent
from main import VideoCircularBuffer, draw_hud


def test_config_structure_and_thresholds():
    """Verify that config.py contains all mandatory operational thresholds."""
    cfg = EviGuardConfig()
    assert cfg.get("system")["imgsz"] == 320
    assert cfg.get("module_a_devices")["phone_min_area"] == 1200.0
    assert cfg.get("module_a_devices")["phone_conf_threshold"] == 0.40
    assert cfg.get("module_a_devices")["book_conf_threshold"] == 0.35
    assert cfg.get("module_a_devices")["smartwatch_wrist_radius"] == 40.0
    assert cfg.get("module_b_signalling")["face_separation_min_px"] == 85.0
    assert cfg.get("module_b_signalling")["signalling_hold_frames"] == 30
    assert cfg.get("module_c_pose_intrusions")["yaw_limit"] == 20.0
    assert cfg.get("module_c_pose_intrusions")["pitch_down_limit"] == 22.0
    assert cfg.get("module_c_pose_intrusions")["absence_frames_threshold"] == 90


def test_head_pose_estimator_math():
    """Verify camera matrix and 3D canonical points."""
    estimator = HeadPoseEstimator(CONFIG)
    cam_mat, dist = estimator.get_camera_matrix(640, 480)
    assert cam_mat.shape == (3, 3)
    assert cam_mat[0, 0] == 640.0  # focal length
    assert cam_mat[0, 2] == 320.0  # cx
    assert cam_mat[1, 2] == 240.0  # cy
    assert CANONICAL_3D_POINTS.shape == (6, 3)
    assert len(LANDMARK_INDICES) == 6


def test_detected_object_area_and_aspect_ratio():
    """Verify DetectedObject geometry calculations."""
    # Box 100x200
    obj = DetectedObject(
        class_id=67,
        class_name="cell phone",
        confidence=0.85,
        box=[50.0, 50.0, 150.0, 250.0]
    )
    assert obj.area == 20000.0
    assert round(obj.aspect_ratio, 2) == 2.0


def test_telemetry_engine_integrity_deductions():
    """Verify sliding-window temporal counters and probabilistic integrity quotient."""
    telemetry = TelemetryEngine(CONFIG)
    assert telemetry.total_penalties == 0.0

    # 1. Simulate Clean Frame
    clean_bundle = DetectionBundle(
        frame_index=1,
        timestamp=100.0,
        objects=[DetectedObject(class_id=0, class_name="person", confidence=0.90, box=[100, 100, 300, 400])],
        hand_signalling=[],
        head_pose=HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, success=True),
        face_detected=True,
        face_count=1,
        person_count=1
    )
    state = telemetry.update(clean_bundle)
    assert state.integrity_quotient == 100.0
    assert state.threat_level == "NORMAL"

    # 2. Simulate Cell Phone Detection (Module A: -30%, Linear Prob 90%-98%)
    phone_bundle = DetectionBundle(
        frame_index=2,
        timestamp=101.0,
        objects=[
            DetectedObject(
                class_id=67,
                class_name="cell phone",
                confidence=0.70,
                box=[50, 50, 120, 190],
                is_malpractice=True,
                malpractice_type="PHONE_DETECTED"
            )
        ],
        hand_signalling=[],
        head_pose=HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, success=True),
        face_detected=True,
        face_count=1,
        person_count=1
    )
    state_phone = telemetry.update(phone_bundle)
    assert state_phone.integrity_quotient == 70.0  # 100 - 30
    assert state_phone.trigger_mp4_dump is True
    assert 0.90 <= state_phone.active_cheat_probabilities.get("PHONE_DETECTED", 0) <= 0.98

    # 3. Simulate MCQ Hand Signalling Hold (Module B: 30 frames -> -10%)
    sig_bundle = DetectionBundle(
        frame_index=3,
        timestamp=105.0,
        objects=[],
        hand_signalling=[
            HandSignallingResult(
                is_raised=True,
                extended_fingers=3,
                gesture_label="MCQ SIGNALLING (3 Fingers)",
                is_signalling_malpractice=True,
                wrist_chin_dist=120.0
            )
        ],
        head_pose=HeadPoseResult(yaw=0.0, pitch=0.0, roll=0.0, success=True),
        face_detected=True,
        face_count=1,
        person_count=1
    )
    # Feed 30 continuous frames
    for i in range(30):
        sig_bundle.frame_index = 10 + i
        state_sig = telemetry.update(sig_bundle)

    assert state_sig.integrity_quotient == 60.0  # 70 - 10


def test_video_circular_buffer_and_hud():
    """Verify circular ring buffer and HUD visual rendering."""
    buf = VideoCircularBuffer(capacity=10, output_dir="data/evidence_clips")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(15):
        buf.append(frame)
    assert len(buf.buffer) == 10

    # Test HUD rendering
    bundle = DetectionBundle(
        frame_index=1,
        timestamp=1.0,
        objects=[DetectedObject(class_id=0, class_name="person", confidence=0.88, box=[100, 100, 300, 400])],
        hand_signalling=[],
        head_pose=HeadPoseResult(yaw=5.0, pitch=-3.0, roll=1.0, success=True, nose_2d=(320, 240)),
        face_detected=True,
        face_count=1,
        person_count=1
    )
    state = TelemetryState(
        frame_index=1,
        timestamp=1.0,
        integrity_quotient=95.0,
        threat_level="NORMAL",
        active_violations=[],
        active_cheat_probabilities={},
        total_incidents=0,
        trigger_mp4_dump=False,
        mp4_dump_reason="",
        fps=30.0
    )
    hud_frame = draw_hud(frame, bundle, state)
    assert hud_frame.shape == (480, 640, 3)

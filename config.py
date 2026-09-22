"""
EviGuard AI Proctoring System - Global Configuration Module
Contains all operational thresholds, bounding area cutoffs, camera intrinsic assumptions,
and probabilistic threat scoring weights for YOLO26 + MediaPipe inference.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple


CONFIG: Dict[str, Any] = {
    # System & Model Architecture
    "system": {
        "app_name": "EviGuard Proctored Inference Engine",
        "version": "2.0.0",
        "device": "auto",  # 'cuda', 'cpu', or 'auto'
        "imgsz": 640,
        "nms_free": True,  # Native YOLO26 end-to-end NMS-free mode
    },

    # Model Checkpoint Paths
    "models": {
        "yolo_detection": "yolo26n.pt",
        "yolo_pose": "yolo26n-pose.pt",
        "fallback_detection": "yolov8n.pt",
        "fallback_pose": "yolov8n-pose.pt",
    },

    # Video Capture & Circular Ring Buffer
    "video": {
        "source": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "buffer_pre_roll_s": 3,
        "buffer_post_roll_s": 3,
        "evidence_dir": "data/evidence_clips",
        "cooldown_seconds": 3.0,
    },

    # Camera Intrinsic Assumptions (Pinhole Model)
    "camera_intrinsics": {
        "focal_length_factor": 1.0,  # fx = fy = width * focal_length_factor
        "zero_distortion": True,
    },

    # Module A: Unauthorized Physical Materials & Devices (Malpractice Likelihood: 90% - 98%)
    "module_a_devices": {
        # Cell Phone (COCO 67)
        "phone_conf_threshold": 0.40,
        "phone_min_area": 1200.0,
        "phone_aspect_ratio_min": 1.30,
        "phone_aspect_ratio_max": 2.70,

        # Book / Notebook (COCO 73)
        "book_conf_threshold": 0.35,

        # Smartwatch Detection (Wrist Keypoint 9 & 10 ROI)
        "smartwatch_enable": True,
        "smartwatch_wrist_radius": 40.0,  # px radius around wrist
        "smartwatch_conf_threshold": 0.35,
        "smartwatch_contrast_threshold": 45.0,

        # Loose Paper / Cheat Notes Workspace Heuristic
        "paper_heuristic_enable": True,
        "paper_min_area": 2200.0,
        "paper_aspect_ratio_min": 1.10,
        "paper_aspect_ratio_max": 1.85,
        "paper_stable_frames": 20,

        # Scoring & Penalties
        "cheat_prob_min": 0.90,
        "cheat_prob_max": 0.98,
        "severity": "CRITICAL",
        "integrity_penalty": 30.0,  # -30% Integrity Quotient
        "trigger_evidence_dump": True,
    },

    # Module B: MCQ Hand & Finger Signalling (Malpractice Likelihood: 50% - 60%)
    "module_b_signalling": {
        "enable": True,
        "max_num_hands": 2,
        "min_detection_confidence": 0.50,
        "min_tracking_confidence": 0.50,
        
        # Raised Hand Criteria
        "wrist_y_max": 0.88,  # Normalized frame height (wrist above bottom 12% of screen)
        
        # Face Separation Safeguard (Discards chin-resting/ear-scratching)
        "face_separation_min_px": 85.0,  # Euclidean dist(wrist, chin) > 85 px
        
        # Finger Counting Criteria
        "target_finger_counts": [1, 2, 3, 4],  # MCQ options A, B, C, D
        "signalling_hold_frames": 30,  # ~1.0s at 30 FPS
        
        # Scoring & Penalties
        "cheat_prob_min": 0.50,
        "cheat_prob_max": 0.60,
        "severity": "WARNING",
        "integrity_penalty": 10.0,  # -10% Integrity Quotient
        "trigger_evidence_dump": False,
    },

    # Module C: Secondary Intrusions & Pose Deviations
    "module_c_pose_intrusions": {
        # Head Pose 3D Euler Thresholds (Degrees)
        "yaw_limit": 20.0,
        "pitch_down_limit": 22.0,  # Looking down at lap/desk
        "pitch_up_limit": -18.0,   # Looking up
        "roll_limit": 20.0,
        "pose_sustained_frames": 45,  # >= 1.5s hold triggers critical alert
        "pose_cheat_prob_min": 0.85,
        "pose_cheat_prob_max": 0.90,
        "pose_integrity_penalty": 20.0,

        # Candidate Absence / Seat Abandonment
        "absence_frames_threshold": 90,  # ~3.0s continuous absence
        "absence_cheat_prob_min": 0.85,
        "absence_cheat_prob_max": 0.92,
        "absence_integrity_penalty": 20.0,

        # Secondary Person Intruder Validation
        "person_conf_threshold": 0.50,
        "person_min_area": 5000.0,
        "person_separation_dist_px": 150.0,
        "person_nms_iou": 0.45,
        "intruder_integrity_penalty": 30.0,
    },

    # Integrity Quotient Aggregation Base
    "integrity_quotient": {
        "base_score": 100.0,
        "min_score": 0.0,
    }
}


@dataclass
class EviGuardConfig:
    """Strongly-typed runtime configuration wrapper."""
    raw: Dict[str, Any] = field(default_factory=lambda: CONFIG)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

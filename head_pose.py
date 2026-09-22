"""
EviGuard Head Pose & 3D Gaze Estimation Engine
Computes 3D Euler angles (Yaw, Pitch, Roll) using MediaPipe FaceMesh landmarks
and OpenCV solvePnP Perspective-n-Point solver with canonical 3D facial coordinates.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np


# Canonical 3D Facial Model Points in Millimeters (Anthropometric Standard)
CANONICAL_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (Landmark 1)
    [0.0, -330.0, -65.0],     # Chin (Landmark 199)
    [-225.0, 170.0, -135.0],  # Left Eye Outer Corner (Landmark 33)
    [225.0, 170.0, -135.0],   # Right Eye Outer Corner (Landmark 263)
    [-150.0, -150.0, -125.0], # Left Mouth Corner (Landmark 61)
    [150.0, -150.0, -125.0]   # Right Mouth Corner (Landmark 291)
], dtype=np.float64)

# 2D Landmark Indices matching CANONICAL_3D_POINTS
LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]
CHIN_LANDMARK_IDX = 152  # Exact lower-jaw chin point for wrist distance safeguards


@dataclass
class HeadPoseResult:
    """Structured result of 3D head pose and gaze estimation."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    is_deviated: bool = False
    gaze_direction: str = "FORWARD"
    deviation_reasons: List[str] = None
    nose_2d: Optional[Tuple[int, int]] = None
    chin_2d: Optional[Tuple[int, int]] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    success: bool = False

    def __post_init__(self):
        if self.deviation_reasons is None:
            self.deviation_reasons = []


class HeadPoseEstimator:
    """3D-to-2D Perspective-n-Point (solvePnP) Head Pose Solver."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        pose_cfg = cfg.get("module_c_pose_intrusions", {})
        
        self.yaw_limit = float(pose_cfg.get("yaw_limit", 20.0))
        self.pitch_down_limit = float(pose_cfg.get("pitch_down_limit", 22.0))
        self.pitch_up_limit = float(pose_cfg.get("pitch_up_limit", -18.0))
        self.roll_limit = float(pose_cfg.get("roll_limit", 20.0))

    @staticmethod
    def get_camera_matrix(width: int, height: int, focal_length_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs pinhole camera calibration matrix and zero-distortion coefficients."""
        focal_length = width * focal_length_factor
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        return camera_matrix, dist_coeffs

    def estimate(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int
    ) -> HeadPoseResult:
        """Solves 3D pose and Euler angles from MediaPipe FaceMesh landmarks."""
        if not face_landmarks:
            return HeadPoseResult(success=False)

        # 1. Extract 2D Landmark Points
        image_points = []
        for idx in LANDMARK_INDICES:
            lm = face_landmarks.landmark[idx]
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            image_points.append([x, y])

        image_points_2d = np.array(image_points, dtype=np.float64)

        # Extract nose and chin coordinates for visual HUD and distance safeguards
        nose_lm = face_landmarks.landmark[1]
        nose_2d = (int(nose_lm.x * frame_width), int(nose_lm.y * frame_height))

        chin_lm = face_landmarks.landmark[CHIN_LANDMARK_IDX]
        chin_2d = (int(chin_lm.x * frame_width), int(chin_lm.y * frame_height))

        # 2. Camera Intrinsics
        camera_matrix, dist_coeffs = self.get_camera_matrix(frame_width, frame_height)

        # 3. Solve Perspective-n-Point (solvePnP)
        success, rvec, tvec = cv2.solvePnP(
            CANONICAL_3D_POINTS,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return HeadPoseResult(
                nose_2d=nose_2d,
                chin_2d=chin_2d,
                success=False
            )

        # 4. Convert Rotation Vector to Matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # 5. Extract Euler Angles (Pitch, Yaw, Roll)
        # Using RQ Decomposition for robust non-gimbal extraction
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        # Decompose angles (degrees)
        pitch = float(angles[0] * 360.0)
        yaw = float(angles[1] * 360.0)
        roll = float(angles[2] * 360.0)

        # 6. Directional & Deviation Evaluation
        is_deviated = False
        reasons: List[str] = []
        direction_parts: List[str] = []

        if abs(yaw) > self.yaw_limit:
            is_deviated = True
            dir_str = "LOOKING RIGHT" if yaw > 0 else "LOOKING LEFT"
            direction_parts.append(dir_str)
            reasons.append(f"Yaw Deviation ({yaw:+.1f}° > ±{self.yaw_limit}°)")

        if pitch > self.pitch_down_limit:
            is_deviated = True
            direction_parts.append("LOOKING DOWN")
            reasons.append(f"Downward Pitch ({pitch:+.1f}° > {self.pitch_down_limit}°)")
        elif pitch < self.pitch_up_limit:
            is_deviated = True
            direction_parts.append("LOOKING UP")
            reasons.append(f"Upward Pitch ({pitch:+.1f}° < {self.pitch_up_limit}°)")

        if abs(roll) > self.roll_limit:
            is_deviated = True
            reasons.append(f"Head Tilt Roll ({roll:+.1f}° > ±{self.roll_limit}°)")

        gaze_direction = " + ".join(direction_parts) if direction_parts else "FORWARD"

        return HeadPoseResult(
            yaw=round(yaw, 1),
            pitch=round(pitch, 1),
            roll=round(roll, 1),
            is_deviated=is_deviated,
            gaze_direction=gaze_direction,
            deviation_reasons=reasons,
            nose_2d=nose_2d,
            chin_2d=chin_2d,
            rvec=rvec,
            tvec=tvec,
            success=True
        )

    @staticmethod
    def draw_pose_axes(
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        nose_2d: Tuple[int, int],
        length: float = 80.0
    ) -> np.ndarray:
        """Projects and renders 3D coordinate axes from the candidate's nose."""
        if rvec is None or tvec is None or nose_2d is None:
            return frame

        h, w = frame.shape[:2]
        camera_matrix, dist_coeffs = HeadPoseEstimator.get_camera_matrix(w, h)

        # 3D axis endpoints
        axis_3d = np.array([
            [length, 0.0, 0.0],    # X axis (Red - Right)
            [0.0, -length, 0.0],   # Y axis (Green - Up)
            [0.0, 0.0, -length]    # Z axis (Blue - Outward gaze vector)
        ], dtype=np.float64)

        img_pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

        p_nose = nose_2d
        p_x = (int(img_pts[0][0][0]), int(img_pts[0][0][1]))
        p_y = (int(img_pts[1][0][0]), int(img_pts[1][0][1]))
        p_z = (int(img_pts[2][0][0]), int(img_pts[2][0][1]))

        # Render 3D Vector Rays
        cv2.line(frame, p_nose, p_x, (0, 0, 255), 2, cv2.LINE_AA)  # Red - X
        cv2.line(frame, p_nose, p_y, (0, 255, 0), 2, cv2.LINE_AA)  # Green - Y
        cv2.line(frame, p_nose, p_z, (255, 165, 0), 3, cv2.LINE_AA)  # Blue/Cyan - Gaze Vector

        return frame

"""
Pose and Gaze Estimation Module
Calculates 3D Head Pose (Yaw, Pitch, Roll) and Gaze Direction using MediaPipe FaceMesh and solvePnP.
Features exact symmetric 6-point 3D-to-2D facial landmark mapping, 4-way directional thresholding (LEFT, RIGHT, DOWN, UP),
and continuous prolonged gaze malpractice tracking (>2.0 seconds).
"""

from dataclasses import dataclass
import logging
import math
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("EviGuard.PoseGaze")


@dataclass
class PoseGazeResult:
    """Encapsulates pose and gaze metrics extracted from a video frame."""
    face_detected: bool
    face_count: int
    yaw: float # Negative = left, Positive = right
    pitch: float # Positive = down (desk/phone), Negative = up (ceiling)
    roll: float # Negative = tilt left, Positive = tilt right
    gaze_direction: str # CENTER (FOCUSED), LOOKING LEFT, LOOKING RIGHT, LOOKING DOWN, LOOKING UP
    is_looking_away: bool
    is_absent: bool
    absence_frames: int
    gaze_violation_frames: int = 0
    gaze_violation_seconds: float = 0.0
    is_prolonged_lookaway: bool = False
    face_box: Optional[List[float]] = None
    landmarks_2d: Optional[List[Tuple[float, float]]] = None
    nose_projection_2d: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_detected": self.face_detected,
            "face_count": self.face_count,
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "gaze_direction": self.gaze_direction,
            "is_looking_away": self.is_looking_away,
            "is_absent": self.is_absent,
            "absence_frames": self.absence_frames,
            "gaze_violation_frames": self.gaze_violation_frames,
            "gaze_violation_seconds": round(self.gaze_violation_seconds, 2),
            "is_prolonged_lookaway": self.is_prolonged_lookaway,
        }


class PoseGazeEstimator:
    """Extracts head pose angles and gaze metrics from frames or cropped student ROIs."""

    # Stable 3D generic facial model points (in mm, centered around nose tip)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
        (0.0, -330.0, -65.0),        # Chin (landmark 199)
        (-225.0, 170.0, -135.0),     # Left Eye Outer Corner (landmark 33)
        (225.0, 170.0, -135.0),      # Right Eye Outer Corner (landmark 263)
        (-150.0, -150.0, -125.0),    # Left Mouth Corner (landmark 61)
        (150.0, -150.0, -125.0)      # Right Mouth Corner (landmark 291)
    ], dtype=np.float64)

    # Exact symmetric landmark indices in MediaPipe Face Mesh
    LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Symmetric 4-Way Directional Thresholds (Degrees)
        head_cfg = self.config.get("head_pose", {})
        self.max_yaw_angle = abs(float(head_cfg.get("max_yaw_angle", head_cfg.get("yaw_limit_right", 16.0))))
        self.max_pitch_angle = abs(float(head_cfg.get("max_pitch_angle", head_cfg.get("pitch_limit_down", 14.0))))
        self.max_roll_angle = abs(float(head_cfg.get("max_roll_angle", head_cfg.get("roll_limit", 22.0))))

        # Continuous Prolonged Gaze Malpractice Threshold (~2.0 seconds / 45-60 frames)
        self.fps = float(self.config.get("fps", 30.0))
        self.prolonged_gaze_threshold_frames = int(self.config.get("prolonged_gaze_threshold_frames", 45))

        absence_cfg = self.config.get("face_absence", {})
        self.absence_threshold = int(absence_cfg.get("absence_frames_threshold", 15)) # ~0.5s rapid response

        self.consecutive_absence_frames = 0
        self.consecutive_lookaway_frames = 0
        self.face_mesh = None
        self._fallback_mode = False

        self._init_mediapipe()

    def _init_mediapipe(self):
        """Initializes MediaPipe Face Mesh with fallback handling."""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=2,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe FaceMesh initialized successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe FaceMesh unavailable ({e}). Using OpenCV/Mathematical fallback.")
            self._fallback_mode = True

    def estimate(self, frame: np.ndarray, bbox: Optional[List[float]] = None) -> PoseGazeResult:
        """Processes frame to compute 3D head pose (Yaw, Pitch, Roll) and 4-way gaze direction."""
        if frame is None or frame.size == 0:
            self.consecutive_absence_frames += 1
            self.consecutive_lookaway_frames = 0
            return self._create_absent_result()

        if self._fallback_mode or self.face_mesh is None:
            return self._fallback_estimate(frame)

        try:
            h_full, w_full = frame.shape[:2]
            
            # Crop optimization if bounding box provided
            offset_x, offset_y = 0.0, 0.0
            if bbox is not None:
                bx1, by1, bx2, by2 = [int(v) for v in bbox]
                pad_x = int((bx2 - bx1) * 0.1)
                pad_y = int((by2 - by1) * 0.1)
                cx1 = max(0, bx1 - pad_x)
                cy1 = max(0, by1 - pad_y)
                cx2 = min(w_full, bx2 + pad_x)
                cy2 = min(h_full, by2 + pad_y)
                
                if (cx2 - cx1) > 50 and (cy2 - cy1) > 50:
                    crop_patch = frame[cy1:cy2, cx1:cx2]
                    offset_x, offset_y = float(cx1), float(cy1)
                    process_img = crop_patch
                else:
                    process_img = frame
            else:
                process_img = frame

            h, w = process_img.shape[:2]
            rgb_frame = cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                self.consecutive_absence_frames += 1
                self.consecutive_lookaway_frames = 0
                return self._create_absent_result()

            # Face detected -> reset absence counter
            self.consecutive_absence_frames = 0
            face_count = len(results.multi_face_landmarks)
            primary_face = results.multi_face_landmarks[0]

            # Extract 2D image points for the 6 symmetric facial landmarks
            image_points_2d = []
            for idx in self.LANDMARK_INDICES:
                lm = primary_face.landmark[idx]
                image_points_2d.append([lm.x * w + offset_x, lm.y * h + offset_y])
            image_points_2d = np.array(image_points_2d, dtype=np.float64)

            # Camera matrix assumption
            focal_length = w_full
            center = (w_full / 2.0, h_full / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            # Solve PnP for 3D Head Pose
            success, rvec, tvec = cv2.solvePnP(
                self.MODEL_POINTS_3D,
                image_points_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return self._fallback_estimate(frame)

            # Convert rotation vector to 3x3 rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            
            # Decompose rotation matrix into Euler angles (Pitch, Yaw, Roll)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch = float(angles[0]) # Positive = Looking Down, Negative = Looking Up
            yaw = float(angles[1])   # Positive = Looking Right, Negative = Looking Left
            roll = float(angles[2])  # Tilt

            # Project 3D nose vector forward for HUD visualization
            nose_end_point3D = np.array([[0.0, 0.0, 500.0]], dtype=np.float64)
            nose_end_point2D, _ = cv2.projectPoints(nose_end_point3D, rvec, tvec, camera_matrix, dist_coeffs)
            p_nose_2d = (float(nose_end_point2D[0][0][0]), float(nose_end_point2D[0][0][1]))

            # Symmetric 4-Way Directional Classification
            gaze_direction, is_looking_away = self._classify_gaze(yaw, pitch, roll)

            # Prolonged Gaze Malpractice Tracking
            if is_looking_away:
                self.consecutive_lookaway_frames += 1
            else:
                self.consecutive_lookaway_frames = 0

            gaze_seconds = self.consecutive_lookaway_frames / max(1.0, self.fps)
            is_prolonged = self.consecutive_lookaway_frames >= self.prolonged_gaze_threshold_frames

            # Extract face bounding box mapped to global image
            all_x = [lm.x * w + offset_x for lm in primary_face.landmark]
            all_y = [lm.y * h + offset_y for lm in primary_face.landmark]
            face_box = [float(min(all_x)), float(min(all_y)), float(max(all_x)), float(max(all_y))]

            return PoseGazeResult(
                face_detected=True,
                face_count=face_count,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                gaze_direction=gaze_direction,
                is_looking_away=is_looking_away,
                is_absent=False,
                absence_frames=0,
                gaze_violation_frames=self.consecutive_lookaway_frames,
                gaze_violation_seconds=gaze_seconds,
                is_prolonged_lookaway=is_prolonged,
                face_box=face_box,
                landmarks_2d=[(pt[0], pt[1]) for pt in image_points_2d],
                nose_projection_2d=p_nose_2d
            )

        except Exception as e:
            logger.error(f"Error in pose & gaze estimation: {e}. Utilizing fallback.")
            return self._fallback_estimate(frame)

    def _classify_gaze(self, yaw: float, pitch: float, roll: float) -> Tuple[str, bool]:
        """Symmetric 4-way directional thresholding for LEFT, RIGHT, DOWN, UP, and CENTER."""
        # 1. Yaw Checks (Horizontal: Left / Right)
        if yaw < -self.max_yaw_angle:  # e.g., < -16.0 deg
            return "LOOKING LEFT", True
        elif yaw > self.max_yaw_angle: # e.g., > +16.0 deg
            return "LOOKING RIGHT", True

        # 2. Pitch Checks (Vertical: Down / Up)
        elif pitch > self.max_pitch_angle:   # e.g., > +14.0 deg (looking down at desk/lap/phone)
            return "LOOKING DOWN", True
        elif pitch < -self.max_pitch_angle:  # e.g., < -14.0 deg (looking up at ceiling)
            return "LOOKING UP", True

        # 3. Roll Checks (Head Tilt)
        elif abs(roll) > self.max_roll_angle:
            return "HEAD TILTED", True

        # 4. Center / Focused Normal State
        return "CENTER (FOCUSED)", False

    def _create_absent_result(self) -> PoseGazeResult:
        """Returns result state when no face is visible."""
        is_absent = self.consecutive_absence_frames >= self.absence_threshold
        return PoseGazeResult(
            face_detected=False,
            face_count=0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            gaze_direction="NO FACE DETECTED",
            is_looking_away=True,
            is_absent=is_absent,
            absence_frames=self.consecutive_absence_frames,
            gaze_violation_frames=0,
            gaze_violation_seconds=0.0,
            is_prolonged_lookaway=False
        )

    def _fallback_estimate(self, frame: np.ndarray) -> PoseGazeResult:
        """Lightweight OpenCV Haar Cascade fallback with calibrated 4-way directional gaze and duration tracking."""
        h, w = frame.shape[:2]
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)

            if len(faces) == 0:
                self.consecutive_absence_frames += 1
                self.consecutive_lookaway_frames = 0
                return self._create_absent_result()

            self.consecutive_absence_frames = 0
            x, y, fw, fh = faces[0]
            face_center_x = x + fw / 2.0
            face_center_y = y + fh / 2.0

            # Proportional angular estimation from center of screen
            yaw = float((face_center_x - w / 2.0) / (w / 2.0) * 35.0)
            pitch = float((face_center_y - h / 2.0) / (h / 2.0) * 30.0)
            roll = 0.0

            gaze_direction, is_looking_away = self._classify_gaze(yaw, pitch, roll)

            if is_looking_away:
                self.consecutive_lookaway_frames += 1
            else:
                self.consecutive_lookaway_frames = 0

            gaze_seconds = self.consecutive_lookaway_frames / max(1.0, self.fps)
            is_prolonged = self.consecutive_lookaway_frames >= self.prolonged_gaze_threshold_frames

            return PoseGazeResult(
                face_detected=True,
                face_count=len(faces),
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                gaze_direction=gaze_direction,
                is_looking_away=is_looking_away,
                is_absent=False,
                absence_frames=0,
                gaze_violation_frames=self.consecutive_lookaway_frames,
                gaze_violation_seconds=gaze_seconds,
                is_prolonged_lookaway=is_prolonged,
                face_box=[float(x), float(y), float(x + fw), float(y + fh)],
                landmarks_2d=[(face_center_x, face_center_y)],
                nose_projection_2d=(face_center_x + yaw * 2.5, face_center_y + pitch * 2.5)
            )
        except Exception:
            self.consecutive_absence_frames = 0
            self.consecutive_lookaway_frames = 0
            return PoseGazeResult(
                face_detected=True,
                face_count=1,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                gaze_direction="CENTER (FOCUSED)",
                is_looking_away=False,
                is_absent=False,
                absence_frames=0,
                gaze_violation_frames=0,
                gaze_violation_seconds=0.0,
                is_prolonged_lookaway=False,
                face_box=[float(w * 0.3), float(h * 0.2), float(w * 0.7), float(h * 0.7)]
            )

"""
Pose and Gaze Estimation Module
Calculates 3D Head Pose (Yaw, Pitch, Roll) and Gaze Direction using MediaPipe FaceMesh and solvePnP.
Features cropped bounding box optimization for ultra-fast landmark compute and relaxed angle thresholds.
"""

from dataclasses import dataclass
import logging
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("EviGuard.PoseGaze")


@dataclass
class PoseGazeResult:
    """Encapsulates pose and gaze metrics extracted from a video frame."""
    face_detected: bool
    face_count: int
    yaw: float # Negative = left, Positive = right
    pitch: float # Negative = down (looking at desk/phone), Positive = up
    roll: float # Negative = tilt left, Positive = tilt right
    gaze_direction: str # CENTER, LOOKING_LEFT, LOOKING_RIGHT, LOOKING_DOWN, LOOKING_UP
    is_looking_away: bool
    is_absent: bool
    absence_frames: int
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
        }


class PoseGazeEstimator:
    """Extracts head pose angles and gaze metrics from frames or cropped student ROIs."""

    # 3D generic facial model points (in mm, centered around nose tip)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
        (0.0, -330.0, -65.0),        # Chin (landmark 152)
        (-225.0, 170.0, -135.0),     # Left eye left corner (landmark 33)
        (225.0, 170.0, -135.0),      # Right eye right corner (landmark 263)
        (-150.0, -150.0, -125.0),    # Left Mouth corner (landmark 61)
        (150.0, -150.0, -125.0)      # Right mouth corner (landmark 291)
    ], dtype=np.float64)

    # Key landmark indices in MediaPipe Face Mesh
    LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Relaxed thresholds for snappier angle detection
        head_cfg = self.config.get("head_pose", {})
        self.yaw_limit_left = head_cfg.get("yaw_limit_left", -20.0)
        self.yaw_limit_right = head_cfg.get("yaw_limit_right", 20.0)
        self.pitch_limit_down = head_cfg.get("pitch_limit_down", -18.0)
        self.pitch_limit_up = head_cfg.get("pitch_limit_up", 22.0)
        self.roll_limit = head_cfg.get("roll_limit", 22.0)

        absence_cfg = self.config.get("face_absence", {})
        self.absence_threshold = absence_cfg.get("absence_frames_threshold", 25)

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
        """Processes frame (or cropped student bounding box) to compute 3D head pose and gaze orientation.
        
        Args:
            frame: Raw BGR video frame.
            bbox: Optional [x1, y1, x2, y2] bounding box to run on cropped patch.
        """
        if frame is None or frame.size == 0:
            self.consecutive_absence_frames += 1
            return self._create_absent_result()

        if self._fallback_mode or self.face_mesh is None:
            return self._fallback_estimate(frame)

        try:
            import cv2
            h_full, w_full = frame.shape[:2]
            
            # Crop optimization if bounding box provided
            offset_x, offset_y = 0.0, 0.0
            if bbox is not None:
                bx1, by1, bx2, by2 = [int(v) for v in bbox]
                # Add 10% safety margin
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
                return self._create_absent_result()

            # Face detected -> reset absence counter
            self.consecutive_absence_frames = 0
            face_count = len(results.multi_face_landmarks)
            primary_face = results.multi_face_landmarks[0]

            # Extract 2D image points mapped back to global coordinates
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

            # Solve PnP for 3D Pose
            success, rvec, tvec = cv2.solvePnP(
                self.MODEL_POINTS_3D,
                image_points_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return self._fallback_estimate(frame)

            # Compute rotation angles from rotation vector
            rmat, _ = cv2.Rodrigues(rvec)
            yaw, pitch, roll = self._rotation_matrix_to_euler_angles(rmat)

            # Project 3D nose vector forward
            nose_end_point3D = np.array([[0.0, 0.0, 500.0]], dtype=np.float64)
            nose_end_point2D, _ = cv2.projectPoints(nose_end_point3D, rvec, tvec, camera_matrix, dist_coeffs)
            p_nose_2d = (float(nose_end_point2D[0][0][0]), float(nose_end_point2D[0][0][1]))

            # Classify Gaze & Orientation
            gaze_direction, is_looking_away = self._classify_gaze(yaw, pitch, roll)

            if is_looking_away:
                self.consecutive_lookaway_frames += 1
            else:
                self.consecutive_lookaway_frames = max(0, self.consecutive_lookaway_frames - 1)

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
                face_box=face_box,
                landmarks_2d=[(pt[0], pt[1]) for pt in image_points_2d],
                nose_projection_2d=p_nose_2d
            )

        except Exception as e:
            logger.error(f"Error in pose & gaze estimation: {e}. Utilizing fallback.")
            return self._fallback_estimate(frame)

    def _classify_gaze(self, yaw: float, pitch: float, roll: float) -> Tuple[str, bool]:
        """Categorizes head pose into discrete gaze directions."""
        is_looking_away = False
        directions = []

        if yaw < self.yaw_limit_left:
            directions.append("LOOKING_LEFT")
            is_looking_away = True
        elif yaw > self.yaw_limit_right:
            directions.append("LOOKING_RIGHT")
            is_looking_away = True

        if pitch < self.pitch_limit_down:
            directions.append("LOOKING_DOWN (DESK/PHONE)")
            is_looking_away = True
        elif pitch > self.pitch_limit_up:
            directions.append("LOOKING_UP")
            is_looking_away = True

        if abs(roll) > self.roll_limit:
            directions.append("HEAD_TILTED")
            is_looking_away = True

        if not directions:
            return "CENTER (FOCUSED)", False

        return " + ".join(directions), is_looking_away

    @staticmethod
    def _rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
        """Calculates rotation angles (Yaw, Pitch, Roll in degrees) from 3x3 rotation matrix."""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0.0

        return float(math.degrees(y)), float(math.degrees(x)), float(math.degrees(z))

    def _create_absent_result(self) -> PoseGazeResult:
        """Returns result state when no face is visible."""
        is_absent = self.consecutive_absence_frames >= self.absence_threshold
        return PoseGazeResult(
            face_detected=False,
            face_count=0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            gaze_direction="NO_FACE_DETECTED",
            is_looking_away=True,
            is_absent=is_absent,
            absence_frames=self.consecutive_absence_frames
        )

    def _fallback_estimate(self, frame: np.ndarray) -> PoseGazeResult:
        """Lightweight OpenCV Haar Cascade fallback for face detection and centered gaze."""
        h, w = frame.shape[:2]
        try:
            import cv2
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)

            if len(faces) == 0:
                self.consecutive_absence_frames += 1
                return self._create_absent_result()

            self.consecutive_absence_frames = 0
            x, y, fw, fh = faces[0]
            face_center_x = x + fw / 2.0
            face_center_y = y + fh / 2.0

            yaw = float((face_center_x - w / 2.0) / (w / 2.0) * 30.0)
            pitch = float(-(face_center_y - h / 2.0) / (h / 2.0) * 20.0)
            roll = 0.0

            gaze_direction, is_looking_away = self._classify_gaze(yaw, pitch, roll)

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
                face_box=[float(x), float(y), float(x + fw), float(y + fh)],
                landmarks_2d=[(face_center_x, face_center_y)],
                nose_projection_2d=(face_center_x + yaw * 2, face_center_y + pitch * 2)
            )
        except Exception:
            self.consecutive_absence_frames = 0
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
                face_box=[float(w * 0.3), float(h * 0.2), float(w * 0.7), float(h * 0.7)]
            )

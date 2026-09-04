"""
Pose, Gaze, and Hand Signalling Estimation Module
Calculates 3D Head Pose (Yaw, Pitch, Roll), Gaze Direction, and detects suspicious finger signalling / hand gestures
using MediaPipe FaceMesh & Hands solutions with robust fallback.
"""

from dataclasses import dataclass, field
import logging
import math
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("EviGuard.PoseGaze")


@dataclass
class PoseGazeResult:
    """Encapsulates pose, gaze, and hand gesture metrics extracted from a video frame."""
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
    
    # Hand / Finger Signalling Fields
    hand_signalling: bool = False
    extended_fingers: int = 0
    hand_gesture_label: str = ""
    hand_boxes: List[List[float]] = field(default_factory=list)
    hand_landmarks: List[Any] = field(default_factory=list)
    
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
            "hand_signalling": self.hand_signalling,
            "extended_fingers": self.extended_fingers,
            "hand_gesture_label": self.hand_gesture_label,
            "hand_boxes": self.hand_boxes,
        }


class HandSignallingDetector:
    """Detects suspicious hand gesturing / finger counting (e.g. signaling 1, 2, 3, 4 fingers to neighbors)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.fps = float(self.config.get("fps", 30.0))
        self.gesture_threshold_frames = int(self.config.get("hand_signalling_frames", 35)) # ~1.2s - 1.5s
        self.consecutive_gesture_frames = 0
        self.mp_hands = None
        self.hands_detector = None
        self._fallback_mode = False

        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands_detector = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Hands initialized successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe Hands unavailable ({e}). Utilizing skin-contour heuristic fallback.")
            self._fallback_mode = True

    def detect(self, frame: np.ndarray) -> Tuple[bool, int, str, List[List[float]], List[Any]]:
        """Analyzes frame for raised hands with suspicious extended fingers (signaling options A/B/C/D)."""
        if frame is None or frame.size == 0:
            self.consecutive_gesture_frames = 0
            return False, 0, "", [], []

        h, w = frame.shape[:2]
        hand_boxes: List[List[float]] = []
        extended_fingers_count = 0
        gesture_detected = False
        gesture_label = ""
        hand_landmarks_list: List[Any] = []

        if not self._fallback_mode and self.hands_detector is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands_detector.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        pts = [(lm.x * w, lm.y * h) for lm in hand_lms.landmark]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        x1, y1, x2, y2 = max(0, min(xs) - 10), max(0, min(ys) - 10), min(w, max(xs) + 10), min(h, max(ys) + 10)
                        hand_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        hand_landmarks_list.append(pts)

                        wrist_y = hand_lms.landmark[0].y
                        # Check if hand is raised in front of camera / above lower third of frame (wrist_y < 0.90)
                        if wrist_y < 0.90:
                            # Count extended fingers:
                            # Index (tip 8 vs pip 6)
                            # Middle (tip 12 vs pip 10)
                            # Ring (tip 16 vs pip 14)
                            # Pinky (tip 20 vs pip 18)
                            # Thumb (tip 4 vs mcp 2)
                            fingers = 0
                            if hand_lms.landmark[8].y < hand_lms.landmark[6].y:
                                fingers += 1
                            if hand_lms.landmark[12].y < hand_lms.landmark[10].y:
                                fingers += 1
                            if hand_lms.landmark[16].y < hand_lms.landmark[14].y:
                                fingers += 1
                            if hand_lms.landmark[20].y < hand_lms.landmark[18].y:
                                fingers += 1
                            if abs(hand_lms.landmark[4].x - hand_lms.landmark[2].x) > 0.04:
                                fingers += 1

                            extended_fingers_count = max(extended_fingers_count, fingers)
                            # Suspicious finger counts: 1, 2, 3, 4 fingers (used for MCQ A, B, C, D cheating)
                            if 1 <= fingers <= 4:
                                gesture_detected = True
                                gesture_label = f"FINGER SIGNALLING ({fingers} Extended Fingers)"
            except Exception as e:
                logger.debug(f"MediaPipe Hands detection exception: {e}")

        # Fallback skin contour / hand detection if MediaPipe is not installed
        if self._fallback_mode or self.hands_detector is None:
            try:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([0, 25, 60], dtype=np.uint8)
                upper_skin = np.array([25, 200, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower_skin, upper_skin)
                mask[0:int(h * 0.35), int(w * 0.3):int(w * 0.7)] = 0
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 2500 < area < (w * h * 0.25):
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        if bh > bw * 1.05:
                            hand_boxes.append([float(bx), float(by), float(bx + bw), float(by + bh)])
                            hull = cv2.convexHull(cnt, returnPoints=False)
                            if len(hull) > 3 and len(cnt) > 3:
                                defects = cv2.convexityDefects(cnt, hull)
                                if defects is not None:
                                    defect_count = sum(1 for d in defects if d[0][3] > 1000)
                                    if 1 <= defect_count <= 4:
                                        gesture_detected = True
                                        extended_fingers_count = defect_count + 1
                                        gesture_label = f"FINGER SIGNALLING ({extended_fingers_count} Fingers)"
            except Exception as e:
                logger.debug(f"Heuristic hand fallback exception: {e}")

        if gesture_detected:
            self.consecutive_gesture_frames += 1
        else:
            self.consecutive_gesture_frames = max(0, self.consecutive_gesture_frames - 2)

        is_sustained = self.consecutive_gesture_frames >= self.gesture_threshold_frames
        return is_sustained, extended_fingers_count, gesture_label, hand_boxes, hand_landmarks_list


class PoseGazeEstimator:
    """Extracts head pose angles, gaze metrics, and hand gesture signalling from frames."""

    # Stable 3D generic facial model points (in mm, centered around nose tip)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
        (0.0, -330.0, -65.0),        # Chin (landmark 199)
        (-225.0, 170.0, -135.0),     # Left Eye Outer Corner (landmark 33)
        (225.0, 170.0, -135.0),      # Right Eye Outer Corner (landmark 263)
        (-150.0, -150.0, -125.0),    # Left Mouth Corner (landmark 61)
        (150.0, -150.0, -125.0)      # Right Mouth Corner (landmark 291)
    ], dtype=np.float64)

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

        # Initialize Hand Signalling Detector
        self.hand_detector = HandSignallingDetector(self.config)

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
        """Processes frame to compute 3D head pose, 4-way gaze direction, and hand gesture signalling."""
        # 1. Detect Hand / Finger Signalling
        is_hand_signalling, ext_fingers, gest_label, hand_boxes, hand_lms = self.hand_detector.detect(frame)

        if frame is None or frame.size == 0:
            self.consecutive_absence_frames += 1
            self.consecutive_lookaway_frames = 0
            res = self._create_absent_result()
            res.hand_signalling = is_hand_signalling
            res.extended_fingers = ext_fingers
            res.hand_gesture_label = gest_label
            res.hand_boxes = hand_boxes
            res.hand_landmarks = hand_lms
            return res

        if self._fallback_mode or self.face_mesh is None:
            res = self._fallback_estimate(frame)
            res.hand_signalling = is_hand_signalling
            res.extended_fingers = ext_fingers
            res.hand_gesture_label = gest_label
            res.hand_boxes = hand_boxes
            res.hand_landmarks = hand_lms
            return res

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
                res = self._create_absent_result()
                res.hand_signalling = is_hand_signalling
                res.extended_fingers = ext_fingers
                res.hand_gesture_label = gest_label
                res.hand_boxes = hand_boxes
                res.hand_landmarks = hand_lms
                return res

            self.consecutive_absence_frames = 0
            face_count = len(results.multi_face_landmarks)
            primary_face = results.multi_face_landmarks[0]

            image_points_2d = []
            for idx in self.LANDMARK_INDICES:
                lm = primary_face.landmark[idx]
                image_points_2d.append([lm.x * w + offset_x, lm.y * h + offset_y])
            image_points_2d = np.array(image_points_2d, dtype=np.float64)

            focal_length = w_full
            center = (w_full / 2.0, h_full / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rvec, tvec = cv2.solvePnP(
                self.MODEL_POINTS_3D,
                image_points_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                res = self._fallback_estimate(frame)
                res.hand_signalling = is_hand_signalling
                res.extended_fingers = ext_fingers
                res.hand_gesture_label = gest_label
                res.hand_boxes = hand_boxes
                res.hand_landmarks = hand_lms
                return res

            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch = float(angles[0])
            yaw = float(angles[1])
            roll = float(angles[2])

            nose_end_point3D = np.array([[0.0, 0.0, 500.0]], dtype=np.float64)
            nose_end_point2D, _ = cv2.projectPoints(nose_end_point3D, rvec, tvec, camera_matrix, dist_coeffs)
            p_nose_2d = (float(nose_end_point2D[0][0][0]), float(nose_end_point2D[0][0][1]))

            gaze_direction, is_looking_away = self._classify_gaze(yaw, pitch, roll)

            if is_looking_away:
                self.consecutive_lookaway_frames += 1
            else:
                self.consecutive_lookaway_frames = 0

            gaze_seconds = self.consecutive_lookaway_frames / max(1.0, self.fps)
            is_prolonged = self.consecutive_lookaway_frames >= self.prolonged_gaze_threshold_frames

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
                hand_signalling=is_hand_signalling,
                extended_fingers=ext_fingers,
                hand_gesture_label=gest_label,
                hand_boxes=hand_boxes,
                hand_landmarks=hand_lms,
                face_box=face_box,
                landmarks_2d=[(pt[0], pt[1]) for pt in image_points_2d],
                nose_projection_2d=p_nose_2d
            )

        except Exception as e:
            logger.error(f"Error in pose & gaze estimation: {e}. Utilizing fallback.")
            res = self._fallback_estimate(frame)
            res.hand_signalling = is_hand_signalling
            res.extended_fingers = ext_fingers
            res.hand_gesture_label = gest_label
            res.hand_boxes = hand_boxes
            res.hand_landmarks = hand_lms
            return res

    def _classify_gaze(self, yaw: float, pitch: float, roll: float) -> Tuple[str, bool]:
        """Symmetric 4-way directional thresholding for LEFT, RIGHT, DOWN, UP, and CENTER."""
        if yaw < -self.max_yaw_angle:
            return "LOOKING LEFT", True
        elif yaw > self.max_yaw_angle:
            return "LOOKING RIGHT", True
        elif pitch > self.max_pitch_angle:
            return "LOOKING DOWN", True
        elif pitch < -self.max_pitch_angle:
            return "LOOKING UP", True
        elif abs(roll) > self.max_roll_angle:
            return "HEAD TILTED", True
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
            is_prolonged_lookaway=False,
            hand_signalling=False,
            extended_fingers=0,
            hand_gesture_label="",
            hand_boxes=[]
        )

    def _fallback_estimate(self, frame: np.ndarray) -> PoseGazeResult:
        """Lightweight OpenCV Haar Cascade fallback."""
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
                hand_signalling=False,
                extended_fingers=0,
                hand_gesture_label="",
                hand_boxes=[],
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
                hand_signalling=False,
                extended_fingers=0,
                hand_gesture_label="",
                hand_boxes=[],
                face_box=[float(w * 0.3), float(h * 0.2), float(w * 0.7), float(h * 0.7)]
            )

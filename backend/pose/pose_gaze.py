"""
Pose, Gaze, and Hand Signalling Estimation Module
Calculates 3D Head Pose (Yaw, Pitch, Roll), Gaze Direction, and detects suspicious finger signalling / hand gestures
using modern MediaPipe Tasks (FaceLandmarker & HandLandmarker) with legacy and mathematical fallbacks.
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import math
import os
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("EviGuard.PoseGaze")


@dataclass
class PoseGazeResult:
    """Encapsulates pose, gaze, and hand gesture metrics extracted from a video frame."""
    face_detected: bool
    face_count: int
    yaw: float  # Negative = left, Positive = right
    pitch: float  # Positive = down (desk/phone), Negative = up (ceiling)
    roll: float  # Negative = tilt left, Positive = tilt right
    gaze_direction: str  # CENTER (FOCUSED), LOOKING LEFT, LOOKING RIGHT, LOOKING DOWN, LOOKING UP
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
    """Detects suspicious hand gesturing / finger counting with landmark stability and temporal persistence."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.fps = float(self.config.get("fps", 30.0))
        self.stability_threshold_px = float(self.config.get("hand_stability_px", 45.0))
        
        # Temporal voting buffer
        self.recent_gestures = deque(maxlen=5)
        self.prev_hand_landmarks: List[List[Tuple[float, float]]] = []
        
        self.task_detector = None
        self.legacy_detector = None
        self._fallback_mode = False

        self._init_detector()

    def _init_detector(self):
        # 1. Try MediaPipe Tasks HandLandmarker
        task_model_path = self.config.get("hand_task_model", "hand_landmarker.task")
        if os.path.exists(task_model_path):
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                base_options = python.BaseOptions(model_asset_path=task_model_path)
                options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
                self.task_detector = vision.HandLandmarker.create_from_options(options)
                logger.info(f"MediaPipe Tasks HandLandmarker initialized successfully from {task_model_path}.")
                return
            except Exception as e:
                logger.warning(f"MediaPipe Tasks HandLandmarker init failed: {e}")

        # 2. Try MediaPipe legacy solutions
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.legacy_detector = mp.solutions.hands.Hands(
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe legacy Hands initialized successfully.")
                return
        except Exception:
            pass

        # 3. Fallback
        logger.info("HandSignallingDetector using skin contour / heuristic fallback.")
        self._fallback_mode = True

    def _calculate_landmark_displacement(self, current_pts: List[Tuple[float, float]]) -> float:
        """Computes mean Euclidean displacement of hand landmarks between consecutive frames."""
        if not self.prev_hand_landmarks or not current_pts:
            return 0.0
        
        prev_pts = self.prev_hand_landmarks[0]
        if len(prev_pts) != len(current_pts):
            return 0.0

        displacements = [
            math.hypot(c[0] - p[0], c[1] - p[1])
            for c, p in zip(current_pts, prev_pts)
        ]
        return float(np.mean(displacements))

    def detect(self, frame: np.ndarray) -> Tuple[bool, int, str, List[List[float]], List[Any]]:
        """Analyzes frame for raised hands with extended fingers (signaling options A/B/C/D)."""
        if frame is None or frame.size == 0:
            self.recent_gestures.append(0)
            self.prev_hand_landmarks = []
            return False, 0, "", [], []

        h, w = frame.shape[:2]
        hand_boxes: List[List[float]] = []
        extended_fingers_count = 0
        gesture_detected = False
        gesture_label = ""
        hand_landmarks_list: List[Any] = []

        # 1. Modern Tasks Detector
        if self.task_detector is not None:
            try:
                import mediapipe as mp
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.task_detector.detect(mp_image)

                if result.hand_landmarks:
                    for lms in result.hand_landmarks:
                        pts = [(lm.x * w, lm.y * h) for lm in lms]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        x1, y1, x2, y2 = max(0, min(xs) - 10), max(0, min(ys) - 10), min(w, max(xs) + 10), min(h, max(ys) + 10)
                        hand_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        hand_landmarks_list.append(pts)

                        wrist_y = lms[0].y
                        if wrist_y < 0.92:
                            fingers = 0
                            # Index (8 vs 6)
                            if lms[8].y < lms[6].y:
                                fingers += 1
                            # Middle (12 vs 10)
                            if lms[12].y < lms[10].y:
                                fingers += 1
                            # Ring (16 vs 14)
                            if lms[16].y < lms[14].y:
                                fingers += 1
                            # Pinky (20 vs 18)
                            if lms[20].y < lms[18].y:
                                fingers += 1
                            # Thumb (4 vs 2)
                            if abs(lms[4].x - lms[2].x) > 0.035 or (lms[4].y < lms[3].y and lms[4].y < lms[2].y):
                                fingers += 1

                            extended_fingers_count = max(extended_fingers_count, fingers)
                            if 1 <= fingers <= 4:
                                gesture_detected = True
                                gesture_label = f"FINGER SIGNALLING ({fingers} Extended Fingers)"
            except Exception as e:
                logger.debug(f"MediaPipe Tasks HandLandmarker exception: {e}")

        # 2. Legacy MediaPipe Detector
        elif self.legacy_detector is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.legacy_detector.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        pts = [(lm.x * w, lm.y * h) for lm in hand_lms.landmark]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        x1, y1, x2, y2 = max(0, min(xs) - 10), max(0, min(ys) - 10), min(w, max(xs) + 10), min(h, max(ys) + 10)
                        hand_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        hand_landmarks_list.append(pts)

                        wrist_y = hand_lms.landmark[0].y
                        if wrist_y < 0.92:
                            fingers = 0
                            if hand_lms.landmark[8].y < hand_lms.landmark[6].y:
                                fingers += 1
                            if hand_lms.landmark[12].y < hand_lms.landmark[10].y:
                                fingers += 1
                            if hand_lms.landmark[16].y < hand_lms.landmark[14].y:
                                fingers += 1
                            if hand_lms.landmark[20].y < hand_lms.landmark[18].y:
                                fingers += 1
                            if abs(hand_lms.landmark[4].x - hand_lms.landmark[2].x) > 0.035:
                                fingers += 1

                            extended_fingers_count = max(extended_fingers_count, fingers)
                            if 1 <= fingers <= 4:
                                gesture_detected = True
                                gesture_label = f"FINGER SIGNALLING ({fingers} Extended Fingers)"
            except Exception as e:
                logger.debug(f"MediaPipe legacy Hands exception: {e}")

        # 3. Fallback skin contour / hand heuristic
        else:
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

        self.prev_hand_landmarks = hand_landmarks_list

        # Append to length-5 buffer and require 2 of 5 frames (rapid yet stable)
        self.recent_gestures.append(1 if gesture_detected else 0)
        is_sustained = sum(self.recent_gestures) >= 2 or (len(self.recent_gestures) < 2 and gesture_detected)

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
        self.absence_threshold = int(absence_cfg.get("absence_frames_threshold", 15))

        self.consecutive_absence_frames = 0
        self.consecutive_lookaway_frames = 0
        
        self.task_detector = None
        self.legacy_face_mesh = None
        self._fallback_mode = False

        # Initialize Hand Signalling Detector
        self.hand_detector = HandSignallingDetector(self.config)

        self._init_face_mesh()

    def _init_face_mesh(self):
        """Initializes MediaPipe FaceLandmarker Task or legacy FaceMesh."""
        task_model_path = self.config.get("face_task_model", "face_landmarker.task")
        if os.path.exists(task_model_path):
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                base_options = python.BaseOptions(model_asset_path=task_model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=2
                )
                self.task_detector = vision.FaceLandmarker.create_from_options(options)
                logger.info(f"MediaPipe Tasks FaceLandmarker initialized successfully from {task_model_path}.")
                return
            except Exception as e:
                logger.warning(f"MediaPipe Tasks FaceLandmarker init failed: {e}")

        # Legacy FaceMesh
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.legacy_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=2,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe legacy FaceMesh initialized successfully.")
                return
        except Exception:
            pass

        logger.info("FaceMesh unavailable. Using OpenCV Haar Cascade fallback.")
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

        # 2. Modern MediaPipe Tasks FaceLandmarker
        if self.task_detector is not None:
            try:
                import mediapipe as mp
                h_full, w_full = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.task_detector.detect(mp_image)

                if not result.face_landmarks:
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
                face_count = len(result.face_landmarks)
                primary_face = result.face_landmarks[0]

                # Extract 2D points for PnP
                image_points_2d = []
                for idx in self.LANDMARK_INDICES:
                    lm = primary_face[idx]
                    image_points_2d.append([lm.x * w_full, lm.y * h_full])
                image_points_2d = np.array(image_points_2d, dtype=np.float64)

                focal_length = float(w_full)
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

                all_x = [lm.x * w_full for lm in primary_face]
                all_y = [lm.y * h_full for lm in primary_face]
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
                logger.debug(f"MediaPipe Tasks FaceLandmarker execution error: {e}")

        # 3. Legacy MediaPipe FaceMesh
        if self.legacy_face_mesh is not None:
            try:
                h_full, w_full = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.legacy_face_mesh.process(rgb_frame)

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
                    image_points_2d.append([lm.x * w_full, lm.y * h_full])
                image_points_2d = np.array(image_points_2d, dtype=np.float64)

                focal_length = float(w_full)
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

                if success:
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

                    all_x = [lm.x * w_full for lm in primary_face.landmark]
                    all_y = [lm.y * h_full for lm in primary_face.landmark]
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
                logger.debug(f"MediaPipe legacy FaceMesh execution error: {e}")

        # 4. Mathematical / Haar Cascade Fallback
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

            yaw = 0.0
            pitch = 0.0
            roll = 0.0

            try:
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                face_roi_gray = gray[y:y + int(fh * 0.65), x:x + fw]
                eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.15, 3)
                if len(eyes) >= 2:
                    eyes_sorted = sorted(eyes, key=lambda e: e[0])
                    e1_center_x = eyes_sorted[0][0] + eyes_sorted[0][2] / 2.0
                    e2_center_x = eyes_sorted[-1][0] + eyes_sorted[-1][2] / 2.0
                    eye_mid_x = (e1_center_x + e2_center_x) / 2.0
                    yaw = float(((eye_mid_x - fw / 2.0) / (fw / 2.0 + 1e-6)) * 40.0)

                    e1_center_y = eyes_sorted[0][1] + eyes_sorted[0][3] / 2.0
                    e2_center_y = eyes_sorted[-1][1] + eyes_sorted[-1][3] / 2.0
                    eye_mid_y = (e1_center_y + e2_center_y) / 2.0
                    pitch = float(((eye_mid_y - fh * 0.35) / (fh * 0.35 + 1e-6)) * 30.0)
            except Exception:
                yaw = 0.0
                pitch = 0.0

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
                face_box=[float(x), float(y), float(x + fw), float(y + fh)]
            )
        except Exception as e:
            logger.debug(f"OpenCV fallback estimation exception: {e}")
            return self._create_absent_result()

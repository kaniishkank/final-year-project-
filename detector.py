"""
EviGuard Core Vision & Proctored Inference Engine
Integrates Ultralytics YOLO26 (NMS-free end-to-end), MediaPipe Hands,
MediaPipe FaceMesh, Smartwatch Wrist ROI analyzer, and Desk Paper Heuristics.
"""

import math
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np

from config import CONFIG
from head_pose import HeadPoseEstimator, HeadPoseResult, CHIN_LANDMARK_IDX

logger = logging.getLogger("EviGuard.Detector")


@dataclass
class DetectedObject:
    """Standardized representation of a detected entity."""
    class_id: int
    class_name: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]
    area: float = 0.0
    aspect_ratio: float = 1.0
    is_malpractice: bool = False
    malpractice_type: str = ""

    def __post_init__(self):
        w = abs(self.box[2] - self.box[0])
        h = abs(self.box[3] - self.box[1])
        self.area = w * h
        self.aspect_ratio = max(w, h) / (min(w, h) + 1e-6)


@dataclass
class HandSignallingResult:
    """Output from MediaPipe Hands gesture and finger counting analysis."""
    is_raised: bool = False
    extended_fingers: int = 0
    gesture_label: str = ""
    is_signalling_malpractice: bool = False
    wrist_chin_dist: float = 0.0
    wrist_point: Optional[Tuple[int, int]] = None
    hand_box: Optional[List[float]] = None
    landmarks: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class DetectionBundle:
    """Unified container for all vision signals in a single video frame."""
    frame_index: int
    timestamp: float
    objects: List[DetectedObject]
    hand_signalling: List[HandSignallingResult]
    head_pose: HeadPoseResult
    face_detected: bool
    face_count: int
    person_count: int
    has_critical_object: bool = False
    smartwatch_detected: bool = False
    paper_detected: bool = False


class ProctoredInferenceEngine:
    """Asynchronous/real-time multi-modal inference pipeline for EviGuard."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or CONFIG
        self.sys_cfg = self.config.get("system", {})
        self.mod_a = self.config.get("module_a_devices", {})
        self.mod_b = self.config.get("module_b_signalling", {})
        self.mod_c = self.config.get("module_c_pose_intrusions", {})

        self.imgsz = int(self.sys_cfg.get("imgsz", 640))
        self.head_pose_solver = HeadPoseEstimator(self.config)

        # 1. Initialize YOLO26 / Ultralytics
        self.yolo_model = None
        self.yolo_pose_model = None
        self._load_yolo_models()

        # 2. Initialize MediaPipe FaceMesh
        self.mp_face_mesh = None
        self.face_mesh = None
        self._load_facemesh()

        # 3. Initialize MediaPipe Hands
        self.mp_hands = None
        self.hands_detector = None
        self._load_hands()

    def _load_yolo_models(self):
        """Loads YOLO26 detection and pose checkpoints with local fallback."""
        try:
            from ultralytics import YOLO
            model_path = self.config.get("models", {}).get("yolo_detection", "yolo26n.pt")
            if not os.path.exists(model_path):
                # Check for alternative local checkpoints
                for alt in ["yolo26n.pt", "yolov8n.pt"]:
                    if os.path.exists(alt):
                        model_path = alt
                        break

            logger.info(f"Loading YOLO26 model from {model_path}...")
            self.yolo_model = YOLO(model_path)
            logger.info("YOLO26 model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load YOLO26 detection model ({e}). Using heuristic fallback.")
            self.yolo_model = None

        # Load Pose model for wrist ROI if available
        try:
            from ultralytics import YOLO
            pose_path = self.config.get("models", {}).get("yolo_pose", "yolo26n-pose.pt")
            if not os.path.exists(pose_path):
                for alt in ["yolo26n-pose.pt", "yolov8n-pose.pt"]:
                    if os.path.exists(alt):
                        pose_path = alt
                        break
            if os.path.exists(pose_path):
                self.yolo_pose_model = YOLO(pose_path)
                logger.info(f"YOLO26 Pose model loaded from {pose_path}.")
        except Exception:
            self.yolo_pose_model = None

    def _load_facemesh(self):
        """Loads MediaPipe FaceMesh model."""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe FaceMesh loaded successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe FaceMesh unavailable ({e}).")
            self.face_mesh = None

    def _load_hands(self):
        """Loads MediaPipe Hands model."""
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands_detector = self.mp_hands.Hands(
                max_num_hands=int(self.mod_b.get("max_num_hands", 2)),
                min_detection_confidence=float(self.mod_b.get("min_detection_confidence", 0.5)),
                min_tracking_confidence=float(self.mod_b.get("min_tracking_confidence", 0.5))
            )
            logger.info("MediaPipe Hands loaded successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe Hands unavailable ({e}).")
            self.hands_detector = None

    def _check_smartwatch_wrist_roi(
        self,
        frame: np.ndarray,
        wrist_point: Tuple[int, int]
    ) -> bool:
        """Inspects radius R=40px around wrist for high-contrast screen / smartwatch contours."""
        if not self.mod_a.get("smartwatch_enable", True) or wrist_point is None:
            return False

        h, w = frame.shape[:2]
        wx, wy = wrist_point
        radius = int(self.mod_a.get("smartwatch_wrist_radius", 40.0))

        x1 = max(0, wx - radius)
        y1 = max(0, wy - radius)
        x2 = min(w, wx + radius)
        y2 = min(h, wy + radius)

        if (x2 - x1) < 15 or (y2 - y1) < 15:
            return False

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Check standard deviation of contrast in wrist ROI
        contrast = np.std(gray)
        if contrast > float(self.mod_a.get("smartwatch_contrast_threshold", 45.0)):
            # Detect circular or rectangular smartwatch bezel contour
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 120.0 < area < 1600.0:
                    return True
        return False

    def _detect_loose_paper_workspace(
        self,
        frame: np.ndarray,
        student_bbox: Optional[Union[List[float], Tuple[float, ...]]] = None
    ) -> List[DetectedObject]:
        """Contour detection with candidate exclusion mask and desk-only ROI for paper cheat notes.

        Safeguards:
        1. Candidate Exclusion Masking (face + 35% downward past chin set to 0).
        2. Desk-Only ROI (desk_roi_y_min = int(h * 0.60)).
        3. Color & Saturation Thresholding (HSV lower_white = [0, 0, 160], upper_white = [180, 45, 255] with (5, 5) kernel).
        4. Contour Geometry Filters (1500 < area < 35000, 0.5 <= aspect_ratio <= 3.0, confidence=0.92).
        """
        paper_objects: List[DetectedObject] = []
        if not self.mod_a.get("paper_heuristic_enable", True) or frame is None or frame.size == 0:
            return paper_objects

        h, w = frame.shape[:2]
        try:
            desk_roi_y_min = int(h * 0.60)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 160], dtype=np.uint8)
            upper_white = np.array([180, 45, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_white, upper_white)

            # Desk-Only ROI: Restrict to lower portion of the frame
            mask[0:desk_roi_y_min, :] = 0

            # Candidate Exclusion Masking
            if student_bbox is not None and len(student_bbox) >= 4:
                sx1, sy1, sx2, sy2 = [int(v) for v in student_bbox[:4]]
                if (sy2 - sy1) > int(h * 0.45):
                    chin_y = sy1 + int((sy2 - sy1) * 0.40)
                    face_h = chin_y - sy1
                    exclusion_y2 = min(h, int(chin_y + face_h * 0.35))
                else:
                    exclusion_y2 = min(h, int(sy2 + (sy2 - sy1) * 0.35))

                ex1 = max(0, sx1)
                ex2 = min(w, sx2)
                ey1 = max(0, sy1)
                ey2 = min(h, exclusion_y2)
                if ex2 > ex1 and ey2 > ey1:
                    mask[ey1:ey2, ex1:ex2] = 0

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (1500.0 < area < 35000.0):
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 8 or bh < 8:
                    continue

                if x <= 4 or y <= 4 or (x + bw) >= (w - 4) or (y + bh) >= (h - 4):
                    continue

                aspect_ratio = float(bw) / float(bh + 1e-6)
                if not (0.5 <= aspect_ratio <= 3.0):
                    continue

                box = [float(x), float(y), float(x + bw), float(y + bh)]
                paper_objects.append(
                    DetectedObject(
                        class_id=73,
                        class_name="unauthorized paper/notes",
                        confidence=0.92,
                        box=box,
                        is_malpractice=True,
                        malpractice_type="UNAUTHORIZED_MATERIAL"
                    )
                )
        except Exception as e:
            logger.debug(f"Paper detection exception: {e}")

        return paper_objects

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0
    ) -> DetectionBundle:
        """Executes full multi-modal detection bundle on a single frame."""
        if frame is None or frame.size == 0:
            return DetectionBundle(
                frame_index=frame_index,
                timestamp=timestamp,
                objects=[],
                hand_signalling=[],
                head_pose=HeadPoseResult(),
                face_detected=False,
                face_count=0,
                person_count=0
            )

        h, w = frame.shape[:2]
        detected_objects: List[DetectedObject] = []
        has_critical = False
        smartwatch_flag = False

        # 1. YOLO26 Detection Inference (Native NMS-Free)
        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=0.25,
                    verbose=False
                )
                for r in results:
                    if r.boxes is None:
                        continue
                    for i in range(len(r.boxes)):
                        box = r.boxes.xyxy[i].cpu().numpy().tolist()
                        conf = float(r.boxes.conf[i].cpu().numpy())
                        cls_id = int(r.boxes.cls[i].cpu().numpy())
                        cls_name = r.names.get(cls_id, str(cls_id)).lower()

                        bw = abs(box[2] - box[0])
                        bh = abs(box[3] - box[1])
                        area = bw * bh
                        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

                        # Module A: Cell Phone (COCO 67)
                        if cls_name in ("cell phone", "phone") or cls_id == 67:
                            min_area = float(self.mod_a.get("phone_min_area", 800.0))
                            min_ar = float(self.mod_a.get("phone_aspect_ratio_min", 1.0))
                            max_ar = float(self.mod_a.get("phone_aspect_ratio_max", 3.5))
                            conf_cut = float(self.mod_a.get("phone_conf_threshold", 0.38))
                            if conf >= conf_cut and area >= min_area and min_ar <= aspect_ratio <= max_ar:
                                detected_objects.append(
                                    DetectedObject(
                                        class_id=67,
                                        class_name="cell phone",
                                        confidence=conf,
                                        box=box,
                                        is_malpractice=True,
                                        malpractice_type="PHONE_DETECTED"
                                    )
                                )
                                has_critical = True

                        # Module A: Book / Notebook (COCO 73)
                        elif cls_name in ("book", "notebook", "paper") or cls_id == 73:
                            if conf >= float(self.mod_a.get("book_conf_threshold", 0.35)):
                                detected_objects.append(
                                    DetectedObject(
                                        class_id=73,
                                        class_name="unauthorized paper/notes",
                                        confidence=conf,
                                        box=box,
                                        is_malpractice=True,
                                        malpractice_type="UNAUTHORIZED_MATERIAL"
                                    )
                                )
                                has_critical = True

                        # Person (COCO 0)
                        elif cls_name == "person" or cls_id == 0:
                            if conf >= float(self.mod_c.get("person_conf_threshold", 0.50)):
                                detected_objects.append(
                                    DetectedObject(
                                        class_id=0,
                                        class_name="person",
                                        confidence=conf,
                                        box=box
                                    )
                                )
            except Exception as e:
                logger.error(f"YOLO26 inference error: {e}")

        # 2. Add Paper Workspace Heuristic
        student_boxes = [d.box for d in detected_objects if d.class_name == "person"]
        student_bbox = student_boxes[0] if student_boxes else None
        paper_dets = self._detect_loose_paper_workspace(frame, student_bbox=student_bbox)
        for pd in paper_dets:
            # Overlap check
            duplicate = False
            for d in detected_objects:
                if d.class_name == "unauthorized paper/notes":
                    if max(0, min(pd.box[2], d.box[2]) - max(pd.box[0], d.box[0])) > 20:
                        duplicate = True
                        break
            if not duplicate:
                detected_objects.append(pd)
                has_critical = True

        # 3. MediaPipe FaceMesh & Head Pose Evaluation
        head_pose_res = HeadPoseResult()
        face_count = 0
        chin_point: Optional[Tuple[int, int]] = None

        if self.face_mesh is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fm_results = self.face_mesh.process(rgb)
                if fm_results.multi_face_landmarks:
                    face_count = len(fm_results.multi_face_landmarks)
                    primary_face = fm_results.multi_face_landmarks[0]
                    head_pose_res = self.head_pose_solver.estimate(primary_face, w, h)
                    chin_point = head_pose_res.chin_2d
            except Exception as e:
                logger.debug(f"FaceMesh error: {e}")

        # 4. MediaPipe Hands & Finger Signalling Analysis
        hand_results: List[HandSignallingResult] = []
        if self.hands_detector is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_data = self.hands_detector.process(rgb)
                if hand_data.multi_hand_landmarks:
                    for h_lms in hand_data.multi_hand_landmarks:
                        wrist_lm = h_lms.landmark[0]
                        wrist_px = (int(wrist_lm.x * w), int(wrist_lm.y * h))
                        wrist_y_norm = wrist_lm.y

                        # Check Smartwatch on Wrist
                        if self._check_smartwatch_wrist_roi(frame, wrist_px):
                            smartwatch_flag = True
                            has_critical = True
                            detected_objects.append(
                                DetectedObject(
                                    class_id=99,
                                    class_name="smartwatch",
                                    confidence=0.88,
                                    box=[float(wrist_px[0]-35), float(wrist_px[1]-35), float(wrist_px[0]+35), float(wrist_px[1]+35)],
                                    is_malpractice=True,
                                    malpractice_type="SMARTWATCH_DETECTED"
                                )
                            )

                        # Raised Hand Criterion: wrist_y < 0.88
                        is_raised = wrist_y_norm < float(self.mod_b.get("wrist_y_max", 0.88))

                        # Face Separation Safeguard: dist(wrist, chin) > 85 px
                        wrist_chin_dist = 999.0
                        face_separated = True
                        if chin_point is not None:
                            wrist_chin_dist = math.hypot(wrist_px[0] - chin_point[0], wrist_px[1] - chin_point[1])
                            if wrist_chin_dist < float(self.mod_b.get("face_separation_min_px", 85.0)):
                                face_separated = False  # Candidate resting chin on hand or scratching ear

                        # Finger Counting (Tip vs PIP joint height)
                        fingers = 0
                        if h_lms.landmark[8].y < h_lms.landmark[6].y:   # Index
                            fingers += 1
                        if h_lms.landmark[12].y < h_lms.landmark[10].y: # Middle
                            fingers += 1
                        if h_lms.landmark[16].y < h_lms.landmark[14].y: # Ring
                            fingers += 1
                        if h_lms.landmark[20].y < h_lms.landmark[18].y: # Pinky
                            fingers += 1
                        if abs(h_lms.landmark[4].x - h_lms.landmark[2].x) > 0.04: # Thumb
                            fingers += 1

                        is_signalling = (
                            is_raised and
                            face_separated and
                            fingers in self.mod_b.get("target_finger_counts", [1, 2, 3, 4])
                        )
                        label = f"MCQ SIGNALLING ({fingers} Fingers)" if is_signalling else ""

                        # Extract hand bounding box
                        h_pts = [(int(lm.x * w), int(lm.y * h)) for lm in h_lms.landmark]
                        xs = [p[0] for p in h_pts]
                        ys = [p[1] for p in h_pts]
                        hand_box = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

                        hand_results.append(
                            HandSignallingResult(
                                is_raised=is_raised,
                                extended_fingers=fingers,
                                gesture_label=label,
                                is_signalling_malpractice=is_signalling,
                                wrist_chin_dist=round(wrist_chin_dist, 1),
                                wrist_point=wrist_px,
                                hand_box=hand_box,
                                landmarks=h_pts
                            )
                        )
            except Exception as e:
                logger.debug(f"Hands detection error: {e}")

        person_count = sum(1 for d in detected_objects if d.class_name == "person")

        return DetectionBundle(
            frame_index=frame_index,
            timestamp=timestamp,
            objects=detected_objects,
            hand_signalling=hand_results,
            head_pose=head_pose_res,
            face_detected=(face_count > 0 or head_pose_res.success),
            face_count=face_count,
            person_count=person_count,
            has_critical_object=has_critical,
            smartwatch_detected=smartwatch_flag,
            paper_detected=len(paper_dets) > 0
        )

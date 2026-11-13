import collections
import math
import os
import sys
import types
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO

# MediaPipe Solutions compatibility shim for Python 3.14+
if not hasattr(mp, "solutions"):
    mp.solutions = types.SimpleNamespace()

    class FaceMeshShim:
        def __init__(self, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
            face_task = "face_landmarker.task" if os.path.exists("face_landmarker.task") else os.path.join(os.path.dirname(__file__), "face_landmarker.task")
            if os.path.exists(face_task):
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                base = python.BaseOptions(model_asset_path=face_task)
                opt = vision.FaceLandmarkerOptions(base_options=base, num_faces=max_num_faces)
                self.detector = vision.FaceLandmarker.create_from_options(opt)
            else:
                self.detector = None

        def process(self, rgb):
            out = types.SimpleNamespace(multi_face_landmarks=[])
            if self.detector is None:
                return out
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = self.detector.detect(mp_img)
                if res.face_landmarks:
                    for face in res.face_landmarks:
                        lms = [types.SimpleNamespace(x=p.x, y=p.y, z=getattr(p, "z", 0.0)) for p in face]
                        out.multi_face_landmarks.append(types.SimpleNamespace(landmark=lms))
            except Exception:
                pass
            return out

    class HandsShim:
        def __init__(self, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.4):
            hand_task = "hand_landmarker.task" if os.path.exists("hand_landmarker.task") else os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
            if os.path.exists(hand_task):
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                base = python.BaseOptions(model_asset_path=hand_task)
                opt = vision.HandLandmarkerOptions(base_options=base, num_hands=max_num_hands)
                self.detector = vision.HandLandmarker.create_from_options(opt)
            else:
                self.detector = None

        def process(self, rgb):
            out = types.SimpleNamespace(multi_hand_landmarks=[])
            if self.detector is None:
                return out
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = self.detector.detect(mp_img)
                if res.hand_landmarks:
                    for hand in res.hand_landmarks:
                        lms = [types.SimpleNamespace(x=p.x, y=p.y, z=getattr(p, "z", 0.0)) for p in hand]
                        out.multi_hand_landmarks.append(types.SimpleNamespace(landmark=lms))
            except Exception:
                pass
            return out

    mp.solutions.face_mesh = types.SimpleNamespace(FaceMesh=FaceMeshShim)
    mp.solutions.hands = types.SimpleNamespace(Hands=HandsShim)


class EviGuardRobustEngine:
    """
    Optimized Lightweight Unified Threat Engine with Specialized Hand-Held Paper & Chit Inspector.
    - YOLOv8 inference at imgsz=320 with FP16 half-precision on GPU / CPU.
    - Target Object Coverage: 'paper', 'chit', 'document', 'note', 'book', 'phone', 'cell phone' (conf >= 0.35)
      -> VIOLATION: UNAUTHORIZED_EXAM_MATERIAL (threat=95, red box).
    - Computer Vision Paper-Slip Extractor: Hand ROI crop, HSV thresholding (V > 200, S < 45),
      solid rectangular contour 600 - 8,000 px held for >= 3 consecutive frames.
    - Facial Core Masking (50px radius suppression around nose tip landmark 1).
    - Alternating Frame Skip (heavy YOLO checks every 2nd frame).
    - 2-of-4 frame rolling temporal stability buffer.
    """
    UNIFIED_LABEL = "VIOLATION: UNAUTHORIZED_EXAM_MATERIAL"
    UNIFIED_THREAT = 95
    UNIFIED_COLOR = (0, 0, 255)  # BGR Red

    def __init__(self):
        candidate_weights = [
            "paper_chit_detector.pt",
            "sentinelvision_large.pt",
            "sentinelvision_yolov8n.pt",
            "yolov8n.pt"
        ]
        weights_path = next((p for p in candidate_weights if os.path.exists(p)), "yolov8n.pt")
        self.model = YOLO(weights_path)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_half = bool(torch.cuda.is_available())

        # MediaPipe detectors
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )

        # Frame skip and temporal persistence buffer (2 out of 4 frames)
        self.frame_count = 0
        self.history = collections.deque(maxlen=4)
        self.CONFIRM_THRESHOLD = 2
        self.cached_detections: List[Dict[str, Any]] = []

        # Handheld paper chit consecutive frames tracker (>= 3 frames required)
        self.chit_consecutive_frames = 0
        # Isolated LAB-space paper chit detector tracker (>= 3 frames required)
        self.lab_chit_streak = 0

        # Compatibility aliases
        self.phone_hist = self.history
        self.book_hist = self.history
        self.paper_hist = self.history
        self.FAST_TRIGGER = self.CONFIRM_THRESHOLD
        self.CONFIRMATION_THRESHOLD = self.CONFIRM_THRESHOLD
        self.target_names = {
            "cell phone": self.UNIFIED_LABEL,
            "phone": self.UNIFIED_LABEL,
            "book": self.UNIFIED_LABEL,
            "notebook": self.UNIFIED_LABEL,
            "paper": self.UNIFIED_LABEL,
            "notes": self.UNIFIED_LABEL,
            "chit": self.UNIFIED_LABEL,
            "document": self.UNIFIED_LABEL,
            "laptop": self.UNIFIED_LABEL,
            "headphone": self.UNIFIED_LABEL,
            "tv": self.UNIFIED_LABEL,
        }
        self.prohibited_keywords = self.target_names

    @property
    def phone_history(self):
        return self.history

    @property
    def book_history(self):
        return self.history

    @property
    def paper_history(self):
        return self.history

    @property
    def chit_history(self):
        return self.history

    @property
    def phone_persistence(self):
        return sum(self.history)

    @property
    def book_persistence(self):
        return sum(self.history)

    @property
    def paper_persistence(self):
        return sum(self.history)

    @property
    def chit_persistence(self):
        return sum(self.history)

    @property
    def phone_frames(self):
        return sum(self.history)

    @property
    def book_frames(self):
        return sum(self.history)

    @property
    def paper_frames(self):
        return sum(self.history)

    @property
    def chit_frames(self):
        return sum(self.history)

    @property
    def PERSISTENCE_THRESHOLD(self):
        return self.CONFIRM_THRESHOLD

    @property
    def FRAME_CONFIRMATION(self):
        return self.CONFIRM_THRESHOLD

    @staticmethod
    def is_invalid_head_overlap(
        phone_box: List[float],
        face_box: Optional[Union[List[float], Tuple[float, ...]]] = None,
        central_features: Optional[List[Tuple[int, int]]] = None,
        hand_boxes: Optional[List[Any]] = None
    ) -> bool:
        """Determines if a candidate detection bounding box falsely covers candidate's inner face core."""
        if not phone_box or len(phone_box) < 4:
            return False

        px1, py1, px2, py2 = [float(v) for v in phone_box[:4]]
        cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0

        if central_features:
            for fx, fy in central_features:
                if (fx - 50 <= cx <= fx + 50) and (fy - 50 <= cy <= fy + 50):
                    return True
            return False

        if face_box and len(face_box) >= 4:
            fx1, fy1, fx2, fy2 = [float(v) for v in face_box[:4]]
            fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            if (fcx - 50 <= cx <= fcx + 50) and (fcy - 50 <= cy <= fcy + 50):
                return True

        return False

    def classify_detection(self, raw_class_name: str, x1: int, y1: int, x2: int, y2: int, conf: float) -> Optional[str]:
        """
        Target Object Coverage:
        Any detection containing 'paper', 'chit', 'document', 'note', 'book', 'phone', or 'cell phone'
        maps directly to VIOLATION: UNAUTHORIZED_EXAM_MATERIAL (Threat 95, Red Box).
        Confidence: >= 0.25 for books; >= 0.35 for all other objects.
        """
        cls_lower = raw_class_name.lower()
        is_book = ("book" in cls_lower or "textbook" in cls_lower or "notebook" in cls_lower)
        min_conf = 0.25 if is_book else 0.35
        if conf < min_conf:
            return None

        prohibited = (
            "paper", "chit", "document", "note", "book", "phone", "cell phone",
            "laptop", "headphone", "earphone", "tv", "display", "screen"
        )
        if any(kw in cls_lower for kw in prohibited):
            return self.UNIFIED_LABEL
        return None

    def _get_secondary_tag(self, raw_class_name: str) -> str:
        """Extract lightweight secondary metadata tag for forensic audit trail."""
        cls_lower = raw_class_name.lower()
        if "chit" in cls_lower or "paper" in cls_lower or "document" in cls_lower or "note" in cls_lower:
            return "notes"
        elif "headphone" in cls_lower or "earphone" in cls_lower:
            return "headphone"
        elif "phone" in cls_lower:
            return "phone"
        elif "book" in cls_lower or "textbook" in cls_lower or "notebook" in cls_lower:
            return "book"
        elif "laptop" in cls_lower or "tv" in cls_lower or "screen" in cls_lower or "display" in cls_lower:
            return "device"
        return "material"

    def _detect_handheld_paper(
        self,
        frame: np.ndarray,
        hand_landmarks_list: List[Any],
        nose_pt: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Computer Vision Paper-Slip Extractor (Inside Hand ROI):
        For tiny chits that raw YOLO might miss:
        When MediaPipe detects hand coordinates (wrist and knuckles), crop the bounding box around the candidate's hand.
        Run a rapid high-contrast white contour check:
        - Convert hand crop to HSV.
        - Threshold for high brightness (V > 200) and low saturation (S < 45) (characteristic of bright white paper against human skin).
        - If a distinct, solid rectangular contour with area between 600 px and 8,000 px is held inside the hand boundary for >= 3 consecutive frames, flag:
          VIOLATION: UNAUTHORIZED_EXAM_MATERIAL (PAPER CHIT DETECTED) (Threat 95, Red Box).
        """
        if not hand_landmarks_list:
            self.chit_consecutive_frames = 0
            return []

        h, w, _ = frame.shape
        found_candidate_chit = False
        detected_chits = []

        for hl in hand_landmarks_list:
            pts = [(int(p.x * w), int(p.y * h)) for p in hl.landmark]
            if not pts:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            min_x = max(0, min(xs) - 25)
            max_x = min(w, max(xs) + 25)
            min_y = max(0, min(ys) - 25)
            max_y = min(h, max(ys) + 25)

            if (max_x - min_x) < 20 or (max_y - min_y) < 20:
                continue

            hand_crop = frame[min_y:max_y, min_x:max_x]
            if hand_crop.size == 0:
                continue

            # Convert to HSV and threshold for white paper
            hsv = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200], dtype=np.uint8)
            upper_white = np.array([180, 45, 255], dtype=np.uint8)
            white_mask = cv2.inRange(hsv, lower_white, upper_white)

            # Rapid morphological clean-up (remove single pixel noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            clean_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Area between 600 px and 8,000 px
                if area < 600 or area > 8000:
                    continue

                rx, ry, rw, rh = cv2.boundingRect(cnt)
                rect_area = rw * rh
                solidity = float(area) / rect_area if rect_area > 0 else 0
                aspect_ratio = float(rw) / rh if rh > 0 else 0

                # Distinct, solid rectangular contour
                if solidity < 0.45 or aspect_ratio < 0.30 or aspect_ratio > 3.3:
                    continue

                # Map back to global frame coordinates
                gx1 = min_x + rx
                gy1 = min_y + ry
                gx2 = gx1 + rw
                gy2 = gy1 + rh
                gcx, gcy = (gx1 + gx2) // 2, (gy1 + gy2) // 2

                # Facial Core Masking: Drop if center point falls inside nose-tip boundary (radius 50px)
                if nose_pt:
                    if math.hypot(gcx - nose_pt[0], gcy - nose_pt[1]) <= 50:
                        continue
                    if abs(gcx - nose_pt[0]) <= 50 and abs(gcy - nose_pt[1]) <= 50:
                        continue

                found_candidate_chit = True
                detected_chits.append((gx1, gy1, gx2, gy2))
                break

        if found_candidate_chit:
            self.chit_consecutive_frames += 1
        else:
            self.chit_consecutive_frames = 0

        # Require >= 3 consecutive frames to flag
        if self.chit_consecutive_frames >= 3 and detected_chits:
            results = []
            for (gx1, gy1, gx2, gy2) in detected_chits:
                results.append({
                    "label": "VIOLATION: UNAUTHORIZED_EXAM_MATERIAL (PAPER CHIT DETECTED)",
                    "box": (gx1, gy1, gx2, gy2),
                    "conf": 0.90,
                    "class_name": "unauthorized_exam_material",
                    "secondary_tag": "notes",
                    "threat": self.UNIFIED_THREAT,
                    "category": "notes",
                    "color": self.UNIFIED_COLOR
                })
            return results

        return []

    def _detect_hand_paper_chit(self, frame: np.ndarray, hand_landmarks: Any) -> Optional[Dict[str, Any]]:
        """
        Isolated LAB-Space Paper Chit Detector:
        Runs only when MediaPipe Hands detects a hand and no YOLO phone/book violation is already active on that hand.
        Uses lighting-invariant CIE LAB color-space (L > 145, neutral chroma: |A-128| < 16, B < 146),
        adaptive contour area (110 - 9000 px) & aspect ratio (0.15 - 6.5), polygon check (4-8 vertices),
        fingertip-proximity contact verification (18px margin with fingertip/DIP landmarks or >= 2 hand landmarks),
        outer boundary contrast check against local hand/skin ring (paper_L - ring_L >= 22.0),
        and fast 2-frame trigger with 1-frame grace period.
        """
        # Initialize internal streak and grace attributes if not already present
        if not hasattr(self, "_chit_detected_streak"):
            self._chit_detected_streak = 0
            self._chit_grace_counter = 0
            self._chit_last_result = None
            self._chit_last_frame = 0

        if hand_landmarks is None or not hasattr(hand_landmarks, "landmark"):
            return None

        h, w, _ = frame.shape
        hand_pts_full = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark]
        if not hand_pts_full:
            return None

        xs = [p[0] for p in hand_pts_full]
        ys = [p[1] for p in hand_pts_full]

        # 1. Hand ROI Extraction with 25px padding to prevent clipping pinched chits extending beyond fingertips
        min_x = max(0, min(xs) - 25)
        max_x = min(w, max(xs) + 25)
        min_y = max(0, min(ys) - 25)
        max_y = min(h, max(ys) + 25)

        if (max_x - min_x) < 30 or (max_y - min_y) < 30:
            return None

        crop = frame[min_y:max_y, min_x:max_x]
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
            return None

        # 2. Lighting-Invariant Real Paper in CIE LAB (L > 145, neutral chroma)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        # White/ruled paper has high L and neutral A/B (~128), unlike skin
        paper_mask = (
            (L > 145) & (np.abs(A.astype(int) - 128) < 16) & (B.astype(int) < 146)
        ).astype(np.uint8) * 255

        # Single 3x3 morphological close to connect ink lines across the paper
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Contour Extraction & Filtering
        contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_candidate = None

        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Lower minimum contour area: 110 to 9000 pixels
                if area < 110 or area > 9000:
                    continue

                rx, ry, rw, rh = cv2.boundingRect(cnt)
                if rw <= 0 or rh <= 0:
                    continue

                # Bounding aspect ratio between 0.15 and 6.5 (supports narrow horizontal or vertical folded slips)
                ar = float(rw) / float(rh)
                if ar < 0.15 or ar > 6.5:
                    continue

                # Contour solidity (area / hull_area) >= 0.55
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0.0
                if solidity < 0.55:
                    continue

                # Polygon approximation has between 4 and 8 vertices (rejects irregular wall gaps)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) < 4 or len(approx) > 8:
                    continue

                # Map back to full-frame coordinates
                gx1 = min_x + rx
                gy1 = min_y + ry
                gx2 = gx1 + rw
                gy2 = gy1 + rh

                # Fingertip-Proximity Contact (18px Margin — Still Blocks Background Wall)
                exp_x1 = max(0, gx1 - 18)
                exp_y1 = max(0, gy1 - 18)
                exp_x2 = min(w, gx2 + 18)
                exp_y2 = min(h, gy2 + 18)

                fingertip_dip_indices = {3, 4, 7, 8, 11, 12, 15, 16, 19, 20}
                fingertip_touch = any(
                    exp_x1 <= hand_pts_full[idx][0] <= exp_x2 and exp_y1 <= hand_pts_full[idx][1] <= exp_y2
                    for idx in fingertip_dip_indices
                    if idx < len(hand_pts_full)
                )
                total_touching = sum(
                    1 for (lx, ly) in hand_pts_full
                    if exp_x1 <= lx <= exp_x2 and exp_y1 <= ly <= exp_y2
                )
                if not (fingertip_touch or total_touching >= 2):
                    continue

                # Boundary Contrast Check (Replaces Internal Canny Density)
                # Dilate candidate contour by ~12px (25x25 ellipse) to create surrounding boundary ring
                paper_cnt_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                cv2.drawContours(paper_cnt_mask, [cnt], -1, 255, -1)

                ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                dilated_mask = cv2.dilate(paper_cnt_mask, ring_kernel)
                ring_mask = cv2.subtract(dilated_mask, paper_cnt_mask)

                paper_pixels = paper_cnt_mask > 0
                ring_pixels = ring_mask > 0

                if np.count_nonzero(paper_pixels) == 0 or np.count_nonzero(ring_pixels) == 0:
                    continue

                paper_L_mean = float(np.mean(L[paper_pixels]))
                ring_L_mean = float(np.mean(L[ring_pixels]))

                # Paper candidate must be at least +22 lightness units brighter than surrounding ring
                if (paper_L_mean - ring_L_mean) < 22.0:
                    continue

                # Valid paper slip detected
                detected_candidate = {
                    "label": "VIOLATION: UNAUTHORIZED_EXAM_MATERIAL (PAPER CHIT)",
                    "box": (gx1, gy1, gx2, gy2),
                    "conf": 0.92,
                    "class_name": "unauthorized_exam_material",
                    "secondary_tag": "notes",
                    "threat": self.UNIFIED_THREAT,
                    "category": "notes",
                    "color": self.UNIFIED_COLOR
                }
                break

        # Fast 2-Frame Trigger (with 1-Frame Grace)
        if detected_candidate is not None:
            self._chit_detected_streak += 1
            self._chit_grace_counter = 0
            self._chit_last_result = detected_candidate
            self._chit_last_frame = self.frame_count

            # 2-frame trigger: on frame 2+, set lab_chit_streak = 2 so process() increments to 3
            if self._chit_detected_streak >= 2:
                self.lab_chit_streak = 2

            return detected_candidate

        # Candidate not detected on this hand: check 1-frame grace for active confirmed chit detections
        is_relevant_hand = True
        if self._chit_last_result is not None and "box" in self._chit_last_result:
            bx1, by1, bx2, by2 = self._chit_last_result["box"]
            hand_cx = float(np.mean(xs))
            hand_cy = float(np.mean(ys))
            is_relevant_hand = (bx1 - 90 <= hand_cx <= bx2 + 90) and (by1 - 90 <= hand_cy <= by2 + 90)

        frame_diff = self.frame_count - self._chit_last_frame
        if (
            is_relevant_hand
            and self._chit_detected_streak >= 2
            and self._chit_grace_counter < 1
            and frame_diff <= 2
            and self._chit_last_result is not None
        ):
            self._chit_grace_counter += 1
            self._chit_last_frame = self.frame_count
            self.lab_chit_streak = 2
            return self._chit_last_result

        # If empty second hand was evaluated while chit is in first hand, don't reset streak
        if not is_relevant_hand:
            return None

        # Reset counters when grace period expires or streak is broken on relevant hand
        self._chit_detected_streak = 0
        self._chit_grace_counter = 0
        self._chit_last_result = None
        self.lab_chit_streak = 0
        return None

    def process(
        self,
        frame: np.ndarray,
        student_face_box: Optional[Union[List[float], Tuple[float, ...]]] = None,
        hand_boxes: Optional[List[Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, int, int, int]]]:
        """
        Processes frame with:
        - Nose core exclusion (50px radius suppression).
        - MediaPipe hand anchor verification (within 65px of wrist/knuckles or desk conf >= 0.55).
        - Specialized Hand-Held Paper & Small Chit Inspector.
        - Alternating frame skip inference at imgsz=320.
        - 2-of-4 frame rolling temporal stability buffer.
        """
        h, w, _ = frame.shape
        self.frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Nose Core Extraction (Landmark 1)
        nose_pt: Optional[Tuple[int, int]] = None
        face_core: Optional[Tuple[int, int, int, int]] = None
        face_out = self.mp_face.process(rgb)
        if face_out.multi_face_landmarks:
            lms = face_out.multi_face_landmarks[0].landmark
            if len(lms) > 1:
                nx, ny = int(lms[1].x * w), int(lms[1].y * h)
                nose_pt = (nx, ny)
                face_core = (nx - 50, ny - 50, nx + 50, ny + 50)
        elif student_face_box is not None and len(student_face_box) >= 4:
            fx1, fy1, fx2, fy2 = [float(v) for v in student_face_box[:4]]
            fcx, fcy = int((fx1 + fx2) / 2), int((fy1 + fy2) / 2)
            nose_pt = (fcx, fcy)
            face_core = (fcx - 50, fcy - 50, fcx + 50, fcy + 50)

        # 2. Hand-Anchor Extraction (Wrist 0, Knuckles 5, 9, 13, 17)
        hand_pts: List[Tuple[int, int]] = []
        hand_lms = []
        hand_out = self.mp_hands.process(rgb)
        if hand_out.multi_hand_landmarks:
            hand_lms = hand_out.multi_hand_landmarks
            for hl in hand_out.multi_hand_landmarks:
                for idx in (0, 5, 9, 13, 17):
                    if idx < len(hl.landmark):
                        pt = hl.landmark[idx]
                        hand_pts.append((int(pt.x * w), int(pt.y * h)))

        if not hand_pts and hand_boxes:
            for hb in hand_boxes:
                if len(hb) >= 4:
                    hx1, hy1, hx2, hy2 = hb[:4]
                    hand_pts.append((int((hx1 + hx2) / 2), int((hy1 + hy2) / 2)))

        # 3. Fast YOLO Inference (Every 2nd Frame at imgsz=320)
        raw_detections: List[Dict[str, Any]] = []
        if self.frame_count % 2 == 0:
            predict_kwargs = {
                "source": frame,
                "conf": 0.25,
                "imgsz": 320,
                "verbose": False
            }
            if self.use_half:
                predict_kwargs["half"] = True
                predict_kwargs["device"] = self.device

            results = self.model.predict(**predict_kwargs)[0]

            for b in results.boxes:
                cls_id = int(b.cls[0].item())
                cls_name = self.model.names[cls_id].lower()
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # A. Classify unified malpractice
                matched_label = self.classify_detection(cls_name, x1, y1, x2, y2, conf)
                if not matched_label:
                    continue

                is_book = ("book" in cls_name or "textbook" in cls_name or "notebook" in cls_name)

                # B. Facial Core Masking (50px radius suppression around nose tip landmark 1)
                # Disabled for books only: open books held near the face must still trigger
                if not is_book and nose_pt:
                    if math.hypot(cx - nose_pt[0], cy - nose_pt[1]) <= 50:
                        continue
                    if abs(cx - nose_pt[0]) <= 50 and abs(cy - nose_pt[1]) <= 50:
                        continue

                # C. MediaPipe Hand-Anchor Verification
                # Bypassed for books only: fingers often grip far edge or hidden behind cover
                if not is_book:
                    is_near_hand = False
                    if hand_pts:
                        for hx, hy in hand_pts:
                            if (x1 - 65 <= hx <= x2 + 65) and (y1 - 65 <= hy <= y2 + 65):
                                is_near_hand = True
                                break
                            if math.hypot(cx - hx, cy - hy) <= 65:
                                is_near_hand = True
                                break

                    is_desk_resting = (conf >= 0.55 and cy >= (h * 0.50))

                    if not (is_near_hand or is_desk_resting):
                        continue

                sec_tag = self._get_secondary_tag(cls_name)
                raw_detections.append({
                    "label": self.UNIFIED_LABEL,
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "class_name": "unauthorized_exam_material",
                    "secondary_tag": sec_tag,
                    "threat": self.UNIFIED_THREAT,
                    "category": sec_tag,
                    "color": self.UNIFIED_COLOR
                })

            # Check Computer Vision Paper-Slip Extractor (Inside Hand ROI)
            paper_chits = self._detect_handheld_paper(frame, hand_lms, nose_pt)
            if paper_chits:
                raw_detections.extend(paper_chits)

            self.cached_detections = raw_detections
        else:
            # Check paper chits also on odd frames to maintain smooth consecutive tracking
            paper_chits = self._detect_handheld_paper(frame, hand_lms, nose_pt)
            raw_detections = list(self.cached_detections)
            if paper_chits:
                raw_detections.extend(paper_chits)

        # Check Isolated LAB-Space Paper Chit Detector on detected hands
        # (Runs only when MediaPipe Hands detects a hand and no YOLO phone/book violation is already active on that hand)
        found_lab_chit = None
        if hand_lms:
            for hl in hand_lms:
                h_xs = [int(p.x * w) for p in hl.landmark]
                h_ys = [int(p.y * h) for p in hl.landmark]
                if h_xs and h_ys:
                    hx1, hx2 = min(h_xs) - 15, max(h_xs) + 15
                    hy1, hy2 = min(h_ys) - 15, max(h_ys) + 15
                    yolo_active = any(
                        d.get("secondary_tag") in ("phone", "book") and
                        not (d["box"][2] < hx1 or d["box"][0] > hx2 or d["box"][3] < hy1 or d["box"][1] > hy2)
                        for d in raw_detections if "box" in d and len(d["box"]) >= 4
                    )
                    if not yolo_active:
                        chit = self._detect_hand_paper_chit(frame, hl)
                        if chit:
                            found_lab_chit = chit
                            break

        if found_lab_chit is not None:
            self.lab_chit_streak += 1
            if self.lab_chit_streak >= 3:
                raw_detections.append(found_lab_chit)
        else:
            self.lab_chit_streak = 0

        # 4. Temporal Stability Buffer: Require positive detection in 2 of the last 4 frames
        # Instant 1-Frame Trigger for Books & 3-Frame Confirmed Paper Chits
        has_book = any(d.get("secondary_tag") == "book" for d in raw_detections)
        has_confirmed_chit = any("PAPER CHIT" in d.get("label", "") for d in raw_detections)

        has_detection = len(raw_detections) > 0
        self.history.append(has_detection)

        confirmed: List[Dict[str, Any]] = []
        if has_book or has_confirmed_chit:
            confirmed = list(raw_detections)
        elif sum(self.history) >= self.CONFIRM_THRESHOLD:
            confirmed = list(raw_detections)

        return confirmed, face_core

    def draw_violations(self, frame: np.ndarray, violations: Any) -> np.ndarray:
        """Visual annotation HUD helper (Unified red box and clean violation banner)."""
        if not violations:
            return frame
        items = violations[0] if isinstance(violations, tuple) else violations
        if not isinstance(items, list):
            return frame
        out = frame.copy()
        for v in items:
            box = v.get("box")
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = map(int, box)
            conf = v.get("conf", 0.0)
            label_val = v.get("label", "")
            if "(PAPER CHIT)" in label_val or "(PAPER CHIT DETECTED)" in label_val:
                caption = f"VIOLATION: UNAUTHORIZED_EXAM_MATERIAL [PAPER CHIT] ({conf*100:.0f}%)" if conf > 0 else "VIOLATION: UNAUTHORIZED_EXAM_MATERIAL [PAPER CHIT]"
            else:
                sec_tag = v.get("secondary_tag", v.get("category", "")).upper()
                tag_str = f" [{sec_tag}]" if sec_tag else ""
                caption = f"VIOLATION: UNAUTHORIZED_EXAM_MATERIAL{tag_str} ({conf*100:.0f}%)" if conf > 0 else f"VIOLATION: UNAUTHORIZED_EXAM_MATERIAL{tag_str}"

            # Red box (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), self.UNIFIED_COLOR, 2)
            (lw, lh), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(out, (x1, max(0, y1 - 22)), (x1 + lw + 8, max(0, y1)), self.UNIFIED_COLOR, -1)
            cv2.putText(out, caption, (x1 + 4, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def reset_tracks(self):
        """Reset temporal rolling buffers."""
        self.history.clear()
        self.cached_detections = []
        self.chit_consecutive_frames = 0
        self.lab_chit_streak = 0
        self.frame_count = 0

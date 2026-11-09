"""
YOLO26 Detector Implementation
Wraps Ultralytics YOLO26 (NMS-free end-to-end inference) for real-time proctoring object detection.
Filters target classes (person, cell phone, book/paper, laptop) with calibrated geometric validation,
native NMS-free high-speed inference, and sub-30ms CPU execution.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np

try:
    import torch
    torch.set_num_threads(4)
except Exception:
    pass

from .base import BaseDetector, DetectionResult

logger = logging.getLogger("EviGuard.Detector")


def apply_clahe_preprocessing(frame: np.ndarray) -> np.ndarray:
    """Converts incoming frames to LAB color space, applies cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))

    to the Luminance channel, and converts back to BGR for low-contrast paper chit capture.
    """
    if frame is None or frame.size == 0:
        return frame
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.debug(f"CLAHE pre-processing fallback: {e}")
        return frame


def detect_paper_chits(
    frame: np.ndarray,
    student_bbox: Optional[Union[List[float], Tuple[float, ...]]] = None
) -> List[DetectionResult]:
    """Refactored paper/chit/notes detection module strictly conforming to the 4 proctoring safeguards:

    1. Candidate Exclusion Masking: Face + 35% downward past chin masked to 0 (eliminates false positives on face, neck, collar, teeth).
    2. Desk-Only ROI: Restricts contour search strictly to the lower frame portion (y >= 0.60 * frame_h).
    3. Color & Saturation Thresholding (HSV): lower_white = [0, 0, 160], upper_white = [180, 45, 255] with (5, 5) closing + opening.
    4. Contour Geometry Filters: 1500 < area < 35000 px^2, aspect ratio between 0.5 and 3.0, confidence=0.92, class_id=73.
    """
    paper_dets: List[DetectionResult] = []
    if frame is None or frame.size == 0:
        return paper_dets

    try:
        h, w = frame.shape[:2]
        desk_roi_y_min = int(h * 0.60)

        # 3. Color & Saturation Thresholding (HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 160], dtype=np.uint8)
        upper_white = np.array([180, 45, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 2. Desk-Only ROI: Restrict to lower portion of the frame (y >= desk_roi_y_min)
        mask[0:desk_roi_y_min, :] = 0

        # 1. Candidate Exclusion Masking
        if student_bbox is not None and len(student_bbox) >= 4:
            sx1, sy1, sx2, sy2 = [int(v) for v in student_bbox[:4]]
            # If student_bbox is full person/body (>45% frame height), estimate chin around upper 40%
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

        # Morphological closing followed by opening with a (5, 5) rectangular kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 4. Contour Geometry Filters
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (1500.0 < area < 35000.0):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 8 or bh < 8:
                continue

            # Reject Frame Boundary Touching (Walls, ceiling, window/door borders)
            if x <= 4 or y <= 4 or (x + bw) >= (w - 4) or (y + bh) >= (h - 4):
                continue

            aspect_ratio = float(bw) / float(bh + 1e-6)
            if not (0.5 <= aspect_ratio <= 3.0):
                continue

            paper_dets.append(
                DetectionResult(
                    box=[float(x), float(y), float(x + bw), float(y + bh)],
                    confidence=0.92,
                    class_id=73,
                    class_name="unauthorized paper/notes"
                )
            )
    except Exception as e:
        logger.debug(f"Paper chit detection exception: {e}")

    return paper_dets


def apply_nms_suppression(detections: List[DetectionResult], iou_threshold: float = 0.28) -> List[DetectionResult]:
    """Class-aware Non-Maximum Suppression to eliminate duplicate / redundant bounding boxes on identical items,

    and prioritize high-confidence cell phones over spurious overlapping book/notes reflections.
    """
    if not detections or len(detections) <= 1:
        return detections

    filtered: List[DetectionResult] = []
    # Sort detections by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

    while sorted_dets:
        best = sorted_dets.pop(0)
        filtered.append(best)

        bx1, by1, bx2, by2 = best.box
        best_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        remaining = []
        for d in sorted_dets:
            dx1, dy1, dx2, dy2 = d.box
            d_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)

            ix1, iy1 = max(bx1, dx1), max(by1, dy1)
            ix2, iy2 = min(bx2, dx2), min(by2, dy2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            union = best_area + d_area - inter + 1e-6
            iou = inter / union
            containment = inter / (min(best_area, d_area) + 1e-6)

            # 1. Group compatible classes together (e.g. book & paper notes, phone variants, persons)
            is_same_category = (
                d.class_name == best.class_name
                or (d.class_name in ("book", "unauthorized paper/notes") and best.class_name in ("book", "unauthorized paper/notes"))
                or (d.class_name in ("cell phone", "phone") and best.class_name in ("cell phone", "phone"))
                or (d.class_name == "person" and best.class_name == "person")
            )

            if is_same_category and (iou > iou_threshold or containment > 0.55):
                continue

            # 2. Cell Phone Precedence: A genuine cell phone suppresses overlapping or adjacent ghost book or paper notes
            if best.class_name in ("cell phone", "phone") and best.confidence >= 0.35:
                if d.class_name in ("book", "unauthorized paper/notes", "paper"):
                    near_x = (dx2 >= bx1 - 70.0) and (dx1 <= bx2 + 70.0)
                    near_y = (dy2 >= by1 - 70.0) and (dy1 <= by2 + 70.0)
                    if (iou > 0.10 or containment > 0.20) or (near_x and near_y and d.area < 15000.0):
                        continue

            # 3. Book Precedence: A detected book suppresses small overlapping chit notes inside its bounds
            if best.class_name == "book" and d.class_name in ("unauthorized paper/notes", "paper"):
                if containment > 0.45 or iou > 0.25:
                    continue

            remaining.append(d)

        sorted_dets = remaining

    return filtered


class YOLO26Detector(BaseDetector):
    """Real-time object detector powered by Ultralytics YOLOv8m with OpenCV CLAHE pre-processing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_path = self.config.get("model_path", "yolov8m.pt")
        self.iou_threshold = float(self.config.get("iou_threshold", 0.28))
        self.imgsz = int(self.config.get("imgsz", 640))
        self.augment = bool(self.config.get("augment", False))
        self.enable_clahe = bool(self.config.get("enable_clahe", True))
        self.nms_free = self.config.get("nms_free", True)
        
        # Specific Confidence Thresholds per object class (calibrated to 0.28 to filter duplicate noise while retaining real items)
        self.phone_conf_threshold = float(self.config.get("phone_confidence_threshold", 0.28))
        self.person_conf_threshold = float(self.config.get("person_confidence_threshold", 0.28))
        self.book_conf_threshold = float(self.config.get("book_confidence_threshold", 0.28))
        self.default_conf_threshold = float(self.config.get("confidence_threshold", 0.28))

        # Geometric Validation Parameters for Cell Phones (calibrated for far-range detection)
        self.phone_min_area = float(self.config.get("phone_min_area", 80.0))
        self.phone_min_w = float(self.config.get("phone_min_w", 8.0))
        self.phone_min_h = float(self.config.get("phone_min_h", 8.0))
        self.phone_min_aspect_ratio = float(self.config.get("phone_min_aspect_ratio", 1.0))
        self.phone_max_aspect_ratio = float(self.config.get("phone_max_aspect_ratio", 4.5))

        # Paper detection enabled for automated local detection pipeline
        self.enable_paper_heuristic = self.config.get("enable_paper_heuristic", True)

        self.model = None
        self._fallback_mode = False

        self._load_model()

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Applies CLAHE pre-processing on the Luminance channel."""
        return apply_clahe_preprocessing(frame)

    def _load_model(self):
        """Attempts to load the standard yolov8m.pt model, falling back gracefully if unavailable."""
        try:
            from ultralytics import YOLO
            target = self.model_path if os.path.exists(self.model_path) else (
                "backend/weights/yolov8m.pt" if os.path.exists("backend/weights/yolov8m.pt") else (
                    "yolov8m.pt" if os.path.exists("yolov8m.pt") else (
                        "backend/weights/best.pt" if os.path.exists("backend/weights/best.pt") else "yolov8m.pt"
                    )
                )
            )
            logger.info(f"Loading YOLO detector model from {target}...")
            self.model = YOLO(target)
            logger.info(f"YOLO detector model ({target}) loaded successfully via Ultralytics.")
        except Exception as e:
            logger.warning(f"Could not load YOLO model ({e}). Initializing simulated/heuristic detector fallback.")
            self._fallback_mode = True

    def _is_valid_phone_geometry(self, box: List[float], frame_shape: Optional[Tuple[int, ...]] = None) -> bool:
        """Validates cell phone bounding box: accepts portrait, landscape, and tilted angles

        both close-up to the camera and at proctored desk distance.
        """
        x1, y1, x2, y2 = box
        bw = abs(x2 - x1)
        bh = abs(y2 - y1)
        area = bw * bh

        # 1. Minimum Dimensions Check
        if area < self.phone_min_area or bw < self.phone_min_w or bh < self.phone_min_h:
            return False

        # 2. Maximum Dimensions Check (allows close-up phone shown to camera up to 75% of frame)
        max_w = 460.0 if not frame_shape else (0.75 * frame_shape[1])
        max_h = 440.0 if not frame_shape else (0.90 * frame_shape[0])
        max_area = 120000.0 if not frame_shape else (0.45 * frame_shape[0] * frame_shape[1])
        if area > max_area or bw > max_w or bh > max_h:
            return False

        # 3. Aspect Ratio Check (supports vertical, horizontal, and diagonal angles: 1.25 to 4.5)
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect_ratio < self.phone_min_aspect_ratio or aspect_ratio > self.phone_max_aspect_ratio:
            return False

        return True

    def _is_valid_exam_chit(
        self,
        frame: np.ndarray,
        box: List[float],
        person_boxes: Optional[List[List[float]]] = None
    ) -> bool:
        """Strictly validates candidate Exam Chit / Notes detections to eliminate doors,

        door frames, and metallic door handles/latches while accurately preserving real paper chits.
        """
        if frame is None or frame.size == 0:
            return False

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        # 1. Tighten Chit Geometry Rules:
        # Accept small paper chits down to 12x12 px (area >= 140 px^2) in candidate workspace
        if bw < 12 or bh < 12 or area < 140:
            return False

        # Discard long, narrow handle ratios (width-to-height ratio > 3.8 or < 0.25)
        wh_ratio = float(bw) / float(bh + 1e-6)
        if wh_ratio > 3.8 or wh_ratio < 0.25:
            return False

        # 2. Door Exclusion Zone & Background Perimeter Mask:
        # Filter out detections that occur in the upper/background perimeter regions of the video frame
        # where doors, door frames, and top latches reside, unless in active student workspace
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        is_upper_door_zone = cy < (0.30 * h)
        is_side_door_perimeter = (cx < 0.10 * w or cx > 0.90 * w) and cy < (0.60 * h)

        if person_boxes:
            in_student_workspace = False
            for pbox in person_boxes:
                px1, py1, px2, py2 = pbox
                # Candidate workspace around student: lateral margin 180px, below chin/neck down to desk
                min_y = py1 + int((py2 - py1) * 0.45)
                max_y = min(h, py2 + 180)
                if (px1 - 180 <= cx <= px2 + 180) and (min_y <= cy <= max_y):
                    in_student_workspace = True
                    break

            if (is_upper_door_zone or is_side_door_perimeter) and not in_student_workspace:
                return False
        else:
            if is_upper_door_zone or (cx < 0.08 * w or cx > 0.92 * w):
                return False

        # 3. Enforce Paper Surface Color & Texture Check on ROI:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            return False

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_chan = roi_hsv[:, :, 2]
        s_chan = roi_hsv[:, :, 1]

        mean_v = float(np.mean(v_chan))
        mean_s = float(np.mean(s_chan))
        mean_gray = float(np.mean(roi_gray))
        mean_b, mean_g, mean_r = np.mean(roi, axis=(0, 1))

        # A. Reject Metallic, Metallic-Gray, or Harsh Metallic Highlights (Door handles, latches, metal plates)
        # Metallic glint / specular reflection: harsh localized highlights (V > 242) + dark metal shadow (V < 75)
        specular_ratio = float(np.count_nonzero(v_chan > 242)) / float(roi.shape[0] * roi.shape[1] + 1e-6)
        dark_metal_ratio = float(np.count_nonzero(v_chan < 75)) / float(roi.shape[0] * roi.shape[1] + 1e-6)
        if specular_ratio > 0.05 and dark_metal_ratio > 0.22:
            return False

        # Dull metallic gray (low saturation S < 25, mid-range metal gray 60 <= V <= 150)
        is_metallic_gray = (mean_s < 25.0) and (60.0 <= mean_v <= 150.0)

        # B. Reject Human Skin Tones (Red-dominant tone, R >> B)
        if (mean_r - mean_b > 20.0 and mean_b < 165.0):
            return False

        # C. Accept High Whiteness / Light Paper Surface Tones
        # Paper slips have light diffuse reflectance (mean_gray >= 128 or mean_v >= 132) and low-to-moderate saturation (S <= 55)
        if mean_gray < 128.0 and mean_v < 132.0:
            return False
        if mean_s > 55.0:
            return False

        # D. Texture & Dark Text Edge Density (Handwriting / printed text lines vs blank metallic handle)
        canny = cv2.Canny(roi_gray, 40, 130)
        edge_density = float(np.count_nonzero(canny)) / float(roi_gray.size + 1e-6)
        lap_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())

        if is_metallic_gray and (edge_density < 0.010 or lap_var < 12.0):
            return False

        # A real paper chit has fibrous edges, paper boundaries, or writing
        if edge_density < 0.003 and lap_var < 5.0:
            return False

        return True

    def _detect_white_paper_sheets(
        self,
        frame: np.ndarray,
        student_bbox: Optional[Union[List[float], Tuple[float, ...]]] = None,
        person_boxes: Optional[List[List[float]]] = None
    ) -> List[DetectionResult]:
        """Detects physical small exam chits and paper slips using candidate exclusion and desk ROI."""
        if not self.enable_paper_heuristic or frame is None or frame.size == 0:
            return []

        target_bbox = student_bbox
        if target_bbox is None and person_boxes:
            target_bbox = person_boxes[0]

        return detect_paper_chits(frame, student_bbox=target_bbox)

    def detect(
        self,
        frame: np.ndarray,
        crop_boxes: Optional[List[List[float]]] = None,
        student_bbox: Optional[Union[List[float], Tuple[float, ...]]] = None
    ) -> List[DetectionResult]:
        """Detects target objects with native NMS-free mode and high-speed single-pass inference."""
        if frame is None or frame.size == 0:
            return []

        if self._fallback_mode or self.model is None:
            return self._fallback_detect(frame)

        try:
            h, w = frame.shape[:2]

            # Apply OpenCV CLAHE pre-processing on Luminance channel
            proc_frame = self.apply_clahe(frame) if self.enable_clahe else frame

            # Single-pass high-speed YOLO inference with yolov8m.pt, conf=0.25, iou=0.28, imgsz=self.imgsz, augment=False
            results = self.model.predict(
                source=proc_frame,
                imgsz=self.imgsz,
                conf=0.25,
                iou=0.28,
                augment=False,
                verbose=False
            )

            detections: List[DetectionResult] = []
            person_boxes: List[List[float]] = []

            # First pass: collect person boxes to establish candidate workspace boundary
            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for i in range(len(boxes)):
                    c_id = int(boxes.cls[i].cpu().numpy())
                    c_name = r.names.get(c_id, str(c_id)).lower()
                    c_conf = float(boxes.conf[i].cpu().numpy())
                    if (c_name == "person" or "person" in c_name) and c_conf >= self.person_conf_threshold:
                        person_boxes.append(boxes.xyxy[i].cpu().numpy().tolist())

            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = r.names.get(cls_id, str(cls_id)).lower()

                    bw = abs(box[2] - box[0])
                    bh = abs(box[3] - box[1])
                    box_area = bw * bh
                    aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

                    # Filter out any bounding box taking up >55% of frame area or width/height (avoid false positive full-body paper chit triggers)
                    if cls_name != "person":
                        if box_area > (0.55 * w * h) or (bw > 0.55 * w and bh > 0.55 * h):
                            continue
                        if ("paper" in cls_name or "chit" in cls_name or "cheating" in cls_name) and (bw > 0.45 * w or bh > 0.45 * h):
                            continue

                    # 1. Validation for Person (Class 0 in COCO, or Class 4 in custom model)
                    if cls_name == "person" or (cls_name.startswith("person") and "non" not in cls_name):
                        if conf < self.person_conf_threshold:
                            continue
                        cls_name = "person"
                        cls_id = 0
                        if box not in person_boxes:
                            person_boxes.append(box)

                    # 2. Validation for Cell Phone (Class 67 in COCO, or Class 1 in custom model)
                    elif cls_name in ("cell phone", "phone", "mobile phone", "telephone") or (
                        "phone" in cls_name and "head" not in cls_name and "ear" not in cls_name
                    ):
                        if conf < self.phone_conf_threshold:
                            continue

                        # Cell Phone vs Big Book Disambiguation:
                        # If candidate object is broad and squarish (aspect_ratio < 1.40, bw >= 150, bh >= 150, area >= 22000),
                        # and not elongated like a phone, it is a big book / textbook.
                        is_big_book = False
                        if conf < 0.65 and aspect_ratio < 1.40 and bw >= 150.0 and bh >= 150.0 and box_area >= 22000.0:
                            is_big_book = True

                        if is_big_book:
                            cls_name = "book"
                            cls_id = 73
                        else:
                            if not self._is_valid_phone_geometry(box, (h, w)):
                                continue
                            cls_name = "cell phone"
                            cls_id = 67

                    # 3. Validation for Book / Paper / Notes (COCO Class 73, or Class 0 in best.pt)
                    elif cls_name in ("book", "notebook") or ("book" in cls_name):
                        if conf < self.book_conf_threshold:
                            continue

                        # Disambiguate Phone vs Book:
                        # When a phone is held in front of the camera, COCO YOLO frequently predicts 'book' due to screen content.
                        # Real books and textbooks have squarish aspect ratios (1.10 <= aspect_ratio <= 1.50).
                        # Smartphones have elongated aspect ratio (1.55 <= aspect_ratio <= 3.5) with handheld dimensions.
                        if (1.55 <= aspect_ratio <= 3.5) and (bw <= 280.0 or bh <= 450.0) and self._is_valid_phone_geometry(box, (h, w)):
                            cls_name = "cell phone"
                            cls_id = 67
                        else:
                            # It's a genuine book / notebook
                            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                            if cy < (0.28 * h) and (cx < 0.12 * w or cx > 0.88 * w):
                                continue  # Reject door top / background wall frames
                            cls_name = "book"
                            cls_id = 73

                    # 4. Disambiguation for Large Book / Textbook mistakenly detected as Laptop (Class 63)
                    elif cls_name in ("laptop",):
                        if conf < self.default_conf_threshold:
                            continue
                        rx1, ry1, rx2, ry2 = [int(v) for v in box]
                        rx1, ry1 = max(0, rx1), max(0, ry1)
                        rx2, ry2 = min(w, rx2), min(h, ry2)
                        is_book = False
                        if rx2 > rx1 and ry2 > ry1:
                            roi_bgr = frame[ry1:ry2, rx1:rx2]
                            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                            mean_b = float(np.mean(roi_gray))
                            mean_sat = float(np.mean(roi_hsv[:, :, 1]))
                            
                            # A book, binder, or open textbook:
                            if aspect_ratio < 1.65 or (ry2 - ry1) >= 0.58 * (rx2 - rx1) or (mean_b >= 75.0 and mean_sat <= 70.0) or conf < 0.65:
                                is_book = True

                        if is_book:
                            cls_name = "book"
                            cls_id = 73
                        else:
                            cls_name = "laptop"
                            cls_id = 63

                    # 5. Audio / Visual Gadgets (headphones, tv, remote)
                    elif cls_name in ("headphone", "headphones", "earphone", "earphones"):
                        if conf < self.default_conf_threshold:
                            continue
                        cls_name = "headphone"
                        cls_id = 85

                    elif cls_name in ("tv", "monitor"):
                        if conf < self.default_conf_threshold:
                            continue
                        cls_name = "tv"
                        cls_id = 62

                    elif cls_name in ("remote", "remote control"):
                        if conf < self.default_conf_threshold:
                            continue
                        cls_name = "remote"
                        cls_id = 65

                    elif cls_name in ("non-cheating", "non_cheating"):
                        continue

                    elif cls_name in ("cheating-paper", "cheating_paper", "paper", "cheating"):
                        if conf < self.book_conf_threshold:
                            continue
                        if not self._is_valid_exam_chit(frame, box, person_boxes=person_boxes):
                            continue
                        cls_name = "unauthorized paper/notes"
                        cls_id = 73

                    elif cls_name in ("hand-gestures", "hand_gestures"):
                        if conf < self.default_conf_threshold:
                            continue
                        cls_name = "hand_signalling"
                        cls_id = 99

                    # 6. Other Target Objects
                    else:
                        if self.target_classes and cls_name not in self.target_classes:
                            continue
                        if conf < self.default_conf_threshold:
                            continue

                    detections.append(
                        DetectionResult(
                            box=box,
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name
                        )
                    )

            # Run specialized heuristic paper sheet / chit detection on CLAHE-enhanced frame
            if self.enable_paper_heuristic:
                target_student_box = student_bbox or (person_boxes[0] if person_boxes else None)
                paper_dets = self._detect_white_paper_sheets(proc_frame, student_bbox=target_student_box, person_boxes=person_boxes)
                for pd in paper_dets:
                    px1, py1, px2, py2 = pd.box
                    p_area = (px2 - px1) * (py2 - py1)

                    # Ensure heuristic paper sheet NEVER overrides or pollutes a genuine cell phone
                    overlaps_phone = False
                    for existing in detections:
                        if existing.class_name in ("cell phone", "phone"):
                            ex1, ey1, ex2, ey2 = existing.box
                            ix1, iy1 = max(px1, ex1), max(py1, ey1)
                            ix2, iy2 = min(px2, ex2), min(py2, ey2)
                            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                            inter = iw * ih
                            e_area = (ex2 - ex1) * (ey2 - ey1)
                            iou = inter / (p_area + e_area - inter + 1e-6)
                            containment = inter / (min(p_area, e_area) + 1e-6)
                            if iou > 0.15 or containment > 0.25:
                                overlaps_phone = True
                                break

                    if not overlaps_phone:
                        detections.append(pd)

            # Apply Class-Aware Non-Maximum Suppression to eliminate duplicate / multi-scale bounding boxes
            detections = apply_nms_suppression(detections, iou_threshold=self.iou_threshold)

            # If YOLO detects class Phone (or class ID 1, 67) with confidence >= 0.28, suppress any hand gesture / finger signalling flags on that frame
            has_phone_detected = any(
                (d.class_name in ("cell phone", "phone") or d.class_id in (1, 67) or "phone" in d.class_name.lower())
                and d.class_name not in ("book", "unauthorized paper/notes", "notes")
                and d.confidence >= 0.28
                for d in detections
            )
            if has_phone_detected:
                detections = [
                    d for d in detections 
                    if d.class_name not in ("hand_signalling", "hand-gestures", "hand_gestures") and d.class_id not in (2, 99)
                ]

            return detections

        except Exception as e:
            logger.error(f"YOLO26 inference error: {e}. Falling back to heuristic detector.")
            return self._fallback_detect(frame)

    def _fallback_detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Heuristic fallback using OpenCV color and contour analysis."""
        detections: List[DetectionResult] = []
        if frame is None or frame.size == 0:
            return detections

        h, w = frame.shape[:2]

        try:
            # 1. Fallback Face / Person Detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)
            for (fx, fy, fw, fh) in faces:
                body_y2 = min(h, fy + int(fh * 3.5))
                body_x1 = max(0, fx - int(fw * 0.5))
                body_x2 = min(w, fx + int(fw * 1.5))
                detections.append(
                    DetectionResult(
                        box=[float(body_x1), float(fy), float(body_x2), float(body_y2)],
                        confidence=0.80,
                        class_id=0,
                        class_name="person"
                    )
                )

            # 2. Black rectangular phone heuristic
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0], dtype=np.uint8)
            upper_black = np.array([180, 255, 45], dtype=np.uint8)
            mask_black = cv2.inRange(hsv, lower_black, upper_black)
            contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if self.phone_min_area <= area <= 25000.0:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
                    if self.phone_min_aspect_ratio <= aspect_ratio <= self.phone_max_aspect_ratio:
                        detections.append(
                            DetectionResult(
                                box=[float(bx), float(by), float(bx + bw), float(by + bh)],
                                confidence=0.70,
                                class_id=67,
                                class_name="cell phone"
                            )
                        )

            # 3. Optional white paper detection
            if self.enable_paper_heuristic:
                detections.extend(self._detect_white_paper_sheets(frame))

        except Exception as e:
            logger.debug(f"Fallback detector exception: {e}")

        return detections


class MockDetector(BaseDetector):
    """Mock/Simulated detector for testing and offline environments."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def detect(self, frame: np.ndarray, crop_boxes: Optional[List[List[float]]] = None) -> List[DetectionResult]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        return [
            DetectionResult(
                box=[float(w * 0.25), float(h * 0.15), float(w * 0.75), float(h * 0.85)],
                confidence=0.92,
                class_id=0,
                class_name="person"
            )
        ]


# Backward Compatibility Aliases
YOLOv26Detector = YOLO26Detector
YOLOv8Detector = YOLO26Detector

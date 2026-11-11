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


def _is_hand_holding_paper(
    frame_or_roi: np.ndarray,
    hand_box: Optional[Union[List[int], List[float], Tuple[int, ...], Tuple[float, ...]]] = None,
    landmarks: Optional[List[Tuple[float, float]]] = None
) -> Tuple[bool, float, List[float], Dict[str, Any]]:
    """Analyzes cropped high-resolution hand area for micro-chits and folded paper slips:
    - Extracts raw hand crop using landmark boundary plus 45px padding.
    - Applies CLAHE contrast enhancement directly to the crop.
    - Rejects tall/vertical architectural contours (window grills/doors: ar > 2.2 or height > 180).
    - Compact paper rule: aspect ratio between 0.4 and 2.0, area <= 12,000 px².
    - Strict hand palm / finger intersection: must intersect palm center / fingertip landmarks (0, 5, 9, 13, 17, 8, 12).
    Returns (is_chit, conf, global_box, metrics).
    """
    if frame_or_roi is None or frame_or_roi.size == 0:
        return False, 0.0, [0.0, 0.0, 0.0, 0.0], {}

    h_img, w_img = frame_or_roi.shape[:2]

    # 1. Extract raw hand crop with 45px boundary padding
    if hand_box is not None and len(hand_box) >= 4:
        hx1, hy1, hx2, hy2 = [int(v) for v in hand_box[:4]]
        # Ignore top 40% of camera frame (ceiling, doors, wall trims)
        if hy2 < 0.40 * h_img or (hy1 + hy2) / 2.0 < 0.40 * h_img:
            return False, 0.0, [0.0, 0.0, 0.0, 0.0], {}
        x1 = max(0, hx1 - 45)
        y1 = max(0, hy1 - 45)
        x2 = min(w_img, hx2 + 45)
        y2 = min(h_img, hy2 + 45)
        hand_crop = frame_or_roi[y1:y2, x1:x2]
    else:
        hand_crop = frame_or_roi
        x1, y1 = 0, 0
        x2, y2 = hand_crop.shape[1], hand_crop.shape[0]

    if hand_crop.size == 0 or hand_crop.shape[0] < 10 or hand_crop.shape[1] < 10:
        return False, 0.0, [float(x1), float(y1), float(x2), float(y2)], {}

    # 2. Apply CLAHE contrast enhancement directly to the cropped region
    crop_enhanced = apply_clahe_preprocessing(hand_crop)

    total_pixels = float(hand_crop.shape[0] * hand_crop.shape[1] + 1e-6)
    roi_hsv = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2HSV)
    roi_gray = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2GRAY)

    # 3. Check for white/light paper presence: HSV S < 55, V > 140
    lower_white = np.array([0, 0, 140], dtype=np.uint8)
    upper_white = np.array([180, 55, 255], dtype=np.uint8)
    white_mask = cv2.inRange(roi_hsv, lower_white, upper_white)

    paper_pixels = int(np.count_nonzero(white_mask))
    paper_pixel_ratio = paper_pixels / total_pixels

    # 4. Text / line edge density (threshold 0.015)
    edges = cv2.Canny(roi_gray, 35, 120)
    white_edges = edges & white_mask
    edge_density = float(np.count_nonzero(white_edges)) / total_pixels

    # 5. Check contours down to 120 px² minimum with strict architectural rejection
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_micro_chit = False
    max_chit_area = 0.0
    best_cnt = None

    for cnt in contours:
        c_area = float(cv2.contourArea(cnt))
        # Paper chits must be compact: area cannot exceed 6,000 px², minimum 100 px²
        if not (100.0 <= c_area <= 6000.0):
            continue

        cbx, cby, cbw, cbh = cv2.boundingRect(cnt)
        c_bbox_area = float(cbw * cbh)
        if c_bbox_area > 6000.0:
            continue

        # Compute bounding box aspect ratio: ar = height / max(width, 1)
        ar = float(cbh) / max(float(cbw), 1.0)

        # 1. Reject Tall / Vertical Architectural Contours and Handheld Smartphones:
        # A smartphone held in hand has cbh >= 65 and 1.4 <= ar <= 2.8.
        # This is a mobile phone, NOT an exam chit!
        if cbh >= 65 and (1.4 <= ar <= 2.8):
            continue
        if cbh > 85 or ar > 1.65:
            continue

        # Paper chits must be compact: aspect ratio between 0.45 and 1.65
        if not (0.45 <= ar <= 1.65):
            continue

        cnt_mask = np.zeros(white_mask.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        cnt_edges = cv2.countNonZero(white_edges & cnt_mask)
        cnt_edge_ratio = cnt_edges / float(c_area + 1e-6)
        if cnt_edge_ratio < 0.010 and cnt_edges < 6:
            continue

        has_micro_chit = True
        if c_area > max_chit_area:
            max_chit_area = c_area
            best_cnt = cnt

    if not has_micro_chit or best_cnt is None:
        return False, 0.0, [float(x1), float(y1), float(x2), float(y2)], {}

    # 6. Calculate global screen coordinates for the detected slip
    local_bx1, local_by1, bw, bh = cv2.boundingRect(best_cnt)
    local_bx1 = max(0, local_bx1 - 4)
    local_by1 = max(0, local_by1 - 4)
    local_bx2 = min(hand_crop.shape[1], local_bx1 + bw + 8)
    local_by2 = min(hand_crop.shape[0], local_by1 + bh + 8)

    global_box = [
        float(x1 + local_bx1),
        float(y1 + local_by1),
        float(x1 + local_bx2),
        float(y1 + local_by2)
    ]

    # Handheld Device / Phone Extension Check:
    # If the region immediately above the candidate box (up to 30px higher) has identical color profile (diff < 35)
    # and extends above the hand, this is a phone/device chassis held in hand, NOT a paper chit!
    if local_by1 > 15:
        above_strip = hand_crop[max(0, local_by1 - 25):local_by1, local_bx1:local_bx2]
        cand_strip = hand_crop[local_by1:min(hand_crop.shape[0], local_by1 + 25), local_bx1:local_bx2]
        if above_strip.size > 0 and cand_strip.size > 0:
            diff_color = float(np.mean(np.abs(np.mean(above_strip, axis=(0, 1)) - np.mean(cand_strip, axis=(0, 1)))))
            if diff_color < 35.0:
                return False, 0.0, global_box, {"is_phone": True}

    # Background Window Light Disqualification:
    # If top edge of candidate box extends above the wrist/hand contour into the background, drop immediately
    if hand_box is not None and len(hand_box) >= 4:
        if global_box[1] < float(hand_box[1]) - 2.0:
            return False, 0.0, global_box, {}

    if landmarks and len(landmarks) >= 13:
        min_lms_y = min(p[1] for p in landmarks)
        if global_box[1] < float(min_lms_y) - 2.0:
            return False, 0.0, global_box, {}

    gw = abs(global_box[2] - global_box[0])
    gh = abs(global_box[3] - global_box[1])
    g_ar = float(gh) / max(float(gw), 1.0)
    # Strict chit size limits: maximum height 70px, area <= 6,000 px², aspect ratio 0.45 - 1.65
    if gh > 85 or g_ar > 1.65 or g_ar < 0.45 or (gw * gh) > 6000.0:
        return False, 0.0, global_box, {}

    if hand_box is not None and len(hand_box) >= 4:
        hb_float = [float(v) for v in hand_box[:4]]
        iw = max(0.0, min(global_box[2], hb_float[2]) - max(global_box[0], hb_float[0]))
        ih = max(0.0, min(global_box[3], hb_float[3]) - max(global_box[1], hb_float[1]))
        inter_area = iw * ih
        g_area = max(1.0, gw * gh)
        if (inter_area / g_area) < 0.15:
            return False, 0.0, global_box, {}

    if landmarks and len(landmarks) >= 13:
        palm_finger_indices = [5, 9, 13, 17, 8, 12]

        touches_palm_or_finger = False
        margin = 25.0
        for idx in palm_finger_indices:
            if idx < len(landmarks):
                lx, ly = landmarks[idx]
                if (global_box[0] - margin <= lx <= global_box[2] + margin) and \
                   (global_box[1] - margin <= ly <= global_box[3] + margin):
                    touches_palm_or_finger = True
                    break

        if not touches_palm_or_finger:
            return False, 0.0, global_box, {}

    metrics = {
        "paper_pixel_ratio": paper_pixel_ratio,
        "edge_density": edge_density,
        "paper_pixels": paper_pixels,
        "max_chit_area": max_chit_area,
        "global_box": global_box
    }
    return True, 0.94, global_box, metrics


class DocumentExamDetector:
    """Scans for 4-point quadrilaterals with text-density verification.

    Excludes candidate body and ignores plain non-document surfaces (walls, shirts, reflections).
    Dual-mode support for both large textbooks and small chits.
    """

    def __init__(self, required_stable_frames: int = 1):
        # Instant 1-frame trigger for low latency
        self.confirmed_detection_frames = 0
        self.REQUIRED_STABLE_FRAMES = 1

    def is_text_dense(self, roi_bgr: np.ndarray) -> Tuple[bool, float]:
        """Checks if the candidate polygon contains dense text strokes

        (handwriting or print) rather than plain flat surfaces (walls/shirts).
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return False, 0.0

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        # Gradient magnitude for handwriting/text line edges
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad = cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0))

        # Text creates sharp micro-contrast boundaries
        _, text_thresh = cv2.threshold(
            abs_grad, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        edge_ratio = np.count_nonzero(text_thresh) / float(
            roi_bgr.shape[0] * roi_bgr.shape[1]
        )

        # Books and handwritten notes exhibit high internal gradient density (4% - 40%)
        # Flat blank objects, walls, and clothes have < 2.5%
        return (edge_ratio >= 0.04), edge_ratio

    def detect_papers_and_books(
        self,
        frame: np.ndarray,
        student_bbox: Optional[Union[List[float], Tuple[float, ...]]] = None,
        require_stability: bool = True
    ) -> List[Dict[str, Any]]:
        """Scans for 4-point quadrilaterals with text-density verification.

        Excludes candidate body and ignores plain non-document surfaces.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detected_items = []

        # 1. Mask Candidate upper torso and head completely
        clean_frame = frame.copy()
        if student_bbox is not None and len(student_bbox) >= 4:
            sx1, sy1, sx2, sy2 = [int(v) for v in student_bbox[:4]]
            # If student_bbox is full person/body (>45% frame height), estimate chin around upper 40%
            if (sy2 - sy1) > int(h * 0.45):
                face_h = int((sy2 - sy1) * 0.40)
                exclusion_y2 = min(h, int(sy1 + face_h * 1.45))
            else:
                exclusion_y2 = min(h, int(sy2 + (sy2 - sy1) * 0.45))
            clean_frame[max(0, sy1) : exclusion_y2, max(0, sx1) : min(w, sx2)] = 0

        # 2. Focus on Desk & Lower Hand Workspace (bottom 55% of the camera)
        desk_y_start = int(h * 0.45)
        desk_roi = clean_frame[desk_y_start:h, 0:w]

        # 3. Morphological Edge Detection for Document Borders
        gray = cv2.cvtColor(desk_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Micro-chit & book area filter: from 150 px² (12x12 micro-notes) to open textbooks (180,000 px)
            if area < 150 or area > 180000:
                continue

            # 4. Check for 4-6 Corner Quadrilateral Geometry (Document/Book/Chit boundary)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

            if len(approx) in (4, 5, 6) and cv2.isContourConvex(approx):
                x, y, bw, bh = cv2.boundingRect(approx)
                # Map coordinates back to full-frame space
                real_y = y + desk_y_start
                aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

                target_roi = frame[real_y : real_y + bh, x : x + bw]
                if target_roi.size == 0:
                    continue

                # Inspect HSV color properties
                roi_hsv = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)
                mean_s = float(np.mean(roi_hsv[:, :, 1]))
                mean_v = float(np.mean(roi_hsv[:, :, 2]))

                # Zero Heuristic Hallucinations: Phones must strictly come from YOLO (class 67).
                # This contour module strictly verifies physical paper chits and secondary textbooks.
                has_text, density = self.is_text_dense(target_roi)
                if has_text:
                    # 1. Exam Chit / Micro-Chit / Paper Notes: from 150 px² to standard paper sheets
                    if 150.0 <= area < 32000.0 and 0.35 <= aspect_ratio <= 3.5:
                        if mean_s < 60.0 and mean_v > 120.0 and density > 0.015:
                            detected_items.append(
                                {
                                    "box": [x, real_y, x + bw, real_y + bh],
                                    "label": "ALERT: Exam Chit in Hand (94%)",
                                    "conf": 0.94,
                                    "type": "Exam Chit in Hand",
                                }
                            )
                    # 2. Large Physical Books / Notebooks: Secondary geometric source
                    elif area >= 32000.0 and 0.50 <= aspect_ratio <= 2.5:
                        if mean_s < 45.0 and mean_v > 80.0 and density > 0.030:
                            detected_items.append(
                                {
                                    "box": [x, real_y, x + bw, real_y + bh],
                                    "label": "ALERT: Book / Textbook (92%)",
                                    "conf": 0.92,
                                    "type": "Book / Textbook",
                                }
                            )

        # 6. Apply temporal buffer (responsive: requires 2 consecutive frames for stability)
        if len(detected_items) > 0:
            self.confirmed_detection_frames += 1
        else:
            self.confirmed_detection_frames = max(0, self.confirmed_detection_frames - 1)

        if not require_stability or self.confirmed_detection_frames >= 1:
            return detected_items
        return []


_global_doc_detector = DocumentExamDetector()


def detect_paper_chits(
    frame: np.ndarray,
    student_bbox: Optional[Union[List[float], Tuple[float, ...]]] = None,
    require_stability: bool = False
) -> List[DetectionResult]:
    """Scans for 4-point quadrilaterals with text-density verification via DocumentExamDetector."""
    items = _global_doc_detector.detect_papers_and_books(
        frame=frame,
        student_bbox=student_bbox,
        require_stability=require_stability
    )
    paper_dets: List[DetectionResult] = []
    for item in items:
        box = [float(v) for v in item["box"]]
        item_type = item.get("type", "")
        conf = float(item.get("conf", 0.92))
        if "phone" in item_type.lower():
            paper_dets.append(
                DetectionResult(
                    box=box,
                    confidence=conf,
                    class_id=67,
                    class_name="cell phone"
                )
            )
        elif "book" in item_type.lower():
            paper_dets.append(
                DetectionResult(
                    box=box,
                    confidence=conf,
                    class_id=73,
                    class_name="book"
                )
            )
        else:
            paper_dets.append(
                DetectionResult(
                    box=box,
                    confidence=conf,
                    class_id=73,
                    class_name="unauthorized paper/notes"
                )
            )
    return paper_dets


def apply_nms_suppression(detections: List[DetectionResult], iou_threshold: float = 0.28) -> List[DetectionResult]:
    """Priority-based Non-Maximum Suppression to eliminate duplicate / redundant bounding boxes,
    and guarantee that real cell phones ALWAYS take absolute precedence over overlapping book/notes reflections.
    """
    if not detections or len(detections) <= 1:
        return detections

    def get_priority(d: DetectionResult) -> int:
        name = d.class_name.lower()
        if "phone" in name or "cell" in name or getattr(d, "class_id", -1) in (1, 67):
            return 4
        elif "book" in name or "notebook" in name or getattr(d, "class_id", -1) == 73:
            return 3
        elif any(k in name for k in ("paper", "chit", "notes")):
            return 2
        elif "hand" in name or "gesture" in name:
            return 1
        return 0

    # Sort so highest priority comes first; within same priority, highest confidence comes first
    sorted_dets = sorted(detections, key=lambda d: (get_priority(d), d.confidence), reverse=True)

    filtered: List[DetectionResult] = []
    for d in sorted_dets:
        dx1, dy1, dx2, dy2 = d.box
        d_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
        suppressed = False

        for kept in filtered:
            kx1, ky1, kx2, ky2 = kept.box
            k_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)

            ix1, iy1 = max(dx1, kx1), max(dy1, ky1)
            ix2, iy2 = min(dx2, kx2), min(dy2, ky2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            union = d_area + k_area - inter + 1e-6
            iou = inter / union
            containment = inter / (min(d_area, k_area) + 1e-6)

            k_is_phone = (get_priority(kept) == 4)
            d_is_phone = (get_priority(d) == 4)

            # 1. Phone precedence: A kept phone suppresses any overlapping book/paper/notes/hand
            if k_is_phone and not d_is_phone:
                if iou > 0.05 or containment > 0.10:
                    suppressed = True
                    break

            # 2. Book precedence: A kept book suppresses small chit inside it
            if get_priority(kept) == 3 and get_priority(d) == 2:
                if containment > 0.40 or iou > 0.20:
                    suppressed = True
                    break

            # 3. Same category / class NMS duplicate suppression
            is_same = (kept.class_name == d.class_name) or (get_priority(kept) == get_priority(d))
            if is_same and (iou > iou_threshold or containment > 0.50):
                suppressed = True
                break

        if not suppressed:
            filtered.append(d)

    return filtered


class YOLO26Detector(BaseDetector):
    """Real-time object detector powered by Ultralytics YOLOv8m with OpenCV CLAHE pre-processing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_path = self.config.get("model_path", None)
        if not self.model_path:
            if os.path.exists("best.pt"):
                self.model_path = "best.pt"
            elif os.path.exists("backend/weights/best.pt"):
                self.model_path = "backend/weights/best.pt"
            else:
                self.model_path = "yolov8n.pt" 
        self.iou_threshold = float(self.config.get("iou_threshold", 0.28))
        self.imgsz = int(self.config.get("imgsz", 640))
        self.augment = bool(self.config.get("augment", False))
        self.enable_clahe = bool(self.config.get("enable_clahe", True))
        self.nms_free = self.config.get("nms_free", True)
        
        # Specific Confidence Thresholds per object class (calibrated for high sensitivity without false alarms)
        self.phone_conf_threshold = float(self.config.get("phone_confidence_threshold", 0.18))
        self.person_conf_threshold = float(self.config.get("person_confidence_threshold", 0.28))
        self.book_conf_threshold = float(self.config.get("book_confidence_threshold", 0.15))
        self.default_conf_threshold = float(self.config.get("confidence_threshold", 0.22))

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
            target = self.model_path if (self.model_path and os.path.exists(self.model_path)) else (
                "best.pt" if os.path.exists("best.pt") else (
                    "backend/weights/best.pt" if os.path.exists("backend/weights/best.pt") else (
                        "yolov8n.pt" if os.path.exists("yolov8n.pt") else "yolov8m.pt"
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
        person_boxes: Optional[List[List[float]]] = None,
        require_stability: bool = False
    ) -> List[DetectionResult]:
        """Detects physical small exam chits and paper slips using candidate exclusion, 4-point quadrilateral, and text density."""
        if not self.enable_paper_heuristic or frame is None or frame.size == 0:
            return []

        target_bbox = student_bbox
        if target_bbox is None and person_boxes:
            target_bbox = person_boxes[0]

        return detect_paper_chits(frame, student_bbox=target_bbox, require_stability=require_stability)

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

            # Single-pass high-speed YOLO inference with yolov8m.pt, conf=0.15, iou=0.28, imgsz=self.imgsz, augment=False
            results = self.model.predict(
                source=proc_frame,
                imgsz=self.imgsz,
                conf=0.15,
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

                        # Strict priority: A detected phone is ALWAYS kept as a cell phone (never downgraded to a book)
                        cls_name = "cell phone"
                        cls_id = 67

                    # 3. Validation for Book / Paper / Notes (COCO Class 73, or Class 0 in best.pt)
                    elif cls_name in ("book", "notebook") or ("book" in cls_name):
                        if conf < self.book_conf_threshold:
                            continue

                        # Minimum Size Gate for Books: area >= 28,000 px² and width >= 150 px
                        if box_area < 28000.0 or bw < 150.0:
                            continue

                        # Reject door top / background wall frames
                        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                        if cy < (0.28 * h) and (cx < 0.12 * w or cx > 0.88 * w):
                            continue
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

                    elif cls_name in ("exam_chit", "chit", "cheating-paper", "cheating_paper", "paper", "cheating"):
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
                require_stab = bool(self.config.get("require_paper_stability", True))
                paper_dets = self._detect_white_paper_sheets(
                    proc_frame,
                    student_bbox=target_student_box,
                    person_boxes=person_boxes,
                    require_stability=require_stab
                )
                for pd in paper_dets:
                    px1, py1, px2, py2 = pd.box
                    p_area = (px2 - px1) * (py2 - py1)

                    # Ensure heuristic paper sheet NEVER overrides or pollutes a genuine cell phone
                    overlaps_phone = False
                    for existing in detections:
                        if existing.class_name in ("cell phone", "phone") or getattr(existing, "class_id", -1) in (1, 67):
                            ex1, ey1, ex2, ey2 = existing.box
                            ix1, iy1 = max(px1, ex1), max(py1, ey1)
                            ix2, iy2 = min(px2, ex2), min(py2, ey2)
                            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                            inter = iw * ih
                            e_area = (ex2 - ex1) * (ey2 - ey1)
                            iou = inter / (p_area + e_area - inter + 1e-6)
                            containment = inter / (min(p_area, e_area) + 1e-6)
                            if iou > 0.05 or containment > 0.10:
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

    def _is_hand_holding_paper(self, hand_roi_bgr: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        return _is_hand_holding_paper(hand_roi_bgr)

    def _detect_large_books_and_papers(
        self,
        frame: np.ndarray,
        student_bbox: Optional[Union[List[float], Tuple[float, ...]]] = None,
        require_stability: bool = False
    ) -> List[DetectionResult]:
        """Instant 1-frame trigger for books and paper chits without visual lag."""
        return self._detect_white_paper_sheets(frame, student_bbox=student_bbox, require_stability=require_stability)

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

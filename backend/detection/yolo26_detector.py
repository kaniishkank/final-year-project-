"""
YOLO26 Detector Implementation
Wraps Ultralytics YOLO26 (NMS-free end-to-end inference) for real-time proctoring object detection.
Filters target classes (person, cell phone, book/paper, laptop) with calibrated geometric validation.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from .base import BaseDetector, DetectionResult

logger = logging.getLogger("EviGuard.Detector")


class YOLO26Detector(BaseDetector):
    """Real-time object detector powered by Ultralytics YOLO26."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_path = self.config.get("model_path", "yolo26n.pt")
        self.iou_threshold = float(self.config.get("iou_threshold", 0.45))
        self.imgsz = int(self.config.get("imgsz", 320))
        
        # Specific Confidence Thresholds per object class
        self.phone_conf_threshold = float(self.config.get("phone_confidence_threshold", 0.38))
        self.person_conf_threshold = float(self.config.get("person_confidence_threshold", 0.45))
        self.book_conf_threshold = float(self.config.get("book_confidence_threshold", 0.35))
        self.default_conf_threshold = float(self.config.get("confidence_threshold", 0.30))

        # Geometric Validation Parameters for Cell Phones
        self.phone_min_area = float(self.config.get("phone_min_area", 800.0))
        self.phone_min_w = float(self.config.get("phone_min_w", 20.0))
        self.phone_min_h = float(self.config.get("phone_min_h", 20.0))
        self.phone_min_aspect_ratio = float(self.config.get("phone_min_aspect_ratio", 1.0))
        self.phone_max_aspect_ratio = float(self.config.get("phone_max_aspect_ratio", 3.6))

        # Paper detection enabling (disabled by default to avoid false alerts on white shirts/walls)
        self.enable_paper_heuristic = self.config.get("enable_paper_heuristic", False)

        self.model = None
        self._fallback_mode = False

        self._load_model()

    def _load_model(self):
        """Attempts to load the YOLO26 model, falling back gracefully if unavailable."""
        try:
            from ultralytics import YOLO
            target = self.model_path if os.path.exists(self.model_path) else (
                "yolo26n.pt" if os.path.exists("yolo26n.pt") else (
                    "yolov8n.pt" if os.path.exists("yolov8n.pt") else self.model_path
                )
            )
            logger.info(f"Loading YOLO26 model from {target}...")
            self.model = YOLO(target)
            logger.info(f"YOLO26 model ({target}) loaded successfully via Ultralytics.")
        except Exception as e:
            logger.warning(f"Could not load YOLO26 model ({e}). Initializing simulated/heuristic detector fallback.")
            self._fallback_mode = True

    def _is_valid_phone_geometry(self, box: List[float]) -> bool:
        """Validates cell phone bounding box: accepts portrait, landscape, and diagonal tilt orientations."""
        x1, y1, x2, y2 = box
        bw = abs(x2 - x1)
        bh = abs(y2 - y1)
        area = bw * bh

        # 1. Minimum Area Check
        if area < self.phone_min_area or bw < self.phone_min_w or bh < self.phone_min_h:
            return False

        # 2. Aspect Ratio Check (supports vertical, horizontal, and 45-degree diagonal angles)
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect_ratio < self.phone_min_aspect_ratio or aspect_ratio > self.phone_max_aspect_ratio:
            return False

        return True

    def _detect_white_paper_sheets(self, frame: np.ndarray, person_boxes: Optional[List[List[float]]] = None) -> List[DetectionResult]:
        """Heuristic detector for white paper sheets strictly on desk surface, excluding person's clothing and background walls."""
        paper_dets: List[DetectionResult] = []
        if not self.enable_paper_heuristic or frame is None or frame.size == 0:
            return paper_dets

        try:
            h, w = frame.shape[:2]
            
            # Analyze lower half of screen (desk workspace area only)
            desk_y_start = int(h * 0.45)
            desk_roi = frame[desk_y_start:h, 0:w]
            
            # Convert to HSV to isolate bright, low-saturation white sheet regions
            hsv = cv2.cvtColor(desk_roi, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 195], dtype=np.uint8)
            upper_white = np.array([180, 50, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_white, upper_white)

            # Mask out any detected person torso region
            if person_boxes:
                for pbox in person_boxes:
                    px1, py1, px2, py2 = [int(v) for v in pbox]
                    # Map to desk ROI coordinates
                    ry1 = max(0, py1 - desk_y_start)
                    ry2 = min(h - desk_y_start, py2 - desk_y_start)
                    rx1 = max(0, px1)
                    rx2 = min(w, px2)
                    if ry2 > ry1 and rx2 > rx1:
                        mask[ry1:ry2, rx1:rx2] = 0

            # Morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 3000.0 or area > (h * w * 0.30):
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
                if 1.15 <= aspect_ratio <= 1.85:
                    rect_area = bw * bh
                    fill_ratio = area / (rect_area + 1e-6)
                    if fill_ratio > 0.65:
                        paper_dets.append(
                            DetectionResult(
                                box=[float(x), float(y + desk_y_start), float(x + bw), float(y + desk_y_start + bh)],
                                confidence=0.85,
                                class_id=73,
                                class_name="unauthorized paper/notes"
                            )
                        )
        except Exception as e:
            logger.debug(f"Paper sheet detection exception: {e}")

        return paper_dets

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detects target objects (persons, cell phones, books/paper, laptops) with calibrated validation."""
        if frame is None or frame.size == 0:
            return []

        if self._fallback_mode or self.model is None:
            return self._fallback_detect(frame)

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.default_conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            detections: List[DetectionResult] = []
            person_boxes: List[List[float]] = []

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy().tolist() # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = r.names.get(cls_id, str(cls_id)).lower()

                    # 1. Validation for Cell Phone (Class 67)
                    if cls_name in ("cell phone", "phone") or cls_id == 67:
                        if conf < self.phone_conf_threshold:
                            continue
                        if not self._is_valid_phone_geometry(box):
                            continue
                        cls_name = "cell phone"

                    # 2. Validation for Book / Paper / Notes (COCO Class 73)
                    elif cls_name in ("book", "notebook", "paper") or cls_id == 73:
                        if conf < self.book_conf_threshold:
                            continue
                        cls_name = "unauthorized paper/notes"

                    # 3. Validation for Person (Class 0)
                    elif cls_name == "person" or cls_id == 0:
                        if conf < self.person_conf_threshold:
                            continue
                        cls_name = "person"
                        person_boxes.append(box)

                    # 4. Other Target Objects (e.g. laptop)
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

            # Integrate Paper Sheet Heuristic Detections if explicitly enabled
            if self.enable_paper_heuristic:
                heuristic_papers = self._detect_white_paper_sheets(frame, person_boxes=person_boxes)
                for hp in heuristic_papers:
                    is_duplicate = False
                    for d in detections:
                        if d.class_name in ("unauthorized paper/notes", "book"):
                            x1 = max(hp.box[0], d.box[0])
                            y1 = max(hp.box[1], d.box[1])
                            x2 = min(hp.box[2], d.box[2])
                            y2 = min(hp.box[3], d.box[3])
                            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                            a1 = (hp.box[2] - hp.box[0]) * (hp.box[3] - hp.box[1])
                            if inter / (a1 + 1e-6) > 0.40:
                                is_duplicate = True
                                break
                    if not is_duplicate:
                        detections.append(hp)

            return detections
        except Exception as e:
            logger.error(f"Error during YOLO26 detection: {e}. Using fallback.")
            return self._fallback_detect(frame)

    def _fallback_detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Heuristic/simulated fallback when YOLO26 is not available."""
        h, w = frame.shape[:2]
        detections: List[DetectionResult] = []

        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, fw, fh) in faces:
                px1 = max(0, x - int(fw * 0.5))
                py1 = max(0, y - int(fh * 0.3))
                px2 = min(w, x + fw + int(fw * 0.5))
                py2 = min(h, y + fh * 3)
                detections.append(
                    DetectionResult(
                        box=[float(px1), float(py1), float(px2), float(py2)],
                        confidence=0.88,
                        class_id=0,
                        class_name="person"
                    )
                )
        except Exception:
            detections.append(
                DetectionResult(
                    box=[float(w * 0.2), float(h * 0.1), float(w * 0.8), float(h * 0.9)],
                    confidence=0.85,
                    class_id=0,
                    class_name="person"
                )
            )

        if self.enable_paper_heuristic:
            paper_sheets = self._detect_white_paper_sheets(frame)
            detections.extend(paper_sheets)

        return detections


# Aliases for backward compatibility and alternate namings
YOLOv26Detector = YOLO26Detector
YOLOv8Detector = YOLO26Detector


class MockDetector(BaseDetector):
    """Detector for automated testing and deterministic scenario simulation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.injected_detections: List[DetectionResult] = []

    def set_injected_detections(self, detections: List[DetectionResult]):
        """Injects explicit detections for testing."""
        self.injected_detections = detections

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if self.injected_detections:
            return self.injected_detections

        h, w = (frame.shape[:2]) if frame is not None else (480, 640)
        return [
            DetectionResult(
                box=[float(w * 0.25), float(h * 0.15), float(w * 0.75), float(h * 0.85)],
                confidence=0.92,
                class_id=0,
                class_name="person"
            )
        ]

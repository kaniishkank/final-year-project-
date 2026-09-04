"""
YOLOv8 Detector Implementation
Wraps Ultralytics YOLOv8 for real-time object detection with target class filtering,
confidence thresholds per class, and heuristic geometric filtering for cell phones to eliminate false positives.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from .base import BaseDetector, DetectionResult

logger = logging.getLogger("EviGuard.Detector")


class YOLOv8Detector(BaseDetector):
    """Object detector powered by Ultralytics YOLOv8 with heuristic geometric filtering."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_path = self.config.get("model_path", "yolov8n.pt")
        self.iou_threshold = float(self.config.get("iou_threshold", 0.45))
        self.imgsz = int(self.config.get("imgsz", 320))
        
        # Specific Confidence Thresholds per object class
        self.phone_conf_threshold = float(self.config.get("phone_confidence_threshold", 0.55))
        self.person_conf_threshold = float(self.config.get("person_confidence_threshold", 0.50))
        self.default_conf_threshold = float(self.config.get("confidence_threshold", 0.32))

        # Heuristic Filtering Parameters for Cell Phones
        self.phone_min_area = float(self.config.get("phone_min_area", 2800.0)) # 40 x 70 px minimum
        self.phone_min_w = float(self.config.get("phone_min_w", 35.0))
        self.phone_min_h = float(self.config.get("phone_min_h", 45.0))
        self.phone_min_aspect_ratio = float(self.config.get("phone_min_aspect_ratio", 1.35))
        self.phone_max_aspect_ratio = float(self.config.get("phone_max_aspect_ratio", 2.65))

        self.model = None
        self._fallback_mode = False

        self._load_model()

    def _load_model(self):
        """Attempts to load the YOLOv8 model, falling back gracefully if unavailable."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            logger.info("YOLOv8 model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load YOLOv8 model ({e}). Initializing simulated/heuristic detector fallback.")
            self._fallback_mode = True

    def _is_valid_phone_geometry(self, box: List[float]) -> bool:
        """Heuristic validation for cell phones: rejects tiny or narrow cylindrical objects (pens, inhalers, lip balm)."""
        x1, y1, x2, y2 = box
        bw = abs(x2 - x1)
        bh = abs(y2 - y1)
        area = bw * bh

        # 1. Minimum Area Check (minimum 40x70 = 2800 px)
        if area < self.phone_min_area or bw < self.phone_min_w or bh < self.phone_min_h:
            return False

        # 2. Aspect Ratio Check (smartphones are typically ~1.6:1 to 2.3:1)
        # Accounting for portrait or landscape orientation:
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect_ratio < self.phone_min_aspect_ratio or aspect_ratio > self.phone_max_aspect_ratio:
            return False

        return True

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detects target objects (persons, cell phones, books, laptops) with geometric validation."""
        if frame is None or frame.size == 0:
            return []

        if self._fallback_mode or self.model is None:
            return self._fallback_detect(frame)

        try:
            # Run YOLOv8 prediction with base confidence threshold
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.default_conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            detections: List[DetectionResult] = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy().tolist() # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = r.names.get(cls_id, str(cls_id)).lower()

                    # Target class filtering
                    if self.target_classes and cls_name not in self.target_classes:
                        continue

                    # 1. Specific Validation for Cell Phone (Class 67)
                    if cls_name in ("cell phone", "phone") or cls_id == 67:
                        if conf < self.phone_conf_threshold: # >= 0.55 cutoff
                            continue
                        if not self._is_valid_phone_geometry(box):
                            # Rejected: non-phone geometry (inhaler, pen, lip balm)
                            continue

                    # 2. Specific Validation for Person (Class 0)
                    elif cls_name == "person" or cls_id == 0:
                        if conf < self.person_conf_threshold: # >= 0.50 cutoff
                            continue

                    # 3. Other Allowed Objects (books, laptops, etc.)
                    else:
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

            return detections
        except Exception as e:
            logger.error(f"Error during YOLOv8 detection: {e}. Using fallback.")
            return self._fallback_detect(frame)

    def _fallback_detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Heuristic/simulated fallback when YOLOv8 is not available."""
        h, w = frame.shape[:2]
        detections: List[DetectionResult] = []

        try:
            import cv2
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

        return detections


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

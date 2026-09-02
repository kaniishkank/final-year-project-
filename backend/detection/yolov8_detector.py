"""
YOLOv8 Detector Implementation
Wraps Ultralytics YOLOv8 for real-time object detection with target class filtering.
Includes a lightweight fallback detector for testing and simulation.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from .base import BaseDetector, DetectionResult

logger = logging.getLogger("EviGuard.Detector")


class YOLOv8Detector(BaseDetector):
    """Object detector powered by Ultralytics YOLOv8."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_path = self.config.get("model_path", "yolov8n.pt")
        self.iou_threshold = self.config.get("iou_threshold", 0.45)
        self.imgsz = self.config.get("imgsz", 320)
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

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detects target objects (persons, cell phones, books, laptops) in the frame."""
        if frame is None or frame.size == 0:
            return []

        if self._fallback_mode or self.model is None:
            return self._fallback_detect(frame)

        try:
            # Run YOLOv8 prediction with downscaled inference size for high FPS
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.confidence_threshold,
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
                    cls_name = r.names.get(cls_id, str(cls_id))

                    # Filter to target exam proctoring classes if specified
                    if not self.target_classes or cls_name in self.target_classes:
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
        """Heuristic/simulated fallback when YOLOv8 is not available.
        Uses basic frame dimensions and OpenCV Haar / simple region heuristics if available.
        """
        h, w = frame.shape[:2]
        detections: List[DetectionResult] = []

        try:
            import cv2
            # Try basic Haar Cascade face detection to simulate 'person' detection if available
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, fw, fh) in faces:
                # Expand face box slightly to represent upper body person box
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
            # Fallback default: assume central person bounding box for testing
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

        # Default synthetic candidate
        h, w = (frame.shape[:2]) if frame is not None else (480, 640)
        return [
            DetectionResult(
                box=[float(w * 0.25), float(h * 0.15), float(w * 0.75), float(h * 0.85)],
                confidence=0.92,
                class_id=0,
                class_name="person"
            )
        ]

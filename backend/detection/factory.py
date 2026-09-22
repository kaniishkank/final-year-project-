"""
Detector Factory
Instantiates detectors based on configuration settings (YOLO26 / YOLOv26 / YOLOv8 / Mock).
"""

from typing import Dict, Any, Optional
from .base import BaseDetector
from .yolo26_detector import YOLO26Detector, YOLOv26Detector, YOLOv8Detector, MockDetector


class DetectorFactory:
    """Factory for creating object detector instances."""

    @staticmethod
    def create_detector(detector_type: str = "yolo26", config: Optional[Dict[str, Any]] = None) -> BaseDetector:
        detector_type = (detector_type or "yolo26").lower()
        if detector_type in ("yolo26", "yolov26", "yolo", "yolov8", "ultralytics"):
            return YOLO26Detector(config)
        elif detector_type in ("mock", "test"):
            return MockDetector(config)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. Supported: 'yolo26', 'yolov26', 'yolov8', 'mock'")

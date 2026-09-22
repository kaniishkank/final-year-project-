"""
Detector Factory
Instantiates detectors based on configuration settings.
"""

from typing import Dict, Any, Optional
from .base import BaseDetector
from .yolov8_detector import YOLOv8Detector, MockDetector

# Alias for YOLO26
YOLO26Detector = YOLOv8Detector


class DetectorFactory:
    """Factory for creating object detector instances."""

    @staticmethod
    def create_detector(detector_type: str = "yolo26", config: Optional[Dict[str, Any]] = None) -> BaseDetector:
        detector_type = detector_type.lower()
        if detector_type in ("yolo26", "yolov26", "yolo", "yolov8"):
            return YOLOv8Detector(config)
        elif detector_type in ("mock", "test"):
            return MockDetector(config)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. Supported: 'yolo26', 'yolov8', 'mock'")

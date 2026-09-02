"""
Detector Factory
Instantiates detectors based on configuration settings.
"""

from typing import Dict, Any, Optional
from .base import BaseDetector
from .yolov8_detector import YOLOv8Detector, MockDetector


class DetectorFactory:
    """Factory for creating object detector instances."""

    @staticmethod
    def create_detector(detector_type: str = "yolov8", config: Optional[Dict[str, Any]] = None) -> BaseDetector:
        detector_type = detector_type.lower()
        if detector_type == "yolov8":
            return YOLOv8Detector(config)
        elif detector_type in ("mock", "test"):
            return MockDetector(config)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. Supported: 'yolov8', 'mock'")

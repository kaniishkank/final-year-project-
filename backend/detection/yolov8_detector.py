"""
YOLO26 / YOLOv8 Detector Compatibility Shim
Redirects to YOLO26Detector implementation.
"""

from .yolo26_detector import (
    YOLO26Detector,
    YOLOv26Detector,
    YOLOv8Detector,
    MockDetector,
)

__all__ = ["YOLO26Detector", "YOLOv26Detector", "YOLOv8Detector", "MockDetector"]

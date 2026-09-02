"""
Base Detector Module
Defines standard DetectionResult data structure and BaseDetector abstract class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class DetectionResult:
    """Represents a single detected object in a video frame."""
    box: List[float] # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def center(self) -> List[float]:
        return [(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0]

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": [round(coord, 1) for coord in self.box],
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
        }


class BaseDetector(ABC):
    """Abstract base class for object detectors in EviGuard."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.45)
        self.target_classes = set(self.config.get("target_classes", ["person", "cell phone", "book", "laptop"]))

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Performs object detection on an image/frame.
        
        Args:
            frame: BGR image numpy array.
            
        Returns:
            List of DetectionResult objects.
        """
        pass

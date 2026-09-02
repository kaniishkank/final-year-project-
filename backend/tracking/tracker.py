"""
Multi-Object and Person Tracker Module
Maintains identity persistence across frames using IoU & centroid matching.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ..detection.base import DetectionResult


class TrackedObject:
    """Represents an active track over multiple video frames."""

    def __init__(self, track_id: int, detection: DetectionResult):
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.box = list(detection.box)
        self.confidence = detection.confidence
        self.disappeared_count = 0
        self.total_frames = 1
        self.history = [list(self.box)]

    @property
    def center(self) -> List[float]:
        return [(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0]

    def update(self, detection: DetectionResult):
        """Updates the track state with a new matching detection."""
        self.box = list(detection.box)
        self.confidence = detection.confidence
        self.disappeared_count = 0
        self.total_frames += 1
        self.history.append(list(self.box))
        if len(self.history) > 30:
            self.history.pop(0)

    def mark_missed(self):
        """Increments the missed detection frame count."""
        self.disappeared_count += 1


class PersonTracker:
    """IoU and Centroid-based tracker for exam candidates and objects."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_disappeared = self.config.get("max_disappeared_frames", 30)
        self.iou_threshold = self.config.get("iou_distance_threshold", 0.3)
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedObject] = {}

    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """Calculates Intersection over Union (IoU) between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Matches incoming detections to existing tracks and assigns track IDs.
        
        Args:
            detections: List of DetectionResults from the object detector.
            
        Returns:
            List of DetectionResults with assigned track_id.
        """
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].mark_missed()
                if self.tracks[track_id].disappeared_count > self.max_disappeared:
                    del self.tracks[track_id]
            return []

        if not self.tracks:
            # First frame with detections: initialize all as new tracks
            for det in detections:
                det.track_id = self.next_track_id
                self.tracks[self.next_track_id] = TrackedObject(self.next_track_id, det)
                self.next_track_id += 1
            return detections

        # Build IoU cost matrix between existing active tracks and current detections
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

        for i, t_id in enumerate(track_ids):
            for j, det in enumerate(detections):
                # Penalty for class mismatch
                if self.tracks[t_id].class_name != det.class_name:
                    cost_matrix[i, j] = 0.0
                else:
                    cost_matrix[i, j] = self.compute_iou(self.tracks[t_id].box, det.box)

        # Greedy bipartite matching
        matched_tracks = set()
        matched_detections = set()

        if cost_matrix.size > 0:
            # Flatten indices sorted descending by IoU
            sorted_indices = np.dstack(np.unravel_index(np.argsort(-cost_matrix.ravel()), cost_matrix.shape))[0]
            for row, col in sorted_indices:
                if row in matched_tracks or col in matched_detections:
                    continue
                if cost_matrix[row, col] >= self.iou_threshold:
                    t_id = track_ids[row]
                    detections[col].track_id = t_id
                    self.tracks[t_id].update(detections[col])
                    matched_tracks.add(row)
                    matched_detections.add(col)

        # Handle unmatched existing tracks
        for i, t_id in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[t_id].mark_missed()
                if self.tracks[t_id].disappeared_count > self.max_disappeared:
                    del self.tracks[t_id]

        # Handle unmatched detections -> register new tracks
        for j, det in enumerate(detections):
            if j not in matched_detections:
                det.track_id = self.next_track_id
                self.tracks[self.next_track_id] = TrackedObject(self.next_track_id, det)
                self.next_track_id += 1

        return detections

    def get_person_count(self) -> int:
        """Returns the number of actively tracked persons in the current frame."""
        return sum(1 for t in self.tracks.values() if t.class_name == "person" and t.disappeared_count == 0)

    def reset(self):
        """Clears all track state."""
        self.tracks.clear()
        self.next_track_id = 1

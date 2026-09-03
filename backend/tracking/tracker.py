"""
Multi-Object and Person Tracker Module
Maintains identity persistence across frames using IoU & centroid matching.
Includes robust Non-Maximum Suppression (NMS), confidence filtering, and primary candidate identification.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..detection.base import DetectionResult


class TrackedObject:
    """Represents an active track over multiple video frames."""

    def __init__(self, track_id: int, detection: DetectionResult, is_primary: bool = False):
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.box = list(detection.box)
        self.confidence = detection.confidence
        self.is_primary = is_primary
        self.disappeared_count = 0
        self.total_frames = 1
        self.history = [list(self.box)]

    @property
    def center(self) -> List[float]:
        return [(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0]

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])

    def update(self, detection: DetectionResult, is_primary: bool = False):
        """Updates the track state with a new matching detection."""
        self.box = list(detection.box)
        self.confidence = detection.confidence
        self.is_primary = is_primary
        self.disappeared_count = 0
        self.total_frames += 1
        self.history.append(list(self.box))
        if len(self.history) > 30:
            self.history.pop(0)

    def mark_missed(self):
        """Increments the missed detection frame count."""
        self.disappeared_count += 1


class PersonTracker:
    """Robust Person & Object Tracker with NMS, candidate isolation, and anti-double counting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_disappeared = self.config.get("max_disappeared_frames", 30)
        self.iou_threshold = self.config.get("iou_distance_threshold", 0.35)
        self.person_conf_threshold = self.config.get("person_conf_threshold", 0.55)
        self.person_nms_iou = self.config.get("person_nms_iou", 0.45)
        self.min_person_area = self.config.get("min_person_area", 6000.0) # Filter out tiny noise boxes
        
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

    @staticmethod
    def compute_intersection_over_min(box1: List[float], box2: List[float]) -> float:
        """Calculates overlap relative to the smaller bounding box (containment metric)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        min_area = min(area1, area2)

        return intersection / min_area if min_area > 0 else 0.0

    def _filter_and_nms_persons(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Filters low-confidence persons and applies NMS to prevent double counting a single individual."""
        person_dets = [d for d in detections if d.class_name == "person" and d.confidence >= self.person_conf_threshold]
        other_dets = [d for d in detections if d.class_name != "person"]

        if not person_dets:
            return other_dets

        # Sort persons descending by confidence * area
        person_dets.sort(key=lambda d: d.confidence * d.area, reverse=True)

        kept_persons: List[DetectionResult] = []
        for det in person_dets:
            # Filter tiny background artifacts
            if det.area < self.min_person_area and kept_persons:
                continue

            should_keep = True
            for kept in kept_persons:
                iou = self.compute_iou(det.box, kept.box)
                i_min = self.compute_intersection_over_min(det.box, kept.box)
                
                # If high IoU or one box is inside another, suppress duplicate
                if iou >= self.person_nms_iou or i_min >= 0.70:
                    should_keep = False
                    break

            if should_keep:
                kept_persons.append(det)

        return kept_persons + other_dets

    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Matches filtered detections to existing tracks and assigns accurate IDs."""
        # 1. Apply confidence filtering & NMS
        filtered_detections = self._filter_and_nms_persons(detections)

        if not filtered_detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].mark_missed()
                if self.tracks[track_id].disappeared_count > self.max_disappeared:
                    del self.tracks[track_id]
            return []

        # Identify primary candidate person (largest & most central person box)
        person_candidates = [d for d in filtered_detections if d.class_name == "person"]
        primary_det = None
        if person_candidates:
            primary_det = max(person_candidates, key=lambda d: d.area)

        if not self.tracks:
            # Initialize tracks
            for det in filtered_detections:
                is_prim = (det is primary_det)
                det.track_id = 1 if is_prim else self.next_track_id
                if not is_prim and det.track_id == 1:
                    self.next_track_id += 1
                    det.track_id = self.next_track_id

                self.tracks[det.track_id] = TrackedObject(det.track_id, det, is_primary=is_prim)
                if det.track_id >= self.next_track_id:
                    self.next_track_id = det.track_id + 1
            return filtered_detections

        # Build IoU matching matrix
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(filtered_detections)), dtype=np.float32)

        for i, t_id in enumerate(track_ids):
            for j, det in enumerate(filtered_detections):
                if self.tracks[t_id].class_name != det.class_name:
                    cost_matrix[i, j] = 0.0
                else:
                    cost_matrix[i, j] = self.compute_iou(self.tracks[t_id].box, det.box)

        matched_tracks = set()
        matched_detections = set()

        if cost_matrix.size > 0:
            sorted_indices = np.dstack(np.unravel_index(np.argsort(-cost_matrix.ravel()), cost_matrix.shape))[0]
            for row, col in sorted_indices:
                if row in matched_tracks or col in matched_detections:
                    continue
                if cost_matrix[row, col] >= self.iou_threshold:
                    t_id = track_ids[row]
                    det = filtered_detections[col]
                    det.track_id = t_id
                    is_prim = (det is primary_det) or (t_id == 1 and det.class_name == "person")
                    self.tracks[t_id].update(det, is_primary=is_prim)
                    matched_tracks.add(row)
                    matched_detections.add(col)

        # Handle unmatched tracks
        for i, t_id in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[t_id].mark_missed()
                if self.tracks[t_id].disappeared_count > self.max_disappeared:
                    del self.tracks[t_id]

        # Handle new unmatched detections
        for j, det in enumerate(filtered_detections):
            if j not in matched_detections:
                is_prim = (det is primary_det and 1 not in self.tracks)
                new_id = 1 if is_prim else self.next_track_id
                det.track_id = new_id
                self.tracks[new_id] = TrackedObject(new_id, det, is_primary=is_prim)
                self.next_track_id = max(self.next_track_id + 1, new_id + 1)

        return filtered_detections

    def get_person_count(self) -> int:
        """Returns the number of verified, distinct persons in the frame."""
        return sum(
            1 for t in self.tracks.values()
            if t.class_name == "person" and t.disappeared_count == 0
        )

    def reset(self):
        """Clears all tracking state."""
        self.tracks.clear()
        self.next_track_id = 1

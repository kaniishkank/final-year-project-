"""
Custom Multi-Object and Person Tracker Module (CustomTracker)
Maintains identity persistence across frames using IoU & centroid matching.
Implements intermediate-frame track propagation, temporal voting buffers (length 5),
and strict secondary person centroid distance validation (>150px) to prevent false intruder alerts.
NO ByteTrack, DeepSORT, or external tracking libraries used.
"""

from collections import deque
from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from ..detection.base import DetectionResult


class TrackedObject:
    """Represents an active track over multiple video frames with temporal voting buffers."""

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
        
        # 5.1 Temporal Voting Buffers (Length 5)
        self.phone_buffer: deque = deque(maxlen=5)
        self.notes_buffer: deque = deque(maxlen=5)
        self.gesture_buffer: deque = deque(maxlen=5)
        
        # Velocity for intermediate frame propagation (dx, dy)
        self.velocity = [0.0, 0.0]

    @property
    def center(self) -> List[float]:
        return [(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0]

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])

    def update(self, detection: DetectionResult, is_primary: bool = False):
        """Updates the track state with a new matching detection."""
        old_center = self.center
        self.box = list(detection.box)
        new_center = self.center
        self.velocity = [new_center[0] - old_center[0], new_center[1] - old_center[1]]
        self.confidence = detection.confidence
        self.is_primary = is_primary
        self.disappeared_count = 0
        self.total_frames += 1
        self.history.append(list(self.box))
        if len(self.history) > 30:
            self.history.pop(0)

    def record_evidence_frame(self, has_phone: bool = False, has_notes: bool = False, has_gesture: bool = False):
        """Appends current frame status to the length-5 temporal voting buffers."""
        self.phone_buffer.append(1 if has_phone else 0)
        self.notes_buffer.append(1 if has_notes else 0)
        self.gesture_buffer.append(1 if has_gesture else 0)

    @property
    def has_phone_temporal_escalation(self) -> bool:
        """Section 5.1: If phone detected in any 2 of the last 5 frames -> Escalate to CRITICAL."""
        return sum(self.phone_buffer) >= 2

    @property
    def has_notes_temporal_escalation(self) -> bool:
        """If notes detected in any 2 of the last 5 frames -> Escalate to CRITICAL."""
        return sum(self.notes_buffer) >= 2

    @property
    def has_gesture_temporal_escalation(self) -> bool:
        """Section 5.2: Require gesture to persist for 3 of the last 5 frames -> Escalate to SUSPICIOUS."""
        return sum(self.gesture_buffer) >= 3

    def mark_missed(self):
        """Increments the missed detection frame count."""
        self.disappeared_count += 1
        # Propagate phone/notes/gesture buffers with 0 on missed frame
        self.record_evidence_frame(has_phone=False, has_notes=False, has_gesture=False)

    def propagate(self) -> DetectionResult:
        """Propagates track forward on intermediate frames when detection is skipped."""
        self.total_frames += 1
        # Smoothly advance box by dampening velocity
        dx = self.velocity[0] * 0.5
        dy = self.velocity[1] * 0.5
        self.box = [
            self.box[0] + dx,
            self.box[1] + dy,
            self.box[2] + dx,
            self.box[3] + dy
        ]
        self.history.append(list(self.box))
        if len(self.history) > 30:
            self.history.pop(0)
        return DetectionResult(
            box=list(self.box),
            confidence=self.confidence * 0.98,
            class_id=self.class_id,
            class_name=self.class_name,
            track_id=self.track_id
        )


class CustomTracker:
    """Custom Person & Object Tracker with NMS, candidate isolation, and intermediate propagation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_disappeared = int(self.config.get("max_disappeared_frames", 30))
        self.iou_threshold = float(self.config.get("iou_distance_threshold", 0.35))
        self.person_conf_threshold = float(self.config.get("person_conf_threshold", 0.45))
        self.person_nms_iou = float(self.config.get("person_nms_iou", 0.45))
        self.min_person_area = float(self.config.get("min_person_area", 5000.0))
        self.min_person_distance = float(self.config.get("min_person_distance", 150.0))
        
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
        """Calculates overlap relative to the smaller bounding box."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        min_area = min(area1, area2)

        return intersection / min_area if min_area > 0 else 0.0

    @staticmethod
    def compute_centroid_distance(box1: List[float], box2: List[float]) -> float:
        """Computes Euclidean distance between bounding box centers."""
        c1_x, c1_y = (box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0
        c2_x, c2_y = (box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0
        return math.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

    def _filter_and_nms_persons(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Filters low-confidence persons and applies strict NMS + spatial distance check."""
        person_dets = [d for d in detections if d.class_name == "person" and d.confidence >= self.person_conf_threshold]
        other_dets = [d for d in detections if d.class_name != "person"]

        if not person_dets:
            return other_dets

        person_dets.sort(key=lambda d: d.confidence * d.area, reverse=True)

        kept_persons: List[DetectionResult] = []
        for det in person_dets:
            if det.area < self.min_person_area and kept_persons:
                continue

            should_keep = True
            for kept in kept_persons:
                iou = self.compute_iou(det.box, kept.box)
                i_min = self.compute_intersection_over_min(det.box, kept.box)
                dist = self.compute_centroid_distance(det.box, kept.box)
                
                if iou >= self.person_nms_iou or i_min >= 0.55 or dist < self.min_person_distance:
                    should_keep = False
                    break

            if should_keep:
                kept_persons.append(det)

        return kept_persons + other_dets

    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Matches filtered detections to existing tracks and updates state."""
        filtered_detections = self._filter_and_nms_persons(detections)

        if not filtered_detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].mark_missed()
                if self.tracks[track_id].disappeared_count > self.max_disappeared:
                    del self.tracks[track_id]
            return []

        person_candidates = [d for d in filtered_detections if d.class_name == "person"]
        primary_det = None
        if person_candidates:
            primary_det = max(person_candidates, key=lambda d: d.area)

        if not self.tracks:
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

        for i, t_id in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[t_id].mark_missed()
                if self.tracks[t_id].disappeared_count > self.max_disappeared:
                    del self.tracks[t_id]

        for j, det in enumerate(filtered_detections):
            if j not in matched_detections:
                is_prim = (det is primary_det and 1 not in self.tracks)
                new_id = 1 if is_prim else self.next_track_id
                det.track_id = new_id
                self.tracks[new_id] = TrackedObject(new_id, det, is_primary=is_prim)
                self.next_track_id = max(self.next_track_id + 1, new_id + 1)

        return filtered_detections

    def propagate_tracks(self) -> List[DetectionResult]:
        """Propagates identities and bounding boxes forward on intermediate non-inference frames."""
        results: List[DetectionResult] = []
        for t_id, track in list(self.tracks.items()):
            if track.disappeared_count == 0:
                results.append(track.propagate())
        return results

    def get_person_count(self) -> int:
        """Returns count of active tracked persons."""
        return sum(1 for t in self.tracks.values() if t.class_name == "person" and t.disappeared_count == 0)

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Returns TrackedObject for given track_id."""
        return self.tracks.get(track_id)


# Alias for backward compatibility
PersonTracker = CustomTracker

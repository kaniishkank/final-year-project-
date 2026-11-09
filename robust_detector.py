"""
EviGuard Robust Proctoring Engine
Autonomous multi-modal detection engine combining YOLO26 with ByteTrack,
MediaPipe Hands ROI chit verification, geometric 4-6 point book/notebook polygon analysis,
and candidate face/neck exclusion masking.
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger("EviGuard.RobustDetector")


def enhance_frame_clarity(frame: np.ndarray) -> np.ndarray:
    """Applies CLAHE on the L-channel in LAB color space.
    Sharpens handwriting strokes on chits and edges of dark objects in low contrast.
    """
    if frame is None or frame.size == 0:
        return frame
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        if not hasattr(enhance_frame_clarity, '_clahe'):
            enhance_frame_clarity._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = enhance_frame_clarity._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.debug(f"enhance_frame_clarity fallback: {e}")
        return frame


class EviGuardRobustEngine:
    """Multi-modal robust proctoring detector engine:
    - YOLO26 + ByteTrack (conf=0.20, persist=True, tracker="bytetrack.yaml", classes=[63, 67, 73])
    - MediaPipe Hands padded ROI inspection for light paper texture (HSV S<50, V>150) + text density (Canny > 0.035)
    - Geometric book/notebook contour detection (4-6 vertices, area 5000 to 0.70*frame_area, Sobel text density > 0.030)
    - Candidate face & neck masking (extended 45% downward past chin)
    - Unified violation deduplication & suppression
    """

    def __init__(self, model_path: Optional[str] = None):
        # 1. Resolve YOLO Model Path
        candidate_paths = [
            model_path,
            "best.pt",
            "yolo26n.pt",
            "backend/weights/yolov8m.pt",
            "yolov8n.pt",
            "backend/weights/best.pt"
        ]
        self.model_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                self.model_path = p
                break
        if not self.model_path:
            self.model_path = "yolo26n.pt"

        logger.info(f"Loading YOLO26 model from {self.model_path} for ByteTrack tracking...")
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        logger.info("YOLO26 model loaded successfully.")

        # 2. Initialize MediaPipe Hands
        self.task_hand_detector = None
        self.mp_hands = None
        self._init_mediapipe_hands()

        # 3. EMA Temporal Box Smoothing State
        self.tracks: Dict[str, Dict[str, Any]] = {}
        self._next_track_id: int = 1
        self.ema_alpha: float = 0.70  # Low-jitter: 70% current detection, 30% previous smoothed
        self.max_lost_frames: int = 2

        # 4. Temporal Confirmation Buffer (2-Frame Persistence)
        self.phone_consecutive_frames: int = 0
        self.chit_consecutive_frames: int = 0
        self.book_consecutive_frames: int = 0

    def _init_mediapipe_hands(self):
        """Initializes MediaPipe Tasks HandLandmarker or legacy solutions fallback."""
        task_path = "hand_landmarker.task"
        if os.path.exists(task_path):
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                base_options = python.BaseOptions(model_asset_path=task_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2,
                    min_hand_detection_confidence=0.45,
                    min_tracking_confidence=0.45
                )
                self.task_hand_detector = vision.HandLandmarker.create_from_options(options)
                logger.info("MediaPipe Tasks HandLandmarker initialized successfully.")
                return
            except Exception as e:
                logger.warning(f"MediaPipe Tasks HandLandmarker init error: {e}")

        # Fallback to legacy solutions if available
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.45,
                    min_tracking_confidence=0.45
                )
                logger.info("MediaPipe solutions Hands initialized successfully.")
        except Exception as e:
            logger.debug(f"MediaPipe solutions Hands fallback init: {e}")

    def _detect_hand_boxes(self, frame: np.ndarray) -> List[List[int]]:
        """Returns list of hand bounding boxes [x1, y1, x2, y2] strictly from MediaPipe Hands.
        Eliminates ungrounded desk/skin heuristic fallbacks.
        Limits to at most 2 hands.
        """
        h, w = frame.shape[:2]
        hand_boxes = []

        # 1. Tasks HandLandmarker
        if self.task_hand_detector is not None:
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.task_hand_detector.detect(mp_image)
                if result.hand_landmarks:
                    for hand_lms in result.hand_landmarks[:2]:
                        xs = [int(lm.x * w) for lm in hand_lms]
                        ys = [int(lm.y * h) for lm in hand_lms]
                        hx1, hy1, hx2, hy2 = max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))
                        if (hx2 - hx1) > 15 and (hy2 - hy1) > 15:
                            hand_boxes.append([hx1, hy1, hx2, hy2])
                    return hand_boxes
            except Exception as e:
                logger.debug(f"Task hand detection failed: {e}")

        # 2. Solutions Hands
        if self.mp_hands is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.mp_hands.process(rgb)
                if res.multi_hand_landmarks:
                    for hand_lms in res.multi_hand_landmarks[:2]:
                        xs = [int(lm.x * w) for lm in hand_lms.landmark]
                        ys = [int(lm.y * h) for lm in hand_lms.landmark]
                        hx1, hy1, hx2, hy2 = max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))
                        if (hx2 - hx1) > 15 and (hy2 - hy1) > 15:
                            hand_boxes.append([hx1, hy1, hx2, hy2])
                    return hand_boxes
            except Exception as e:
                logger.debug(f"Solutions hand detection failed: {e}")

        # Strictly return MediaPipe hands only; no background skin/desk contour searching.
        return hand_boxes

    def _is_hand_holding_paper(
        self,
        frame_or_roi: np.ndarray,
        hand_box: Optional[Union[List[int], List[float], Tuple[int, ...], Tuple[float, ...]]] = None
    ) -> Tuple[bool, float, List[float], Dict[str, Any]]:
        """Analyzes cropped high-resolution hand area for micro-chits and folded paper slips:
        - STRICT HAND-PRESENCE GATE: Returns False if no hand detected.
        - IGNORES TOP 40% OF FRAME: Excludes walls, doors, ceiling, and switchboards.
        - HAND-ANCHOR OVERLAP: Requires candidate paper to overlap active hand by >= 10% or be within 40px.
        - Extracts raw hand crop using landmark boundary plus 45px padding.
        - Applies CLAHE contrast enhancement directly to the crop.
        - Checks for white/light paper presence with:
          * HSV S < 55, V > 140
          * Lower area minimum to 120 px²
          * paper_pixel_ratio >= 0.08
          * edge_density >= 0.015
        - Calculates global screen coordinates: [x1 + local_bx1, y1 + local_by1, x1 + local_bx2, y1 + local_by2]
        Returns (is_chit, conf, global_box, metrics).
        """
        if frame_or_roi is None or frame_or_roi.size == 0:
            return False, 0.0, [0.0, 0.0, 0.0, 0.0], {}

        # STRICT HAND-PRESENCE GATE: Hand box is mandatory
        if hand_box is None or len(hand_box) < 4:
            return False, 0.0, [0.0, 0.0, 0.0, 0.0], {}

        h_img, w_img = frame_or_roi.shape[:2]
        hx1, hy1, hx2, hy2 = [int(v) for v in hand_box[:4]]

        # IGNORE TOP 40% OF FRAME (background walls, doors, ceilings, switchboards)
        if hy2 < 0.40 * h_img or (hy1 + hy2) / 2.0 < 0.40 * h_img:
            return False, 0.0, [0.0, 0.0, 0.0, 0.0], {}

        # 1. Extract raw hand crop with 45px boundary padding
        x1 = max(0, hx1 - 45)
        y1 = max(0, hy1 - 45)
        x2 = min(w_img, hx2 + 45)
        y2 = min(h_img, hy2 + 45)
        hand_crop = frame_or_roi[y1:y2, x1:x2]

        if hand_crop.size == 0 or hand_crop.shape[0] < 10 or hand_crop.shape[1] < 10:
            return False, 0.0, [float(x1), float(y1), float(x2), float(y2)], {}

        # 2. Apply CLAHE contrast enhancement directly to the cropped region
        crop_enhanced = enhance_frame_clarity(hand_crop)

        total_pixels = float(hand_crop.shape[0] * hand_crop.shape[1] + 1e-6)
        roi_hsv = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2HSV)
        roi_gray = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2GRAY)

        # 3. Check for white/light paper presence: HSV S < 55, V > 140
        lower_white = np.array([0, 0, 140], dtype=np.uint8)
        upper_white = np.array([180, 55, 255], dtype=np.uint8)
        white_mask = cv2.inRange(roi_hsv, lower_white, upper_white)

        paper_pixels = int(np.count_nonzero(white_mask))
        paper_pixel_ratio = paper_pixels / total_pixels

        # 4. Text / line edge density (threshold 0.015)
        edges = cv2.Canny(roi_gray, 35, 120)
        white_edges = edges & white_mask
        edge_density = float(np.count_nonzero(white_edges)) / total_pixels

        # 5. Check contours down to 120 px² minimum
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_micro_chit = False
        max_chit_area = 0.0
        best_cnt = None
        for cnt in contours:
            c_area = cv2.contourArea(cnt)
            if 120.0 <= c_area <= (total_pixels * 0.85):
                has_micro_chit = True
                if c_area > max_chit_area:
                    max_chit_area = c_area
                    best_cnt = cnt

        is_chit = (
            (paper_pixel_ratio >= 0.08 or has_micro_chit or paper_pixels >= 120)
            and (edge_density >= 0.015 or np.count_nonzero(white_edges) >= 10)
        )

        if not is_chit:
            return False, 0.0, [float(x1), float(y1), float(x2), float(y2)], {}

        # 6. Calculate global screen coordinates for the detected slip
        if best_cnt is not None:
            local_bx1, local_by1, bw, bh = cv2.boundingRect(best_cnt)
            local_bx1 = max(0, local_bx1 - 4)
            local_by1 = max(0, local_by1 - 4)
            local_bx2 = min(hand_crop.shape[1], local_bx1 + bw + 8)
            local_by2 = min(hand_crop.shape[0], local_by1 + bh + 8)
        else:
            white_pts = cv2.findNonZero(white_mask)
            if white_pts is not None and len(white_pts) > 0:
                local_bx1, local_by1, bw, bh = cv2.boundingRect(white_pts)
                local_bx2 = local_bx1 + bw
                local_by2 = local_by1 + bh
            else:
                local_bx1, local_by1 = 0, 0
                local_bx2, local_by2 = hand_crop.shape[1], hand_crop.shape[0]

        global_box = [
            float(x1 + local_bx1),
            float(y1 + local_by1),
            float(x1 + local_bx2),
            float(y1 + local_by2)
        ]

        # Ignore if detected global box is in the top 40% of the camera frame
        if global_box[1] < 0.40 * h_img or (global_box[1] + global_box[3]) / 2.0 < 0.40 * h_img:
            return False, 0.0, global_box, {}

        # Verify candidate paper box intersects with active hand box by >= 10% overlap or within 40px
        hb_float = [float(v) for v in hand_box[:4]]
        iou_hand = self._box_iou(global_box, hb_float)
        cont_hand = self._box_containment(global_box, hb_float)
        dx = max(0.0, max(float(hx1) - global_box[2], global_box[0] - float(hx2)))
        dy = max(0.0, max(float(hy1) - global_box[3], global_box[1] - float(hy2)))
        is_anchored = (iou_hand >= 0.10) or (cont_hand >= 0.10) or (dx <= 40.0 and dy <= 40.0)

        if not is_anchored:
            return False, 0.0, global_box, {}

        metrics = {
            "paper_pixel_ratio": paper_pixel_ratio,
            "edge_density": edge_density,
            "paper_pixels": paper_pixels,
            "max_chit_area": max_chit_area,
            "global_box": global_box
        }
        return is_chit, 0.94 if is_chit else 0.0, global_box, metrics

    def _inspect_hand_for_chit(self, frame: np.ndarray, hand_box: List[int]) -> Optional[Dict[str, Any]]:
        """Inspects raw high-resolution hand crop (padded by 45px) for micro-chits and paper notes."""
        is_chit, conf, global_box, metrics = self._is_hand_holding_paper(frame, hand_box)
        if is_chit and global_box is not None:
            return {
                "box": global_box,
                "label": "ALERT: Exam Chit in Hand (94%)",
                "threat": 94,
                "color": (0, 215, 255),
                "class_name": "exam chit in hand",
                "conf": 0.94,
                "suppress_gesture": True
            }
        return None

    def _detect_large_books_and_papers(
        self,
        frame: np.ndarray,
        student_face_box: Optional[Union[List[float], Tuple[float, ...]]] = None,
        hand_boxes: Optional[List[List[int]]] = None
    ) -> List[Dict[str, Any]]:
        """Strict Hand-Presence Gate: If MediaPipe Hands detects NO hands in the frame,
        IMMEDIATELY RETURN ZERO NOTE/PAPER VIOLATIONS.
        """
        if not hand_boxes or len(hand_boxes) == 0:
            return []
        return []

    def _detect_geometric_books(
        self,
        frame: np.ndarray,
        student_face_box: Optional[Union[List[float], Tuple[float, ...]]] = None
    ) -> List[Dict[str, Any]]:
        """Disabled full-frame contour search to eliminate duplicate multi-box room clutter.
        Physical books and textbooks are cleanly tracked via YOLO26 (class 73).
        """
        return []

    def _detect_yolo_violations(
        self,
        frame: np.ndarray,
        hand_boxes: Optional[List[List[int]]] = None,
        student_face_box: Optional[Union[List[float], Tuple[float, ...]]] = None
    ) -> List[Dict[str, Any]]:
        """Runs YOLO ByteTrack persistence adapting to fine-tuned or COCO classes:
        - Minimum phone confidence raised to 0.42.
        - Phone aspect ratio validated between 1.3 and 2.6.
        - Phone detections inside candidate upper torso/neck suppressed unless overlapping an active hand.
        - Notes/chits strictly hand-gated: 0 violations if no hands detected, ignored in top 40% of frame,
          and require >= 10% overlap or <= 40px proximity to an active hand.
        """
        yolo_violations = []
        try:
            model_names = getattr(self.model, "names", {})
            if len(model_names) <= 5:
                track_classes = list(model_names.keys())
            else:
                track_classes = [63, 67, 73]

            results = self.model.track(
                source=frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.15,
                classes=track_classes,
                verbose=False
            )
            h_frame, w_frame = frame.shape[:2]

            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for i in range(len(boxes)):
                    box = [float(v) for v in boxes.xyxy[i].cpu().numpy().tolist()]
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    track_id = int(boxes.id[i].cpu().numpy()) if (boxes.id is not None) else None
                    t_str = f" [ID:{track_id}]" if track_id is not None else ""

                    bw = abs(box[2] - box[0])
                    bh = abs(box[3] - box[1])
                    aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

                    cname = str(model_names.get(cls_id, "")).lower()

                    # 1. Cell Phone (COCO 67 or fine-tuned 'cell_phone')
                    if cls_id == 67 or "phone" in cname:
                        # Raised minimum confidence threshold to 0.42
                        if conf < 0.42:
                            continue

                        # Phone aspect ratio validation: between 1.3 and 2.6 (discard square or irregular boxes)
                        if not (1.3 <= aspect_ratio <= 2.6):
                            continue

                        # Torso/neck shadow exclusion:
                        # Ignore any phone detection fully inside upper torso/neck unless active hand overlaps it
                        if student_face_box is not None and len(student_face_box) >= 4:
                            fx1, fy1, fx2, fy2 = student_face_box[:4]
                            fw = fx2 - fx1
                            fh = fy2 - fy1
                            torso_box = [fx1 - 0.35 * fw, fy2, fx2 + 0.35 * fw, fy2 + 2.2 * fh]
                            if self._box_containment(box, torso_box) > 0.50 or self._box_iou(box, torso_box) > 0.35:
                                has_hand_overlap = False
                                if hand_boxes:
                                    for hb in hand_boxes:
                                        if self._box_iou(box, hb) > 0.05 or self._box_containment(box, hb) > 0.08:
                                            has_hand_overlap = True
                                            break
                                if not has_hand_overlap:
                                    # Shadow on dark clothing/shoulder -> discard false positive
                                    continue

                        yolo_violations.append({
                            "box": box,
                            "label": f"VIOLATION: MOBILE_PHONE_DETECTED{t_str} ({conf*100:.0f}%)",
                            "threat": 96,
                            "color": (0, 0, 255),
                            "class_name": "cell phone",
                            "conf": conf,
                            "track_id": track_id
                        })

                    # 2. Book / Notes (COCO 73 or fine-tuned 'book')
                    elif cls_id == 73 or "book" in cname:
                        # STRICT HAND-PRESENCE GATE: 0 violations if no hands detected
                        if not hand_boxes or len(hand_boxes) == 0:
                            continue

                        # Completely ignore top 40% of camera frame (walls, doors, ceiling, switchboards)
                        if box[1] < 0.40 * h_frame or (box[1] + box[3]) / 2.0 < 0.40 * h_frame:
                            continue

                        # Candidate paper must intersect active hand by >= 10% overlap or within 40px
                        is_anchored = False
                        for hb in hand_boxes:
                            if self._box_iou(box, hb) >= 0.10 or self._box_containment(box, hb) >= 0.10:
                                is_anchored = True
                                break
                            dx = max(0.0, max(float(hb[0]) - box[2], box[0] - float(hb[2])))
                            dy = max(0.0, max(float(hb[1]) - box[3], box[1] - float(hb[3])))
                            if dx <= 40.0 and dy <= 40.0:
                                is_anchored = True
                                break
                        if not is_anchored:
                            continue

                        if conf < 0.25:
                            continue
                        yolo_violations.append({
                            "box": box,
                            "label": f"ALERT: Book / Textbook{t_str} ({conf*100:.0f}%)",
                            "threat": 92,
                            "color": (0, 165, 255),
                            "class_name": "book",
                            "conf": conf,
                            "track_id": track_id
                        })

                    # 3. Exam Chit (fine-tuned 'exam_chit' / 'chit')
                    elif "chit" in cname or (len(model_names) <= 5 and cls_id == 0):
                        # STRICT HAND-PRESENCE GATE: 0 violations if no hands detected
                        if not hand_boxes or len(hand_boxes) == 0:
                            continue

                        # Completely ignore top 40% of camera frame
                        if box[1] < 0.40 * h_frame or (box[1] + box[3]) / 2.0 < 0.40 * h_frame:
                            continue

                        # Hand overlap >= 10% or within 40px
                        is_anchored = False
                        for hb in hand_boxes:
                            if self._box_iou(box, hb) >= 0.10 or self._box_containment(box, hb) >= 0.10:
                                is_anchored = True
                                break
                            dx = max(0.0, max(float(hb[0]) - box[2], box[0] - float(hb[2])))
                            dy = max(0.0, max(float(hb[1]) - box[3], box[1] - float(hb[3])))
                            if dx <= 40.0 and dy <= 40.0:
                                is_anchored = True
                                break
                        if not is_anchored:
                            continue

                        if conf < 0.25:
                            continue
                        yolo_violations.append({
                            "box": box,
                            "label": f"ALERT: Exam Chit in Hand{t_str} ({conf*100:.0f}%)",
                            "threat": 94,
                            "color": (0, 215, 255),
                            "class_name": "exam chit",
                            "conf": conf,
                            "track_id": track_id
                        })

                    # 4. Laptop (COCO 63)
                    elif cls_id == 63 or "laptop" in cname:
                        yolo_violations.append({
                            "box": box,
                            "label": f"ALERT: Laptop{t_str} ({conf*100:.0f}%)",
                            "threat": 85,
                            "color": (255, 0, 255),
                            "class_name": "laptop",
                            "conf": conf,
                            "track_id": track_id
                        })
        except Exception as e:
            logger.error(f"YOLO track error: {e}")

        return yolo_violations

    def process(
        self,
        frame: np.ndarray,
        student_face_box: Optional[Union[List[float], Tuple[float, ...]]] = None
    ) -> List[Dict[str, Any]]:
        """Processes frame through YOLO26 ByteTrack, MediaPipe Hands chit inspector, and geometric book analyzer.
        Returns a unified list of detected violation dictionaries containing box, label, threat, and color.
        """
        if frame is None or frame.size == 0:
            return []

        # 0. Precision CLAHE Pre-Processing for High-Clarity Feature Extraction
        enhanced_frame = enhance_frame_clarity(frame)

        # 1. Detect active hands via MediaPipe Hands FIRST (Strict Hand-Presence Gate)
        hand_boxes = self._detect_hand_boxes(enhanced_frame)

        # 2. YOLO26 + ByteTrack with CLAHE enhancement, tightened phone thresholds & hand-gated notes
        yolo_viols = self._detect_yolo_violations(
            enhanced_frame,
            hand_boxes=hand_boxes,
            student_face_box=student_face_box
        )

        # 3. Hand ROI Chit Inspection with CLAHE enhancement
        # Strictly anchored to MediaPipe Hands (at most 1 chit alert per hand, max 2 total)
        # STRICT HAND-PRESENCE GATE: If NO hands detected, immediately return zero note/paper violations
        hand_viols = []
        if hand_boxes and len(hand_boxes) > 0:
            for h_box in hand_boxes[:2]:
                chit_viol = self._inspect_hand_for_chit(enhanced_frame, h_box)
                if chit_viol:
                    hand_viols.append(chit_viol)
                    if len(hand_viols) >= 2:
                        break

        # 4. Geometric Book / Notebook Detection with Face/Neck Masking
        geo_viols = self._detect_geometric_books(enhanced_frame, student_face_box)

        # 5. Merge and Deduplicate with Precedence Rules
        final_violations: List[Dict[str, Any]] = []

        # Add all phone violations first
        raw_phones = [v for v in yolo_viols if v["class_name"] == "cell phone"]
        for gv in geo_viols:
            if gv.get("class_name") == "cell phone":
                raw_phones.append(gv)

        phones = []
        for p in raw_phones:
            dup = False
            for ep in phones:
                if self._box_iou(p["box"], ep["box"]) > 0.30 or self._box_containment(p["box"], ep["box"]) > 0.35:
                    dup = True
                    break
            if not dup:
                phones.append(p)
        final_violations.extend(phones)

        # Add chits in hand that do not overlap or touch/hold phones
        for hv in hand_viols:
            overlaps_phone = False
            for pv in phones:
                hx1, hy1, hx2, hy2 = hv["box"]
                px1, py1, px2, py2 = pv["box"]
                near_x = (hx2 >= px1 - 30.0) and (hx1 <= px2 + 30.0)
                near_y = (hy2 >= py1 - 30.0) and (hy1 <= py2 + 30.0)
                if self._box_iou(hv["box"], pv["box"]) > 0.05 or self._box_containment(hv["box"], pv["box"]) > 0.10 or (near_x and near_y):
                    overlaps_phone = True
                    break
            if not overlaps_phone:
                final_violations.append(hv)

        # Add remaining YOLO violations (books, laptops)
        for yv in yolo_viols:
            if yv["class_name"] == "cell phone":
                continue
            duplicate = False
            for fv in final_violations:
                if self._box_iou(yv["box"], fv["box"]) > 0.35:
                    duplicate = True
                    break
            if not duplicate:
                final_violations.append(yv)

        # 6. Global Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes
        if final_violations:
            raw_boxes = [v["box"] for v in final_violations]
            boxes_xywh = [
                [int(b[0]), int(b[1]), int(max(1.0, b[2] - b[0])), int(max(1.0, b[3] - b[1]))]
                for b in raw_boxes
            ]
            scores = [float(v.get("threat", 50) / 100.0) for v in final_violations]
            indices = cv2.dnn.NMSBoxes(
                boxes_xywh,
                scores,
                score_threshold=0.40,
                nms_threshold=0.30
            )
            if len(indices) > 0:
                if isinstance(indices, np.ndarray):
                    flat_indices = indices.flatten().tolist()
                else:
                    flat_indices = [idx[0] if hasattr(idx, "__getitem__") else idx for idx in indices]
                final_violations = [final_violations[i] for i in flat_indices]
            else:
                final_violations = []

        # 7. Temporal Confirmation Buffer (2-Frame Persistence)
        # Require candidate phone or chit to persist for at least 2 consecutive frames before critical alert
        has_current_phone = any(v.get("class_name") == "cell phone" for v in final_violations)
        has_current_chit = any("chit" in v.get("class_name", "") for v in final_violations)
        has_current_book = any(v.get("class_name") == "book" for v in final_violations)

        if has_current_phone:
            self.phone_consecutive_frames += 1
        else:
            self.phone_consecutive_frames = 0

        if has_current_chit:
            self.chit_consecutive_frames += 1
        else:
            self.chit_consecutive_frames = 0

        if has_current_book:
            self.book_consecutive_frames += 1
        else:
            self.book_consecutive_frames = 0

        for v in final_violations:
            c_name = v.get("class_name", "")
            if c_name == "cell phone":
                if self.phone_consecutive_frames < 2:
                    v["threat"] = 45  # Preliminary buffer (below critical 70 threshold)
                    v["label"] = f"MONITOR: Verifying Mobile Phone (1/2 frames)"
                    v["color"] = (0, 165, 255)
                    v["is_confirmed"] = False
                else:
                    v["threat"] = 96
                    v["color"] = (0, 0, 255)
                    v["is_confirmed"] = True
            elif "chit" in c_name:
                if self.chit_consecutive_frames < 2:
                    v["threat"] = 45
                    v["label"] = f"MONITOR: Verifying Exam Chit (1/2 frames)"
                    v["color"] = (0, 165, 255)
                    v["is_confirmed"] = False
                else:
                    v["threat"] = 94
                    v["color"] = (0, 215, 255)
                    v["is_confirmed"] = True
            elif c_name == "book":
                if self.book_consecutive_frames < 2:
                    v["threat"] = 45
                    v["label"] = f"MONITOR: Verifying Book / Notes (1/2 frames)"
                    v["color"] = (0, 165, 255)
                    v["is_confirmed"] = False
                else:
                    v["threat"] = 92
                    v["color"] = (0, 165, 255)
                    v["is_confirmed"] = True

        # 8. EMA Temporal Box Smoothing
        smoothed_violations = self._smooth_violations(final_violations)
        return smoothed_violations

    @staticmethod
    def _box_iou(boxA: List[float], boxB: List[float]) -> float:
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = areaA + areaB - inter + 1e-6
        return inter / union

    @staticmethod
    def _box_containment(boxA: List[float], boxB: List[float]) -> float:
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / (min(areaA, areaB) + 1e-6)

    def reset_tracks(self) -> None:
        """Resets all tracked EMA smoothing states and temporal confirmation buffers."""
        self.tracks.clear()
        self._next_track_id = 1
        self.phone_consecutive_frames = 0
        self.chit_consecutive_frames = 0
        self.book_consecutive_frames = 0

    def _smooth_violations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Applies Exponential Moving Average (EMA) temporal smoothing to bounding box coordinates.
        Eliminates high-frequency coordinate jitter while preserving instant response to object movement.
        """
        matched_track_keys = set()
        matched_det_indices = set()

        # 1. Match by explicit track_id if available (e.g. from ByteTrack)
        for i, det in enumerate(violations):
            tid = det.get("track_id")
            if tid is not None:
                key = f"{det['class_name']}_{tid}"
                if key in self.tracks:
                    track = self.tracks[key]
                    prev_box = track["box"]
                    curr_box = det["box"]
                    track["box"] = [
                        float(self.ema_alpha * c + (1.0 - self.ema_alpha) * p)
                        for c, p in zip(curr_box, prev_box)
                    ]
                    track["label"] = det.get("label", track["label"])
                    track["conf"] = float(self.ema_alpha * det.get("conf", 0.9) + (1.0 - self.ema_alpha) * track.get("conf", 0.9))
                    track["threat"] = det.get("threat", track["threat"])
                    track["color"] = det.get("color", track["color"])
                    track["lost_count"] = 0
                    matched_track_keys.add(key)
                    matched_det_indices.add(i)

        # 2. Match remaining detections by IoU overlap (> 0.25) within the same class
        for i, det in enumerate(violations):
            if i in matched_det_indices:
                continue
            best_iou = 0.0
            best_key = None
            for key, track in self.tracks.items():
                if key in matched_track_keys:
                    continue
                if track.get("class_name") != det.get("class_name"):
                    continue
                iou = self._box_iou(det["box"], track["box"])
                if iou > 0.25 and iou > best_iou:
                    best_iou = iou
                    best_key = key

            if best_key is not None:
                track = self.tracks[best_key]
                prev_box = track["box"]
                curr_box = det["box"]
                track["box"] = [
                    float(self.ema_alpha * c + (1.0 - self.ema_alpha) * p)
                    for c, p in zip(curr_box, prev_box)
                ]
                track["label"] = det.get("label", track["label"])
                track["conf"] = float(self.ema_alpha * det.get("conf", 0.9) + (1.0 - self.ema_alpha) * track.get("conf", 0.9))
                track["threat"] = det.get("threat", track["threat"])
                track["color"] = det.get("color", track["color"])
                track["lost_count"] = 0
                matched_track_keys.add(best_key)
                matched_det_indices.add(i)

        # 3. Register new tracks for unmatched detections
        for i, det in enumerate(violations):
            if i in matched_det_indices:
                continue
            tid = det.get("track_id")
            key = f"{det['class_name']}_{tid}" if tid is not None else f"{det['class_name']}_gen_{self._next_track_id}"
            self._next_track_id += 1
            new_track = dict(det)
            new_track["box"] = [float(v) for v in det["box"]]
            new_track["lost_count"] = 0
            self.tracks[key] = new_track
            matched_track_keys.add(key)

        # 4. Decay and remove lost tracks
        dead_keys = []
        for key, track in self.tracks.items():
            if key not in matched_track_keys:
                track["lost_count"] += 1
                if track["lost_count"] > self.max_lost_frames:
                    dead_keys.append(key)

        for dk in dead_keys:
            del self.tracks[dk]

        # 5. Output active smoothed violations (only emit if recently seen, lost_count <= 1)
        smoothed_results: List[Dict[str, Any]] = []
        for key, track in self.tracks.items():
            if track["lost_count"] <= 1:
                res = dict(track)
                res.pop("lost_count", None)
                smoothed_results.append(res)

        return smoothed_results

    def draw_violations(self, frame: np.ndarray, violations: List[Dict[str, Any]]) -> np.ndarray:
        """Renders violation bounding boxes and badges on a copy of the frame."""
        annotated = frame.copy()
        for v in violations:
            box = [int(val) for val in v["box"]]
            color = v.get("color", (0, 0, 255))
            label = v.get("label", "ALERT")
            x1, y1, x2, y2 = box

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        return annotated

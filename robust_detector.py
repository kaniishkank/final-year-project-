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
        """Returns list of hand bounding boxes [x1, y1, x2, y2] from MediaPipe Hands or skin heuristic."""
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
                    for hand_lms in result.hand_landmarks:
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
                    for hand_lms in res.multi_hand_landmarks:
                        xs = [int(lm.x * w) for lm in hand_lms.landmark]
                        ys = [int(lm.y * h) for lm in hand_lms.landmark]
                        hx1, hy1, hx2, hy2 = max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))
                        if (hx2 - hx1) > 15 and (hy2 - hy1) > 15:
                            hand_boxes.append([hx1, hy1, hx2, hy2])
                    return hand_boxes
            except Exception as e:
                logger.debug(f"Solutions hand detection failed: {e}")

        # 3. Desk skin contour heuristic fallback (bottom 60% of frame)
        try:
            desk_y = int(h * 0.40)
            roi = frame[desk_y:h, :]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 200, 255]))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 1800 < area < 35000:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    hand_boxes.append([x, y + desk_y, x + bw, y + desk_y + bh])
        except Exception:
            pass

        return hand_boxes

    def _inspect_hand_for_chit(self, frame: np.ndarray, hand_box: List[int]) -> Optional[Dict[str, Any]]:
        """Inspects hand bounding box padded by 40px for light-colored paper texture (HSV S<50, V>150)
        and text/line edge density (Canny > 0.035). If found, returns violation dict.
        """
        h, w = frame.shape[:2]
        hx1, hy1, hx2, hy2 = hand_box
        px1 = max(0, hx1 - 40)
        py1 = max(0, hy1 - 40)
        px2 = min(w, hx2 + 40)
        py2 = min(h, hy2 + 40)

        roi = frame[py1:py2, px1:px2]
        if roi.size == 0 or roi.shape[0] < 15 or roi.shape[1] < 15:
            return None

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        total_pixels = float(roi.shape[0] * roi.shape[1] + 1e-6)

        # 1. Light-colored paper texture: S < 50, V > 150
        paper_mask = (roi_hsv[:, :, 1] < 50) & (roi_hsv[:, :, 2] > 150)
        paper_pixels = int(np.count_nonzero(paper_mask))
        paper_ratio = paper_pixels / total_pixels

        # 2. Text/line edge density: Canny > 0.035
        edges = cv2.Canny(roi_gray, 40, 140)
        edge_density = float(np.count_nonzero(edges)) / total_pixels

        # Sobel gradient density for handwriting / printed text micro-contrast
        gx = cv2.Sobel(roi_gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(roi_gray, cv2.CV_16S, 0, 1, ksize=3)
        ag = cv2.convertScaleAbs(cv2.addWeighted(gx, 0.5, gy, 0.5, 0))
        _, text_thresh = cv2.threshold(ag, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sobel_density = float(np.count_nonzero(text_thresh)) / total_pixels

        if (paper_pixels >= 450 or paper_ratio >= 0.08) and (edge_density > 0.035 or sobel_density > 0.035):
            return {
                "box": [float(px1), float(py1), float(px2), float(py2)],
                "label": "ALERT: EXAM CHIT IN HAND (93%)",
                "threat": 93,
                "color": (0, 140, 255),
                "class_name": "unauthorized paper/notes",
                "conf": 0.93
            }
        return None

    def _detect_geometric_books(
        self,
        frame: np.ndarray,
        student_face_box: Optional[Union[List[float], Tuple[float, ...]]] = None
    ) -> List[Dict[str, Any]]:
        """Detects books/notebooks using 4-6 point contour approximation and internal text density.
        Enforces Candidate Face & Neck Masking (45% downward past chin).
        """
        h, w = frame.shape[:2]
        clean_frame = frame.copy()

        # Enforce Candidate Face & Neck Masking: zero out face and 45% past chin
        if student_face_box is not None and len(student_face_box) >= 4:
            fx1, fy1, fx2, fy2 = [int(v) for v in student_face_box[:4]]
            if (fy2 - fy1) > int(h * 0.45):
                face_h = int((fy2 - fy1) * 0.40)
                exclusion_y2 = min(h, int(fy1 + face_h * 1.45))
            else:
                exclusion_y2 = min(h, int(fy2 + (fy2 - fy1) * 0.45))
            clean_frame[max(0, fy1):exclusion_y2, max(0, fx1):min(w, fx2)] = 0

        # Scan mid-to-lower frame (eye-level and desk: bottom 70% of frame)
        y_start = int(h * 0.30)
        roi_search = clean_frame[y_start:h, 0:w]
        gray = cv2.cvtColor(roi_search, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        book_violations = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Sensitive bounds: small chits (350 px) to big open textbooks
            if not (500.0 < area < (0.75 * w * h)):
                continue
            bx, by, bw_c, bh_c = cv2.boundingRect(cnt)
            if min(bw_c, bh_c) < 20:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

            # 4-6 point quadrilateral or open book / chit structure
            if 4 <= len(approx) <= 6:
                x, y, bw, bh = cv2.boundingRect(approx)
                real_y = y + y_start
                aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)

                target_roi = frame[real_y:real_y+bh, x:x+bw]
                if target_roi.size == 0:
                    continue

                roi_hsv = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)
                mean_s = float(np.mean(roi_hsv[:, :, 1]))
                mean_v = float(np.mean(roi_hsv[:, :, 2]))

                # A. Smartphone Check: illuminated saturated screen with handheld phone scale
                is_phone_candidate = (
                    (3000.0 <= area <= 85000.0) and (min(bw, bh) >= 45.0 and max(bw, bh) >= 70.0) and
                    (1.20 <= aspect_ratio <= 3.8) and
                    (mean_s >= 40.0 or mean_v < 60.0)
                )
                if is_phone_candidate:
                    book_violations.append({
                        "box": [float(x), float(real_y), float(x + bw), float(real_y + bh)],
                        "label": "VIOLATION: MOBILE_PHONE_DETECTED (95%)",
                        "threat": 95,
                        "color": (0, 0, 255),
                        "class_name": "cell phone",
                        "conf": 0.95
                    })
                    continue

                rg = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(rg, cv2.CV_16S, 1, 0, ksize=3)
                gy = cv2.Sobel(rg, cv2.CV_16S, 0, 1, ksize=3)
                ag = cv2.convertScaleAbs(cv2.addWeighted(gx, 0.5, gy, 0.5, 0))
                _, tt = cv2.threshold(ag, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                sobel_density = float(np.count_nonzero(tt)) / float(target_roi.shape[0] * target_roi.shape[1] + 1e-6)

                if sobel_density > 0.025:
                    if mean_s < 55.0 and mean_v > 80.0:
                        item_type = "Big Book / Notebook" if area > 16000.0 and aspect_ratio < 1.70 else "Exam Chit / Notes"
                        threat = 90 if "Book" in item_type else 92
                        color = (0, 165, 255) if "Book" in item_type else (0, 140, 255)
                        book_violations.append({
                            "box": [float(x), float(real_y), float(x + bw), float(real_y + bh)],
                            "label": f"ALERT: {item_type} (94%)",
                            "threat": threat,
                            "color": color,
                            "class_name": "book" if "Book" in item_type else "unauthorized paper/notes",
                            "conf": 0.94
                        })

        return book_violations

    def _detect_yolo_violations(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Runs YOLO26 with native ByteTrack persistence for classes [63, 67, 73]."""
        yolo_violations = []
        try:
            results = self.model.track(
                source=frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.20,
                classes=[63, 67, 73],
                verbose=False
            )
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

                    # 1. Cell Phone (COCO 67)
                    if cls_id == 67:
                        # Strict priority: Always keep as cell phone!
                        yolo_violations.append({
                            "box": box,
                            "label": f"VIOLATION: MOBILE_PHONE_DETECTED{t_str} ({conf*100:.0f}%)",
                            "threat": 95,
                            "color": (0, 0, 255),
                            "class_name": "cell phone",
                            "conf": conf,
                            "track_id": track_id
                        })

                    # 2. Book (COCO 73)
                    elif cls_id == 73:
                        # Phone Disambiguation: handheld dimensions or phone-compatible aspect ratio
                        is_likely_phone = (
                            (bw * bh < 85000.0) and (1.20 <= aspect_ratio <= 3.8) and
                            (bw <= 340.0 or bh <= 460.0)
                        )
                        if is_likely_phone:
                            yolo_violations.append({
                                "box": box,
                                "label": f"VIOLATION: MOBILE_PHONE_DETECTED{t_str} ({conf*100:.0f}%)",
                                "threat": 95,
                                "color": (0, 0, 255),
                                "class_name": "cell phone",
                                "conf": conf,
                                "track_id": track_id
                            })
                        else:
                            yolo_violations.append({
                                "box": box,
                                "label": f"ALERT: Book / Textbook{t_str} ({conf*100:.0f}%)",
                                "threat": 90,
                                "color": (0, 165, 255),
                                "class_name": "book",
                                "conf": conf,
                                "track_id": track_id
                            })

                    # 3. Laptop (COCO 63)
                    elif cls_id == 63:
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
            logger.error(f"YOLO26 track error: {e}")

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

        # 1. YOLO26 + ByteTrack
        yolo_viols = self._detect_yolo_violations(frame)

        # 2. Hand ROI Chit Inspection
        hand_viols = []
        hand_boxes = self._detect_hand_boxes(frame)
        for h_box in hand_boxes:
            chit_viol = self._inspect_hand_for_chit(frame, h_box)
            if chit_viol:
                hand_viols.append(chit_viol)

        # 3. Geometric Book / Notebook Detection with Face/Neck Masking
        geo_viols = self._detect_geometric_books(frame, student_face_box)

        # 4. Merge and Deduplicate with Precedence Rules
        # - Priority 1: Cell Phone (suppresses overlapping books/chits)
        # - Priority 2: Chit In Hand (suppresses generic overlapping book bounding boxes)
        # - Priority 3: Books/Notebooks
        final_violations: List[Dict[str, Any]] = []

        # Add all phone violations first (deduplicated across YOLO and geometric detectors)
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

        # Add geometric book violations that do not duplicate existing YOLO or hand violations
        for gv in geo_viols:
            duplicate = False
            for fv in final_violations:
                if self._box_iou(gv["box"], fv["box"]) > 0.25 or self._box_containment(gv["box"], fv["box"]) > 0.40:
                    duplicate = True
                    break
            if not duplicate:
                final_violations.append(gv)

        return final_violations

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

"""
EviGuard Proctored Inference Engine - Standalone Real-Time Execution Loop
Runs YOLO26 (NMS-free), MediaPipe Hands & FaceMesh in an OpenCV webcam stream.
Renders real-time HUD bounding boxes, 3D head pose axes, telemetry overlays,
and records rolling 6-second MP4 evidence clips upon critical malpractice triggers.
"""

import collections
import os
import sys
import threading
import time
from typing import Deque, Optional
import cv2
import numpy as np

from config import CONFIG
from detector import ProctoredInferenceEngine, DetectionBundle
from head_pose import HeadPoseEstimator
from telemetry import TelemetryEngine, TelemetryState


class VideoCircularBuffer:
    """Circular rolling frame memory ring buffer with background MP4 video dumper."""

    def __init__(self, capacity: int = 180, output_dir: str = "data/evidence_clips"):
        self.capacity = capacity
        self.output_dir = output_dir
        self.buffer: Deque[np.ndarray] = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()
        os.makedirs(output_dir, exist_ok=True)

    def append(self, frame: np.ndarray):
        with self.lock:
            self.buffer.append(frame.copy())

    def dump_clip_async(self, incident_label: str, fps: float = 30.0):
        """Spawns a background thread to encode and save the current buffer to MP4."""
        with self.lock:
            frames_to_write = list(self.buffer)

        if not frames_to_write:
            return

        thread = threading.Thread(
            target=self._write_video,
            args=(frames_to_write, incident_label, fps),
            daemon=True
        )
        thread.start()

    def _write_video(self, frames: list, label: str, fps: float):
        try:
            h, w = frames[0].shape[:2]
            clean_label = "".join(c if c.isalnum() else "_" for c in label)[:20]
            ts = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"evidence_{ts}_{clean_label}.mp4")

            # MP4V codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filepath, fourcc, max(10.0, fps), (w, h))

            for f in frames:
                writer.write(f)
            writer.release()
            print(f"[EVIDENCE DUMP] Saved 6s forensic clip: {filepath}")
        except Exception as e:
            print(f"[EVIDENCE DUMP ERROR] {e}")


def draw_hud(
    frame: np.ndarray,
    bundle: DetectionBundle,
    telemetry: TelemetryState
) -> np.ndarray:
    """Renders comprehensive HUD overlays: bounding boxes, 3D gaze axes, top telemetry bar."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # 1. Top Telemetry Bar Background
    bar_height = 68
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
    cv2.line(annotated, (0, bar_height), (w, bar_height), (79, 70, 229), 2)

    # 2. Integrity Quotient Display
    iq = telemetry.integrity_quotient
    if iq >= 85.0:
        iq_color = (74, 222, 128)  # Green
        status_text = "AUTHENTIC"
    elif iq >= 60.0:
        iq_color = (251, 191, 36)  # Amber
        status_text = "ELEVATED RISK"
    else:
        iq_color = (239, 68, 68)   # Red
        status_text = "MALPRACTICE"

    # Title & Engine
    cv2.putText(annotated, "EviGuard AI [YOLO26 + MediaPipe]", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 245, 249), 2, cv2.LINE_AA)

    # Telemetry Line: IQ & Status
    cv2.putText(annotated, f"INTEGRITY: {iq:.1f}%", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, iq_color, 2, cv2.LINE_AA)
    cv2.putText(annotated, f"[{status_text}]", (210, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, iq_color, 2, cv2.LINE_AA)

    # Head Pose & FPS on Top Right
    pose_str = f"Y:{bundle.head_pose.yaw:+.0f} P:{bundle.head_pose.pitch:+.0f} R:{bundle.head_pose.roll:+.0f}"
    cv2.putText(annotated, pose_str, (w - 230, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"FPS: {telemetry.fps:.0f}", (w - 85, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 2, cv2.LINE_AA)

    # 3. Active Violations & Probabilities Sub-Banner
    if telemetry.active_violations:
        viol_banner_y = bar_height + 28
        cv2.rectangle(annotated, (0, bar_height), (w, viol_banner_y), (185, 28, 28), -1)
        prob_strs = [f"{k} ({telemetry.active_cheat_probabilities.get(k, 0.9)*100:.0f}%)" for k in telemetry.active_violations[:2]]
        viol_text = "ALERT: " + " | ".join(prob_strs)
        cv2.putText(annotated, viol_text, (12, viol_banner_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Render Detected Objects
    for obj in bundle.objects:
        x1, y1, x2, y2 = [int(v) for v in obj.box]
        if obj.is_malpractice:
            color = (0, 0, 230)  # Red for malpractice (Phone, Paper, Smartwatch)
            label = f"{obj.class_name.upper()} ({obj.confidence*100:.0f}%)"
        elif obj.class_name == "person":
            color = (34, 197, 94)  # Green for Candidate
            label = f"Student (Active) {obj.confidence*100:.0f}%"
        else:
            color = (255, 200, 0)
            label = f"{obj.class_name} {obj.confidence*100:.0f}%"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label Badge
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Render Hand Signalling
    for hand in bundle.hand_signalling:
        if hand.hand_box:
            hx1, hy1, hx2, hy2 = [int(v) for v in hand.hand_box]
            h_color = (0, 165, 255) if hand.is_signalling_malpractice else (200, 200, 200)
            cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), h_color, 2)
            if hand.gesture_label:
                cv2.putText(annotated, hand.gesture_label, (hx1, hy1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 2, cv2.LINE_AA)

    # 6. Render 3D Head Pose Axes on Nose Tip
    if bundle.head_pose.success and bundle.head_pose.rvec is not None:
        HeadPoseEstimator.draw_pose_axes(
            annotated,
            bundle.head_pose.rvec,
            bundle.head_pose.tvec,
            bundle.head_pose.nose_2d,
            length=60.0
        )

    # 7. Bottom Navigation Instructions
    cv2.putText(annotated, "Press [Q] or [ESC] to Exit Proctoring", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

    return annotated


def main():
    """Main execution loop for real-time proctoring."""
    print("=" * 65)
    print("  EviGuard Proctored Inference Engine (YOLO26 + MediaPipe)")
    print("=" * 65)

    # 1. Initialize Engines
    engine = ProctoredInferenceEngine(CONFIG)
    telemetry_tracker = TelemetryEngine(CONFIG)
    ring_buffer = VideoCircularBuffer(capacity=180, output_dir=CONFIG.get("video", {}).get("evidence_dir", "data/evidence_clips"))

    # 2. Initialize Camera
    cam_src = CONFIG.get("video", {}).get("source", 0)
    width = int(CONFIG.get("video", {}).get("width", 640))
    height = int(CONFIG.get("video", {}).get("height", 480))

    cap = cv2.VideoCapture(cam_src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video capture source: {cam_src}")
        sys.exit(1)

    print("[SYSTEM] Starting real-time proctoring inference stream...")
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop video file or continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx += 1
            now = time.time()

            # Append to Circular Evidence Ring Buffer
            ring_buffer.append(frame)

            # Process Frame through Multi-Modal Engine
            bundle = engine.process_frame(frame, frame_idx, now)

            # Update Telemetry & Probabilistic Integrity Quotient
            telemetry = telemetry_tracker.update(bundle)

            # Check Critical Video Dump Trigger
            if telemetry.trigger_mp4_dump:
                ring_buffer.dump_clip_async(telemetry.mp4_dump_reason, fps=telemetry.fps)

            # Render Visual HUD
            display_frame = draw_hud(frame, bundle, telemetry)

            # Display Window
            cv2.imshow("EviGuard AI Proctoring Studio", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[SYSTEM] Session terminated by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 65)
        print("  SESSION INTEGRITY SUMMARY REPORT")
        print("=" * 65)
        print(f"Total Frames Processed: {frame_idx}")
        print(f"Final Integrity Quotient: {telemetry_tracker.total_penalties:.1f}% deduction -> {max(0.0, 100.0 - telemetry_tracker.total_penalties):.1f}%")
        print(f"Total Incidents Logged:  {len(telemetry_tracker.incidents)}")
        for inc in telemetry_tracker.incidents:
            print(f"  - [{inc.timestamp_str}] {inc.anomaly_type} ({inc.severity}) | Cheat Prob: {inc.cheat_probability*100:.1f}% | -{inc.integrity_penalty}%")
        print("=" * 65)


if __name__ == "__main__":
    main()

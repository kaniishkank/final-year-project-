"""
Seed Script for EviGuard
Populates data/eviguard.db with realistic exam sessions, telemetry metrics, snapshots, and incident records.
"""

from datetime import datetime, timedelta
import os
import cv2
import numpy as np
from backend.db.models import DatabaseManager

def seed():
    print("[EviGuard] Seeding database and mock evidence...")
    
    # Ensure evidence clips dir exists
    evidence_dir = "data/evidence_clips"
    os.makedirs(evidence_dir, exist_ok=True)
    
    db = DatabaseManager.get_instance("sqlite:///data/eviguard.db")
    
    # 1. Create Sample Session
    session_id = "EXAM_2026_CS401_01"
    candidate_id = "STD-2024-889"
    candidate_name = "Alex Johnson"
    exam_title = "CS401: Advanced AI & Machine Learning"
    
    session = db.create_session(session_id, candidate_id, candidate_name, exam_title)
    print(f"  [+] Created session: {session_id} for {candidate_name}")

    # Generate sample snapshot 1: Cell Phone in Hand
    snap1_path = os.path.join(evidence_dir, f"{session_id}_inc1_snapshot.jpg")
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img1[:] = (40, 35, 45)
    cv2.rectangle(img1, (100, 340), (540, 480), (75, 70, 85), -1) # Desk
    # Candidate looking down
    cv2.circle(img1, (320, 210), 70, (210, 180, 160), -1)
    cv2.circle(img1, (300, 230), 8, (50, 40, 30), -1)
    cv2.circle(img1, (340, 230), 8, (50, 40, 30), -1)
    cv2.ellipse(img1, (320, 390), (140, 120), 0, 0, 180, (110, 80, 75), -1)
    # Phone
    cv2.rectangle(img1, (380, 340), (450, 430), (20, 20, 20), -1)
    cv2.rectangle(img1, (385, 345), (445, 425), (220, 240, 255), -1)
    # Bounding box & warning label
    cv2.rectangle(img1, (375, 335), (455, 435), (0, 0, 255), 2)
    cv2.putText(img1, "ALERT: cell phone (94%)", (350, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(snap1_path, img1)

    # Generate sample MP4 video clip for incident 1
    clip1_path = os.path.join(evidence_dir, f"{session_id}_inc1_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip1_path, fourcc, 15.0, (640, 480))
    for i in range(45): # 3-second clip
        f = img1.copy()
        cv2.putText(f, f"REC [EVIGUARD VAULT] 00:0{i//15}:0{i%15}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        writer.write(f)
    writer.release()
    print("  [+] Generated mock video clip & snapshot for Incident #1")

    # Generate sample snapshot 2: Multiple Persons in Room
    snap2_path = os.path.join(evidence_dir, f"{session_id}_inc2_snapshot.jpg")
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2[:] = (40, 35, 45)
    # Primary candidate
    cv2.circle(img2, (380, 220), 65, (220, 190, 170), -1)
    cv2.ellipse(img2, (380, 390), (130, 110), 0, 0, 180, (120, 90, 80), -1)
    cv2.rectangle(img2, (260, 140), (500, 460), (255, 180, 0), 2)
    cv2.putText(img2, "Person #1 (Candidate)", (260, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
    # Secondary intruder
    cv2.circle(img2, (140, 190), 55, (180, 150, 140), -1)
    cv2.ellipse(img2, (140, 350), (100, 90), 0, 0, 180, (80, 100, 130), -1)
    cv2.rectangle(img2, (50, 120), (230, 420), (0, 0, 255), 2)
    cv2.putText(img2, "ALERT: Person #2 (Intruder)", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(snap2_path, img2)

    # 2. Insert Incident 1: Unauthorized Mobile Device
    db.log_incident(
        session_id=session_id,
        frame_index=142,
        violation_type="PHONE_DETECTED",
        severity="CRITICAL",
        risk_score=88.5,
        confidence=0.94,
        reason_summary="Unauthorized Mobile Device (Cell Phone) in Workspace",
        reason_narrative="At 14:12:08, candidate Alex Johnson exhibited sustained downward head pitch (-28.4°) for 4.6 seconds coinciding with an unauthorized cell phone detected in hand with 94.2% model confidence.",
        evidence_clip_path=clip1_path,
        evidence_snapshot_path=snap1_path,
        details={
            "factor_attribution": {"cell_phone": 65.0, "head_pose_deviation": 35.0},
            "recommended_action": "Issue immediate verbal warning; instruct candidate to relocate phone outside test perimeter.",
            "pose_gaze": {"yaw": -4.2, "pitch": -28.4, "roll": 2.1, "gaze_direction": "LOOKING_DOWN (DESK/PHONE)"}
        }
    )

    # 3. Insert Incident 2: Multiple Persons
    db.log_incident(
        session_id=session_id,
        frame_index=380,
        violation_type="MULTIPLE_PERSONS",
        severity="HIGH",
        risk_score=76.0,
        confidence=0.91,
        reason_summary="Multiple Persons Detected in Testing Environment",
        reason_narrative="At 14:24:45, a secondary individual entered the camera field of view in the background behind candidate Alex Johnson for over 3.2 seconds.",
        evidence_clip_path=None,
        evidence_snapshot_path=snap2_path,
        details={
            "factor_attribution": {"multiple_persons": 80.0, "unusual_movement": 20.0},
            "recommended_action": "Request immediate 360-degree webcam scan to ensure solitary exam conditions.",
            "pose_gaze": {"yaw": 12.0, "pitch": 2.5, "roll": 0.0, "gaze_direction": "CENTER (FOCUSED)"}
        }
    )

    # 4. Insert telemetry metrics to populate timeline chart
    for f in range(1, 100):
        # Baseline low risk with a peak at f=35 (incident 1) and f=75 (incident 2)
        if 30 <= f <= 45:
            score = 65.0 + (f - 30) * 1.5
            violations = ["PHONE_DETECTED", "GAZE_DOWN"]
            phone = True
            count = 1
        elif 70 <= f <= 85:
            score = 70.0 + (f - 70) * 0.8
            violations = ["MULTIPLE_PERSONS"]
            phone = False
            count = 2
        else:
            score = max(0.0, 5.0 + (f % 5) * 2.0)
            violations = []
            phone = False
            count = 1

        db.log_metric(
            session_id=session_id,
            frame_index=f * 10,
            risk_score=score,
            person_count=count,
            phone_detected=phone,
            yaw=float((f % 7) - 3),
            pitch=float(-25.0 if 30 <= f <= 45 else (f % 5) - 2),
            active_violations=violations
        )

    print("  [+] Seeded 100 telemetry metric logs for timeline plotting")
    print("[EviGuard] Seeding completed successfully!")

if __name__ == "__main__":
    seed()

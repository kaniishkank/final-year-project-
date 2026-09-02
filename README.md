# 🛡️ EviGuard: AI-Powered Exam Proctoring & Evidence Analysis System

EviGuard is a multi-modal computer vision and Explainable AI (XAI) automated exam proctoring framework designed to detect academic dishonesty in real-time, generate tamper-evident video/snapshot logs, and present audit justifications for proctors and educators.

---

## 🌟 Key Features

1. **Multi-Object Detection (`backend/detection/`)**:
   - Integrated with **YOLOv8** to identify unauthorized electronic devices (cell phones, secondary laptops) and unauthorized study materials (books/notes).
   - Extensible factory pattern with fallback simulation for automated testing.

2. **Temporal Entity Tracking (`backend/tracking/`)**:
   - IoU and Centroid-based tracker ensuring persistent tracking of the candidate and detecting unauthorized secondary persons entering the test environment.

3. **3D Head Pose & Gaze Analysis (`backend/pose/`)**:
   - 3D facial landmark estimation via Perspective-n-Point (`solvePnP`) computing **Euler angles (Yaw, Pitch, Roll)**.
   - Detects sustained downward gaze (desk/phone), lateral gaze shifts (looking away), and candidate absence.

4. **Dynamic Risk Engine (`backend/scoring/`)**:
   - Weighted multi-modal violation scoring with **Exponential Moving Average (EMA)** smoothing and temporal decay.
   - Categorizes threats into `LOW` (0-30), `MEDIUM` (31-70), and `HIGH` (71-100) risk bands.

5. **Explainable AI (XAI) & Audit Generation (`backend/explainability/`)**:
   - Translates raw model signals into factor attribution percentages and human-readable natural language audit reports.

6. **Automated Evidence Vault (`data/evidence_clips/`)**:
   - Circular rolling frame buffer (Ring Buffer) automatically recording pre-roll and post-roll video clips (`.mp4`) and high-resolution snapshots (`.jpg`) when critical violations occur.

7. **Interactive Web Dashboard (`frontend/dashboard.py`)**:
   - Built with **Streamlit** and **Plotly**.
   - **Live Proctoring**: Live annotated video feed with HUD overlays, semi-circular risk gauge, active violation badges, and live simulation triggers.
   - **Incident Vault**: Interactive video playback of evidence clips with XAI factor breakdown and proctor verdict submission (Confirm / False Positive / Dismiss).
   - **Analytics & Export**: Summary charts and downloadable formal markdown integrity reports.
   - **Sensitivity Tuning**: Dynamic configuration sliders for weights, gaze thresholds, and camera parameters.

---

## 📂 Project Architecture

```
eviguard/
├── backend/
│   ├── detection/
│   │   ├── base.py              # Base detector interface & DetectionResult dataclass
│   │   ├── yolov8_detector.py   # Ultralytics YOLOv8 detector with fallback
│   │   └── factory.py           # DetectorFactory
│   ├── tracking/
│   │   └── tracker.py           # Multi-entity & PersonTracker
│   ├── pose/
│   │   └── pose_gaze.py         # 3D Head Pose (solvePnP) & Gaze Estimator
│   ├── scoring/
│   │   └── risk_engine.py       # Dynamic Risk Engine & EMA smoothing
│   ├── explainability/
│   │   └── reason_generator.py  # XAI Reason Generator & Factor Attribution
│   ├── db/
│   │   └── models.py            # SQLAlchemy database models & manager
│   └── pipeline.py              # End-to-end Orchestration Pipeline & Ring Buffer
├── frontend/
│   └── dashboard.py             # Streamlit interactive proctoring UI
├── data/
│   ├── evidence_clips/          # Recorded MP4 incident clips & snapshots
│   └── eviguard.db              # SQLite session and incident database
├── tests/
│   └── test_pipeline.py         # Pytest unit & integration test suite
├── config.yaml                  # System configuration parameters
├── requirements.txt             # Python dependencies
└── run.py                       # Launcher script
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m pytest tests/test_pipeline.py -v
```

### 3. Launch EviGuard Dashboard
```bash
python run.py
```
*Or via Streamlit directly:*
```bash
streamlit run frontend/dashboard.py
```

Open your browser at `http://localhost:8501` to access the EviGuard proctoring console.

---

## 🧪 Testing with the Interactive Simulator
In the **Live Proctoring** tab, use the built-in demo buttons:
- **📱 Phone Detected**: Injects an unauthorized mobile device detection into the frame.
- **👥 Multiple People**: Injects an unauthorized secondary person in the proctoring frame.
- **👀 Looking Away**: Simulates candidate looking away from the primary display.
- **🚪 Candidate Absent**: Simulates candidate stepping away from the camera.

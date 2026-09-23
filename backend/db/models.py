"""
EviGuard Database Models & Persistence Layer
Provides SQLAlchemy ORM models for exam sessions, detected incidents, and real-time risk logs.
"""

from datetime import datetime
import json
import os
import queue
import threading
import time
from typing import Dict, Any, List, Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    desc
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, scoped_session

Base = declarative_base()


class ExamSession(Base):
    """Represents an individual proctoring exam session for a candidate."""
    __tablename__ = "exam_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    candidate_id = Column(String(64), nullable=False)
    candidate_name = Column(String(128), nullable=False)
    exam_title = Column(String(128), nullable=False)
    start_time = Column(DateTime, default=datetime.now, nullable=False)
    end_time = Column(DateTime, nullable=True)
    status = Column(String(32), default="ACTIVE") # ACTIVE, COMPLETED, TERMINATED
    total_incidents = Column(Integer, default=0)
    avg_risk_score = Column(Float, default=0.0)
    peak_risk_score = Column(Float, default=0.0)
    integrity_index = Column(Float, default=100.0) # 100 - weighted violation index

    # Relationships
    incidents = relationship("Incident", back_populates="session", cascade="all, delete-orphan")
    risk_metrics = relationship("RiskMetricLog", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "exam_title": self.exam_title,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "total_incidents": self.total_incidents,
            "avg_risk_score": round(self.avg_risk_score, 2),
            "peak_risk_score": round(self.peak_risk_score, 2),
            "integrity_index": round(self.integrity_index, 2),
        }


class Incident(Base):
    """Represents a flagged security or behavioral violation during a session."""
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("exam_sessions.session_id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    frame_index = Column(Integer, default=0)
    violation_type = Column(String(64), nullable=False) # e.g. PHONE_DETECTED, MULTIPLE_PERSONS, GAZE_AWAY, FACE_ABSENT
    severity = Column(String(32), default="HIGH") # LOW, MEDIUM, HIGH, CRITICAL
    risk_score = Column(Float, nullable=False)
    confidence = Column(Float, default=1.0)
    reason_summary = Column(String(256), nullable=False)
    reason_narrative = Column(Text, nullable=False)
    evidence_clip_path = Column(String(256), nullable=True)
    evidence_snapshot_path = Column(String(256), nullable=True)
    details_json = Column(Text, default="{}") # JSON string containing bounding boxes, angles, etc.
    proctor_verdict = Column(String(32), default="PENDING") # PENDING, CONFIRMED, FALSE_POSITIVE, DISMISSED
    proctor_notes = Column(Text, nullable=True)

    session = relationship("ExamSession", back_populates="incidents")

    def to_dict(self) -> Dict[str, Any]:
        try:
            details = json.loads(self.details_json) if self.details_json else {}
        except Exception:
            details = {}

        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else None,
            "frame_index": self.frame_index,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "risk_score": round(self.risk_score, 1),
            "confidence": round(self.confidence, 2),
            "reason_summary": self.reason_summary,
            "reason_narrative": self.reason_narrative,
            "evidence_clip_path": self.evidence_clip_path,
            "evidence_snapshot_path": self.evidence_snapshot_path,
            "details": details,
            "proctor_verdict": self.proctor_verdict,
            "proctor_notes": self.proctor_notes,
        }


class RiskMetricLog(Base):
    """High-frequency continuous telemetry log for plotting timeline metrics."""
    __tablename__ = "risk_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("exam_sessions.session_id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    frame_index = Column(Integer, default=0)
    risk_score = Column(Float, default=0.0)
    person_count = Column(Integer, default=1)
    phone_detected = Column(Boolean, default=False)
    yaw = Column(Float, default=0.0)
    pitch = Column(Float, default=0.0)
    active_violations = Column(String(256), default="")

    session = relationship("ExamSession", back_populates="risk_metrics")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.strftime("%H:%M:%S") if self.timestamp else "",
            "frame_index": self.frame_index,
            "risk_score": round(self.risk_score, 1),
            "person_count": self.person_count,
            "phone_detected": self.phone_detected,
            "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1),
            "active_violations": self.active_violations.split(",") if self.active_violations else [],
        }


# Database Connection and Helper Functions

def get_engine_and_session_factory(db_url: str = "sqlite:///data/eviguard.db"):
    """Creates database engine and scoped sessionmaker."""
    # Ensure data directory exists if sqlite
    if "sqlite:///" in db_url:
        db_path = db_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})
    Base.metadata.create_all(engine)
    session_factory = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
    return engine, session_factory


class DatabaseManager:
    """Singleton/Manager for database interactions throughout the pipeline and dashboard."""
    _instance = None

    def __init__(self, db_url: str = "sqlite:///data/eviguard.db"):
        self.db_url = db_url
        self.engine, self.SessionFactory = get_engine_and_session_factory(db_url)
        self._metric_queue: queue.Queue = queue.Queue(maxsize=2000)
        self._metric_worker_running = True
        self._metric_worker_thread = threading.Thread(target=self._process_metric_queue, daemon=True)
        self._metric_worker_thread.start()

    @classmethod
    def get_instance(cls, db_url: str = "sqlite:///data/eviguard.db") -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = cls(db_url)
        return cls._instance

    def _process_metric_queue(self):
        """Background thread worker to batch commit metric telemetry to SQLite without blocking vision FPS."""
        while self._metric_worker_running:
            try:
                batch = []
                try:
                    first_item = self._metric_queue.get(timeout=0.25)
                    batch.append(first_item)
                    while len(batch) < 50:
                        try:
                            batch.append(self._metric_queue.get_nowait())
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue

                if batch:
                    db = self.SessionFactory()
                    try:
                        for item in batch:
                            metric = RiskMetricLog(
                                session_id=item["session_id"],
                                frame_index=item["frame_index"],
                                risk_score=item["risk_score"],
                                person_count=item["person_count"],
                                phone_detected=item["phone_detected"],
                                yaw=item["yaw"],
                                pitch=item["pitch"],
                                active_violations=item["active_violations"]
                            )
                            db.add(metric)
                        db.commit()
                    except Exception:
                        db.rollback()
                    finally:
                        db.close()
            except Exception:
                time.sleep(0.05)

    def flush_metrics(self):
        """Immediately drains and writes any pending in-memory telemetry metrics to SQLite."""
        batch = []
        while not self._metric_queue.empty():
            try:
                batch.append(self._metric_queue.get_nowait())
            except queue.Empty:
                break

        if batch:
            db = self.SessionFactory()
            try:
                for item in batch:
                    metric = RiskMetricLog(
                        session_id=item["session_id"],
                        frame_index=item["frame_index"],
                        risk_score=item["risk_score"],
                        person_count=item["person_count"],
                        phone_detected=item["phone_detected"],
                        yaw=item["yaw"],
                        pitch=item["pitch"],
                        active_violations=item["active_violations"]
                    )
                    db.add(metric)
                db.commit()
            except Exception:
                db.rollback()
            finally:
                db.close()

    def create_session(self, session_id: str, candidate_id: str, candidate_name: str, exam_title: str) -> ExamSession:
        db = self.SessionFactory()
        try:
            existing = db.query(ExamSession).filter_by(session_id=session_id).first()
            if existing:
                return existing

            session_obj = ExamSession(
                session_id=session_id,
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                exam_title=exam_title,
                start_time=datetime.now(),
                status="ACTIVE"
            )
            db.add(session_obj)
            db.commit()
            db.refresh(session_obj)
            return session_obj
        finally:
            db.close()

    def end_session(self, session_id: str) -> Optional[ExamSession]:
        self.flush_metrics()
        db = self.SessionFactory()
        try:
            session_obj = db.query(ExamSession).filter_by(session_id=session_id).first()
            if not session_obj:
                return None

            session_obj.end_time = datetime.now()
            session_obj.status = "COMPLETED"

            # Compute stats
            incidents = db.query(Incident).filter_by(session_id=session_id).all()
            metrics = db.query(RiskMetricLog).filter_by(session_id=session_id).all()

            session_obj.total_incidents = len(incidents)
            if metrics:
                scores = [m.risk_score for m in metrics]
                session_obj.avg_risk_score = float(sum(scores) / len(scores))
                session_obj.peak_risk_score = float(max(scores))
            
            # Integrity Index calculation (100 - penalties)
            penalty = sum([30 if inc.severity == "CRITICAL" else (20 if inc.severity == "HIGH" else 10) for inc in incidents])
            session_obj.integrity_index = max(0.0, float(100.0 - penalty))

            db.commit()
            db.refresh(session_obj)
            return session_obj
        finally:
            db.close()

    def log_incident(
        self,
        session_id: str,
        frame_index: int,
        violation_type: str,
        severity: str,
        risk_score: float,
        confidence: float,
        reason_summary: str,
        reason_narrative: str,
        evidence_clip_path: Optional[str] = None,
        evidence_snapshot_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Incident:
        db = self.SessionFactory()
        try:
            incident = Incident(
                session_id=session_id,
                frame_index=frame_index,
                violation_type=violation_type,
                severity=severity,
                risk_score=risk_score,
                confidence=confidence,
                reason_summary=reason_summary,
                reason_narrative=reason_narrative,
                evidence_clip_path=evidence_clip_path,
                evidence_snapshot_path=evidence_snapshot_path,
                details_json=json.dumps(details or {}),
                proctor_verdict="PENDING"
            )
            db.add(incident)

            # Update session incident count
            session_obj = db.query(ExamSession).filter_by(session_id=session_id).first()
            if session_obj:
                session_obj.total_incidents = (session_obj.total_incidents or 0) + 1
                if risk_score > (session_obj.peak_risk_score or 0.0):
                    session_obj.peak_risk_score = risk_score

            db.commit()
            db.refresh(incident)
            return incident
        finally:
            db.close()

    def log_metric(
        self,
        session_id: str,
        frame_index: int,
        risk_score: float,
        person_count: int,
        phone_detected: bool,
        yaw: float,
        pitch: float,
        active_violations: List[str]
    ):
        """Asynchronously queues metric log for zero-latency frame throughput."""
        item = {
            "session_id": session_id,
            "frame_index": frame_index,
            "risk_score": float(risk_score),
            "person_count": int(person_count),
            "phone_detected": bool(phone_detected),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "active_violations": ",".join(active_violations) if isinstance(active_violations, list) else str(active_violations)
        }
        try:
            self._metric_queue.put_nowait(item)
        except queue.Full:
            pass

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        self.flush_metrics()
        db = self.SessionFactory()
        try:
            sessions = db.query(ExamSession).order_by(desc(ExamSession.start_time)).all()
            return [s.to_dict() for s in sessions]
        finally:
            db.close()

    def get_session_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        db = self.SessionFactory()
        try:
            s = db.query(ExamSession).filter_by(session_id=session_id).first()
            return s.to_dict() if s else None
        finally:
            db.close()

    def get_session_incidents(self, session_id: str) -> List[Dict[str, Any]]:
        db = self.SessionFactory()
        try:
            incidents = db.query(Incident).filter_by(session_id=session_id).order_by(Incident.timestamp).all()
            return [i.to_dict() for i in incidents]
        finally:
            db.close()

    def get_session_metrics(self, session_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        self.flush_metrics()
        db = self.SessionFactory()
        try:
            metrics = db.query(RiskMetricLog).filter_by(session_id=session_id).order_by(RiskMetricLog.frame_index).limit(limit).all()
            return [m.to_dict() for m in metrics]
        finally:
            db.close()

    def update_incident_verdict(self, incident_id: int, verdict: str, notes: Optional[str] = None) -> bool:
        db = self.SessionFactory()
        try:
            incident = db.query(Incident).filter_by(id=incident_id).first()
            if not incident:
                return False
            incident.proctor_verdict = verdict
            if notes is not None:
                incident.proctor_notes = notes
            db.commit()
            return True
        finally:
            db.close()

    def purge_all_data(self) -> bool:
        """Completely purges all historical sessions, incidents, risk metrics, and clips."""
        db = self.SessionFactory()
        try:
            db.query(RiskMetricLog).delete()
            db.query(Incident).delete()
            db.query(ExamSession).delete()
            db.commit()

            # Clean evidence clips directory if it exists
            clips_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "evidence_clips")
            if os.path.exists(clips_dir):
                for f in os.listdir(clips_dir):
                    f_path = os.path.join(clips_dir, f)
                    try:
                        if os.path.isfile(f_path):
                            os.remove(f_path)
                    except Exception:
                        pass
            return True
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()


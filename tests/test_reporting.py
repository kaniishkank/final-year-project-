"""
Unit & Integration Tests for Candidate Malpractice & Integrity Report Generator
"""

import os
import pytest
from backend.db.models import DatabaseManager
from backend.reporting.report_generator import (
    ReportGenerator,
    generate_candidate_pdf_report,
    generate_candidate_csv_report
)


@pytest.fixture
def mock_db(tmp_path):
    """Creates a temporary sqlite db for isolated testing."""
    db_file = tmp_path / "test_reporting.db"
    db = DatabaseManager(f"sqlite:///{db_file}")
    
    # Create test session
    session_id = "TEST_EXAM_999"
    db.create_session(
        session_id=session_id,
        candidate_id="STD-888",
        candidate_name="Jane Doe",
        exam_title="Computer Vision & Ethics"
    )

    # Log several diverse incidents
    db.log_incident(
        session_id=session_id,
        frame_index=15,
        violation_type="PHONE_DETECTED",
        severity="CRITICAL",
        risk_score=92.5,
        confidence=0.945,
        reason_summary="Unauthorized mobile phone in hand",
        reason_narrative="Candidate held a smartphone in active view for 4.2s.",
        evidence_clip_path="evidence/clip_001.mp4",
        evidence_snapshot_path="evidence/snap_001.jpg"
    )

    db.log_incident(
        session_id=session_id,
        frame_index=65,
        violation_type="UNAUTHORIZED_MATERIAL",
        severity="HIGH",
        risk_score=78.0,
        confidence=0.882,
        reason_summary="Physical notes / paper detected on desk",
        reason_narrative="Candidate referenced paper sheet on the desk.",
        evidence_clip_path="evidence/clip_002.mp4"
    )

    db.log_incident(
        session_id=session_id,
        frame_index=140,
        violation_type="SUSPICIOUS_SIGNALLING",
        severity="HIGH",
        risk_score=85.0,
        confidence=0.910,
        reason_summary="Suspicious 2-finger gesture towards camera",
        reason_narrative="Candidate displayed 2 fingers indicative of signalling answer options.",
        evidence_clip_path="evidence/clip_003.mp4"
    )

    return db, session_id


def test_pdf_report_generation(mock_db):
    """Verifies that PDF report is generated with valid structure and PDF header."""
    db, session_id = mock_db
    pdf_bytes = generate_candidate_pdf_report(session_id, db)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


def test_pdf_report_empty_session(tmp_path):
    """Verifies that PDF report generates gracefully when session has 0 incidents."""
    db_file = tmp_path / "test_empty.db"
    db = DatabaseManager(f"sqlite:///{db_file}")
    session_id = "CLEAN_EXAM_101"
    db.create_session(
        session_id=session_id,
        candidate_id="STD-CLEAN",
        candidate_name="Clean Student",
        exam_title="Pure Integrity Test"
    )

    pdf_bytes = generate_candidate_pdf_report(session_id, db)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-")


def test_csv_report_generation(mock_db):
    """Verifies that CSV audit export contains correct headers and candidate data."""
    db, session_id = mock_db
    csv_str = generate_candidate_csv_report(session_id, db)

    assert isinstance(csv_str, str)
    assert "Incident ID,Timestamp,Frame Index" in csv_str
    assert "PHONE_DETECTED" in csv_str
    assert "UNAUTHORIZED_MATERIAL" in csv_str
    assert "SUSPICIOUS_SIGNALLING" in csv_str
    assert "Jane Doe" in csv_str
    assert "STD-888" in csv_str


def test_report_generator_oop_interface(mock_db):
    """Verifies ReportGenerator class interface."""
    db, session_id = mock_db
    rg = ReportGenerator(db)
    
    data = rg.get_session_data(session_id)
    assert data["session"]["candidate_name"] == "Jane Doe"
    assert len(data["incidents"]) == 3

    pdf = rg.generate_pdf(session_id)
    csv_text = rg.generate_csv(session_id)
    assert len(pdf) > 0
    assert len(csv_text) > 0

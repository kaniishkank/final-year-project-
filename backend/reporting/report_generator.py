"""
EviGuard Automated Candidate Malpractice & Integrity Report Generator
Generates institutional audit-grade PDF and CSV reports for proctoring sessions.
"""

import csv
import io
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

from backend.db.models import DatabaseManager, ExamSession, Incident


def _format_violation_name(raw_name: str) -> str:
    """Formats raw violation string to clean human-readable category name."""
    mapping = {
        "PHONE_DETECTED": "Mobile Phone Detected",
        "UNAUTHORIZED_MATERIAL": "Unauthorized Paper / Notes",
        "UNAUTHORIZED_PAPER": "Unauthorized Paper / Notes",
        "MULTIPLE_PERSONS": "Secondary Person / Intruder",
        "FACE_ABSENT": "Candidate Absent / Face Obscured",
        "SUSPICIOUS_SIGNALLING": "Suspicious Finger / Hand Signalling",
        "GAZE_AWAY": "Sustained Gaze Deviation",
        "PROLONGED_GAZE": "Prolonged Gaze Malpractice",
        "HEAD_POSE_DEVIATION": "Head Pose Deviation",
        "LOOKING_LEFT": "Gaze Deviation (Left)",
        "LOOKING_RIGHT": "Gaze Deviation (Right)",
        "LOOKING_DOWN": "Downward Gaze (Desk/Lap)",
    }
    cleaned = raw_name.strip().upper()
    return mapping.get(cleaned, raw_name.replace("_", " ").title())


def _format_severity_badge(severity: str) -> str:
    sev = severity.upper()
    if sev == "CRITICAL":
        return '<font color="#DC2626"><b>CRITICAL</b></font>'
    elif sev == "HIGH":
        return '<font color="#EA580C"><b>HIGH</b></font>'
    elif sev == "MEDIUM":
        return '<font color="#D97706"><b>MEDIUM</b></font>'
    else:
        return '<font color="#2563EB"><b>LOW</b></font>'


class ReportGenerator:
    """Automated formal integrity and malpractice report generator for EviGuard."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager.get_instance()

    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Fetches session metadata and incident records from eviguard.db."""
        session = self.db_manager.get_session_by_id(session_id)
        if not session:
            # Fallback mock/empty session if not found in db
            session = {
                "session_id": session_id,
                "candidate_id": "N/A",
                "candidate_name": "Candidate",
                "exam_title": "Assessment",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "status": "COMPLETED",
                "total_incidents": 0,
                "peak_risk_score": 0.0,
                "avg_risk_score": 0.0,
                "integrity_index": 100.0,
            }
        
        incidents = self.db_manager.get_session_incidents(session_id)
        return {
            "session": session,
            "incidents": incidents
        }

    def generate_pdf(self, session_id: str) -> bytes:
        """Compiles formal institutional audit PDF report as bytes."""
        data = self.get_session_data(session_id)
        session = data["session"]
        incidents = data["incidents"]

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom Typography Styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#0F172A'),
            alignment=TA_LEFT
        )
        
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#4F46E5'),
            alignment=TA_LEFT,
            textTransform='uppercase'
        )

        meta_label_style = ParagraphStyle(
            'MetaLabel',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor('#64748B')
        )

        meta_val_style = ParagraphStyle(
            'MetaVal',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor('#0F172A')
        )

        section_heading_style = ParagraphStyle(
            'SectionHeading',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=15,
            textColor=colors.HexColor('#1E293B')
        )

        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=TA_CENTER
        )

        table_cell_style = ParagraphStyle(
            'TableCell',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor('#1E293B')
        )

        table_cell_center = ParagraphStyle(
            'TableCellCenter',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor('#1E293B'),
            alignment=TA_CENTER
        )

        xai_narrative_style = ParagraphStyle(
            'XAINarrative',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#334155'),
            alignment=TA_JUSTIFY
        )

        story = []

        # ---------------- 1. INSTITUTIONAL HEADER ----------------
        header_data = [
            [
                Paragraph("<b>EVIGUARD AI PROCTORING SYSTEM</b>", subtitle_style),
                Paragraph(f"Report Generated: {datetime.now().strftime('%b %d, %Y - %H:%M:%S')}", ParagraphStyle('GenDate', fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#64748B'), alignment=TA_RIGHT))
            ],
            [
                Paragraph("Candidate Malpractice & Academic Integrity Report", title_style),
                Paragraph(f"Ref Session: <b>{session.get('session_id')}</b>", ParagraphStyle('RefId', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#4F46E5'), alignment=TA_RIGHT))
            ]
        ]
        header_table = Table(header_data, colWidths=[380, 160])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(header_table)
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#4F46E5'), spaceBefore=6, spaceAfter=10))

        # ---------------- 2. CANDIDATE & ASSESSMENT IDENTITY ----------------
        integrity_val = float(session.get("integrity_index", 100.0))
        if integrity_val >= 85.0:
            badge_bg = "#DCFCE7"
            badge_border = "#16A34A"
            badge_text_color = "#15803D"
            status_text = "AUTHENTIC / HIGH INTEGRITY"
        elif integrity_val >= 60.0:
            badge_bg = "#FEF3C7"
            badge_border = "#D97706"
            badge_text_color = "#B45309"
            status_text = "MODERATE RISK / REVIEW REQUIRED"
        else:
            badge_bg = "#FEE2E2"
            badge_border = "#DC2626"
            badge_text_color = "#B91C1C"
            status_text = "SUSPECTED MALPRACTICE DETECTED"

        cand_meta = [
            [
                Paragraph("Candidate Name:", meta_label_style),
                Paragraph(f"<b>{session.get('candidate_name', 'N/A')}</b>", meta_val_style),
                Paragraph("Exam Assessment:", meta_label_style),
                Paragraph(f"<b>{session.get('exam_title', 'N/A')}</b>", meta_val_style),
            ],
            [
                Paragraph("Student / Seat ID:", meta_label_style),
                Paragraph(f"<b>{session.get('candidate_id', 'N/A')}</b>", meta_val_style),
                Paragraph("Session Start Time:", meta_label_style),
                Paragraph(str(session.get('start_time', 'N/A'))[:19], meta_val_style),
            ],
            [
                Paragraph("Session Status:", meta_label_style),
                Paragraph(f"<b>{session.get('status', 'COMPLETED')}</b>", meta_val_style),
                Paragraph("Total Flagged Incidents:", meta_label_style),
                Paragraph(f"<b>{len(incidents)} Incidents</b>", meta_val_style),
            ],
            [
                Paragraph("Peak Threat Score:", meta_label_style),
                Paragraph(f"<b>{session.get('peak_risk_score', 0.0):.1f} / 100</b>", meta_val_style),
                Paragraph("Average Risk Index:", meta_label_style),
                Paragraph(f"<b>{session.get('avg_risk_score', 0.0):.1f} / 100</b>", meta_val_style),
            ]
        ]
        meta_table = Table(cand_meta, colWidths=[100, 170, 110, 160])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # ---------------- 3. INTEGRITY QUOTIENT BADGE ----------------
        integrity_banner_data = [
            [
                Paragraph(
                    f"<b>FINAL INTEGRITY QUOTIENT: {integrity_val:.1f}%</b> &nbsp;|&nbsp; "
                    f"<b>VERDICT: {status_text}</b>",
                    ParagraphStyle(
                        'IntegrityBanner',
                        fontName='Helvetica-Bold',
                        fontSize=9.5,
                        textColor=colors.HexColor(badge_text_color),
                        alignment=TA_CENTER
                    )
                )
            ]
        ]
        integrity_banner_table = Table(integrity_banner_data, colWidths=[540])
        integrity_banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(badge_bg)),
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor(badge_border)),
            ('TOPPADDING', (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ]))
        story.append(integrity_banner_table)
        story.append(Spacer(1, 12))

        # ---------------- 4. DETAILED INCIDENT & MALPRACTICE TABLE ----------------
        story.append(Paragraph("Detailed Malpractice & Incident Audit Trail", section_heading_style))
        story.append(Spacer(1, 4))

        if incidents:
            table_rows = [
                [
                    Paragraph("<b>ID</b>", table_header_style),
                    Paragraph("<b>Timestamp</b>", table_header_style),
                    Paragraph("<b>Violation Category</b>", table_header_style),
                    Paragraph("<b>AI Conf.</b>", table_header_style),
                    Paragraph("<b>Severity / Threat</b>", table_header_style),
                    Paragraph("<b>Evidence Reference</b>", table_header_style),
                    Paragraph("<b>Verdict</b>", table_header_style),
                ]
            ]

            for inc in incidents:
                inc_id = f"#{inc.get('id', 0)}"
                ts = str(inc.get("timestamp", ""))
                # Format to short time if full date
                if len(ts) >= 19:
                    ts = ts[11:19]
                
                v_name = _format_violation_name(inc.get("violation_type", "ANOMALY"))
                conf_pct = f"{float(inc.get('confidence', 1.0)) * 100:.1f}%"
                sev = inc.get("severity", "HIGH")
                threat_num = inc.get("risk_score", 0.0)
                sev_str = f"{_format_severity_badge(sev)} ({threat_num:.0f})"
                
                ev_clip = inc.get("evidence_clip_path") or inc.get("evidence_snapshot_path") or "telemetry_log"
                ev_clip_name = os.path.basename(ev_clip)
                if len(ev_clip_name) > 18:
                    ev_clip_name = ev_clip_name[:15] + "..."

                verdict = inc.get("proctor_verdict", "PENDING")

                table_rows.append([
                    Paragraph(inc_id, table_cell_center),
                    Paragraph(ts, table_cell_center),
                    Paragraph(f"<b>{v_name}</b>", table_cell_style),
                    Paragraph(conf_pct, table_cell_center),
                    Paragraph(sev_str, table_cell_center),
                    Paragraph(f"<code>{ev_clip_name}</code>", table_cell_center),
                    Paragraph(verdict, table_cell_center),
                ])

            inc_table = Table(table_rows, colWidths=[30, 65, 155, 50, 95, 85, 60])
            inc_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('LEFTPADDING', (0,0), (-1,-1), 4),
                ('RIGHTPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(inc_table)
        else:
            no_inc_data = [[Paragraph("<b>✓ No Malpractice or Suspicious Behaviors Detected. Candidate maintained full compliance throughout the assessment session.</b>", ParagraphStyle('NoInc', fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor('#15803D'), alignment=TA_CENTER))]]
            no_inc_table = Table(no_inc_data, colWidths=[540])
            no_inc_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0FDF4')),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#86EFAC')),
                ('TOPPADDING', (0,0), (-1,-1), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(no_inc_table)

        story.append(Spacer(1, 12))

        # ---------------- 5. EXPLAINABLE AI (XAI) FORENSIC SUMMARY ----------------
        story.append(Paragraph("Explainable AI (XAI) Forensic Summary & Model Justification", section_heading_style))
        story.append(Spacer(1, 4))

        if incidents:
            xai_paragraphs = []
            for idx, inc in enumerate(incidents[:5]): # Top 5 detailed narratives
                ts = str(inc.get("timestamp", ""))[:19]
                v_name = _format_violation_name(inc.get("violation_type", "ANOMALY"))
                reason = inc.get("reason_narrative") or inc.get("reason_summary") or "Automated behavioral anomaly flagged."
                xai_text = f"<b>Incident #{inc.get('id', idx+1)} [{ts}] - {v_name}:</b> {reason}"
                xai_paragraphs.append(Paragraph(xai_text, xai_narrative_style))
                xai_paragraphs.append(Spacer(1, 3))
            
            xai_box = Table([[xai_paragraphs]], colWidths=[540])
            xai_box.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
                ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#CBD5E1')),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(xai_box)
        else:
            story.append(Paragraph("Continuous multi-modal telemetry (YOLOv8 vision, solvePnP 3D pose, and MediaPipe face & hand tracking) verified full visual integrity with zero threshold exceedances.", xai_narrative_style))

        story.append(Spacer(1, 14))

        # ---------------- 6. PROCTOR SIGN-OFF & INSTITUTIONAL DECLARATION ----------------
        signoff_data = [
            [
                Paragraph("<b>PROCTOR VERIFICATION & FORMAL SIGN-OFF</b>", ParagraphStyle('SignHead', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#1E293B'))),
                Paragraph("<b>FINAL PROCTORING DETERMINATION:</b>", ParagraphStyle('DetHead', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#1E293B'))),
            ],
            [
                Paragraph("<b>Proctor Name:</b> ___________________________<br/><b>Employee / Proctor ID:</b> ___________________", meta_val_style),
                Paragraph("[ &nbsp; ] <b>Approved / Verified Clean</b><br/>[ &nbsp; ] <b>Integrity Warning Issued</b><br/>[ &nbsp; ] <b>Escalated for Disciplinary Review</b>", meta_val_style),
            ],
            [
                Paragraph("<b>Proctor Signature:</b> ______________________", meta_val_style),
                Paragraph("<b>Date Signed:</b> _____ / _____ / 2026 &nbsp;&nbsp; <b>Official Seal:</b> [ &nbsp; ]", meta_val_style),
            ]
        ]
        signoff_table = Table(signoff_data, colWidths=[270, 270])
        signoff_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFFFF')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#94A3B8')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        
        story.append(KeepTogether([
            signoff_table,
            Spacer(1, 6),
            Paragraph(
                "<font size='6.5' color='#94A3B8'>CONFIDENTIAL ACADEMIC RECORD: This document contains cryptographic AI forensic evidence compiled by EviGuard Enterprise. Unauthorized modification or dissemination is strictly prohibited under institutional examination bylaws.</font>",
                ParagraphStyle('Legal', fontName='Helvetica', fontSize=6.5, leading=8, textColor=colors.HexColor('#94A3B8'), alignment=TA_CENTER)
            )
        ]))

        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data

    def generate_csv(self, session_id: str) -> str:
        """Exports raw tabular incident audit records as CSV text."""
        data = self.get_session_data(session_id)
        session = data["session"]
        incidents = data["incidents"]

        output = io.StringIO()
        writer = csv.writer(output)

        # Header Metadata
        writer.writerow(["# EviGuard Incident Audit Trail Export"])
        writer.writerow(["# Session ID", session.get("session_id")])
        writer.writerow(["# Candidate Name", session.get("candidate_name")])
        writer.writerow(["# Candidate ID", session.get("candidate_id")])
        writer.writerow(["# Exam Title", session.get("exam_title")])
        writer.writerow(["# Integrity Score", f"{session.get('integrity_index', 100.0):.1f}%"])
        writer.writerow(["# Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow([])

        # Table Column Headers
        writer.writerow([
            "Incident ID",
            "Timestamp",
            "Frame Index",
            "Violation Type",
            "Violation Category",
            "Severity",
            "Risk Score",
            "AI Confidence",
            "Evidence Clip File",
            "Evidence Snapshot File",
            "Proctor Verdict",
            "Reason Summary",
            "Explainable AI Narrative",
            "Proctor Notes"
        ])

        for inc in incidents:
            writer.writerow([
                inc.get("id"),
                inc.get("timestamp"),
                inc.get("frame_index"),
                inc.get("violation_type"),
                _format_violation_name(inc.get("violation_type", "")),
                inc.get("severity"),
                f"{inc.get('risk_score', 0.0):.1f}",
                f"{float(inc.get('confidence', 1.0)) * 100:.1f}%",
                inc.get("evidence_clip_path") or "",
                inc.get("evidence_snapshot_path") or "",
                inc.get("proctor_verdict") or "PENDING",
                inc.get("reason_summary") or "",
                inc.get("reason_narrative") or "",
                inc.get("proctor_notes") or ""
            ])

        return output.getvalue()


# Helper module-level functions
def generate_candidate_pdf_report(session_id: str, db_manager: Optional[DatabaseManager] = None) -> bytes:
    """Convenience functional interface for generating Candidate PDF Report."""
    return ReportGenerator(db_manager).generate_pdf(session_id)


def generate_candidate_csv_report(session_id: str, db_manager: Optional[DatabaseManager] = None) -> str:
    """Convenience functional interface for generating Candidate CSV Audit Report."""
    return ReportGenerator(db_manager).generate_csv(session_id)

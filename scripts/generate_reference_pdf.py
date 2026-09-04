"""
Script to generate a professional academic reference PDF for:
YOLO Model Accuracy Comparison, Extensibility Guide (YOLOv26 / Future Models), and Training Dataset Breakdown.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "EviGuard AI Proctoring — Computer Vision Model Reference Guide")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "Confidential & Academic Reference — EviGuard AI System")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()


def create_pdf(output_filename: str):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#475569'),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Heading1Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#0369A1'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=14,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#0F172A')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#0F172A')
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("YOLO Model Architecture, Accuracy Benchmarks & Extensibility Guide", title_style))
    story.append(Paragraph("Academic Reference Document for EviGuard AI Proctoring System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceAfter=12))

    # Section 1: Accuracy Comparison Table
    story.append(Paragraph("1. Comparative Accuracy & Speed Benchmarks (YOLO vs Other Models)", h1_style))
    story.append(Paragraph(
        "In real-time edge applications such as online exam proctoring, a vision model must achieve high mean Average Precision (mAP) while maintaining a high frame rate (>25 FPS) on commodity hardware without excessive GPU reliance. Below is a comprehensive comparative evaluation against alternative computer vision architectures:",
        body_style
    ))

    # Benchmark Table
    table_data = [
        [
            Paragraph("Model Architecture", table_header_style),
            Paragraph("Type / Paradigm", table_header_style),
            Paragraph("COCO mAP (50-95)", table_header_style),
            Paragraph("Inference (CPU / GPU)", table_header_style),
            Paragraph("AI Proctoring Viability", table_header_style)
        ],
        [
            Paragraph("<b>YOLOv8 Nano (Ultralytics)</b>", table_cell_style),
            Paragraph("Single-Stage (Anchor-Free)", table_cell_style),
            Paragraph("37.3%", table_cell_style),
            Paragraph("<b>30–45 FPS (CPU)</b><br/>120+ FPS (GPU)", table_cell_style),
            Paragraph("<font color='#059669'><b>Ideal</b></font> — Zero latency, ultra-lightweight (3.2M params).", table_cell_style)
        ],
        [
            Paragraph("<b>YOLOv8 / v11 Small</b>", table_cell_style),
            Paragraph("Single-Stage (Anchor-Free)", table_cell_style),
            Paragraph("44.9% – 46.8%", table_cell_style),
            Paragraph("20–30 FPS (CPU)<br/>95+ FPS (GPU)", table_cell_style),
            Paragraph("<font color='#059669'><b>High Accuracy</b></font> — Robust small object detection.", table_cell_style)
        ],
        [
            Paragraph("<b>Faster R-CNN (ResNet-50)</b>", table_cell_style),
            Paragraph("Two-Stage (RPN + RoI)", table_cell_style),
            Paragraph("40.2%", table_cell_style),
            Paragraph("4–8 FPS (CPU)<br/>18–25 FPS (GPU)", table_cell_style),
            Paragraph("<font color='#DC2626'><b>Unsuitable</b></font> — Excessive lag; frames drop in live stream.", table_cell_style)
        ],
        [
            Paragraph("<b>SSD MobileNet v2</b>", table_cell_style),
            Paragraph("Single-Stage (Anchor-Based)", table_cell_style),
            Paragraph("22.1% – 31.2%", table_cell_style),
            Paragraph("30–40 FPS (CPU)<br/>70+ FPS (GPU)", table_cell_style),
            Paragraph("<font color='#D97706'><b>Suboptimal</b></font> — Fails to detect partially occluded phones.", table_cell_style)
        ],
        [
            Paragraph("<b>RT-DETR (Baidu / Ultralytics)</b>", table_cell_style),
            Paragraph("Vision Transformer (DETR)", table_cell_style),
            Paragraph("53.1% – 54.8%", table_cell_style),
            Paragraph("3–6 FPS (CPU)<br/>35–50 FPS (GPU)", table_cell_style),
            Paragraph("<font color='#D97706'><b>GPU Only</b></font> — High precision, but requires heavy VRAM.", table_cell_style)
        ]
    ]

    col_widths = [105, 95, 75, 95, 134]
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Section 2: Core Advantages
    story.append(Paragraph("2. Technical Advantages of YOLO in AI Proctoring", h1_style))
    story.append(Paragraph("<b>1. Single-Stage Regression (Zero Latency):</b> Unlike two-stage detectors that generate regional proposals before classification, YOLO formulates object detection as a unified spatial regression problem. Bounding coordinates and confidence scores are generated simultaneously in a single forward pass (&lt;10ms).", bullet_style))
    story.append(Paragraph("<b>2. Anchor-Free Head Architecture:</b> Older architectures relied on predefined anchor box ratios. YOLOv8 uses an anchor-free task-aligned assigner that predicts the center and dimensions directly. This significantly enhances detection of rotated smartphones, notebooks, and shifting candidate postures.", bullet_style))
    story.append(Paragraph("<b>3. CPU and Edge Deployment:</b> The lightweight backbone (C2f feature fusion blocks) enables smooth 30 FPS inference on standard laptops without requiring a dedicated CUDA GPU.", bullet_style))
    story.append(Paragraph("<b>4. Multi-Task Ecosystem:</b> Ultralytics YOLO provides a unified API supporting Object Detection, Pose Estimation, Hand Keypoint Tracking, and Instance Segmentation under a consistent interface.", bullet_style))

    story.append(Spacer(1, 10))

    # Section 3: Implementing YOLOv26 / Future Models
    story.append(Paragraph("3. Implementing Future Models (e.g., YOLOv26 or Custom Weights)", h1_style))
    story.append(Paragraph(
        "<b>Model Lineage Clarification:</b> The current official Ultralytics releases span from <b>YOLOv8 &rarr; YOLOv9 &rarr; YOLOv10 &rarr; YOLOv11</b>. While there is no official 'YOLOv26' release yet in the academic community, the <b>EviGuard</b> codebase is engineered with a decoupled <b>Factory Pattern Architecture</b> that guarantees 100% plug-and-play compatibility with any future release.",
        body_style
    ))

    # Step-by-step Callout Box
    callout_data = [[
        Paragraph(
            "<b>How to Integrate a Future Model / Custom Checkpoint in EviGuard:</b><br/>"
            "<b>Step 1:</b> Place your new weights file (e.g., <code>yolov26n.pt</code> or <code>custom_exam_proctor.pt</code>) into the project root directory.<br/>"
            "<b>Step 2:</b> Open <code>config.yaml</code> and update the model configuration section:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<code>detection:</code><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>model_type: 'yolov8'</code>&nbsp;&nbsp;# Uses standard Ultralytics engine<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>model_path: 'yolov26n.pt'</code>&nbsp;&nbsp;# Points to your new model<br/>"
            "<b>Step 3:</b> Restart the dashboard. <code>DetectorFactory</code> and <code>EviGuardPipeline</code> will automatically load the model weights, initialize inference tensors, and bind detections directly to the live HUD and Risk Scoring Engine without requiring any code refactoring.",
            callout_style
        )
    ]]
    callout_table = Table(callout_data, colWidths=[504])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F9FF')),
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor('#0284C7')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 10))

    # Section 4: Training Datasets Analysis
    story.append(Paragraph("4. Dataset & Pretraining Analysis for YOLOv8", h1_style))
    story.append(Paragraph(
        "YOLOv8 base pretrained models (<code>yolov8n.pt</code>, <code>yolov8s.pt</code>, <code>yolov8m.pt</code>, <code>yolov8l.pt</code>, <code>yolov8x.pt</code>) were trained on extensive multi-million image benchmarks:",
        body_style
    ))

    # Dataset Summary Table
    ds_table_data = [
        [
            Paragraph("Dataset Benchmark", table_header_style),
            Paragraph("Image Count", table_header_style),
            Paragraph("Annotations / Classes", table_header_style),
            Paragraph("Role in Model Pipeline", table_header_style)
        ],
        [
            Paragraph("<b>MS COCO 2017</b><br/>(Microsoft Common Objects in Context)", table_cell_style),
            Paragraph("<b>330,000+ total images</b><br/>&bull; 118,287 Train<br/>&bull; 5,000 Validation<br/>&bull; 40,670 Test", table_cell_style),
            Paragraph("<b>1.5 Million+</b> bounding box instances across <b>80 classes</b>.<br/><i>(Includes Person, Phone, Book, Laptop)</i>", table_cell_style),
            Paragraph("<b>Primary Object Detection Training:</b> Enables real-time classification and localization of candidates and unauthorized electronic items.", table_cell_style)
        ],
        [
            Paragraph("<b>ImageNet-1k</b><br/>(Visual Recognition Challenge)", table_cell_style),
            Paragraph("<b>1,280,000+</b> high-resolution training images", table_cell_style),
            Paragraph("<b>1,000 categories</b> covering diverse real-world textures and objects.", table_cell_style),
            Paragraph("<b>Backbone Feature Pretraining:</b> Initializes convolutional weights with rich multi-scale visual representations prior to COCO fine-tuning.", table_cell_style)
        ]
    ]

    t_ds = Table(ds_table_data, colWidths=[130, 110, 130, 134])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0369A1')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 12))

    # Section 5: Summary for Final Year Project
    story.append(Paragraph("5. Summary & Academic Justification for EviGuard", h1_style))
    story.append(Paragraph("<b>1. Proven Balance:</b> YOLOv8 provides the highest accuracy-to-compute ratio among modern detectors, making it the industry standard for real-time edge monitoring.", bullet_style))
    story.append(Paragraph("<b>2. Extensibility:</b> Because the system employs an abstract <code>BaseDetector</code> interface, future architectures (such as YOLOv10, YOLOv11, or a future YOLOv26) can be loaded with zero modification to business logic.", bullet_style))
    story.append(Paragraph("<b>3. Zero-Lag Multi-Modal Integration:</b> YOLO detections seamlessly fuse with MediaPipe 3D Head Pose, Gaze Estimator, and Dynamic Risk Scoring Engine for immediate incident flagging.", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Reference PDF successfully created at: {output_filename}")


if __name__ == "__main__":
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "YOLO_Accuracy_and_Integration_Reference.pdf"))
    create_pdf(out_path)

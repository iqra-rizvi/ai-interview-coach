"""
backend/utils/pdf_generator.py
Builds a professional PDF interview performance report using ReportLab.
"""
from __future__ import annotations
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# ─── Color palette ────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1e3a5f")
TEAL   = colors.HexColor("#0d9488")
LIGHT  = colors.HexColor("#f0f9ff")
GRAY   = colors.HexColor("#64748b")
GREEN  = colors.HexColor("#16a34a")
RED    = colors.HexColor("#dc2626")
AMBER  = colors.HexColor("#d97706")
WHITE  = colors.white
BLACK  = colors.black


def _score_color(score: float) -> colors.Color:
    if score >= 8:
        return GREEN
    elif score >= 5:
        return AMBER
    return RED


def build_pdf_report(
    path: str,
    candidate_name: str,
    role: str,
    interview_type: str,
    avg_score: float,
    executive_summary: str,
    qa_pairs: list[dict],
    session_date: str,
) -> None:
    """Generate and save the full PDF report to `path`."""
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "title", fontSize=22, textColor=NAVY, spaceAfter=4,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "subtitle", fontSize=11, textColor=GRAY, spaceAfter=2,
        fontName="Helvetica", alignment=TA_CENTER,
    )
    h2_style = ParagraphStyle(
        "h2", fontSize=14, textColor=NAVY, spaceBefore=16, spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    h3_style = ParagraphStyle(
        "h3", fontSize=11, textColor=TEAL, spaceBefore=10, spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "body", fontSize=10, textColor=BLACK, leading=15,
        fontName="Helvetica", spaceAfter=6,
    )
    italic_style = ParagraphStyle(
        "italic", fontSize=10, textColor=GRAY, leading=14,
        fontName="Helvetica-Oblique", spaceAfter=6,
    )
    label_style = ParagraphStyle(
        "label", fontSize=9, textColor=WHITE,
        fontName="Helvetica-Bold",
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("🤖 AI Interview Coach", title_style))
    story.append(Paragraph("Mock Interview Performance Report", subtitle_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL))
    story.append(Spacer(1, 0.4 * cm))

    # ── Candidate info table ──────────────────────────────────────────────────
    info_data = [
        ["Candidate", candidate_name,   "Date",           session_date],
        ["Target Role", role,            "Interview Type", interview_type.title()],
        ["Avg. Score",  f"{avg_score}/10", "Questions",    str(len(qa_pairs))],
    ]
    info_table = Table(info_data, colWidths=[3 * cm, 6.5 * cm, 3.5 * cm, 4 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT),
        ("TEXTCOLOR",  (0, 0), (0, -1), NAVY),
        ("TEXTCOLOR",  (2, 0), (2, -1), NAVY),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTNAME",   (1, 0), (1, -1), "Helvetica"),
        ("FONTNAME",   (3, 0), (3, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5 * cm))

    # ── Score bar (visual) ────────────────────────────────────────────────────
    scores = [qa.get("score", 0) for qa in qa_pairs]
    score_row = [f"Q{i+1}\n{s:.1f}" for i, s in enumerate(scores)]
    score_colors = [_score_color(s) for s in scores]

    if scores:
        score_data = [score_row]
        col_w = 17 * cm / max(len(scores), 1)
        score_table = Table(score_data, colWidths=[col_w] * len(scores), rowHeights=[1.2 * cm])
        ts = TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
        ])
        for i, c in enumerate(score_colors):
            ts.add("BACKGROUND", (i, 0), (i, 0), c)
        score_table.setStyle(ts)
        story.append(Paragraph("Score Overview", h2_style))
        story.append(score_table)
        story.append(Spacer(1, 0.3 * cm))

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", h2_style))
    for para in executive_summary.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))
    story.append(Spacer(1, 0.3 * cm))

    # ── Detailed Q&A Breakdown ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Detailed Question Breakdown", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL))
    story.append(Spacer(1, 0.3 * cm))

    for qa in qa_pairs:
        q_num = qa.get("question_number", "?")
        score = qa.get("score", 0)
        sc = _score_color(score)

        # Question header
        story.append(Paragraph(f"Question {q_num}  •  Topic: {qa.get('topic', '—')}", h3_style))

        # Score badge
        badge = Table([[f"Score: {score:.1f} / 10"]], colWidths=[4 * cm], rowHeights=[0.6 * cm])
        badge.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), sc),
            ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(badge)
        story.append(Spacer(1, 0.2 * cm))

        story.append(Paragraph(f"<b>Q:</b> {qa.get('question', '')}", body_style))
        story.append(Paragraph(f"<b>A:</b> {qa.get('answer', '(no answer)')}", italic_style))

        if qa.get("strengths"):
            story.append(Paragraph("<b>Strengths:</b>", body_style))
            for s in qa["strengths"]:
                story.append(Paragraph(f"  ✓  {s}", body_style))

        if qa.get("weaknesses"):
            story.append(Paragraph("<b>Areas to improve:</b>", body_style))
            for w in qa["weaknesses"]:
                story.append(Paragraph(f"  ✗  {w}", body_style))

        if qa.get("feedback"):
            story.append(Paragraph("<b>Coach's feedback:</b>", body_style))
            story.append(Paragraph(qa["feedback"], italic_style))

        if qa.get("ideal_answer_outline"):
            story.append(Paragraph("<b>Ideal answer outline:</b>", body_style))
            story.append(Paragraph(qa["ideal_answer_outline"], body_style))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
        story.append(Spacer(1, 0.3 * cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Generated by AI Interview Coach · Keep practicing — you've got this! 🚀",
        ParagraphStyle("footer", fontSize=9, textColor=GRAY, alignment=TA_CENTER),
    ))

    doc.build(story)

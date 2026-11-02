"""
backend/utils/pdf_generator.py
Builds a professional, auditable PDF performance report using ReportLab.
Implements XML sanitization, 5-dimensional psychometric rubric breakdown,
verbatim evidence citation display, and pedagogical remediation curriculum.
"""
from __future__ import annotations
import html
from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ─── Color palette ────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1e3a5f")
TEAL   = colors.HexColor("#0d9488")
LIGHT  = colors.HexColor("#f0fdf4")
GRAY   = colors.HexColor("#64748b")
GREEN  = colors.HexColor("#16a34a")
RED    = colors.HexColor("#dc2626")
AMBER  = colors.HexColor("#d97706")
WHITE  = colors.white
BLACK  = colors.black
BORDER = colors.HexColor("#cbd5e1")
CODE_BG = colors.HexColor("#f1f5f9")


def _esc(val: Any) -> str:
    """XML-escape strings so ReportLab never crashes on code tokens like <T>, &, >."""
    if val is None:
        return ""
    text = str(val).strip()
    return html.escape(text)


def _score_color(score: float) -> colors.Color:
    if score >= 7.5:
        return GREEN
    elif score >= 5.0:
        return AMBER
    return RED


def build_pdf_report(
    path: str,
    candidate_name: str,
    role: str,
    interview_type: str,
    avg_score: float,
    executive_summary: str,
    qa_pairs: List[Dict[str, Any]],
    session_date: str,
    radar: Optional[Dict[str, float]] = None,
    curriculum: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Generate and save the full AgentHire PDF report to `path`."""
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "title", fontSize=22, textColor=NAVY, spaceAfter=4,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "subtitle", fontSize=10, textColor=GRAY, spaceAfter=2,
        fontName="Helvetica", alignment=TA_CENTER,
    )
    h2_style = ParagraphStyle(
        "h2", fontSize=13, textColor=NAVY, spaceBefore=14, spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    h3_style = ParagraphStyle(
        "h3", fontSize=11, textColor=TEAL, spaceBefore=8, spaceAfter=3,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "body", fontSize=9, textColor=BLACK, leading=14,
        fontName="Helvetica", spaceAfter=4,
    )
    italic_style = ParagraphStyle(
        "italic", fontSize=9, textColor=colors.HexColor("#334155"), leading=13,
        fontName="Helvetica-Oblique", spaceAfter=4,
    )
    quote_style = ParagraphStyle(
        "quote", fontSize=8.5, textColor=colors.HexColor("#1e293b"), leading=12,
        fontName="Courier", backColor=CODE_BG, spaceAfter=4, leftIndent=8,
    )
    footer_style = ParagraphStyle(
        "footer", fontSize=8, textColor=GRAY, alignment=TA_CENTER,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("AgentHire: Multi-Agent Interview Platform", title_style))
    story.append(Paragraph("Psychometrically Calibrated Candidate Assessment & Diagnostic Report", subtitle_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL))
    story.append(Spacer(1, 0.3 * cm))

    # ── Candidate Information Table ───────────────────────────────────────────
    info_data = [
        ["Candidate", _esc(candidate_name), "Evaluation Date", _esc(session_date)],
        ["Target Position", _esc(role), "Screening Modality", _esc(interview_type.title())],
        ["Composite Score", f"{avg_score:.1f} / 10.0", "Items Answered", str(len(qa_pairs))],
    ]
    info_table = Table(info_data, colWidths=[3.2 * cm, 6.2 * cm, 3.8 * cm, 4.2 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR",  (0, 0), (0, -1), NAVY),
        ("TEXTCOLOR",  (2, 0), (2, -1), NAVY),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTNAME",   (1, 0), (1, -1), "Helvetica"),
        ("FONTNAME",   (3, 0), (3, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── 5-Dimensional Psychometric Competency Radar Breakdown ─────────────────
    if radar:
        story.append(Paragraph("Five Orthogonal Psychometric Competency Dimensions", h2_style))
        radar_header = ["Evaluation Dimension", "Score Anchor (1 to 5)", "Assessed Competency Level"]
        radar_rows = [radar_header]
        for dim_name, score_val in radar.items():
            level = "Mastery (Advanced)" if score_val >= 4.2 else ("Competent" if score_val >= 3.0 else "Developing (Gap)")
            radar_rows.append([dim_name, f"{score_val:.2f} / 5.0", level])

        radar_table = Table(radar_rows, colWidths=[6.5 * cm, 4.5 * cm, 6.4 * cm])
        radar_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
            ("GRID",       (0, 0), (-1, -1), 0.3, BORDER),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#f8fafc")]),
        ]))
        story.append(radar_table)
        story.append(Spacer(1, 0.4 * cm))

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(Paragraph("Executive Performance Synthesis", h2_style))
    paragraphs = (executive_summary or "").split("\n\n")
    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(_esc(para.strip()), body_style))
    story.append(Spacer(1, 0.3 * cm))

    # ── Actionable Pedagogical Remediation Plan ───────────────────────────────
    if curriculum:
        story.append(Paragraph("Curated Pedagogical Remediation Curriculum", h2_style))
        for item in curriculum:
            d_name = _esc(item.get("dimension", ""))
            ref = _esc(item.get("reference", ""))
            action = _esc(item.get("action", ""))
            story.append(Paragraph(f"• <b>{d_name} Focus:</b> {ref}", body_style))
            story.append(Paragraph(f"  <i>Recommended Action:</i> {action}", italic_style))
        story.append(Spacer(1, 0.3 * cm))

    # ── Detailed Q&A and Verbatim Grounded Audit Trail ────────────────────────
    if qa_pairs:
        story.append(PageBreak())
        story.append(Paragraph("Verbatim Grounded Evidentiary Audit Trail", title_style))
        story.append(Paragraph("Detailed turn transcript with verbatim evidence verification gate records", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1, color=TEAL))
        story.append(Spacer(1, 0.3 * cm))

        for idx, qa in enumerate(qa_pairs):
            q_num = qa.get("question_number", idx + 1)
            score = float(qa.get("score", 0.0) or 0.0)
            sc = _score_color(score)
            tier = _esc(qa.get("pedagogical_tier", "Tier 1: Foundational"))
            topic = _esc(qa.get("topic", "General Technical"))

            item_flow = []
            item_flow.append(Paragraph(f"Question {q_num}  •  {tier}  •  Topic: {topic}", h3_style))

            badge = Table([[f"Composite Score: {score:.1f} / 10.0"]], colWidths=[4.8 * cm], rowHeights=[0.55 * cm])
            badge.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), sc),
                ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            item_flow.append(badge)
            item_flow.append(Spacer(1, 0.15 * cm))

            item_flow.append(Paragraph(f"<b>Prompt:</b> {_esc(qa.get('question', ''))}", body_style))
            item_flow.append(Paragraph(f"<b>Candidate Answer:</b> {_esc(qa.get('answer', '(No answer recorded)'))}", italic_style))

            # Verbatim Evidentiary Citations
            rubric_scores = qa.get("rubric_scores") or {}
            if rubric_scores:
                item_flow.append(Paragraph("<b>Evidentiary Grounding Audit:</b>", body_style))
                for d_name, r_item in rubric_scores.items():
                    if isinstance(r_item, dict):
                        q_quote = r_item.get("verbatim_evidence_quote", "")
                        r_score = r_item.get("score", 3)
                        r_rat = r_item.get("evaluator_rationale", "")
                    elif hasattr(r_item, "verbatim_evidence_quote"):
                        q_quote = r_item.verbatim_evidence_quote
                        r_score = r_item.score
                        r_rat = r_item.evaluator_rationale
                    else:
                        continue
                    item_flow.append(
                        Paragraph(f"<b>[{_esc(d_name)} — Score: {r_score}/5]:</b> {_esc(r_rat)}", body_style)
                    )
                    if q_quote and "[UNGROUNDED" not in q_quote and "[NO" not in q_quote:
                        item_flow.append(Paragraph(f"<i>Verbatim Citation:</i> \"{_esc(q_quote)}\"", quote_style))

            if qa.get("feedback"):
                item_flow.append(Paragraph(f"<b>Pedagogical Coach Feedback:</b> {_esc(qa.get('feedback'))}", italic_style))

            item_flow.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#e2e8f0")))
            item_flow.append(Spacer(1, 0.25 * cm))

            story.append(KeepTogether(item_flow))
    else:
        story.append(Paragraph("<i>No interview questions were completed in this session.</i>", body_style))

    # ── Ethical Compliance Footer ─────────────────────────────────────────────
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "AgentHire AI System · Compliant with EU AI Act (Title III Art. 6) & IEEE Ethically Aligned Design · Advisory Decision-Support Only",
        footer_style,
    ))

    doc.build(story)

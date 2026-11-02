"""
backend/api/routes/reports.py
Report generation and download endpoints for AgentHire.
"""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from loguru import logger

from backend.models.database import get_db, InterviewSession, QAPair, User
from backend.models.schemas import SessionReport, QAPairResponse, RubricScore
from backend.agents.report_agent import ReportAgent

router = APIRouter()
_reporter = ReportAgent()


@router.post("/{session_id}/generate", response_model=SessionReport)
def generate_report(session_id: str, db: Session = Depends(get_db)):
    """
    Synthesize 5-dimensional psychometric performance report,
    compute radar averages, compile PDF, and return diagnostic artifacts.
    """
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user = db.query(User).filter(User.id == session.user_id).first()
    candidate_name = user.name if user else "Candidate"

    pairs = (
        db.query(QAPair)
        .filter(QAPair.session_id == session_id, QAPair.answer.isnot(None))
        .order_by(QAPair.question_number, QAPair.is_followup)
        .all()
    )

    if not pairs:
        raise HTTPException(status_code=400, detail="No answered questions found in this session")

    qa_dicts = []
    qa_responses = []

    for p in pairs:
        rubric_parsed = {}
        if p.rubric_breakdown:
            try:
                raw_r = json.loads(p.rubric_breakdown)
                rubric_parsed = {k: RubricScore(**v) for k, v in raw_r.items()}
            except Exception:
                pass

        qa_dicts.append({
            "question_number": p.question_number,
            "question": p.question,
            "answer": p.answer,
            "topic": p.topic or "",
            "score": p.score or 0.0,
            "strengths": json.loads(p.strengths) if p.strengths else [],
            "weaknesses": json.loads(p.weaknesses) if p.weaknesses else [],
            "feedback": p.feedback or "",
            "ideal_answer_outline": p.ideal_answer_outline or "",
            "rubric_scores": {k: v.model_dump() for k, v in rubric_parsed.items()} if rubric_parsed else None,
            "pedagogical_tier": p.pedagogical_tier,
            "difficulty_lambda": p.difficulty_lambda,
            "is_followup": p.is_followup,
        })

        qa_responses.append(QAPairResponse(
            id=str(p.id),
            question_number=p.question_number,
            question=p.question,
            answer=p.answer,
            score=p.score,
            strengths=json.loads(p.strengths) if p.strengths else None,
            weaknesses=json.loads(p.weaknesses) if p.weaknesses else None,
            feedback=p.feedback,
            ideal_answer_outline=p.ideal_answer_outline,
            topic=p.topic,
            rubric_scores=rubric_parsed if rubric_parsed else None,
            pedagogical_tier=p.pedagogical_tier,
            difficulty_lambda=p.difficulty_lambda,
            is_followup=bool(p.is_followup),
        ))

    summary, pdf_path, avg_score, radar, curriculum = _reporter.compile_report(
        session_id=session_id,
        candidate_name=candidate_name,
        role=session.role,
        interview_type=session.interview_type,
        qa_pairs=qa_dicts,
    )

    # Persist results to DB
    session.avg_score = avg_score
    session.report_path = pdf_path
    session.status = "completed"
    db.commit()

    detected_gaps = []
    if session.candidate_profile:
        try:
            c_data = json.loads(session.candidate_profile)
            detected_gaps = c_data.get("detected_competency_gaps", [])
        except Exception:
            pass

    # Windows-safe filename extraction
    pdf_filename = Path(pdf_path).name if pdf_path else None
    pdf_url = f"/reports/{pdf_filename}" if pdf_filename else None

    return SessionReport(
        session_id=session_id,
        candidate_name=candidate_name,
        role=session.role,
        interview_type=session.interview_type,
        avg_score=avg_score,
        total_questions=len(pairs),
        executive_summary=summary,
        qa_pairs=qa_responses,
        report_pdf_url=pdf_url,
        rubric_radar=radar,
        competency_divergence=session.competency_divergence,
        detected_competency_gaps=detected_gaps,
        remediation_plan=curriculum,
    )


@router.get("/{session_id}/download")
def download_report(session_id: str, db: Session = Depends(get_db)):
    """Download the generated PDF report for a session."""
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.report_path or not Path(session.report_path).exists():
        raise HTTPException(status_code=404, detail="Report file not found. Call /generate first.")

    return FileResponse(
        path=session.report_path,
        media_type="application/pdf",
        filename=f"AgentHire_Report_{session_id[:8]}.pdf",
    )

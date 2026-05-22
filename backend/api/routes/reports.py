"""
backend/api/routes/reports.py
Report endpoints — generate and download session PDF reports.
"""
from __future__ import annotations
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from loguru import logger

from backend.models.database import get_db, InterviewSession, QAPair, User
from backend.models.schemas import SessionReport, QAPairResponse
from backend.agents.report_agent import ReportAgent

router = APIRouter()
_reporter = ReportAgent()


@router.post("/{session_id}/generate", response_model=SessionReport)
def generate_report(session_id: str, db: Session = Depends(get_db)):
    """
    Generate (or re-generate) the PDF performance report for a session.
    """
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user = db.query(User).filter(User.id == session.user_id).first()
    candidate_name = user.name if user else "Candidate"

    pairs = (
        db.query(QAPair)
        .filter(QAPair.session_id == session_id, QAPair.answer.isnot(None))
        .order_by(QAPair.question_number)
        .all()
    )

    if not pairs:
        raise HTTPException(status_code=400, detail="No answered questions found in this session")

    qa_dicts = [
        {
            "question_number": p.question_number,
            "question": p.question,
            "answer": p.answer,
            "topic": p.topic or "",
            "score": p.score or 0,
            "strengths": json.loads(p.strengths) if p.strengths else [],
            "weaknesses": json.loads(p.weaknesses) if p.weaknesses else [],
            "feedback": p.feedback or "",
            "ideal_answer_outline": p.ideal_answer_outline or "",
        }
        for p in pairs
    ]

    summary, pdf_path, avg_score = _reporter.compile_report(
        session_id=session_id,
        candidate_name=candidate_name,
        role=session.role,
        interview_type=session.interview_type,
        qa_pairs=qa_dicts,
    )

    # Persist to DB
    session.avg_score = avg_score
    session.report_path = pdf_path
    db.commit()

    qa_responses = [
        QAPairResponse(
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
        )
        for p in pairs
    ]

    pdf_url = f"/reports/{pdf_path.split('/')[-1]}" if pdf_path else None

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
    )


@router.get("/{session_id}/download")
def download_report(session_id: str, db: Session = Depends(get_db)):
    """Download the PDF report for a completed session."""
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.report_path:
        raise HTTPException(status_code=404, detail="Report not yet generated. Call /generate first.")

    return FileResponse(
        path=session.report_path,
        media_type="application/pdf",
        filename=f"interview_report_{session_id[:8]}.pdf",
    )

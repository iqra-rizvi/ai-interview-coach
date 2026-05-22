"""
backend/api/routes/sessions.py
Session management endpoints:
  POST /api/sessions/             — create a new interview session
  POST /api/sessions/{id}/start   — generate the first question
  POST /api/sessions/{id}/answer  — submit answer → get eval + feedback + next Q
  GET  /api/sessions/{id}         — get session details
  GET  /api/sessions/{id}/history — full Q&A history
  POST /api/sessions/{id}/finish  — force-complete and generate report
"""
from __future__ import annotations
import uuid
import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger

from backend.models.database import get_db, InterviewSession, QAPair, User
from backend.models.schemas import (
    SessionCreate, SessionResponse, AnswerSubmit,
    AnswerFeedbackResponse, NextQuestionResponse, EvaluationResult,
    SessionReport, QAPairResponse,
)
from backend.agents.interviewer_agent import InterviewerAgent
from backend.agents.evaluator_agent import EvaluatorAgent
from backend.agents.feedback_agent import FeedbackAgent
from backend.agents.report_agent import ReportAgent
from backend.utils.config import get_settings

settings = get_settings()
router = APIRouter()

# Agent singletons shared across requests
_interviewer = InterviewerAgent()
_evaluator   = EvaluatorAgent()
_feedback    = FeedbackAgent()
_reporter    = ReportAgent()

# In-memory session context (in prod use Redis)
_session_context: dict[str, dict] = {}


def _get_session_or_404(session_id: str, db: Session) -> InterviewSession:
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ─── Create session ───────────────────────────────────────────────────────────

@router.post("/", response_model=SessionResponse, status_code=201)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    """Create a new interview session for a user."""
    # Ensure user exists (auto-create if not)
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        user = User(id=payload.user_id, name="Candidate")
        db.add(user)
        db.commit()

    session = InterviewSession(
        id=str(uuid.uuid4()),
        user_id=payload.user_id,
        role=payload.role,
        interview_type=payload.interview_type,
        status="active",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    # Init in-memory context
    _session_context[session.id] = {
        "question_number": 0,
        "covered_topics": [],
        "resume_summary": "No resume uploaded.",
    }

    logger.info(f"Session created: {session.id} | {session.role} | {session.interview_type}")
    return session


# ─── Start (generate first question) ─────────────────────────────────────────

@router.post("/{session_id}/start", response_model=NextQuestionResponse)
def start_session(session_id: str, db: Session = Depends(get_db)):
    """Generate the opening question for a session."""
    session = _get_session_or_404(session_id, db)
    ctx = _session_context.setdefault(session_id, {
        "question_number": 0,
        "covered_topics": [],
        "resume_summary": "No resume uploaded.",
    })

    ctx["question_number"] = 1
    question, topic = _interviewer.generate_first_question(
        session_id=session_id,
        role=session.role,
        interview_type=session.interview_type,
        resume_summary=ctx["resume_summary"],
        max_questions=settings.max_questions,
    )

    # Persist question to DB (answer will be filled later)
    qa = QAPair(
        id=str(uuid.uuid4()),
        session_id=session_id,
        question_number=1,
        question=question,
        topic=topic,
    )
    db.add(qa)
    db.commit()

    ctx["current_qa_id"] = qa.id
    ctx["current_question"] = question
    ctx["current_topic"] = topic

    return NextQuestionResponse(
        question=question,
        question_number=1,
        topic=topic,
        is_final=(1 >= settings.max_questions),
    )


# ─── Submit answer ────────────────────────────────────────────────────────────

@router.post("/{session_id}/answer", response_model=AnswerFeedbackResponse)
def submit_answer(
    session_id: str,
    payload: AnswerSubmit,
    db: Session = Depends(get_db),
):
    """
    Submit a candidate's answer.
    Returns: evaluation + feedback + next question (or completion notice).
    """
    session = _get_session_or_404(session_id, db)
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    ctx = _session_context.get(session_id)
    if not ctx:
        raise HTTPException(status_code=400, detail="Session not started — call /start first")

    q_num = ctx["question_number"]
    current_question = ctx.get("current_question", "")

    # 1. Evaluate the answer
    evaluation = _evaluator.evaluate(
        session_id=session_id,
        question=current_question,
        answer=payload.answer,
        role=session.role,
        interview_type=session.interview_type,
    )

    # 2. Generate coaching feedback
    feedback_text = _feedback.generate_feedback(
        session_id=session_id,
        question=current_question,
        answer=payload.answer,
        evaluation=evaluation,
        role=session.role,
        resume_summary=ctx["resume_summary"],
        topic=ctx.get("current_topic", ""),
    )

    # 3. Persist to DB
    qa_record = db.query(QAPair).filter(
        QAPair.session_id == session_id,
        QAPair.question_number == q_num,
    ).first()

    if qa_record:
        qa_record.answer = payload.answer
        qa_record.score = evaluation["score"]
        qa_record.strengths = json.dumps(evaluation.get("strengths", []))
        qa_record.weaknesses = json.dumps(evaluation.get("weaknesses", []))
        qa_record.feedback = feedback_text
        qa_record.ideal_answer_outline = evaluation.get("ideal_answer_outline", "")
        db.commit()

    # Update covered topics
    current_topic = ctx.get("current_topic", "")
    if current_topic and current_topic not in ctx["covered_topics"]:
        ctx["covered_topics"].append(current_topic)

    # 4. Decide next step
    next_q_num = q_num + 1
    session_complete = next_q_num > settings.max_questions

    eval_response = EvaluationResult(
        score=evaluation["score"],
        strengths=evaluation.get("strengths", []),
        weaknesses=evaluation.get("weaknesses", []),
        ideal_answer_outline=evaluation.get("ideal_answer_outline", ""),
        follow_up_suggested=evaluation.get("follow_up_suggested", ""),
    )

    next_q_response = None
    if not session_complete:
        # Generate next question
        is_final = next_q_num >= settings.max_questions
        next_question, next_topic = _interviewer.generate_next_question(
            session_id=session_id,
            role=session.role,
            interview_type=session.interview_type,
            resume_summary=ctx["resume_summary"],
            last_answer=payload.answer,
            question_number=next_q_num,
            max_questions=settings.max_questions,
            covered_topics=ctx["covered_topics"],
            is_final=is_final,
        )

        # Persist next Q
        next_qa = QAPair(
            id=str(uuid.uuid4()),
            session_id=session_id,
            question_number=next_q_num,
            question=next_question,
            topic=next_topic,
        )
        db.add(next_qa)
        db.commit()

        ctx["question_number"] = next_q_num
        ctx["current_question"] = next_question
        ctx["current_topic"] = next_topic
        ctx["current_qa_id"] = next_qa.id

        next_q_response = NextQuestionResponse(
            question=next_question,
            question_number=next_q_num,
            topic=next_topic,
            is_final=is_final,
        )
    else:
        # Mark session complete
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        db.commit()

    return AnswerFeedbackResponse(
        evaluation=eval_response,
        feedback=feedback_text,
        next_question=next_q_response,
        session_complete=session_complete,
    )


# ─── Get session ──────────────────────────────────────────────────────────────

@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    return _get_session_or_404(session_id, db)


# ─── Full Q&A history ─────────────────────────────────────────────────────────

@router.get("/{session_id}/history", response_model=List[QAPairResponse])
def get_history(session_id: str, db: Session = Depends(get_db)):
    _get_session_or_404(session_id, db)
    pairs = (
        db.query(QAPair)
        .filter(QAPair.session_id == session_id)
        .order_by(QAPair.question_number)
        .all()
    )
    result = []
    for p in pairs:
        result.append(QAPairResponse(
            id=p.id,
            question_number=p.question_number,
            question=p.question,
            answer=p.answer,
            score=p.score,
            strengths=json.loads(p.strengths) if p.strengths else None,
            weaknesses=json.loads(p.weaknesses) if p.weaknesses else None,
            feedback=p.feedback,
            ideal_answer_outline=p.ideal_answer_outline,
            topic=p.topic,
        ))
    return result


# ─── Force finish ─────────────────────────────────────────────────────────────

@router.post("/{session_id}/finish")
def finish_session(session_id: str, db: Session = Depends(get_db)):
    """
    Manually complete the session and trigger report generation.
    Useful if the user wants to end early.
    """
    session = _get_session_or_404(session_id, db)
    ctx = _session_context.get(session_id, {})

    pairs = (
        db.query(QAPair)
        .filter(QAPair.session_id == session_id)
        .order_by(QAPair.question_number)
        .all()
    )

    qa_dicts = []
    for p in pairs:
        if p.answer:
            qa_dicts.append({
                "question_number": p.question_number,
                "question": p.question,
                "answer": p.answer,
                "topic": p.topic or "",
                "score": p.score or 0,
                "strengths": json.loads(p.strengths) if p.strengths else [],
                "weaknesses": json.loads(p.weaknesses) if p.weaknesses else [],
                "feedback": p.feedback or "",
                "ideal_answer_outline": p.ideal_answer_outline or "",
            })

    user = db.query(User).filter(User.id == session.user_id).first()
    candidate_name = user.name if user else "Candidate"

    summary, pdf_path, avg_score = _reporter.compile_report(
        session_id=session_id,
        candidate_name=candidate_name,
        role=session.role,
        interview_type=session.interview_type,
        qa_pairs=qa_dicts,
    )

    session.status = "completed"
    session.avg_score = avg_score
    session.report_path = pdf_path
    session.completed_at = datetime.utcnow()
    db.commit()

    return {
        "message": "Session completed",
        "avg_score": avg_score,
        "report_path": pdf_path,
        "executive_summary": summary,
    }

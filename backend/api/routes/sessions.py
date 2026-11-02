"""
backend/api/routes/sessions.py
Session management endpoints for AgentHire:
  POST /api/sessions/             — create a new interview session
  POST /api/sessions/{id}/start   — generate the first question
  POST /api/sessions/{id}/answer  — submit answer -> get 5D eval + feedback + next Q/probe
  GET  /api/sessions/{id}         — get session details
  GET  /api/sessions/{id}/history — full Q&A history with evidentiary audit
  POST /api/sessions/{id}/finish  — complete and generate report
"""
from __future__ import annotations
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from loguru import logger

from backend.models.database import get_db, InterviewSession, QAPair, User
from backend.models.schemas import (
    SessionCreate, SessionResponse, AnswerSubmit,
    AnswerFeedbackResponse, NextQuestionResponse, EvaluationResult,
    SessionReport, QAPairResponse, RubricScore,
)
from backend.agents.interviewer_agent import InterviewerAgent
from backend.agents.evaluator_agent import EvaluatorAgent
from backend.agents.feedback_agent import FeedbackAgent
from backend.agents.report_agent import ReportAgent
from backend.utils.config import get_settings

settings = get_settings()
router = APIRouter()

_interviewer = InterviewerAgent()
_evaluator   = EvaluatorAgent()
_feedback    = FeedbackAgent()
_reporter    = ReportAgent()

# In-memory session tracking
_session_context: Dict[str, Dict[str, Any]] = {}


def _get_session_or_404(session_id: str, db: Session) -> InterviewSession:
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ─── Create session ───────────────────────────────────────────────────────────

@router.post("/", response_model=SessionResponse, status_code=201)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    """Create a new interview session."""
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
        job_description=payload.job_description,
        status="active",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    _session_context[session.id] = {
        "question_number": 0,
        "covered_topics": [],
        "resume_summary": "No resume uploaded.",
        "competency_gaps": [],
        "is_followup": False,
    }

    logger.info(f"AgentHire session initialized: {session.id} | {session.role} ({session.interview_type})")
    return session


# ─── Start Session ────────────────────────────────────────────────────────────

@router.post("/{session_id}/start", response_model=NextQuestionResponse)
def start_session(session_id: str, db: Session = Depends(get_db)):
    """Generate the opening Tier 1 question for a session."""
    session = _get_session_or_404(session_id, db)
    ctx = _session_context.setdefault(session_id, {
        "question_number": 0,
        "covered_topics": [],
        "resume_summary": "No resume uploaded.",
        "competency_gaps": [],
        "is_followup": False,
    })

    # Restore gaps from session if available
    gaps = ctx.get("competency_gaps", [])
    if not gaps and session.candidate_profile:
        try:
            profile_data = json.loads(session.candidate_profile)
            gaps = profile_data.get("detected_competency_gaps", [])
            ctx["competency_gaps"] = gaps
        except Exception:
            pass

    ctx["question_number"] = 1
    question, topic, tier, diff = _interviewer.generate_first_question(
        session_id=session_id,
        role=session.role,
        interview_type=session.interview_type,
        resume_summary=ctx["resume_summary"],
        max_questions=settings.max_questions,
        competency_gaps=gaps,
    )

    qa = QAPair(
        id=str(uuid.uuid4()),
        session_id=session_id,
        question_number=1,
        question=question,
        topic=topic,
        pedagogical_tier=tier,
        difficulty_lambda=diff,
        is_followup=False,
    )
    db.add(qa)
    db.commit()

    ctx["current_qa_id"] = qa.id
    ctx["current_question"] = question
    ctx["current_topic"] = topic
    ctx["current_tier"] = tier
    ctx["current_difficulty"] = diff
    ctx["is_followup"] = False

    return NextQuestionResponse(
        question=question,
        question_number=1,
        topic=topic,
        pedagogical_tier=tier,
        difficulty_lambda=diff,
        target_competency=topic,
        is_followup=False,
        is_final=(1 >= settings.max_questions),
    )


# ─── Submit Answer ────────────────────────────────────────────────────────────

@router.post("/{session_id}/answer", response_model=AnswerFeedbackResponse)
def submit_answer(
    session_id: str,
    payload: AnswerSubmit,
    db: Session = Depends(get_db),
):
    """
    Submit candidate answer -> execute 5D RAE evaluation with verbatim evidence gate ->
    generate DFP feedback -> evaluate adaptive probing branching condition.
    """
    session = _get_session_or_404(session_id, db)
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    ctx = _session_context.get(session_id)
    if not ctx:
        raise HTTPException(status_code=400, detail="Session not started — call /start first")

    q_num = ctx["question_number"]
    current_question = ctx.get("current_question", "")
    is_current_followup = ctx.get("is_followup", False)

    # 1. RAE-Agent Evidentiary Evaluation
    evaluation = _evaluator.evaluate(
        session_id=session_id,
        question=current_question,
        answer=payload.answer,
        role=session.role,
        interview_type=session.interview_type,
        is_followup=is_current_followup,
    )

    # 2. DFP-Agent Coaching Feedback
    feedback_text = _feedback.generate_feedback(
        session_id=session_id,
        question=current_question,
        answer=payload.answer,
        evaluation=evaluation,
        role=session.role,
        resume_summary=ctx["resume_summary"],
        topic=ctx.get("current_topic", ""),
    )

    # Serialize rubric scores for database
    rubric_dict = {}
    for dim, r in evaluation.get("rubric_scores", {}).items():
        if hasattr(r, "model_dump"):
            rubric_dict[dim] = r.model_dump()
        else:
            rubric_dict[dim] = r

    # 3. Persist answered question to DB
    qa_record = db.query(QAPair).filter(
        QAPair.session_id == session_id,
        QAPair.question_number == q_num,
        QAPair.is_followup == is_current_followup,
    ).first()

    if not qa_record:
        qa_record = db.query(QAPair).filter(
            QAPair.session_id == session_id,
            QAPair.question_number == q_num,
        ).order_by(QAPair.created_at.desc()).first()

    if qa_record:
        qa_record.answer = payload.answer
        qa_record.score = evaluation["score"]
        qa_record.strengths = json.dumps(evaluation.get("strengths", []))
        qa_record.weaknesses = json.dumps(evaluation.get("weaknesses", []))
        qa_record.feedback = feedback_text
        qa_record.ideal_answer_outline = evaluation.get("ideal_answer_outline", "")
        qa_record.rubric_breakdown = json.dumps(rubric_dict)
        db.commit()

    # Track topic coverage
    current_topic = ctx.get("current_topic", "")
    if current_topic and current_topic not in ctx["covered_topics"]:
        ctx["covered_topics"].append(current_topic)

    # 4. Check Adaptive Probing Branching (Section 4.3):
    # Condition: (c_1 >= 3) AND (c_4 <= 3) AND (not is_followup)
    requires_probing = evaluation.get("requires_probing", False) and not is_current_followup

    eval_response = EvaluationResult(
        score=evaluation["score"],
        composite_5=evaluation.get("composite_5", 3.0),
        strengths=evaluation.get("strengths", []),
        weaknesses=evaluation.get("weaknesses", []),
        ideal_answer_outline=evaluation.get("ideal_answer_outline", ""),
        follow_up_suggested=evaluation.get("follow_up_suggested", ""),
        rubric_scores=evaluation.get("rubric_scores", {}),
        evidence_grounding_verified=evaluation.get("evidence_grounding_verified", True),
        requires_probing=requires_probing,
        probing_focus=evaluation.get("probing_focus"),
        is_followup=is_current_followup,
    )

    next_q_response = None
    probe_triggered = False
    probe_expl = None

    if requires_probing:
        # Branch to AIS-Agent adaptive follow-up probe!
        probe_triggered = True
        probe_expl = (
            "Adaptive Probing Condition Activated: Candidate demonstrated technical accuracy (c_1 >= 3) "
            "but solution completeness was partial (c_4 <= 3). Probing knowledge boundary."
        )
        logger.info(f"[Branching] Triggering adaptive probe for session {session_id} Q{q_num}")

        probe_question, probe_topic, probe_tier, probe_diff = _interviewer.synthesize_followup_probe(
            question=current_question,
            candidate_answer=payload.answer,
            evaluation=evaluation,
            role=session.role,
        )

        probe_qa = QAPair(
            id=str(uuid.uuid4()),
            session_id=session_id,
            question_number=q_num,  # Attached to same parent question turn
            question=probe_question,
            topic=probe_topic,
            pedagogical_tier=probe_tier,
            difficulty_lambda=probe_diff,
            is_followup=True,
            parent_question_id=qa_record.id if qa_record else None,
        )
        db.add(probe_qa)
        db.commit()

        ctx["current_question"] = probe_question
        ctx["current_topic"] = probe_topic
        ctx["current_tier"] = probe_tier
        ctx["current_difficulty"] = probe_diff
        ctx["current_qa_id"] = probe_qa.id
        ctx["is_followup"] = True

        next_q_response = NextQuestionResponse(
            question=probe_question,
            question_number=q_num,
            topic=probe_topic,
            pedagogical_tier=probe_tier,
            difficulty_lambda=probe_diff,
            target_competency=probe_topic,
            is_followup=True,
            is_final=False,
        )
        session_complete = False

    else:
        # Advance to next core question item
        next_q_num = q_num + 1
        session_complete = next_q_num > settings.max_questions
        ctx["is_followup"] = False

        if not session_complete:
            is_final = next_q_num >= settings.max_questions
            gaps = ctx.get("competency_gaps", [])

            next_question, next_topic, next_tier, next_diff = _interviewer.generate_next_question(
                session_id=session_id,
                role=session.role,
                interview_type=session.interview_type,
                resume_summary=ctx["resume_summary"],
                last_answer=payload.answer,
                question_number=next_q_num,
                max_questions=settings.max_questions,
                covered_topics=ctx["covered_topics"],
                competency_gaps=gaps,
                is_final=is_final,
            )

            next_qa = QAPair(
                id=str(uuid.uuid4()),
                session_id=session_id,
                question_number=next_q_num,
                question=next_question,
                topic=next_topic,
                pedagogical_tier=next_tier,
                difficulty_lambda=next_diff,
                is_followup=False,
            )
            db.add(next_qa)
            db.commit()

            ctx["question_number"] = next_q_num
            ctx["current_question"] = next_question
            ctx["current_topic"] = next_topic
            ctx["current_tier"] = next_tier
            ctx["current_difficulty"] = next_diff
            ctx["current_qa_id"] = next_qa.id

            next_q_response = NextQuestionResponse(
                question=next_question,
                question_number=next_q_num,
                topic=next_topic,
                pedagogical_tier=next_tier,
                difficulty_lambda=next_diff,
                target_competency=next_topic,
                is_followup=False,
                is_final=is_final,
            )
        else:
            session.status = "completed"
            session.completed_at = datetime.utcnow()
            db.commit()

    return AnswerFeedbackResponse(
        evaluation=eval_response,
        feedback=feedback_text,
        next_question=next_q_response,
        session_complete=session_complete,
        adaptive_probe_triggered=probe_triggered,
        probe_explanation=probe_expl,
    )


# ─── Get Session ──────────────────────────────────────────────────────────────

@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    return _get_session_or_404(session_id, db)


# ─── Q&A History ──────────────────────────────────────────────────────────────

@router.get("/{session_id}/history", response_model=List[QAPairResponse])
def get_history(session_id: str, db: Session = Depends(get_db)):
    _get_session_or_404(session_id, db)
    pairs = (
        db.query(QAPair)
        .filter(QAPair.session_id == session_id)
        .order_by(QAPair.question_number, QAPair.is_followup)
        .all()
    )
    result = []
    for p in pairs:
        rubric_parsed = None
        if p.rubric_breakdown:
            try:
                raw_rubric = json.loads(p.rubric_breakdown)
                rubric_parsed = {k: RubricScore(**v) for k, v in raw_rubric.items()}
            except Exception:
                pass

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
            rubric_scores=rubric_parsed,
            pedagogical_tier=p.pedagogical_tier,
            difficulty_lambda=p.difficulty_lambda,
            is_followup=bool(p.is_followup),
        ))
    return result


# ─── Force Finish ─────────────────────────────────────────────────────────────

@router.post("/{session_id}/finish")
def finish_session(session_id: str, db: Session = Depends(get_db)):
    """Manually complete session and trigger report generation."""
    session = _get_session_or_404(session_id, db)

    pairs = (
        db.query(QAPair)
        .filter(QAPair.session_id == session_id, QAPair.answer.isnot(None))
        .order_by(QAPair.question_number, QAPair.is_followup)
        .all()
    )

    qa_dicts = []
    for p in pairs:
        rubric_data = None
        if p.rubric_breakdown:
            try:
                rubric_data = json.loads(p.rubric_breakdown)
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
            "rubric_scores": rubric_data,
            "pedagogical_tier": p.pedagogical_tier,
            "difficulty_lambda": p.difficulty_lambda,
            "is_followup": p.is_followup,
        })

    user = db.query(User).filter(User.id == session.user_id).first()
    candidate_name = user.name if user else "Candidate"

    summary, pdf_path, avg_score, radar, curriculum = _reporter.compile_report(
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
        "rubric_radar": radar,
        "curriculum": curriculum,
    }

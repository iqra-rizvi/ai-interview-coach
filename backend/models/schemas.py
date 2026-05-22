"""
backend/models/schemas.py
Pydantic v2 schemas for request validation and API responses.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


# ─── User ────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    name: str
    email: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Session ─────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    user_id: str
    role: str = Field(..., examples=["Software Engineer", "Data Scientist", "Product Manager"])
    interview_type: str = Field(..., pattern="^(technical|hr|mixed)$")


class SessionResponse(BaseModel):
    id: str
    user_id: str
    role: str
    interview_type: str
    status: str
    avg_score: Optional[float]
    resume_filename: Optional[str]
    report_path: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}


# ─── Q&A ─────────────────────────────────────────────────────────────────────

class AnswerSubmit(BaseModel):
    session_id: str
    question_number: int
    answer: str = Field(..., min_length=1, max_length=5000)


class EvaluationResult(BaseModel):
    score: float
    strengths: List[str]
    weaknesses: List[str]
    ideal_answer_outline: str
    follow_up_suggested: str = ""


class QAPairResponse(BaseModel):
    id: str
    question_number: int
    question: str
    answer: Optional[str]
    score: Optional[float]
    strengths: Optional[List[str]]
    weaknesses: Optional[List[str]]
    feedback: Optional[str]
    ideal_answer_outline: Optional[str]
    topic: Optional[str]

    model_config = {"from_attributes": True}


# ─── Interview flow ───────────────────────────────────────────────────────────

class NextQuestionResponse(BaseModel):
    question: str
    question_number: int
    topic: Optional[str]
    is_final: bool


class AnswerFeedbackResponse(BaseModel):
    evaluation: EvaluationResult
    feedback: str
    next_question: Optional[NextQuestionResponse]
    session_complete: bool


# ─── Report ──────────────────────────────────────────────────────────────────

class SessionReport(BaseModel):
    session_id: str
    candidate_name: str
    role: str
    interview_type: str
    avg_score: float
    total_questions: int
    executive_summary: str
    qa_pairs: List[QAPairResponse]
    report_pdf_url: Optional[str]


# ─── Resume upload ────────────────────────────────────────────────────────────

class ResumeUploadResponse(BaseModel):
    session_id: str
    filename: str
    chroma_collection_id: str
    summary: str
    message: str

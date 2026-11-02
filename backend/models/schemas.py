"""
backend/models/schemas.py
Pydantic v2 schemas for request validation, API responses, and
LangGraph state contracts (as specified in AgentHire Section 10 Listing A.1).
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Dict, Literal, Any
from pydantic import BaseModel, Field, field_validator


# ─── User schemas ────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    name: str
    email: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── AgentHire Section 10: Multi-Agent State Contracts ───────────────────────

class CandidateProfile(BaseModel):
    candidate_id: str = Field(description="Unique anonymized candidate identifier")
    technical_skills: List[str] = Field(default_factory=list, description="Extracted programming languages, frameworks, toolsets")
    experience_years: float = Field(default=0.0, description="Total normalized professional experience in years")
    verified_projects: List[Dict[str, str]] = Field(default_factory=list, description="Project titles, descriptions, and architectural stacks")
    detected_competency_gaps: List[str] = Field(default_factory=list, description="Core job requirements missing from candidate resume")
    sanitized_cv_text: Optional[str] = Field(default=None, description="PII-sanitized text of the candidate CV")


class QuestionItem(BaseModel):
    question_id: str = Field(description="Unique question identifier")
    target_competency: str = Field(description="Primary technical competency evaluated by this item")
    pedagogical_tier: Literal[
        "Tier 1: Foundational",
        "Tier 2: Systems Architecture",
        "Tier 3: Situational STAR",
    ] = Field(default="Tier 1: Foundational")
    difficulty_lambda: int = Field(ge=1, le=5, default=3, description="Difficulty calibration index from 1 to 5")
    question_text: str = Field(description="Prompt text presented to candidate")
    is_followup: bool = Field(default=False, description="Flag indicating whether item is an adaptive probe")
    parent_question_id: Optional[str] = Field(default=None, description="Reference to root question if followup")


class RubricScore(BaseModel):
    dimension_name: str = Field(description="One of the five orthogonal evaluation criteria")
    score: int = Field(ge=1, le=5, description="Integer rating between 1 and 5")
    verbatim_evidence_quote: str = Field(description="Exact verbatim substring extracted from candidate transcript")
    evaluator_rationale: str = Field(description="Structured justification explaining score based on quote")
    evidence_verified: bool = Field(default=True, description="Whether verbatim evidence passed the verification gate")

    @field_validator("verbatim_evidence_quote")
    def validate_non_empty_quote(cls, v: str) -> str:
        cleaned = v.strip() if v else ""
        # If quote is empty, provide a standard ungrounded token
        if not cleaned:
            return "[NO_VERBATIM_EVIDENCE_PROVIDED]"
        return cleaned


class EvaluationRecord(BaseModel):
    question_id: str
    scores: Dict[str, RubricScore]
    composite_score: float = Field(ge=0.0, le=5.0)
    requires_probing: bool = Field(default=False)
    probing_focus: Optional[str] = None
    evidence_grounded: bool = Field(default=True)


class SessionStateSchema(BaseModel):
    session_id: str
    job_role_title: str
    job_requirements: List[str] = Field(default_factory=list)
    candidate_profile: Optional[CandidateProfile] = None
    competency_divergence: Dict[str, float] = Field(default_factory=dict)
    divergence_score: float = Field(default=0.0, description="Overall divergence Delta(C, J) in [0, 1]")
    question_battery: List[QuestionItem] = Field(default_factory=list)
    current_turn_index: int = 0
    transcript_history: List[Dict[str, str]] = Field(default_factory=list)
    evaluation_records: Dict[str, EvaluationRecord] = Field(default_factory=dict)
    diagnostic_report: Optional[Dict[str, Any]] = None


# ─── API Session schemas ─────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    user_id: str
    role: str = Field(..., examples=["Software Engineer", "Data Scientist", "Product Manager"])
    interview_type: str = Field(..., pattern="^(technical|hr|mixed)$")
    job_description: Optional[str] = None


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


# ─── Q&A and Evaluation schemas ──────────────────────────────────────────────

class AnswerSubmit(BaseModel):
    session_id: str
    question_number: int
    answer: str = Field(..., min_length=1, max_length=5000)


class EvaluationResult(BaseModel):
    score: float = Field(description="Composite score scaled to 0-10 for display")
    composite_5: float = Field(default=3.0, description="Composite score on 1-5 scale")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    ideal_answer_outline: str = ""
    follow_up_suggested: str = ""
    rubric_scores: Dict[str, RubricScore] = Field(default_factory=dict)
    evidence_grounding_verified: bool = True
    requires_probing: bool = False
    probing_focus: Optional[str] = None
    is_followup: bool = False


class QAPairResponse(BaseModel):
    id: str
    question_number: int
    question: str
    answer: Optional[str]
    score: Optional[float]
    strengths: Optional[List[str]] = None
    weaknesses: Optional[List[str]] = None
    feedback: Optional[str] = None
    ideal_answer_outline: Optional[str] = None
    topic: Optional[str] = None
    rubric_scores: Optional[Dict[str, RubricScore]] = None
    pedagogical_tier: Optional[str] = None
    difficulty_lambda: Optional[int] = None
    is_followup: Optional[bool] = False

    model_config = {"from_attributes": True}


# ─── Interview Flow schemas ──────────────────────────────────────────────────

class NextQuestionResponse(BaseModel):
    question: str
    question_number: int
    topic: Optional[str] = None
    pedagogical_tier: Optional[str] = "Tier 1: Foundational"
    difficulty_lambda: Optional[int] = 3
    target_competency: Optional[str] = None
    is_followup: bool = False
    is_final: bool = False


class AnswerFeedbackResponse(BaseModel):
    evaluation: EvaluationResult
    feedback: str
    next_question: Optional[NextQuestionResponse] = None
    session_complete: bool
    adaptive_probe_triggered: bool = False
    probe_explanation: Optional[str] = None


# ─── Report schemas ──────────────────────────────────────────────────────────

class SessionReport(BaseModel):
    session_id: str
    candidate_name: str
    role: str
    interview_type: str
    avg_score: float
    total_questions: int
    executive_summary: str
    qa_pairs: List[QAPairResponse]
    report_pdf_url: Optional[str] = None
    rubric_radar: Optional[Dict[str, float]] = None
    competency_divergence: Optional[float] = None
    detected_competency_gaps: Optional[List[str]] = None
    remediation_plan: Optional[List[Dict[str, str]]] = None


# ─── Resume Upload schemas ───────────────────────────────────────────────────

class ResumeUploadResponse(BaseModel):
    session_id: str
    filename: str
    chroma_collection_id: str
    summary: str
    message: str
    extracted_skills: List[str] = Field(default_factory=list)
    competency_divergence: float = Field(default=0.0)
    detected_gaps: List[str] = Field(default_factory=list)
    pii_sanitized: bool = True

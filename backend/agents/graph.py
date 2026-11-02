"""
backend/agents/graph.py
LangGraph stateful multi-agent orchestration for AgentHire.
Implements the 5-Agent Architecture and Conditional Probing Loop
as specified in AgentHire Chapter 4 and Appendix 10 Listing A.2.
"""
from __future__ import annotations
import operator
from typing import TypedDict, List, Dict, Optional, Annotated, Any
from loguru import logger

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.models.schemas import SessionStateSchema, RubricScore, EvaluationRecord, QuestionItem, CandidateProfile
from backend.agents.interviewer_agent import InterviewerAgent
from backend.agents.evaluator_agent import EvaluatorAgent
from backend.agents.feedback_agent import FeedbackAgent
from backend.agents.report_agent import ReportAgent
from backend.utils.config import get_settings

settings = get_settings()

_interviewer = InterviewerAgent()
_evaluator = EvaluatorAgent()
_feedback = FeedbackAgent()
_reporter = ReportAgent()


# ─── Shared state schema for interactive sessions ────────────────────────────

class QARecord(TypedDict):
    question_number: int
    question: str
    answer: str
    topic: str
    score: float
    strengths: List[str]
    weaknesses: List[str]
    ideal_answer_outline: str
    feedback: str
    rubric_scores: Optional[Dict[str, Any]]
    pedagogical_tier: Optional[str]
    difficulty_lambda: Optional[int]
    is_followup: Optional[bool]


class InterviewState(TypedDict):
    session_id: str
    user_id: str
    candidate_name: str
    role: str
    interview_type: str
    max_questions: int
    resume_summary: str
    competency_gaps: List[str]
    competency_divergence: float

    # Progress tracking
    question_number: int
    covered_topics: List[str]
    qa_history: Annotated[List[QARecord], operator.add]

    # Current turn state
    current_question: str
    current_topic: str
    current_tier: str
    current_difficulty: int
    current_answer: str
    is_followup: bool

    # Flow control
    session_complete: bool
    requires_probing: bool
    probing_focus: Optional[str]

    # Outputs
    executive_summary: str
    avg_score: float
    report_pdf_path: str
    radar_scores: Dict[str, float]


# ─── Node functions ───────────────────────────────────────────────────────────

def generate_question(state: InterviewState) -> dict:
    """DQG-Agent / AIS-Agent: Synthesize next question or adaptive probe."""
    q_num = state.get("question_number", 1)
    is_final = q_num >= state.get("max_questions", 10)
    gaps = state.get("competency_gaps", [])

    if q_num == 1:
        question, topic, tier, diff = _interviewer.generate_first_question(
            session_id=state["session_id"],
            role=state["role"],
            interview_type=state["interview_type"],
            resume_summary=state.get("resume_summary", ""),
            max_questions=state.get("max_questions", 10),
            competency_gaps=gaps,
        )
    else:
        last_qa = state["qa_history"][-1] if state.get("qa_history") else {}
        question, topic, tier, diff = _interviewer.generate_next_question(
            session_id=state["session_id"],
            role=state["role"],
            interview_type=state["interview_type"],
            resume_summary=state.get("resume_summary", ""),
            last_answer=last_qa.get("answer", ""),
            question_number=q_num,
            max_questions=state.get("max_questions", 10),
            covered_topics=state.get("covered_topics", []),
            competency_gaps=gaps,
            is_final=is_final,
        )

    logger.info(f"[Graph] Q{q_num} generated ({tier}): {question[:60]}...")
    return {
        "current_question": question,
        "current_topic": topic,
        "current_tier": tier,
        "current_difficulty": diff,
        "is_followup": False,
    }


def evaluate_answer(state: InterviewState) -> dict:
    """RAE-Agent: 5-dimensional rubric scoring with Verbatim Evidence Verification Gate."""
    eval_result = _evaluator.evaluate(
        session_id=state["session_id"],
        question=state["current_question"],
        answer=state["current_answer"],
        role=state["role"],
        interview_type=state["interview_type"],
        is_followup=state.get("is_followup", False),
    )
    return {"_eval_result": eval_result}


def generate_feedback_and_record(state: InterviewState) -> dict:
    """DFP-Agent: Generate feedback and commit evaluated turn to history."""
    eval_result = state.get("_eval_result", {})

    feedback = _feedback.generate_feedback(
        session_id=state["session_id"],
        question=state["current_question"],
        answer=state["current_answer"],
        evaluation=eval_result,
        role=state["role"],
        resume_summary=state.get("resume_summary", ""),
        topic=state.get("current_topic", ""),
    )

    record: QARecord = {
        "question_number": state["question_number"],
        "question": state["current_question"],
        "answer": state["current_answer"],
        "topic": state["current_topic"],
        "score": eval_result.get("score", 0.0),
        "strengths": eval_result.get("strengths", []),
        "weaknesses": eval_result.get("weaknesses", []),
        "ideal_answer_outline": eval_result.get("ideal_answer_outline", ""),
        "feedback": feedback,
        "rubric_scores": eval_result.get("rubric_scores", {}),
        "pedagogical_tier": state.get("current_tier", "Tier 1: Foundational"),
        "difficulty_lambda": state.get("current_difficulty", 3),
        "is_followup": state.get("is_followup", False),
    }

    new_topics = state.get("covered_topics", []) + [state.get("current_topic", "")]
    return {
        "qa_history": [record],
        "covered_topics": new_topics,
        "_eval_result": None,
        "requires_probing": eval_result.get("requires_probing", False),
        "probing_focus": eval_result.get("probing_focus"),
    }


def probe_followup_node(state: InterviewState) -> dict:
    """AIS-Agent: Synthesize targeted adaptive boundary probe."""
    last_qa = state["qa_history"][-1] if state.get("qa_history") else {}
    eval_record = {"probing_focus": state.get("probing_focus", ""), "ideal_answer_outline": last_qa.get("ideal_answer_outline", "")}

    probe_question, topic, tier, diff = _interviewer.synthesize_followup_probe(
        question=last_qa.get("question", ""),
        candidate_answer=last_qa.get("answer", ""),
        evaluation=eval_record,
        role=state["role"],
    )

    logger.info(f"[Graph: Adaptive Probe] Triggered follow-up: {probe_question[:60]}...")
    return {
        "current_question": probe_question,
        "current_topic": topic,
        "current_tier": tier,
        "current_difficulty": diff,
        "is_followup": True,
        "requires_probing": False,
    }


def should_probe_or_advance(state: InterviewState) -> str:
    """
    Conditional edge transition for adaptive probing (Section 4.3):
    Condition: (c_1 >= 3) AND (c_4 <= 3) AND (not is_followup)
    """
    if state.get("requires_probing", False) and not state.get("is_followup", False):
        return "probe_followup_node"

    next_q = state["question_number"] + 1
    if next_q > state.get("max_questions", 10):
        return "compile_report"
    return "advance_turn"


def increment_question(state: InterviewState) -> dict:
    return {"question_number": state["question_number"] + 1}


def compile_report(state: InterviewState) -> dict:
    """DFP-Agent: Generate comprehensive PDF report with radar breakdown."""
    summary, pdf_path, avg_score, radar, curriculum = _reporter.compile_report(
        session_id=state["session_id"],
        candidate_name=state.get("candidate_name", "Candidate"),
        role=state["role"],
        interview_type=state["interview_type"],
        qa_pairs=state.get("qa_history", []),
    )
    return {
        "executive_summary": summary,
        "report_pdf_path": pdf_path,
        "avg_score": avg_score,
        "radar_scores": radar,
        "session_complete": True,
    }


# ─── Graph assembly ───────────────────────────────────────────────────────────

def build_agenthire_graph() -> Any:
    """Build and compile the full AgentHire stateful computation graph."""
    graph = StateGraph(InterviewState)

    graph.add_node("generate_question", generate_question)
    graph.add_node("evaluate_answer", evaluate_answer)
    graph.add_node("generate_feedback", generate_feedback_and_record)
    graph.add_node("probe_followup_node", probe_followup_node)
    graph.add_node("increment_question", increment_question)
    graph.add_node("compile_report", compile_report)

    graph.set_entry_point("generate_question")
    graph.add_edge("generate_question", "evaluate_answer")
    graph.add_edge("evaluate_answer", "generate_feedback")

    graph.add_conditional_edges(
        "generate_feedback",
        should_probe_or_advance,
        {
            "probe_followup_node": "probe_followup_node",
            "advance_turn": "increment_question",
            "compile_report": "compile_report",
        },
    )

    graph.add_edge("probe_followup_node", "evaluate_answer")
    graph.add_edge("increment_question", "generate_question")
    graph.add_edge("compile_report", END)

    memory = MemorySaver()
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["evaluate_answer"],
    )


interview_graph = build_agenthire_graph()

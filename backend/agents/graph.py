"""
backend/agents/graph.py
LangGraph stateful multi-agent orchestration.

Graph flow:
  START
    │
    ▼
  [generate_question]  ←──────────────────────┐
    │                                          │
    ▼                                          │
  [await_answer]  (human-in-the-loop)         │
    │                                          │
    ▼                                          │
  [evaluate_answer]                            │
    │                                          │
    ▼                                          │
  [generate_feedback]                          │
    │                                          │
    ▼                                          │
  [decide_next] ── more questions? ────────────┘
    │
    └── session complete ──▶ [compile_report] ──▶ END
"""
from __future__ import annotations
from typing import TypedDict, List, Optional, Annotated
import operator
from loguru import logger

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.agents.interviewer_agent import InterviewerAgent
from backend.agents.evaluator_agent import EvaluatorAgent
from backend.agents.feedback_agent import FeedbackAgent
from backend.agents.report_agent import ReportAgent
from backend.utils.config import get_settings

settings = get_settings()


# ─── Shared state schema ─────────────────────────────────────────────────────

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


class InterviewState(TypedDict):
    # Session config
    session_id: str
    user_id: str
    candidate_name: str
    role: str
    interview_type: str
    max_questions: int
    resume_summary: str

    # Progress tracking
    question_number: int
    covered_topics: List[str]
    qa_history: Annotated[List[QARecord], operator.add]  # Append-only

    # Current turn
    current_question: str
    current_topic: str
    current_answer: str

    # Flow control
    session_complete: bool

    # Outputs
    executive_summary: str
    avg_score: float
    report_pdf_path: str


# ─── Agent singletons ─────────────────────────────────────────────────────────

_interviewer = InterviewerAgent()
_evaluator = EvaluatorAgent()
_feedback = FeedbackAgent()
_reporter = ReportAgent()


# ─── Node functions ───────────────────────────────────────────────────────────

def generate_question(state: InterviewState) -> dict:
    """InterviewerAgent generates the next question."""
    q_num = state["question_number"]
    is_final = q_num >= state["max_questions"]

    if q_num == 1:
        question, topic = _interviewer.generate_first_question(
            session_id=state["session_id"],
            role=state["role"],
            interview_type=state["interview_type"],
            resume_summary=state["resume_summary"],
            max_questions=state["max_questions"],
        )
    else:
        last_qa = state["qa_history"][-1] if state["qa_history"] else {}
        question, topic = _interviewer.generate_next_question(
            session_id=state["session_id"],
            role=state["role"],
            interview_type=state["interview_type"],
            resume_summary=state["resume_summary"],
            last_answer=last_qa.get("answer", ""),
            question_number=q_num,
            max_questions=state["max_questions"],
            covered_topics=state["covered_topics"],
            is_final=is_final,
        )

    logger.info(f"[Graph] Q{q_num} generated: {question[:60]}...")
    return {
        "current_question": question,
        "current_topic": topic,
    }


def evaluate_answer(state: InterviewState) -> dict:
    """EvaluatorAgent scores the current answer."""
    evaluation = _evaluator.evaluate(
        session_id=state["session_id"],
        question=state["current_question"],
        answer=state["current_answer"],
        role=state["role"],
        interview_type=state["interview_type"],
    )
    return {"_eval_result": evaluation}  # Temp key — consumed by feedback node


def generate_feedback_and_record(state: InterviewState) -> dict:
    """FeedbackAgent generates coaching text, then records the full QA."""
    # Retrieve eval from previous node via state (LangGraph passes full state)
    eval_result = state.get("_eval_result", {})

    feedback = _feedback.generate_feedback(
        session_id=state["session_id"],
        question=state["current_question"],
        answer=state["current_answer"],
        evaluation=eval_result,
        role=state["role"],
        resume_summary=state["resume_summary"],
        topic=state["current_topic"],
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
    }

    new_topics = state["covered_topics"] + [state["current_topic"]]
    return {
        "qa_history": [record],
        "covered_topics": new_topics,
        "_eval_result": None,  # Clear temp key
    }


def decide_next(state: InterviewState) -> str:
    """Router: continue asking or compile final report."""
    next_q = state["question_number"] + 1
    if next_q > state["max_questions"]:
        return "compile_report"
    return "generate_question"


def compile_report(state: InterviewState) -> dict:
    """ReportAgent compiles the full PDF report."""
    summary, pdf_path, avg_score = _reporter.compile_report(
        session_id=state["session_id"],
        candidate_name=state["candidate_name"],
        role=state["role"],
        interview_type=state["interview_type"],
        qa_pairs=state["qa_history"],
    )
    return {
        "executive_summary": summary,
        "report_pdf_path": pdf_path,
        "avg_score": avg_score,
        "session_complete": True,
    }


def increment_question(state: InterviewState) -> dict:
    """Bump the question counter after processing each answer."""
    return {"question_number": state["question_number"] + 1}


# ─── Graph assembly ───────────────────────────────────────────────────────────

def build_interview_graph() -> StateGraph:
    """Build and compile the LangGraph interview workflow."""
    graph = StateGraph(InterviewState)

    # Add nodes
    graph.add_node("generate_question", generate_question)
    graph.add_node("evaluate_answer", evaluate_answer)
    graph.add_node("generate_feedback", generate_feedback_and_record)
    graph.add_node("increment_question", increment_question)
    graph.add_node("compile_report", compile_report)

    # Entry
    graph.set_entry_point("generate_question")

    # generate_question → (await human answer — handled externally) → evaluate_answer
    # The "await_answer" step is handled by the API layer: when the user submits
    # an answer, we update the state and resume from evaluate_answer.
    graph.add_edge("generate_question", "evaluate_answer")
    graph.add_edge("evaluate_answer", "generate_feedback")
    graph.add_edge("generate_feedback", "increment_question")

    # Conditional: more questions or finish?
    graph.add_conditional_edges(
        "increment_question",
        decide_next,
        {
            "generate_question": "generate_question",
            "compile_report": "compile_report",
        },
    )
    graph.add_edge("compile_report", END)

    # Compile with in-memory checkpointing (swap MemorySaver for RedisCheckpointer in prod)
    memory = MemorySaver()
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["evaluate_answer"],  # Pause here to wait for human answer
    )


# Module-level compiled graph instance
interview_graph = build_interview_graph()

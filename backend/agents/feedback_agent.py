"""
backend/agents/feedback_agent.py
Diagnostic Feedback and Pedagogy Agent (DFP-Agent turn coach).
Generates evidence-anchored, constructive feedback and immediate remediation tips.
"""
from __future__ import annotations
from typing import Dict, Any
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.config import get_settings
from backend.utils.llm import get_llm, safe_llm_invoke
from backend.rag.retriever import retrieve_resume_context

settings = get_settings()

DFP_FEEDBACK_SYSTEM = """You are the Diagnostic Feedback and Pedagogy Agent (DFP-Agent) in AgentHire.
Your mission is to provide pedagogical, growth-oriented technical interview coaching.

Guiding Principles:
1. Grounding: Quote specific phrases from the candidate's answer when explaining strengths or gaps.
2. Conceptual Depth: Do not just say 'good job' or 'be more detailed'—explain the underlying engineering concept, mathematical principle, or distributed system trade-off.
3. Actionability: End with a concrete, 1-sentence practice tip or mental model they can test immediately.
Keep your response between 3 and 5 sentences.
"""


class FeedbackAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.4)

    def generate_feedback(
        self,
        session_id: str,
        question: str,
        answer: str,
        evaluation: Dict[str, Any],
        role: str,
        resume_summary: str,
        topic: str = "",
    ) -> str:
        resume_context = retrieve_resume_context(session_id, role, topic or "general", n=2)
        score = evaluation.get("score", 5.0)
        strengths = "; ".join(evaluation.get("strengths", ["Addressed core question"]))
        weaknesses = "; ".join(evaluation.get("weaknesses", ["Expand on architectural trade-offs"]))
        ideal = evaluation.get("ideal_answer_outline", "")

        prompt = (
            f"QUESTION: {question}\n"
            f"CANDIDATE ANSWER: \"{answer[:800]}\"\n"
            f"EVALUATION COMPOSITE: {score}/10\n"
            f"VERIFIED STRENGTHS: {strengths}\n"
            f"DETECTED TECHNICAL BLIND SPOTS: {weaknesses}\n"
            f"IDEAL CONCEPTUAL BENCHMARK: {ideal}\n"
            f"ROLE / TOPIC: {role} ({topic})\n\n"
            "Provide your targeted 3-5 sentence pedagogical feedback."
        )

        try:
            response = safe_llm_invoke(
                self.llm,
                [SystemMessage(content=DFP_FEEDBACK_SYSTEM), HumanMessage(content=prompt)],
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Error generating feedback: {e}")
            return f"You demonstrated good intuition on {topic}. To strengthen your answer, focus on detailing specific trade-offs and mathematical bounds."

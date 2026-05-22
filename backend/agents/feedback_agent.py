from __future__ import annotations
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from backend.utils.config import get_settings
from backend.utils.prompt_templates import FEEDBACK_SYSTEM, FEEDBACK_PROMPT
from backend.rag.retriever import retrieve_resume_context

settings = get_settings()


class FeedbackAgent:
    def __init__(self):
        self.llm = ChatGroq(model=settings.llm_model, temperature=0.6, groq_api_key=settings.groq_api_key)

    def generate_feedback(self, session_id, question, answer, evaluation, role, resume_summary, topic=""):
        resume_context = retrieve_resume_context(session_id, role, topic or "general", n=3)
        prompt = FEEDBACK_PROMPT.format(
            question=question, answer=answer[:1000],
            score=evaluation.get("score", 5),
            strengths="; ".join(evaluation.get("strengths", [])),
            weaknesses="; ".join(evaluation.get("weaknesses", [])),
            ideal_answer_outline=evaluation.get("ideal_answer_outline", ""),
            resume_summary=resume_summary[:400], role=role,
        )
        response = self.llm.invoke([SystemMessage(content=FEEDBACK_SYSTEM), HumanMessage(content=prompt)])
        return response.content.strip()

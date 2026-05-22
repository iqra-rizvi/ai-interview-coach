from __future__ import annotations
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from backend.utils.config import get_settings
from backend.utils.prompt_templates import (
    INTERVIEWER_SYSTEM, INTERVIEWER_FIRST_QUESTION,
    INTERVIEWER_NEXT_QUESTION, INTERVIEWER_CLOSING,
    TECHNICAL_TOPICS, HR_TOPICS,
)
from backend.rag.retriever import retrieve_for_question_generation

settings = get_settings()


class InterviewerAgent:
    def __init__(self):
        self.llm = ChatGroq(model=settings.llm_model, temperature=0.8, groq_api_key=settings.groq_api_key)

    def _get_next_topic(self, role, interview_type, question_number, covered_topics):
        if interview_type == "hr":
            pool = HR_TOPICS
        elif interview_type == "technical":
            pool = TECHNICAL_TOPICS.get(role, TECHNICAL_TOPICS["Software Engineer"])
        else:
            tech = TECHNICAL_TOPICS.get(role, TECHNICAL_TOPICS["Software Engineer"])
            pool = tech if question_number % 2 == 0 else HR_TOPICS
        uncovered = [t for t in pool if t not in covered_topics]
        return uncovered[0] if uncovered else pool[question_number % len(pool)]

    def generate_first_question(self, session_id, role, interview_type, resume_summary, max_questions):
        topic = "tell me about yourself"
        rag_context = retrieve_for_question_generation(session_id, role, topic, interview_type)
        system = INTERVIEWER_SYSTEM.format(
            role=role, interview_type=interview_type, question_number=1,
            max_questions=max_questions, resume_summary=resume_summary, covered_topics="none yet",
        )
        response = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=INTERVIEWER_FIRST_QUESTION)])
        return response.content.strip(), topic

    def generate_next_question(self, session_id, role, interview_type, resume_summary,
                                last_answer, question_number, max_questions, covered_topics, is_final=False):
        topic = self._get_next_topic(role, interview_type, question_number, covered_topics)
        rag_context = retrieve_for_question_generation(session_id, role, topic, interview_type)
        system = INTERVIEWER_SYSTEM.format(
            role=role, interview_type=interview_type, question_number=question_number,
            max_questions=max_questions, resume_summary=resume_summary,
            covered_topics=", ".join(covered_topics) or "none yet",
        )
        user_prompt = INTERVIEWER_CLOSING if is_final else INTERVIEWER_NEXT_QUESTION.format(
            last_answer=last_answer[:800], rag_context=rag_context[:600],
        )
        response = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user_prompt)])
        return response.content.strip(), topic

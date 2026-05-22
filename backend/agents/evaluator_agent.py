from __future__ import annotations
import json, re
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from backend.utils.config import get_settings
from backend.utils.prompt_templates import EVALUATOR_SYSTEM, EVALUATOR_PROMPT
from backend.rag.retriever import retrieve_for_evaluation

settings = get_settings()

_DEFAULT = {
    "score": 5.0, "strengths": ["Answer provided"],
    "weaknesses": ["Could not parse evaluation"],
    "ideal_answer_outline": "A strong answer addresses the question with concrete examples.",
    "follow_up_suggested": "",
}

def _parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return _DEFAULT


class EvaluatorAgent:
    def __init__(self):
        self.llm = ChatGroq(model=settings.llm_model, temperature=0.2, groq_api_key=settings.groq_api_key)

    def evaluate(self, session_id, question, answer, role, interview_type):
        if not answer or len(answer.strip()) < 5:
            return {**_DEFAULT, "score": 0, "weaknesses": ["No meaningful answer provided"]}
        rag_context = retrieve_for_evaluation(session_id, question, role)
        prompt = EVALUATOR_PROMPT.format(
            question=question, answer=answer[:1500],
            interview_type=interview_type, role=role, rag_context=rag_context[:600],
        )
        response = self.llm.invoke([SystemMessage(content=EVALUATOR_SYSTEM), HumanMessage(content=prompt)])
        result = _parse_json(response.content)
        result["score"] = max(0.0, min(10.0, float(result.get("score", 5.0))))
        logger.info(f"[Evaluator] Score: {result['score']}/10")
        return result

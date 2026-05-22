from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from backend.utils.config import get_settings
from backend.utils.prompt_templates import REPORT_SUMMARY_PROMPT
from backend.utils.pdf_generator import build_pdf_report

settings = get_settings()


class ReportAgent:
    def __init__(self):
        self.llm = ChatGroq(model=settings.llm_model, temperature=0.4, groq_api_key=settings.groq_api_key)

    def generate_executive_summary(self, candidate_name, role, interview_type, qa_pairs):
        scores = [qa["score"] for qa in qa_pairs if qa.get("score") is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
        all_strengths, all_weaknesses = [], []
        for qa in qa_pairs:
            all_strengths.extend(qa.get("strengths") or [])
            all_weaknesses.extend(qa.get("weaknesses") or [])
        prompt = REPORT_SUMMARY_PROMPT.format(
            candidate_name=candidate_name, role=role, interview_type=interview_type,
            total_questions=len(qa_pairs), avg_score=avg_score,
            scores_list=", ".join(f"{s}/10" for s in scores),
            all_strengths="; ".join(all_strengths[:10]),
            all_weaknesses="; ".join(all_weaknesses[:10]),
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def compile_report(self, session_id, candidate_name, role, interview_type, qa_pairs, output_dir="./data/reports"):
        executive_summary = self.generate_executive_summary(candidate_name, role, interview_type, qa_pairs)
        scores = [qa["score"] for qa in qa_pairs if qa.get("score") is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(output_dir, f"report_{session_id[:8]}_{timestamp}.pdf")
        build_pdf_report(
            path=pdf_path, candidate_name=candidate_name, role=role,
            interview_type=interview_type, avg_score=avg_score,
            executive_summary=executive_summary, qa_pairs=qa_pairs,
            session_date=datetime.now().strftime("%B %d, %Y"),
        )
        return executive_summary, pdf_path, avg_score

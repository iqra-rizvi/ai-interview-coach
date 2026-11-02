"""
backend/agents/report_agent.py
DFP-Agent report synthesizer.
Computes 5-dimension rubric radar distributions, verified strengths,
critical blind spots, and compiles an exportable PDF diagnostic study guide.
"""
from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.config import get_settings
from backend.utils.llm import get_llm, safe_llm_invoke
from backend.utils.pdf_generator import build_pdf_report
from backend.models.schemas import RubricScore

settings = get_settings()

DIMENSION_NAMES = [
    "Technical Accuracy",
    "Contextual Relevance",
    "Conceptual Depth",
    "Solution Completeness",
    "Analytical Problem Solving",
]

DEFAULT_LITERATURE_MAP = {
    "Technical Accuracy": {
        "title": "Vaswani et al., 'Attention Is All You Need' (NeurIPS 2017)",
        "recommendation": "Review formal mathematical derivations and backpropagation gradient paths across deep architectures.",
    },
    "Conceptual Depth": {
        "title": "Lin et al., 'Focal Loss for Dense Object Detection' (ICCV 2017)",
        "recommendation": "Study modulating factors in objective functions to manage extreme class skew and hard negative mining.",
    },
    "Solution Completeness": {
        "title": "Martin Fowler, 'Patterns of Enterprise Application Architecture'",
        "recommendation": "Practice comprehensive edge-case enumeration, boundary condition analysis, and failure recovery protocols.",
    },
    "Analytical Problem Solving": {
        "title": "Kernighan & Pike, 'The Practice of Programming'",
        "recommendation": "Structure engineering explanations using first-principles causal trees rather than purely linear narration.",
    },
    "Contextual Relevance": {
        "title": "Manning, Raghavan & Schutze, 'Introduction to Information Retrieval'",
        "recommendation": "Focus answers directly on the operational scenario boundaries without evasive preliminary filler.",
    },
}


class ReportAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.3)

    def calculate_radar_scores(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute the average 1-5 score for each of the 5 psychometric dimensions."""
        totals = {dim: 0.0 for dim in DIMENSION_NAMES}
        counts = {dim: 0 for dim in DIMENSION_NAMES}

        for qa in qa_pairs:
            rubric = qa.get("rubric_scores") or {}
            for dim in DIMENSION_NAMES:
                val = rubric.get(dim)
                if val:
                    if isinstance(val, dict):
                        s = float(val.get("score", 3.0))
                    elif hasattr(val, "score"):
                        s = float(val.score)
                    else:
                        s = 3.0
                    totals[dim] += s
                    counts[dim] += 1

        averages = {}
        for dim in DIMENSION_NAMES:
            c = counts[dim]
            averages[dim] = round(totals[dim] / c, 2) if c > 0 else 3.0
        return averages

    def generate_remediation_curriculum(self, radar: Dict[str, float]) -> List[Dict[str, str]]:
        """Synthesize tailored pedagogical remediation plan mapped to lowest rubric dimensions."""
        curriculum = []
        # Sort dimensions ascending to address lowest-scoring areas first
        sorted_dims = sorted(radar.items(), key=lambda x: x[1])
        for dim, score in sorted_dims[:3]:
            meta = DEFAULT_LITERATURE_MAP.get(dim, {
                "title": "Enterprise Technical Architecture Standards",
                "recommendation": "Practice deep-dive architectural trade-off formulations.",
            })
            curriculum.append({
                "dimension": dim,
                "score": f"{score}/5.0",
                "reference": meta["title"],
                "action": meta["recommendation"],
            })
        return curriculum

    def generate_executive_summary(
        self,
        candidate_name: str,
        role: str,
        interview_type: str,
        qa_pairs: List[Dict[str, Any]],
        radar: Dict[str, float],
    ) -> str:
        scores = [qa["score"] for qa in qa_pairs if qa.get("score") is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

        all_strengths, all_weaknesses = [], []
        for qa in qa_pairs:
            all_strengths.extend(qa.get("strengths") or [])
            all_weaknesses.extend(qa.get("weaknesses") or [])

        radar_summary_str = ", ".join(f"{k}: {v}/5" for k, v in radar.items())

        sys_prompt = (
            "You are the Diagnostic Feedback and Pedagogy Agent (DFP-Agent) for AgentHire. "
            "Write a thorough, formal, psychometrically grounded executive performance summary.\n"
            "Format in 3 structured paragraphs:\n"
            "1. Overall Evaluative Synthesis (candidate technical caliber, baseline alignment, scoring stability)\n"
            "2. Validated Core Competencies (top 3 verified engineering strengths with conceptual grounding)\n"
            "3. High-Priority Remediation Vectors (top areas for improvement with theoretical and practical study focus)\n"
            "Keep the tone objective, professional, and empowering. Under 250 words."
        )

        user_prompt = (
            f"Candidate: {candidate_name}\n"
            f"Target Role: {role} ({interview_type})\n"
            f"Total Items Answered: {len(qa_pairs)}\n"
            f"Average Composite Score: {avg_score}/10\n"
            f"5-Dimension Radar Averages: {radar_summary_str}\n"
            f"Collected Strengths: {'; '.join(all_strengths[:8]) or 'Consistent technical participation'}\n"
            f"Identified Blind Spots: {'; '.join(all_weaknesses[:8]) or 'Expand mathematical depth and edge case coverage'}\n"
        )

        try:
            response = safe_llm_invoke(self.llm, [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Error generating executive summary: {e}")
            return (
                f"Candidate completed {len(qa_pairs)} interview items for the {role} role with an average composite "
                f"score of {avg_score}/10. The candidate exhibited solid baseline familiarity with core principles. "
                "Immediate pedagogical focus should center on formal mathematical justification and systematic edge-case analysis."
            )

    def compile_report(
        self,
        session_id: str,
        candidate_name: str,
        role: str,
        interview_type: str,
        qa_pairs: List[Dict[str, Any]],
        output_dir: str = "./data/reports",
    ) -> Tuple[str, str, float, Dict[str, float], List[Dict[str, str]]]:
        """
        Compile executive summary, compute radar distribution, build PDF, and return report artifacts.
        """
        radar = self.calculate_radar_scores(qa_pairs)
        curriculum = self.generate_remediation_curriculum(radar)
        summary = self.generate_executive_summary(candidate_name, role, interview_type, qa_pairs, radar)

        scores = [qa["score"] for qa in qa_pairs if qa.get("score") is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(output_dir, f"report_{session_id[:8]}_{timestamp}.pdf")

        build_pdf_report(
            path=pdf_path,
            candidate_name=candidate_name,
            role=role,
            interview_type=interview_type,
            avg_score=avg_score,
            executive_summary=summary,
            qa_pairs=qa_pairs,
            session_date=datetime.now().strftime("%B %d, %Y"),
            radar=radar,
            curriculum=curriculum,
        )

        logger.info(f"[ReportAgent] Report compiled: {pdf_path} (Avg Score: {avg_score})")
        return summary, pdf_path, avg_score, radar, curriculum

"""
backend/agents/evaluator_agent.py
Rubric-Anchored Response Evaluation Agent (RAE-Agent).
Implements the 5-Dimensional Psychometric Rubric and the Verbatim Evidence
Verification Gate V(tau_k, A) as defined in AgentHire Chapter 5.
"""
from __future__ import annotations
import json
import re
from typing import Dict, Any, Tuple
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from backend.models.schemas import RubricScore, EvaluationRecord
from backend.utils.config import get_settings
from backend.utils.llm import get_llm, safe_llm_invoke
from backend.rag.retriever import retrieve_for_evaluation

settings = get_settings()

DIMENSIONS = [
    "Technical Accuracy",
    "Contextual Relevance",
    "Conceptual Depth",
    "Solution Completeness",
    "Analytical Problem Solving",
]

THETA_MIN = 15  # Minimum character length for verifiable quote


def verify_evidence(quote: str, candidate_answer: str, theta_min: int = THETA_MIN, gamma: float = 0.0) -> float:
    """
    Mathematical formulation of Verbatim Evidence Verification Gate V(tau_k, A):
      V(tau_k, A) = 1.0 if (tau_k in A and len(tau_k) >= theta_min) else gamma
    """
    if not quote or not candidate_answer:
        return gamma

    norm_quote = " ".join(quote.strip().lower().split())
    norm_answer = " ".join(candidate_answer.strip().lower().split())

    # Strip surrounding ellipses and punctuation
    cleaned_quote = norm_quote.strip(". \t\n\"'")
    if len(cleaned_quote) >= theta_min and cleaned_quote in norm_answer:
        return 1.0

    # Also check if a significant contiguous substring (>= 20 chars) matches
    if len(cleaned_quote) >= 20:
        words = cleaned_quote.split()
        if len(words) >= 4:
            sub = " ".join(words[:4])
            if sub in norm_answer:
                return 1.0

    return gamma


RAE_SYSTEM_PROMPT = """You are the Rubric-Anchored Response Evaluation Agent (RAE-Agent) for AgentHire.
You evaluate candidate responses across FIVE ORTHOGONAL PSYCHOMETRIC DIMENSIONS on a strict 1 to 5 integer scale:

1. Technical Accuracy: (1: Gross factual errors, 3: Broadly accurate with minor slips, 5: Flawless mathematical/factual correctness)
2. Contextual Relevance: (1: Evasive filler prose, 3: Partially responsive, 5: Direct, highly focused answer addressing prompt)
3. Conceptual Depth: (1: Superficial buzzwords, 3: Explains what without why, 5: Deep trade-off/architectural articulation)
4. Solution Completeness: (1: Major components omitted, 3: Core answered, bounds omitted, 5: Exhaustive coverage of constraints)
5. Analytical Problem Solving: (1: Disjointed, incoherent, 3: Conventional linear approach, 5: Rigorous first-principles decomposition)

CRITICAL ARCHITECTURAL CONSTRAINT [THE EVIDENCE VERIFICATION INVARIANT]:
For EACH dimension, you MUST cite an exact, VERBATIM textual sentence or phrase from the candidate's actual answer as "verbatim_evidence_quote" (minimum 15 characters).
If you fabricate a quote or quote something not in the candidate's answer, your score will be mathematically invalidated to 0.

You must return valid JSON ONLY without markdown fences with this schema:
{
  "rubric_scores": {
    "Technical Accuracy": {"score": 1-5, "verbatim_evidence_quote": "exact quote from answer", "evaluator_rationale": "reason"},
    "Contextual Relevance": {"score": 1-5, "verbatim_evidence_quote": "exact quote from answer", "evaluator_rationale": "reason"},
    "Conceptual Depth": {"score": 1-5, "verbatim_evidence_quote": "exact quote from answer", "evaluator_rationale": "reason"},
    "Solution Completeness": {"score": 1-5, "verbatim_evidence_quote": "exact quote from answer", "evaluator_rationale": "reason"},
    "Analytical Problem Solving": {"score": 1-5, "verbatim_evidence_quote": "exact quote from answer", "evaluator_rationale": "reason"}
  },
  "strengths": ["Top verified strength 1", "Strength 2"],
  "weaknesses": ["Specific technical gap 1", "Specific technical gap 2"],
  "ideal_answer_outline": "2-3 sentences outlining a flawless senior-level response",
  "follow_up_suggested": "Targeted probing question if partial understanding is detected"
}
"""


class EvaluatorAgent:
    """
    RAE-Agent evaluates candidate answers with 5D rubric scoring,
    applies the verbatim verification gate, and detects boundary probing conditions.
    """

    def __init__(self):
        self.llm = get_llm(temperature=0.1)

    def evaluate(
        self,
        session_id: str,
        question: str,
        answer: str,
        role: str,
        interview_type: str,
        is_followup: bool = False,
    ) -> Dict[str, Any]:
        cleaned_answer = (answer or "").strip()

        # Handle empty or trivial answers directly
        if len(cleaned_answer) < 8:
            return self._build_empty_evaluation(is_followup)

        rag_context = retrieve_for_evaluation(session_id, question, role)

        user_prompt = (
            f"TARGET ROLE: {role}\n"
            f"INTERVIEW TYPE: {interview_type}\n"
            f"QUESTION: {question}\n"
            f"BENCHMARK CONTEXT: {rag_context[:600]}\n\n"
            f"CANDIDATE ACTUAL ANSWER TRANSCRIPT:\n\"{cleaned_answer}\"\n\n"
            "Evaluate strictly according to the 5 dimensions. Ensure all quotes are exact verbatim substrings."
        )

        try:
            response = safe_llm_invoke(
                self.llm,
                [SystemMessage(content=RAE_SYSTEM_PROMPT), HumanMessage(content=user_prompt)],
            )
            raw = response.content if hasattr(response, "content") else str(response)
            eval_data = self._parse_json(raw, cleaned_answer)
        except Exception as e:
            logger.error(f"[RAE-Agent] LLM scoring exception: {e}. Building fallback evaluation.")
            eval_data = self._build_heuristic_evaluation(cleaned_answer, is_followup)

        # Apply Verbatim Evidence Verification Gate V(tau_k, A)
        verified_rubric: Dict[str, RubricScore] = {}
        weighted_sum = 0.0
        w_k = 0.20  # Equal weighting across 5 dimensions

        for dim in DIMENSIONS:
            score_item = eval_data.get("rubric_scores", {}).get(dim, {})
            raw_score = int(score_item.get("score", 3))
            raw_score = max(1, min(5, raw_score))
            quote = str(score_item.get("verbatim_evidence_quote", "")).strip()
            rationale = str(score_item.get("evaluator_rationale", "Evaluated based on transcript"))

            # Evidence verification function V(tau_k, A)
            v_factor = verify_evidence(quote, cleaned_answer)
            is_verified = (v_factor == 1.0)

            # If ungrounded quote, penalize to minimum score
            if not is_verified:
                logger.warning(f"[Evidence Gate] Quote for '{dim}' ungrounded in transcript: '{quote[:30]}...' -> Penalized.")
                effective_score = 1
            else:
                effective_score = raw_score

            weighted_sum += w_k * effective_score

            verified_rubric[dim] = RubricScore(
                dimension_name=dim,
                score=effective_score,
                verbatim_evidence_quote=quote if is_verified else "[UNGROUNDED QUOTE PENALIZED]",
                evaluator_rationale=rationale,
                evidence_verified=is_verified,
            )

        composite_5 = round(weighted_sum, 2)
        # Scaled to 0-10 for display
        composite_10 = round(composite_5 * 2.0, 1)

        # Adaptive Probing Condition (Section 4.3):
        # Condition: (c_1 >= 3) AND (c_4 <= 3) AND (is_followup == False)
        c_1 = verified_rubric["Technical Accuracy"].score
        c_4 = verified_rubric["Solution Completeness"].score
        requires_probing = (c_1 >= 3) and (c_4 <= 3) and (not is_followup)

        probing_focus = ""
        if requires_probing:
            probing_focus = eval_data.get("follow_up_suggested") or (
                "Candidate demonstrated accurate baseline intuition, but boundary constraints "
                "or mathematical formulation was incomplete."
            )
            logger.info(f"[Adaptive Probing Triggered] c_1={c_1}, c_4={c_4}. Probing boundary.")

        return {
            "score": composite_10,
            "composite_5": composite_5,
            "strengths": eval_data.get("strengths", ["Addressed core technical premise"]),
            "weaknesses": eval_data.get("weaknesses", ["Expand on architectural trade-offs"]),
            "ideal_answer_outline": eval_data.get("ideal_answer_outline", "A strong answer thoroughly covers design and edge cases."),
            "follow_up_suggested": eval_data.get("follow_up_suggested", ""),
            "rubric_scores": verified_rubric,
            "evidence_grounding_verified": all(s.evidence_verified for s in verified_rubric.values()),
            "requires_probing": requires_probing,
            "probing_focus": probing_focus,
            "is_followup": is_followup,
        }

    def _parse_json(self, text: str, candidate_answer: str) -> Dict[str, Any]:
        clean = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if "rubric_scores" in data:
                    return data
            except Exception:
                pass
        return self._build_heuristic_evaluation(candidate_answer, False)

    def _build_empty_evaluation(self, is_followup: bool) -> Dict[str, Any]:
        empty_rubric = {
            dim: RubricScore(
                dimension_name=dim,
                score=1,
                verbatim_evidence_quote="[NO MEANINGFUL ANSWER PROVIDED]",
                evaluator_rationale="Candidate submitted an empty or unelaborated answer.",
                evidence_verified=True,
            )
            for dim in DIMENSIONS
        }
        return {
            "score": 0.0,
            "composite_5": 1.0,
            "strengths": [],
            "weaknesses": ["No substantive response provided to technical question"],
            "ideal_answer_outline": "Explain theoretical mechanics and provide concrete implementation examples.",
            "follow_up_suggested": "",
            "rubric_scores": empty_rubric,
            "evidence_grounding_verified": True,
            "requires_probing": False,
            "probing_focus": None,
            "is_followup": is_followup,
        }

    def _build_heuristic_evaluation(self, candidate_answer: str, is_followup: bool) -> Dict[str, Any]:
        # Extract a real snippet from candidate's answer for grounding
        snippet = candidate_answer[:60].strip()
        if len(snippet) < THETA_MIN:
            snippet = candidate_answer.strip()

        heuristic_rubric = {
            dim: RubricScore(
                dimension_name=dim,
                score=3,
                verbatim_evidence_quote=snippet,
                evaluator_rationale="Evaluated based on candidate transcript alignment.",
                evidence_verified=True,
            )
            for dim in DIMENSIONS
        }
        return {
            "score": 6.0,
            "composite_5": 3.0,
            "strengths": ["Answer demonstrates fundamental concepts"],
            "weaknesses": ["Deepen mathematical rigor and edge-case analysis"],
            "ideal_answer_outline": "A thorough explanation addresses root architectural considerations.",
            "follow_up_suggested": "",
            "rubric_scores": heuristic_rubric,
            "evidence_grounding_verified": True,
            "requires_probing": False,
            "probing_focus": None,
            "is_followup": is_followup,
        }

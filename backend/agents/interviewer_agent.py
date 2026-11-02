"""
backend/agents/interviewer_agent.py
Combines the Dynamic Question Generation Agent (DQG-Agent) and
Adaptive Interview Simulation Agent (AIS-Agent) from AgentHire Sections 3.3 and 3.4.
Enforces 3 Pedagogical Tiers, difficulty scaling lambda in [1, 5], sliding window context,
and real-time knowledge-boundary follow-up probing.
"""
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.config import get_settings
from backend.utils.llm import get_llm, safe_llm_invoke
from backend.utils.prompt_templates import TECHNICAL_TOPICS, HR_TOPICS
from backend.rag.retriever import retrieve_for_question_generation

settings = get_settings()

TIERS = [
    "Tier 1: Foundational",
    "Tier 2: Systems Architecture",
    "Tier 3: Situational STAR",
]


class InterviewerAgent:
    """
    DQG-Agent & AIS-Agent:
    Synthesizes tiered technical items, manages conversational turn-taking,
    and executes adaptive boundary follow-up probes.
    """

    def __init__(self):
        self.llm = get_llm(temperature=0.7)

    def _determine_tier_and_difficulty(self, question_number: int, max_questions: int) -> Tuple[str, int]:
        """Progressively scale pedagogical tier and difficulty lambda in [1, 5]."""
        ratio = question_number / max(max_questions, 1)
        if ratio <= 0.35:
            tier = "Tier 1: Foundational"
            difficulty = 2 if question_number == 1 else 3
        elif ratio <= 0.70:
            tier = "Tier 2: Systems Architecture"
            difficulty = 4
        else:
            tier = "Tier 3: Situational STAR"
            difficulty = 4 if ratio < 0.9 else 5
        return tier, difficulty

    def _get_next_topic(
        self,
        role: str,
        interview_type: str,
        question_number: int,
        covered_topics: List[str],
        competency_gaps: List[str] | None = None,
    ) -> str:
        # Prioritize detected competency gaps from the CPA-Agent if available!
        if competency_gaps:
            unaddressed_gaps = [g for g in competency_gaps if g not in covered_topics]
            if unaddressed_gaps:
                return unaddressed_gaps[0]

        if interview_type == "hr":
            pool = HR_TOPICS
        elif interview_type == "technical":
            pool = TECHNICAL_TOPICS.get(role, TECHNICAL_TOPICS["Software Engineer"])
        else:
            tech = TECHNICAL_TOPICS.get(role, TECHNICAL_TOPICS["Software Engineer"])
            pool = tech if question_number % 2 == 0 else HR_TOPICS

        uncovered = [t for t in pool if t not in covered_topics]
        return uncovered[0] if uncovered else pool[question_number % len(pool)]

    def generate_first_question(
        self,
        session_id: str,
        role: str,
        interview_type: str,
        resume_summary: str,
        max_questions: int,
        competency_gaps: List[str] | None = None,
    ) -> Tuple[str, str, str, int]:
        """
        Generate initial question.
        Returns: (question_text, topic, pedagogical_tier, difficulty_lambda)
        """
        tier = "Tier 1: Foundational"
        difficulty = 2
        topic = "System Introduction & Technical Background"

        rag_context = retrieve_for_question_generation(session_id, role, topic, interview_type)

        sys_prompt = (
            f"You are the Lead Technical Interviewer (AIS-Agent) for AgentHire conducting a mock screen.\n"
            f"Candidate Role: {role}\n"
            f"Interview Mode: {interview_type}\n"
            f"Pedagogical Tier: {tier} (Difficulty {difficulty}/5)\n"
            f"Candidate Profile: {resume_summary[:400]}\n\n"
            "Greet the candidate professionally in 1 sentence, then pose your first foundational technical/background "
            "question to set a baseline. Keep it concise, engaging, and clear. Output ONLY the interviewer dialogue."
        )

        try:
            response = safe_llm_invoke(self.llm, [SystemMessage(content=sys_prompt), HumanMessage(content="Start interview.")])
            question = response.content.strip()
        except Exception as e:
            logger.warning(f"Error generating first question: {e}")
            question = f"Welcome to your technical screen for the {role} position. To start, walk me through your engineering background and the core architecture of a project you recently built."

        return question, topic, tier, difficulty

    def generate_next_question(
        self,
        session_id: str,
        role: str,
        interview_type: str,
        resume_summary: str,
        last_answer: str,
        question_number: int,
        max_questions: int,
        covered_topics: List[str],
        competency_gaps: List[str] | None = None,
        is_final: bool = False,
    ) -> Tuple[str, str, str, int]:
        """
        Generate subsequent tiered question using sliding context window.
        Returns: (question_text, topic, pedagogical_tier, difficulty_lambda)
        """
        tier, difficulty = self._determine_tier_and_difficulty(question_number, max_questions)
        topic = self._get_next_topic(role, interview_type, question_number, covered_topics, competency_gaps)
        rag_context = retrieve_for_question_generation(session_id, role, topic, interview_type)

        sys_prompt = (
            f"You are the AIS-Agent driving turn {question_number} of {max_questions} in AgentHire.\n"
            f"Target Role: {role} ({interview_type})\n"
            f"Pedagogical Tier: {tier} (Difficulty: {difficulty}/5)\n"
            f"Focus Topic / Target Competency: {topic}\n"
            f"Resume Profile: {resume_summary[:300]}\n"
            f"Knowledge Base Context: {rag_context[:400]}\n\n"
            "Rules:\n"
            "- Ask exactly ONE focused question targeting this tier.\n"
            "- Do NOT give commentary, feedback, or say 'Good job!'.\n"
            "- If this is Tier 2, ask about systems trade-offs or scalability.\n"
            "- If Tier 3, pose a situational production debugging / post-mortem scenario.\n"
            "- Return ONLY the question string."
        )

        user_content = (
            f"Candidate's previous response summary: \"{last_answer[:400]}\"\n"
            f"Please present the next question for Topic: {topic}."
        )

        try:
            response = safe_llm_invoke(
                self.llm,
                [SystemMessage(content=sys_prompt), HumanMessage(content=user_content)],
            )
            question = response.content.strip()
        except Exception as e:
            logger.warning(f"Error generating next question: {e}")
            question = f"Regarding {topic}: Can you explain the core design trade-offs you encounter when implementing this in a production system?"

        return question, topic, tier, difficulty

    def synthesize_followup_probe(
        self,
        question: str,
        candidate_answer: str,
        evaluation: Dict[str, Any],
        role: str,
    ) -> Tuple[str, str, str, int]:
        """
        AIS-Agent Adaptive Probing (Section 4.3):
        When candidate response satisfies (c_1 >= 3) AND (c_4 <= 3), synthesize
        a surgical follow-up probe targeting the unelaborated boundary.
        """
        probing_focus = evaluation.get("probing_focus", "")
        ideal_outline = evaluation.get("ideal_answer_outline", "")

        sys_prompt = (
            f"You are the Adaptive Interview Simulation Agent (AIS-Agent) executing an adaptive boundary probe.\n"
            f"Role: {role}\n"
            f"Prior Question: \"{question}\"\n"
            f"Candidate's Answer: \"{candidate_answer[:500]}\"\n"
            f"Identified Boundary / Gap: {probing_focus}\n"
            f"Ideal Consideration: {ideal_outline}\n\n"
            "Task: Generate a direct, professional follow-up probing question. "
            "Acknowledge the valid technical point they made in one phrase, and immediately probe "
            "the specific boundary, mathematical detail, or edge case they omitted.\n"
            "Example style: 'You correctly mentioned applying weighted cross-entropy, but how exactly does the loss formula incorporate class frequencies, and what is the mathematical intuition behind Focal Loss in comparison?'\n"
            "Return ONLY the probing question text without preamble."
        )

        try:
            response = safe_llm_invoke(
                self.llm,
                [SystemMessage(content=sys_prompt), HumanMessage(content="Synthesize follow-up probe.")],
            )
            probe_text = response.content.strip()
        except Exception as e:
            logger.warning(f"Error synthesizing probe: {e}")
            probe_text = f"You touched on the core mechanism, but could you elaborate specifically on how you handle the underlying boundary conditions and failure modes in that scenario?"

        return probe_text, "Adaptive Boundary Probe", "Tier 2: Systems Architecture", 4

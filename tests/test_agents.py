"""
tests/test_agents.py
Rigorous unit and integration test suite for AgentHire.
Covers 5D rubric evaluation, verbatim evidence verification gate V(tau_k, A),
adaptive probing condition, competency divergence Delta(C, J), PII blinding,
and PDF generation safety.
"""
from __future__ import annotations
import pytest
import json
import os
from unittest.mock import MagicMock, patch

from backend.models.schemas import RubricScore, CandidateProfile
from backend.agents.evaluator_agent import EvaluatorAgent, verify_evidence
from backend.agents.interviewer_agent import InterviewerAgent
from backend.rag.ingestion import (
    chunk_text,
    clean_text,
    sanitize_pii,
    compute_competency_divergence,
)
from backend.utils.pdf_generator import build_pdf_report, _esc


class TestEvidenceVerificationGate:
    """Tests for the Verbatim Evidence Verification Gate V(tau_k, A) (Eq. 5.2)."""

    def test_verbatim_quote_verified(self):
        answer = "Vanishing gradients occur because backpropagation multiplies activation derivatives across deep layers."
        quote = "multiplies activation derivatives"
        v_score = verify_evidence(quote, answer, theta_min=15)
        assert v_score == 1.0

    def test_fabricated_quote_invalidated(self):
        answer = "I built a web scraper in Python using BeautifulSoup."
        hallucinated_quote = "distributed Kubernetes cluster with Raft consensus"
        v_score = verify_evidence(hallucinated_quote, answer, theta_min=15)
        assert v_score == 0.0

    def test_short_quote_below_threshold_invalidated(self):
        answer = "Yes, I know PyTorch well."
        short_quote = "PyTorch"  # Less than theta_min=15 characters
        v_score = verify_evidence(short_quote, answer, theta_min=15)
        assert v_score == 0.0


class TestEvaluatorAgent:
    """Tests for the Rubric-Anchored Response Evaluation Agent (RAE-Agent)."""

    def test_5d_rubric_evaluation(self):
        candidate_answer = (
            "Accuracy is misleading because a trivial model predicting non-fraud every time gets 99.8% accuracy. "
            "PR-AUC is much better than ROC-AUC because ROC-AUC is flattered by the huge number of true negatives."
        )
        mock_response = {
            "rubric_scores": {
                "Technical Accuracy": {
                    "score": 5,
                    "verbatim_evidence_quote": "trivial model predicting non-fraud every time gets 99.8% accuracy",
                    "evaluator_rationale": "Accurate calculation of majority class baseline.",
                },
                "Contextual Relevance": {
                    "score": 5,
                    "verbatim_evidence_quote": "PR-AUC is much better than ROC-AUC",
                    "evaluator_rationale": "Directly confronts imbalanced metric trade-offs.",
                },
                "Conceptual Depth": {
                    "score": 4,
                    "verbatim_evidence_quote": "flattered by the huge number of true negatives",
                    "evaluator_rationale": "Identifies false positive rate denominator skew.",
                },
                "Solution Completeness": {
                    "score": 2,
                    "verbatim_evidence_quote": "Accuracy is misleading because",
                    "evaluator_rationale": "Omitted objective function formulation and focal loss.",
                },
                "Analytical Problem Solving": {
                    "score": 4,
                    "verbatim_evidence_quote": "PR-AUC is much better than ROC-AUC because",
                    "evaluator_rationale": "First-principles metric reasoning.",
                },
            },
            "strengths": ["Clear metric intuition"],
            "weaknesses": ["Omitted loss function"],
            "ideal_answer_outline": "Formulate weighted cross-entropy or Focal Loss.",
            "follow_up_suggested": "How would you mathematically formulate class weights?",
        }

        with patch("backend.agents.evaluator_agent.get_llm"):
            with patch("backend.agents.evaluator_agent.retrieve_for_evaluation", return_value=""):
                agent = EvaluatorAgent.__new__(EvaluatorAgent)
                agent.llm = MagicMock()
                with patch("backend.agents.evaluator_agent.safe_llm_invoke") as mock_invoke:
                    mock_invoke.return_value = MagicMock(content=json.dumps(mock_response))
                    result = agent.evaluate(
                        session_id="s1",
                        question="Why is accuracy misleading?",
                        answer=candidate_answer,
                        role="Machine Learning Engineer",
                        interview_type="technical",
                    )

        assert result["score"] > 0
        assert "Technical Accuracy" in result["rubric_scores"]
        assert result["requires_probing"] is True  # c_1 >= 3 and c_4 <= 2 triggers adaptive probing!

    def test_empty_answer_scores_zero(self):
        with patch("backend.agents.evaluator_agent.get_llm"):
            agent = EvaluatorAgent.__new__(EvaluatorAgent)
            agent.llm = MagicMock()
            result = agent.evaluate("s1", "Explain BST", "   ", "Software Engineer", "technical")
        assert result["score"] == 0.0
        assert result["requires_probing"] is False


class TestCompetencyDivergenceAndPII:
    """Tests for CPA-Agent divergence calculation and PII blinding (Section 3.2 & 7.2)."""

    def test_pii_sanitization(self):
        raw_cv = (
            "Iqra Rizvi\n"
            "Email: iqra@example.com, Phone: +1 (555) 234-5678\n"
            "Graduated from Amity University Uttar Pradesh (2023 - 2027)\n"
            "Portfolio: https://github.com/iqra-rizvi\n"
            "Built distributed PyTorch pipeline."
        )
        sanitized = sanitize_pii(raw_cv)
        assert "iqra@example.com" not in sanitized
        assert "[REDACTED_EMAIL]" in sanitized
        assert "[REDACTED_PHONE]" in sanitized
        assert "https://github.com/iqra-rizvi" not in sanitized
        assert "Built distributed PyTorch pipeline." in sanitized

    def test_divergence_computation_basic(self):
        candidate_skills = ["Python", "PyTorch", "Scikit-learn", "Git"]
        overall_div, per_skill, gaps = compute_competency_divergence(
            candidate_skills=candidate_skills,
            target_role="Machine Learning Engineer",
        )
        assert 0.0 <= overall_div <= 1.0
        assert isinstance(per_skill, dict)
        assert len(per_skill) > 0


class TestAdaptiveInterviewer:
    """Tests for tiered item synthesis and adaptive boundary probing."""

    def test_adaptive_probe_synthesis(self):
        interviewer = InterviewerAgent.__new__(InterviewerAgent)
        interviewer.llm = MagicMock()

        mock_probe = "You mentioned weighted loss, but how does Focal Loss incorporate gamma to down-weight easy negatives?"
        with patch("backend.agents.interviewer_agent.safe_llm_invoke") as mock_invoke:
            mock_invoke.return_value = MagicMock(content=mock_probe)
            probe_text, topic, tier, diff = interviewer.synthesize_followup_probe(
                question="Explain imbalanced classification.",
                candidate_answer="We can use weighted loss.",
                evaluation={"probing_focus": "Omitted Focal Loss mathematical intuition"},
                role="Machine Learning Engineer",
            )
            assert "Focal Loss" in probe_text
            assert tier == "Tier 2: Systems Architecture"
            assert diff == 4


class TestPDFGenerator:
    """Tests for ReportLab PDF generation, XML escaping, and radar inclusion."""

    def test_pdf_with_special_characters_no_crash(self, tmp_path):
        out_path = str(tmp_path / "test_report.pdf")
        # Include characters that notoriously break ReportLab XML parser: <T>, &, >
        malicious_code_text = "When writing template<typename T> & using a < b && x > y, memory safe."

        build_pdf_report(
            path=out_path,
            candidate_name="Iqra Rizvi <Test>",
            role="Software Engineer & Cloud Systems",
            interview_type="mixed",
            avg_score=8.5,
            executive_summary=f"Summary with XML tokens: {malicious_code_text}",
            qa_pairs=[{
                "question_number": 1,
                "question": "Explain C++ generics: <T> & containers?",
                "answer": malicious_code_text,
                "topic": "Generics & Templates",
                "score": 8.5,
                "strengths": ["Used <typename T> correctly"],
                "weaknesses": ["Did not mention concepts & constraints"],
                "feedback": "Great explanation of template<T> instantiation.",
                "rubric_scores": {
                    "Technical Accuracy": {
                        "score": 5,
                        "verbatim_evidence_quote": "template<typename T>",
                        "evaluator_rationale": "Correct generic syntax.",
                    }
                },
            }],
            session_date="October 2026",
            radar={"Technical Accuracy": 4.5, "Conceptual Depth": 4.0},
        )
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000

    def test_pdf_empty_qa_pairs_no_crash(self, tmp_path):
        out_path = str(tmp_path / "empty_report.pdf")
        build_pdf_report(
            path=out_path,
            candidate_name="Test Candidate",
            role="Data Scientist",
            interview_type="technical",
            avg_score=0.0,
            executive_summary="Candidate concluded session early.",
            qa_pairs=[],
            session_date="October 2026",
        )
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 500


class TestFastAPIIntegration:
    """End-to-end API lifecycle test using FastAPI TestClient."""

    def test_full_session_flow(self):
        from fastapi.testclient import TestClient
        from backend.api.main import app

        client = TestClient(app)

        # 1. Health check
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        # 2. Create session
        create_resp = client.post("/api/sessions/", json={
            "user_id": "test-user-001",
            "role": "Software Engineer",
            "interview_type": "technical",
        })
        assert create_resp.status_code == 201
        session_id = create_resp.json()["id"]

        # 3. Start session -> generate first question
        with patch("backend.agents.interviewer_agent.safe_llm_invoke") as mock_invoke:
            mock_invoke.return_value = MagicMock(content="Explain how hash tables achieve average O(1) lookups.")
            start_resp = client.post(f"/api/sessions/{session_id}/start")
            assert start_resp.status_code == 200
            q_data = start_resp.json()
            assert "question" in q_data
            assert q_data["question_number"] == 1
            assert q_data["pedagogical_tier"] == "Tier 1: Foundational"

        # 4. Finish session -> generate report
        finish_resp = client.post(f"/api/sessions/{session_id}/finish")
        assert finish_resp.status_code == 200
        assert "avg_score" in finish_resp.json()


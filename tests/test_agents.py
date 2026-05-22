"""
tests/test_agents.py
Unit tests — run with: pytest tests/ -v
"""
import pytest
import json
from unittest.mock import MagicMock, patch


class TestEvaluatorAgent:
    def test_valid_json_response(self):
        mock_json = json.dumps({
            "score": 7, "strengths": ["Clear", "Examples"],
            "weaknesses": ["Too brief"],
            "ideal_answer_outline": "A great answer would...",
            "follow_up_suggested": "",
        })
        with patch("backend.agents.evaluator_agent.ChatGroq") as mock_cls:
            with patch("backend.agents.evaluator_agent.retrieve_for_evaluation", return_value=""):
                from backend.agents.evaluator_agent import EvaluatorAgent
                agent = EvaluatorAgent.__new__(EvaluatorAgent)
                agent.llm = MagicMock()
                agent.llm.invoke.return_value = MagicMock(content=mock_json)
                result = agent.evaluate("s", "What is a BST?", "A tree...", "Software Engineer", "technical")
        assert result["score"] == 7.0

    def test_empty_answer_scores_zero(self):
        with patch("backend.agents.evaluator_agent.ChatGroq"):
            with patch("backend.agents.evaluator_agent.retrieve_for_evaluation", return_value=""):
                from backend.agents.evaluator_agent import EvaluatorAgent
                agent = EvaluatorAgent.__new__(EvaluatorAgent)
                agent.llm = MagicMock()
                result = agent.evaluate("s", "q", "   ", "role", "hr")
        assert result["score"] == 0


class TestIngestion:
    def test_chunk_text_basic(self):
        from backend.rag.ingestion import chunk_text
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=400, overlap=80)
        assert len(chunks) > 1

    def test_clean_text(self):
        from backend.rag.ingestion import clean_text
        result = clean_text("Hello   \n\n  World\x00")
        assert "Hello" in result and "World" in result and "\x00" not in result


class TestPDFGenerator:
    def test_pdf_created(self, tmp_path):
        from backend.utils.pdf_generator import build_pdf_report
        out = str(tmp_path / "test.pdf")
        build_pdf_report(
            path=out, candidate_name="Test User", role="Software Engineer",
            interview_type="mixed", avg_score=8.0,
            executive_summary="Strong performance.",
            qa_pairs=[{
                "question_number": 1, "question": "Tell me about yourself.",
                "answer": "I am a software engineer.", "topic": "introduction",
                "score": 8.0, "strengths": ["Clear"], "weaknesses": ["Add metrics"],
                "feedback": "Good intro.", "ideal_answer_outline": "Background + skills.",
            }],
            session_date="January 1, 2025",
        )
        import os
        assert os.path.exists(out) and os.path.getsize(out) > 1000

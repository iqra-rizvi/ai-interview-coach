"""
backend/api/websocket.py
WebSocket endpoint for real-time streaming interviews.
"""
from __future__ import annotations
import uuid
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from backend.agents.interviewer_agent import InterviewerAgent
from backend.agents.evaluator_agent import EvaluatorAgent
from backend.agents.feedback_agent import FeedbackAgent
from backend.agents.report_agent import ReportAgent
from backend.utils.config import get_settings

settings = get_settings()
router = APIRouter()

_interviewer = InterviewerAgent()
_evaluator   = EvaluatorAgent()
_feedback    = FeedbackAgent()
_reporter    = ReportAgent()


async def _send(ws: WebSocket, data: dict):
    await ws.send_text(json.dumps(data))


@router.websocket("/interview")
async def interview_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"[WS] New connection -> session {session_id}")

    state = {
        "session_id": session_id,
        "candidate_name": "Candidate",
        "role": "",
        "interview_type": "mixed",
        "resume_summary": "No resume provided.",
        "question_number": 0,
        "covered_topics": [],
        "qa_history": [],
        "current_question": "",
        "current_topic": "",
    }

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action", "")

            if action == "start":
                state["role"]           = msg.get("role", "Software Engineer")
                state["interview_type"] = msg.get("interview_type", "mixed")
                state["candidate_name"] = msg.get("candidate_name", "Candidate")
                state["resume_summary"] = msg.get("resume_summary", "No resume provided.")
                state["question_number"] = 1

                question, topic = _interviewer.generate_first_question(
                    session_id=session_id, role=state["role"],
                    interview_type=state["interview_type"],
                    resume_summary=state["resume_summary"],
                    max_questions=settings.max_questions,
                )
                state["current_question"] = question
                state["current_topic"] = topic

                await _send(websocket, {
                    "type": "question", "question": question,
                    "question_number": 1, "topic": topic,
                    "is_final": 1 >= settings.max_questions,
                })

            elif action == "answer":
                answer = msg.get("text", "").strip()
                if not answer:
                    await _send(websocket, {"type": "error", "message": "Empty answer"})
                    continue

                q_num = state["question_number"]

                evaluation = _evaluator.evaluate(
                    session_id=session_id, question=state["current_question"],
                    answer=answer, role=state["role"],
                    interview_type=state["interview_type"],
                )
                await _send(websocket, {
                    "type": "evaluation",
                    "score": evaluation["score"],
                    "strengths": evaluation.get("strengths", []),
                    "weaknesses": evaluation.get("weaknesses", []),
                    "ideal": evaluation.get("ideal_answer_outline", ""),
                })

                feedback_text = _feedback.generate_feedback(
                    session_id=session_id, question=state["current_question"],
                    answer=answer, evaluation=evaluation, role=state["role"],
                    resume_summary=state["resume_summary"], topic=state["current_topic"],
                )
                await _send(websocket, {"type": "feedback", "text": feedback_text})

                if state["current_topic"] not in state["covered_topics"]:
                    state["covered_topics"].append(state["current_topic"])

                state["qa_history"].append({
                    "question_number": q_num,
                    "question": state["current_question"],
                    "answer": answer,
                    "topic": state["current_topic"],
                    **evaluation,
                    "feedback": feedback_text,
                })

                next_q_num = q_num + 1
                if next_q_num > settings.max_questions:
                    summary, pdf_path, avg_score = _reporter.compile_report(
                        session_id=session_id, candidate_name=state["candidate_name"],
                        role=state["role"], interview_type=state["interview_type"],
                        qa_pairs=state["qa_history"],
                    )
                    pdf_filename = pdf_path.split("/")[-1] if pdf_path else ""
                    await _send(websocket, {
                        "type": "complete", "avg_score": avg_score,
                        "report_url": f"/reports/{pdf_filename}",
                        "executive_summary": summary,
                    })
                else:
                    is_final = next_q_num >= settings.max_questions
                    next_q, next_topic = _interviewer.generate_next_question(
                        session_id=session_id, role=state["role"],
                        interview_type=state["interview_type"],
                        resume_summary=state["resume_summary"],
                        last_answer=answer, question_number=next_q_num,
                        max_questions=settings.max_questions,
                        covered_topics=state["covered_topics"], is_final=is_final,
                    )
                    state["question_number"] = next_q_num
                    state["current_question"] = next_q
                    state["current_topic"] = next_topic
                    await _send(websocket, {
                        "type": "question", "question": next_q,
                        "question_number": next_q_num,
                        "topic": next_topic, "is_final": is_final,
                    })

            elif action == "finish":
                if state["qa_history"]:
                    summary, pdf_path, avg_score = _reporter.compile_report(
                        session_id=session_id, candidate_name=state["candidate_name"],
                        role=state["role"], interview_type=state["interview_type"],
                        qa_pairs=state["qa_history"],
                    )
                    pdf_filename = pdf_path.split("/")[-1] if pdf_path else ""
                    await _send(websocket, {
                        "type": "complete", "avg_score": avg_score,
                        "report_url": f"/reports/{pdf_filename}",
                        "executive_summary": summary,
                    })
                else:
                    await _send(websocket, {"type": "error", "message": "No answers recorded yet"})

    except WebSocketDisconnect:
        logger.info(f"[WS] Disconnected: {session_id}")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
        try:
            await _send(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass

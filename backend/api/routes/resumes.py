"""
backend/api/routes/resumes.py
Resume ingestion, PII sanitization, Candidate Profile Analysis (CPA-Agent),
and Competency Divergence Delta(C, J) computation.
"""
from __future__ import annotations
import os
import uuid
import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from loguru import logger

from backend.models.database import get_db, InterviewSession
from backend.models.schemas import ResumeUploadResponse
from backend.rag.ingestion import (
    ingest_resume,
    generate_resume_summary,
    extract_candidate_profile,
    compute_competency_divergence,
)
from backend.rag.vectorstore import add_resume_chunks
from backend.api.routes.sessions import _session_context
from backend.utils.config import get_settings
from backend.utils.llm import get_llm

settings = get_settings()
router = APIRouter()
ALLOWED = {".pdf", ".docx", ".doc", ".txt"}
RESUMES_DIR = "./data/resumes"


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Ingest CV, execute PII sanitization (blinding), run CPA-Agent extraction,
    and compute the Competency Divergence Metric Delta(C, J).
    """
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED}")

    os.makedirs(RESUMES_DIR, exist_ok=True)
    saved_path = os.path.join(RESUMES_DIR, f"{session_id}_{uuid.uuid4().hex[:8]}{ext}")

    content = await file.read()
    with open(saved_path, "wb") as f:
        f.write(content)

    # Ingest with automated PII sanitization
    clean_text, chunks, sanitized_text = ingest_resume(saved_path)

    collection_name = f"resume_{session_id}"
    add_resume_chunks(
        collection_name=collection_name,
        chunks=chunks,
        metadata={"session_id": session_id, "filename": file.filename, "role": session.role},
    )

    llm = get_llm(temperature=0.2)

    # CPA-Agent Candidate Profile extraction & Competency Divergence
    candidate_profile = extract_candidate_profile(sanitized_text, llm, role=session.role)
    overall_div, per_skill_div, detected_gaps = compute_competency_divergence(
        candidate_skills=candidate_profile.technical_skills,
        target_role=session.role,
    )
    candidate_profile.detected_competency_gaps = detected_gaps

    summary = generate_resume_summary(sanitized_text, llm)

    # Persist profile and divergence to session record
    session.resume_filename = file.filename
    session.resume_chroma_id = collection_name
    session.candidate_profile = candidate_profile.model_dump_json()
    session.competency_divergence = overall_div
    db.commit()

    # Update in-memory session context
    ctx = _session_context.setdefault(session_id, {
        "question_number": 0,
        "covered_topics": [],
        "competency_gaps": [],
    })
    ctx["resume_summary"] = summary
    ctx["competency_gaps"] = detected_gaps
    ctx["competency_divergence"] = overall_div
    ctx["candidate_profile"] = candidate_profile.model_dump()

    logger.info(
        f"[CPA-Agent] CV Analyzed: {len(candidate_profile.technical_skills)} skills extracted, "
        f"Divergence Delta(C, J) = {overall_div:.3f}, Gaps: {detected_gaps}"
    )

    return ResumeUploadResponse(
        session_id=session_id,
        filename=file.filename,
        chroma_collection_id=collection_name,
        summary=summary,
        message=f"Resume processed with PII blinding — {len(chunks)} chunks indexed.",
        extracted_skills=candidate_profile.technical_skills,
        competency_divergence=overall_div,
        detected_gaps=detected_gaps,
        pii_sanitized=True,
    )

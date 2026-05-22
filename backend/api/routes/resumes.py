from __future__ import annotations
import os, uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from loguru import logger
from backend.models.database import get_db, InterviewSession
from backend.models.schemas import ResumeUploadResponse
from backend.rag.ingestion import ingest_resume, generate_resume_summary
from backend.rag.vectorstore import add_resume_chunks
from backend.api.routes.sessions import _session_context
from backend.utils.config import get_settings
from langchain_groq import ChatGroq

settings = get_settings()
router = APIRouter()
ALLOWED = {".pdf", ".docx", ".doc", ".txt"}
RESUMES_DIR = "./data/resumes"


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(session_id: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'")
    os.makedirs(RESUMES_DIR, exist_ok=True)
    saved_path = os.path.join(RESUMES_DIR, f"{session_id}_{uuid.uuid4().hex[:8]}{ext}")
    content = await file.read()
    with open(saved_path, "wb") as f:
        f.write(content)
    full_text, chunks = ingest_resume(saved_path)
    collection_name = f"resume_{session_id}"
    add_resume_chunks(collection_name=collection_name, chunks=chunks,
                      metadata={"session_id": session_id, "filename": file.filename, "role": session.role})
    llm = ChatGroq(model=settings.llm_model, temperature=0.2, groq_api_key=settings.groq_api_key)
    summary = generate_resume_summary(full_text, llm)
    session.resume_filename = file.filename
    session.resume_chroma_id = collection_name
    db.commit()
    ctx = _session_context.setdefault(session_id, {"question_number": 0, "covered_topics": []})
    ctx["resume_summary"] = summary
    return ResumeUploadResponse(
        session_id=session_id, filename=file.filename,
        chroma_collection_id=collection_name, summary=summary,
        message=f"Resume processed — {len(chunks)} chunks indexed.",
    )

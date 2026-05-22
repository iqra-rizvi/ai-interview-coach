from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple
from loguru import logger


def _extract_pdf(path):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)

def _extract_docx(path):
    from docx import Document
    return "\n".join(p.text for p in Document(path).paragraphs)

def _extract_txt(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def extract_text(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf": return _extract_pdf(file_path)
    elif ext in (".docx", ".doc"): return _extract_docx(file_path)
    elif ext == ".txt": return _extract_txt(file_path)
    else: raise ValueError(f"Unsupported: {ext}")

def clean_text(text):
    return re.sub(r"[^\x20-\x7E\n]", "", re.sub(r"\s+", " ", text)).strip()

def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words): break
        start = end - overlap
    return chunks

def ingest_resume(file_path):
    raw = extract_text(file_path)
    clean = clean_text(raw)
    chunks = chunk_text(clean)
    logger.info(f"Resume: {len(chunks)} chunks")
    return clean, chunks

def generate_resume_summary(full_text, llm):
    from langchain_core.messages import HumanMessage
    prompt = (
        "Summarize this resume in 3-5 sentences: years of experience, "
        "core skills, key projects, target role suitability.\n\n"
        f"RESUME:\n{full_text[:3000]}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content if hasattr(response, "content") else str(response)

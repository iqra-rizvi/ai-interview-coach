"""
backend/rag/ingestion.py
Document ingestion, PII sanitization (blinding), entity extraction,
and semantic Competency Divergence modeling Delta(C, J) as specified in AgentHire Section 3.2.
"""
from __future__ import annotations
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from loguru import logger
from backend.models.schemas import CandidateProfile
from backend.rag.embeddings import get_embedding_function


# ─── PII Sanitization & Blinding (Section 7.2) ───────────────────────────────

def sanitize_pii(text: str) -> str:
    """
    Purge personally identifiable information (PII) to enforce demographic
    parity and algorithmic fairness aligned with EU AI Act Title III Article 6.
    """
    # Emails
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", text)
    # Phone numbers
    text = re.sub(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[REDACTED_PHONE]", text)
    # URLs / LinkedIn / GitHub
    text = re.sub(r"https?://\S+|www\.\S+", "[REDACTED_LINK]", text)
    # Graduation years (proxy for age) e.g., 2012-2016, Class of 2024
    text = re.sub(r"\b(19\d{2}|20\d{2})\s*[-–—to]+\s*(19\d{2}|20\d{2})\b", "[GRADUATION_YEARS_ACCREDITED]", text)
    # Common university patterns
    text = re.sub(r"\b(?:University|Institute|College|Academy|Polytechnic)\s+of\s+[A-Za-z\s]+", "[UNIVERSITY_ACCREDITED]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Za-z\s]+\s+(?:University|Institute|College)\b", "[UNIVERSITY_ACCREDITED]", text, flags=re.IGNORECASE)
    return text


# ─── Raw text extraction ─────────────────────────────────────────────────────

def _extract_pdf(path: str) -> str:
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


def _extract_docx(path: str) -> str:
    from docx import Document
    return "\n".join(p.text for p in Document(path).paragraphs)


def _extract_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return _extract_docx(file_path)
    elif ext == ".txt":
        return _extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported document format: {ext}")


def clean_text(text: str) -> str:
    # Retain standard ASCII printable characters and newlines
    cleaned = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def ingest_resume(file_path: str) -> Tuple[str, List[str], str]:
    """
    Ingest, clean, sanitize PII, and chunk resume.
    Returns: (clean_text, chunks, sanitized_text)
    """
    raw = extract_text(file_path)
    clean = clean_text(raw)
    sanitized = sanitize_pii(clean)
    chunks = chunk_text(sanitized)
    logger.info(f"Ingested document '{Path(file_path).name}': {len(chunks)} chunks, PII sanitized.")
    return clean, chunks, sanitized


# ─── Competency Divergence Metric Delta(C, J) (Section 3.2, Eq 3.1) ──────────

ROLE_BENCHMARK_SKILLS: Dict[str, List[str]] = {
    "Software Engineer": [
        "Data Structures & Algorithms",
        "System Design & Microservices",
        "Object-Oriented Programming (OOP)",
        "Relational Databases & SQL Optimization",
        "REST APIs & Web Frameworks",
        "Version Control (Git) & CI/CD",
        "Concurrency & Multithreading",
        "Unit Testing & Integration Testing",
    ],
    "Machine Learning Engineer": [
        "Supervised & Unsupervised Learning",
        "Deep Learning (PyTorch / TensorFlow)",
        "Model Evaluation under Class Imbalance",
        "Feature Engineering & Selection",
        "Cost-Sensitive Loss Formulations",
        "Model Deployment & Inference Optimization",
        "Vector Databases & Embeddings",
        "Distributed Training & GPU Acceleration",
    ],
    "Data Scientist": [
        "Statistical Inference & Hypothesis Testing",
        "Machine Learning Algorithms",
        "Exploratory Data Analysis (Pandas, NumPy)",
        "A/B Testing & Metric Design",
        "Data Visualization & Storytelling",
        "Feature Engineering",
        "SQL Query Optimization",
        "Business Metric Translation",
    ],
    "DevOps Engineer": [
        "CI/CD Pipeline Automation",
        "Containerization (Docker)",
        "Container Orchestration (Kubernetes)",
        "Infrastructure as Code (Terraform)",
        "Cloud Platforms (AWS / GCP / Azure)",
        "Observability & Monitoring (Prometheus, Grafana)",
        "Linux Systems Administration & Shell Scripting",
        "Security & Secret Management",
    ],
    "Product Manager": [
        "Product Discovery & User Research",
        "Roadmap Prioritization Frameworks",
        "KPI Definition & Funnel Analytics",
        "Stakeholder & Engineering Alignment",
        "Agile / Scrum Methodologies",
        "Market Research & Competitive Analysis",
        "Go-To-Market (GTM) Strategy",
        "Feature Specification (PRD Writing)",
    ],
}


def compute_competency_divergence(
    candidate_skills: List[str],
    target_role: str,
    custom_requirements: List[str] | None = None,
) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Computes Competency Divergence Metric Delta(C, J):
      Delta(C, J) = 1 - (1 / |R_core|) * sum_{r in R_core} max_{s in S_tech} CosineSim(e_s, e_r)
    Returns:
      overall_divergence: float in [0, 1]
      per_skill_divergence: Dict[str, float]
      critical_gaps: List[str] (skills where Delta >= 0.35)
    """
    required_skills = custom_requirements or ROLE_BENCHMARK_SKILLS.get(
        target_role, ROLE_BENCHMARK_SKILLS["Software Engineer"]
    )
    if not candidate_skills:
        # Complete divergence if no skills recognized
        return 1.0, {r: 1.0 for r in required_skills}, required_skills

    embed_fn = get_embedding_function()
    try:
        cand_embeds = np.array(embed_fn.embed_documents(candidate_skills))
        req_embeds = np.array(embed_fn.embed_documents(required_skills))

        # Normalize embedding vectors to unit length
        cand_norm = cand_embeds / np.linalg.norm(cand_embeds, axis=1, keepdims=True)
        req_norm = req_embeds / np.linalg.norm(req_embeds, axis=1, keepdims=True)

        # Cosine similarity matrix: shape (|required|, |candidate|)
        sim_matrix = np.matmul(req_norm, cand_norm.T)

        per_skill_divergence = {}
        critical_gaps = []
        div_sum = 0.0

        for idx, skill in enumerate(required_skills):
            max_sim = float(np.max(sim_matrix[idx]))
            # Cosine similarity clip to [-1, 1], mapped to divergence in [0, 1]
            clamped_sim = max(0.0, min(1.0, max_sim))
            divergence = round(1.0 - clamped_sim, 3)
            per_skill_divergence[skill] = divergence
            div_sum += divergence
            if divergence >= 0.35:
                critical_gaps.append(skill)

        overall_divergence = round(div_sum / len(required_skills), 3)
        return overall_divergence, per_skill_divergence, critical_gaps
    except Exception as e:
        logger.warning(f"Error computing divergence matrix: {e}. Falling back to keyword matching.")
        cand_lower = {s.lower() for s in candidate_skills}
        per_skill = {}
        gaps = []
        for r in required_skills:
            matched = any(w in " ".join(cand_lower) for w in r.lower().split() if len(w) > 3)
            score = 0.1 if matched else 0.8
            per_skill[r] = score
            if score > 0.35:
                gaps.append(r)
        return round(len(gaps) / max(len(required_skills), 1), 3), per_skill, gaps


# ─── Candidate Profile Extraction ────────────────────────────────────────────

def extract_candidate_profile(sanitized_text: str, llm, role: str = "Software Engineer") -> CandidateProfile:
    """
    Extract structured candidate profile C = <S_tech, E_exp, P_proj> from sanitized CV.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    sys_prompt = (
        "You are an expert technical resume parser for the CPA-Agent. "
        "Extract structured candidate information from the sanitized resume. "
        "Return valid JSON ONLY without markdown fences with these exact keys:\n"
        "{\n"
        '  "technical_skills": ["skill1", "skill2", ...],\n'
        '  "experience_years": <float>,\n'
        '  "verified_projects": [{"title": "Project Name", "description": "Short summary", "stack": "Python, ..."}],\n'
        '  "detected_competency_gaps": ["skill that seems missing for role"]\n'
        "}"
    )
    user_prompt = f"Target Role: {role}\n\nRESUME TEXT:\n{sanitized_text[:3500]}"

    try:
        from backend.utils.llm import safe_llm_invoke
        response = safe_llm_invoke(llm, [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        raw_content = response.content if hasattr(response, "content") else str(response)

        # Parse JSON
        cleaned = re.sub(r"```(?:json)?", "", raw_content).strip().strip("`").strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return CandidateProfile(
                candidate_id="CAND-" + re.sub(r"\D", "", sanitized_text[:100])[:6] or "CAND-001",
                technical_skills=data.get("technical_skills", []),
                experience_years=float(data.get("experience_years", 0.0)),
                verified_projects=data.get("verified_projects", []),
                detected_competency_gaps=data.get("detected_competency_gaps", []),
                sanitized_cv_text=sanitized_text[:2000],
            )
    except Exception as e:
        logger.warning(f"Profile extraction LLM parse exception: {e}")

    # Fallback heuristic extraction
    heuristic_skills = re.findall(
        r"\b(Python|Java|C\+\+|SQL|Docker|Kubernetes|PyTorch|TensorFlow|React|FastAPI|Git|AWS|Linux|Pandas|Scikit-learn)\b",
        sanitized_text,
        re.IGNORECASE,
    )
    unique_skills = sorted(list({s.title() for s in heuristic_skills}))
    return CandidateProfile(
        candidate_id="CAND-DEFAULT",
        technical_skills=unique_skills,
        experience_years=1.0,
        verified_projects=[],
        detected_competency_gaps=[],
        sanitized_cv_text=sanitized_text[:2000],
    )


def generate_resume_summary(full_text: str, llm) -> str:
    from langchain_core.messages import HumanMessage
    from backend.utils.llm import safe_llm_invoke

    prompt = (
        "Summarize this candidate profile in 3-5 concise, objective sentences: "
        "years of experience, core technical stack, key architectures built, and suitability.\n\n"
        f"CANDIDATE PROFILE:\n{full_text[:3000]}"
    )
    try:
        response = safe_llm_invoke(llm, [HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"Resume summary error: {e}")
        return "Experienced technical candidate with verified background in software engineering."

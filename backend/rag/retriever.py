from __future__ import annotations
from typing import List
from loguru import logger
from backend.rag.vectorstore import query_collection

KB_COLLECTION = "interview_knowledge_base"


def _merge(lists: List[List[str]], max_total: int = 6) -> str:
    seen, merged = set(), []
    for lst in lists:
        for item in lst:
            key = item[:100]
            if key not in seen:
                seen.add(key)
                merged.append(item)
    return "\n\n---\n\n".join(merged[:max_total]) or "No relevant context found."


def retrieve_for_question_generation(session_id, role, topic, interview_type, n=3) -> str:
    query = f"Interview questions for {role} about {topic} ({interview_type})"
    resume_docs = query_collection(f"resume_{session_id}", query, n)
    kb_docs = query_collection(KB_COLLECTION, query, n)
    return _merge([resume_docs, kb_docs])


def retrieve_for_evaluation(session_id, question, role, n=3) -> str:
    query = f"Ideal answer evaluation criteria: {question} for {role}"
    kb_docs = query_collection(KB_COLLECTION, query, n)
    return _merge([kb_docs])


def retrieve_resume_context(session_id, role, topic, n=4) -> str:
    query = f"Skills and experience relevant to {role} {topic}"
    resume_docs = query_collection(f"resume_{session_id}", query, n)
    return _merge([resume_docs])

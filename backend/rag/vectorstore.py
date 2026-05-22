"""
backend/rag/vectorstore.py
FAISS-based vector store — no C++ compilation needed.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List
from loguru import logger
from backend.utils.config import get_settings

settings = get_settings()
STORE_DIR = Path(settings.chroma_persist_dir)
STORE_DIR.mkdir(parents=True, exist_ok=True)

_index_cache: dict = {}


def _get_embeddings():
    from backend.rag.embeddings import get_embedding_function
    return get_embedding_function()


def add_resume_chunks(collection_name: str, chunks: List[str], metadata: dict = None) -> int:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    embed_fn = _get_embeddings()
    meta = metadata or {}
    docs = [Document(page_content=chunk, metadata=meta) for chunk in chunks]

    try:
        if collection_name in _index_cache:
            _index_cache[collection_name].add_documents(docs)
        else:
            save_path = str(STORE_DIR / collection_name)
            if os.path.exists(save_path):
                index = FAISS.load_local(save_path, embed_fn, allow_dangerous_deserialization=True)
                index.add_documents(docs)
            else:
                index = FAISS.from_documents(docs, embed_fn)
            _index_cache[collection_name] = index

        _index_cache[collection_name].save_local(str(STORE_DIR / collection_name))
        logger.info(f"Stored {len(chunks)} chunks in '{collection_name}'")
    except Exception as e:
        logger.error(f"Error storing chunks: {e}")
        raise
    return len(chunks)


def add_knowledge_base_docs(chunks: List[str], source: str,
                             collection_name: str = "interview_knowledge_base") -> int:
    return add_resume_chunks(collection_name, chunks, metadata={"source": source})


def query_collection(collection_name: str, query: str, n_results: int = 5) -> List[str]:
    from langchain_community.vectorstores import FAISS
    try:
        embed_fn = _get_embeddings()
        save_path = str(STORE_DIR / collection_name)

        if collection_name not in _index_cache:
            if not os.path.exists(save_path):
                return []
            index = FAISS.load_local(save_path, embed_fn, allow_dangerous_deserialization=True)
            _index_cache[collection_name] = index

        results = _index_cache[collection_name].similarity_search(query, k=min(n_results, 5))
        return [doc.page_content for doc in results]
    except Exception as e:
        logger.warning(f"Query failed for '{collection_name}': {e}")
        return []


def delete_collection(collection_name: str) -> None:
    _index_cache.pop(collection_name, None)
    logger.info(f"Deleted collection '{collection_name}' from cache")

from functools import lru_cache
from loguru import logger


@lru_cache(maxsize=1)
def get_embedding_function():
    logger.info("Loading local embedding model: all-MiniLM-L6-v2")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

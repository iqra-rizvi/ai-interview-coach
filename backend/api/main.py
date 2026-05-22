"""
backend/api/main.py
FastAPI application — registers routers, handles startup/shutdown,
configures CORS, and serves health check.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import os

from backend.models.database import init_db
from backend.api.routes import sessions, resumes, reports
from backend.api.websocket import router as ws_router
from backend.utils.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting AI Interview Coach API...")

    # Ensure data directories exist
    for directory in ["./data/resumes", "./data/reports", "./data/vectordb"]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ready: {directory}")

    # Initialize SQLite database
    init_db()
    logger.info("Database initialized")

    logger.info(f"LLM model: {settings.llm_model}")
    logger.info(f"Embedding backend: {settings.embedding_backend}")
    logger.info("API ready ✓")

    yield  # Application running

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down AI Interview Coach API...")


app = FastAPI(
    title="AI Interview Coach",
    description=(
        "AI-powered mock interview platform with RAG, multi-agent evaluation, "
        "real-time feedback, and PDF performance reports."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files — serve generated PDF reports for download
app.mount("/reports", StaticFiles(directory="./data/reports"), name="reports")

# Routers
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])
app.include_router(resumes.router,  prefix="/api/resumes",  tags=["Resumes"])
app.include_router(reports.router,  prefix="/api/reports",  tags=["Reports"])
app.include_router(ws_router,       prefix="/ws",           tags=["WebSocket"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": "AI Interview Coach",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "model": settings.llm_model}

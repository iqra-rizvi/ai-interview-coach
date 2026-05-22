"""
backend/models/database.py
SQLAlchemy ORM models — Users, Sessions, Questions, Answers.
"""
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime,
    ForeignKey, Boolean, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from backend.utils.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)  # UUID
    name = Column(String(200), nullable=False)
    email = Column(String(300), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("InterviewSession", back_populates="user", cascade="all, delete")


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    role = Column(String(200), nullable=False)
    interview_type = Column(String(50), nullable=False)  # technical | hr | mixed
    status = Column(String(30), default="active")        # active | completed | abandoned
    avg_score = Column(Float, nullable=True)
    resume_filename = Column(String(500), nullable=True)
    resume_chroma_id = Column(String(200), nullable=True)  # ChromaDB collection name
    report_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")
    qa_pairs = relationship("QAPair", back_populates="session", cascade="all, delete")


class QAPair(Base):
    __tablename__ = "qa_pairs"

    id = Column(String(36), primary_key=True)  # UUID
    session_id = Column(String(36), ForeignKey("interview_sessions.id"), nullable=False)
    question_number = Column(Integer, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    score = Column(Float, nullable=True)          # 0-10
    strengths = Column(Text, nullable=True)       # JSON list stored as string
    weaknesses = Column(Text, nullable=True)      # JSON list stored as string
    feedback = Column(Text, nullable=True)
    ideal_answer_outline = Column(Text, nullable=True)
    topic = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("InterviewSession", back_populates="qa_pairs")


# ─── Engine & Session factory ───────────────────────────────────────────────

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # SQLite only
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables. Call on startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and ensures it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

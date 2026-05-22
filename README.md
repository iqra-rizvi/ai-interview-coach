# 🤖 AI Interview Coach

An AI-powered interview preparation platform that simulates technical and HR interviews with personalized, real-time feedback — built with **RAG**, **multi-agent AI (LangGraph)**, **FastAPI**, and **Streamlit**.

> 💡 **100% Free** — Uses Groq (free LLM API) + local sentence-transformers (free embeddings). No credit card needed.

---

## 🏗️ Architecture

```
User (Streamlit UI)
        │
        ▼
FastAPI Orchestrator  ←→  WebSocket streaming
        │
        ▼
LangGraph Multi-Agent System
  ├── InterviewerAgent   — generates contextual questions
  ├── EvaluatorAgent     — scores answers (0-10) with rubric
  ├── FeedbackAgent      — personalized improvement tips
  └── ReportAgent        — compiles full PDF session report
        │
        ▼
RAG Pipeline (LangChain + ChromaDB)
  ├── Resume ingestion & chunking
  ├── Local sentence-transformers embeddings (FREE)
  ├── ChromaDB vector store
  └── Semantic retriever (top-k)
        │
        ▼
Data Stores
  ├── SQLite   — users, sessions, scores
  ├── ChromaDB — resume + Q&A vectors
  └── File system — uploads, PDF reports
```

---

## 📁 Project Structure

```
ai-interview-coach/
├── backend/
│   ├── agents/
│   │   ├── interviewer_agent.py
│   │   ├── evaluator_agent.py
│   │   ├── feedback_agent.py
│   │   ├── report_agent.py
│   │   └── graph.py
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   └── retriever.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── sessions.py
│   │   │   ├── resumes.py
│   │   │   └── reports.py
│   │   └── websocket.py
│   ├── models/
│   │   ├── database.py
│   │   └── schemas.py
│   └── utils/
│       ├── pdf_generator.py
│       ├── prompt_templates.py
│       └── config.py
├── frontend/
│   └── app.py
├── data/
│   ├── resumes/
│   ├── reports/
│   └── vectordb/
├── tests/
│   └── test_agents.py
├── requirements.txt
├── .env.example
├── run.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Get a FREE Groq API key
- Go to **https://console.groq.com**
- Sign up (free, no credit card)
- Click **API Keys → Create API Key**
- Copy the key (starts with `gsk_...`)

### 2. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/ai-interview-coach.git
cd ai-interview-coach
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Configure
```bash
copy .env.example .env
notepad .env
```
Set your key:
```
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=llama3-8b-8192
```

### 4. Run
```bash
python run.py
```

Open **http://localhost:8501** for the UI.
API docs at **http://localhost:8000/docs**.

---

## ✨ Features

| Feature | Tech |
|---|---|
| Resume-aware questions | RAG + ChromaDB |
| Technical & HR modes | Prompt engineering |
| Real-time feedback | FastAPI + Streamlit |
| Answer scoring (0–10) | EvaluatorAgent |
| Coaching suggestions | FeedbackAgent |
| Full PDF report | ReportAgent + ReportLab |
| Session history | SQLite |
| 100% Free | Groq + local embeddings |

---

## 🛠️ Tech Stack

- **LangChain + LangGraph** — multi-agent orchestration
- **Groq (llama3-8b-8192)** — free LLM
- **ChromaDB** — vector store
- **sentence-transformers** — free local embeddings
- **FastAPI** — REST API + WebSocket
- **Streamlit** — frontend UI
- **SQLAlchemy** — SQLite database
- **ReportLab** — PDF reports

---

## 📄 License
MIT

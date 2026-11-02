# 🤖 AgentHire: Multi-Agent Intelligent Interview Preparation Platform

**AgentHire: An Agentic AI-Based Intelligent Interview Preparation and Candidate Evaluation Platform**
*B.Tech Computer Science & Engineering Minor Project Report Implementation*
*Author: Iqra Rizvi · Amity University Uttar Pradesh (ASET Lucknow)*

---

## 🏛️ System Architecture

AgentHire decomposes the technical assessment lifecycle into **five discrete, finite-state agents** orchestrated via a stateful computation graph in **LangGraph**:

```
Candidate CV + Target Job Description
                │
                ▼
  [1. CPA-Agent] Candidate Profile Analysis
      ├── PII Sanitization & Blinding (EU AI Act Title III Art. 6 / IEEE compliant)
      ├── Skills Entity Extraction: C = <S_tech, E_exp, P_proj>
      └── Competency Divergence Calculation: Δ(C, J) via Cosine Distance
                │
                ▼
  [2. DQG-Agent] Dynamic Question Generation
      ├── Tier 1: Foundational Theoretical Mechanics (λ in [1, 5])
      ├── Tier 2: Applied Systems Design & Scalability
      └── Tier 3: Situational Engineering Problem Solving (STAR)
                │
                ▼
  [3. AIS-Agent] Adaptive Interview Simulation
      ├── Stateful turn-taking dialogue loop & sliding context window
      └── Adaptive Probing Condition: (c_1 >= 3) AND (c_4 <= 3) AND (not is_followup)
                │
                ▼
  [4. RAE-Agent] Rubric-Anchored Response Evaluation
      ├── 5 Orthogonal Dimensions: Accuracy (c_1), Relevance (c_2), Depth (c_3), Completeness (c_4), Problem Solving (c_5)
      └── Verbatim Evidence Verification Gate: V(τ_k, A) = 1.0 if (τ_k in A and len >= 15) else 0.0
                │
                ▼
  [5. DFP-Agent] Diagnostic Feedback & Pedagogy
      ├── 5-Axis Competency Radar Computation (Plotly)
      ├── Verified Strengths & Technical Blind Spots
      └── Curated Pedagogical Remediation Curriculum & Exportable PDF Dossier
```

---

## 🛠️ Tech Stack & Prerequisites

- **Language & Runtime:** Python 3.11+ / 3.12+ (64-bit)
- **Agent Orchestrator:** LangGraph 0.2.28 + LangChain 0.3.7
- **Inference Backbone:** Groq API (`openai/gpt-oss-20b` or `qwen/qwen3.8-27b`) with automatic fallback resilience
- **Local Embedding Vectorizer:** `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dimensional dense vectors)
- **Vector Index:** FAISS (pre-compiled Windows CPU engine)
- **Backend Microservice:** FastAPI 0.115 + Uvicorn + WebSockets + Pydantic v2
- **Persistence:** SQLite relational database with SQLAlchemy ORM
- **Frontend User Experience:** Streamlit 1.39 + Plotly 7.1 (interactive polar radar & progress telemetry)
- **Report Engine:** ReportLab 4.2 with XML-sanitization and auditable evidentiary tables

---

## 🚀 One-Command Launch (Presentation Mode)

### Step 1: Virtual Environment
Ensure your virtual environment is activated:
```powershell
.\venv\Scripts\activate
```

### Step 2: Configure Environment
Confirm `.env` has your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_key_here
LLM_MODEL=openai/gpt-oss-20b
APP_HOST=0.0.0.0
APP_PORT=8000
API_BASE_URL=http://localhost:8000
```

### Step 3: Launch Both Services
Run the launcher:
```powershell
python run.py
```
- **Streamlit Interactive UI:** [http://localhost:8501](http://localhost:8501)
- **FastAPI OpenAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 Running the Verification Test Suite

AgentHire includes an auditable 11-test suite validating the evidence gate, adaptive probing, PII blinding, divergence computation, and end-to-end API lifecycle:

```powershell
.\venv\Scripts\python -m pytest tests/ -v
```

Expected output:
```
tests/test_agents.py::TestEvidenceVerificationGate::test_verbatim_quote_verified PASSED
tests/test_agents.py::TestEvidenceVerificationGate::test_fabricated_quote_invalidated PASSED
tests/test_agents.py::TestEvidenceVerificationGate::test_short_quote_below_threshold_invalidated PASSED
tests/test_agents.py::TestEvaluatorAgent::test_5d_rubric_evaluation PASSED
tests/test_agents.py::TestEvaluatorAgent::test_empty_answer_scores_zero PASSED
tests/test_agents.py::TestCompetencyDivergenceAndPII::test_pii_sanitization PASSED
tests/test_agents.py::TestCompetencyDivergenceAndPII::test_divergence_computation_basic PASSED
tests/test_agents.py::TestAdaptiveInterviewer::test_adaptive_probe_synthesis PASSED
tests/test_agents.py::TestPDFGenerator::test_pdf_with_special_characters_no_crash PASSED
tests/test_agents.py::TestPDFGenerator::test_pdf_empty_qa_pairs_no_crash PASSED
tests/test_agents.py::TestFastAPIIntegration::test_full_session_flow PASSED
======================= 11 passed in ~30s =======================
```

---

## 🎯 Live Presentation Demo Script (Step-by-Step)

During your evaluation / viva, follow this 4-minute demonstration flow:

1. **Mode 1: Profile Calibration & CPA-Agent Blinding**
   - Open [http://localhost:8501](http://localhost:8501).
   - Enter your name and select target role: **Machine Learning Engineer**.
   - Upload any sample resume from `data/resumes/` (or your own).
   - Point out the **PII Blinding Chip** (`🛡️ PII Stripped: Compliant with EU AI Act`) and the **Competency Divergence Metric $\Delta(C, J)$** gauge. Show how skills are extracted and critical gaps are highlighted.
   - Click **"Initialize Multi-Agent Assessment Session"**.

2. **Mode 2: Conversational Simulation & Adaptive Probing**
   - Show how the **DQG-Agent** generates a **Tier 1: Foundational** item with calibrated difficulty $\lambda$.
   - Submit an answer that is technically accurate but incomplete (e.g., mention weighted cross-entropy for imbalanced data without explaining focal loss).
   - Highlight the **Adaptive Boundary Probing Alert**:
     `🎯 Adaptive Probing Condition Activated: (c_1 >= 3) AND (c_4 <= 3)`.
   - Expand the **5-Dimensional Psychometric Rubric** card to show the verbatim quotes verifying your answer with zero hallucination.

3. **Mode 3: Diagnostic Pedagogical Dashboard**
   - Click **"Conclude Session"** to synthesize the report.
   - Show the **Plotly 5-Axis Competency Radar Chart** mapping your strength profile across Technical Accuracy, Relevance, Depth, Completeness, and Problem Solving.
   - Review the **Curated Remediation Curriculum** referencing foundational literature (e.g. Lin et al. Focal Loss, Vaswani et al. Attention).
   - Click **"Download Official Performance Report (PDF)"** to open the generated ReportLab PDF dossier.

4. **Mode 4: Scientific Benchmark Matrix**
   - Click **"Benchmark & Validation Matrix"** on the sidebar.
   - Walk through Table 6.1 and 6.2 from your project report showing ICC inter-rater reliability agreement ($0.84$ vs $0.51$) and scoring determinism ($\sigma^2 \le 0.05$).

---

## 🌐 24/7 Cloud Deployment Guide

AgentHire includes pre-configured deployment manifests and an automatic background API initiator, allowing 24/7 zero-downtime hosting on any major cloud platform.

### Option 1: Streamlit Community Cloud (Recommended — 100% Free 24/7)
1. Push this repository to your GitHub account:
   ```bash
   git add .
   git commit -m "Configure 24/7 cloud deployment manifests"
   git push origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New App"** and configure:
   - **Repository:** `iqra-rizvi/ai-interview-coach`
   - **Branch:** `main`
   - **Main file path:** `frontend/app.py`
4. Expand **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   LLM_MODEL = "openai/gpt-oss-20b"
   ```
5. Click **"Deploy"**. AgentHire's automatic background microservice manager will boot the FastAPI engine internally and expose the live application 24/7 at `https://<your-app-name>.streamlit.app`.

---

### Option 2: Render.com (1-Click Blueprint — Free 24/7 Web Service)
1. Sign up / Log in to [render.com](https://render.com).
2. Click **New +** → **Blueprint**.
3. Select your GitHub repository (`ai-interview-coach`).
4. Render will automatically detect `render.yaml`.
5. Enter your `GROQ_API_KEY` when prompted in the environment variables form.
6. Click **Apply**. Render will run `deploy_runner.py` and provide a permanent HTTPS URL (`https://agenthire.onrender.com`).

---

### Option 3: Railway.app
1. Go to [railway.app](https://railway.app) and create a new project.
2. Select **"Deploy from GitHub repo"** → `ai-interview-coach`.
3. In the project **Variables** tab, add:
   - `GROQ_API_KEY` = `gsk_your_groq_key`
4. In **Settings → Networking**, click **Generate Domain**.
5. Railway uses `railway.json` and `Procfile` to run both services 24/7.

---

### Option 4: Docker / Cloud VPS (AWS, GCP, DigitalOcean, Hetzner)
Run locally or on any cloud server with a single command:
```bash
docker compose up -d --build
```
Access the application on port `8501`.


"""
frontend/app.py
Streamlit UI for AI Interview Coach.

Pages:
  🏠 Home        — role/type selection, resume upload
  🎤 Interview   — live mock interview chat
  📊 Report      — performance report with scores
  📁 History     — past sessions
"""
import streamlit as st
import requests
import uuid
import time
import os
from pathlib import Path

API_URL = os.getenv("API_BASE_URL", "http://localhost:8080")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d9488 100%);
        padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
        text-align: center; color: white;
    }
    .score-card {
        background: #1e293b; border-radius: 12px; padding: 1.5rem;
        border-left: 4px solid #0d9488; margin: 0.5rem 0;
    }
    .question-box {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
    }
    .feedback-box {
        background: #0f2d1f; border-left: 4px solid #16a34a;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
    .score-high  { color: #4ade80; font-weight: bold; font-size: 1.4rem; }
    .score-mid   { color: #fbbf24; font-weight: bold; font-size: 1.4rem; }
    .score-low   { color: #f87171; font-weight: bold; font-size: 1.4rem; }
    div[data-testid="stChatMessage"] { background: #1e293b; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "user_id": str(uuid.uuid4()),
        "candidate_name": "Candidate",
        "session_id": None,
        "page": "home",
        "role": "Software Engineer",
        "interview_type": "mixed",
        "resume_uploaded": False,
        "resume_summary": "",
        "qa_history": [],      # List of {question, answer, score, feedback, ...}
        "current_question": None,
        "current_topic": None,
        "question_number": 0,
        "session_complete": False,
        "report_data": None,
        "chat_messages": [],   # For display
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
s = st.session_state


# ─── API helpers ──────────────────────────────────────────────────────────────
def api_post(endpoint: str, **kwargs) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{endpoint}", **kwargs, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def score_color_class(score: float) -> str:
    if score >= 8:
        return "score-high"
    elif score >= 5:
        return "score-mid"
    return "score-low"


def score_emoji(score: float) -> str:
    if score >= 8:   return "🟢"
    elif score >= 5: return "🟡"
    return "🔴"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Interview Coach")
    st.divider()

    if st.button("🏠 Home", use_container_width=True):
        s.page = "home"
    if st.button("🎤 Interview", use_container_width=True, disabled=not s.session_id):
        s.page = "interview"
    if st.button("📊 Report", use_container_width=True, disabled=not s.session_complete):
        s.page = "report"

    st.divider()
    if s.session_id:
        st.caption(f"Session: `{s.session_id[:8]}...`")
        st.caption(f"Role: **{s.role}**")
        st.caption(f"Type: **{s.interview_type.title()}**")
        if s.question_number > 0:
            from backend.utils.config import get_settings
            try:
                max_q = get_settings().max_questions
            except Exception:
                max_q = 10
            st.progress(min(s.question_number / max_q, 1.0), text=f"Q {s.question_number}/{max_q}")

    st.divider()
    st.caption("Built with LangGraph · RAG · FastAPI · Streamlit")


# ═══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if s.page == "home":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Interview Coach</h1>
        <p>Ace your next interview with AI-powered mock sessions, real-time feedback, and personalized reports.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Your Profile")
        s.candidate_name = st.text_input("Your name", value=s.candidate_name)

        st.subheader("🎯 Interview Setup")
        s.role = st.selectbox("Target Role", [
            "Software Engineer",
            "Data Scientist",
            "DevOps Engineer",
            "Product Manager",
            "Machine Learning Engineer",
            "Backend Developer",
            "Frontend Developer",
            "Full Stack Developer",
        ])
        s.interview_type = st.radio(
            "Interview Type",
            ["technical", "hr", "mixed"],
            format_func=lambda x: {
                "technical": "💻 Technical",
                "hr": "🤝 HR / Behavioral",
                "mixed": "⚡ Mixed (Recommended)",
            }[x],
            horizontal=True,
        )

    with col2:
        st.subheader("📄 Upload Your Resume (Optional)")
        st.caption("Uploading your resume lets the AI ask questions tailored to your background.")

        resume_file = st.file_uploader(
            "Upload resume",
            type=["pdf", "docx", "txt"],
            help="PDF, DOCX, or TXT — max 10MB",
        )

        if s.resume_uploaded:
            st.success("✅ Resume indexed successfully!")
            if s.resume_summary:
                with st.expander("Resume Summary"):
                    st.write(s.resume_summary)

        st.subheader("ℹ️ How it works")
        st.markdown("""
        1. **Set up** your role and interview type above
        2. **Upload** your resume for personalized questions *(optional)*
        3. **Start** the mock interview
        4. **Answer** each question in the chat interface
        5. **Receive** instant scores, feedback, and coaching tips
        6. **Download** your full PDF performance report
        """)

    st.divider()

    if st.button("🚀 Start Mock Interview", type="primary", use_container_width=True):
        with st.spinner("Setting up your interview session..."):
            # Create session via API
            result = api_post("/api/sessions/", json={
                "user_id": s.user_id,
                "role": s.role,
                "interview_type": s.interview_type,
            })

            if result:
                s.session_id = result["id"]
                s.qa_history = []
                s.chat_messages = []
                s.question_number = 0
                s.session_complete = False
                s.current_question = None

                # Upload resume if provided
                if resume_file and not s.resume_uploaded:
                    resume_result = api_post(
                        "/api/resumes/upload",
                        data={"session_id": s.session_id},
                        files={"file": (resume_file.name, resume_file.getvalue(), resume_file.type)},
                    )
                    if resume_result:
                        s.resume_uploaded = True
                        s.resume_summary = resume_result.get("summary", "")

                # Start interview (get first question)
                q_result = api_post(f"/api/sessions/{s.session_id}/start")
                if q_result:
                    s.current_question = q_result["question"]
                    s.current_topic = q_result.get("topic", "")
                    s.question_number = 1
                    s.chat_messages.append({
                        "role": "assistant",
                        "content": q_result["question"],
                        "topic": q_result.get("topic", ""),
                        "question_number": 1,
                    })
                    s.page = "interview"
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVIEW PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif s.page == "interview":
    st.markdown(f"## 🎤 Mock Interview — {s.role}")

    if s.session_complete:
        st.success("✅ Interview complete! Check your report.")
        if st.button("📊 View Report", type="primary"):
            s.page = "report"
            st.rerun()
        st.stop()

    # Chat display
    chat_container = st.container()
    with chat_container:
        for msg in s.chat_messages:
            if msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    q_num = msg.get("question_number", "")
                    topic = msg.get("topic", "")
                    if q_num:
                        st.caption(f"Question {q_num} · {topic}")
                    st.markdown(msg["content"])
            elif msg["role"] == "user":
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(msg["content"])
            elif msg["role"] == "feedback":
                with st.container():
                    data = msg.get("data", {})
                    score = data.get("score", 0)
                    emoji = score_emoji(score)

                    col1, col2, col3 = st.columns([1, 2, 2])
                    with col1:
                        st.metric("Score", f"{score}/10", delta=None)
                    with col2:
                        if data.get("strengths"):
                            st.markdown("**✅ Strengths:**")
                            for s_item in data["strengths"][:2]:
                                st.markdown(f"• {s_item}")
                    with col3:
                        if data.get("weaknesses"):
                            st.markdown("**📈 Improve:**")
                            for w in data["weaknesses"][:2]:
                                st.markdown(f"• {w}")

                    if data.get("feedback"):
                        st.markdown(f"""
                        <div class="feedback-box">
                            💬 <strong>Coach:</strong> {data['feedback']}
                        </div>
                        """, unsafe_allow_html=True)
                    st.divider()

    # Answer input
    if not s.session_complete:
        answer = st.chat_input("Type your answer here...", key="answer_input")

        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("⏹ End Early", help="Finish interview and generate report"):
                with st.spinner("Generating your report..."):
                    result = api_post(f"/api/sessions/{s.session_id}/finish")
                    if result:
                        s.session_complete = True
                        s.report_data = result
                        s.page = "report"
                        st.rerun()

        if answer:
            # Add user message to chat
            s.chat_messages.append({"role": "user", "content": answer})

            with st.spinner("Evaluating your answer..."):
                result = api_post(
                    f"/api/sessions/{s.session_id}/answer",
                    json={
                        "session_id": s.session_id,
                        "question_number": s.question_number,
                        "answer": answer,
                    }
                )

            if result:
                eval_data = result.get("evaluation", {})
                feedback_text = result.get("feedback", "")

                # Add feedback card to chat
                s.chat_messages.append({
                    "role": "feedback",
                    "data": {
                        **eval_data,
                        "feedback": feedback_text,
                    }
                })

                # Store in history
                s.qa_history.append({
                    "question": s.current_question,
                    "answer": answer,
                    "score": eval_data.get("score", 0),
                    "strengths": eval_data.get("strengths", []),
                    "weaknesses": eval_data.get("weaknesses", []),
                    "feedback": feedback_text,
                    "topic": s.current_topic,
                })

                if result.get("session_complete"):
                    s.session_complete = True
                    # Auto-generate report
                    report = api_post(f"/api/reports/{s.session_id}/generate")
                    s.report_data = report
                    s.chat_messages.append({
                        "role": "assistant",
                        "content": "🎉 That's a wrap! Your interview is complete. Click **View Report** to see your performance analysis and download your PDF report.",
                    })
                else:
                    next_q = result.get("next_question", {})
                    if next_q:
                        s.current_question = next_q["question"]
                        s.current_topic = next_q.get("topic", "")
                        s.question_number = next_q["question_number"]
                        s.chat_messages.append({
                            "role": "assistant",
                            "content": next_q["question"],
                            "topic": next_q.get("topic", ""),
                            "question_number": next_q["question_number"],
                        })

                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif s.page == "report":
    st.markdown("## 📊 Performance Report")

    report = s.report_data
    if not report and s.session_id:
        with st.spinner("Generating report..."):
            report = api_post(f"/api/reports/{s.session_id}/generate")
            s.report_data = report

    if not report:
        st.warning("No report data available. Complete an interview first.")
        st.stop()

    avg_score = report.get("avg_score", 0)
    total_q = report.get("total_questions", 0)

    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Score", f"{avg_score}/10")
    with col2:
        st.metric("Questions Answered", total_q)
    with col3:
        st.metric("Role", report.get("role", "—"))
    with col4:
        st.metric("Interview Type", report.get("interview_type", "—").title())

    st.divider()

    # Score visualization
    qa_pairs = report.get("qa_pairs", []) or s.qa_history
    if qa_pairs:
        st.subheader("📈 Score Breakdown")
        scores = [qa.get("score", 0) for qa in qa_pairs]
        labels = [f"Q{i+1}" for i in range(len(scores))]

        import json as _json
        chart_data = {
            "Question": labels,
            "Score": scores,
        }

        # Use st.bar_chart
        import pandas as pd
        df = pd.DataFrame(chart_data).set_index("Question")
        st.bar_chart(df, color="#0d9488")

    # Executive Summary
    if report.get("executive_summary"):
        st.subheader("📝 Executive Summary")
        st.info(report["executive_summary"])

    # Q&A Breakdown
    st.subheader("🔍 Detailed Breakdown")
    for i, qa in enumerate(qa_pairs):
        score = qa.get("score", 0)
        with st.expander(
            f"{score_emoji(score)} Q{i+1} — Score: {score}/10 · {qa.get('topic', '')}",
            expanded=(i == 0),
        ):
            st.markdown(f"**Question:** {qa.get('question', '')}")
            st.markdown(f"**Your Answer:** {qa.get('answer', '*No answer recorded*')}")

            col1, col2 = st.columns(2)
            with col1:
                strengths = qa.get("strengths") or []
                if strengths:
                    st.markdown("**✅ Strengths:**")
                    for s_item in strengths:
                        st.markdown(f"• {s_item}")
            with col2:
                weaknesses = qa.get("weaknesses") or []
                if weaknesses:
                    st.markdown("**📈 Areas to Improve:**")
                    for w in weaknesses:
                        st.markdown(f"• {w}")

            if qa.get("ideal_answer_outline"):
                st.markdown(f"**💡 Ideal answer outline:** {qa['ideal_answer_outline']}")

            if qa.get("feedback"):
                st.markdown(f"""
                <div class="feedback-box">
                    💬 <strong>Coach's Feedback:</strong> {qa['feedback']}
                </div>
                """, unsafe_allow_html=True)

    # Download PDF
    st.divider()
    pdf_url = report.get("report_pdf_url")
    if pdf_url:
        full_url = f"{API_URL}{pdf_url}"
        st.markdown(f"### 📥 Download Your Report")
        st.markdown(f"[⬇️ Download PDF Report]({full_url})")
    elif s.session_id:
        if st.button("📥 Download PDF Report", type="primary"):
            pdf_bytes = requests.get(f"{API_URL}/api/reports/{s.session_id}/download").content
            st.download_button(
                "⬇️ Save PDF",
                data=pdf_bytes,
                file_name=f"interview_report_{s.session_id[:8]}.pdf",
                mime="application/pdf",
            )

    # Start new interview
    st.divider()
    if st.button("🔄 Start New Interview", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key != "user_id":
                del st.session_state[key]
        init_state()
        st.rerun()

"""
deploy_runner.py
Universal production 24/7 cloud deployment runner for AgentHire.
Compatible with Render, Railway, Hugging Face Spaces, Fly.io, Heroku, or Docker VPS.

Architecture:
  - Binds FastAPI internally to 127.0.0.1:8000
  - Binds Streamlit publicly to 0.0.0.0:$PORT (default 8501 or 10000)
  - Forwards requests internally with zero public port exposure
"""
import os
import sys
import time
import signal
import subprocess
from pathlib import Path


def main():
    print("=" * 60)
    print("🚀 AgentHire: 24/7 Cloud Deployment Runner Starting...")
    print("=" * 60)

    # Pre-create data directories for persistence
    for p in ["./data/resumes", "./data/reports", "./data/vectordb"]:
        os.makedirs(p, exist_ok=True)

    # Determine assigned cloud ports
    public_port = os.getenv("PORT", os.getenv("SERVER_PORT", "8501"))
    internal_api_port = "8000"

    # Direct Streamlit requests to internal FastAPI
    os.environ["API_BASE_URL"] = f"http://127.0.0.1:{internal_api_port}"

    print(f"✓ Public Web Port  : {public_port}")
    print(f"✓ Internal API Port: {internal_api_port}")
    print(f"✓ Model Endpoint   : {os.getenv('LLM_MODEL', 'openai/gpt-oss-20b')}")

    # 1. Start FastAPI backend
    print("\n[1/2] Starting FastAPI microservice...")
    api_cmd = [
        sys.executable, "-m", "uvicorn", "backend.api.main:app",
        "--host", "127.0.0.1",
        "--port", internal_api_port,
        "--workers", "1"
    ]
    api_proc = subprocess.Popen(api_cmd, env={**os.environ})

    # Wait for backend to be ready
    time.sleep(3)

    # 2. Start Streamlit frontend
    print(f"\n[2/2] Starting Streamlit frontend on port {public_port}...")
    ui_cmd = [
        sys.executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", str(public_port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    ui_proc = subprocess.Popen(ui_cmd, env={**os.environ})

    print("\n✅ AgentHire is live and listening!")
    print(f"🌐 Public URL accessible via your cloud domain on port {public_port}\n")

    def shutdown(sig, frame):
        print("\n[AgentHire] Gracefully shutting down...")
        try:
            ui_proc.terminate()
            api_proc.terminate()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep runner alive while processes run
    try:
        ui_proc.wait()
    finally:
        shutdown(None, None)


if __name__ == "__main__":
    main()

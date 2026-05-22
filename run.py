"""
run.py — One-command launcher for AI Interview Coach
Usage:
    python run.py              # starts API + UI
    python run.py --api-only
    python run.py --ui-only
"""
import subprocess, sys, os, signal, time
from pathlib import Path


def check_env():
    env_path = Path(".env")
    if not env_path.exists():
        import shutil
        shutil.copy(".env.example", ".env")
        print("\nERROR: .env file was missing — created it from .env.example")
        print("Open .env and add your GROQ_API_KEY, then run again.\n")
        sys.exit(1)

    env = {}
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()

    key = env.get("GROQ_API_KEY", "")
    if not key or key.startswith("gsk_...") or key == "":
        print("\nERROR: GROQ_API_KEY not set in .env")
        print("  1. Go to https://console.groq.com")
        print("  2. Sign up free → API Keys → Create Key")
        print("  3. Paste it in .env as: GROQ_API_KEY=gsk_xxxx\n")
        sys.exit(1)

    for k, v in env.items():
        os.environ.setdefault(k, v)

    print("✓ Environment OK")
    print(f"✓ Model: {env.get('LLM_MODEL', 'llama3-8b-8192')}")


def main():
    args = sys.argv[1:]
    api_only = "--api-only" in args
    ui_only  = "--ui-only"  in args

    check_env()
    procs = []

    print("\n🤖 AI Interview Coach")
    print("-" * 40)

    if not ui_only:
        print("Starting FastAPI   → http://localhost:8000")
        print("API Docs           → http://localhost:8000/docs")
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.api.main:app",
             "--host", "0.0.0.0", "--port", "8000", "--reload"],
            env={**os.environ},
        ))
        time.sleep(3)

    if not api_only:
        print("Starting Streamlit → http://localhost:8501")
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
             "--server.port", "8501", "--server.headless", "true"],
            env={**os.environ},
        ))

    print("\nPress Ctrl+C to stop.\n")

    def shutdown(sig, frame):
        print("\nShutting down...")
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        for p in procs:
            if p.poll() is not None:
                shutdown(None, None)
        time.sleep(1)


if __name__ == "__main__":
    main()

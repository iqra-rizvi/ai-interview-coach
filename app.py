"""
app.py — Root launcher redirecting to frontend/app.py for Streamlit Community Cloud.
Allows deployment using either `app.py` or `frontend/app.py` as the entrypoint.
"""
import os
import sys
import runpy
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Target frontend application
target_app = root_dir / "frontend" / "app.py"

if __name__ == "__main__":
    runpy.run_path(str(target_app), run_name="__main__")
else:
    # When imported by Streamlit runner
    runpy.run_path(str(target_app))

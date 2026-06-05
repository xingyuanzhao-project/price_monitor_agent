"""
Local entry point for the Price Monitor Agent application.

Installs dependencies, builds the frontend, and starts the FastAPI server.
Opens browser automatically.

Usage:
    python run.py
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
BACKEND_STATIC_DIR = PROJECT_ROOT / "backend" / "static"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def ensure_python_dependencies() -> None:
    """Install Python dependencies from requirements.txt."""
    print("Checking Python dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE), "--quiet"],
        check=True,
    )
    print("Python dependencies OK.")


def build_frontend_if_needed() -> None:
    """Build the frontend if the static directory is missing or empty."""
    if BACKEND_STATIC_DIR.is_dir() and any(BACKEND_STATIC_DIR.iterdir()):
        print("Frontend build already exists.")
        return
    print("Building frontend...")
    subprocess.run(["npm", "install"], cwd=str(FRONTEND_DIR), check=True)
    subprocess.run(["npm", "run", "build"], cwd=str(FRONTEND_DIR), check=True)
    print("Frontend build complete.")


def open_browser_after_delay(url: str, delay_seconds: float = 2.0) -> None:
    """Open browser after a short delay (runs in daemon thread)."""
    time.sleep(delay_seconds)
    webbrowser.open(url)


def main() -> None:
    """Entry point: install deps, build frontend, start server, open browser."""
    ensure_python_dependencies()

    try:
        build_frontend_if_needed()
    except (FileNotFoundError, subprocess.CalledProcessError) as build_error:
        print(f"Frontend build skipped: {build_error}")
        print("Run the frontend dev server separately: cd frontend && npm run dev")

    host = "0.0.0.0"
    port = 8000
    url = f"http://localhost:{port}"

    print(f"\nStarting Price Monitor Agent at {url}")
    print("Press Ctrl+C to stop.\n")

    threading.Thread(target=open_browser_after_delay, args=(url,), daemon=True).start()

    import uvicorn
    uvicorn.run("backend.main:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()

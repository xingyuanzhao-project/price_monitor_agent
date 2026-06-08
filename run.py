"""
Local entry point for the Price Monitor Agent application.

Installs dependencies, builds the frontend, and starts the FastAPI server.
Opens browser automatically. Server host/port are driven by environment
variables (PRICE_MONITOR_HOST, PRICE_MONITOR_PORT) via ServerSettings.

Usage:
    python run.py
"""

import shutil
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


def _resolve_npm() -> str:
    """Return the full path to npm. Raises FileNotFoundError if not installed."""
    npm_path = shutil.which("npm")
    if npm_path is None:
        raise FileNotFoundError(
            "npm not found on PATH. Install Node.js: https://nodejs.org/"
        )
    return npm_path


def ensure_python_dependencies() -> None:
    """Install Python dependencies from requirements.txt."""
    print("Checking Python dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE), "--quiet"],
        check=True,
    )
    print("Python dependencies OK.")


def build_frontend() -> None:
    """Build the frontend. Fails loudly if npm/node are missing."""
    npm = _resolve_npm()
    print("Building frontend...")
    subprocess.run([npm, "install"], cwd=str(FRONTEND_DIR), check=True)
    subprocess.run(
        [npm, "run", "build"],
        cwd=str(FRONTEND_DIR),
        check=True,
    )
    print("Frontend build complete.")


def open_browser_after_delay(url: str, delay_seconds: float = 2.0) -> None:
    """Open browser after a short delay (runs in daemon thread)."""
    time.sleep(delay_seconds)
    webbrowser.open(url)


def main() -> None:
    """Entry point: install deps, build frontend, start server, open browser."""
    ensure_python_dependencies()
    build_frontend()

    from backend.server_settings import get_server_settings
    settings = get_server_settings()
    host = settings.host
    port = settings.port
    url = f"http://localhost:{port}"

    print(f"\nStarting Price Monitor Agent at {url}")
    print("Press Ctrl+C to stop.\n")

    threading.Thread(target=open_browser_after_delay, args=(url,), daemon=True).start()

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

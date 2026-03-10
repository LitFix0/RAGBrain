"""
RAGBrain — Entry Point
Run: python backend/main.py
Waits for backend to be ready, then opens browser automatically.
"""

import sys
import os
import threading
import webbrowser
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

def wait_then_open_browser():
    """Poll the backend until it's ready, then open the browser."""
    url = "http://localhost:8000/status"
    for _ in range(30):  # try for up to 30 seconds
        try:
            urllib.request.urlopen(url, timeout=1)
            # Backend is ready!
            webbrowser.open("http://localhost:8000")
            return
        except Exception:
            time.sleep(1)
    # Fallback: open anyway after 30s
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    threading.Thread(target=wait_then_open_browser, daemon=True).start()

    uvicorn.run(
        "backend.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
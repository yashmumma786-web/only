"""
Deployment entry point for StoneStocks FastAPI server.

Binds port 5000 immediately (before any heavy imports) so the Replit
deployment health check succeeds.  The heavy imports in main.py
(torch, open_clip, transformers, cv2 – ~60 s on a cold container) run
AFTER the socket is already listening, so they no longer race against
the 60-second port-open deadline.

The dev workflow continues to use  `python main.py`  (hot-reload intact).
"""
import socket
import os

# ── 1. Bind the port right now, before any heavy imports ──────────────
PORT = int(os.getenv("PORT", "5000"))
_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
_sock.bind(("0.0.0.0", PORT))
_sock.listen(128)          # socket is now open; health check will pass

# ── 2. Heavy app imports happen here (~60 s on cold start) ─────────────
import uvicorn              # lightweight – just the uvicorn library

# ── 3. Hand the pre-bound socket to uvicorn, no reload in production ───
uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=PORT,
    fd=_sock.fileno(),      # use the already-bound socket
    reload=False,           # no file-watcher in production
    log_level="info",
)

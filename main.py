import os
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# Load environment variables first
load_dotenv()

# Force offline mode for Hugging Face Hub / Transformers to prevent network calls
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

logging.getLogger("httpx").setLevel(logging.WARNING)

# Module imports
from src.admin_auth import router as admin_router
from src.modules.ingestion.api import router as dataset_router
from src.modules.ml_inference import services as ml_services
from src.modules.orchestrator.api import router as orchestrator_router
from src.modules.search.api import router as search_router, pages_router as search_pages_router
from src.modules.search.gateway_api import router as gateway_router
from src.modules.search.internal_api import router as internal_search_router
from src.modules.similar.router import router as similar_router
from src.modules.taxonomy import services as taxonomy_services
from src.modules.taxonomy.api import router as tag_router
from src.modules.search.api import _load_all_stone_data_batched
from src.modules.ml_inference.foundation_worker import start_worker_if_needed as start_fw
from src.modules.orchestrator.lightness_worker import start_worker_if_needed as start_lw
from src.modules.orchestrator.vein_colour_worker import start_worker_if_needed as start_vw

# App Settings
APP_TITLE = "StoneStocks Offline Visual Trainer (MVP)"
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _safe_start_worker(start_func, success_msg: str, name: str) -> None:
    """Helper to safely start background workers during startup."""
    try:
        if start_func():
            print(f"[Startup] {success_msg}")
    except Exception as e:
        if "no such table" not in str(e):
            print(f"[Startup] {name} skipped: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    async def _warm_cache() -> None:
        try:
            await asyncio.to_thread(_load_all_stone_data_batched)
            print("[Startup] Stone cache warmed successfully")
        except Exception as exc:
            print(f"[Startup] Stone cache warm-up skipped: {exc}")

    asyncio.create_task(_warm_cache())

    # Mark stale runs
    try:
        ml_services.cleanup_stale_runs()
    except Exception as e:
        print(f"[Startup] Failed to mark stale runs interrupted: {e}")

    # Start workers safely using the helper function

    _safe_start_worker(start_fw, "Foundation worker resumed (pending work found)", "Foundation worker startup")
    _safe_start_worker(start_lw, "Lightness backfill worker launched (Task #375)", "Lightness worker startup")
    _safe_start_worker(start_vw, "Vein colour backfill worker resumed (pending work found)", "Vein colour worker startup")

    yield
    # --- Shutdown ---
    pass


# Check startup secrets
if not os.environ.get("REPLIT_INTERNAL_SECRET"):
    print("[Startup] ERROR: REPLIT_INTERNAL_SECRET is not set. All /internal/* endpoints are rejecting requests.")
else:
    print("[Startup] REPLIT_INTERNAL_SECRET is configured.")


# Initialize taxonomy database
taxonomy_services.init_db()

app = FastAPI(title=APP_TITLE, lifespan=lifespan)

# Middleware
SESSION_SECRET = os.environ.get("SESSION_SECRET", "dev-session-secret-change-me")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# Register Routers
routers = [
    admin_router,
    tag_router,
    orchestrator_router,
    dataset_router,
    gateway_router,  # Main Server Gateway registered before search APIs
    search_router,
    search_pages_router,
    similar_router,
    internal_search_router,
]
for router in routers:
    app.include_router(router)


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"request": request, "title": APP_TITLE, "runs": []}
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        reload=True,
        log_level="info",
        reload_excludes=[".local", ".local/", "cache", "cache/", "data", "data/*"],
    )
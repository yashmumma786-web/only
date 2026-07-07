# StoneStocks Offline Visual Trainer (MVP)

A FastAPI application for visual search and classification of stone/rock imagery using PyTorch, OpenCLIP, and Transformers.

## How to run

The workflow **"Start application"** runs `python main.py` on port 5000.

For production/deployment use `start.py` instead — it pre-binds the port before loading heavy ML dependencies to beat the 60-second health-check deadline.

## Stack

- **Backend**: FastAPI + Uvicorn (Python 3.11)
- **ML**: PyTorch, OpenCLIP, Transformers, OpenCV (headless), Ultralytics
- **Storage**: SQLite (per-module databases), filesystem for embeddings (`.npy`) and image caches
- **Templates**: Jinja2

## Modules

| Module | Purpose |
|---|---|
| `ingestion` | Dataset imports via CSV upload |
| `ml_inference` | Embedding generation, CV analysis, foundation model jobs |
| `orchestrator` | Background workers (lightness, vein colour backfill) |
| `search` | Visual + name search with facets and gateway API |
| `similar` | Similar-stone explorer |
| `taxonomy` | Tag classification, sibling votes, fingerprints |

## Required secrets

| Secret | Purpose |
|---|---|
| `SESSION_SECRET` | Starlette session middleware |
| `ADMIN_PASSWORD` | Admin login |
| `REPLIT_INTERNAL_SECRET` | Protects `/internal/*` endpoints |
| `DATASETS_DB_PATH` | Path to datasets SQLite DB |
| `FOUNDATION_DB_PATH` | Path to foundation model jobs DB |
| `ENRICHMENT_DB_PATH` | Path to enrichment jobs DB |
| `EMBEDDINGS_CACHE_DIR` | Directory for `.npy` embedding files |
| `IMAGE_CACHE_DIR` | Directory for cached images |
| `SEARCH_LOGS_DB_PATH` | Path to search logs DB |
| `FINGERPRINTS_DB_PATH` | Path to taxonomy fingerprints DB |
| `SIBLING_VOTES_DB_PATH` | Path to sibling votes DB |
| `SIMILAR_EXPLORER_DB_PATH` | Path to similar explorer DB |
| `STOCK_TAGS_DB_PATH` | Path to stock tags DB |
| `STOCK_TAGS_BACKUP_DIR` | Directory for stock tags backups |
| `ANALYSIS_CACHE_DIR` | Directory for analysis cache |

## Known fixes applied on Replit

- `opencv-python` (GUI build) replaced with `opencv-python-headless` — `libxcb.so.1` is not available on Replit's headless servers.
- `MergedStockTags` re-exported from `src/modules/taxonomy/services.py` (was only defined in `repository.py`).

## User preferences

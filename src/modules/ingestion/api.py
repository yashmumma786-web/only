"""
Ingestion API Aggregator
"""

from fastapi import APIRouter
from src.modules.ingestion.admin_router import router as admin_router
from src.modules.ingestion.analyser_router import router as analyser_router

router = APIRouter()
router.include_router(admin_router)
router.include_router(analyser_router)

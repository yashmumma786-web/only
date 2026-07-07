# Cache manager utility to handle cross-module cache invalidation callbacks
import logging

logger = logging.getLogger(__name__)

_invalidation_callbacks = []

def register_invalidation_callback(callback) -> None:
    """Register a callback to be run when the stone cache is invalidated."""
    if callback not in _invalidation_callbacks:
        _invalidation_callbacks.append(callback)

def invalidate_stone_cache() -> None:
    """Trigger all registered stone cache invalidation callbacks."""
    logger.info("Triggering global stone cache invalidation callbacks...")
    for cb in _invalidation_callbacks:
        try:
            cb()
        except Exception as e:
            logger.error(f"Error running cache invalidation callback: {e}")

_color_cache_invalidator = None

def register_color_cache_invalidator(fn) -> None:
    """Register the callback for invalidating the color cache."""
    global _color_cache_invalidator
    _color_cache_invalidator = fn

def notify_color_cache_invalidator(company_stone_id: str) -> None:
    """Trigger the registered color cache invalidation callback."""
    if _color_cache_invalidator:
        try:
            _color_cache_invalidator(company_stone_id)
        except Exception as e:
            logger.error(f"Error running color cache invalidator for {company_stone_id}: {e}")

from typing import Dict, List, Optional, Any, Tuple
from src.modules.taxonomy import repository as _repo
from src.modules.taxonomy.repository import PATTERN_FAMILY_ALIASES, MergedStockTags
from pathlib import Path
from src.modules.taxonomy import color_rules
from src.modules.taxonomy.aggregator import aggregate_stock, _fingerprint_to_tag_dict
from src.modules.taxonomy import storage_sqlite as storage
from src.modules.taxonomy import sibling_votes

def get_pattern_family_aliases() -> Dict[str, str]:
    """Retrieve pattern family aliases."""
    return PATTERN_FAMILY_ALIASES



def get_family_hue_stats(family_name: str) -> List[Dict[str, Any]]:
    """Retrieve hue statistics for a specific family."""
    return _repo.get_family_hue_stats(family_name)


def get_block_hue_stats(block_id: str) -> List[Dict[str, Any]]:
    """Retrieve hue statistics for a specific block."""
    return _repo.get_block_hue_stats(block_id)


def get_batch_facet_data(
    company_stone_ids: List[str],
    tag_names: List[str],
    agg_fields: List[str],
    ovr_fields: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Get batch facet data including AI, manual, aggregated, and override records."""
    return _repo.get_batch_facet_data(company_stone_ids, tag_names, agg_fields, ovr_fields)




def get_stone_color_info(company_stone_id: str) -> Dict[str, Optional[str]]:
    """Retrieve base color and AI color information for a stone."""
    return _repo.get_stone_color_info(company_stone_id)


def get_page_hydration_data(company_stone_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve metadata/tag hydration data for page rendering."""
    return _repo.get_page_hydration_data(company_stone_ids)



def get_stones_missing_ai_tag(company_stone_ids: List[str], tag_name: str) -> List[str]:
    """Get stones from the batch that do not have the specified AI tag."""
    return _repo.get_stones_missing_ai_tag(company_stone_ids, tag_name)


def save_lightness_tags(company_stone_id: str, mean_L: Optional[float], base_color_hint: Optional[str] = None) -> bool:
    """Save lightness math and band to the AI tags database."""
    if mean_L is None:
        return False
    
    if base_color_hint is None:
        tags_dict = _repo.get_stone_tags(company_stone_id)
        manual_color = tags_dict.get("manual_tags", {}).get("base_color", {}).get("value")
        ai_color = tags_dict.get("ai_tags", {}).get("base_color", {}).get("value")
        base_color_hint = manual_color or ai_color
        
    band = color_rules.classify_lightness_band(mean_L, base_color_hint)
    
    tags = {
        "stone_lightness_mean": (mean_L, 1.0),
        "stone_lightness": (band, 1.0)
    }
    return _repo.set_ai_tags_batch(company_stone_id, tags)


def write_ml_predictions_batch(company_stone_id: str, predictions: Dict[str, Any]) -> bool:
    """Write an arbitrary batch of ML predictions to the AI tags database."""
    if not company_stone_id or not predictions:
        return False
    tags_to_write = {}
    for k, v in predictions.items():
        if k == "company_stone_id": continue
        # Default confidence to 1.0 for ML outputs if not provided
        if isinstance(v, tuple) and len(v) == 2:
            tags_to_write[k] = v
        else:
            tags_to_write[k] = (v, 1.0)
    return _repo.set_ai_tags_batch(company_stone_id, tags_to_write)


def get_lightness_pending(company_stone_ids: List[str]) -> List[str]:
    """Retrieve stones that have base_color but no stone_lightness in stock_tags_ai."""
    return _repo.get_lightness_pending(company_stone_ids)


def check_integrity(db_path: Optional[Path] = None) -> bool:
    """Check integrity of the tags database (or a specific backup db)."""
    return _repo.check_integrity(db_path)


def backup_tags_db(backup_path: Path) -> None:
    """Backup the tags database."""
    _repo.backup_tags_db(backup_path)


def snapshot_tag_counts(cids: set) -> Dict[str, Tuple[int, int]]:
    """Return {tag_name: (n_rows, n_stones)} snapshot for the given stone IDs."""
    return _repo.snapshot_tag_counts(cids)


def has_ai_main_color(stone_id: str) -> bool:
    """True iff ``stock_tags_ai`` has a non-NULL ``main_color`` row."""
    return _repo.has_ai_main_color(stone_id)


def set_ai_tags_batch(company_stone_id: str, tags: Dict[str, Tuple[Any, float]]) -> bool:
    """Set AI tags in batch."""
    return _repo.set_ai_tags_batch(company_stone_id, tags)


def save_enrichment_results(
    cv_data: Dict[str, dict],
    predictions: Dict[str, dict],
) -> Dict[str, Any]:
    """
    Takes CV data and ML predictions from the Orchestrator,
    aggregates CV data, applies model predictions, and writes
    fingerprints and tags to taxonomy DB.
    """


    built = {}
    predictions_tags = {}

    for sid, data in cv_data.items():
        try:
            analyses = data.get("analyses") or []
            image_paths = data.get("image_paths") or []
            image_ids = data.get("image_ids") or []

            fingerprint = aggregate_stock(sid, analyses, image_paths, image_ids)
            
            # Apply model predictions
            pred = predictions.get(sid)
            if pred:
                if pred.get("base_color"):
                    bc = pred["base_color"]
                    fingerprint.ai_base_color = bc["label"]
                    fingerprint.ai_base_color_conf = bc["confidence"]
                    fingerprint.ai_base_color_method = bc["method"]
                if pred.get("design"):
                    ds = pred["design"]
                    fingerprint.ai_design = ds["label"]
                    fingerprint.ai_design_conf = ds["confidence"]
                    fingerprint.ai_design_method = ds["method"]
                if pred.get("tonality"):
                    ton = pred["tonality"]
                    fingerprint.ai_tonality = ton["label"]
                    fingerprint.ai_tonality_conf = ton["confidence"]
                    fingerprint.ai_tonality_method = ton["method"]

            if fingerprint.ai_base_color:
                fingerprint.base_color = fingerprint.ai_base_color.lower()
                fingerprint.base_color_conf = fingerprint.ai_base_color_conf

            if fingerprint.ai_design:
                fingerprint.design = fingerprint.ai_design.lower()
                fingerprint.design_conf = fingerprint.ai_design_conf

            if fingerprint.ai_tonality:
                fingerprint.tonality = fingerprint.ai_tonality.lower()
                fingerprint.tonality_conf = fingerprint.ai_tonality_conf

            built[sid] = fingerprint
            storage.save_fingerprint(fingerprint)
            predictions_tags[sid] = _fingerprint_to_tag_dict(fingerprint)

            # Sync tags to stock_tags_ai
            sync_tags = {}
            if fingerprint.base_color:
                sync_tags["base_color"] = (fingerprint.base_color, fingerprint.base_color_conf)
                sync_tags["main_color"] = (fingerprint.base_color, fingerprint.base_color_conf)
            if fingerprint.design:
                sync_tags["design"] = (fingerprint.design, fingerprint.design_conf)
                
                pf_val = fingerprint.design
                if pf_val == "plain": pf_val = "uniform"
                elif pf_val == "veiny": pf_val = "veined"
                elif pf_val == "spotty": pf_val = "breccia"
                
                sync_tags["pattern_family"] = (pf_val, fingerprint.design_conf)
                
            if fingerprint.tonality:
                sync_tags["tonality"] = (fingerprint.tonality, fingerprint.tonality_conf)
            if sync_tags:
                _repo.set_ai_tags_batch(sid, sync_tags)
        except Exception as e:
            print(f"[TAXONOMY-SERVICES] Failed to save/aggregate enrichment for {sid}: {e}")

    return {
        "built": built,
        "predictions": predictions_tags,
    }


def set_sibling_vote(
    reference_company_stone_id: str,
    sibling_company_stone_id: str,
    vote: int,
    admin_user_id: Optional[str] = None,
) -> bool:
    """Record a sibling vote (UP=1, DOWN=-1) on a pair of stones."""
    return sibling_votes.set_vote(
        company_stone_id_a=reference_company_stone_id,
        company_stone_id_b=sibling_company_stone_id,
        vote=vote,
        admin_user_id=admin_user_id,
    )


def get_suppressed_siblings(stone_id: str) -> List[str]:
    """Get list of stone IDs marked as not related (vote = -1) to the given stone."""
    return sibling_votes.get_suppressed_siblings(stone_id)


def init_db() -> None:
    """Initialize the taxonomy database."""
    _repo.init_db()


def get_pattern_family_values() -> List[str]:
    """Get the valid list of taxonomy pattern families."""
    return list(_repo.TAXONOMY_PATTERN_FAMILY_VALUES)







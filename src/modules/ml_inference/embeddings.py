"""
V3 Embeddings - Stock-level visual embeddings for classification

Uses OpenCLIP (ViT-B/32) to compute per-image embeddings, then aggregates
to stock-level embeddings using quality-weighted mean.

Caches embeddings to: cache/embeddings_v3/{company_stone_id}.npy
"""

import json
import hashlib
import requests
import time
import torch
import open_clip
import cv2
import numpy as np
import gc
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import os
EMBEDDINGS_CACHE_DIR = Path(os.environ.get("EMBEDDINGS_CACHE_DIR"))
EMBEDDINGS_META_FILE = EMBEDDINGS_CACHE_DIR / "embeddings_meta.json"
IMAGE_CACHE_DIR = Path(os.environ.get("IMAGE_CACHE_DIR"))
STONESTOCKS_BASE_URL = "https://images.stonestocks.com"

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _ensure_dirs():
    """Ensure cache directories exist."""
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_clip_model():
    """Load OpenCLIP model lazily."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    
    if _clip_model is not None:
        return _clip_model, _clip_preprocess
    
    print("[V3 Embeddings] Loading OpenCLIP ViT-B/32...")
    
    pretrained_options = [
        'openai',
        'laion2b_s34b_b79k',
        'laion400m_e32',
        None
    ]
    
    for pretrained in pretrained_options:
        try:
            if pretrained:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained=pretrained,
                    device='cpu'
                )
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    device='cpu'
                )
            
            model.eval()
            _clip_model = model
            _clip_preprocess = preprocess
            print(f"[V3 Embeddings] Loaded OpenCLIP with pretrained='{pretrained}'")
            return model, preprocess
            
        except Exception as e:
            print(f"[V3 Embeddings] Failed with pretrained='{pretrained}': {e}")
            continue
    
    raise RuntimeError("Failed to load OpenCLIP model")


def _download_image(image_path: str, timeout: int = 10) -> Optional[Image.Image]:
    """Download image from stonestocks or load from cache."""
    cache_key = hashlib.md5(image_path.encode()).hexdigest()[:16]
    cached_path = IMAGE_CACHE_DIR / f"{cache_key}.jpg"
    
    try:
        if cached_path.exists():
            img = Image.open(cached_path)
            img.load()
            return img.convert("RGB")
        
        if image_path.startswith("http"):
            url = image_path
        else:
            url = f"{STONESTOCKS_BASE_URL}/{image_path}"
        
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cached_path, 'wb') as f:
                f.write(response.content)
            img = Image.open(BytesIO(response.content))
            img.load()
            return img.convert("RGB")
    except Exception as e:
        print(f"[V3 Embeddings] Failed to download {image_path}: {e}")
    
    return None


def _compute_image_quality_weight(image: Image.Image) -> float:
    """Compute quality weight for an image.
    
    weight = 1 / (1 + blur) * exposure_score
    
    Simplified version without mask detection.
    """
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = 1.0 / (1.0 + 1000.0 / (laplacian_var + 1))
    
    mean_val = gray.mean()
    if mean_val < 50:
        exposure_score = mean_val / 50.0
    elif mean_val > 200:
        exposure_score = (255 - mean_val) / 55.0
    else:
        exposure_score = 1.0
    
    weight = blur_score * max(0.2, exposure_score)
    
    return max(0.1, min(1.0, weight))


def compute_image_embedding(image: Image.Image) -> Optional[np.ndarray]:
    """Compute embedding for a single image."""
    model, preprocess = _load_clip_model()
    
    try:
        img_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            embedding = model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        result = embedding.cpu().numpy().flatten()
        del img_tensor, embedding
        return result
    
    except Exception as e:
        print(f"[V3 Embeddings] Failed to compute embedding: {e}")
        return None


def compute_stock_embedding(
    image_paths: List[str],
    max_images: int = 10
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Compute stock-level embedding from multiple images.
    
    Uses quality-weighted mean of per-image embeddings.
    
    Returns:
        (embedding, metadata) where metadata includes:
        - image_count_used
        - average_weight
        - errors (list of failed images)
    """
    _ensure_dirs()
    
    embeddings = []
    weights = []
    errors = []
    
    paths_to_process = image_paths[:max_images]
    
    for path in paths_to_process:
        image = _download_image(path)
        if image is None:
            errors.append(path)
            continue
        
        embedding = compute_image_embedding(image)
        if embedding is None:
            image.close()
            del image
            errors.append(path)
            continue
        
        weight = _compute_image_quality_weight(image)
        image.close()
        del image
        
        embeddings.append(embedding)
        weights.append(weight)
    
    if not embeddings:
        return None, {
            "image_count_used": 0,
            "average_weight": 0.0,
            "errors": errors
        }
    
    embeddings_array = np.stack(embeddings)
    weights_array = np.array(weights)
    
    weights_normalized = weights_array / weights_array.sum()
    stock_embedding = np.average(embeddings_array, axis=0, weights=weights_normalized)
    
    stock_embedding = stock_embedding / np.linalg.norm(stock_embedding)
    
    metadata = {
        "image_count_used": len(embeddings),
        "average_weight": float(weights_array.mean()),
        "errors": errors
    }
    
    return stock_embedding, metadata


def save_stock_embedding(
    company_stone_id: str,
    embedding: np.ndarray,
    metadata: Dict[str, Any],
    skip_meta_json: bool = False,
) -> bool:
    """Save stock embedding to cache."""
    _ensure_dirs()
    
    try:
        embedding_path = EMBEDDINGS_CACHE_DIR / f"{company_stone_id}.npy"
        np.save(str(embedding_path), embedding)
        
        if not skip_meta_json:
            all_meta = load_embeddings_metadata()
            all_meta[company_stone_id] = {
                **metadata,
                "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            save_embeddings_metadata(all_meta)
        
        return True
    except Exception as e:
        print(f"[V3 Embeddings] Failed to save embedding for {company_stone_id}: {e}")
        return False


def load_stock_embedding(company_stone_id: str) -> Optional[np.ndarray]:
    """Load stock embedding from cache."""
    embedding_path = EMBEDDINGS_CACHE_DIR / f"{company_stone_id}.npy"
    
    if not embedding_path.exists():
        return None
    
    try:
        return np.load(str(embedding_path))
    except Exception as e:
        print(f"[V3 Embeddings] Failed to load embedding for {company_stone_id}: {e}")
        return None


def load_embeddings_metadata() -> Dict[str, Dict]:
    """Load all embeddings metadata."""
    if not EMBEDDINGS_META_FILE.exists():
        return {}
    
    try:
        with open(EMBEDDINGS_META_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_embeddings_metadata(metadata: Dict[str, Dict]):
    """Save all embeddings metadata."""
    _ensure_dirs()
    with open(EMBEDDINGS_META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

_GC_EVERY = 25

def build_all_embeddings(
    manifest_df,
    max_images_per_stock: int = 5,
    force_rebuild: bool = False,
    progress_callback=None,
) -> Dict[str, Any]:
    """Build embeddings for all stocks in manifest.
    
    Args:
        manifest_df: DataFrame with company_stone_id and image_asset_id columns
        max_images_per_stock: Max images to use per stock
        force_rebuild: If True, rebuild even if cached
        progress_callback: Optional callable(done, total, stone_id) called after each stone
        
    Returns:
        Dict with build statistics
    """
    _ensure_dirs()
    
    stocks_by_id = manifest_df.groupby("company_stone_id")["image_asset_id"].apply(list).to_dict()
    
    built = 0
    skipped = 0
    failed = 0
    done = 0
    
    total = len(stocks_by_id)
    
    for i, (company_stone_id, image_paths) in enumerate(stocks_by_id.items()):
        if (i + 1) % 50 == 0:
            print(f"[V3 Embeddings] Processing {i+1}/{total}...")
        
        if not force_rebuild:
            existing = load_stock_embedding(company_stone_id)
            if existing is not None:
                del existing
                skipped += 1
                done += 1
                if progress_callback:
                    progress_callback(done, total, company_stone_id)
                continue
        
        embedding, metadata = compute_stock_embedding(image_paths, max_images_per_stock)
        
        if embedding is not None:
            save_stock_embedding(company_stone_id, embedding, metadata, skip_meta_json=True)
            del embedding, metadata
            built += 1
        else:
            failed += 1

        done += 1
        if progress_callback:
            progress_callback(done, total, company_stone_id)

        if done % _GC_EVERY == 0:
            gc.collect()
    
    print(f"[V3 Embeddings] Done: built={built} skipped={skipped} failed={failed}/{total}")
    return {
        "total_stocks": total,
        "built": built,
        "skipped": skipped,
        "failed": failed
    }


def compute_embedding_aggregates(
    embeddings: np.ndarray, outlier_threshold: float = 0.78
) -> Dict[str, Any]:
    """Pure mathematical function to compute centroids and cosine similarities."""
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    cosines = embeddings @ centroid
    best_idx = int(np.argmax(cosines))
    consistency_score = float(np.median(cosines))
    outlier_count = int(np.sum(cosines < outlier_threshold))

    return {
        "best_idx": best_idx,
        "centroid_embedding": centroid.tolist(),
        "consistency_score": consistency_score,
        "outlier_count": outlier_count,
    }


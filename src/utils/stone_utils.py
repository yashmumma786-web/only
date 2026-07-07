import re
from pathlib import Path
from typing import Optional
import numpy as np

EMBEDDINGS_CACHE_DIR = Path("cache/embeddings_v3")

NAME_STOPWORDS = {
    "stone", "gem", "natural", "slab", "tile", "polished", "honed", 
    "leathered", "brushed", "cm", "mm", "x", "the", "a", "an"
}

SIZE_PATTERN = re.compile(r'\b\d+(?:\.\d+)?(?:cm|mm|x|\s*x\s*)\b', re.IGNORECASE)

def normalize_name(name: str) -> str:
    """
    Normalize stone name for matching:
    - lowercase
    - remove punctuation
    - remove stopwords
    - remove numeric sizes/thickness tokens
    - collapse whitespace
    """
    if not name:
        return ""
    
    text = name.lower()
    text = SIZE_PATTERN.sub(' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in NAME_STOPWORDS and not w.isdigit()]
    return ' '.join(words).strip()
"""
In-memory image fetching from URLs.

Fetches image bytes via HTTP, loads into numpy array in memory.
No disk writes, no temporary files, no persistence.
Bytes are discarded immediately after conversion to numpy array.
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

FETCH_TIMEOUT = 15
MAX_IMAGE_SIZE = 10 * 1024 * 1024

USER_AGENT = "StoneStocks-Analyzer/1.0"


def fetch_image_from_url(url: str, timeout: int = FETCH_TIMEOUT) -> Optional[np.ndarray]:
    """
    Fetch image from URL and return as BGR numpy array (OpenCV format).
    
    - Fetches bytes via HTTP GET
    - Loads into PIL Image from BytesIO (no disk)
    - Converts to numpy array
    - Discards bytes immediately
    
    Returns None if fetch fails or image is invalid.
    """
    if not url:
        return None
    
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()
        
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_IMAGE_SIZE:
            logger.warning(f"Image too large: {content_length} bytes, skipping {url[:100]}")
            return None
        
        image_bytes = response.content
        
        if len(image_bytes) > MAX_IMAGE_SIZE:
            logger.warning(f"Image too large after download: {len(image_bytes)} bytes")
            return None
        
        bytes_io = io.BytesIO(image_bytes)
        pil_image = Image.open(bytes_io)
        
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        rgb_array = np.array(pil_image)
        
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        del image_bytes
        del bytes_io
        del pil_image
        del rgb_array
        
        return bgr_array
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching image: {url[:100]}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"HTTP error fetching image: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error loading image from URL: {e}")
        return None


def fetch_images_from_urls(urls: List[str], max_images: int = 3) -> List[np.ndarray]:
    """
    Fetch multiple images from URLs, returning list of BGR numpy arrays.
    
    Stops after max_images successful fetches.
    Invalid URLs or failed fetches are skipped.
    """
    images = []
    
    for url in urls:
        if len(images) >= max_images:
            break
        
        img = fetch_image_from_url(url)
        if img is not None:
            images.append(img)
    
    return images


def crop_to_slab_region(
    img: np.ndarray,
    left_pct: float = 0.10,
    right_pct: float = 0.10,
    top_pct: float = 0.10,
    bottom_pct: float = 0.20
) -> Optional[np.ndarray]:
    """
    Crop image to focus on slab region, removing background and label strip.
    
    Default crops:
    - 10% from left (remove background/shadows)
    - 10% from right (remove background/shadows)
    - 10% from top (remove background/shadows)
    - 20% from bottom (remove label strip)
    
    This leaves approximately 80% width x 70% height = 56% of image area,
    focused on the central slab region.
    
    Returns cropped image (in-memory, no disk writes).
    """
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    left = int(w * left_pct)
    right = int(w * (1 - right_pct))
    top = int(h * top_pct)
    bottom = int(h * (1 - bottom_pct))
    
    left = max(0, left)
    right = min(w, right)
    top = max(0, top)
    bottom = min(h, bottom)
    
    if right <= left or bottom <= top:
        return img
    
    cropped = img[top:bottom, left:right]
    
    return cropped

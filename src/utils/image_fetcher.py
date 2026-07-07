"""
Shared image fetching utilities with Cloudflare URL normalization.
"""
from urllib.parse import unquote

CHROME_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

SESSION_HEADERS = {
    "User-Agent": CHROME_USER_AGENT,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://images.stonestocks.com/",
}


def normalize_url(url: str) -> str:
    """
    Handles Cloudflare image-resize URLs like:
    https://images.stonestocks.com/cdn-cgi/image/width=1200,quality=85/https://images.stonestocks.com/images/....jpg
    """
    if not url:
        return url

    if "/cdn-cgi/image/" in url:
        after = url.split("/cdn-cgi/image/", 1)[1]   # 'width=.../https://images...'
        after = after.split("/", 1)[1]               # 'https://images.stonestocks.com/images/...'
        after = unquote(after)                       # in case it is URL-encoded
        if after.startswith("http://") or after.startswith("https://"):
            return after
        return "https://images.stonestocks.com/" + after.lstrip("/")

    return url

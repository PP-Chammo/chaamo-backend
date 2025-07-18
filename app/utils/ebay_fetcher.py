import httpx
from typing import Optional
import asyncio
import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:107.0) Gecko/20100101 Firefox/107.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
]

DEFAULT_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

async def fetch_html(url: str, params: Optional[dict[str, str]] = None, headers: Optional[dict[str, str]] = None) -> str:
    if params is None:
        params = {}
    # Pick a random User-Agent for every request
    ua = random.choice(USER_AGENTS)
    merged_headers = dict(DEFAULT_HEADERS)
    merged_headers["User-Agent"] = ua
    if headers:
        merged_headers.update(headers)
    # Increased, human-like randomized delay (0.5-1.2s)
    await asyncio.sleep(random.uniform(0.5, 1.2))
    async with httpx.AsyncClient(headers=merged_headers, follow_redirects=True, http2=True) as client:
        try:
            response = await client.get(url, params=params, timeout=20.0)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                print(f"[WARN] 403 Forbidden for url: {url} - Skipping.")
                return ""
            print(f"HTTP error occurred: {e}")
            raise
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            raise


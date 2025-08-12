import random
import asyncio
import httpx
from urllib.parse import urlparse

from typing import Optional

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
]

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
    # Extra headers to better mimic browsers
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

def get_random_headers() -> dict:
    """
    Generate random headers for HTTP requests to avoid detection.
    """
    headers = dict(DEFAULT_HEADERS)
    headers["User-Agent"] = random.choice(USER_AGENTS)
    return headers

async def httpx_get_content(url: str, params: Optional[dict[str, str]] = None, headers: Optional[dict[str, str]] = None) -> str:
    if params is None:
        params = {}
    if headers is None:
        headers = get_random_headers()

    # Ensure a reasonable Referer and Host headers
    try:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        headers.setdefault("Referer", origin)
        headers.setdefault("Host", parsed.netloc)
    except Exception:
        pass

    # Small human-like jitter
    await asyncio.sleep(random.uniform(0.25, 0.9))

    # Simple retry with backoff and UA rotation
    attempts = 3
    for attempt in range(1, attempts + 1):
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, http2=True, timeout=httpx.Timeout(20.0)) as client:
            try:
                req = client.build_request('GET', url, params=params)
                print(f"httpx Scraping URL: {req.url} (attempt {attempt}/{attempts})")
                response = await client.send(req)
                if response.status_code in (429, 403):
                    # Rotate UA and wait then retry
                    headers["User-Agent"] = random.choice(USER_AGENTS)
                    await asyncio.sleep(1.0 * attempt)
                    continue
                response.raise_for_status()
                return response.text
            except (httpx.ReadTimeout, httpx.ConnectTimeout):
                await asyncio.sleep(0.5 * attempt)
                continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (403, 429):
                    await asyncio.sleep(1.0 * attempt)
                    continue
                raise
            except httpx.NetworkError:
                await asyncio.sleep(0.5 * attempt)
                continue

    return ""

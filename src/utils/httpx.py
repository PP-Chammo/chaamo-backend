import random
import asyncio
import httpx
from urllib.parse import urlparse, urlencode, urljoin

from typing import Optional

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
]

ACCEPT_LANGS = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.8",
    "en-US,en-GB;q=0.7,en;q=0.6",
]

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
    # Extra headers to better mimic browsers
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

def get_random_headers() -> dict:
    """
    Generate random headers for HTTP requests to avoid detection.
    """
    headers = dict(DEFAULT_HEADERS)
    headers["User-Agent"] = random.choice(USER_AGENTS)
    # Randomize client hints slightly
    if "sec-ch-ua" not in headers:
        headers["sec-ch-ua"] = '"Chromium";v="120", "Not(A:Brand";v="24", "Google Chrome";v="120"'
    if "sec-ch-ua-mobile" not in headers:
        headers["sec-ch-ua-mobile"] = "?0"
    if "sec-ch-ua-platform" not in headers:
        headers["sec-ch-ua-platform"] = random.choice(['"Windows"', '"macOS"'])
    headers["Accept-Language"] = random.choice(ACCEPT_LANGS)
    return headers

def _build_referer(url: str, params: dict[str, str]) -> str:
    """Build a realistic Referer including query params when available."""
    try:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if params:
            return f"{origin}{parsed.path}?{urlencode(params)}"
        return origin
    except Exception:
        return url

async def httpx_get_content(
    url: str,
    params: Optional[dict[str, str]] = None,
    headers: Optional[dict[str, str]] = None,
    *,
    attempts: int = 5,
    request_timeout: float = 25.0,
    jitter_min: float = 0.6,
    jitter_max: float = 1.5,
) -> str:
    if params is None:
        params = {}
    if headers is None:
        headers = get_random_headers()

    # Ensure a reasonable Referer and Host headers
    try:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        headers.setdefault("Referer", _build_referer(url, params))
        headers.setdefault("Host", parsed.netloc)
    except Exception:
        pass

    # Small human-like jitter
    try:
        await asyncio.sleep(random.uniform(float(jitter_min), float(jitter_max)))
    except Exception:
        await asyncio.sleep(0.5)

    # Simple retry with backoff and UA rotation
    # Create a single client per call to persist cookies across retries
    use_http2 = bool(random.getrandbits(1))
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0)
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, http2=use_http2, timeout=httpx.Timeout(float(request_timeout)), limits=limits, trust_env=True) as client:
        for attempt in range(1, attempts + 1):
            try:
                req = client.build_request('GET', url, params=params, headers=headers)
                print(f"httpx Scraping URL: {req.url} (attempt {attempt}/{attempts})")
                response = await client.send(req)
                if response.status_code in (429, 403):
                    # Rotate UA and wait then retry
                    headers["User-Agent"] = random.choice(USER_AGENTS)
                    backoff = min(8.0, (1.5 ** attempt)) + random.uniform(0.2, 0.8)
                    await asyncio.sleep(backoff)
                    continue
                response.raise_for_status()
                text = response.text
                # Handle interstitial redirect pages (meta refresh / JS redirects / Refresh header)
                # Try a small number of internal follows to resolve final HTML
                for _ in range(2):
                    # 1) HTTP Refresh header
                    refresh = response.headers.get("Refresh") or response.headers.get("refresh")
                    redirect_url = None
                    if refresh and "url=" in refresh.lower():
                        try:
                            part = refresh.split("url=", 1)[1].strip().strip("'\"")
                            redirect_url = urljoin(str(req.url), part)
                        except Exception:
                            redirect_url = None
                    # 2) Meta refresh
                    if not redirect_url:
                        import re as _re
                        m = _re.search(r"<meta[^>]*http-equiv=\s*\"?refresh\"?[^>]*content=\s*\"?\d+\s*;\s*url=([^\"'>]+)\"?", text, flags=_re.I)
                        if m:
                            redirect_url = urljoin(str(req.url), m.group(1))
                    # 3) JS location redirects
                    if not redirect_url:
                        for pat in [
                            r"location\.href\s*=\s*['\"]([^'\"]+)['\"]",
                            r"window\.location\.replace\(\s*['\"]([^'\"]+)['\"]\s*\)",
                            r"location\.assign\(\s*['\"]([^'\"]+)['\"]\s*\)"
                        ]:
                            import re as _re
                            m = _re.search(pat, text, flags=_re.I)
                            if m:
                                redirect_url = urljoin(str(req.url), m.group(1))
                                break
                    if not redirect_url:
                        break
                    # Follow the discovered redirect within the same client
                    await asyncio.sleep(random.uniform(0.2, 0.6))
                    response = await client.get(redirect_url)
                    response.raise_for_status()
                    text = response.text
                    # If the new page is normal, break on next loop naturally
                # Detect common bot-block/interstitial pages and retry with backoff
                block_indicators = [
                    "verify you are a human",
                    "please verify yourself",
                    "access denied",
                    "captcha",
                    "to continue, please",
                    "your browser will redirect to your requested content shortly",
                    "please wait"
                ]
                if any(s in text.lower() for s in block_indicators):
                    headers["User-Agent"] = random.choice(USER_AGENTS)
                    backoff = min(8.0, (1.5 ** attempt)) + random.uniform(0.3, 1.0)
                    await asyncio.sleep(backoff)
                    continue
                return text
            except (httpx.ReadTimeout, httpx.ConnectTimeout):
                await asyncio.sleep(0.5 * attempt + random.uniform(0.1, 0.5))
                continue
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (403, 429):
                    await asyncio.sleep(1.0 * attempt + random.uniform(0.2, 0.6))
                    continue
                raise
            except httpx.NetworkError:
                await asyncio.sleep(0.5 * attempt + random.uniform(0.2, 0.8))
                continue

    return ""

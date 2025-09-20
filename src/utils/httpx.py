"""Clean HTTP client utilities with Zyte proxy support and anti-bot measures."""

import asyncio
import os
import random
import re
from typing import Optional
from urllib.parse import urlencode, urljoin, urlparse
from dotenv import load_dotenv

# Load environment variables from .env file

import httpx

from src.utils.logger import get_logger, httpx_logger

logger = get_logger("httpx")

load_dotenv()

# Disable verbose httpx logging to prevent spam
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

# Modern browser user agents for anti-bot measures
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

# Accept language headers for internationalization
ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.9,es;q=0.8",
    "en-GB,en-US;q=0.9,en;q=0.8",
    "en-US,en;q=0.9,fr;q=0.8",
    "en-GB,en;q=0.9,de;q=0.8",
]

# Zyte API configuration
ZYTE_API_KEY = os.environ.get("ZYTE_API_ACCESS")
ZYTE_PROXY_URL = os.environ.get("ZYTE_PROXY_URL")

# Elite proxy servers - tested and working for eBay scraping
PROXY_LIST = []

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
    headers["Accept-Language"] = random.choice(ACCEPT_LANGUAGES)

    # Randomize client hints based on user agent
    ua = headers["User-Agent"]
    if "chrome" in ua.lower():
        chrome_versions = ["120", "121", "122"]
        version = random.choice(chrome_versions)
        headers["sec-ch-ua"] = (
            f'"Chromium";v="{version}", "Not(A:Brand";v="24", "Google Chrome";v="{version}"'
        )
        headers["sec-ch-ua-mobile"] = "?0"
        headers["sec-ch-ua-platform"] = random.choice(
            ['"Windows"', '"macOS"', '"Linux"']
        )
    elif "firefox" in ua.lower():
        # Firefox doesn't send sec-ch-ua headers
        headers.pop("sec-ch-ua", None)
        headers.pop("sec-ch-ua-mobile", None)
        headers.pop("sec-ch-ua-platform", None)
    elif "safari" in ua.lower():
        # Safari on macOS
        headers["sec-ch-ua-platform"] = '"macOS"'
        headers["sec-ch-ua-mobile"] = "?0"

    # Add viewport variations
    headers["viewport-width"] = str(random.randint(1200, 1920))

    return headers


def get_zyte_proxy() -> Optional[str]:
    """Get Zyte proxy URL with authentication."""
    if not ZYTE_API_KEY:
        logger.warning("🔑 Zyte API key not found in environment")
        return None

    # Zyte proxy format: http://api_key:@api.zyte.com:8011/
    return f"http://{ZYTE_API_KEY}:@{ZYTE_PROXY_URL}/"


def get_random_proxy() -> Optional[str]:
    """Get traditional proxy URL (excludes Zyte which uses different format)."""
    # Fallback to traditional proxy list
    if not PROXY_LIST:
        logger.info("📡 No proxies available, using direct connection")
        return None

    # Pure random selection for better distribution and eBay blocking avoidance
    selected_proxy = random.choice(PROXY_LIST)
    return f"http://{selected_proxy['ip']}:{selected_proxy['port']}"


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
    attempts: int = 2,
    request_timeout: float = 25.0,
    jitter_min: float = 0.6,
    jitter_max: float = 1.5,
    use_proxy: bool = True,
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

    # Enhanced human-like jitter with random variation
    try:
        base_delay = random.uniform(float(jitter_min), float(jitter_max))
        # Add extra randomness to mimic human behavior
        human_variance = random.uniform(0.1, 0.4)
        total_delay = base_delay + human_variance
        await asyncio.sleep(total_delay)
    except Exception:
        await asyncio.sleep(random.uniform(0.5, 1.2))

    from src.utils.logger import setup_logger

    logger = setup_logger("chaamo.httpx")

    # Try proxy first with SSL fallback strategy
    connection_attempts = []
    if use_proxy:
        # Try Zyte proxy with SSL fallback strategy
        zyte_proxy = get_zyte_proxy()
        if zyte_proxy:
            # First try Zyte with SSL verification
            connection_attempts.append(("zyte_ssl", zyte_proxy))
            # Then try Zyte without SSL verification
            connection_attempts.append(("zyte_no_ssl", zyte_proxy))
        else:
            # Fallback to traditional proxies
            proxy_url = get_random_proxy()
            connection_attempts.append(("direct", None))
            if proxy_url:
                connection_attempts.append(("proxy", proxy_url))
            elif not PROXY_LIST:  # If proxy list is empty, fallback to direct
                httpx_logger.info(
                    "🔄 Proxy list is empty, using direct connection fallback"
                )

    # Always add direct connection as final fallback
    connection_attempts.append(("direct", None))

    for connection_type, proxy_config in connection_attempts:
        # Log connection attempt
        if connection_type == "zyte_ssl":
            httpx_logger.info("🔐 Use Zyte SSL verification")
        elif connection_type == "zyte_no_ssl":
            httpx_logger.info("🔓 Use Zyte without SSL verification")
        elif connection_type == "proxy":
            httpx_logger.info(f"🌐 Use Proxy {proxy_config}")
        else:
            httpx_logger.info("🔗 Use Direct")

        use_http2 = bool(random.getrandbits(1))
        limits = httpx.Limits(
            max_keepalive_connections=3, max_connections=10, keepalive_expiry=30.0
        )

        try:
            # Configure SSL verification and proxy settings
            verify_ssl = False
            proxy_setting = None

            if connection_type == "zyte_ssl":
                # Zyte proxy with SSL verification enabled
                verify_ssl = (
                    "/usr/local/share/ca-certificates/zyte-ca.crt"
                    if os.path.exists("/usr/local/share/ca-certificates/zyte-ca.crt")
                    else True
                )
                proxy_setting = proxy_config
            elif connection_type == "zyte_no_ssl":
                # Zyte proxy with SSL verification disabled
                verify_ssl = False
                proxy_setting = proxy_config
            elif connection_type == "proxy":
                # Traditional proxy without SSL verification
                verify_ssl = False
                proxy_setting = proxy_config

            async with httpx.AsyncClient(
                headers=headers,
                follow_redirects=True,
                http2=use_http2,
                timeout=httpx.Timeout(float(request_timeout)),
                limits=limits,
                trust_env=True,
                max_redirects=10,
                proxy=proxy_setting,
                verify=verify_ssl,
            ) as client:
                success = await _attempt_requests(
                    client, url, params, headers, attempts, connection_type
                )
                if success:
                    logger.info(f"✅ {connection_type} connection successful")
                    return success
        except Exception as e:
            logger.warning(f"⚠️ {connection_type} connection failed: {e}")
            continue

    # If all connection attempts failed
    logger.error(
        "❌ All connection attempts failed (Zyte SSL -> Zyte no-SSL -> Direct)"
    )
    return ""


async def _attempt_requests(client, url, params, headers, attempts, connection_type):
    """Attempt requests with retry logic for a specific client connection."""
    for attempt in range(1, attempts + 1):
        try:
            req = client.build_request("GET", url, params=params, headers=headers)
            logger.info(
                f"🔍 Scraping URL: {req.url} (attempt {attempt}/{attempts}) via {connection_type}"
            )
            response = await client.send(req)

            if response.status_code in (429, 403, 503):
                # Enhanced rate limiting handling with longer backoff
                headers["User-Agent"] = random.choice(USER_AGENTS)
                headers["Accept-Language"] = random.choice(ACCEPT_LANGUAGES)
                if response.status_code == 429:
                    backoff = min(20.0, (2.5**attempt)) + random.uniform(2.0, 5.0)
                    logger.warning(
                        f"⚠️ Rate limited (429), waiting {backoff:.1f}s before retry {attempt}"
                    )
                else:
                    backoff = min(12.0, (2.0**attempt)) + random.uniform(1.0, 3.0)
                    logger.warning(
                        f"⚠️ Access denied ({response.status_code}), waiting {backoff:.1f}s before retry {attempt}"
                    )
                await asyncio.sleep(backoff)
                continue

            response.raise_for_status()
            text = response.text

            # Handle interstitial redirect pages (meta refresh / JS redirects / Refresh header)
            for _ in range(2):
                # 1) HTTP Refresh header
                refresh = response.headers.get("Refresh") or response.headers.get(
                    "refresh"
                )
                redirect_url = None
                if refresh and "url=" in refresh.lower():
                    try:
                        part = refresh.split("url=", 1)[1].strip().strip("'\"")
                        redirect_url = urljoin(str(req.url), part)
                    except Exception:
                        redirect_url = None
                # 2) Meta refresh
                if not redirect_url:
                    m = re.search(
                        r"<meta[^>]*http-equiv=\s*\"?refresh\"?[^>]*content=\s*\"?\d+\s*;\s*url=([^\"'>]+)\"?",
                        text,
                        flags=re.I,
                    )
                    if m:
                        redirect_url = urljoin(str(req.url), m.group(1))
                # 3) JS location redirects
                if not redirect_url:
                    for pat in [
                        r"location\.href\s*=\s*['\"]([^'\"]+)['\"]",
                        r"window\.location\.replace\(\s*['\"]([^'\"]+)['\"]\s*\)",
                        r"location\.assign\(\s*['\"]([^'\"]+)['\"]\s*\)",
                    ]:
                        m = re.search(pat, text, flags=re.I)
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

            # Detect common bot-block/interstitial pages and retry with backoff
            block_indicators = [
                "verify you are a human",
                "please verify yourself",
                "access denied",
                "captcha",
                "to continue, please",
                "your browser will redirect to your requested content shortly",
                "please wait",
                "splashui/challenge",
                "ebay.com/splashui",
                "challenge?ap=1",
                "blocked for unusual activity",
                "automated requests",
            ]

            # Check both URL and content for challenge indicators
            current_url = str(response.url).lower()
            is_challenge = any(s in text.lower() for s in block_indicators) or any(
                s in current_url for s in ["challenge", "splash", "verify"]
            )

            if is_challenge:
                logger.info(
                    f"⚠️ Detected challenge/redirect page, retrying with new session (attempt {attempt})"
                )
                headers["User-Agent"] = random.choice(USER_AGENTS)
                headers.update(
                    {
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": (
                            "same-origin" if "ebay" in current_url else "none"
                        ),
                        "Sec-Fetch-User": "?1",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    }
                )
                backoff = min(12.0, (2.0**attempt)) + random.uniform(1.0, 3.0)
                await asyncio.sleep(backoff)
                continue

            return text

        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ProxyError):
            logger.info(f"⚠️ Connection/Proxy error on attempt {attempt}, retrying...")
            await asyncio.sleep(0.5 * attempt + random.uniform(0.1, 0.5))
            continue
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (403, 429):
                logger.info(
                    f"⚠️ HTTP {e.response.status_code} error, waiting before retry..."
                )
                await asyncio.sleep(1.0 * attempt + random.uniform(0.2, 0.6))
                continue
            raise
        except httpx.NetworkError as e:
            logger.info(f"⚠️ Zyte SSL error: retrying...")
            await asyncio.sleep(0.5 * attempt + random.uniform(0.2, 0.8))
            continue

    return None

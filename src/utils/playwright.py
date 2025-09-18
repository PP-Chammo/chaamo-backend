import asyncio
import random
import subprocess
import os
from typing import Optional
from urllib.parse import urlencode

# src/utils/playwright.py

# Optional imports; guard if Playwright not installed
try:
    import playwright_stealth
    from playwright_stealth import stealth as Stealth
except Exception:  # pragma: no cover
    playwright_stealth = None
    Stealth = None

try:
    from playwright.async_api import (
        async_playwright,
        TimeoutError as PlaywrightTimeoutError,
    )

    _HAS_PLAYWRIGHT = True
except Exception:  # pragma: no cover
    async_playwright = None
    PlaywrightTimeoutError = Exception
    _HAS_PLAYWRIGHT = False

from src.utils.logger import scraper_logger

# Set this to True for debug (shows browser window), False for normal headless mode
headless = True

logger = scraper_logger

# User agents and accept-languages
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
ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.9,es;q=0.8",
    "en-GB,en-US;q=0.9,en;q=0.8",
    "en-US,en;q=0.9,fr;q=0.8",
    "en-GB,en;q=0.9,de;q=0.8",
]

# module-scoped singletons for playwright and browser
_playwright_obj = None
_browser_obj = None

# tune concurrency: how many parallel pages we allow
_PAGE_SEMAPHORE = asyncio.Semaphore(4)


def _is_production() -> bool:
    env = (os.environ.get("ENV") or os.environ.get("APP_ENV") or "development").lower()
    return env == "production"


def is_playwright_installed() -> bool:
    # Chromium browser path detection (platform independent)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium
            path = getattr(browser, "executable_path", None)
            return bool(path and os.path.exists(path))
    except Exception:
        return False


def ensure_playwright_browsers():
    if _is_production():
        logger.info("Playwright disabled in production; skipping browser install")
        return
    if not is_playwright_installed():
        logger.info("🔧 Installing Playwright browsers (first run)...")
        subprocess.run(["playwright", "install", "chromium"], check=True)
    else:
        logger.info("Playwright browsers already installed.")


def get_random_headers() -> dict:
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": random.choice(ACCEPT_LANGUAGES),
    }
    return headers


async def _ensure_browser():
    """
    Start playwright and a single Chromium browser process (kept open).
    We create a new context per-request to allow varying user-agent / locale per request.
    """
    global _playwright_obj, _browser_obj
    if _is_production():
        raise RuntimeError("Playwright is disabled in production environment")
    if not _HAS_PLAYWRIGHT:
        raise RuntimeError(
            "Playwright is not installed or unavailable in this environment"
        )
    if _browser_obj:
        return
    _playwright_obj = await async_playwright().start()
    _browser_obj = await _playwright_obj.chromium.launch(headless=headless)


async def playwright_get_content(
    url: str,
    params: Optional[dict[str, str]] = None,
    headers: Optional[dict[str, str]] = None,
    *,
    attempts: int = 2,
    request_timeout: float = 25.0,
    jitter_min: float = 0.6,
    jitter_max: float = 1.5,
    use_proxy: bool = True,  # Ignored for playwright, kept for API compatibility
) -> str:
    """
    Fetch page content using playwright. Reuses browser process, creates a new context per request.
    We create a new context per-request to allow varying user-agent / locale per request.
    Returns HTML content string or raises last exception if all attempts fail.
    """
    if _is_production():
        raise RuntimeError("Playwright content fetch is disabled in production")
    if params is None:
        params = {}
    if headers is None:
        headers = get_random_headers()

    # Build URL with params
    if params:
        q = urlencode(params)
        full_url = f"{url}?{q}" if "?" not in url else f"{url}&{q}"
    else:
        full_url = url

    # Human-like jitter
    try:
        base_delay = random.uniform(float(jitter_min), float(jitter_max))
        human_variance = random.uniform(0.1, 0.4)
        total_delay = base_delay + human_variance
        await asyncio.sleep(total_delay)
    except Exception:
        await asyncio.sleep(random.uniform(0.5, 1.2))

    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            await _ensure_browser()
            async with _PAGE_SEMAPHORE:
                # create context per-request to allow custom UA/locale
                context = await _browser_obj.new_context(
                    user_agent=headers.get("User-Agent"),
                    locale=headers.get("Accept-Language", "en-US"),
                    viewport={
                        "width": random.randint(1200, 1920),
                        "height": random.randint(900, 1080),
                    },
                )
                try:
                    if Stealth is not None:
                        stealth = Stealth()
                        await stealth.apply_stealth_async(context)
                except Exception:
                    # stealth may fail sometimes; continue but log
                    logger.debug("stealth_async failed or skipped")

                page = await context.new_page()
                # Set extra headers (except User-Agent/Accept-Language handled above)
                extra_headers = {
                    k: v
                    for k, v in headers.items()
                    if k not in ("User-Agent", "Accept-Language")
                }
                if extra_headers:
                    await context.set_extra_http_headers(extra_headers)

                logger.info(f"🌐 Navigating to {full_url}")
                await page.goto(full_url, timeout=int(request_timeout * 1000))
                # small wait to allow JS to render
                await asyncio.sleep(random.uniform(0.2, 0.5))
                content = await page.content()
                await page.close()
                await context.close()
                return content
        except PlaywrightTimeoutError as e:
            logger.warning(f"⚠️ Playwright timeout on attempt {attempt}: {e}")
            last_exc = e
        except Exception as e:
            logger.warning(f"⚠️ Playwright error on attempt {attempt}: {e}")
            last_exc = e

        # Rotate user agent and language for next attempt
        headers = get_random_headers()
        await asyncio.sleep(random.uniform(0.5, 1.5))

    logger.error(f"❌ All playwright_get_content attempts failed for {url}")
    if last_exc:
        raise last_exc
    return ""


async def close_playwright():
    """Gracefully close playwright/browser. Call this on app shutdown."""
    global _playwright_obj, _browser_obj
    try:
        if _browser_obj:
            await _browser_obj.close()
        if _playwright_obj:
            await _playwright_obj.stop()
    finally:
        _playwright_obj = None
        _browser_obj = None

import subprocess
import random

from typing import Optional
from playwright.async_api import async_playwright
import os
import sys

headless = False

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
]

STEALTH_JS = '''
// Pass basic navigator checks
defineProperty = Object.defineProperty;
defineProperty(navigator, 'webdriver', {get: () => undefined});
window.chrome = { runtime: {} };
window.navigator.permissions.query = (parameters) => (
  parameters.name === 'notifications' ?
    Promise.resolve({ state: Notification.permission }) :
    window.navigator.permissions.query(parameters)
);
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
'''

selectors = [
    "h1:has-text('results for')",
    ".srp-controls__count-heading",
]

USER_DATA_DIR = os.environ.get("PLAYWRIGHT_USER_DATA_DIR", None)

def get_random_headers():
    # No-op for Playwright, kept for compatibility
    return {}

async def playwright_get_content(url: str, params: Optional[dict[str, str]] = None, headers: Optional[dict[str, str]] = None, user_data_dir: Optional[str] = None) -> str:
    if params:
        from urllib.parse import urlencode
        url = f"{url}?{urlencode(params)}"
    user_data = user_data_dir or USER_DATA_DIR
    async with async_playwright() as p:
        if user_data:
            browser = await p.chromium.launch_persistent_context(user_data, headless=headless)
            page = await browser.new_page()
        else:
            random_user_agent = random.choice(USER_AGENTS)
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context(user_agent=random_user_agent)
            page = await context.new_page()
        await page.add_init_script(STEALTH_JS)
        if headers:
            await page.set_extra_http_headers(headers)
        await page.goto(url, wait_until="domcontentloaded", timeout=90000)
        found = False
        for selector in selectors:
            try:
                await page.wait_for_selector(selector, timeout=5000)
                found = True
                break
            except:
                print("skipping")
                continue
        if not found:
            await browser.close()
            return ""
        html = await page.content()
        await browser.close()
        return html


def setup_playwright():
    try:
        import playwright
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
    subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)

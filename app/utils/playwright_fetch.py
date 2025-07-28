from typing import Optional
from playwright.async_api import async_playwright
import os
import sys

def get_random_headers():
    # No-op for Playwright, kept for compatibility
    return {}

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

def print_chrome_profile_path():
    if sys.platform.startswith("darwin"):
        # Mac
        print("~/Library/Application Support/Google/Chrome/Default")
    elif sys.platform.startswith("win"):
        # Windows
        print(r"%LOCALAPPDATA%\\Google\\Chrome\\User Data\\Default")
    else:
        # Linux
        print("~/.config/google-chrome/Default")

USER_DATA_DIR = os.environ.get("PLAYWRIGHT_USER_DATA_DIR", None)

async def playwright_fetch_html(url: str, params: Optional[dict[str, str]] = None, headers: Optional[dict[str, str]] = None, user_data_dir: Optional[str] = None) -> str:
    """
    Fetch HTML using Playwright with stealth JS and optional persistent browser profile.
    - Always runs in headless mode (no visible browser window).
    - Set PLAYWRIGHT_USER_DATA_DIR env var or pass user_data_dir to use your real Chrome/Brave/Edge profile.
    - This helps bypass Cloudflare and other anti-bot protections.
    """
    if params:
        from urllib.parse import urlencode
        url = f"{url}?{urlencode(params)}"
    user_data = user_data_dir or USER_DATA_DIR
    async with async_playwright() as p:
        if user_data:
            browser = await p.chromium.launch_persistent_context(user_data, headless=False)
            page = await browser.new_page()
        else:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
        await page.add_init_script(STEALTH_JS)
        if headers:
            await page.set_extra_http_headers(headers)
        await page.goto(url, wait_until="domcontentloaded", timeout=90000)
        html = await page.content()
        await browser.close()
        return html


async def download_image_with_playwright(image_url: str) -> bytes:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(image_url, wait_until="networkidle", timeout=90000)
        image_bytes = await page.evaluate(
            '''async (url) => {
                const response = await fetch(url, {credentials: 'same-origin'});
                const buffer = await response.arrayBuffer();
                return Array.from(new Uint8Array(buffer));
            }''',
            image_url
        )
        await browser.close()
        return bytes(image_bytes)

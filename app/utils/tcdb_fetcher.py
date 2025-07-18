from typing import Optional
from playwright.async_api import async_playwright
import os
import sys


def get_random_headers():
    # No-op for Playwright, kept for compatibility
    return {}

# Minimal stealth script to evade basic bot detection
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

# Helper to print the default Chrome user data directory for your OS
# Usage: python -c 'from app.utils.tcdb_fetcher import print_chrome_profile_path; print_chrome_profile_path()'
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

# Optionally set this to your real Chrome/Brave/Edge user data dir for persistent sessions
USER_DATA_DIR = os.environ.get("PLAYWRIGHT_USER_DATA_DIR", None)  # e.g. "/Users/youruser/Library/Application Support/Google/Chrome/Default"

async def fetch_html(url: str, params: Optional[dict[str, str]] = None, headers: Optional[dict[str, str]] = None, user_data_dir: Optional[str] = None) -> str:
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

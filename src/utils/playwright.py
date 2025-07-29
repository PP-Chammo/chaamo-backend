import subprocess
import random
from typing import Optional
from playwright.async_api import async_playwright
import os
import sys
import traceback
import base64

headless = True

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
]

STEALTH_JS = '''
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

# Selector yang lebih konsisten untuk eBay Sold Items
selectors = [
    "ul.srp-results",  # container utama hasil
    "li.s-item",       # satu item
    ".srp-controls__count-heading",  # total hasil
]

USER_DATA_DIR = os.environ.get("PLAYWRIGHT_USER_DATA_DIR", None)


def get_random_headers():
    return {}


async def playwright_get_content(
    url: str,
    params: Optional[dict[str, str]] = None,
    headers: Optional[dict[str, str]] = None,
    user_data_dir: Optional[str] = None
) -> str:
    if params:
        from urllib.parse import urlencode
        url = f"{url}?{urlencode(params)}"

    user_data = user_data_dir or USER_DATA_DIR
    random_user_agent = random.choice(USER_AGENTS)
    print(f"[INFO] Scraping URL: {url}")
    print(f"[INFO] User-Agent: {random_user_agent}")

    try:
        async with async_playwright() as p:
            if user_data:
                browser = await p.chromium.launch_persistent_context(
                    user_data,
                    headless=headless,
                    user_agent=random_user_agent
                )
                page = await browser.new_page()
            else:
                browser = await p.chromium.launch(
                    headless=headless,
                    args=[
                        "--headless=chrome",
                        "--no-sandbox",
                        "--disable-gpu",
                        "--disable-dev-shm-usage",
                        "--disable-setuid-sandbox",
                        "--no-zygote",
                        "--single-process"
                    ]
                )
                context = await browser.new_context(user_agent=random_user_agent)
                page = await context.new_page()

            page.on("console", lambda msg: print("[PAGE LOG]:", msg.text))

            await page.add_init_script(STEALTH_JS)

            await page.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1",
                **(headers or {})
            })

            try:
                print("[DEBUG] Navigating to page...")
                await page.goto(url, wait_until="domcontentloaded", timeout=90000)
                print("[DEBUG] Page navigation complete.")
            except Exception as e:
                print("[ERROR] Failed to navigate to page:")
                traceback.print_exc()
                await browser.close()
                return ""

            found = False
            for selector in selectors:
                try:
                    print(f"[DEBUG] Waiting for selector: {selector}")
                    await page.wait_for_selector(selector, timeout=7000)
                    found = True
                    print(f"[DEBUG] Selector found: {selector}")
                    break
                except Exception as e:
                    print(f"[WARN] Selector not found: {selector}")
                    traceback.print_exc()

            if not found:
                print("[WARN] No selector matched — saving debug info...")
                await page.screenshot(path="/tmp/debug.png", full_page=True)
                html_debug = await page.content()
                with open("/tmp/debug.html", "w") as f:
                    f.write(html_debug)
                print("[DEBUG] Saved /tmp/debug.png and debug.html")
                await browser.close()
                return ""

            html = await page.content()
            print(f"[DEBUG] HTML length: {len(html)}")
            await browser.close()
            return html

    except Exception as e:
        print("[ERROR] Exception during Playwright scraping:")
        traceback.print_exc()
        return ""


def setup_playwright():
    try:
        import playwright
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
    subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps"], check=True)

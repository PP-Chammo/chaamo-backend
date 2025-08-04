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
    print('----------')
    print(f"[INFO] Scraping URL: {url}")

    browser = None
    context = None
    page = None
    
    try:
        print("[DEBUG] Starting Playwright session...")
        async with async_playwright() as p:
            print("[DEBUG] Playwright instance created successfully")
            
            # Always use persistent context for better stability in containers
            print("[DEBUG] Using persistent context...")
            browser = await p.chromium.launch_persistent_context(
                "/tmp/playwright-user-data",
                headless=headless,
                user_agent=random_user_agent,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu"
                ],
                # Add context options for better redirect handling
                ignore_https_errors=True,
                extra_http_headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1"
                }
            )
            print("[DEBUG] Persistent context created successfully")
            
            print("[DEBUG] Creating new page...")
            page = await browser.new_page()
            print("[DEBUG] Page created successfully")

            page_errors = []

            page.on("console", lambda msg: (
                page_errors.append(msg.text)
                if msg.type == "error" else None
            ))

            await page.add_init_script(STEALTH_JS)
            print("[DEBUG] Stealth script added")

            await page.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1",
                **(headers or {})
            })
            print("[DEBUG] Headers set")

            # First, test basic internet connectivity
            try:
                print("[DEBUG] Testing internet connectivity...")
                await page.goto("https://httpbin.org/ip", wait_until="load", timeout=30000)
                ip_info = await page.content()
                print(f"[DEBUG] Internet connectivity test successful: {ip_info[:200]}...")
            except Exception as e:
                print(f"[ERROR] Internet connectivity test failed: {e}")
                return ""
            
            # Now navigate to the actual target URL with proper redirect handling
            try:
                print("[DEBUG] Navigating to target page...")
                # Use 'load' instead of 'domcontentloaded' to ensure redirects are followed
                await page.goto(url, wait_until="load", timeout=60000)
                print("[DEBUG] Page navigation complete.")
                
                # Wait a bit for any JavaScript redirects to complete
                await page.wait_for_timeout(2000)
                print("[DEBUG] Waited for potential JavaScript redirects.")
                
            except Exception as e:
                print(f"[ERROR] Failed to navigate to page: {e}")
                traceback.print_exc()
                return ""

            # Simple approach: get content immediately after navigation
            print("[DEBUG] Getting page content immediately...")
            try:
                html = await page.content()
                print(f"[DEBUG] Retrieved HTML content, length: {len(html)}")
                return html
            except Exception as e:
                print(f"[ERROR] Failed to get page content: {e}")
                # Try to get content using a different method
                try:
                    html = await page.evaluate("() => document.documentElement.outerHTML")
                    print(f"[DEBUG] Retrieved HTML content via evaluate, length: {len(html)}")
                    return html
                except Exception as e2:
                    print(f"[ERROR] Failed to get content via evaluate: {e2}")
                    return ""

    except Exception as e:
        print(f"[ERROR] Exception during Playwright scraping: {e}")
        traceback.print_exc()
        return ""
    finally:
        # Clean up resources
        try:
            if page:
                print("[DEBUG] Closing page...")
                await page.close()
        except Exception as e:
            print(f"[WARN] Error closing page: {e}")
            
        # Note: Persistent context (browser) is automatically closed when the async context exits
        print("[DEBUG] Cleanup completed")


def setup_playwright():
    try:
        import playwright
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
    subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps"], check=True)

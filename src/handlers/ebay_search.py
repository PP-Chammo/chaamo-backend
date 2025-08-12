import math
import re
from typing import Optional
from bs4 import BeautifulSoup
import dateutil.parser as date_parser
import pytz

from src.models.ebay import Region, base_target_url
from src.utils.supabase import supabase
from src.utils.httpx import httpx_get_content

def _region_code(region: Region | str) -> str:
    """Return canonical region code 'us' or 'uk' from Region enum or string."""
    return region.value if isinstance(region, Region) else region

async def ebay_search_handler(query: str, region: Region, master_card_id: Optional[str] = None):
    region_str = _region_code(region)
    url = f"{base_target_url[region_str]}/sch/i.html"
    
    try:
        total_pages = await get_ebay_page_count(query, region_str)
        print(f"Total pages: {total_pages}")
    except Exception as e:
        print(f"❌ Error getting page count: {e}")
        return {"error": "Failed to get page count", "details": str(e)}
    
    total_result = []
    for page in range(1, total_pages + 1):
        result_per_page = []
        print('----------')
        print(f"scraping page {page}...")
        params = {
            "_nkw": query,
            "_sacat": "0",
            "_from": "R40",
            "rt": "nc",
            "LH_Sold": "1",
            "LH_Complete": "1",
            "Country/Region of Manufacture": "United States" if region_str == "us" else "United Kingdom",
            "_ipg": 240,
            "_pgn": page
        }
        
        try:
            html_text = await httpx_get_content(url=url, params=params)
            if not html_text:
                print(f"⚠️  No HTML content received for page {page}")
                continue
        except Exception as e:
            print(f"❌ Error scraping page {page}: {e}")
            continue
        soup = BeautifulSoup(html_text, 'lxml')
        items = soup.select('li.s-item, li.s-card')
        for item in items:
            title_element = item.select_one('.s-item__title span, .s-card__title span')
            if not title_element or title_element.get_text() == "Shop on eBay":
                continue
            link_element = item.select_one('.s-item__link, .su-link')
            img_element = item.select_one('.s-item__image-wrapper img, .s-card__image-wrapper img, .image-treatment img')
            price_element = item.select_one('.s-item__price, .s-card__price')
            sold_date_element = item.select_one('.s-item__caption .POSITIVE, .s-card__caption .POSITIVE, .s-item__caption .positive, .s-card__caption .positive')
            if not all([title_element, img_element, link_element, price_element, sold_date_element]):
                continue
            post_id = link_element.get('href').split('?')[0].split('/')[-1]
            title = title_element.get_text()
            image_url = img_element.get('src') or img_element.get('data-src') or img_element.get('data-image-src')
            sold_at = get_sold_at_by_region(sold_date_element.get_text().replace('Sold', '').strip(), region_str)
            price_text = price_element.get_text().strip() if price_element else ''
            currency_match = re.search(r'([£$€])', price_text)
            currency = currency_match.group(1) if currency_match else '$'
            # Normalize price: remove symbols/commas, handle ranges like "12.34 to 56.78" or "12.34-56.78"
            normalized = re.sub(r'[£$€]', '', price_text)
            normalized = re.sub(r',', '', normalized)
            normalized = re.sub(r'(\d+(?:\.\d+)?)\s*(?:to|-)\s*[£$€]?\s*(\d+(?:\.\d+)?)', r'\1-\2', normalized, flags=re.I)
            nums = re.findall(r'\d+(?:\.\d+)?', normalized)
            if not nums:
                continue
            price = float(nums[-1])
            condition, grading_company, grade = get_condition_and_grade(title)
            post_url = link_element.get('href')
            result_per_page.append({
                "id": post_id,
                "master_card_id": master_card_id,
                "title": title,
                "image_url": image_url,
                "region": region_str,
                "sold_at": sold_at,
                "price": price,
                "currency": currency,
                "condition": condition,
                "grading_company": grading_company,
                "grade": grade,
                "post_url": post_url,
            })

        if len(result_per_page) > 0:
            try:
                ebay_posts_payload = deduplicate_by_id(result_per_page)
                if supabase:
                    res = supabase.table("ebay_posts").upsert(ebay_posts_payload, on_conflict="id").execute()
                    upserted = len(res.data) if hasattr(res, "data") and res.data is not None else len(ebay_posts_payload)
                    print(f"Upsert ebay_posts {upserted} rows")
                else:
                    print("Supabase client not initialized; skipping upsert")
                total_result += ebay_posts_payload
            except Exception as e:
                print(f"Supabase upsert ebay_posts error: {e}")

    print(f"Found {len(total_result)} posts")
    return {
        "total": len(total_result),
        "result": total_result,
    }




async def get_ebay_page_count(query: str, region: Region | str) -> int:
    region_str = _region_code(region)
    url = f"{base_target_url[region_str]}/sch/i.html"
    params = {
        "_nkw": query,
        "_sacat": "0",
        "_from": "R40",
        "rt": "nc",
        "LH_Sold": "1",
        "LH_Complete": "1",
        "Country/Region of Manufacture": "United States" if region_str == "us" else "United Kingdom",
        "_pgn": "1"
    }
    html_text = await httpx_get_content(url=url, params=params)
    soup = BeautifulSoup(html_text, 'lxml')
    count_container = soup.select_one(".srp-controls__count-heading, .result-count__count-heading")
    if not count_container:
        return 1
    bold = count_container.find("span", class_="BOLD")
    raw_text = (bold.get_text(strip=True) if bold else count_container.get_text(" ", strip=True))
    # Extract first integer (handles commas)
    m = re.search(r"(\d[\d,]*)", raw_text)
    if not m:
        return 1
    total_results = int(m.group(1).replace(",", ""))
    total_pages = max(1, math.ceil(total_results / 240))
    return total_pages


def deduplicate_by_id(records: list[dict]) -> list[dict]:
    seen = {}
    for record in records:
        seen[record["id"]] = record
    return list(seen.values())


def get_condition_and_grade(title: str):
    grading_companies = ["PSA", "BGS", "SGC", "CGC", "CSG", "HGA", "GMA", "Beckett"]
    condition = "raw"
    grading_company = None
    grade = None
    for company in grading_companies:
        if company in title.upper():
            condition = "graded"
            grading_company = company
            match = re.search(rf"{company} ?([0-9]{{1,2}}(?:\.[0-9])?)", title.upper())
            if match:
                grade = match.group(1)
            break
    return condition, grading_company, grade


def get_sold_at_by_region(sold_at_raw, region: Region | str):
    region_str = _region_code(region)
    try:
        dt = date_parser.parse(sold_at_raw, fuzzy=True, dayfirst=True)
        tz = pytz.timezone('Europe/London' if region_str == 'uk' else 'America/New_York')
        dt = dt if dt.tzinfo else tz.localize(dt)
        return dt.isoformat()
    except Exception:
        return sold_at_raw

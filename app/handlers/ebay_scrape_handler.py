import re
import os
import aiofiles
import pytz
from dateutil import parser as date_parser

from bs4 import BeautifulSoup
from typing import List, Dict, Any

from app.services.supabase import supabase
from app.utils.httpx_fetch import httpx_fetch_html
from app.utils.playwright_fetch import download_image_with_playwright
from app.utils.compare_image import compare_image_by_url

base_target_url = {
    "us": "https://www.ebay.com",
    "uk": "https://www.ebay.co.uk"
}

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
    "DNT": "1"
}

def parse_condition_and_grade(title: str):
    # Simple logic: if 'PSA', 'BGS', 'SGC', 'CGC', 'CSG', 'HGA', 'GMA', 'Beckett' in title, it's graded
    # Otherwise, raw
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

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

def parse_sold_at(sold_at_raw, region):
    try:
        dt = date_parser.parse(sold_at_raw, fuzzy=True, dayfirst=True)
        tz = pytz.timezone('Europe/London' if region == 'uk' else 'America/New_York')
        dt = dt if dt.tzinfo else tz.localize(dt)
        return dt.isoformat()
    except Exception:
        return sold_at_raw

async def discover_solds(query: str, region: str, page: str) -> List[Dict[str, Any]]:
    master_card_id = None
    canonical_image_path = None
    response = (supabase.table("master_cards").select("id, canonical_image_url").eq("name", query).limit(1).execute())
    if response.data and len(response.data) > 0:
        master_card_id = response.data[0]["id"]
    if response.data[0]["canonical_image_url"]:
        image_bytes = await download_image_with_playwright(response.data[0]["canonical_image_url"])
        temp_dir = os.path.join(os.path.dirname(__file__), '../../temp')
        os.makedirs(temp_dir, exist_ok=True)
        filename = sanitize_filename(query) + '.jpg'
        print(filename)
        file_path = os.path.join(temp_dir, filename)
        print(file_path)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(image_bytes)
        canonical_image_path = file_path
    url = f"{base_target_url[region]}/sch/i.html"
    params = {
        "_nkw": query,
        "_sacat": "0",
        "_from": "R40",
        "rt": "nc",
        "LH_Sold": "1",
        "LH_Complete": "1",
        "Country/Region of Manufacture": "United States" if region == "us" else "United Kingdom",
        "_pgn": page
    }
    html_text = await httpx_fetch_html(url=url, params=params)
    soup = BeautifulSoup(html_text, 'lxml')
    items = soup.select('li.s-item, li.s-card')
    result = []
    for item in items:
        title_element = item.select_one('.s-item__title span, .s-card__title span')
        print(title_element)
        if not title_element or title_element.get_text() == "Shop on eBay":
            continue
        title = title_element.get_text()
        link_element = item.select_one('.s-item__link, .su-link')
        img_element = item.select_one('.s-item__image-wrapper img, .s-card__image-wrapper img, .image-treatment img')
        price_element = item.select_one('.s-item__price, .s-card__price')
        sold_date_element = item.select_one('.s-item__caption .POSITIVE, .s-card__caption .POSITIVE, .s-item__caption .positive, .s-card__caption .positive')
        if not all([title_element, img_element, link_element, price_element, sold_date_element]):
            continue
        if canonical_image_path and img_element:
            compared_result = compare_image_by_url(canonical_image_path, img_element.get('src'))
            compared_score = float(compared_result['score']) if compared_result['score'] else 0
            if compared_result["match"] == False:
                print(f"❌ {format(compared_score * 100, '.2f')}% --- {img_element.get('src')}")
                continue
            else:
                status = '✅' if compared_result['match'] == True else '⚠️ '
                print(f"{status} {format(compared_score * 100, '.2f')}% --- {img_element.get('src')}")
            price_text = price_element.get_text().strip() if price_element else ''
            currency_match = re.search(r'([£$€])', price_text)
            currency = currency_match.group(1) if currency_match else '$'
            item_id = link_element.get('href').split('?')[0].split('/')[-1]
            cleaned_price = re.sub(r'(\d+\.\d+)\s+to\s+[£$€]?(\d+\.\d+)', r'\1-\2', re.sub(r'[£$€]', '', price_text))
            cleaned_price = float(re.findall(r'\d+\.\d+', cleaned_price)[-1])
            sold_at = parse_sold_at(sold_date_element.get_text().replace('Sold', '').strip(), region)
            condition, grading_company, grade = parse_condition_and_grade(title)
            result.append({
                "id": item_id,
                "master_card_id": master_card_id,
                "source_image_url": img_element.get('src'),
                "region": region,
                "sold_at": sold_at,
                "price": cleaned_price,
                "currency": currency,
                "condition": condition,
                "grading_company": grading_company,
                "grade": grade,
                "source_listing_url": link_element.get('href'),
            })
        else:
            if not canonical_image_path:
                print("No canonical image path found, skipping")
            else:
                print(f"No match {img_element.get('src')}, skipping")

    if result:
        upserts = []
        for sold_obj in result:
            upserts.append(sold_obj)
        try:
            res = (supabase.table("scraped_sold_history").upsert(upserts, on_conflict="id").execute())
            print(f"Upsert scraped_sold_history {len(res.data)} rows")
        except Exception as e:
            print(f"Supabase upsert scraped_sold_history error: {e}")
    print(f"Found {len(result)} sold items")
    return result

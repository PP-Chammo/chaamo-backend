import math
import re
import asyncio
from typing import Optional, List, Dict
from src.models.category import CategoryId
from bs4 import BeautifulSoup
import dateutil.parser as date_parser
import pytz

from src.models.ebay import Region, base_target_url
from src.utils.supabase import supabase
from src.utils.httpx import httpx_get_content

# ---------------- Main scraper ----------------
async def ebay_scrape_handler(
    region: Region,
    category_id: Optional[CategoryId] = None,
    query: Optional[str] = None,
    user_card_id: Optional[str] = None,
    max_pages: int = 50,
    page_retries: int = 5,
):
    region_str = _region_code(region)
    base_url = f"{base_target_url[region_str]}/sch/i.html"

    effective_query = _resolve_query(query, user_card_id)
    if not effective_query:
        return {"error": "Missing query", "details": "Provide query or a valid user_card_id with a name"}

    # fetch master_card_id once
    mc_id = None
    user_card_category_id = category_id
    if user_card_id and supabase:
        try:
            sel = supabase.table("user_cards").select("master_card_id, category_id").eq("id", user_card_id).limit(1).execute()
            data = getattr(sel, "data", None) or []
            if data:
                mc_id = data[0].get("master_card_id")
                user_card_category_id = data[0].get("category_id")
        except Exception as e:
            print(f"⚠️ Failed to fetch master_card_id from user_cards: {e}")

    # Use provided category_id or fallback to user_card's category_id
    effective_category_id = category_id.value if category_id else user_card_category_id

    # determine total pages. If max_pages provided, skip expensive count call
    try:
        mp = int(max_pages)
    except Exception:
        mp = 1
    if mp and mp >= 1:
        total_pages = mp
    else:
        try:
            total_pages = await get_ebay_page_count(effective_query, region_str)
        except Exception as e:
            print(f"❌ Error getting page count: {e}")
            total_pages = 1

    # scrape pages in parallel with retry
    tasks = [
        scrape_page_with_retry(page, total_pages, base_url, effective_query, mc_id, effective_category_id, region_str, retries=page_retries)
        for page in range(1, total_pages + 1)
    ]
    results = await asyncio.gather(*tasks)
    total_result = [item for sublist in results for item in sublist if sublist]

    print(f"Found total {len(total_result)} posts (before dedupe)")
    total_result = deduplicate_by_id(total_result)

    if not total_result:
        # gagal total -> update last_sold_price=0 + log
        if user_card_id and supabase:
            try:
                supabase.table("user_cards").update({"last_sold_price": 0, "last_sold_currency": "GBP" if region == "uk" else "USD"}).eq("id", user_card_id).execute()
                supabase.table("failed_scrape_cards").insert({
                    "user_card_id": user_card_id,
                    "name": effective_query
                }).execute()
                print(f"⚠️ Logged failed scrape for user_card_id={user_card_id}")
            except Exception as e:
                print(f"❌ Failed to log failed scrape: {e}")
        return {"total": 0, "result": []}

    # Enrich a subset with HD images from listing pages to improve quality (disabled by default)
    enable_hd_enrich = False
    if enable_hd_enrich:
        try:
            await enrich_records_with_hd_images(total_result, max_concurrency=10, max_items=200)
        except Exception as e:
            print(f"⚠️ Image enrichment step failed: {e}")

    # Upsert in batches to avoid statement timeouts
    try:
        upsert_ebay_posts_chunked(total_result, user_card_id)
    except Exception as e:
        print(f"❌ Final upsert error (batched): {e}")

    # update user_cards.last_sold_* from latest sold_at
    if user_card_id and supabase and total_result:
        try:
            latest = max(total_result, key=lambda rec: safe_parse_date(rec.get("sold_at")))
            update_payload = {
                "last_sold_currency": latest.get("currency"),
                "last_sold_price": latest.get("price"),
                "last_sold_post_url": latest.get("post_url"),
                "last_sold_at": latest.get("sold_at")
            }
            supabase.table("user_cards").update(update_payload).eq("id", user_card_id).execute()
            print(f"✅ Updated user_cards {user_card_id} with last sold info")
        except Exception as e:
            print(f"❌ Failed to update user_cards last sold: {e}")

    return {"total": len(total_result), "result": total_result}


# ---------------- Page scraper with retry ----------------
async def scrape_page_with_retry(page, total_pages, base_url, query, mc_id, category_id, region_str, retries: int = 5):
    for attempt in range(1, retries + 1):
        try:
            print(f"Scraping page {page}/{total_pages} (attempt {attempt}) ...")
            params = {
                "_nkw": query,
                "_sacat": "0",
                "_from": "R40",
                "rt": "nc",
                "LH_Sold": "1",
                "LH_Complete": "1",
                "Country/Region of Manufacture": "United States" if region_str == "us" else "United Kingdom",
                "_ipg": 240,
                "_pgn": page,
                "_sop": 13
            }
            html_text = await httpx_get_content(
                url=base_url,
                params=params,
                attempts=retries,
                request_timeout=10.0,
                jitter_min=0.2,
                jitter_max=0.6,
            )
            if not html_text:
                raise ValueError("Empty HTML content")

            soup = BeautifulSoup(html_text, "lxml")
            items = soup.select("li.s-item, li.s-card")
            page_records = []

            for item in items:
                title_el = item.select_one(".s-item__title, .s-card__title")
                if not title_el:
                    continue
                title = title_el.get_text(" ", strip=True)
                if title.lower().strip() in ("", "shop on ebay"):
                    continue

                link_el = item.select_one(".s-item__link, .su-link")
                if not link_el or not link_el.get("href"):
                    continue
                post_url = link_el.get("href")
                post_id = extract_post_id_from_url(post_url)

                img_el = item.select_one(".s-item__image-wrapper img, .s-card__image-wrapper img, .image-treatment img")
                image_url = safe_img_src(img_el)
                image_hd_url = upgrade_ebay_img_to_hd(image_url)

                price_el = item.select_one(".s-item__price, .s-card__price")
                price_text = price_el.get_text(" ", strip=True) if price_el else ""
                price_value, currency = parse_price_and_value(price_text, region_str)
                if price_value is None:
                    continue

                sold_caption = item.select_one(".s-item__caption .POSITIVE, .s-card__caption .POSITIVE, .s-item__caption .positive, .s-card__caption .positive")
                raw_sold = sold_caption.get_text(" ", strip=True) if sold_caption else ""
                if not raw_sold and "sold" not in item.get_text(" ").lower():
                    continue
                sold_at = get_sold_at_by_region(raw_sold.replace("Sold", "").strip(), region_str)

                condition, grading_company, grade = get_condition_and_grade(title)

                page_records.append({
                    "id": post_id,
                    "master_card_id": mc_id,
                    "name": title,
                    "image_url": image_url,
                    "image_hd_url": image_hd_url,
                    "region": region_str,
                    "sold_at": sold_at,
                    "price": price_value,
                    "currency": currency,
                    "condition": condition,
                    "grading_company": grading_company,
                    "grade": grade,
                    "post_url": post_url,
                    "category_id": category_id
                })

            if page_records:
                return deduplicate_by_id(page_records)

        except Exception as e:
            print(f"⚠️ Error scraping page {page} attempt {attempt}: {e}")

    print(f"❌ Failed to scrape page {page} after {retries} attempts")
    return []


# ---------------- Helpers ----------------
def _region_code(region: Region | str) -> str:
    return region.value if hasattr(region, "value") else region

def _currency_code_from_text(text: str, region_hint: str) -> str:
    if not text:
        return "USD" if region_hint == "us" else "GBP"
    t = text.upper()
    if "CA$" in t or "C$" in t or "CAD" in t:
        return "CAD"
    if "AU$" in t or "A$" in t or "AUD" in t:
        return "AUD"
    if "£" in text:
        return "GBP"
    if "€" in text:
        return "EUR"
    if "$" in text:
        return "USD"
    return "USD" if region_hint == "us" else "GBP"

def _resolve_query(query: Optional[str], user_card_id: Optional[str]) -> Optional[str]:
    if query and query.strip():
        return query.strip()
    if user_card_id and supabase:
        try:
            res = supabase.table("user_cards").select("custom_name, master_card_id").eq("id", user_card_id).limit(1).execute()
            data = getattr(res, "data", None) or []
            if data:
                uc = data[0]
                master_card_id = uc.get("master_card_id")
                custom_name = uc.get("custom_name")
                if master_card_id:
                    mc_res = supabase.table("master_cards").select("name").eq("id", master_card_id).limit(1).execute()
                    mc_data = getattr(mc_res, "data", None) or []
                    if mc_data and mc_data[0].get("name"):
                        return mc_data[0]["name"].strip()
                if custom_name:
                    return custom_name.strip()
        except Exception as e:
            print(f"⚠️ Failed to resolve query from user_cards: {e}")
    return None

def parse_price_and_value(price_text: str, region_hint: str):
    if not price_text:
        return None, _currency_code_from_text("", region_hint)
    txt = price_text.replace("\u00A0", " ")
    currency = _currency_code_from_text(txt, region_hint)
    cleaned = re.sub(r'(?i)(GBP|USD|CAD|AUD|EUR|CA\$|AU\$|\$|£|€)', ' ', txt)
    cleaned = re.sub(r',', '', cleaned)
    nums = re.findall(r'\d+(?:\.\d+)?', cleaned)
    if not nums:
        return None, currency
    try:
        return float(nums[-1]), currency
    except Exception:
        return None, currency

def extract_post_id_from_url(url: str) -> str:
    if not url:
        return ""
    m = re.search(r'/itm/(?:[^/]+/)?(\d+)', url)
    if m:
        return m.group(1)
    return url.split('?')[0].rstrip('/').split('/')[-1]

def safe_img_src(img_el) -> Optional[str]:
    if not img_el:
        return None
    for attr in ("src", "data-src", "data-image-src", "srcset"):
        v = img_el.get(attr)
        if v:
            if attr == "srcset":
                return v.split(",")[0].split()[0]
            return v
    return None

def deduplicate_by_id(records: List[Dict]) -> List[Dict]:
    seen = {}
    for r in records:
        seen[r["id"]] = r
    return list(seen.values())

def upsert_ebay_posts_chunked(records: List[Dict], user_card_id: Optional[str] = None, initial_batch_size: int = 2000):
    """
    Upsert `records` into `ebay_posts` in batches, merging user_card_id into user_card_ids per-chunk.
    On statement timeout or other errors, the chunk is split into halves and retried, down to single-record upserts.
    """
    if not supabase or not records:
        return

    def merge_user_ids_for_chunk(chunk: List[Dict]):
        if not user_card_id:
            return
        ids = [p.get("id") for p in chunk if p.get("id")]
        if not ids:
            return
        try:
            existing_res = supabase.table("ebay_posts").select("id,user_card_ids").in_("id", ids).execute()
            existing_map = {r["id"]: (r.get("user_card_ids") or []) for r in (getattr(existing_res, "data", None) or [])}
        except Exception as e:
            print(f"⚠️ Failed to fetch existing user_card_ids for chunk: {e}")
            existing_map = {}

        for p in chunk:
            cur = [x for x in (existing_map.get(p.get("id"), []) or []) if x]
            if user_card_id and user_card_id not in cur:
                cur.append(user_card_id)
            p["user_card_ids"] = cur

    def process_chunk(chunk: List[Dict], batch_size: int):
        # If the chunk is larger than batch_size, split into sub-batches first
        if len(chunk) > batch_size:
            for i in range(0, len(chunk), batch_size):
                process_chunk(chunk[i:i+batch_size], batch_size)
            return

        # Merge user_card_ids for this chunk before upsert
        merge_user_ids_for_chunk(chunk)

        try:
            res = supabase.table("ebay_posts").upsert(chunk, on_conflict="id").execute()
            print(f"✅ Upserted chunk of {len(getattr(res, 'data', chunk))} rows")
        except Exception as e:
            msg = str(e)
            print(f"⚠️ Upsert error on chunk size {len(chunk)}: {e}")
            # On statement timeout or similar, split and retry; otherwise also split as fallback
            if len(chunk) <= 1:
                print("❌ Skipping single-record chunk due to repeated error")
                return
            mid = len(chunk) // 2
            left = chunk[:mid]
            right = chunk[mid:]
            # Reduce batch size to be conservative after an error
            smaller_batch = max(1, batch_size // 2)
            process_chunk(left, smaller_batch)
            process_chunk(right, smaller_batch)

    process_chunk(records, initial_batch_size)

def get_sold_at_by_region(sold_at_raw, region: Region | str):
    region_str = _region_code(region)
    try:
        dt = date_parser.parse(sold_at_raw, fuzzy=True, dayfirst=True)
        tz = pytz.timezone('Europe/London' if region_str == 'uk' else 'America/New_York')
        dt = dt if dt.tzinfo else tz.localize(dt)
        return dt.isoformat()
    except Exception:
        return sold_at_raw

def get_condition_and_grade(title: str):
    grading_companies = ["PSA", "BGS", "SGC", "CGC", "CSG", "HGA", "GMA", "Beckett"]
    condition = "raw"
    grading_company = None
    grade = None
    for company in grading_companies:
        if company in (title or "").upper():
            condition = "graded"
            grading_company = company
            m = re.search(rf"{company} ?([0-9]{{1,2}}(?:\.[0-9])?)", (title or "").upper())
            if m:
                grade = m.group(1)
            break
    return condition, grading_company, grade

def safe_parse_date(v):
    try:
        return date_parser.parse(v)
    except Exception:
        return date_parser.parse("1970-01-01T00:00:00Z")

def upgrade_ebay_img_to_hd(url: Optional[str]) -> Optional[str]:
    """Normalize i.ebayimg.com URLs to s-l1600 while preserving extension."""
    if not url:
        return None
    try:
        if "i.ebayimg.com" in url:
            return re.sub(r"/s-l[0-9]{2,4}", "/s-l1600", url)
    except Exception:
        return url
    return url

def extract_hd_image_from_listing_html(html: str) -> Optional[str]:
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "lxml")
        # Prefer explicit meta/link tags
        candidates = []
        og = soup.find("meta", attrs={"property": "og:image"})
        if og and og.get("content"):
            candidates.append(og.get("content"))
        tw = soup.find("meta", attrs={"name": "twitter:image"})
        if tw and tw.get("content"):
            candidates.append(tw.get("content"))
        link_img = soup.find("link", attrs={"rel": ["image_src", "preload", "canonical"]})
        if link_img and link_img.get("href"):
            candidates.append(link_img.get("href"))

        for c in candidates:
            if c and "i.ebayimg.com" in c:
                return c

        # Fallback: regex scan for any i.ebayimg.com URL in the HTML
        matches = re.findall(r"https?://i\\.ebayimg\\.com/[^'\"<>\\s]+", html)
        if not matches:
            return None
        # Prefer URLs that already contain s-l size token
        for m in matches:
            if "/s-l" in m:
                return m
        return matches[0]
    except Exception:
        return None

async def enrich_records_with_hd_images(records: List[Dict], max_concurrency: int = 8, max_items: int = 200):
    sem = asyncio.Semaphore(max_concurrency)

    async def _enrich(rec: Dict):
        if not rec or rec.get("image_hd_url"):
            return
        post_url = rec.get("post_url")
        if not post_url:
            return
        async with sem:
            html = await httpx_get_content(url=post_url)
            if not html:
                return
            hd = extract_hd_image_from_listing_html(html)
            if hd:
                rec["image_hd_url"] = upgrade_ebay_img_to_hd(hd)

    # Limit scope to avoid too many requests
    work = [r for r in records if r and r.get("post_url")][: max(0, max_items)]
    await asyncio.gather(*[ _enrich(r) for r in work ])

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
    if not html_text:
        return 1
    soup = BeautifulSoup(html_text, "lxml")
    count_container = soup.select_one(".srp-controls__count-heading, .result-count__count-heading")
    if not count_container:
        return 1
    bold = count_container.find("span", class_="BOLD")
    raw_text = (bold.get_text(strip=True) if bold else count_container.get_text(" ", strip=True))
    m = re.search(r"(\d[\d,]*)", raw_text)
    if not m:
        return 1
    total_results = int(m.group(1).replace(",", ""))
    return max(1, math.ceil(total_results / 240))

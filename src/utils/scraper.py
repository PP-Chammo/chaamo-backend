import os
import re
import math
import json
import pytz
import httpx
import random
import asyncio
import difflib
import unicodedata
import dateutil.parser as date_parser
from typing import Optional, Any, Dict, List
from rapidfuzz import fuzz
from typing import Optional, Any, Dict, List
from bs4 import BeautifulSoup

from urllib.parse import urlencode
from src.models.category import CategoryId
from src.models.ebay import Region, base_target_url
from src.utils.httpx import httpx_get_content
from src.utils.supabase import supabase
from src.utils.logger import scraper_logger


# ===============================================================
# Extract year range
# ===============================================================
def extract_year_range(text: str) -> list[str]:
    # extract year pattern at the start, can be 1994 or 1994-96
    m = re.match(r"(\d{4})(?:-(\d{2,4}))?", text)
    if not m:
        return []

    start = int(m.group(1))
    end_str = m.group(2)

    if end_str:
        if len(end_str) == 2:
            end = int(str(start)[:2] + end_str)
        else:
            end = int(end_str)
        return [str(y) for y in range(start, end + 1)]
    else:
        return [str(start)]


# ===============================================================
# Upsert card_sets (supabase)
# ===============================================================
async def upsert_card_sets(card_sets: list[dict[str, any]]) -> dict[str, any]:
    """
    Upsert card_sets into supabase table `card_sets`.
    Return:
      {
        "count": int,            # number of records returned by supabase (accumulated)
        "records": List[Dict],   # list of returned representations
        "errors": List[str]      # errors, if any
      }
    """
    results = {"count": 0, "records": [], "errors": []}
    if not card_sets:
        return results

    try:
        db_records = []
        for card_set in card_sets:
            db_record = {
                "name": card_set.get("name", ""),
                "category_id": card_set.get("category_id", ""),
                "platform": card_set.get("platform", ""),
                "platform_set_id": card_set.get("platform_set_id", ""),
                "years": card_set.get("years", []),
                "link": card_set.get("link", ""),
                "browse_type": card_set.get("browse_type", ""),
            }
            db_records.append(db_record)

        # upsert in batches, accumulate returned representations
        BATCH = 500
        for i in range(0, len(db_records), BATCH):
            batch = db_records[i : i + BATCH]
            resp = (
                supabase.table("card_sets")
                .upsert(
                    batch, on_conflict="platform_set_id", returning="representation"
                )
                .execute()
            )
            batch_records = resp.data or []
            results["records"].extend(batch_records)
            results["count"] += len(batch_records)

        scraper_logger.info(f"✅ Upserted {results['count']} sets to card_sets")
        return results
    except Exception as e:
        results["errors"].append(str(e))
        scraper_logger.error(f"❌ Failed to upsert card_sets: {e}")
        return results


# ===============================================================
# Upsert master_cards (supabase)
# ===============================================================
async def upsert_master_cards(cards: list[dict[str, any]]) -> dict[str, any]:
    """
    Upsert cards into supabase table `master_cards`.
    Return structure same as upsert_card_sets: {count, records, errors}
    """
    results = {"count": 0, "records": [], "errors": []}
    if not cards:
        return results

    try:
        db_records = []
        for card in cards:
            years = card.get("years", []) or []
            db_record = {
                "category_id": card.get("category_id", ""),
                "set_id": card.get("set_id", ""),
                "platform": card.get("platform", ""),
                "platform_card_id": card.get("platform_card_id", ""),
                "name": card.get("name", ""),
                "card_number": card.get("card_number", ""),
                "canonical_image_url": card.get("canonical_image_url", ""),
                "link": card.get("link", ""),
                "attributes": card.get("attributes", {}),
                "years": years,
                "year": years[0] if years else None,
            }
            db_records.append(db_record)

        # upsert in batches, accumulate returned representations
        BATCH = 500
        for i in range(0, len(db_records), BATCH):
            batch = db_records[i : i + BATCH]
            resp = (
                supabase.table("master_cards")
                .upsert(
                    batch, on_conflict="platform_card_id", returning="representation"
                )
                .execute()
            )
            batch_records = resp.data or []
            results["records"].extend(batch_records)
            results["count"] += len(batch_records)

        scraper_logger.info(f"✅ Upserted {results['count']} cards to master_cards")
        return results
    except Exception as e:
        results["errors"].append(str(e))
        scraper_logger.error(f"❌ Failed to upsert master_cards: {e}")
        return results


# ===============================================================
# Get HTML Content
# ===============================================================
async def scrape_ebay_html(
    region: Region,
    query: str | None = None,
    category_id: CategoryId | None = None,
    card_id: str | None = None,
    max_pages: int = 50,
    page_retries: int = 3,
    disable_proxy: bool = False,
) -> list[str]:
    scraper_logger.info(
        f"[scrape_ebay_html] params: query={query}, category_id={category_id}, card_id={card_id}"
    )
    # (same logic as before) - build config, request first page to determine total pages, loop pages
    final_query = query
    final_category_id = category_id

    # -----------------------------------------------
    # Mode: card_id
    # -----------------------------------------------
    if card_id:
        try:
            response = (
                supabase.table("cards")
                .select("canonical_title, category_id")
                .eq("id", card_id)
                .limit(1)
                .execute()
            )
            if response.data:
                card_data = response.data[0]
                final_query = card_data.get("canonical_title")
                final_category_id = (
                    CategoryId(card_data.get("category_id"))
                    if card_data.get("category_id")
                    else None
                )
                scraper_logger.info(
                    f"✅ [Scraping Mode: card_id] ------------------------------------"
                )
        except Exception as e:
            scraper_logger.warning(f"⚠️ Error resolving card_id {card_id}: {e}")
    # -----------------------------------------------
    # Mode: query + category_id
    # -----------------------------------------------
    if query and category_id:
        scraper_logger.info(
            f"✅ [Scraping Mode:] ------------------ query + category_id ------------------"
        )

    config = {
        "query": final_query,
        "category_id": final_category_id.value if final_category_id else None,
        "region_str": region.value,
        "base_url": f"{base_target_url[region.value]}/sch/i.html",
    }

    total_pages = 1
    if max_pages > 1:
        try:
            params = {
                "_nkw": config["query"],
                "_sacat": "0",
                "_from": "R40",
                "rt": "nc",
                "LH_Sold": "1",
                "LH_Complete": "1",
                "Country/Region of Manufacture": (
                    "United Kingdom"
                    if config["region_str"] == "uk"
                    else "United States"
                ),
                "_ipg": "240",
                "_pgn": "1",
                "_sop": "12",
            }
            scraper_logger.info(
                f"📡 [Scraping eBay start] with query --- {config['base_url']}?{urlencode(params)}"
            )
            html = await httpx_get_content(
                config["base_url"], params=params, use_proxy=not disable_proxy
            )
            if html:
                soup = BeautifulSoup(html, "html.parser")
                count_element = soup.select_one(
                    ".srp-controls__count-heading, .result-count__count-heading"
                )
                if count_element:
                    bold = count_element.find("span", class_="BOLD")
                    text = (
                        bold.get_text(strip=True)
                        if bold
                        else count_element.get_text(" ", strip=True)
                    )
                    match = re.search(r"(\d[\d,]*)", text)
                    if match:
                        total_results = int(match.group(1).replace(",", ""))
                        total_pages = max(1, math.ceil(total_results / 240))
        except Exception as e:
            scraper_logger.warning(f"⚠️ Error getting page count: {e}")

    total_pages = min(max_pages, total_pages)
    scraper_logger.info(f"🚀 [Scraping {total_pages} pages] ---- '{final_query}'")

    html_pages = []
    for page in range(1, total_pages + 1):
        if page > 1:
            delay = random.uniform(1.0, 3.5) + (page - 1) * random.uniform(0.2, 0.8)
            await asyncio.sleep(delay)

        page_html = None
        for attempt in range(1, page_retries + 1):
            try:
                params = {
                    "_nkw": config["query"],
                    "_sacat": (
                        str(config["category_id"]) if config["category_id"] else "0"
                    ),
                    "_from": "R40",
                    "rt": "nc",
                    "LH_Sold": "1",
                    "LH_Complete": "1",
                    "Country/Region of Manufacture": (
                        "United Kingdom"
                        if config["region_str"] == "uk"
                        else "United States"
                    ),
                    "_ipg": "240",
                    "_pgn": str(page),
                    "_sop": "12",
                }
                scraper_logger.info(
                    f"📡 [eBay URL] --- {config['base_url']}?{urlencode(params)}"
                )
                timeout = random.uniform(12.0, 18.0)
                jitter_min = random.uniform(0.5, 1.2)
                jitter_max = random.uniform(1.5, 3.0)
                page_html = await httpx_get_content(
                    config["base_url"],
                    params=params,
                    attempts=2,
                    request_timeout=timeout,
                    jitter_min=jitter_min,
                    jitter_max=jitter_max,
                    use_proxy=not disable_proxy,
                )
                if page_html:
                    break
            except Exception as e:
                scraper_logger.warning(
                    f"⚠️ Error scraping page {page} attempt {attempt}: {e}"
                )
                if attempt < page_retries:
                    await asyncio.sleep(random.uniform(2.0, 5.0))
        if page_html:
            html_pages.append(page_html)

    scraper_logger.info(f"📄 Successfully scraped {len(html_pages)} HTML pages")
    return html_pages


# ===============================================================
# Extract Ebay HTML content to data
# ===============================================================
async def extract_ebay_post_data(
    html_pages: list[str], region: str, category_id: int | None = None
) -> list[dict[str, any]]:
    all_posts: list[dict[str, any]] = []
    for html in html_pages:
        soup = BeautifulSoup(html, "html.parser")
        items = soup.select("li[data-listingid]")
        for item in items:
            try:
                listing_id = item.get("data-listingid")
                if not listing_id:
                    continue

                # title
                title = None
                title_el = item.select_one(".s-card__title .su-styled-text")
                if title_el and title_el.get_text().strip():
                    title = title_el.get_text(strip=True)
                if not title or title.lower() == "shop on ebay":
                    continue

                # price / currency
                price = None
                currency = None
                price_el = item.select_one(".su-styled-text.s-card__price")
                if price_el:
                    price_text = price_el.get_text(strip=True)
                    if "£" in price_text:
                        currency = "GBP"
                        price_match = re.search(
                            r"£([\d,]+\.?\d*)", price_text.replace(",", "")
                        )
                    elif "$" in price_text:
                        currency = "USD"
                        price_match = re.search(
                            r"\$([\d,]+\.?\d*)", price_text.replace(",", "")
                        )
                    elif "€" in price_text:
                        currency = "EUR"
                        price_match = re.search(
                            r"€([\d,]+\.?\d*)", price_text.replace(",", "")
                        )
                    else:
                        price_match = re.search(
                            r"([\d,]+\.?\d*)", price_text.replace(",", "")
                        )
                        currency = "GBP" if region == "uk" else "USD"
                    if price_match:
                        price = price_match.group(1)
                if not price:
                    continue
                if not currency:
                    currency = "GBP" if region == "uk" else "USD"

                # url
                url = ""
                link_el = item.select_one("a[href*='/itm/']")
                if link_el:
                    url = link_el.get("href")
                if not url:
                    continue

                # images
                image_url = ""
                image_hd_url = ""
                img_el = item.select_one(".s-card__image") or item.select_one(
                    "img[src*='ebayimg.com']"
                )
                if img_el:
                    src = (
                        img_el.get("src")
                        or img_el.get("data-src")
                        or img_el.get("data-defer-load")
                    )
                    if src and "ebayimg.com" in src:
                        if "s-l140" in src:
                            image_url = src
                            image_hd_url = src.replace("s-l140", "s-l1600")
                        elif "s-l500" in src:
                            image_url = src.replace("s-l500", "s-l140")
                            image_hd_url = src.replace("s-l500", "s-l1600")
                        else:
                            image_url = src
                            if src and "i.ebayimg.com" in src:
                                image_hd_url = re.sub(
                                    r"/s-l[0-9]{2,4}", "/s-l1600", src
                                )
                            else:
                                image_hd_url = src

                # sold date
                sold_date = None
                sold_el = item.select_one(".s-card__caption .su-styled-text")
                if sold_el:
                    sold_text = sold_el.get_text(strip=True)
                    if "sold" in sold_text.lower():
                        try:
                            dt = date_parser.parse(sold_text, fuzzy=True, dayfirst=True)
                            tz = pytz.timezone(
                                "Europe/London"
                                if region == "uk"
                                else "America/New_York"
                            )
                            dt = dt if dt.tzinfo else tz.localize(dt)
                            sold_date = dt.isoformat()
                        except Exception:
                            sold_date = None
                if not sold_date:
                    sold_date = None

                post_data: dict[str, any] = {
                    "id": listing_id,
                    "title": title,
                    "image_url": image_url,
                    "image_hd_url": image_hd_url,
                    "normalized_title": title.lower(),
                    "sold_date": sold_date,
                    "sold_price": price,
                    "sold_currency": currency,
                    "sold_post_url": url,
                    "embedding": None,  # Will be filled in next step
                }

                all_posts.append(post_data)

            except Exception as e:
                scraper_logger.warning(f"⚠️ Failed to extract item data: {e}")
                continue

    # dedupe by url
    seen_urls = set()
    unique_posts: list[dict[str, any]] = []
    for post in all_posts:
        url = post.get("sold_post_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_posts.append(post)

    scraper_logger.info(
        f"📦 Extracted {len(unique_posts)} unique eBay post items (after normalization)"
    )
    return unique_posts


# ===============================================================
# Update user card with selected post
# ===============================================================
async def update_card(
    selected_post: dict[str, any], gpt_reason: str, card_id: str
) -> dict[str, any]:
    results = {"card_updated": False, "errors": []}
    try:
        update_data = {
            "last_sold_post_url": selected_post.get("sold_post_url"),
            "last_sold_price": selected_post.get("sold_price"),
            "last_sold_currency": selected_post.get("sold_currency"),
            "last_sold_at": selected_post.get("sold_date"),
            "last_sold_debug": gpt_reason,
        }
        supabase.table("cards").update(update_data).eq("id", card_id).execute()
        results["card_updated"] = True
        scraper_logger.info(
            f"✅ Updated card {card_id}: price={selected_post.get('sold_price')} {selected_post.get('sold_currency')}"
        )
    except Exception as e:
        results["errors"].append(str(e))
        scraper_logger.error(f"❌ Failed update cards: {e}")
    return results


# ===============================================================
# upsert ebay_posts (supabase)
# ===============================================================
async def upsert_ebay_listings(
    posts: list[dict[str, any]], card_id: str | None = None
) -> dict[str, any]:
    """
    Upsert posts into supabase table `ebay_posts`. If embedding present, store it in embedding column.
    """
    results = {"upserted_count": 0, "errors": []}
    if not posts:
        return results
    try:
        db_records = []
        for post in posts:
            # generate deterministic-ish id if not present
            db_id = post.get("title", "")[:50] + "_" + str(random.randint(1000, 9999))
            db_record = {
                "id": db_id,
                "title": post.get("title", ""),
                "image_url": post.get("image_url", ""),
                "image_hd_url": post.get("image_hd_url", ""),
                "region": post.get("metadata", {}).get("region", ""),
                "sold_at": post.get("sold_date", ""),
                "currency": post.get("sold_currency", "USD"),
                "price": post.get("sold_price", ""),
                "condition": post.get("condition", "raw"),
                "grading_company": post.get("grading_company"),
                "grade": post.get("grade"),
                "post_url": post.get("sold_post_url", ""),
                "blocked": False,
                "user_card_ids": [],
                "normalised_name": post.get("normalized_title", ""),
                "category_id": post.get("metadata", {}).get("category_id"),
            }
            db_records.append(db_record)
        # upsert in batches with robust fallback for missing columns
        BATCH = 500
        for i in range(0, len(db_records), BATCH):
            batch = db_records[i : i + BATCH]

            def _upsert_safe(recs: list[dict[str, any]]):
                while True:
                    try:
                        supabase.table("ebay_posts").upsert(
                            recs, on_conflict="id", returning="minimal"
                        ).execute()
                        return True
                    except Exception as e:
                        msg = str(e)
                        # Example: Could not find the 'normalized_attributes' column of 'ebay_posts'
                        m = re.search(r"Could not find the '([^']+)' column", msg)
                        if m:
                            missing_col = m.group(1)
                            for r in recs:
                                r.pop(missing_col, None)
                            scraper_logger.warning(
                                f"⚠️ Upsert retry: removed missing column '{missing_col}' from batch"
                            )
                            continue
                        # Some Supabase clients return JSON dict-like errors; attempt to parse column name
                        if "normalized_attributes" in msg:
                            for r in recs:
                                r.pop("normalized_attributes", None)
                            scraper_logger.warning(
                                "⚠️ Upsert retry: removed 'normalized_attributes' from batch"
                            )
                            continue
                        if "embedding" in msg:
                            for r in recs:
                                r.pop("embedding", None)
                            scraper_logger.warning(
                                "⚠️ Upsert retry: removed 'embedding' from batch"
                            )
                            continue
                        # If we cannot recover, re-raise
                        raise

            _upsert_safe(batch)
            results["upserted_count"] += len(batch)
        scraper_logger.info(
            f"✅ Upserted {results['upserted_count']} listings to ebay_posts"
        )
    except Exception as e:
        results["errors"].append(str(e))
        scraper_logger.error(f"❌ Failed to upsert listings: {e}")
    return results


# ===============================================================
# get card record by card_id (supabase)
# ===============================================================
async def get_card(card_id: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("cards")
            .select(
                "canonical_title, years, card_set, name, variation, serial_number, number, grading_company, grade_number"
            )
            .eq("id", card_id)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]
    except Exception as e:
        scraper_logger.info(f"⚠️ Failed to resolve card: {e}")
    return None


# ===============================================================
# Normalize text for comparison
# ===============================================================
def normalized_text(val: Any) -> Optional[str]:
    if not val or val == "":
        return None
    return str(val).strip().lower()


# ===============================================================
# word_match
# ===============================================================
def word_match(
    title: Optional[str],
    filter: Optional[str],
    word_threshold: Optional[int] = None,
) -> bool:
    """Return True if title matches filter using token and fuzzy checks with count-based threshold.

    - title: text to search within (can be None; treated as empty string)
    - filter: space-delimited string of tokens to check (None/empty => False)
    - word_threshold: minimum number of tokens/phrases that must match (>= 1). If None or > token count, require all tokens.

    Special rules:
    - Tokens are split on whitespace and evaluated independently (even single-character tokens like "1").
    - Use RapidFuzz per token as a flexible fallback (no static alias expansions).
    """
    if not filter:
        return True
    # Normalize title
    title_text = (title or "").lower().strip()
    filter_text = str(filter).lower().strip()
    # Build collapsed variants to catch cases like 'logo fractor' vs 'logofractor' or 'logo-fractor'
    title_collapsed = title_text.replace(" ", "")
    filter_collapsed = filter_text.replace(" ", "")
    # Alphanumeric-only squeeze (remove all non a-z0-9)
    title_squeezed = re.sub(r"[^a-z0-9]+", "", title_text)
    filter_squeezed = re.sub(r"[^a-z0-9]+", "", filter_text)
    # Tokenize the filter on whitespace; do not merge short tokens
    tokens = [t for t in filter_text.split() if t]
    if not tokens:
        return False

    token_count = len(tokens)
    # Compute required matches based on word_threshold (default: require all)
    required = (
        token_count
        if not word_threshold or word_threshold <= 0
        else min(word_threshold, token_count)
    )

    # 1) If requiring all tokens, allow an early perfect whole-string equality
    if required == token_count:
        if fuzz.token_set_ratio(title_text, filter_text) >= 100:
            return True
        if fuzz.token_set_ratio(title_collapsed, filter_collapsed) >= 100:
            return True
        if fuzz.token_set_ratio(title_squeezed, filter_squeezed) >= 100:
            return True

    # 2) Count token coverage (exact or fuzzy) for tokens
    matched = 0
    fuzzy_token_min_ratio = 80  # fixed per-token fuzzy similarity threshold
    for t in tokens:
        tc = t.replace(" ", "")
        ts = re.sub(r"[^a-z0-9]+", "", t)
        if t in title_text or tc in title_collapsed or ts in title_squeezed:
            matched += 1
            continue
        # last resort fuzzy per-token when available
        if (
            fuzz.partial_ratio(t, title_text) >= fuzzy_token_min_ratio
            or fuzz.partial_ratio(tc, title_collapsed) >= fuzzy_token_min_ratio
            or fuzz.partial_ratio(ts, title_squeezed) >= fuzzy_token_min_ratio
        ):
            matched += 1

    return matched >= required


# ===============================================================
# Helper functions to check if all required attributes are matched
# ===============================================================
def all_present_matched(
    attribute_filters: List[str], req: List[str], matched: Dict[str, bool]
) -> bool:
    filtered_req = [k for k in req if k in attribute_filters]
    return all(matched.get(k, False) for k in filtered_req)


# ===============================================================
# Build strict promisable candidates
# ===============================================================
def build_strict_promisable_candidates(
    attribute_filters: Dict[str, Any],
    posts: List[Dict],
) -> Dict[str, List[Dict]]:
    low_req = ["years", "name"]
    med_req = ["years", "card_set", "name"]
    high_req = ["years", "card_set", "name", "variation"]

    # -----------------------------------------------------------
    # Normalize card-side values once
    # -----------------------------------------------------------
    unmatch_posts: List[Dict] = []
    match_posts: List[Dict] = []
    # Debug counters
    low_cnt = med_cnt = high_cnt = unmatch_cnt = fallback_cnt = 0

    # Log incoming filters once for debugging
    try:
        dbg_filters = {
            k: v
            for k, v in attribute_filters.items()
            if k
            in ["years", "card_set", "name", "variation", "serial_number", "number"]
        }
        scraper_logger.info(
            f"[build_candidates] filters: {json.dumps(dbg_filters, ensure_ascii=False)}"
        )
    except Exception:
        pass

    card_years = normalized_text(attribute_filters.get("years", ""))
    card_set = normalized_text(attribute_filters.get("card_set", ""))
    card_name = normalized_text(attribute_filters.get("name", ""))
    card_variation = normalized_text(attribute_filters.get("variation", ""))
    card_serial = normalized_text(attribute_filters.get("serial_number", ""))
    card_number = normalized_text(attribute_filters.get("number", ""))
    # card_grading = normalized_text(attribute_filters.get("grading_company", ""))
    # card_grade = normalized_text(attribute_filters.get("grade_number", ""))

    # print(f"card_years: {card_years}")
    # print(f"card_set: {card_set}")
    # print(f"card_name: {card_name}")
    # print(f"card_variation: {card_variation}")
    # print(f"card_serial: {card_serial}")
    # print(f"card_number: {card_number}")
    # print(f"card_grading: {card_grading}")
    # print(f"card_grade: {card_grade}")

    for ebay_post in posts:
        # -----------------------------------------------------------
        # Post normalization helpers
        # -----------------------------------------------------------
        title = ebay_post.get("normalized_title")
        matched = {
            # Years: match any of the provided years (threshold=1)
            "years": word_match(title, card_years, 1),
            "card_set": word_match(title, card_set, 2),
            "name": word_match(title, card_name),
            "variation": word_match(title, card_variation, 0),
            "serial_number": word_match(title, card_serial),
            "number": word_match(title, card_number),
            # "grading_company": word_match(title, card_grading),
            # "grade_number": word_match(title, card_grade),
        }

        if all_present_matched(attribute_filters, low_req, matched):
            matched["candidate_score"] = "low"
            low_cnt += 1
        elif all_present_matched(attribute_filters, med_req, matched):
            matched["candidate_score"] = "medium"
            med_cnt += 1
        elif all_present_matched(attribute_filters, high_req, matched):
            matched["candidate_score"] = "high"
            high_cnt += 1
        else:
            matched["candidate_score"] = "unmatch"
            unmatch_cnt += 1

        ebay_post["matched"] = matched

        if matched["candidate_score"] == "unmatch":
            # Fallback: allow GPT to refine when set+name look good, even if year is missing
            if matched.get("card_set") and matched.get("name"):
                matched["candidate_score"] = "fallback"
                match_posts.append(ebay_post)
                fallback_cnt += 1
            else:
                unmatch_posts.append(ebay_post)
        else:
            match_posts.append(ebay_post)

    # Log summary counts and up to 5 sample unmatches for diagnosis
    try:
        scraper_logger.info(
            f"[build_candidates] counts -> low={low_cnt}, med={med_cnt}, high={high_cnt}, fallback={fallback_cnt}, unmatch={unmatch_cnt}"
        )
        if unmatch_cnt and len(unmatch_posts) > 0:
            sample = []
            for p in unmatch_posts[:5]:
                sample.append(
                    {
                        "title": p.get("normalized_title") or p.get("title"),
                        "matched": p.get("matched"),
                    }
                )
            # scraper_logger.info(
            #     f"[build_candidates] unmatch_sample: {json.dumps(sample, ensure_ascii=False)}"
            # )
    except Exception:
        pass

    return {"unmatch_posts": unmatch_posts, "match_posts": match_posts}


# ===============================================================
# GPT selection using model gpt-4o-mini
# ===============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com")
GPT_MODEL = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")


async def select_best_candidate_with_gpt(
    canonical_title: Optional[str],
    match_posts: List[Dict],
) -> Optional[Dict]:
    """Return the single best candidate (full post) chosen by GPT and finalized by most recent sold_date.

    Flow:
    - Build a compact list from ALL `match_posts` for GPT (index + id + titles).
    - GPT returns indices of likely matches.
    - Map indices back to `match_posts` to obtain the candidate list.
    - Print the compact entries for the selected indices for debugging.
    - Choose the most recent by sold_date and return that single post.
    """
    if len(match_posts) == 0:
        scraper_logger.info("❌ No match posts found; selection returns None")
        return None

    compact: List[Dict] = []
    for idx, match_post in enumerate(match_posts):
        compact.append(
            {
                "i": idx,  # index used for selection
                "id": match_post.get("id"),
                "raw": str(match_post.get("title")),
                "normalized": str(match_post.get("normalized_title")),
            }
        )

    scraper_logger.info(f"[compact count before send] --- {len(compact)} posts")
    scraper_logger.info("--------------------------------------")

    system_prompt = (
        "You are an EXPERT trading-card matcher. Decide whether each candidate refers to the SAME underlying card as the canonical title. Apply these principles: "
        "1) Core identity is defined by: product line/set/series + subject (player/character/team/item/trainer/stadium) + card number when provided. "
        "2) Normalize case, punctuation, hyphens, common abbreviations, and spacing; treat equivalent naming and order-insensitive wording as the same. "
        "3) Set/product line must be the same family (brand + series). Different product lines are NOT matches. "
        "4) If the canonical includes a card number, prefer candidates with the same number; ignore '#' and leading zeros and tolerate separators. If a candidate shows a different number, EXCLUDE it. "
        "5) Year is informative but NOT required; do not exclude solely for missing/different year when the core identity matches. "
        "6) Variations/parallels: When the canonical specifies a specific variant (color/finish/parallel/print-run, autograph, memorabilia, promo/pre-release), EXCLUDE candidates that explicitly state a different variant. When the canonical is silent, ACCEPT base and minor surface-finish variants typical to the product line; treat grading as non-identity. "
        "7) Quantity/lot terms (e.g., playset, x2/x4, set of N) do not alter identity—do not exclude for quantity alone. "
        "8) Ignore boilerplate (game/sport name, generic words like 'card', condition terms) unless they convey identity (e.g., team vs player). "
        "Return a JSON object: {matches: [indices], reason: string}. Include indices only when the core identity rules are satisfied. If none qualify, return an empty list and briefly explain why."
    )

    user_prompt = {
        "canonical_title": canonical_title or "",
        "candidates": compact,
        "instructions": (
            "Compare 'canonical_title' with each candidate using both 'raw' and 'normalized' titles. "
            "Include an index ONLY IF the core identity matches: same product line/set/series (semantic family), same subject (player/character/team/item/trainer/stadium), and — when the canonical provides it — the same card number (ignoring '#', leading zeros, and separators). "
            "Year is optional. If the canonical specifies a particular variant, exclude candidates explicitly stating a different variant; otherwise accept typical minor finish variants for the product line. Quantity/lot terms do not affect identity. "
            "Treat serial/print-run and grading as informative signals but not strictly required unless mismatched values change identity. "
            'Return JSON: {"matches": [i1, i2, ...], "reason": <string>}. If none qualify, leave matches empty and provide a concise reason (e.g., set mismatch, different variant, number mismatch, uncertain identity).'
        ),
    }

    try:
        url = OPENAI_BASE.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GPT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        scraper_logger.info(f"[GPT response] --- {content}")
        idxs = parsed.get("matches", [])

        selected_indices: List[int] = [
            i for i in idxs if isinstance(i, int) and 0 <= i < len(match_posts)
        ]

        selected_posts: List[Dict] = [match_posts[i] for i in selected_indices]

        def sold_ts(p: Dict) -> float:
            d = p.get("sold_date") or p.get("sold_at") or ""
            try:
                return date_parser.parse(d, fuzzy=True).timestamp()
            except Exception:
                return 0.0

        scraper_logger.info("--------------------------------------")
        scraper_logger.info(
            f"[candidates] --- {json.dumps(selected_posts, indent=4, ensure_ascii=False)}"
        )

        last_sold_post = max(selected_posts, key=sold_ts)
        return {
            "last_sold_post": last_sold_post,
            "reason": parsed.get("reason"),
        }
    except Exception as e:
        scraper_logger.info(f"⚠️ GPT selection failed: {e}")
        try:
            scraper_logger.info(f"gpt_error: {str(e)}")
        except Exception:
            pass
        return None

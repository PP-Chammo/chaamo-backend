"""Clean, refactored eBay scraping system with 4 main functions."""

import asyncio
import math
import random
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from dataclasses import dataclass
from urllib.parse import urlencode

import dateutil.parser as date_parser
import pytz
from bs4 import BeautifulSoup

from src.utils.logger import get_logger
from src.models.category import CategoryId
from src.models.ebay import Region, base_target_url
from src.utils.httpx import httpx_get_content
from src.utils.supabase import supabase
from src.utils.matching_new import get_last_sold_item

scraper_logger = get_logger("scraper")


# ==============================================================================
# FUNCTION 1: DECIDE MODE AND SCRAPE EBAY -> RETURN HTML CONTENT
# ==============================================================================


async def scrape_ebay_html(
    region: Region,
    query: Optional[str] = None,
    category_id: Optional[CategoryId] = None,
    user_card_id: Optional[str] = None,
    max_pages: int = 50,
    page_retries: int = 3,
    disable_proxy: bool = False,
) -> List[str]:
    """
    Function 1: Decide eBay scrape mode (query+category_id or user_card_id) and scrape through httpx + zyte proxy.

    Args:
        region: eBay region (us/uk)
        query: Search query (for mode 1)
        category_id: Category filter (for mode 1)
        user_card_id: User card ID (for mode 2)
        max_pages: Maximum pages to scrape
        page_retries: Number of retry attempts per page (default: 3)
        disable_proxy: Whether to disable proxy usage

    Returns:
        List[str]: List of HTML content from scraped pages
    """
    # RESOLVE SCRAPING MODE AND PARAMETERS
    final_query = query or "card"
    final_category_id = category_id

    # Mode 2: user_card_id provided - resolve from database
    if user_card_id:
        try:
            response = (
                supabase.table("user_cards")
                .select("custom_name, category_id")
                .eq("id", user_card_id)
                .limit(1)
                .execute()
            )

            if response.data:
                card_data = response.data[0]
                final_query = card_data.get("custom_name") or "card"
                final_category_id = (
                    CategoryId(card_data.get("category_id"))
                    if card_data.get("category_id")
                    else None
                )
                scraper_logger.info(
                    f"✅ Mode 2: Using user_card_id - query: '{final_query}'"
                )
        except Exception as e:
            scraper_logger.warning(
                f"⚠️ Error resolving user_card_id {user_card_id}: {e}"
            )

    # Mode 1: Direct query + category_id
    if query and category_id:
        scraper_logger.info(f"✅ Mode 1: Using direct query: '{query}'")

    # Build base configuration
    config = {
        "query": final_query,
        "category_id": final_category_id.value if final_category_id else None,
        "region_str": region.value,
        "base_url": f"{base_target_url[region.value]}/sch/i.html",
    }

    # GET TOTAL AVAILABLE PAGES
    total_pages = 1
    try:
        # Build search params for page 1
        params = {
            "_nkw": config["query"],
            "_sacat": "0",
            "_from": "R40",
            "rt": "nc",
            "LH_Sold": "1",
            "LH_Complete": "1",
            "Country/Region of Manufacture": (
                "United Kingdom" if config["region_str"] == "uk" else "United States"
            ),
            "_ipg": "240",
            "_pgn": "1",
            "_sop": "13",
        }

        scraper_logger.info(
            f"📡 Scraping eBay URL: {config['base_url']}?{urlencode(params)}"
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
    scraper_logger.info(f"🚀 Scraping {total_pages} pages for query: '{final_query}'")

    # SCRAPE ALL PAGES AND RETURN HTML CONTENT
    html_pages = []
    for page in range(1, total_pages + 1):
        if page > 1:
            # Human-like delay between pages
            delay = random.uniform(1.0, 3.5) + (page - 1) * random.uniform(0.2, 0.8)
            await asyncio.sleep(delay)

        # Scrape single page with retry logic
        page_html = None

        for attempt in range(1, page_retries + 1):
            try:
                # Build search parameters for this page
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
                    "_sop": "13",
                }

                # Anti-bot randomization
                timeout = random.uniform(12.0, 18.0)
                jitter_min = random.uniform(0.5, 1.2)
                jitter_max = random.uniform(1.5, 3.0)

                page_html = await httpx_get_content(
                    config["base_url"],
                    params=params,
                    attempts=5,
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


# ==============================================================================
# FUNCTION 2: EXTRACT HTML AND ITERATE EBAY POST ITEMS -> RETURN ARRAY OF DATA
# ==============================================================================


async def extract_ebay_post_data(
    html_pages: List[str], region: str, category_id: Optional[int] = None
) -> List[Dict]:
    """
    Function 2: Extract HTML and iterate each eBay post item to return array of extracted data.

    Args:
        html_pages: List of HTML content from scraped pages
        region: Region string (us/uk) for currency/timezone handling
        category_id: Category ID for the posts

    Returns:
        List[Dict]: Array of extracted eBay post item data with structure:
        - title: eBay post title
        - image_url: eBay post image URL
        - image_hd_url: eBay post image URL in HD version
        - metadata: Everything related to eBay post (grade, company, set, etc) except sold data
        - sold_date: When item was sold
        - sold_price: Sale price
        - sold_currency: Currency of sale
        - sold_post_url: URL to the sold listing
        - embedding: Title embedding (placeholder for now)
    """
    all_posts = []

    for html in html_pages:
        soup = BeautifulSoup(html, "html.parser")
        # Focus on actual product listings with data-listingid
        items = soup.select("li[data-listingid]")

        for item in items:
            try:
                # EXTRACT DATA FROM SINGLE EBAY ITEM ELEMENT
                # Extract listing ID
                listing_id = item.get("data-listingid")
                if not listing_id:
                    continue

                # Extract title
                title = None
                title_el = item.select_one(".s-card__title .su-styled-text")
                if title_el and title_el.get_text().strip():
                    title = title_el.get_text(strip=True)

                if not title or title.lower() == "shop on ebay":
                    continue

                # Extract price and currency
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

                # Ensure currency is set (fallback if not detected)
                if not currency:
                    currency = "GBP" if region == "uk" else "USD"

                # Extract URL
                url = ""
                link_el = item.select_one("a[href*='/itm/']")
                if link_el:
                    url = link_el.get("href")

                if not url:
                    continue

                # Extract image URLs
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
                            # Upgrade eBay image URL to HD
                            if src and "i.ebayimg.com" in src:
                                image_hd_url = re.sub(
                                    r"/s-l[0-9]{2,4}", "/s-l1600", src
                                )
                            else:
                                image_hd_url = src

                # Extract sold date
                sold_date = None
                sold_el = item.select_one(".s-card__caption .su-styled-text")
                if sold_el:
                    sold_text = sold_el.get_text(strip=True)
                    if "sold" in sold_text.lower():
                        # Parse sold date with proper timezone
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
                            sold_date = datetime.utcnow().isoformat()

                if not sold_date:
                    sold_date = datetime.utcnow().isoformat()

                # Extract condition, grading company, and grade from title
                grading_companies = [
                    "PSA",
                    "BGS",
                    "SGC",
                    "CGC",
                    "CSG",
                    "HGA",
                    "GMA",
                    "Beckett",
                ]
                condition = "raw"
                grading_company = None
                grade = None

                for company in grading_companies:
                    if company in (title or "").upper():
                        condition = "graded"
                        grading_company = company
                        m = re.search(
                            rf"{company} ?([0-9]{{1,2}}(?:\.[0-9])?)",
                            (title or "").upper(),
                        )
                        if m:
                            grade = float(m.group(1))
                        break

                # Normalize card name for search/matching
                normalized_name = ""
                if title:
                    # Convert to lowercase and remove extra spaces
                    normalized = re.sub(r"\s+", " ", title.lower().strip())
                    # Remove common symbols and punctuation
                    normalized = re.sub(r"[^\w\s]", " ", normalized)
                    # Remove extra spaces again after symbol removal
                    normalized_name = re.sub(r"\s+", " ", normalized).strip()

                # Extract meaningful words directly from title as tags
                tags = []
                if title:
                    # Important categories that should always be included
                    important_categories = {
                        "topps",
                        "panini",
                        "futera",
                        "pokemon",
                        "pokémon",
                        "dc",
                        "fortnite",
                        "marvel",
                        "digimon",
                        "wrestling",
                        "yugioh",
                        "yu-gi-oh",
                        "lorcana",
                        "garbage",
                        "pail",
                        "kids",
                    }

                    # Generic stop words to exclude
                    stop_words = {
                        "the",
                        "a",
                        "an",
                        "and",
                        "or",
                        "but",
                        "in",
                        "on",
                        "at",
                        "to",
                        "for",
                        "of",
                        "with",
                        "by",
                        "from",
                        "up",
                        "about",
                        "into",
                        "through",
                        "during",
                        "before",
                        "after",
                        "card",
                        "cards",
                        "trading",
                        "sports",
                        "collectible",
                        "collectibles",
                        "mint",
                        "condition",
                        "nm",
                        "ex",
                        "vg",
                        "poor",
                        "lot",
                        "set",
                        "pack",
                        "box",
                        "case",
                        "sealed",
                        "new",
                        "used",
                        "vintage",
                        "tcg",
                        "ccg",
                        "game",
                        "games",
                        "trading_cards",
                    }

                    # Clean the title and split into words
                    cleaned_title = re.sub(r"[^\w\s\-]", " ", title)
                    words = cleaned_title.lower().split()

                    meaningful_tags = []

                    for word in words:
                        word = word.strip("-").strip()

                        # Skip if empty or too short
                        if not word or len(word) < 2:
                            continue

                        # Always include important category names
                        if word in important_categories:
                            meaningful_tags.append(word)
                            continue

                        # Skip generic stop words
                        if word in stop_words:
                            continue

                        # Include years (4 digits starting with 19 or 20)
                        if re.match(r"^(19|20)\d{2}$", word):
                            meaningful_tags.append(word)
                            continue

                        # Include meaningful words (3+ chars)
                        if len(word) >= 3:
                            if word.isdigit() and not re.match(r"^(19|20)\d{2}$", word):
                                continue
                            meaningful_tags.append(word)
                        elif len(word) == 2 and not word.isdigit():
                            meaningful_tags.append(word)

                    # Remove duplicates and limit to top 8 tags
                    tags = list(dict.fromkeys(meaningful_tags))[:8]

                post_data = {
                    "id": listing_id,
                    "title": title,
                    "image_url": image_url,
                    "image_hd_url": image_hd_url,
                    "metadata": {
                        "condition": condition,
                        "grading_company": grading_company,
                        "grade": grade,
                        "region": region,
                        "category_id": category_id,
                        "normalized_name": normalized_name,
                        "tags": tags,
                    },
                    "sold_date": sold_date,
                    "sold_price": price,
                    "sold_currency": currency,
                    "sold_post_url": url,
                    "embedding": None,  # Placeholder for future embedding functionality
                }

                if post_data:
                    all_posts.append(post_data)

            except Exception as e:
                scraper_logger.warning(f"⚠️ Failed to extract item data: {e}")
                continue

    # REMOVE DUPLICATES BY URL
    seen_urls = set()
    unique_posts = []

    for post in all_posts:
        url = post.get("sold_post_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_posts.append(post)

    scraper_logger.info(f"📦 Extracted {len(unique_posts)} unique eBay post items")

    return unique_posts


# ==============================================================================
# FUNCTION 3: FILTER AND DECIDE BEST EBAY POST -> RETURN SELECTED OBJECT
# ==============================================================================


async def select_best_ebay_post(posts: List[Dict], user_query: str) -> Optional[Dict]:
    """
    Function 3: Filter and decide which eBay post data should be used as last_sold price.
    Uses robust matching system with fuzzy matching, card features, and edge case handling.

    Args:
        posts: List of eBay post data from function 2
        user_query: Original user query for matching

    Returns:
        Optional[Dict]: Single selected eBay post data object, or None if no suitable match
    """
    if not posts:
        return None

    # Use robust card matcher to find best match
    result = await get_last_sold_item(user_query, posts)

    print("--------------------------")
    print(f'Found: {result["found"]}')
    print(f'Match: {result["match"]}')
    print(f'score: {result["score"]}')
    print(f'reason: {result["reason"]}')
    print(f'candidates: {result["candidates"]}')
    print("--------------------------")

    return result


# ==============================================================================
# FUNCTION 4: UPSERT DATABASE AND UPDATE USER_CARDS
# ==============================================================================


async def upsert_ebay_listings(
    posts: List[Dict], user_card_id: Optional[str] = None
) -> Dict:
    """
    Function 4a: Upsert eBay listings to ebay_listings table only.
    Used for both query+category_id and user_card_id modes.

    Args:
        posts: All eBay post data to upsert to ebay_listings table
        user_card_id: User card ID for tracking (optional)

    Returns:
        Dict: Summary of upsert results
    """
    results = {"upserted_count": 0, "errors": []}

    if not posts:
        return results

    try:
        db_records = []
        current_time = datetime.utcnow().isoformat()

        for post in posts:
            # Transform to database schema (basic fields only)
            db_record = {
                "id": post.get("title", "")[:50]
                + "_"
                + str(random.randint(1000, 9999)),
                "name": post.get("title", ""),
                "image_url": post.get("image_url", ""),
                "image_hd_url": post.get("image_hd_url", ""),
                "region": post.get("metadata", {}).get("region", ""),
                "sold_at": post.get("sold_date", ""),
                "currency": post.get("sold_currency", "USD"),  # Required field
                "price": post.get("sold_price", ""),
                "condition": post.get("condition", "raw"),  # Required field
                "grading_company": post.get("grading_company"),
                "grade": post.get("grade"),
                "post_url": post.get("sold_post_url", ""),
                "blocked": False,
                "user_card_ids": [],
                "normalised_name": post.get("normalised_name", ""),
                "category_id": post.get("metadata", {}).get("category_id"),
            }
            db_records.append(db_record)

        # Upsert in batches
        batch_size = 1000
        for i in range(0, len(db_records), batch_size):
            batch = db_records[i : i + batch_size]
            supabase.table("ebay_posts").upsert(
                batch, on_conflict="id", returning="minimal"
            ).execute()
            results["upserted_count"] += len(batch)

        scraper_logger.info(
            f"✅ Upserted {results['upserted_count']} listings to ebay_posts table"
        )

    except Exception as e:
        error_msg = f"Failed to upsert listings: {str(e)}"
        results["errors"].append(error_msg)
        scraper_logger.error(f"❌ {error_msg}")

    return results


async def update_user_card(selected_post: Dict, user_card_id: str) -> Dict:
    """
    Function 4b: Update specific user_cards with selected post data.
    Only used for user_card_id mode.

    Args:
        selected_post: Selected best post for updating user_cards
        user_card_id: User card ID to update

    Returns:
        Dict: Summary of user card update
    """
    results = {"user_card_updated": False, "errors": []}

    try:
        update_data = {
            "last_sold_price": selected_post["sold_price"],
            "last_sold_currency": selected_post["sold_currency"],
            "last_sold_at": selected_post["sold_date"],
            "last_sold_post_url": selected_post["sold_post_url"],
        }
        
        print(user_card_id, update_data)

        supabase.table("user_cards").update(update_data).eq(
            "id", user_card_id
        ).execute()
        results["user_card_updated"] = True

        scraper_logger.info(
            f"✅ Updated user_card {user_card_id}: "
            f"price={selected_post.get('sold_price')} {selected_post.get('sold_currency')}"
        )

    except Exception as e:
        error_msg = f"Failed to update user_cards: {str(e)}"
        results["errors"].append(error_msg)
        scraper_logger.error(f"❌ {error_msg}")

    return results


# ==============================================================================
# MAIN ORCHESTRATOR CLASS AND WORKER SYSTEM
# ==============================================================================


class EbayScraper:
    """Main eBay scraper class orchestrating the 4 core functions."""

    def __init__(self):
        self.region_codes = {"us": "us", "uk": "uk"}

    async def scrape(
        self,
        region: Region,
        query: Optional[str] = None,
        category_id: Optional[CategoryId] = None,
        user_card_id: Optional[str] = None,
        max_pages: int = 50,
        page_retries: int = 3,
        disable_proxy: bool = False,
    ) -> Dict:
        """Main scraping orchestrator using the 4 core functions."""

        try:
            # Resolve query for user_card_id mode early
            resolved_query = query or "card"
            if user_card_id:
                try:
                    response = (
                        supabase.table("user_cards")
                        .select("custom_name, category_id")
                        .eq("id", user_card_id)
                        .limit(1)
                        .execute()
                    )

                    if response.data:
                        card_data = response.data[0]
                        resolved_query = card_data.get("custom_name") or "card"
                        scraper_logger.info(
                            f"🔍 Resolved user_card query: '{resolved_query}'"
                        )
                    else:
                        scraper_logger.warning(
                            f"⚠️ user_card_id {user_card_id} not found in database"
                        )
                except Exception as e:
                    scraper_logger.error(
                        f"❌ Failed to resolve user_card query: {str(e)}"
                    )

            # Function 1: Scrape HTML content
            html_pages = await scrape_ebay_html(
                region=region,
                query=query,
                category_id=category_id,
                user_card_id=user_card_id,
                max_pages=max_pages,
                page_retries=page_retries,
                disable_proxy=disable_proxy,
            )

            if not html_pages:
                return {"total": 0, "result": []}

            # Function 2: Extract post data from HTML
            posts = await extract_ebay_post_data(
                html_pages=html_pages,
                region=region.value,
                category_id=category_id.value if category_id else None,
            )

            # Function 4a: Always upsert to ebay_listings table
            upsert_results = await upsert_ebay_listings(
                posts=posts, user_card_id=user_card_id
            )

            results = {
                "total": len(posts),
                "result": posts,
                "upsert_results": upsert_results,
            }

            # Function 3 & 4b: Only select best post and update user_cards for user_card_id mode
            if user_card_id:
                scraper_logger.info(
                    f"🔍 Using resolved query for matching: '{resolved_query}'"
                )

                matching_result = await select_best_ebay_post(
                    posts=posts, user_query=resolved_query
                )

                # Extract the actual matched post from the result structure
                if matching_result and matching_result.get("match"):
                    selected_post = matching_result["match"]
                    user_update_results = await update_user_card(
                        selected_post=selected_post, user_card_id=user_card_id
                    )
                    results["user_update_results"] = user_update_results
                    scraper_logger.info(
                        f"🎯 User card mode: updated {user_card_id} with best match (score: {matching_result.get('score', 0)}, found: {matching_result.get('found', False)})"
                    )
                else:
                    scraper_logger.info(
                        f"⚠️ No suitable match found for user_card {user_card_id}"
                    )
            else:
                scraper_logger.info(
                    f"📊 Query+Category mode: {len(posts)} listings upserted only"
                )

            scraper_logger.info(f"🎉 Scraping completed: {len(posts)} posts processed")
            return results

        except Exception as e:
            scraper_logger.error(f"❌ Scraping failed: {str(e)}")
            raise


# Worker system classes (preserved for API compatibility)
class WorkerStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkerTask:
    id: str
    query: str
    region: str
    category_id: int
    status: WorkerStatus
    created_at: datetime
    user_card_id: Optional[str] = None
    max_pages: int = 50
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_count: int = 0
    upserted_count: int = 0
    user_updated: bool = False
    results: Optional[dict] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "region": self.region,
            "category_id": self.category_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "user_card_id": self.user_card_id,
            "max_pages": self.max_pages,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "results_count": self.result_count,
            "upserted_count": self.upserted_count,
            "user_updated": self.user_updated,
            "error_message": self.error_message,
        }


class EbayWorkerManager:
    """Worker manager for background eBay scraping tasks."""

    def __init__(self):
        self.active_tasks: Dict[str, WorkerTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=3)

    def create_task(
        self,
        query: Optional[str] = None,
        region: Region = Region.us,
        category_id: Optional[CategoryId] = None,
        user_card_id: Optional[str] = None,
        max_pages: int = 50,
    ) -> WorkerTask:
        """Create new worker task."""
        task_id = str(uuid.uuid4())

        # Simple parameter resolution for task creation
        final_query = query or "card"
        final_category_id = category_id or CategoryId.PANINI

        task = WorkerTask(
            id=task_id,
            query=final_query,
            region=region.value,
            category_id=final_category_id.value,
            status=WorkerStatus.PENDING,
            created_at=datetime.utcnow(),
            user_card_id=user_card_id,
            max_pages=max_pages,
        )

        self.active_tasks[task_id] = task
        return task

    async def _run_scrape_worker(
        self,
        task_id: str,
        region: Region,
        category_id: CategoryId,
        query: str,
        user_card_id: Optional[str],
        max_pages: int,
        master_card_id: Optional[str] = None,
        disable_proxy: bool = False,
    ):
        """Execute scraping task in background."""
        task = self.active_tasks[task_id]

        try:
            task.status = WorkerStatus.RUNNING
            task.started_at = datetime.utcnow()

            # Use the main scraper
            result = await scraper.scrape(
                region=region,
                query=query,
                category_id=category_id,
                user_card_id=user_card_id,
                max_pages=max_pages,
                disable_proxy=disable_proxy,
            )

            task.result_count = result.get("total", 0)
            task.upserted_count = result.get("upsert_results", {}).get(
                "upserted_count", 0
            )
            task.user_updated = bool(
                result.get("user_update_results", {}).get("updated", False)
            )
            task.results = result  # Store full results for API access
            task.status = WorkerStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            scraper_logger.info(
                f"✅ Task {task_id} completed: {task.result_count} posts, {task.upserted_count} upserted, user_updated={task.user_updated}"
            )

        except Exception as e:
            task.status = WorkerStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            scraper_logger.error(f"❌ Task {task_id} failed: {str(e)}")

    def get_task_status(self, task_id: str) -> Optional[WorkerTask]:
        """Get task status by ID."""
        return self.active_tasks.get(task_id)

    def get_all_tasks(self) -> List[WorkerTask]:
        """Get all tasks sorted by created date."""
        return sorted(
            self.active_tasks.values(), key=lambda t: t.created_at, reverse=True
        )

    def cleanup_old_tasks(self, max_tasks: int = 100):
        """Remove old completed tasks to prevent memory buildup."""
        if len(self.active_tasks) <= max_tasks:
            return

        # Keep most recent tasks
        all_tasks = self.get_all_tasks()
        tasks_to_remove = all_tasks[max_tasks:]

        for task in tasks_to_remove:
            if task.status in [WorkerStatus.COMPLETED, WorkerStatus.FAILED]:
                self.active_tasks.pop(task.id, None)


# Initialize instances
scraper = EbayScraper()
worker_manager = EbayWorkerManager()

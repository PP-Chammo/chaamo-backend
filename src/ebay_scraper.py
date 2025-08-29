"""Clean, well-structured eBay scraping system with proper function ordering."""

import asyncio
import math
import random
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import dateutil.parser as date_parser
import pytz
from bs4 import BeautifulSoup

from src.utils.logger import get_logger

scraper_logger = get_logger("scraper")

from src.models.category import CategoryId
from src.models.ebay import Region, base_target_url
from src.utils.httpx import httpx_get_content
from src.utils.supabase import supabase


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def get_condition_and_grade(title: str):
    """Extract condition, grading company, and grade from card title."""
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
                grade = float(m.group(1))
            break
    
    return condition, grading_company, grade


def _normalize_name(name: str) -> str:
    """Normalize card name for search/matching."""
    if not name:
        return ""
    
    # Convert to lowercase and remove extra spaces
    normalized = re.sub(r'\s+', ' ', name.lower().strip())
    
    # Remove common symbols and punctuation
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    
    # Remove extra spaces again after symbol removal
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


# ==============================================================================
# UTILITY FUNCTIONS (No dependencies - called by other functions)
# ==============================================================================


def _safe_parse_date(date_str: str) -> datetime:
    """Safely parse date string."""
    try:
        return date_parser.parse(date_str)
    except Exception:
        return date_parser.parse("1970-01-01T00:00:00Z")


def _upgrade_image_url(url: str) -> str:
    """Upgrade eBay image URL to HD."""
    if url and "i.ebayimg.com" in url:
        return re.sub(r"/s-l[0-9]{2,4}", "/s-l1600", url)
    return url


def _parse_sold_date(sold_text: str, region: str) -> str:
    """Parse sold date with proper timezone."""
    try:
        dt = date_parser.parse(sold_text, fuzzy=True, dayfirst=True)
        tz = pytz.timezone("Europe/London" if region == "uk" else "America/New_York")
        dt = dt if dt.tzinfo else tz.localize(dt)
        return dt.isoformat()
    except Exception:
        return datetime.utcnow().isoformat()


def _build_search_params(config: Dict, page: int) -> Dict:
    """Build eBay search parameters."""
    return {
        "_nkw": config["query"],
        "_sacat": str(config["category_id"]) if config["category_id"] else Null,
        "_from": "R40",
        "rt": "nc",
        "LH_Sold": "1",
        "LH_Complete": "1",
        "Country/Region of Manufacture": (
            "United Kingdom" if config["region_str"] == "uk" else "United States"
        ),
        "_ipg": "240",
        "_pgn": str(page),
        "_sop": "13",
    }


# ==============================================================================
# DATA EXTRACTION FUNCTIONS (Called by scraping functions)
# ==============================================================================


def _extract_item_data(item, config: Dict) -> Optional[Dict]:
    """Extract data from a single eBay item element."""

    # Extract listing ID from data-listingid attribute
    listing_id = item.get("data-listingid")
    if not listing_id:
        return None

    # Title - exact selectors matching the HTML structure
    title = None
    title_el = item.select_one(".s-card__title .su-styled-text")
    if title_el and title_el.get_text().strip():
        title = title_el.get_text(strip=True)

    # Fallback title selectors
    if not title:
        title_selectors = [
            ".s-card__title",
            ".s-item__title",
            "h3 .su-styled-text",
        ]
        for sel in title_selectors:
            title_el = item.select_one(sel)
            if title_el and title_el.get_text().strip():
                title = title_el.get_text(strip=True)
                break

    if not title or title.lower() == "shop on ebay":
        return None

    # Price - exact selector matching HTML structure
    price = None
    currency = None
    price_el = item.select_one(".su-styled-text.s-card__price")

    if price_el:
        price_text = price_el.get_text(strip=True)
        # Extract currency and price
        if "£" in price_text:
            currency = "GBP"
            price_match = re.search(r"£([\d,]+\.?\d*)", price_text.replace(",", ""))
        elif "$" in price_text:
            currency = "USD"
            price_match = re.search(r"\$([\d,]+\.?\d*)", price_text.replace(",", ""))
        elif "€" in price_text:
            currency = "EUR"
            price_match = re.search(r"€([\d,]+\.?\d*)", price_text.replace(",", ""))
        else:
            # Generic numeric extraction
            price_match = re.search(r"([\d,]+\.?\d*)", price_text.replace(",", ""))
            currency = "GBP" if config["region_str"] == "uk" else "USD"

        if price_match:
            price = price_match.group(1)  # Keep as string to match database format

    if not price:
        return None

    # URL - extract from href attribute
    url = ""
    link_el = item.select_one("a[href*='/itm/']")
    if link_el:
        url = link_el.get("href")

    if not url:
        return None

    # Image URLs - extract and create both standard and HD versions
    image_url = ""
    image_hd_url = ""

    # Try multiple image selectors
    img_el = item.select_one(".s-card__image")
    if not img_el:
        img_el = item.select_one("img[src*='ebayimg.com']")
    if not img_el:
        img_el = item.select_one("img[data-defer-load*='ebayimg.com']")

    if img_el:
        # Get image URL from src or data-defer-load
        src = img_el.get("src") or img_el.get("data-defer-load")
        if src and "ebayimg.com" in src:
            # Standard image (s-l140)
            if "s-l140" in src:
                image_url = src
                image_hd_url = src.replace("s-l140", "s-l1600")
            elif "s-l500" in src:
                image_url = src.replace("s-l500", "s-l140")
                image_hd_url = src.replace("s-l500", "s-l1600")
            else:
                image_url = src
                image_hd_url = src

    # Sold date - exact selector matching HTML structure
    sold_text = ""
    sold_el = item.select_one(".s-card__caption .su-styled-text")
    if sold_el:
        sold_text = sold_el.get_text(strip=True)
        # Only use if it contains "Sold" text
        if "sold" not in sold_text.lower():
            sold_text = ""

    sold_date = _parse_sold_date(sold_text, config["region_str"])

    # Extract condition, grading company, and grade from title
    condition, grading_company, grade = get_condition_and_grade(title)

    # Get master_card_id from user_cards if user_card_id is provided
    master_card_id = None
    if config.get("user_card_id"):
        try:
            response = supabase.table("user_cards").select("master_card_id").eq("id", config["user_card_id"]).execute()
            if response.data:
                master_card_id = response.data[0].get("master_card_id")
        except Exception:
            # If query fails, keep master_card_id as None
            pass

    return {
        "id": listing_id,
        "master_card_id": master_card_id,
        "name": title,
        "image_url": image_url,
        "region": config["region_str"],
        "sold_at": sold_date,
        "currency": currency,
        "price": price,
        "condition": condition,
        "grading_company": grading_company,
        "grade": grade,
        "post_url": url,
        "blocked": False,
        "user_card_ids": [],  # Empty array, will be populated during upsert
        "normalised_name": _normalize_name(title),
        "category_id": config["category_id"],
        "image_hd_url": image_hd_url,
    }


def _parse_page_items(html: str, config: Dict) -> List[Dict]:
    """Parse eBay items from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    # Focus on actual product listings with data-listingid (skip promotional items)
    items = soup.select("li[data-listingid]")
    records = []

    scraper_logger.info(f"🔍 Found {len(items)} product items with listing IDs")

    for item in items:
        try:
            record = _extract_item_data(item, config)
            if record and record.get("name") and record.get("price"):
                records.append(record)
        except Exception as e:
            scraper_logger.warning(f"⚠️ Failed to extract item data: {e}")
            continue

    scraper_logger.info(f"✅ Successfully extracted {len(records)} valid records")
    return records


# ==============================================================================
# DATABASE FUNCTIONS (Called by save functions)
# ==============================================================================


def _upsert_records_batch(records: List[Dict], user_card_id: Optional[str]):
    """Insert records into the ebay_posts table, avoiding eBay ID reuse conflicts."""
    batch_size = 1000

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]

        # Add user_card_id if provided
        if user_card_id:
            for record in batch:
                record["user_card_id"] = user_card_id

        try:
            scraper_logger.info(f"💾 Inserting {len(batch)} records to ebay_posts table")
            # Use insert instead of upsert to handle eBay ID reuse
            # eBay reuses listing IDs across different auctions, so we want separate records
            supabase.table("ebay_posts").insert(batch, returning="minimal").execute()
            scraper_logger.info(f"✅ Successfully inserted {len(batch)} records to ebay_posts")
        except Exception as e:
            from src.utils.logger import log_error_with_context

            log_error_with_context(scraper_logger, e, f"batch insert to ebay_posts ({len(batch)} records)")
            scraper_logger.error(f"❌ Failed to insert {len(batch)} records to ebay_posts: {str(e)}")
            
            # If insert fails due to duplicate, try with upsert as fallback
            if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                scraper_logger.warning("🔄 Duplicate detected, using upsert with composite key")
                try:
                    # Add timestamp to make record unique
                    from datetime import datetime
                    current_time = datetime.utcnow().isoformat()
                    
                    for record in batch:
                        record["scraped_at"] = current_time
                    
                    supabase.table("ebay_posts").upsert(batch, on_conflict="id,price,sold_at").execute()
                    scraper_logger.info(f"✅ Successfully upserted {len(batch)} records with composite key")
                except Exception as fallback_error:
                    scraper_logger.error(f"❌ Fallback upsert also failed: {str(fallback_error)}")
                    raise
            else:
                raise  # Re-raise to ensure the error propagates


async def _update_user_card(records: List[Dict], user_card_id: str):
    """Update user card with latest sold information."""
    try:
        latest = max(records, key=lambda r: _safe_parse_date(r.get("sold_at")))

        update_data = {
            "last_sold_currency": latest.get("currency"),
            "last_sold_price": latest.get("price"),
            "last_sold_post_url": latest.get("post_url"),
            "last_sold_at": latest.get("sold_at"),
        }

        supabase.table("user_cards").update(update_data).eq(
            "id", user_card_id
        ).execute()

    except Exception as e:
        from src.utils.logger import log_error_with_context

        log_error_with_context(scraper_logger, e, "updating user card")


async def _handle_failed_scrape(config: Dict):
    """Handle failed scrape by logging and updating user card."""
    user_card_id = config["user_card_id"]
    if not user_card_id:
        return

    try:
        # Update user card with zero values
        supabase.table("user_cards").update(
            {
                "last_sold_price": 0,
                "last_sold_currency": (
                    "GBP" if config["region_str"] == "uk" else "USD"
                ),
            }
        ).eq("id", user_card_id).execute()

        # Log failed scrape
        supabase.table("failed_scrape_cards").insert(
            {"user_card_id": user_card_id, "name": config["query"]}
        ).execute()

    except Exception as e:
        from src.utils.logger import log_error_with_context

        log_error_with_context(scraper_logger, e, "logging failed scrape")


# ==============================================================================
# CONFIGURATION FUNCTIONS (Called by main scraper)
# ==============================================================================


def _resolve_scrape_params(
    query: Optional[str], category_id: Optional[CategoryId], user_card_id: Optional[str]
) -> tuple[str, CategoryId, Optional[str]]:
    """Resolve final query and category_id from either direct params or user_card_id.
    Returns: (final_query, final_category_id, master_card_id)
    """

    # Mode 1: Direct query + category_id provided
    if query and category_id:
        return query, category_id, None

    # Mode 2: user_card_id provided - resolve from database
    if user_card_id:
        try:
            response = (
                supabase.table("user_cards")
                .select("custom_name, category_id, master_card_id")
                .eq("id", user_card_id)
                .limit(1)
                .execute()
            )

            if response.data:
                card_data = response.data[0]
                resolved_query = card_data.get("custom_name") or "card"
                resolved_category = (
                    CategoryId(card_data.get("category_id"))
                    if card_data.get("category_id")
                    else category_id or CategoryId.PANINI
                )
                master_card_id = card_data.get("master_card_id")
                scraper_logger.info(
                    f"✅ Resolved user_card_id {user_card_id[:8]}... to query: '{resolved_query}', category: {resolved_category.value if hasattr(resolved_category, 'value') else resolved_category}"
                )
                return resolved_query, resolved_category, master_card_id
            else:
                scraper_logger.warning(
                    f"⚠️ User card {user_card_id} not found, using fallback"
                )
        except Exception as e:
            scraper_logger.warning(f"⚠️ Error resolving user card {user_card_id}: {e}")

    # Fallback: use provided params or defaults
    final_query = query or "card"
    final_category = category_id or CategoryId.PANINI
    return final_query, final_category, None


def _setup_scrape_config(
    region: Region,
    query: str,
    category_id: CategoryId,
    user_card_id: Optional[str],
    master_card_id: Optional[str] = None,
) -> Dict:
    """Setup scraping configuration with resolved parameters."""
    config = {
        "region": region.value,
        "region_str": region.value,
        "query": query,
        "category_id": category_id.value,
        "user_card_id": user_card_id,
        "base_url": f"{base_target_url[region.value]}/sch/i.html",
        "master_card_id": master_card_id,
    }

    return config


# ==============================================================================
# SCRAPING FUNCTIONS (Called by main scraper)
# ==============================================================================


async def _scrape_single_page(config: Dict, page: int, max_retries: int, disable_proxy: bool = False) -> List[Dict]:
    """Scrape a single page with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            params = _build_search_params(config, page)

            # Anti-bot randomization
            timeout = random.uniform(12.0, 18.0)
            jitter_min = random.uniform(0.5, 1.2)
            jitter_max = random.uniform(1.5, 3.0)

            html = await httpx_get_content(
                config["base_url"],
                params=params,
                attempts=5,
                request_timeout=timeout,
                jitter_min=jitter_min,
                jitter_max=jitter_max,
                use_proxy=not disable_proxy,
            )

            if html:
                return _parse_page_items(html, config)

        except Exception as e:
            scraper_logger.warning(
                f"⚠️ Error scraping page {page} attempt {attempt}: {e}"
            )
            if attempt < max_retries:
                await asyncio.sleep(random.uniform(2.0, 5.0))

    return []


async def _scrape_all_pages(config: Dict, total_pages: int, retries: int, disable_proxy: bool = False) -> List[Dict]:
    """Scrape all pages concurrently with proper delays."""
    all_records = []

    for page in range(1, total_pages + 1):
        # Human-like delay between pages
        if page > 1:
            delay = random.uniform(1.0, 3.5) + (page - 1) * random.uniform(0.2, 0.8)
            await asyncio.sleep(delay)

        records = await _scrape_single_page(config, page, retries, disable_proxy)
        all_records.extend(records)

    return all_records


async def _get_page_count(config: Dict, disable_proxy: bool = False) -> int:
    """Get total pages available for the query."""
    try:
        params = _build_search_params(config, page=1)
        html = await httpx_get_content(
            config["base_url"], params=params, use_proxy=not disable_proxy
        )

        if not html:
            return 1

        soup = BeautifulSoup(html, "html.parser")
        count_element = soup.select_one(
            ".srp-controls__count-heading, .result-count__count-heading"
        )

        if not count_element:
            return 1

        bold = count_element.find("span", class_="BOLD")
        text = (
            bold.get_text(strip=True)
            if bold
            else count_element.get_text(" ", strip=True)
        )

        match = re.search(r"(\d[\d,]*)", text)
        if match:
            total_results = int(match.group(1).replace(",", ""))
            return max(1, math.ceil(total_results / 240))

    except Exception as e:
        scraper_logger.warning(f"⚠️ Error getting page count: {e}")

    return 1


# ==============================================================================
# DATA PROCESSING FUNCTIONS (Called by main scraper)
# ==============================================================================


def _process_records(records: List[Dict], config: Dict) -> List[Dict]:
    """Process and deduplicate records."""
    if not records:
        return []

    # Deduplicate by URL
    seen_urls = set()
    unique_records = []

    for record in records:
        url = record.get("post_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_records.append(record)

    return unique_records


async def _save_results(records: List[Dict], config: Dict):
    """Save results to database and update user card."""
    if not records:
        await _handle_failed_scrape(config)
        return

    # Batch upsert records
    # _upsert_records_batch(records, config["user_card_id"])

    # Update user card with latest sold info
    if config["user_card_id"]:
        await _update_user_card(records, config["user_card_id"])


# ==============================================================================
# MAIN SCRAPER CLASS AND FUNCTIONS
# ==============================================================================


class EbayScraper:
    """Clean, well-structured eBay scraper."""

    def __init__(self):
        self.region_codes = {"us": "us", "uk": "uk"}

    async def scrape(
        self,
        region: Region,
        query: str,
        category_id: CategoryId,
        user_card_id: Optional[str] = None,
        max_pages: int = 50,
        page_retries: int = 3,
        disable_proxy: bool = False,
    ) -> Dict:
        """Main scraping function - clean and well-structured."""

        # 1. Setup and validation
        scrape_config = _setup_scrape_config(region, query, category_id, user_card_id)
        if "error" in scrape_config:
            return scrape_config

        # 2. Determine pages to scrape
        total_pages = min(max_pages, await _get_page_count(scrape_config, disable_proxy))

        # 3. Scrape all pages concurrently for speed
        all_records = await _scrape_all_pages(scrape_config, total_pages, page_retries, disable_proxy)

        # 4. Process and deduplicate results
        processed_records = _process_records(all_records, scrape_config)

        # 5. Save to database and update user card
        await _save_results(processed_records, scrape_config)

        return {"total": len(processed_records), "result": processed_records}


# ==============================================================================
# WORKER MANAGEMENT SYSTEM
# ==============================================================================

from enum import Enum
from dataclasses import dataclass


class WorkerStatus(str, Enum):
    """Worker execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkerTask:
    """Worker task data structure."""

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
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
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
            "error_message": self.error_message,
        }


class EbayWorkerManager:
    """Manages eBay scraping workers and task status."""

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
        """Create new worker task with unified parameter resolution."""
        task_id = str(uuid.uuid4())

        # Resolve final parameters from either mode
        final_query, final_category_id, master_card_id = _resolve_scrape_params(
            query, category_id, user_card_id
        )

        task = WorkerTask(
            id=task_id,
            query=final_query,
            region=region.value,
            category_id=(
                final_category_id.value
                if hasattr(final_category_id, "value")
                else final_category_id
            ),
            status=WorkerStatus.PENDING,
            created_at=datetime.utcnow(),
            user_card_id=user_card_id,
            max_pages=max_pages,
        )

        self.active_tasks[task_id] = task

        # Worker will be started by the FastAPI endpoint

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
        """Execute scraping task in background with unified approach."""
        task = self.active_tasks[task_id]

        try:
            # Update status to running
            task.status = WorkerStatus.RUNNING
            task.started_at = datetime.utcnow()

            # Execute scraping with unified parameters - always uses resolved query
            result = await scraper.scrape(
                region=region,
                query=query,  # Already resolved from either mode
                category_id=category_id,  # Already resolved from either mode
                user_card_id=user_card_id,
                max_pages=max_pages,
                page_retries=3,
                disable_proxy=disable_proxy,
            )

            # Update task with results
            task.result_count = result.get("total", 0)
            task.status = WorkerStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            scraper_logger.info(
                f"✅ Task {task_id} completed successfully with {task.result_count} results"
            )

            # Keep completed tasks in active_tasks so user can check status via /v1/workers

        except Exception as e:
            task.status = WorkerStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            scraper_logger.error(f"❌ Task {task_id} failed: {str(e)}")
            # Keep failed tasks in active_tasks for debugging

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
                del self.active_tasks[task.id]


# ==============================================================================
# GLOBAL INSTANCES AND COMPATIBILITY
# ==============================================================================

# Global scraper instance
scraper = EbayScraper()

# Global worker manager instance
worker_manager = EbayWorkerManager()


# Backwards compatibility function
async def ebay_scrape_handler(*args, **kwargs):
    """Backwards compatibility wrapper."""
    return await scraper.scrape(*args, **kwargs)

import uuid
from enum import Enum
from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from src.utils.logger import scraper_logger
from src.models.category import CategoryId
from src.models.ebay import Region
from src.utils.scraper import (
    scrape_ebay_html,
    extract_ebay_post_data,
    update_card,
    upsert_ebay_listings,
    get_card,
    build_strict_promisable_candidates,
    select_best_candidate_with_gpt,
)

# --------------------------
# Main Handler
# --------------------------
class EbayScraper:
    def __init__(self):
        self.region_codes = {"us": "us", "uk": "uk"}

    async def scrape(
        self,
        region: Region,
        query: Optional[str] = None,
        category_id: Optional[CategoryId] = None,
        card_id: Optional[str] = None,
        max_pages: int = 50,
        disable_proxy: bool = False,
    ) -> Dict[str, Any]:
        try:
            # ===============================================================
            # Step 1: Scrape Ebay html
            # ===============================================================
            html_pages = await scrape_ebay_html(
                region=region,
                # mode: query + category_id
                query=query,
                category_id=category_id,
                # mode: card_id
                card_id=card_id,
                max_pages=max_pages,
                disable_proxy=disable_proxy,
            )

            if not html_pages:
                scraper_logger.error(
                    f'❌ failed fetch ebay results for query: ------ "{query}"'
                )

            # ===============================================================
            # Step 2: Extract data from HTML
            # ===============================================================

            posts = await extract_ebay_post_data(
                html_pages=html_pages,
                region=region.value,
                category_id=category_id.value if category_id else None,
            )

            # ===============================================================
            # Step 3: Store ebay listings into supabase tables ebay_posts
            # ===============================================================

            upsert_results = await upsert_ebay_listings(posts, card_id=card_id)

            # ===============================================================
            # Step 3 (card_id mode only): Building a "promisable candidates" list by strictly matching all non-null fields with medium/high proofs
            # ===============================================================

            if card_id:

                selected_card = await get_card(card_id)
                attribute_filters = {k: v for k, v in selected_card.items() if v}
                # Normalize years without splitting into characters
                try:
                    years_val = selected_card.get("years") if selected_card else None
                    if isinstance(years_val, (list, tuple, set)):
                        yf = " ".join(str(y) for y in years_val if y)
                        if yf:
                            attribute_filters["years"] = yf
                    elif years_val is not None:
                        attribute_filters["years"] = str(years_val)
                    # else: do not force-add years
                except Exception:
                    # On any failure, fall back to removing malformed years to avoid over-filtering
                    attribute_filters.pop("years", None)

                candidates_result = build_strict_promisable_candidates(
                    attribute_filters=attribute_filters,
                    posts=posts,
                )
                match_posts = candidates_result.get("match_posts", [])
                # unmatch_posts = candidates_result.get("unmatch_posts", [])

                # ===============================================================
                # Step 4 (card_id mode only): Selecting the best candidate using GPT
                # ===============================================================
                best_matched_post = None
                last_sold_post = await select_best_candidate_with_gpt(
                    canonical_title=(
                        selected_card.get("canonical_title") if selected_card else None
                    ),
                    match_posts=match_posts,
                )

                if last_sold_post:
                    await update_card(
                        selected_post=last_sold_post.get("last_sold_post"),
                        gpt_reason=last_sold_post.get("reason"),
                        card_id=card_id,
                    )
                    best_matched_post = last_sold_post

                results = {
                    "result_count": len(posts),
                    "upsert_count": len(upsert_results),
                    "best_matched_post": best_matched_post,
                }
                scraper_logger.info(f"[result count] --- {results['result_count']}")
                scraper_logger.info(f"[upsert count] --- {results['upsert_count']}")
                scraper_logger.info(f"[last sold post] --- {results['best_matched_post']}")
                scraper_logger.info(
                    f"🎉 Scraping [card_id] completed: {len(posts)} posts processed"
                )
                return results

            else:
                results = {
                    "result_count": len(posts),
                    "upsert_count": len(upsert_results),
                }
                scraper_logger.info(f"result_count: {results['result_count']}")
                scraper_logger.info(f"upsert_count: {results['upsert_count']}")
                scraper_logger.info(
                    f"🎉 Scraping [query + category_id] completed: {len(posts)} posts processed"
                )
                return results

        except Exception as e:
            scraper_logger.error(f"❌ Scraping failed: {e}")
            raise


# Worker manager kept for compatibility (same as before)
class WorkerStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkerTask:
    id: str
    query: Optional[str]
    region: str
    category_id: Optional[int]
    status: WorkerStatus
    created_at: datetime
    card_id: Optional[str] = None
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
            "card_id": self.card_id,
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
    def __init__(self):
        self.active_tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=3)

    def create_task(
        self,
        query: Optional[str] = None,
        region: Region = Region.us,
        category_id: Optional[CategoryId] = None,
        card_id: Optional[str] = None,
        max_pages: int = 50,
    ) -> WorkerTask:
        task_id = str(uuid.uuid4())
        final_query = query
        final_category_id = category_id
        task = WorkerTask(
            id=task_id,
            query=final_query,
            region=region.value,
            category_id=final_category_id.value if final_category_id else None,
            status=WorkerStatus.PENDING,
            created_at=datetime.utcnow(),
            card_id=card_id,
            max_pages=max_pages,
        )
        self.active_tasks[task_id] = task
        return task

    # API helpers used by src/api/v1/endpoint.py
    def get_all_tasks(self) -> list[WorkerTask]:
        """Return a list of all current tasks (pending/running/completed/failed)."""
        return list(self.active_tasks.values())

    def get_task_status(self, task_id: str) -> Optional[WorkerTask]:
        """Return a single task by ID or None if not found."""
        return self.active_tasks.get(task_id)

    def cleanup_old_tasks(self, max_age_hours: int = 12) -> None:
        """Remove completed tasks older than max_age_hours to avoid memory growth."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_delete = [
            tid
            for tid, t in self.active_tasks.items()
            if t.completed_at and t.completed_at < cutoff
        ]
        for tid in to_delete:
            try:
                del self.active_tasks[tid]
            except KeyError:
                pass

    async def _run_scrape_worker(
        self,
        task_id: str,
        region: Region,
        category_id: Optional[CategoryId],
        query: Optional[str],
        card_id: Optional[str],
        max_pages: int,
        master_card_id: Optional[str] = None,
        disable_proxy: bool = False,
    ):
        task = self.active_tasks[task_id]
        try:
            task.status = WorkerStatus.RUNNING
            task.started_at = datetime.utcnow()
            result = await scraper.scrape(
                region=region,
                query=query,
                category_id=category_id,
                card_id=card_id,
                max_pages=max_pages,
                disable_proxy=disable_proxy,
            )
            task.result_count = result.get("total", 0)
            task.upserted_count = result.get("upsert_results", {}).get(
                "upserted_count", 0
            )
            task.user_updated = bool(
                result.get("user_update_results", {}).get("card_updated", False)
            )
            task.results = result
            task.status = WorkerStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            scraper_logger.info(f"✅ Task {task_id} completed")
        except Exception as e:
            task.status = WorkerStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            scraper_logger.error(f"❌ Task {task_id} failed: {e}")


# init
scraper = EbayScraper()
worker_manager = EbayWorkerManager()

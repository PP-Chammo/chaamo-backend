import uuid
from datetime import datetime
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.models.category import CategoryId
from src.models.ebay import Region
from src.utils.supabase import supabase
from src.utils.scraper import (
    scrape_ebay_html,
    embed_posts,
    extract_ebay_post_data,
    select_best_ebay_post,
    upsert_ebay_listings,
    update_user_card,
)

scraper_logger = get_logger("scraper")


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
        user_card_id: Optional[str] = None,
        max_pages: int = 50,
        page_retries: int = 3,
        disable_proxy: bool = False,
        rerank_with_gpt: bool = True,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
        - Step 1: scrape & normalize
        - Step 2: embed posts
        - Step 3: nn + optional gpt rerank (if user_card_id provided we pick best)
        - Step 4: store vectors + upsert posts, update user_card if applied
        """
        try:
            # resolve user_card query early
            resolved_query = query or "card"
            if user_card_id:
                try:
                    resp = (
                        supabase.table("user_cards")
                        .select("custom_name, category_id")
                        .eq("id", user_card_id)
                        .limit(1)
                        .execute()
                    )
                    if resp.data:
                        resolved_query = (
                            resp.data[0].get("custom_name") or resolved_query
                        )
                except Exception as e:
                    scraper_logger.warning(f"⚠️ Failed to resolve user_card: {e}")

            # ===============================================================
            # Step 1: Scrape Ebay html
            # ===============================================================

            html_pages = await scrape_ebay_html(
                region=region,
                query=query,
                category_id=category_id.value,
                user_card_id=user_card_id,
                max_pages=max_pages,
                page_retries=page_retries,
                disable_proxy=disable_proxy,
            )

            if not html_pages:
                return {"total": 0, "result": []}

            # ===============================================================
            # Step 2: Extract data from HTML
            # ===============================================================

            posts = await extract_ebay_post_data(
                html_pages=html_pages,
                region=region.value,
                category_id=category_id.value if category_id else None,
            )

            # ===============================================================
            # Step 3: Add Embedding into extracted data
            # ===============================================================

            await embed_posts(posts)

            # ===============================================================
            # Step 4: Pick best post using NN + optional GPT rerank
            # ===============================================================

            selected_post_info = None
            if user_card_id:
                selected_post_info = await select_best_ebay_post(
                    posts, resolved_query, top_k=top_k, rerank_with_gpt=rerank_with_gpt
                )

            # ===============================================================
            # Step 5: Store vectors & posts into supabase tables ebay_posts
            # ===============================================================

            upsert_results = await upsert_ebay_listings(
                posts, user_card_id=user_card_id
            )

            # ===============================================================
            # Step 6: Update user_card using selected
            # ===============================================================

            user_update_results = {}
            if (
                user_card_id
                and selected_post_info
                and selected_post_info.get("found")
                and selected_post_info.get("match")
            ):
                selected = selected_post_info["match"]

                # ===============================================================
                # Log selected item details
                # ===============================================================

                scraper_logger.info(
                    f"🎯 Selected best match for user_card {user_card_id}:\n"
                    f"   📝 Title: {selected.get('title')}\n"
                    f"   💰 Price: {selected.get('sold_price')} {selected.get('sold_currency')}\n"
                    f"   🔗 URL: {selected.get('sold_post_url')}\n"
                    f"   📊 Similarity: {selected.get('similarity', 0):.4f}\n"
                    f"   🏷️ Set: {selected.get('metadata', {}).get('normalized_attributes', {}).get('set', 'None')}"
                )

                user_update_results = await update_user_card(selected, user_card_id)

            results = {
                "total": len(posts),
                "result": posts,
                "selected": selected_post_info,
                "upsert_results": upsert_results,
                "user_update_results": user_update_results,
            }
            scraper_logger.info(
                f"🎉 Scraping pipeline completed: {len(posts)} posts processed"
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
    def __init__(self):
        self.active_tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=3)

    def create_task(
        self,
        query: Optional[str] = None,
        region: Region = Region.us,
        category_id: Optional[CategoryId] = None,
        user_card_id: Optional[str] = None,
        max_pages: int = 50,
    ) -> WorkerTask:
        task_id = str(uuid.uuid4())
        final_query = query or "card"
        final_category_id = category_id or CategoryId.TOPPS
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
        task = self.active_tasks[task_id]
        try:
            task.status = WorkerStatus.RUNNING
            task.started_at = datetime.utcnow()
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
                result.get("user_update_results", {}).get("user_card_updated", False)
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

"""Distributed scheduler for eBay scraping tasks with proper multi-instance coordination."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.ebay_scraper import scraper
from src.models.ebay import Region
from src.models.category import CategoryId
from src.utils.supabase import supabase


# ==============================================================================
# CONFIGURATION AND UTILITY FUNCTIONS
# ==============================================================================


def _setup_logging():
    """Setup clean logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def _get_instance_id() -> str:
    """Get unique instance ID for this application instance."""
    import os
    # Use Fly.io instance ID if available, otherwise generate UUID
    return os.environ.get('FLY_ALLOC_ID', str(uuid.uuid4()))


async def _acquire_job_lock(job_name: str, instance_id: str, lock_duration_minutes: int = 30) -> bool:
    """Acquire distributed job lock to prevent duplicate execution across instances."""
    try:
        expires_at = datetime.utcnow() + timedelta(minutes=lock_duration_minutes)
        
        # Try to insert new lock
        result = supabase.table("scheduler_locks").insert({
            "job_name": job_name,
            "instance_id": instance_id,
            "locked_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "status": "active"
        }).execute()
        
        if result.data:
            return True
            
    except Exception as e:
        # Lock already exists or other error - check if we can take over expired lock
        try:
            # Clean up expired locks first
            supabase.table("scheduler_locks").delete().match({
                "job_name": job_name
            }).filter("expires_at", "lt", datetime.utcnow().isoformat()).execute()
            
            # Try to acquire again after cleanup
            result = supabase.table("scheduler_locks").insert({
                "job_name": job_name,
                "instance_id": instance_id,
                "locked_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at.isoformat(),
                "status": "active"
            }).execute()
            
            return bool(result.data)
            
        except Exception:
            return False
    
    return False


async def _release_job_lock(job_name: str, instance_id: str) -> bool:
    """Release job lock after completion."""
    try:
        supabase.table("scheduler_locks").delete().match({
            "job_name": job_name,
            "instance_id": instance_id
        }).execute()
        return True
    except Exception:
        return False


async def _ensure_scheduler_table():
    """Ensure scheduler_locks table exists."""
    try:
        # Test if table exists by trying a simple query
        supabase.table("scheduler_locks").select("job_name").limit(1).execute()
    except Exception:
        # Table likely doesn't exist - this is expected on first run
        # Fly.io deployment will need to create this table manually or via migration
        pass


def _build_scrape_configs() -> List[Dict]:
    """Build clean, non-duplicate scrape configurations."""
    categories = [
        (CategoryId.TOPPS, "Topps card"),
        (CategoryId.PANINI, "Panini card"),
        (CategoryId.POKEMON, "Pokemon card"),
        (CategoryId.MARVEL, "Marvel card"),
        (CategoryId.DC, "DC card"),
        (CategoryId.YU_GI_OH, "Yu-Gi-Oh card"),
        (CategoryId.LORCANA, "Lorcana card"),
        (CategoryId.WRESTLING, "Wrestling card"),
        (CategoryId.DIGIMON, "Digimon card"),
        (CategoryId.FORTNITE, "Fortnite card"),
        (CategoryId.FUTERA, "Futera card"),
        (CategoryId.GARBAGE_PAIL_KIDS, "Garbage Pail Kids card"),
    ]

    regions = [Region.us, Region.uk]
    configs = []

    for category_id, query in categories:
        for region in regions:
            configs.append(
                {"query": query, "category_id": category_id, "region": region}
            )

    return configs


# ==============================================================================
# SCRAPING EXECUTION FUNCTIONS
# ==============================================================================


async def _execute_single_scrape(config: Dict, index: int, total: int, logger) -> Dict:
    """Execute single scrape configuration."""
    query = config["query"]
    category_id = config["category_id"]
    region = config["region"]

    logger.info(f"[{index}/{total}] Scraping: {query} | {region.value.upper()}")

    try:
        result = await scraper.scrape(
            region=region,
            query=query,
            category_id=category_id,
            max_pages=10,
            page_retries=3,
        )

        count = result.get("total", 0)
        logger.info(f"   ✅ Completed: {count} records")

        return {"success": True, "count": count}

    except Exception as e:
        logger.error(f"   ❌ Failed: {e}")
        return {"success": False, "error": str(e)}


async def _execute_scrape_batch(configs: List[Dict], logger) -> Dict:
    """Execute batch of scrape configurations."""
    total_configs = len(configs)
    success_count = 0
    total_scraped = 0

    logger.info(f"🚀 Starting daily scrape batch: {total_configs} configurations")

    for i, config in enumerate(configs, 1):
        result = await _execute_single_scrape(config, i, total_configs, logger)

        if result["success"]:
            success_count += 1
            total_scraped += result["count"]

        # Brief pause between scrapes
        if i < total_configs:
            await asyncio.sleep(2)

    logger.info(
        f"🏁 Daily scrape completed: {success_count}/{total_configs} successful, {total_scraped} total records"
    )

    return {
        "total_configs": total_configs,
        "success_count": success_count,
        "total_scraped": total_scraped,
    }


# ==============================================================================
# SCHEDULER CLASS
# ==============================================================================


class EbayScheduler:
    """Distributed eBay scraping scheduler with multi-instance coordination."""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        self.logger = _setup_logging()
        self.scrape_configs = _build_scrape_configs()
        self.instance_id = _get_instance_id()

    async def run_daily_scrape(self):
        """Execute daily scraping batch with distributed locking."""
        if self.is_running:
            self.logger.warning("Daily scrape already running locally, skipping execution")
            return

        job_name = "daily_ebay_scrape"
        
        # Try to acquire distributed lock
        lock_acquired = await _acquire_job_lock(job_name, self.instance_id, lock_duration_minutes=60)
        
        if not lock_acquired:
            self.logger.info("🔒 Another instance is already running daily scrape, skipping")
            return

        self.logger.info(f"🔑 Acquired job lock for instance {self.instance_id}")
        self.is_running = True

        try:
            await _execute_scrape_batch(self.scrape_configs, self.logger)
        except Exception as e:
            self.logger.error(f"💥 Critical error in daily scrape: {e}")
        finally:
            self.is_running = False
            # Always release lock when done
            await _release_job_lock(job_name, self.instance_id)
            self.logger.info(f"🔓 Released job lock for instance {self.instance_id}")

    def start(self):
        """Start the scheduler - runs daily at 3 AM."""
        self.scheduler.add_job(
            self.run_daily_scrape,
            CronTrigger(hour=3, minute=0),
            id="daily_ebay_scrape",
            name="Daily eBay Scrape Job",
            replace_existing=True,
        )

        self.scheduler.start()
        self.logger.info("📅 eBay scheduler started - runs daily at 03:00 UTC")

    def stop(self):
        """Stop the scheduler cleanly."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            self.logger.info("⏹️  eBay scheduler stopped")


# ==============================================================================
# GLOBAL INSTANCES AND PUBLIC API
# ==============================================================================

# Global scheduler instance
ebay_scheduler = EbayScheduler()


def start_ebay_cronjob():
    """Start the eBay cronjob scheduler."""
    ebay_scheduler.start()


def stop_ebay_cronjob():
    """Stop the eBay cronjob scheduler."""
    ebay_scheduler.stop()

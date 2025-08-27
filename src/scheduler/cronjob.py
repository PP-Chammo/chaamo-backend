import asyncio
import logging
from typing import List, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from src.handlers.ebay_scrape import ebay_scrape_handler
from src.models.ebay import Region
from src.models.category import CategoryId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurable scrape list - add/remove queries as needed
SCRAPE_LIST = [
    {"query": "Topps card", "category_id": CategoryId.TOPPS, "region": Region.uk},
    {"query": "Topps card", "category_id": CategoryId.TOPPS, "region": Region.us},
    {"query": "Panini card", "category_id": CategoryId.PANINI, "region": Region.uk},
    {"query": "Panini card", "category_id": CategoryId.PANINI, "region": Region.us},
    {"query": "Futera card", "category_id": CategoryId.FUTERA, "region": Region.uk},
    {"query": "Futera card", "category_id": CategoryId.FUTERA, "region": Region.us},
    {"query": "Pokemon card", "category_id": CategoryId.POKEMON, "region": Region.uk},
    {"query": "Pokemon card", "category_id": CategoryId.POKEMON, "region": Region.us},
    {"query": "DC card", "category_id": CategoryId.DC, "region": Region.uk},
    {"query": "DC card", "category_id": CategoryId.DC, "region": Region.us},
    {"query": "Fortnite card", "category_id": CategoryId.FORTNITE, "region": Region.uk},
    {"query": "Fortnite card", "category_id": CategoryId.FORTNITE, "region": Region.us},
    {"query": "Marvel card", "category_id": CategoryId.MARVEL, "region": Region.uk},
    {"query": "Marvel card", "category_id": CategoryId.MARVEL, "region": Region.us},
    {"query": "Garbage Pail Kids card", "category_id": CategoryId.GARBAGE_PAIL_KIDS, "region": Region.uk},
    {"query": "Garbage Pail Kids card", "category_id": CategoryId.GARBAGE_PAIL_KIDS, "region": Region.us},
    {"query": "Digimon card", "category_id": CategoryId.DIGIMON, "region": Region.uk},
    {"query": "Digimon card", "category_id": CategoryId.DIGIMON, "region": Region.us},
    {"query": "Wrestling card", "category_id": CategoryId.WRESTLING, "region": Region.uk},
    {"query": "Wrestling card", "category_id": CategoryId.WRESTLING, "region": Region.us},
    {"query": "Yu-Gi-Oh card", "category_id": CategoryId.YU_GI_OH, "region": Region.uk},
    {"query": "Yu-Gi-Oh card", "category_id": CategoryId.YU_GI_OH, "region": Region.us},
    {"query": "Lorcana card", "category_id": CategoryId.LORCANA, "region": Region.uk},
    {"query": "Lorcana card", "category_id": CategoryId.LORCANA, "region": Region.us},
    {"query": "Topps trading card", "category_id": CategoryId.TOPPS, "region": Region.uk},
    {"query": "Topps trading card", "category_id": CategoryId.TOPPS, "region": Region.us},
    {"query": "Panini trading card", "category_id": CategoryId.PANINI, "region": Region.uk},
    {"query": "Panini trading card", "category_id": CategoryId.PANINI, "region": Region.us},
    {"query": "Futera trading card", "category_id": CategoryId.FUTERA, "region": Region.uk},
    {"query": "Futera trading card", "category_id": CategoryId.FUTERA, "region": Region.us},
    {"query": "Pokemon trading card", "category_id": CategoryId.POKEMON, "region": Region.uk},
    {"query": "Pokemon trading card", "category_id": CategoryId.POKEMON, "region": Region.us},
    {"query": "DC trading card", "category_id": CategoryId.DC, "region": Region.uk},
    {"query": "DC trading card", "category_id": CategoryId.DC, "region": Region.us},
    {"query": "Fortnite trading card", "category_id": CategoryId.FORTNITE, "region": Region.uk},
    {"query": "Fortnite trading card", "category_id": CategoryId.FORTNITE, "region": Region.us},
    {"query": "Marvel trading card", "category_id": CategoryId.MARVEL, "region": Region.uk},
    {"query": "Marvel trading card", "category_id": CategoryId.MARVEL, "region": Region.us},
    {"query": "Garbage Pail Kids trading card", "category_id": CategoryId.GARBAGE_PAIL_KIDS, "region": Region.uk},
    {"query": "Garbage Pail Kids trading card", "category_id": CategoryId.GARBAGE_PAIL_KIDS, "region": Region.us},
    {"query": "Wrestling trading card", "category_id": CategoryId.WRESTLING, "region": Region.uk},
    {"query": "Wrestling trading card", "category_id": CategoryId.WRESTLING, "region": Region.us},
    {"query": "Yu-Gi-Oh trading card", "category_id": CategoryId.YU_GI_OH, "region": Region.uk},
    {"query": "Yu-Gi-Oh trading card", "category_id": CategoryId.YU_GI_OH, "region": Region.us},
    {"query": "Lorcana trading card", "category_id": CategoryId.LORCANA, "region": Region.uk},
    {"query": "Lorcana trading card", "category_id": CategoryId.LORCANA, "region": Region.us},
]

class EbayCronJob:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        
    async def run_scrape_batch(self):
        """Run all scrapes in SCRAPE_LIST sequentially"""
        if self.is_running:
            logger.warning("Scrape batch already running, skipping this execution")
            return
            
        self.is_running = True
        logger.info(f"Starting scheduled scrape batch with {len(SCRAPE_LIST)} items")
        
        try:
            for i, scrape_config in enumerate(SCRAPE_LIST, 1):
                query = scrape_config["query"]
                category_id = scrape_config["category_id"]
                region = scrape_config["region"]
                
                logger.info(f"[{i}/{len(SCRAPE_LIST)}] Scraping: {query} | {category_id.name} | {region.value}")
                
                try:
                    result = await ebay_scrape_handler(
                        region=region,
                        category_id=category_id,
                        query=query,
                        max_pages=10,
                        page_retries=3,
                    )
                    
                    total = result.get("total", 0)
                    logger.info(f"   ✅ Completed: {total} records scraped")
                    
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed scraping {query} ({region.value}): {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Critical error in scrape batch: {e}")
        finally:
            self.is_running = False
            logger.info("Scrape batch completed")
    
    def start_scheduler(self):
        """Start the cron scheduler"""
        # Run daily at 3 AM server time
        self.scheduler.add_job(
            self.run_scrape_batch,
            CronTrigger(hour=3, minute=0),
            id='ebay_scrape_job',
            name='Daily eBay Scrape',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info(f"eBay scrape scheduler started - runs daily at {CronTrigger(hour=16, minute=5)} server time")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("eBay scrape scheduler stopped")

# Global scheduler instance
ebay_scheduler = EbayCronJob()

def start_ebay_cronjob():
    """Initialize and start the eBay cronjob scheduler"""
    ebay_scheduler.start_scheduler()

def stop_ebay_cronjob():
    """Stop the eBay cronjob scheduler"""
    ebay_scheduler.stop_scheduler()

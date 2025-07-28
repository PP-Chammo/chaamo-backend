from fastapi import APIRouter
from . import ebay_scrape

router = APIRouter()
router.include_router(ebay_scrape.router)

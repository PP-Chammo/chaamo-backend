from fastapi import APIRouter
from . import bulk_scrape

router = APIRouter()
router.include_router(bulk_scrape.router)

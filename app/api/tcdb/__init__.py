from fastapi import APIRouter
from . import tcdb_scrape

router = APIRouter()
router.include_router(tcdb_scrape.router) 

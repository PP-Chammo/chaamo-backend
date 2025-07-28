from typing import Optional
from fastapi import APIRouter, Query

from src.handlers.ebay_search import ebay_search_handler
from src.models.ebay import Region

router = APIRouter()

@router.get("/ebay_search",  summary="Scrape eBay sold posts")
async def search_endpoint(
    query: str = Query("2024 Topps Thiery Henry", description="Enter keyword. For example: '2023 Topps Merlin Lamine Yamal'"),
    region: Region = Query(Region.uk, description="Choose a region from 'us' or 'uk'"),
    master_card_id: Optional[str] = Query(None, description="(Optional) can get id from master_cards table")
):
    result = await ebay_search_handler(query, region, master_card_id)
    return { "total": len(result), "result": result }

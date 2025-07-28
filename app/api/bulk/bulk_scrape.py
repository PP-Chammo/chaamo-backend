from fastapi import APIRouter, HTTPException, Query
from app.handlers.bulk_scrape_handler import discover_bulk
from app.models.bulk import BulkResponse
from app.models.tcdb import ScrapeTarget

router = APIRouter()

@router.get("/sets", response_model=BulkResponse, summary="Discover card sets by category/brand")
async def discover_sets_endpoint(
    category: ScrapeTarget = Query(..., description="Select a category to bulk scrape all sets and cards")
):
    try:
        result = await discover_bulk(category.value)
        return {
                "status": "success" if len(result) > 0 else "failed",
                "data": result
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException, Query

from app.models.ebay import EbaySoldsResponse, Region, SearchPage
from app.handlers.ebay_scrape_handler import discover_solds

router = APIRouter()

@router.get("/solds", response_model=EbaySoldsResponse, summary="Discover sold listings from eBay")
async def discover_posts_endpoint(
    query: str = Query("", description="Enter the search query for eBay. Example: '2023 Topps Merlin Lamine Yamal'"),
    region: Region = Query(Region.uk, description="Choose a region from 'us' or 'uk'"),
    page: SearchPage = Query(SearchPage.one, description="Choose a page from search results")
):
    try:
        solds = await discover_solds(query, region.value, page.value)
        return EbaySoldsResponse(solds=solds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

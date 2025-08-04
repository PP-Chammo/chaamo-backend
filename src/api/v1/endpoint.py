import asyncio
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from src.handlers.ebay_search import ebay_search_handler
from src.models.ebay import Region
from src.utils.playwright import playwright_get_content

router = APIRouter()

@router.get("/ebay_search",  summary="Scrape eBay sold posts")
async def search_endpoint(
    query: str = Query("2024 Topps Thiery Henry", description="Enter keyword. For example: '2023 Topps Merlin Lamine Yamal'"),
    region: Region = Query(Region.uk, description="Choose a region from 'us' or 'uk'"),
    master_card_id: Optional[str] = Query(None, description="(Optional) can get id from master_cards table")
):
    try:
        # Add timeout to prevent hanging
        result = await asyncio.wait_for(
            ebay_search_handler(query, region, master_card_id),
            timeout=120.0  # 2 minutes timeout
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408, 
            detail="Request timeout - scraping took too long. Please try again."
        )
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/test_playwright", summary="Test Playwright functionality")
async def test_playwright():
    """Test endpoint to verify Playwright is working"""
    try:
        html = await playwright_get_content("https://httpbin.org/html")
        if html and len(html) > 0:
            return {
                "status": "success",
                "message": "Playwright is working correctly",
                "html_length": len(html)
            }
        else:
            return {
                "status": "error",
                "message": "Playwright returned empty content"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Playwright test failed: {str(e)}"
        }

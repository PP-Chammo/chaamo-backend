from fastapi import APIRouter, HTTPException, Query
from app.handlers.tcdb_scrape_handler import discover_sets, discover_cards
from app.models.tcdb import CardsResponse, ScrapeTarget, SetsResponse

router = APIRouter()

@router.get("/sets", response_model=SetsResponse, summary="Discover card sets by category")
async def discover_sets_endpoint(
    category: ScrapeTarget = Query(..., description="Select a category to discover sets for")
):
    try:
        sets = await discover_sets(category)
        return SetsResponse(sets=sets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cards", response_model=CardsResponse, summary="Get cards by set ID")
async def discover_cards_endpoint(
    set_id: int = Query(..., description="Set ID to fetch cards for")
):
    try:
        cards = await discover_cards(set_id)
        return CardsResponse(cards=cards)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

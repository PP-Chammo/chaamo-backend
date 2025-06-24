from fastapi import APIRouter, HTTPException, Query
from httpx import HTTPStatusError, RequestError
from bs4 import BeautifulSoup

from app.models.card import ScrapeResponse, Region, SearchPage
from app.utils.fetcher import fetch_html
from app.handlers.scrape_cards_handler import process_scrape_cards
from app.utils.scrape_helpers import save_debug_html
from app.utils.supabase_helpers import save_to_supabase

router = APIRouter()

@router.get("/scrape_cards", response_model=ScrapeResponse, summary="Scrape eBay sold listings")
async def scrape_cards_endpoint(
    query: str = Query(
        "",
        description="Enter the name of the card you want to search for on eBay. For example: '2023 Topps Merlin Lamine Yamal'"
    ),
    region: Region = Query(
        Region.uk,
        description="Choose a region from 'us' or 'uk'"
    ),
    page: SearchPage = Query(
        SearchPage.one,
        description="Choose a page from search"
    )
):
    """
    Endpoint to scrape data of sold cards from eBay, save it to Supabase,
    and return the results.
    """
    try:
        html_text = await fetch_html(query=query, region=region.value, page=page.value)
    except HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error fetching data from eBay: {e.response.reason_phrase}"
        )
    except RequestError:
        raise HTTPException(status_code=504, detail="Timeout or network error connecting to eBay.")

    if not html_text:
        raise HTTPException(status_code=500, detail="Failed to retrieve HTML content.")

    try:
        grouped_cards = await process_scrape_cards(html_text)

        if not grouped_cards:
            soup = BeautifulSoup(html_text, 'lxml')
            items = soup.select('li.s-item, li.s-card')
            save_debug_html(str(items))
            raise HTTPException(status_code=500, detail="Failed to parse results.")

        await save_to_supabase(grouped_cards)

        return ScrapeResponse(result=grouped_cards)
    except Exception as e:
        soup = BeautifulSoup(html_text, 'lxml')
        items = soup.select('li.s-item, li.s-card')
        save_debug_html(str(items))
        print(f"An unexpected error occurred during processing: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while parsing the data.")

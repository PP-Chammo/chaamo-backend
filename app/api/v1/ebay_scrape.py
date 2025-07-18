from fastapi import APIRouter, HTTPException, Query
from httpx import HTTPStatusError, RequestError
from bs4 import BeautifulSoup

from app.models.card import ScrapeResponse, Region, SearchPage
from app.utils.ebay_fetcher import fetch_html
from app.handlers.ebay_scrape_handler import process_ebay_scrape
from app.utils.ebay_scrape_helpers import save_debug_html
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
    print("\n🚀 Starting ebay_scrape endpoint")
    print(f"   📝 Query: '{query}'")
    print(f"   🌍 Region: {region.value}")
    print(f"   📄 Page: {page.value}")
    
    if not query or not query.strip():
        print("❌ ERROR: Empty or invalid query parameter")
        raise HTTPException(status_code=400, detail="Query parameter is required and cannot be empty")
    
    try:
        print("🌐 Fetching HTML from eBay...")
        ebay_domain = "com" if region.value == "us" else "co.uk"
        url = f"https://www.ebay.{ebay_domain}/sch/i.html"
        params = {
            "_from": "R40",
            "_nkw": query,
            "_sacat": "0",
            "rt": "nc",
            "LH_Sold": "1",
            "LH_Complete": "1",
            "Country/Region of Manufacture": "United States" if region.value == "us" else "United Kingdom",
            "_pgn": page.value
        }
        html_text = await fetch_html(url=url, params=params)
        print(f"   ✅ HTML fetched successfully ({len(html_text)} characters)")
        
    except HTTPStatusError as e:
        print(f"❌ HTTP ERROR: {e.response.status_code} - {e.response.reason_phrase}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error fetching data from eBay: {e.response.reason_phrase}"
        )
    except RequestError as e:
        print(f"❌ REQUEST ERROR: {e}")
        raise HTTPException(status_code=504, detail="Timeout or network error connecting to eBay.")

    if not html_text:
        print("❌ ERROR: Empty HTML content received")
        raise HTTPException(status_code=500, detail="Failed to retrieve HTML content.")

    try:
        print("🔧 Processing scraped data...")
        grouped_cards = await process_ebay_scrape(html_text)

        if not grouped_cards:
            print("❌ ERROR: No grouped cards returned from processing")
            soup = BeautifulSoup(html_text, 'lxml')
            items = soup.select('li.s-item, li.s-card')
            print(f"   🔍 Found {len(items)} potential items in HTML")
            save_debug_html(str(items))
            raise HTTPException(status_code=500, detail="Failed to parse results.")

        print(f"✅ Successfully processed {len(grouped_cards)} card groups")
        
        # Save to Supabase
        print("💾 Saving to Supabase...")
        await save_to_supabase(grouped_cards)
        print("✅ Supabase save operation completed")

        print(f"🎉 Scrape operation successful - returning {len(grouped_cards)} card groups")
        return ScrapeResponse(result=grouped_cards)
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR during processing: {e}")
        soup = BeautifulSoup(html_text, 'lxml')
        items = soup.select('li.s-item, li.s-card')
        print(f"   🔍 Found {len(items)} potential items in HTML for debugging")
        save_debug_html(str(items))
        print("   📝 Full error details:")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An internal error occurred while parsing the data.")

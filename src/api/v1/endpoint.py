import asyncio
from typing import Optional
from urllib.parse import urlencode
from fastapi import APIRouter, Query, HTTPException, Request
from starlette.responses import RedirectResponse

from src.handlers.ebay_scrape import ebay_scrape_handler
from src.models.ebay import Region
from src.models.category import CategoryId
from src.utils.paypal import create_order, capture_order

router = APIRouter()

@router.get("/ebay_scrape", summary="Scrape eBay sold posts")
async def search_endpoint(
    region: Region = Query(Region.uk, description="Choose a region from 'us' or 'uk'"),
    category_id: Optional[CategoryId] = Query(None, description="Category filter - REQUIRED when using 'query'. Choose from available categories."),
    query: Optional[str] = Query(None, description="Search keyword (e.g., '2023 Topps Merlin Lamine Yamal'). Must be used WITH category_id."),
    user_card_id: Optional[str] = Query(None, description="Alternative to query+category_id: Use existing user card ID from user_cards table."),
):
    try:
        # Add timeout to prevent hanging
        result = await asyncio.wait_for(
            ebay_scrape_handler(region, category_id, query, user_card_id),
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
 

# -----------------------------
# PayPal Checkout Endpoints
# Keep paths as /api/v1/paypal/...
# -----------------------------

def _base_url(request: Request) -> str:
    # e.g. https://chaamo-backend.fly.dev
    return str(request.base_url).rstrip("/")


@router.get("/paypal/checkout", summary="Start PayPal checkout")
async def paypal_checkout(
    request: Request,
    amount: str = Query("1.00", description="Payment amount as string, e.g. '1.00'"),
    currency: str = Query("USD", description="Currency code, e.g. 'USD'"),
    redirect: str = Query(..., description="App redirect deep link to return to after payment"),
):
    try:
        base = _base_url(request)
        return_url = f"{base}/api/v1/paypal/return?{urlencode({'redirect': redirect})}"
        cancel_url = f"{base}/api/v1/paypal/cancel?{urlencode({'redirect': redirect})}"

        order = await create_order(amount=amount, currency=currency, return_url=return_url, cancel_url=cancel_url)
        links = order.get("links", [])
        approval = next((l for l in links if l.get("rel") in ("approve", "payer-action")), None)
        if not approval or not approval.get("href"):
            raise HTTPException(status_code=502, detail="Failed to create PayPal approval link")
        return RedirectResponse(url=approval["href"], status_code=302)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Checkout error: {e}")


@router.get("/paypal/return", summary="PayPal return URL")
async def paypal_return(
    redirect: str = Query(..., description="Original app redirect deep link"),
    token: Optional[str] = Query(None, description="PayPal order ID token"),
    PayerID: Optional[str] = Query(None, description="PayPal payer ID (may be absent)"),
):
    if not token:
        params = urlencode({"status": "error"})
        return RedirectResponse(url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302)
    try:
        await capture_order(order_id=token)
        params = urlencode({"status": "success", "orderId": token})
        return RedirectResponse(url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302)
    except Exception:
        params = urlencode({"status": "error", "orderId": token})
        return RedirectResponse(url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302)


@router.get("/paypal/cancel", summary="PayPal cancel URL")
async def paypal_cancel(
    redirect: str = Query(..., description="Original app redirect deep link"),
    token: Optional[str] = Query(None, description="PayPal order ID token"),
):
    params = urlencode({"status": "cancel", "orderId": token or ""})
    return RedirectResponse(url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302)

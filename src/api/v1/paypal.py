from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import RedirectResponse

from src.utils.paypal import create_order, capture_order

router = APIRouter(prefix="/paypal", tags=["PayPal"])


def _base_url(request: Request) -> str:
    # e.g. https://chaamo-backend.fly.dev
    return str(request.base_url).rstrip("/")


@router.get("/checkout", summary="Start PayPal checkout")
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
        # Find approval link
        links = order.get("links", [])
        approval = next((l for l in links if l.get("rel") in ("approve", "payer-action")), None)
        if not approval or not approval.get("href"):
            raise HTTPException(status_code=502, detail="Failed to create PayPal approval link")
        return RedirectResponse(url=approval["href"], status_code=302)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Checkout error: {e}")


@router.get("/return", summary="PayPal return URL")
async def paypal_return(
    redirect: str = Query(..., description="Original app redirect deep link"),
    token: Optional[str] = Query(None, description="PayPal order ID token"),
    PayerID: Optional[str] = Query(None, description="PayPal payer ID (may be absent)"),
):
    # token is the order ID for v2 checkout
    if not token:
        # Missing order token, treat as failure
        params = urlencode({"status": "error"})
        return RedirectResponse(url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302)
    try:
        capture = await capture_order(order_id=token)
        # If capture is successful, redirect back to app with success
        params = urlencode({"status": "success", "orderId": token})
        return RedirectResponse(url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302)
    except Exception:
        params = urlencode({"status": "error", "orderId": token})
        return RedirectResponse(url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302)


@router.get("/cancel", summary="PayPal cancel URL")
async def paypal_cancel(
    redirect: str = Query(..., description="Original app redirect deep link"),
    token: Optional[str] = Query(None, description="PayPal order ID token"),
):
    params = urlencode({"status": "cancel", "orderId": token or ""})
    return RedirectResponse(url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302)

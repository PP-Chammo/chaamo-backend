import asyncio
from typing import Optional
from urllib.parse import urlencode
from fastapi import APIRouter, Query, HTTPException, Request, Depends
from starlette.responses import RedirectResponse

from src.ebay_scraper import worker_manager
from src.models.ebay import Region
from src.models.category import CategoryId
from src.models.api import (
    WorkerListResponse,
    WorkerTaskResponse,
    ScrapeRequest,
    ScrapeStartResponse,
    ErrorResponse,
    PayPalCheckoutRequest,
)
from src.utils.paypal import create_order, capture_order
from src.utils.logger import (
    api_logger,
    log_api_request,
    log_worker_task,
    log_error_with_context,
)

router = APIRouter()


# Dependency injection
def get_worker_manager():
    """Dependency injection for worker manager."""
    return worker_manager


@router.get(
    "/workers", summary="Get all worker tasks", response_model=WorkerListResponse
)
async def get_all_workers(manager=Depends(get_worker_manager)):
    """Get status of all eBay scraping workers."""
    log_api_request(api_logger, "GET", "/workers")

    try:
        tasks = manager.get_all_tasks()
        manager.cleanup_old_tasks()  # Clean up old tasks

        api_logger.info(f"📋 Retrieved {len(tasks)} worker tasks")

        return WorkerListResponse(
            total_tasks=len(tasks),
            tasks=[WorkerTaskResponse(**task.to_dict()) for task in tasks],
        )
    except Exception as e:
        log_error_with_context(api_logger, e, "fetching workers")
        raise HTTPException(status_code=500, detail=f"Error fetching workers: {str(e)}")


@router.get(
    "/workers/{task_id}",
    summary="Get worker task status",
    response_model=WorkerTaskResponse,
)
async def get_worker_status(task_id: str, manager=Depends(get_worker_manager)):
    """Get status of specific eBay scraping worker by task ID."""
    log_api_request(api_logger, "GET", f"/workers/{task_id[:8]}...")

    try:
        task = manager.get_task_status(task_id)

        if not task:
            api_logger.warning(f"🔍 Task {task_id[:8]}... not found")
            raise HTTPException(status_code=404, detail="Task not found")

        api_logger.info(
            f"📋 Retrieved task {task_id[:8]}... status: {task.status.value}"
        )
        return WorkerTaskResponse(**task.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(api_logger, e, f"fetching task {task_id[:8]}...")
        raise HTTPException(status_code=500, detail=f"Error fetching task: {str(e)}")


# Legacy endpoint deprecated - use POST /ebay_scrape for new worker-based approach


@router.post(
    "/ebay_scrape",
    summary="Start eBay scraping worker",
    response_model=ScrapeStartResponse,
)
async def start_scrape_worker(
    region: Region = Query(Region.uk, description="Choose a region from 'us' or 'uk'"),
    category_id: Optional[CategoryId] = Query(
        None, description="Category filter - Required when using 'query', resolved automatically for 'user_card_id'"
    ),
    query: Optional[str] = Query(
        None, description="Search keyword (e.g., '2023 Topps Merlin Lamine Yamal')"
    ),
    user_card_id: Optional[str] = Query(
        None,
        description="Alternative to query: Use existing user card ID from user_cards table",
    ),
    max_pages: int = Query(50, description="Max pages to scrape (lower is faster)"),
    disable_proxy: bool = Query(False, description="Disable proxy usage (default: False - uses Zyte proxy)"),
    manager=Depends(get_worker_manager),
):
    """Start a background eBay scraping worker and return task ID."""

    log_api_request(
        api_logger,
        "POST",
        "/ebay_scrape",
        {
            "region": region.value,
            "category_id": category_id.value if category_id else None,
            "query": query,
            "user_card_id": user_card_id,
            "max_pages": max_pages,
            "disable_proxy": disable_proxy,
        },
    )

    # Validation: enforce proper parameter combinations
    if not query and not user_card_id:
        api_logger.warning(
            "❌ Validation failed: either 'query' + 'category_id' or 'user_card_id' must be provided"
        )
        raise HTTPException(
            status_code=400,
            detail="Either 'query' + 'category_id' or 'user_card_id' must be provided",
        )

    if query and user_card_id:
        api_logger.warning(
            "❌ Validation failed: cannot use both 'query' and 'user_card_id' simultaneously"
        )
        raise HTTPException(
            status_code=400,
            detail="Cannot use both 'query' and 'user_card_id' simultaneously. Choose one.",
        )

    # Require category_id when using query mode
    if query and not category_id:
        api_logger.warning(
            "❌ Validation failed: 'category_id' is required when using 'query'"
        )
        raise HTTPException(
            status_code=400,
            detail="'category_id' is required when using 'query'",
        )

    try:
        # Create worker task - unified approach resolves parameters internally
        task = manager.create_task(
            query=query,
            region=region,
            category_id=category_id,
            user_card_id=user_card_id,
            max_pages=max_pages,
        )
        
        # Start worker in FastAPI async context  
        try:
            # task.category_id is already an int, convert to CategoryId enum
            asyncio.create_task(
                manager._run_scrape_worker(
                    task.id,
                    region,
                    CategoryId(task.category_id),
                    task.query,
                    user_card_id,
                    max_pages,
                    None,  # master_card_id parameter
                    disable_proxy,  # Pass disable_proxy parameter
                )
            )
            api_logger.info(f"⚙️ Worker task {task.id} started successfully")
        except Exception as worker_error:
            api_logger.error(f"❌ Failed to start worker: {worker_error}")
            raise

        log_worker_task(
            api_logger,
            task.id,
            "created",
            f"query='{task.query}' region={region.value}",
        )

        return ScrapeStartResponse(
            task_id=task.id,
            status=task.status.value,
            message="eBay scraping worker started successfully",
            query=task.query,
            region=task.region,
            category_id=task.category_id,
            estimated_duration="2-5 minutes",
        )

    except Exception as e:
        api_logger.error(f"❌ Worker creation failed: {str(e)}")
        api_logger.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        api_logger.error(f"❌ Traceback: {traceback.format_exc()}")
        log_error_with_context(api_logger, e, "starting worker")
        raise HTTPException(status_code=500, detail=f"Failed to start worker: {str(e)}")


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
    redirect: str = Query(
        ..., description="App redirect deep link to return to after payment"
    ),
):
    log_api_request(
        api_logger, "GET", "/paypal/checkout", {"amount": amount, "currency": currency}
    )

    try:
        base = _base_url(request)
        return_url = f"{base}/api/v1/paypal/return?{urlencode({'redirect': redirect})}"
        cancel_url = f"{base}/api/v1/paypal/cancel?{urlencode({'redirect': redirect})}"

        api_logger.info(f"💰 Creating PayPal order: {amount} {currency}")
        order = await create_order(
            amount=amount,
            currency=currency,
            return_url=return_url,
            cancel_url=cancel_url,
        )

        links = order.get("links", [])
        approval = next(
            (l for l in links if l.get("rel") in ("approve", "payer-action")), None
        )
        if not approval or not approval.get("href"):
            api_logger.error("❌ Failed to create PayPal approval link")
            raise HTTPException(
                status_code=502, detail="Failed to create PayPal approval link"
            )

        api_logger.info(f"✅ PayPal order created, redirecting to approval")
        return RedirectResponse(url=approval["href"], status_code=302)

    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(api_logger, e, "PayPal checkout")
        raise HTTPException(status_code=500, detail=f"Checkout error: {e}")


@router.get("/paypal/return", summary="PayPal return URL")
async def paypal_return(
    redirect: str = Query(..., description="Original app redirect deep link"),
    token: Optional[str] = Query(None, description="PayPal order ID token"),
    PayerID: Optional[str] = Query(None, description="PayPal payer ID (may be absent)"),
):
    log_api_request(api_logger, "GET", "/paypal/return", {"token": token})

    if not token:
        api_logger.warning("❌ PayPal return without token")
        params = urlencode({"status": "error"})
        return RedirectResponse(
            url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
        )

    try:
        api_logger.info(f"💰 Capturing PayPal order: {token}")
        await capture_order(order_id=token)
        api_logger.info(f"✅ PayPal payment captured successfully: {token}")

        params = urlencode({"status": "success", "orderId": token})
        return RedirectResponse(
            url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
        )
    except Exception as e:
        log_error_with_context(api_logger, e, f"capturing PayPal order {token}")
        params = urlencode({"status": "error", "orderId": token})
        return RedirectResponse(
            url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
        )


@router.get("/paypal/cancel", summary="PayPal cancel URL")
async def paypal_cancel(
    redirect: str = Query(..., description="Original app redirect deep link"),
    token: Optional[str] = Query(None, description="PayPal order ID token"),
):
    log_api_request(api_logger, "GET", "/paypal/cancel", {"token": token})
    api_logger.info(f"🚫 PayPal payment cancelled: {token or 'no-token'}")

    params = urlencode({"status": "cancel", "orderId": token or ""})
    return RedirectResponse(
        url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
    )

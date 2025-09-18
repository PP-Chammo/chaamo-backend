import asyncio
from typing import Optional, List, Dict, Any
import os
from fastapi import APIRouter, Query, HTTPException, Request, Depends

from src.ebay_scraper import worker_manager as ebay_worker_manager
from src.tcdb_scraper import tcdb_scrape_handler
from src.paypal_handlers import (
    paypal_subscription_handler,
    paypal_subscription_return_handler,
    paypal_subscription_cancel_handler,
    paypal_webhook_subscription_handler,
    paypal_boost_handler,
    paypal_boost_return_handler,
    paypal_webhook_boost_handler,
    paypal_order_handler,
    paypal_order_return_handler,
    paypal_webhook_order_handler,
    paypal_cancel_handler,
)
from src.shippo_handlers import shippo_rates_handler, shippo_webhook_handler
from src.models.ebay import Region
from src.models.tcdb import BrowseDropdown
from src.models.shippo import RateResponse, TransactionPayload
from src.models.category import CategoryId, CategoryDropdown
from src.models.api import (
    WorkerListResponse,
    WorkerTaskResponse,
    ScrapeStartResponse,
)
from src.utils.logger import (
    worker_logger,
    api_logger,
    log_api_request,
    log_worker_task,
)

router = APIRouter()


# Dependency injection
def get_worker_manager():
    """Dependency injection for worker manager."""
    return ebay_worker_manager


# ===============================================================
# EBAY SCRAPER
# ===============================================================


@router.post(
    "/ebay_scrape",
    summary="Start eBay scraping worker",
    response_model=ScrapeStartResponse,
)
async def start_scrape_worker(
    region: Region = Query(Region.uk, description="Choose a region from 'us' or 'uk'"),
    category_id: Optional[CategoryDropdown] = Query(
        None,
        description="Category filter - Required when using 'query', resolved automatically for 'card_id'",
    ),
    query: Optional[str] = Query(
        None, description="Search keyword (e.g., '2023 Topps Merlin Lamine Yamal')"
    ),
    card_id: Optional[str] = Query(
        None,
        description="Alternative to query: Use existing user card ID from cards table",
    ),
    max_pages: int = Query(50, description="Max pages to scrape (lower is faster)"),
    disable_proxy: bool = Query(
        False, description="Disable proxy usage (default: False - uses Zyte proxy)"
    ),
    manager=Depends(get_worker_manager),
):
    # Convert dropdown string to CategoryId
    cat_id = CategoryDropdown.to_category_id(category_id) if category_id else None

    """Start a background eBay scraping worker and return task ID."""

    log_api_request(
        api_logger,
        "POST",
        "/ebay_scrape",
        {
            "region": region.value,
            "category_id": cat_id.value if cat_id else None,
            "query": query,
            "card_id": card_id,
            "max_pages": max_pages,
            "disable_proxy": disable_proxy,
        },
    )

    # Validation: enforce proper parameter combinations
    if not query and not card_id:
        api_logger.warning(
            "❌ Validation failed: either 'query' + 'category_id' or 'card_id' must be provided"
        )
        raise HTTPException(
            status_code=400,
            detail="Either 'query' + 'category_id' or 'card_id' must be provided",
        )

    if query and card_id:
        api_logger.warning(
            "❌ Validation failed: cannot use both 'query' and 'card_id' simultaneously"
        )
        raise HTTPException(
            status_code=400,
            detail="Cannot use both 'query' and 'card_id' simultaneously. Choose one.",
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
            category_id=cat_id,
            card_id=card_id,
            max_pages=max_pages,
        )

        # Start worker in FastAPI async context
        try:
            # task.category_id is already an int, convert to CategoryId enum
            asyncio.create_task(
                manager._run_scrape_worker(
                    task.id,
                    region,
                    CategoryId(task.category_id) if task.category_id else None,
                    task.query,
                    card_id,
                    max_pages,
                    None,  # master_card_id parameter
                    disable_proxy,  # Pass disable_proxy parameter
                )
            )
            worker_logger.info(f"[Worker task] {task.id} started successfully")
        except Exception as worker_error:
            worker_logger.error(f"❌ Failed to start worker: {worker_error}")
            raise

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
        api_logger.exception("Error starting worker: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to start worker: {str(e)}")


@router.get(
    "/ebay_scrape",
    summary="Start eBay scraping worker (GET version)",
    response_model=ScrapeStartResponse,
)
async def start_scrape_worker_get(
    region: Region = Query(Region.uk, description="Choose a region from 'us' or 'uk'"),
    category_id: Optional[CategoryId] = Query(
        None,
        description="Category filter - Required when using 'query', resolved automatically for 'card_id'",
    ),
    query: Optional[str] = Query(
        None, description="Search keyword (e.g., '2023 Topps Merlin Lamine Yamal')"
    ),
    card_id: Optional[str] = Query(
        None,
        description="Alternative to query: Use existing user card ID from cards table",
    ),
    max_pages: int = Query(50, description="Max pages to scrape (lower is faster)"),
    disable_proxy: bool = Query(
        False, description="Disable proxy usage (default: False - uses Zyte proxy)"
    ),
    manager=Depends(get_worker_manager),
):
    """GET version: Start a background eBay scraping worker and return task ID."""

    # Call the existing POST endpoint logic
    return await start_scrape_worker(
        region=region,
        category_id=category_id,
        query=query,
        card_id=card_id,
        max_pages=max_pages,
        disable_proxy=disable_proxy,
        manager=manager,
    )


# ===============================================================
# TCDB SCRAPER
# ===============================================================


@router.post(
    "/tcdb_scrape", summary="Start TCDB scraping - THIS ENDPOINT DISABLED ON PRODUCTION"
)
async def start_tcdb_scrape(
    category_id: CategoryDropdown = Query(
        CategoryDropdown.TOPPS, description="Filter by brand category"
    ),
    browse: Optional[List[BrowseDropdown]] = Query(
        None, description="Filter by browsing category from TCDB website"
    ),
    search_mode: bool = Query(
        True, description="Use search-based scraping method (default: True)"
    ),
):
    """Start a TCDB scraping task based on a brand and optional browse category."""
    env = (os.environ.get("ENV") or os.environ.get("APP_ENV") or "development").lower()
    if env == "production":
        raise HTTPException(
            status_code=403, detail="TCDB scraping is disabled in production"
        )
    cat_id = CategoryDropdown.to_category_id(category_id)
    log_api_request(
        api_logger,
        "POST",
        "/tcdb_scrape",
        {
            "category_id": cat_id.value,
            "browse": [b.value for b in browse] if browse else [],
        },
    )

    try:
        upserted_set_count, upserted_card_count = await tcdb_scrape_handler(
            cat_id, [b.value for b in browse] if browse else [], search_mode
        )
        return {
            "status": "ok",
            "message": f"TCDB scrape for brand completed. Found {upserted_set_count} cards from {upserted_card_count} sets.",
            "category_id": int(cat_id),
            "browse": [b.value for b in browse] if browse else None,
        }
    except Exception as e:
        api_logger.error(e, "starting TCDB scrape")
        raise HTTPException(
            status_code=500, detail=f"Failed to start TCDB scrape: {str(e)}"
        )


# ===============================================================
# WORKER MANAGEMENT
# ===============================================================


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
        api_logger.exception("Error fetching workers: %s", e)
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
        api_logger.exception("Error fetching task %s...: %s", task_id[:8], e)
        raise HTTPException(status_code=500, detail=f"Error fetching task: {str(e)}")


# ===============================================================
# PAYPAL ENDPOINT
# ===============================================================


@router.get("/paypal/subscription", summary="PayPal subscription plan API")
async def paypal_subscription(
    request: Request,
    user_id: str = Query(..., description="User ID"),
    plan_id: str = Query(..., description="Plan ID"),
    redirect: str = Query(
        ..., description="App redirect deep link to return to after payment"
    ),
):
    return await paypal_subscription_handler(
        request=request,
        user_id=user_id,
        plan_id=plan_id,
        redirect=redirect,
    )


@router.get(
    "/paypal/subscription/cancel", summary="PayPal cancel subscription plan API"
)
async def paypal_subscription_cancel(
    redirect: str = Query(..., description="Original app redirect deep link"),
    user_id: Optional[str] = Query(None, description="User ID"),
    plan_id: Optional[str] = Query(None, description="Plan ID"),
    paypal_subscription_id: Optional[str] = Query(
        None, description="PayPal subscription ID token"
    ),
):
    return await paypal_subscription_cancel_handler(
        redirect=redirect,
        user_id=user_id,
        plan_id=plan_id,
        paypal_subscription_id=paypal_subscription_id,
    )


@router.get(
    "/paypal/subscription/return",
    summary="PayPal subscription plan return redirect URL",
)
async def paypal_subscription_return(
    redirect: str = Query(..., description="Original app redirect deep link"),
    user_id: str = Query(..., description="User ID"),
    plan_id: str = Query(..., description="Plan ID"),
    subscription_id: Optional[str] = Query(
        None, description="PayPal subscription ID token"
    ),
):
    return await paypal_subscription_return_handler(
        redirect=redirect,
        user_id=user_id,
        plan_id=plan_id,
        subscription_id=subscription_id,
    )


@router.post("/paypal/webhook/subscription", summary="PayPal subscription plan webhook")
async def paypal_webhooks(request: Request):
    return await paypal_webhook_subscription_handler(request=request)


# Boost-post (mirrors subscription without cancel)
@router.get("/paypal/boost-post", summary="PayPal boost-post API")
async def paypal_boost_post(
    request: Request,
    user_id: str = Query(..., description="User ID (must be listing seller)"),
    listing_id: str = Query(..., description="Listing ID to boost"),
    plan_id: str = Query(..., description="Boost Plan ID"),
    redirect: str = Query(
        ..., description="App redirect deep link to return to after payment"
    ),
):
    return await paypal_boost_handler(
        request=request,
        user_id=user_id,
        listing_id=listing_id,
        plan_id=plan_id,
        redirect=redirect,
    )


@router.get(
    "/paypal/boost-post/return",
    summary="PayPal boost-post return redirect URL",
)
async def paypal_boost_return(
    redirect: str = Query(..., description="Original app redirect deep link"),
    user_id: str = Query(..., description="User ID"),
    listing_id: str = Query(..., description="Listing ID"),
    plan_id: str = Query(..., description="Boost Plan ID"),
    subscription_id: Optional[str] = Query(
        None, description="PayPal subscription ID token"
    ),
):
    return await paypal_boost_return_handler(
        redirect=redirect,
        user_id=user_id,
        listing_id=listing_id,
        plan_id=plan_id,
        subscription_id=subscription_id,
    )


@router.post("/paypal/webhook/boost-post", summary="PayPal boost-post webhook")
async def paypal_webhook_boost(request: Request):
    return await paypal_webhook_boost_handler(request=request)


@router.post("/paypal/order", summary="PayPal order API")
async def create_paypal_order(
    request: Request, payload: TransactionPayload
) -> Dict[str, Any]:
    return await paypal_order_handler(request=request, payload=payload)


@router.get("/paypal/order/return", summary="PayPal order return redirect URL")
async def paypal_return(
    redirect: str = Query(..., description="Original app redirect deep link"),
    token: Optional[str] = Query(None, description="PayPal order ID token"),
    PayerID: Optional[str] = Query(None, description="PayPal payer ID (may be absent)"),
):
    return await paypal_order_return_handler(
        redirect=redirect,
        token=token,
        PayerID=PayerID,
    )


@router.post("/paypal/webhook/order", summary="PayPal order webhook")
async def paypal_webhook_order(request: Request):
    return await paypal_webhook_order_handler(request=request)


@router.get(
    "/paypal/cancel", summary="PayPal cancel redirect URL for all types of PayPal APIs"
)
async def paypal_cancel(
    redirect: str = Query(..., description="Original app redirect deep link"),
    token: Optional[str] = Query(None, description="PayPal order ID token"),
):
    return await paypal_cancel_handler(
        redirect=redirect,
        token=token,
    )


# ===============================================================
# SHIPPO ENDPOINT
# ===============================================================
@router.get(
    "/shippo/rates",
    response_model=RateResponse,
    summary="Shippo Get shipping rates API",
)
async def get_shipping_rates(
    seller_id: str,
    buyer_id: str,
    insurance: Optional[bool] = False,
    insurance_currency: Optional[str] = None,
    insurance_amount: Optional[float] = 0.0,
):
    return await shippo_rates_handler(
        seller_id=seller_id,
        buyer_id=buyer_id,
        insurance=insurance,
        insurance_currency=insurance_currency,
        insurance_amount=insurance_amount,
    )


@router.post("/shippo/webhook", summary="Shippo webhook")
async def shippo_webhook(request: Request):
    return await shippo_webhook_handler(request=request)

import os
import asyncio
import shippo
import re
from shippo.models import components
from typing import Optional, List
from urllib.parse import urlencode
from dotenv import load_dotenv
from fastapi import APIRouter, Query, HTTPException, Request, Depends, status
import httpx
from starlette.responses import RedirectResponse

from src.ebay_scraper import worker_manager as ebay_worker_manager
from src.tcdb_scraper import tcdb_scrape_handler
from src.models.ebay import Region
from src.models.tcdb import BrowseDropdown
from src.models.shippo import RateOption, RateRequest
from src.models.category import CategoryId, CategoryDropdown
from src.models.api import (
    WorkerListResponse,
    WorkerTaskResponse,
    ScrapeStartResponse,
)
from src.utils.supabase import supabase
from src.utils.paypal import create_order, capture_order
from src.utils.logger import (
    api_logger,
    log_api_request,
    log_worker_task,
    log_error_with_context,
)

router = APIRouter()

load_dotenv()

SHIPPO_API_KEY = os.environ.get("SHIPPO_API_KEY")
SHIPPO_ALLOWED_PROVIDERS = os.environ.get("SHIPPO_ALLOWED_PROVIDERS")


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
        description="Category filter - Required when using 'query', resolved automatically for 'user_card_id'",
    ),
    query: Optional[str] = Query(
        None, description="Search keyword (e.g., '2023 Topps Merlin Lamine Yamal')"
    ),
    user_card_id: Optional[str] = Query(
        None,
        description="Alternative to query: Use existing user card ID from user_cards table",
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
            category_id=cat_id,
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


@router.get(
    "/ebay_scrape",
    summary="Start eBay scraping worker (GET version)",
    response_model=ScrapeStartResponse,
)
async def start_scrape_worker_get(
    region: Region = Query(Region.uk, description="Choose a region from 'us' or 'uk'"),
    category_id: Optional[CategoryId] = Query(
        None,
        description="Category filter - Required when using 'query', resolved automatically for 'user_card_id'",
    ),
    query: Optional[str] = Query(
        None, description="Search keyword (e.g., '2023 Topps Merlin Lamine Yamal')"
    ),
    user_card_id: Optional[str] = Query(
        None,
        description="Alternative to query: Use existing user card ID from user_cards table",
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
        user_card_id=user_card_id,
        max_pages=max_pages,
        disable_proxy=disable_proxy,
        manager=manager,
    )


# ===============================================================
# TCDB SCRAPER
# ===============================================================


@router.post("/tcdb_scrape", summary="Start TCDB scraping")
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
        log_error_with_context(api_logger, e, "starting TCDB scrape")
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


# ===============================================================
# PAYPAL CHECKOUT
# ===============================================================


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


# ===============================================================
# SHIPPO API
# ===============================================================

shippo_sdk = shippo.Shippo(api_key_header=SHIPPO_API_KEY)


shippo_sdk = shippo.Shippo(api_key_header=SHIPPO_API_KEY)


@router.get("/shippo/rates", response_model=List[RateOption])
async def get_shipping_rates(
    listing_id: str = Query(..., description="The eBay listing ID"),
    seller_id: str = Query(..., description="The seller's user ID"),
    buyer_id: str = Query(..., description="The buyer's user ID"),
    insurance: Optional[bool] = Query(
        False, description="Whether to include insurance"
    ),
    insurance_amount: Optional[float] = Query(0.0, description="Insurance amount"),
    insurance_currency: Optional[str] = Query(
        "GBP", description="Currency for insurance"
    ),
):
    if not SHIPPO_API_KEY:
        raise HTTPException(status_code=500, detail="SHIPPO_API_KEY not set")

    api_logger.info(f"SHIPPO_ALLOWED_PROVIDERS: {SHIPPO_ALLOWED_PROVIDERS}")

    # Fetch seller & buyer data
    seller = (
        supabase.table("profiles")
        .select("username, phone_number")
        .eq("id", seller_id)
        .execute()
        .data[0]
    )
    seller_addr = (
        supabase.table("user_addresses")
        .select(
            "address_line_1, address_line_2, city, state_province, postal_code, country"
        )
        .eq("user_id", seller_id)
        .execute()
        .data[0]
    )

    buyer = (
        supabase.table("profiles")
        .select("username, phone_number")
        .eq("id", buyer_id)
        .execute()
        .data[0]
    )
    buyer_addr = (
        supabase.table("user_addresses")
        .select(
            "address_line_1, address_line_2, city, state_province, postal_code, country"
        )
        .eq("user_id", buyer_id)
        .execute()
        .data[0]
    )

    # -------------------------
    # Build Shippo address objects
    # -------------------------
    def build_address(user, addr):
        street = addr["address_line_1"]
        if addr.get("address_line_2"):
            street = f"{street}, {addr['address_line_2']}"
        return components.AddressCreateRequest(
            name=user["username"],
            street1=street,
            city=addr["city"],
            state=addr.get("state_province") or "",  # keep empty for UK
            zip=addr["postal_code"],
            country=addr["country"].upper(),
        )

    seller_address = build_address(seller, seller_addr)
    buyer_address = build_address(buyer, buyer_addr)

    # -------------------------
    # Parcel
    # -------------------------
    parcel = components.ParcelCreateRequest(
        length="20",
        width="15",
        height="2",
        distance_unit=components.DistanceUnitEnum.CM,
        weight="0.05",
        mass_unit=components.WeightUnitEnum.KG,
    )

    # -------------------------
    # Insurance
    # -------------------------
    shippo_insurance_amount = None
    shippo_insurance_currency = None
    if insurance and insurance_amount and insurance_amount > 0:
        shippo_insurance_amount = f"{float(insurance_amount):.2f}"
        shippo_insurance_currency = insurance_currency

    # -------------------------
    # Create shipment
    # -------------------------
    shipment_request = components.ShipmentCreateRequest(
        address_from=seller_address,
        address_to=buyer_address,
        parcels=[parcel],
        async_=False,
        insurance_amount=shippo_insurance_amount,
        insurance_currency=shippo_insurance_currency,
    )

    try:
        shipment = shippo_sdk.shipments.create(shipment_request)
    except Exception as e:
        api_logger.error(f"Error creating shipment: {str(e)}")
        log_error_with_context(api_logger, e, "creating shipment")
        raise HTTPException(
            status_code=500, detail=f"Error creating shipment: {str(e)}"
        )

    # Check if shipment has rates
    if not shipment.rates:
        api_logger.error(f"No rates found in shipment: {shipment}")
        raise HTTPException(status_code=502, detail="No shipping rates available")

    raw_rates = shipment.rates

    # -------------------------
    # Parse rates
    # -------------------------
    parsed: List[RateOption] = []
    for r in raw_rates:
        try:
            print(r)
            amount = float(r.amount)
            parsed.append(
                RateOption(
                    id=r.object_id,
                    value=str(amount),
                    label=f"{r.provider} - {r.servicelevel.name}",
                    courier=r.provider,
                    service=r.servicelevel.name,
                    amount=amount,
                    currency=r.currency,
                    estimated_days=r.estimated_days,
                    shippo_rate_id=r.object_id,
                )
            )
        except Exception as ex:
            api_logger.debug("Skipping malformed rate: %s; error: %s", r, ex)

    if SHIPPO_ALLOWED_PROVIDERS:
        parsed = [p for p in parsed if p.courier in SHIPPO_ALLOWED_PROVIDERS]

    parsed.sort(key=lambda x: x.amount)
    if not parsed:
        raise HTTPException(status_code=502, detail="No shipping rates available")

    return parsed


@router.post("/shippo/transactions", response_model=List[RateOption])
async def user_submit_order(listing_id, buyer_id, selected_rate_id):

    vw_chaamo_cards_response = supabase.table("vw_chaamo").select("*").eq("id", listing_id).execute()
    vw_chaamo_card = vw_chaamo_cards_response.data[0]

    order_payload = {
        "listing_id": vw_chaamo_card["id"],
        "user_card_id": vw_chaamo_card["user_card_id"],
        "seller_id": vw_chaamo_card["seller_id"],
        "buyer_id": buyer_id,
        "final_price": "",
        "shipping_fee": "",
        "insurance_fee": "",
        "platform_fee": "",
        "seller_earnings": "",
        "status": "",
        "shipping_address": "", # json object
        "rate_id": selected_rate_id,
        "status": "pending_payment"
    }

    print(order_payload)

    return order_payload

    # Save the order to the database first
    # order =  supabase.table("orders").insert(order_payload).execute()

    # # Create Shippo Transaction
    # transaction = create_shippo_transaction(
    #     order["id"], selected_rate_id, user_email="user@example.com"
    # )

    # # Save transaction_id & payment_url to DB
    # supabase.table("payments").insert(
    #     {
    #         "order_id": order["id"],
    #         "transaction_id": transaction.object_id,
    #         "payment_url": transaction.payment_url,
    #         "status": transaction.status,
    #     }
    # ).execute()

    # # Redirect user to PayPal
    # return transaction.payment_url

from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import RedirectResponse

from src.db.payment import (
    create_subscription_record,
    get_subscription_record,
    update_subscription_status,
)
from src.utils.logger import api_logger, log_api_request, log_error_with_context
from src.utils.paypal import create_subscription, get_subscription_details

router = APIRouter()


def _base_url(request: Request) -> str:
    # e.g. https://chaamo-backend.fly.dev
    return str(request.base_url).rstrip("/")


# -----------------------------
# PayPal Subscription Endpoints
# -----------------------------

@router.get("/subscription/checkout", summary="Start PayPal subscription")
async def paypal_subscription_checkout(
    request: Request,
    paypal_plan_id: str = Query(..., description="PayPal plan ID"),
    membership_plan_id: str = Query(..., description="Membership Plan ID"),
    user_id: str = Query(..., description="User ID for subscription"),
    subscriber_name: str = Query("Subscriber", description="Subscriber name"),
    subscriber_email: str = Query(..., description="Subscriber email"),
    redirect: str = Query(..., description="App redirect deep link"),
):
    """Start PayPal subscription checkout."""
    log_api_request(
        api_logger,
        "GET",
        "/paypal/subscription/checkout",
        {"plan_id": membership_plan_id, "user_id": user_id},
    )

    try:
        base = _base_url(request)
        return_url = f"{base}/api/v1/paypal/subscription/return?{urlencode({'redirect': redirect, 'user_id': user_id})}"
        cancel_url = f"{base}/api/v1/paypal/subscription/cancel?{urlencode({'redirect': redirect})}"

        subscription_record = get_subscription_record(
            user_id=user_id, plan_id=membership_plan_id
        )

        if subscription_record:
            subscription = await get_subscription_details(
                subscription_record["paypal_subscription_id"]
            )
            api_logger.info(f"💰 PayPal subscription found: {subscription}")
        else:
            api_logger.info("💰 No PayPal subscription found, creating new subscription")
            api_logger.info(
                f"💰 Creating PayPal subscription for plan: {membership_plan_id}"
            )

            subscription = await create_subscription(
                paypal_plan_id=paypal_plan_id,
                return_url=return_url,
                cancel_url=cancel_url,
                subscriber_name=subscriber_name,
                subscriber_email=subscriber_email,
                user_id=user_id,
            )
            api_logger.info(f"💰 PayPal subscription created: {subscription}")

            paypal_subscription_id = subscription["id"]
            try:
                await create_subscription_record(
                    user_id=user_id,
                    paypal_subscription_id=paypal_subscription_id,
                    membership_plan_id=membership_plan_id,
                    status="pending",
                )
                api_logger.info(f"💾 Created subscription record for user: {user_id}")
            except Exception as db_error:
                api_logger.error(f"❌ Failed to create subscription record: {db_error}")

        # Get approval link
        links = subscription.get("links", [])
        approval = next((link for link in links if link.get("rel") == "approve"), None)

        if not approval or not approval.get("href"):
            api_logger.error("❌ Failed to create PayPal subscription approval link")
            raise HTTPException(
                status_code=502, detail="Failed to create subscription approval link"
            )

        api_logger.info(f"✅ PayPal subscription created: {subscription.get('id')}")

        return RedirectResponse(url=approval["href"], status_code=302)

    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(api_logger, e, "PayPal subscription checkout")
        raise HTTPException(status_code=500, detail=f"Subscription checkout error: {e}")


@router.get("/subscription/return", summary="PayPal subscription return URL")
async def paypal_subscription_return(
    redirect: str = Query(..., description="Original app redirect deep link"),
    subscription_id: Optional[str] = Query(None, description="PayPal subscription ID"),
    user_id: Optional[str] = Query(None, description="User ID"),
):
    """Handle successful PayPal subscription approval."""
    log_api_request(
        api_logger,
        "GET",
        "/paypal/subscription/return",
        {"subscription_id": subscription_id, "user_id": user_id},
    )

    if not subscription_id:
        api_logger.warning("❌ PayPal subscription return without subscription_id")
        params = urlencode({"status": "error"})
        return RedirectResponse(
            url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
        )

    try:
        # Activate the subscription
        api_logger.info(f"💰 Activating PayPal subscription: {subscription_id}")

        # Get subscription details to verify status
        details = await get_subscription_details(subscription_id)
        status = details.get("status")

        api_logger.info(
            f"✅ PayPal subscription activated: {subscription_id}, status: {status}"
        )

        api_logger.info("details: ", details)

        # Update subscription status in database
        if user_id:
            start_date = details.get("start_time")
            end_date = details.get("billing_info", {}).get("next_billing_time")

            api_logger.info("details: innn user_id ", details)
            api_logger.info(f"Start date: {start_date}, End date: {end_date}")
            update_subscription_status(
                paypal_subscription_id=subscription_id,
                status="active",
                start_date=start_date,
                end_date=end_date,
            )

            api_logger.info(f"💾 Updated subscription to active for user: {user_id}")

        params = urlencode(
            {
                "status": "success",
                "subscription_id": subscription_id,
                "subscription_status": status,
            }
        )
        return RedirectResponse(
            url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
        )

    except Exception as e:
        log_error_with_context(
            api_logger, e, f"activating PayPal subscription {subscription_id}"
        )
        params = urlencode(
            {"status": "error", "subscription_id": subscription_id or ""}
        )
        return RedirectResponse(
            url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
        )


@router.get("/paypal/subscription/cancel", summary="PayPal subscription cancel URL")
async def paypal_subscription_cancel(
    redirect: str = Query(..., description="Original app redirect deep link"),
    subscription_id: Optional[str] = Query(None, description="PayPal subscription ID"),
):
    """Handle PayPal subscription cancellation."""
    log_api_request(
        api_logger,
        "GET",
        "/paypal/subscription/cancel",
        {"subscription_id": subscription_id},
    )
    api_logger.info(
        f"🚫 PayPal subscription cancelled: {subscription_id or 'no-subscription-id'}"
    )

    params = urlencode({"status": "cancel", "subscription_id": subscription_id or ""})
    return RedirectResponse(
        url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
    )

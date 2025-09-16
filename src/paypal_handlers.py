import httpx
import json
from typing import Optional
from shippo.models import components
from urllib.parse import urlencode
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
from starlette.responses import RedirectResponse
from decimal import Decimal, InvalidOperation

from src.utils.shippo import shippo_sdk
from src.models.shippo import TransactionPayload
from src.utils.supabase import (
    supabase_get_subscription,
    supabase_get_subscriptions,
    supabase_get_order,
    supabase_get_orders,
    supabase_get_payment,
    supabase_get_payments,
    supabase_get_membership_plan,
    supabase_get_boost_plan,
    supabase_get_boost_listing,
    supabase_get_boost_listings,
    supabase_get_listing,
    supabase_get_listing_card,
    supabase_get_profile,
    supabase_get_user_address,
    supabase_get_card,
    supabase_mutate_payment,
    supabase_mutate_subscription,
    supabase_mutate_order,
    supabase_mutate_boost_listing,
    supabase_delete_order,
)
from src.utils.currency import format_price
from src.utils.shippo import (
    shippo_get_rate_details,
    shippo_validate_and_merge,
    shippo_build_shipment_and_select_rate,
)
from src.utils.paypal import (
    get_access_token,
    get_api_base,
    paypal_get_base_url,
    paypal_create_subscription,
    paypal_create_order,
    paypal_void_checkout,
    paypal_void_authorization,
    paypal_authorize_order,
    paypal_capture_authorization,
    paypal_get_order,
    paypal_cancel_subscription,
    paypal_fetch_plan_fixed_price,
    paypal_fetch_price_by_subscription_id,
    paypal_compute_adjusted_start,
    paypal_parse_iso_dt,
    paypal_build_gateway_account_info,
)
from src.utils.logger import (
    api_logger,
    log_api_request,
    log_success,
    log_failure,
)


# ===============================================================
# /paypal/subscription
# ===============================================================


async def paypal_subscription_handler(
    request: Request,
    user_id: str,
    plan_id: str,
    redirect: str,
):
    log_api_request(
        api_logger,
        "GET",
        "/paypal/subscription",
        {"user_id": user_id, "plan_id": plan_id, "redirect": redirect},
    )
    try:
        # 1. get paypal_plan_id
        membership_plan = supabase_get_membership_plan(
            {"id": plan_id}, columns="id,paypal_plan_id,subscription_days"
        )
        if not membership_plan:
            api_logger.error(
                "Membership plan not found for plan_id=%s in paypal_subscription_handler",
                plan_id,
            )
            raise HTTPException(status_code=404, detail="Plan not found")
        paypal_plan_id = membership_plan.get("paypal_plan_id")
        if not paypal_plan_id:
            api_logger.error(
                "Plan %s missing paypal_plan_id in membership_plans", plan_id
            )
            raise HTTPException(
                status_code=400, detail="Plan does not have PayPal plan id"
            )

        # Guard: prevent duplicate active subscription period for same user/plan
        now_dt = datetime.now()
        active_subscriptions = supabase_get_subscriptions(
            {"user_id": user_id, "status": "active"},
            columns="id, start_time, end_time, status",
        )
        if active_subscriptions:
            for active_subscription in active_subscriptions:
                start_dt = paypal_parse_iso_dt(active_subscription.get("start_time"))
                end_dt = paypal_parse_iso_dt(active_subscription.get("end_time"))
                # Block only if now is within [start_time, end_time)
                if start_dt and end_dt and start_dt <= now_dt < end_dt:
                    # Already in an active period; block new purchase
                    api_logger.info(
                        "Blocking new subscription: user_id=%s already active for plan_id=%s",
                        user_id,
                        plan_id,
                    )
                    params = urlencode(
                        {
                            "status": "error",
                            "reason": "already_active",
                        }
                    )
                    return RedirectResponse(
                        f"{redirect}{'&' if '?' in redirect else '?'}{params}",
                        status_code=302,
                    )

        # 2. build return/cancel urls (include internal plan_id so return handler can use it)
        base = paypal_get_base_url(request)
        return_url = f"{base}/api/v1/paypal/subscription/return?{urlencode({'redirect': redirect, 'user_id': user_id, 'plan_id': plan_id})}"
        cancel_url = f"{base}/api/v1/paypal/cancel?{urlencode({'redirect': redirect})}"

        subscription_resp = await paypal_create_subscription(
            plan_id=paypal_plan_id,
            return_url=return_url,
            cancel_url=cancel_url,
            user_details=None,
            custom_id=f"plan:{plan_id}",
        )

        approval_url = subscription_resp.get("approval_url")
        subscription_id = subscription_resp.get("subscription_id")
        if not approval_url or not subscription_id:
            api_logger.error(
                "PayPal create-subscription missing approval_url or subscription_id"
            )
            raise HTTPException(status_code=500, detail="Payment provider error")

        # 3. Persist or reuse a single 'pending' subscription per user/plan
        # Reuse or create: prefer updating an existing pending row for this user+plan
        primary_id = None
        pending_subscriptions = supabase_get_subscriptions(
            {"user_id": user_id, "status": "pending"},
            columns="id,payment_id,paypal_subscription_id,status",
        )
        pending_subscription = (
            pending_subscriptions[0] if pending_subscriptions else None
        )
        if pending_subscription:
            primary_id = pending_subscription.get("id")
            # Cancel any linked pending payment and unlink
            pending_payment_id = pending_subscription.get("payment_id")
            if pending_payment_id:
                supabase_mutate_payment(
                    "update", {"status": "cancelled"}, {"id": pending_payment_id}
                )
                supabase_mutate_subscription(
                    "update",
                    {"payment_id": None},
                    {"id": primary_id},
                )
                api_logger.info(
                    "Cancelled and unlinked pending payment %s for subscription %s",
                    pending_payment_id,
                    primary_id,
                )
            # Update existing pending row with the latest PayPal subscription id
            supabase_mutate_subscription(
                "update",
                {"paypal_subscription_id": subscription_id},
                {"id": primary_id},
            )
            log_success(
                api_logger,
                f"Updated pending subscription {primary_id} with paypal_subscription_id={subscription_id}",
            )
            # Cancel any other pending duplicates for safety
            supabase_mutate_subscription(
                "update",
                {"status": "cancelled"},
                {
                    "user_id": user_id,
                    "plan_id": plan_id,
                    "status": "pending",
                    "id": {"neq": primary_id},
                },
            )
            log_success(
                api_logger,
                f"Cancelled duplicate pending subscriptions for user_id={user_id} plan_id={plan_id} keeping id={primary_id}",
            )
        else:
            inserted_subscription = supabase_mutate_subscription(
                "insert",
                {
                    "user_id": user_id,
                    "plan_id": plan_id,
                    "status": "pending",
                    "paypal_subscription_id": subscription_id,
                    "start_time": None,
                    "end_time": None,
                },
            )
            primary_id = (
                inserted_subscription.data[0]["id"]
                if getattr(inserted_subscription, "data", None)
                else None
            )
            log_success(
                api_logger,
                f"Inserted pending subscription id={primary_id} for user_id={user_id} plan_id={plan_id}",
            )

        # 4. Ensure pending payment exists at submit time and link to subscription (idempotent)
        try:
            # Get plan amount from PayPal plan id (source of truth)
            plan_amount, plan_currency = await paypal_fetch_plan_fixed_price(
                paypal_plan_id
            )
            pid = None
            existing_pay = supabase_get_payment(
                {"gateway_order_id": subscription_id}
            )
            if existing_pay:
                pid = existing_pay.get("id")
            else:
                inserted_payment = supabase_mutate_payment(
                    "insert",
                    {
                        "user_id": user_id,
                        "gateway": "paypal",
                        "gateway_order_id": subscription_id,
                        "amount": plan_amount,
                        "currency": (plan_currency or "USD"),
                        "status": "pending",
                    },
                )
                if getattr(inserted_payment, "data", None):
                    pid = inserted_payment.data[0].get("id")
            if pid and primary_id:
                supabase_mutate_subscription(
                    "update", {"payment_id": pid}, {"id": primary_id}
                )
                log_success(
                    api_logger,
                    f"Linked pending payment {pid} to pending subscription {primary_id}",
                )
        except Exception as e:
            api_logger.exception(
                "Submit-time pending payment creation/link failed for subscription: %s",
                e,
            )

        # 5. redirect user to PayPal approval page
        return RedirectResponse(url=approval_url, status_code=302)

    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("PayPal checkout failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Checkout error: {e}")


# ===============================================================
# /paypal/subscription/cancel
# ===============================================================
async def paypal_subscription_cancel_handler(
    redirect: str,
    user_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    paypal_subscription_id: Optional[str] = None,
) -> RedirectResponse:
    log_api_request(
        api_logger,
        "GET",
        "/paypal/subscription/cancel",
        {"user_id": user_id, "plan_id": plan_id, "paypal_subscription_id": paypal_subscription_id},
    )
    try:
        paypal_subscription = None
        # Resolve local subscription row
        if paypal_subscription_id:
            paypal_subscription = supabase_get_subscription(
                {"paypal_subscription_id": paypal_subscription_id}
            )
        elif user_id and plan_id:
            # Prefer pending, else active for this user+plan
            paypal_subscription = supabase_get_subscription(
                {"user_id": user_id, "plan_id": plan_id, "status": "pending"}
            )
            if not paypal_subscription:
                paypal_subscription = supabase_get_subscription(
                    {"user_id": user_id, "plan_id": plan_id, "status": "active"}
                )

        # Attempt remote cancellation if we have a PayPal id
        paypal_id = paypal_subscription_id or (
            paypal_subscription.get("paypal_subscription_id")
            if paypal_subscription
            else None
        )
        if paypal_id:
            try:
                await paypal_cancel_subscription(
                    paypal_id, reason="User cancelled at approval"
                )
                log_success(api_logger, f"Cancelled PayPal subscription {paypal_id} remotely")
            except Exception as e:
                api_logger.exception(
                    "Failed to cancel PayPal subscription %s remotely: %s",
                    paypal_id,
                    e,
                )

        # Update local status to cancelled
        if paypal_subscription and paypal_subscription.get("id"):
            supabase_mutate_subscription(
                "update", {"status": "cancelled"}, {"id": paypal_subscription.get("id")}
            )
            log_success(
                api_logger,
                f"Marked local subscription {paypal_subscription.get('id')} as cancelled",
            )
        elif paypal_id:
            supabase_mutate_subscription(
                "update", {"status": "cancelled"}, {"paypal_subscription_id": paypal_id}
            )
            log_success(
                api_logger,
                f"Marked local subscription with paypal_subscription_id={paypal_id} as cancelled",
            )

        redirect_params = urlencode(
            {"status": "cancel", "subscriptionId": paypal_id or ""}
        )
        return RedirectResponse(
            url=f"{redirect}{'&' if '?' in redirect else '?'}{redirect_params}",
            status_code=302,
        )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("Error handling PayPal subscription cancel: %s", e)
        # Fall back to a safe redirect
        redirect_params = urlencode(
            {"status": "cancel", "subscriptionId": paypal_subscription_id or ""}
        )
        return RedirectResponse(
            url=f"{redirect}{'&' if '?' in redirect else '?'}{redirect_params}",
            status_code=302,
        )


# ===============================================================
# /paypal/subscription/return
# ===============================================================
async def paypal_subscription_return_handler(
    redirect: str,
    user_id: str,
    plan_id: str,
    subscription_id: Optional[str],
) -> RedirectResponse:
    try:
        log_api_request(
            api_logger,
            "GET",
            "/paypal/subscription/return",
            {
                "subscription_id": subscription_id,
                "user_id": user_id,
                "plan_id": plan_id,
            },
        )

        if not all([subscription_id, user_id, plan_id, redirect]):
            api_logger.error("Missing query params on PayPal return")
            params = urlencode({"status": "error", "reason": "missing_params"})
            return RedirectResponse(f"{redirect}?{params}", status_code=302)

        paypal_sub_id = subscription_id  # preserve provided PayPal subscription id
        token = await get_access_token()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{get_api_base().rstrip('/')}/v1/billing/subscriptions/{paypal_sub_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()

        # We do NOT activate locally here; leave subscription as pending until webhook.
        # Ensure internal subscription mapping exists (idempotent) without creating duplicates
        local_subscription_id = None
        existing_subscription = supabase_get_subscription(
            {"paypal_subscription_id": paypal_sub_id}
        )
        if existing_subscription:
            supabase_mutate_subscription(
                "update",
                {
                    "status": existing_subscription.get("status") or "pending",
                },
                {"id": existing_subscription.get("id")},
            )
            local_subscription_id = existing_subscription.get("id")
            log_success(
                api_logger,
                f"Found existing local subscription id={local_subscription_id} for paypal_subscription_id={paypal_sub_id}",
            )
        else:
            # Reuse any existing pending for this user+plan to avoid duplicates
            fallback_subscription = supabase_get_subscription(
                {"user_id": user_id, "plan_id": plan_id, "status": "pending"}
            )
            if fallback_subscription:
                supabase_mutate_subscription(
                    "update",
                    {"paypal_subscription_id": paypal_sub_id},
                    {"id": fallback_subscription.get("id")},
                )
                local_subscription_id = fallback_subscription.get("id")
                log_success(
                    api_logger,
                    f"Linked pending local subscription id={local_subscription_id} to paypal_subscription_id={paypal_sub_id}",
                )
            else:
                inserted_subscription = supabase_mutate_subscription(
                    "insert",
                    {
                        "user_id": user_id,
                        "plan_id": plan_id,
                        "status": "pending",
                        "paypal_subscription_id": paypal_sub_id,
                        "start_time": None,
                        "end_time": None,
                    },
                )
                if not getattr(inserted_subscription, "data", None):
                    api_logger.error(
                        "Failed to insert local subscription mapping on return"
                    )
                else:
                    local_subscription_id = inserted_subscription.data[0].get("id")
                    log_success(
                        api_logger,
                        f"Inserted local subscription mapping id={local_subscription_id} for paypal_subscription_id={paypal_sub_id}",
                    )
        # Insert a pending payment and link it via subscriptions.payment_id (idempotent, explicit steps)
        if local_subscription_id:
            # Step 1: read subscription
            subscription = supabase_get_subscription({"id": local_subscription_id})
            # Step 2: if no linked payment, insert one
            if not subscription or not subscription.get("payment_id"):
                plan_id_val = data.get("plan_id")
                if not plan_id_val:
                    log_failure(
                        api_logger,
                        "PayPal return: missing plan_id in subscription details; skipping pending payment insert",
                    )
                else:
                    plan_amount, plan_currency = await paypal_fetch_plan_fixed_price(
                        plan_id_val
                    )
                    gateway_account_info = await paypal_build_gateway_account_info(
                        paypal_sub_id,
                        "subscription",
                        plan_id,
                        data,
                    )
                    inserted_payment = supabase_mutate_payment(
                        "insert",
                        {
                            "user_id": user_id,
                            "gateway": "paypal",
                            "gateway_order_id": paypal_sub_id,
                            "amount": plan_amount,
                            "currency": (plan_currency or "USD"),
                            "gateway_account_info": gateway_account_info or None,
                            "status": "pending",
                        },
                    )
                    # Step 3: if inserted, link it on subscription
                    if getattr(inserted_payment, "data", None):
                        pid = inserted_payment.data[0].get("id")
                        supabase_mutate_subscription(
                            "update",
                            {"payment_id": pid},
                            {"id": local_subscription_id},
                        )
                        log_success(
                            api_logger,
                            f"Linked payment {pid} to local subscription {local_subscription_id}",
                        )

        # Redirect to client
        redirect_params = urlencode(
            {"status": "success", "subscriptionId": local_subscription_id}
        )
        return RedirectResponse(
            f"{redirect}{'&' if '?' in redirect else '?'}{redirect_params}",
            status_code=302,
        )

    except Exception as e:
        api_logger.exception(
            "Error checking PayPal subscription %s: %s", subscription_id, e
        )
        redirect_out = redirect or "/"
        redirect_params = urlencode(
            {
                "status": "error",
                "subscriptionId": subscription_id or "",
                "reason": "internal_error",
            }
        )
        return RedirectResponse(
            f"{redirect_out}{'&' if '?' in redirect_out else '?'}{redirect_params}",
            status_code=302,
        )


# ===============================================================
# /paypal/webhook/subscription
# ===============================================================
async def paypal_webhook_subscription_handler(request: Request) -> dict:
    try:
        body = await request.json()
        if not isinstance(body, dict):
            log_failure(api_logger, "Subscription webhook: body is not a JSON object")
            body = {}
        event_type = body.get("event_type")
        resource = body.get("resource")
        log_api_request(
            api_logger,
            "POST",
            "/paypal/webhook/subscription triggered",
            {"event_type": event_type},
        )

        if not event_type:
            log_failure(api_logger, "Subscription webhook: missing event_type")
            return {"status": "ignored"}

        if not isinstance(resource, dict) or not resource:
            log_failure(api_logger, f"Subscription webhook: missing resource for event {event_type}")
            return {"status": "ignored"}

        subscription_id = resource.get("id")
        if not subscription_id:
            api_logger.error("PayPal webhook missing subscription ID")
            return {"status": "error", "message": "Missing subscription ID"}

        # -------------------------
        # Subscription Activated
        # -------------------------
        if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
            # Activate locally using plan duration (start now, end now + subscription_days)
            subscription = supabase_get_subscription(
                {"paypal_subscription_id": subscription_id}
            )
            if subscription:
                plan = supabase_get_membership_plan(
                    {"id": subscription.get("plan_id")}, columns="subscription_days"
                )
                if not plan:
                    log_failure(
                        api_logger,
                        "ACTIVATED webhook: membership plan not found; defaulting plan_days=30",
                    )
                    plan_days = 30
                else:
                    plan_days = int(plan.get("subscription_days") or 30)
                adj_start = await paypal_compute_adjusted_start(
                    subscription.get("user_id"),
                    subscription.get("plan_id"),
                    subscription.get("id"),
                )
                end_dt = adj_start + timedelta(days=plan_days)
                supabase_mutate_subscription(
                    "update",
                    {
                        "status": "active",
                        "start_time": adj_start.isoformat(),
                        "end_time": end_dt.isoformat(),
                    },
                    {"id": subscription.get("id")},
                )
                api_logger.info(
                    "Activated subscription %s (user=%s plan=%s) start=%s end=%s",
                    subscription.get("id"),
                    subscription.get("user_id"),
                    subscription.get("plan_id"),
                    adj_start.isoformat(),
                    end_dt.isoformat(),
                )
                # Ensure a pending payment exists and is linked (webhook fallback)
                if not subscription.get("payment_id"):
                    plan_amount, plan_currency = (
                        await paypal_fetch_price_by_subscription_id(subscription_id)
                    )
                    inserted_payment = supabase_mutate_payment(
                        "insert",
                        {
                            "user_id": subscription.get("user_id"),
                            "gateway": "paypal",
                            "gateway_order_id": subscription_id,
                            "amount": plan_amount,
                            "currency": (plan_currency or "USD"),
                            "status": "pending",
                        },
                    )
                    if getattr(inserted_payment, "data", None):
                        pid = inserted_payment.data[0].get("id")
                        supabase_mutate_subscription(
                            "update",
                            {"payment_id": pid},
                            {"id": subscription.get("id")},
                        )
                        api_logger.info(
                            "Webhook ACTIVATED: inserted pending payment %s linked to subscription %s",
                            pid,
                            subscription.get("id"),
                        )
            else:
                api_logger.warning(
                    f"Subscription {subscription_id} ACTIVATED but not found in DB. Skipping activation."
                )

        # -------------------------
        # Subscription Cancelled
        # -------------------------
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            supabase_mutate_subscription(
                "update",
                {"status": "cancelled"},
                {"paypal_subscription_id": subscription_id},
            )
            api_logger.info(
                "Cancelled local subscription with paypal_subscription_id=%s",
                subscription_id,
            )

        # -------------------------
        # Payment Completed
        # -------------------------
        elif event_type == "PAYMENT.SALE.COMPLETED":
            paypal_sub_id = resource.get("billing_agreement_id")
            txn_id = resource.get("id")
            amt_obj = resource.get("amount")
            amount_val = None
            currency_val = None
            if isinstance(amt_obj, dict):
                amount_val = amt_obj.get("total")
                currency_val = amt_obj.get("currency")
            else:
                log_failure(api_logger, "PAYMENT.SALE.COMPLETED: missing amount object in resource")
            if not paypal_sub_id:
                api_logger.error(
                    "Missing billing_agreement_id in PAYMENT.SALE.COMPLETED"
                )
            else:
                subscription = supabase_get_subscription(
                    {"paypal_subscription_id": paypal_sub_id}
                )
                if subscription:
                    # Resolve internal plan_id (fallback: map PayPal subscription.plan_id → membership_plans.id)
                    internal_plan_id = subscription.get("plan_id")
                    if not internal_plan_id:
                        try:
                            access_token = await get_access_token()
                            async with httpx.AsyncClient(timeout=20) as client:
                                sres = await client.get(
                                    f"{get_api_base().rstrip('/')}/v1/billing/subscriptions/{paypal_sub_id}",
                                    headers={"Authorization": f"Bearer {access_token}"},
                                )
                                if sres.status_code == 200:
                                    sjson = sres.json() or {}
                                    pp_plan_id = sjson.get("plan_id")
                                    if pp_plan_id:
                                        plan_row = supabase_get_membership_plan(
                                            {"paypal_plan_id": pp_plan_id}, columns="id"
                                        )
                                        if plan_row:
                                            internal_plan_id = plan_row.get("id")
                        except Exception:
                            api_logger.exception(
                                "Failed to map PayPal plan id to internal plan id for subscription %s",
                                subscription.get("id"),
                            )
                    # One payment per subscription cycle:
                    # 1) Prefer updating the linked pending payment via subscriptions.payment_id
                    # 2) Else, idempotent upsert by gateway_transaction_id and link it
                    # Build gateway account info once for reuse
                    gateway_info = await paypal_build_gateway_account_info(
                        paypal_sub_id,
                        "subscription",
                        internal_plan_id,
                        resource,
                    )
                    updated = False
                    pid = subscription.get("payment_id")
                    if pid:
                        supabase_mutate_payment(
                            "update",
                            {
                                "status": "succeeded",
                                "gateway_transaction_id": txn_id,
                                "gateway_capture_id": txn_id,
                                "gateway_order_id": paypal_sub_id,
                                "amount": amount_val,
                                "currency": currency_val,
                                "gateway_account_info": gateway_info,
                            },
                            {"id": pid},
                        )
                        updated = True
                        api_logger.info(
                            "Updated linked payment %s as succeeded for subscription %s",
                            pid,
                            subscription.get("id"),
                        )
                    if not updated:
                        # Idempotent by gateway_transaction_id
                        existing_txn_row = supabase_get_payment(
                            {"gateway_transaction_id": txn_id}
                        )
                        if existing_txn_row:
                            pid = existing_txn_row["id"]
                            supabase_mutate_payment(
                                "update",
                                {
                                    "user_id": subscription.get("user_id"),
                                    "status": "succeeded",
                                    "gateway_capture_id": txn_id,
                                    "gateway_order_id": paypal_sub_id,
                                    "amount": amount_val,
                                    "currency": currency_val,
                                    "gateway_account_info": gateway_info,
                                },
                                {"id": pid},
                            )
                            # Link payment to subscription if not linked yet
                            if not subscription.get("payment_id"):
                                supabase_mutate_subscription(
                                    "update",
                                    {"payment_id": pid},
                                    {"id": subscription.get("id")},
                                )
                                api_logger.info(
                                    "Linked existing payment %s to subscription %s",
                                    pid,
                                    subscription.get("id"),
                                )
                            api_logger.info(
                                "Marked existing payment %s as succeeded for subscription %s",
                                pid,
                                subscription.get("id"),
                            )
                        else:
                            inserted_payment = supabase_mutate_payment(
                                "insert",
                                {
                                    "user_id": subscription.get("user_id"),
                                    "gateway": "paypal",
                                    "gateway_transaction_id": txn_id,
                                    "gateway_capture_id": txn_id,
                                    "gateway_order_id": paypal_sub_id,
                                    "amount": amount_val,
                                    "currency": currency_val,
                                    "status": "succeeded",
                                    "gateway_account_info": await paypal_build_gateway_account_info(
                                        paypal_sub_id,
                                        "subscription",
                                        internal_plan_id,
                                        resource,
                                    ),
                                },
                            )
                            if getattr(inserted_payment, "data", None):
                                new_pid = inserted_payment.data[0].get("id")
                                supabase_mutate_subscription(
                                    "update",
                                    {"payment_id": new_pid},
                                    {"id": subscription.get("id")},
                                )
                                api_logger.info(
                                    "Inserted succeeded payment %s and linked to subscription %s",
                                    new_pid,
                                    subscription.get("id"),
                                )

                    # Ensure subscription is active with correct dates (fallback if ACTIVATED not processed)
                    if (subscription.get("status") or "").lower() != "active":
                        plan = supabase_get_membership_plan(
                            {"id": subscription.get("plan_id")},
                            columns="subscription_days",
                        )
                        if not plan:
                            log_failure(
                                api_logger,
                                "PAYMENT.SALE.COMPLETED: membership plan not found; defaulting plan_days=30",
                            )
                            plan_days = 30
                        else:
                            plan_days = int(plan.get("subscription_days") or 30)
                        adj_start = await paypal_compute_adjusted_start(
                            subscription.get("user_id"),
                            subscription.get("plan_id"),
                            subscription.get("id"),
                        )
                        end_dt = adj_start + timedelta(days=plan_days)
                        supabase_mutate_subscription(
                            "update",
                            {
                                "status": "active",
                                "start_time": adj_start.isoformat(),
                                "end_time": end_dt.isoformat(),
                            },
                            {"id": subscription.get("id")},
                        )
                        api_logger.info(
                            "Fallback ACTIVATION after capture: subscription %s start=%s end=%s",
                            subscription.get("id"),
                            adj_start.isoformat(),
                            end_dt.isoformat(),
                        )
                else:
                    api_logger.error(
                        f"No subscription found for PayPal ID {paypal_sub_id}"
                    )

        # -------------------------
        # Subscription Updated
        # -------------------------
        elif event_type == "BILLING.SUBSCRIPTION.UPDATED":
            supabase_mutate_subscription(
                "update",
                {
                    "status": resource["status"].lower(),
                    "end_time": resource.get("billing_info", {}).get(
                        "next_billing_time"
                    ),
                },
                {"paypal_subscription_id": subscription_id},
            )

        api_logger.info(f"Handled PayPal webhook: {event_type} for {subscription_id}")
        return {"status": "ok"}

    except Exception as e:
        api_logger.exception("PayPal webhook error: %s", e)
        raise HTTPException(status_code=400, detail="Invalid webhook payload")


# ===============================================================
# /paypal/boost-post
# ===============================================================


async def paypal_boost_handler(
    request: Request,
    user_id: str,
    listing_id: str,
    plan_id: str,
    redirect: str,
):
    """Start PayPal checkout using Subscriptions API for boost-post.
    Differences from membership: uses `boost_plans` and later creates a row in `boost_listings`.
    """
    log_api_request(
        api_logger,
        "GET",
        "/paypal/boost-post",
        {"user_id": user_id, "listing_id": listing_id, "plan_id": plan_id, "redirect": redirect},
    )
    try:
        # Validate listing belongs to the user initiating the boost
        listing = supabase_get_listing(
            {"id": listing_id}, columns="id, seller_id"
        )
        if not listing:
            api_logger.error("Listing not found for boost: %s", listing_id)
            raise HTTPException(status_code=404, detail="Listing not found")
        if (listing.get("seller_id") or "") != user_id:
            api_logger.error(
                "Boost listing seller mismatch: user=%s listing.seller=%s",
                user_id,
                listing.get("seller_id"),
            )
            raise HTTPException(status_code=403, detail="Cannot boost other user's listing")

        # 1) fetch PayPal plan id from boost_plans
        boost_plan = supabase_get_boost_plan(
            {"id": plan_id}, columns="id,paypal_plan_id,boost_days,currency,price"
        )
        if not boost_plan:
            api_logger.error("Boost plan not found: %s", plan_id)
            raise HTTPException(status_code=404, detail="Boost plan not found")
        paypal_plan_id = boost_plan.get("paypal_plan_id")
        if not paypal_plan_id:
            raise HTTPException(status_code=400, detail="Boost plan missing PayPal plan id")

        # 2) build return/cancel urls embedding identifiers
        base = paypal_get_base_url(request)
        return_url = f"{base}/api/v1/paypal/boost-post/return?{urlencode({'redirect': redirect, 'user_id': user_id, 'listing_id': listing_id, 'plan_id': plan_id})}"
        cancel_url = f"{base}/api/v1/paypal/cancel?{urlencode({'redirect': redirect})}"

        subscription_resp = await paypal_create_subscription(
            plan_id=paypal_plan_id,
            return_url=return_url,
            cancel_url=cancel_url,
            user_details=None,
            custom_id=f"listing:{listing_id}|plan:{plan_id}",
        )

        approval_url = subscription_resp.get("approval_url")
        paypal_subscription_id = subscription_resp.get("subscription_id")
        if not approval_url or not paypal_subscription_id:
            api_logger.error("PayPal create-subscription (boost) missing approval_url or subscription_id")
            raise HTTPException(status_code=500, detail="Payment provider error")

        # 3) Create pending payment and pending boost_listings (no gateway_account_info)
        try:
            # Fetch price from PayPal plan (source of truth)
            amount_val, currency_val = await paypal_fetch_plan_fixed_price(paypal_plan_id)

            # Insert payment pending (maps to PayPal subscription via gateway_order_id)
            existing_payment = supabase_get_payment({"gateway_order_id": paypal_subscription_id})
            payment_id = None
            if existing_payment:
                payment_id = existing_payment.get("id")
            else:
                insert_pay = supabase_mutate_payment(
                    "insert",
                    {
                        "user_id": user_id,
                        "gateway": "paypal",
                        "gateway_order_id": paypal_subscription_id,
                        "amount": amount_val,
                        "currency": (currency_val or "USD"),
                        "status": "pending",
                    },
                )
                if not getattr(insert_pay, "data", None):
                    api_logger.error("Failed to insert pending payment for boost")
                else:
                    payment_id = insert_pay.data[0].get("id")

            # Insert boost_listings pending linked to payment
            if payment_id:
                # Avoid duplicate pending boost rows for the same payment
                existing_boost = supabase_get_boost_listing({"payment_id": payment_id})
                if not existing_boost:
                    # Insert pending boost without schedule; schedule will be set on ACTIVATED webhook
                    ins_boost = supabase_mutate_boost_listing(
                        "insert",
                        {
                            "listing_id": listing_id,
                            "plan_id": plan_id,
                            "payment_id": payment_id,
                            "status": "pending",
                        },
                    )
                    if not getattr(ins_boost, "data", None):
                        api_logger.error(
                            "Failed to insert pending boost_listings for listing %s (payment %s)",
                            listing_id,
                            payment_id,
                        )
                    else:
                        log_success(
                            api_logger,
                            f"Inserted pending boost_listings {ins_boost.data[0].get('id')} for listing={listing_id} plan={plan_id}",
                        )
        except Exception as e:
            api_logger.exception("Failed to create pending payment/boost row for boost flow: %s", e)

        # 4) redirect to approval
        return RedirectResponse(url=approval_url, status_code=302)
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("PayPal boost checkout failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Boost checkout error: {e}")


# ===============================================================
# /paypal/boost-post/return
# ===============================================================
async def paypal_boost_return_handler(
    redirect: str,
    user_id: str,
    listing_id: str,
    plan_id: str,
    subscription_id: Optional[str],
) -> RedirectResponse:
    try:
        log_api_request(
            api_logger,
            "GET",
            "/paypal/boost-post/return",
            {
                "subscription_id": subscription_id,
                "user_id": user_id,
                "listing_id": listing_id,
                "plan_id": plan_id,
            },
        )

        if not all([subscription_id, user_id, listing_id, plan_id, redirect]):
            api_logger.error("Boost return: missing params")
            params = urlencode({"status": "error", "reason": "missing_params"})
            return RedirectResponse(f"{redirect}?{params}", status_code=302)

        paypal_sub_id = subscription_id
        token = await get_access_token()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{get_api_base().rstrip('/')}/v1/billing/subscriptions/{paypal_sub_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()

        # Nothing else to persist here; pending payment and boost_listings are created on submit.

        # Redirect back to client
        redirect_params = urlencode({"status": "success", "subscriptionId": paypal_sub_id})
        return RedirectResponse(
            f"{redirect}{'&' if '?' in redirect else '?'}{redirect_params}",
            status_code=302,
        )
    except Exception as e:
        api_logger.exception("Error handling PayPal boost return: %s", e)
        redirect_out = redirect or "/"
        redirect_params = urlencode(
            {"status": "error", "subscriptionId": subscription_id or "", "reason": "internal_error"}
        )
        return RedirectResponse(
            f"{redirect_out}{'&' if '?' in redirect_out else '?'}{redirect_params}",
            status_code=302,
        )


# ===============================================================
# /paypal/webhook/boost-post
# ===============================================================
async def paypal_webhook_boost_handler(request: Request) -> dict:
    try:
        body = await request.json()
        if not isinstance(body, dict):
            log_failure(api_logger, "Boost webhook: body is not a JSON object")
            body = {}
        event_type = body.get("event_type")
        resource = body.get("resource")
        log_api_request(
            api_logger,
            "POST",
            "/paypal/webhook/boost-post triggered",
            {"event_type": event_type},
        )

        if not event_type or not isinstance(resource, dict):
            return {"status": "ignored"}

        subscription_id = resource.get("id")
        # -------------- ACTIVATED --------------
        if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
            # Find pending payment by gateway_order_id
            payment_row = supabase_get_payment({"gateway_order_id": subscription_id})
            if not payment_row:
                # If missing unexpectedly, create a minimal pending payment to allow linking
                try:
                    amount_val, currency_val = await paypal_fetch_price_by_subscription_id(subscription_id)
                except Exception:
                    amount_val, currency_val = None, None
                inserted = supabase_mutate_payment(
                    "insert",
                    {
                        "gateway": "paypal",
                        "gateway_order_id": subscription_id,
                        "amount": amount_val,
                        "currency": (currency_val or "USD"),
                        "status": "pending",
                    },
                )
                payment_row = inserted.data[0] if getattr(inserted, "data", None) else None
            if not payment_row:
                api_logger.error("Boost ACTIVATED but no payment mapping for subscription %s", subscription_id)
                return {"status": "ok"}

            # Find associated boost_listings via payment_id; if missing, parse from PayPal custom_id
            boost_row = supabase_get_boost_listing({"payment_id": payment_row.get("id")})
            listing_id = None
            plan_id = None
            if boost_row:
                listing_id = boost_row.get("listing_id")
                plan_id = boost_row.get("plan_id")
            else:
                # Try to recover mapping from PayPal resource.custom_id
                custom_id = resource.get("custom_id") if isinstance(resource, dict) else None
                if isinstance(custom_id, str) and "listing:" in custom_id and "|" in custom_id:
                    try:
                        parts = dict(
                            p.split(":", 1) for p in custom_id.split("|") if ":" in p
                        )
                        listing_id = parts.get("listing")
                        plan_id = parts.get("plan")
                    except Exception:
                        listing_id, plan_id = None, None
                if not listing_id or not plan_id:
                    api_logger.error(
                        "Boost ACTIVATED but boost_listings not found and custom_id missing/invalid for payment %s",
                        payment_row.get("id"),
                    )
                    return {"status": "ok"}

            # Compute adjusted start to avoid overlapping boosts
            now_dt = datetime.now()
            latest_end = None
            try:
                existing = supabase_get_boost_listings(
                    {"listing_id": listing_id, "status": "active"},
                    columns="id,start_time,end_time,status",
                )
                for b in existing:
                    st = paypal_parse_iso_dt(b.get("start_time"))
                    et = paypal_parse_iso_dt(b.get("end_time"))
                    if et and et > now_dt and (latest_end is None or et > latest_end):
                        latest_end = et
            except Exception:
                api_logger.exception("Failed to compute adjusted boost start; using now")
            start_dt = latest_end if latest_end and latest_end > now_dt else now_dt

            # Determine boost duration from boost_plans
            boost_plan = supabase_get_boost_plan({"id": plan_id}, columns="boost_days")
            boost_days = int((boost_plan or {}).get("boost_days") or 1)
            end_dt = start_dt + timedelta(days=boost_days)

            # Update existing boost_listings to active, or create if missing
            if boost_row:
                upd = supabase_mutate_boost_listing(
                    "update",
                    {
                        "start_time": start_dt.isoformat(),
                        "end_time": end_dt.isoformat(),
                        "status": "active",
                    },
                    {"id": boost_row.get("id")},
                )
                if not getattr(upd, "data", None):
                    api_logger.error("Failed to activate boost_listings %s", boost_row.get("id"))
                else:
                    api_logger.info(
                        "Activated boost %s for listing %s from %s to %s",
                        boost_row.get("id"),
                        listing_id,
                        start_dt.isoformat(),
                        end_dt.isoformat(),
                    )
            else:
                # Insert active row as fallback if none existed
                ins = supabase_mutate_boost_listing(
                    "insert",
                    {
                        "listing_id": listing_id,
                        "plan_id": plan_id,
                        "payment_id": payment_row.get("id"),
                        "start_time": start_dt.isoformat(),
                        "end_time": end_dt.isoformat(),
                        "status": "active",
                    },
                )
                if not getattr(ins, "data", None):
                    api_logger.error(
                        "Failed to insert active boost_listings for listing %s (payment %s)",
                        listing_id,
                        payment_row.get("id"),
                    )
                else:
                    api_logger.info(
                        "Inserted active boost %s for listing %s from %s to %s",
                        ins.data[0].get("id"),
                        listing_id,
                        start_dt.isoformat(),
                        end_dt.isoformat(),
                    )
            return {"status": "ok"}

        # -------------- CANCELLED --------------
        if event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            # mark payment cancelled if exists
            pay = supabase_get_payment({"gateway_order_id": subscription_id})
            if pay:
                supabase_mutate_payment("update", {"status": "cancelled"}, {"id": pay.get("id")})
                boost_row = supabase_get_boost_listing({"payment_id": pay.get("id")})
                if boost_row:
                    supabase_mutate_boost_listing("update", {"status": "cancelled"}, {"id": boost_row.get("id")})
            return {"status": "ok"}

        # -------------- PAYMENT.SALE.COMPLETED --------------
        if event_type == "PAYMENT.SALE.COMPLETED":
            paypal_sub_id = resource.get("billing_agreement_id")
            txn_id = resource.get("id")
            amt_obj = resource.get("amount") or {}
            amount_val = amt_obj.get("total")
            currency_val = amt_obj.get("currency")
            if not paypal_sub_id:
                return {"status": "ignored"}
            # Update pending payment to succeeded and populate gateway_account_info
            existing_txn_row = supabase_get_payment({"gateway_order_id": paypal_sub_id})
            if existing_txn_row:
                # Try to enrich mapping from boost_listings via payment_id
                listing_id = None
                plan_id = None
                boost_row = supabase_get_boost_listing({"payment_id": existing_txn_row.get("id")})
                if boost_row:
                    listing_id = boost_row.get("listing_id")
                    plan_id = boost_row.get("plan_id")
                else:
                    # Fallback: parse custom_id from webhook resource
                    custom_id = resource.get("custom_id") if isinstance(resource, dict) else None
                    if isinstance(custom_id, str) and "listing:" in custom_id and "|" in custom_id:
                        try:
                            parts = dict(p.split(":", 1) for p in custom_id.split("|") if ":" in p)
                            listing_id = listing_id or parts.get("listing")
                            plan_id = plan_id or parts.get("plan")
                        except Exception:
                            listing_id, plan_id = listing_id, plan_id

                supabase_mutate_payment(
                    "update",
                    {
                        "status": "succeeded",
                        "gateway_transaction_id": txn_id,
                        "gateway_capture_id": txn_id,
                        "amount": amount_val,
                        "currency": currency_val,
                        "gateway_account_info": await paypal_build_gateway_account_info(
                            paypal_sub_id,
                            "boost",
                            plan_id,
                            resource,
                        ),
                    },
                    {"id": existing_txn_row.get("id")},
                )
            return {"status": "ok"}

        return {"status": "ignored"}
    except Exception as e:
        api_logger.exception("PayPal boost webhook error: %s", e)
        raise HTTPException(status_code=400, detail="Invalid webhook payload")


# ===============================================================
# /paypal/order
# ===============================================================
async def paypal_order_handler(request: Request, payload: "TransactionPayload"):
    """
    Refactored paypal order handler:
    - safe_handler ensures every error is logged
    - address validation improved (street2/phone/email)
    - handles Shippo 'multiple addresses' with suggestions
    """
    log_api_request(api_logger, "POST", "/paypal/order", {"payload": payload})

    # -------------------------
    # Parameter Extraction
    # -------------------------
    listing_id = payload.listing_id
    buyer_id = payload.buyer_id
    selected_rate_id = payload.selected_rate_id
    selected_rate_amount = payload.selected_rate_amount
    selected_rate_currency = payload.selected_rate_currency
    insurance = payload.insurance
    insurance_amount = payload.insurance_amount
    insurance_currency = payload.insurance_currency
    redirect = payload.redirect

    # -------------------------
    # Load Listing Data
    # -------------------------
    listing_card = supabase_get_listing_card({"id": listing_id})
    if not listing_card:
        api_logger.error(
            "Listing not found for listing_id=%s in paypal_order_handler", listing_id
        )
        raise HTTPException(status_code=404, detail="Listing not found")

    listing = listing_card
    selected_listing_price = (
        listing.get("highest_bid_price", 0)
        if listing.get("listing_type") == "auction"
        else listing.get("start_price", 0)
    )

    # -------------------------
    # Sold guard: if listing already has a paid order, stop and clean pending ones
    # -------------------------
    sold_statuses = ["awaiting_shipment", "shipped", "delivered", "completed"]
    sold_order = supabase_get_order(
        {"listing_id": listing_id, "status": {"in": sold_statuses}}
    )
    if sold_order:
        # Best-effort cleanup of stale awaiting_payment orders (ignore errors)
        try:
            supabase_delete_order(
                {"listing_id": listing_id, "status": "awaiting_payment"}
            )
        except Exception:
            pass
        return {"status": "sold"}

    # -------------------------
    # Load Buyer Currency
    # -------------------------
    profile_row = supabase_get_profile({"id": buyer_id}, columns="currency")
    profile_currency = profile_row.get("currency") if profile_row else None
    if not profile_currency:
        api_logger.error(
            "Buyer currency lookup failed for user_id=%s (profile_row=%s)",
            buyer_id,
            profile_row,
        )
        raise HTTPException(status_code=500, detail="Buyer currency lookup failed")
    buyer_currency = (
        profile_currency or insurance_currency or selected_rate_currency or "USD"
    ).upper()

    # -------------------------
    # Check for existing pending order (we will UPDATE it instead of returning stale gateway URLs)
    # -------------------------
    reuse_order = False
    order_id = None
    old_gateway_order_id = None
    existing_order = supabase_get_order(
        {
            "listing_id": listing_id,
            "buyer_id": buyer_id,
            "status": "awaiting_payment",
        }
    )
    if existing_order:
        order_id = existing_order.get("id")
        old_gateway_order_id = existing_order.get("gateway_order_id")
        reuse_order = bool(order_id)
        api_logger.info(
            "Reusing existing awaiting_payment order %s for buyer=%s listing=%s (will refresh amounts and PayPal order)",
            order_id,
            buyer_id,
            listing_id,
        )

    # -------------------------
    # Load and Validate Seller Address (address_from)
    # -------------------------
    seller_id = listing.get("seller_id")
    if not seller_id:
        api_logger.error(
            "Listing %s missing seller reference (seller_id is null)", listing_id
        )
        raise HTTPException(status_code=500, detail="Listing missing seller reference")

    seller_address = supabase_get_user_address({"user_id": seller_id})
    if not seller_address:
        api_logger.error(
            "No shipping origin address found for seller_id=%s (listing_id=%s)",
            seller_id,
            listing_id,
        )
        raise HTTPException(
            status_code=400, detail="No shipping origin address found for seller"
        )

    seller_profile = supabase_get_profile(
        {"id": seller_id}, columns="username,email,phone_number"
    )
    if not seller_profile:
        api_logger.error(
            "Seller profile not found for seller_id=%s (listing_id=%s)",
            seller_id,
            listing_id,
        )
        raise HTTPException(status_code=500, detail="Seller profile lookup failed")

    seller_address = {
        "name": (seller_profile.get("username") or "").strip() or "Seller",
        "street1": seller_address.get("address_line_1", ""),
        "street2": seller_address.get("address_line_2", "") or "",
        "city": seller_address.get("city", ""),
        "state": seller_address.get("state_province", ""),
        "zip": seller_address.get("postal_code", ""),
        "country": (seller_address.get("country", "") or "").upper(),
        "phone": seller_profile.get("phone_number") or None,
        "email": seller_profile.get("email") or None,
    }

    # Validate and normalize seller address (merged helper)
    seller_address = await shippo_validate_and_merge(seller_address)

    # -------------------------
    # Load and Validate Buyer Address
    # -------------------------
    buyer_address = supabase_get_user_address({"user_id": buyer_id})
    if not buyer_address:
        api_logger.error(
            "No shipping destination address found for buyer_id=%s (listing_id=%s)",
            buyer_id,
            listing_id,
        )
        raise HTTPException(
            status_code=400, detail="No shipping destination address found"
        )

    # Get buyer profile for name, phone, email (since user_addresses table only has address fields)
    buyer_profile = supabase_get_profile(
        {"id": buyer_id}, columns="username,email,phone_number"
    )
    if not buyer_profile:
        api_logger.error(
            "Buyer profile not found for buyer_id=%s (listing_id=%s)",
            buyer_id,
            listing_id,
        )
        raise HTTPException(status_code=500, detail="Buyer profile lookup failed")

    # Build shipping payload including street2, phone, email to improve Shippo match accuracy
    buyer_name = buyer_profile.get("username", "")
    shipping_address = {
        "name": buyer_name or "Buyer",
        "street1": buyer_address.get("address_line_1", ""),
        "street2": buyer_address.get("address_line_2", "") or "",
        "city": buyer_address.get("city", ""),
        "state": buyer_address.get("state_province", ""),
        "zip": buyer_address.get("postal_code", ""),
        "country": buyer_address.get("country", ""),
        "phone": buyer_profile.get("phone_number") or None,
        "email": buyer_profile.get("email") or None,
    }

    # Validate and normalize buyer address (merged helper)
    shipping_address = await shippo_validate_and_merge(shipping_address)

    # -------------------------
    # Build server-side shipment and select canonical Shippo rate
    # -------------------------
    try:
        # Derive hint from client-provided rate id if possible
        client_hint = {}
        try:
            hint = shippo_get_rate_details(selected_rate_id)
            client_hint = hint
        except Exception:
            # Fallback to amount/currency hint only
            client_hint = {
                "amount": (
                    Decimal(str(selected_rate_amount))
                    if selected_rate_amount is not None
                    else None
                ),
                "currency": (selected_rate_currency or "").upper(),
            }

        ins_amount_str = None
        if insurance and insurance_amount and float(insurance_amount) > 0:
            try:
                ins_amount_str = f"{float(insurance_amount):.2f}"
            except Exception:
                ins_amount_str = None

        selection = shippo_build_shipment_and_select_rate(
            address_from=seller_address,
            address_to=shipping_address,
            client_hint=client_hint,
            parcel=None,
            insurance_amount=ins_amount_str,
            insurance_currency=insurance_currency,
        )
        canonical_rate_id = selection["rate_id"]
        canonical_rate_amount = selection["amount"]  # Decimal
        canonical_rate_currency = selection["currency"]
        if not canonical_rate_id:
            raise HTTPException(
                status_code=400, detail="Failed to select shipping rate"
            )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("Server-side shipment build/select rate failed: %s", e)
        raise HTTPException(status_code=400, detail="Unable to prepare shipping rate")

    # -------------------------
    # Format all amounts into buyer currency
    # -------------------------
    try:
        shipping_fee = format_price(
            from_currency=canonical_rate_currency,
            to_currency=buyer_currency,
            amount=(
                float(canonical_rate_amount)
                if canonical_rate_amount is not None
                else 0.0
            ),
        )
        seller_earnings = format_price(
            from_currency=listing.get("currency"),
            to_currency=buyer_currency,
            amount=selected_listing_price,
        )
        insurance_fee = Decimal("0.00")
        if insurance and insurance_amount and insurance_amount > 0:
            insurance_fee = format_price(
                from_currency=insurance_currency,
                to_currency=buyer_currency,
                amount=insurance_amount,
            )
    except (InvalidOperation, TypeError) as e:
        api_logger.exception("Format all amounts into buyer currency failed: %s", e)
        raise HTTPException(status_code=400, detail="Invalid numeric values provided")
    except Exception as e:
        api_logger.exception("Unexpected error formatting amounts: %s", e)
        raise HTTPException(status_code=500, detail="Error formatting amounts")

    # -------------------------
    # Final Price Calculation
    # -------------------------
    try:
        final_price = (
            Decimal(seller_earnings) + Decimal(shipping_fee) + Decimal(insurance_fee)
        )
        final_price = final_price.quantize(Decimal("0.01"))
    except Exception as e:
        api_logger.exception("Final price calc failed: %s", e)
        raise HTTPException(status_code=500, detail="Final price calculation error")

    # -------------------------
    # Create or Update order record (awaiting_payment) - DB authoritative
    # -------------------------
    try:
        if reuse_order and order_id:
            update_fields = {
                "currency": buyer_currency,
                "final_price": str(final_price),
                "seller_earnings": str(seller_earnings),
                "shipping_fee": str(shipping_fee),
                "insurance_fee": str(insurance_fee),
                "status": "awaiting_payment",
                "shipping_address": shipping_address,
                "shipping_rate_id": canonical_rate_id,
            }
            supabase_mutate_order("update", update_fields, {"id": order_id})
        else:
            order_data = {
                "listing_id": listing_id,
                "seller_id": listing.get("seller_id"),
                "buyer_id": buyer_id,
                "currency": buyer_currency,
                "final_price": str(final_price),
                "seller_earnings": str(seller_earnings),
                "shipping_fee": str(shipping_fee),
                "insurance_fee": str(insurance_fee),
                "status": "awaiting_payment",
                "shipping_address": shipping_address,
                "shipping_rate_id": canonical_rate_id,
            }
            insert_resp = supabase_mutate_order("insert", order_data)
            if not getattr(insert_resp, "data", None) or len(insert_resp.data) == 0:
                api_logger.error("DB insert returned empty for order creation")
                raise HTTPException(status_code=500, detail="Database insert failed")
            order_row = insert_resp.data[0]
            order_id = order_row.get("id")
            if not order_id:
                api_logger.error("Order created but no ID returned")
                raise HTTPException(
                    status_code=500, detail="Order created but no ID returned"
                )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("Create/Update order failed: %s", e)
        raise HTTPException(status_code=500, detail="Create/Update order error")

    # -------------------------
    # Create Paypal Approve URL and update order with gateway fields
    # -------------------------
    try:
        # Best-effort: when reusing order, void previous PayPal order id to avoid stale unpaid orders
        if reuse_order and old_gateway_order_id:
            try:
                await paypal_void_checkout(old_gateway_order_id)
                api_logger.info(
                    "Voiding previous PayPal order id %s for reused order %s",
                    old_gateway_order_id,
                    order_id,
                )
            except Exception as ve:
                api_logger.exception(
                    "Failed to void previous PayPal order %s (continuing): %s",
                    old_gateway_order_id,
                    ve,
                )

        base = paypal_get_base_url(request)
        return_url = (
            f"{base}/api/v1/paypal/order/return?{urlencode({'redirect': redirect})}"
        )
        cancel_url = f"{base}/api/v1/paypal/cancel?{urlencode({'redirect': redirect})}"

        amount_breakdown = {
            "item_total": f"{seller_earnings}",
            "shipping": f"{shipping_fee}",
        }
        if insurance_fee and Decimal(str(insurance_fee)) > 0:
            amount_breakdown["handling"] = f"{insurance_fee}"

        # Prefer cards.canonical_title via card_id, then view's title, then name
        card_id_from_listing = listing.get("card_id")
        item_name = None
        if card_id_from_listing:
            card = supabase_get_card(
                {"id": card_id_from_listing}, columns="canonical_title"
            )
            if card:
                item_name = card.get("canonical_title")
            else:
                api_logger.error("Card not found for card_id %s", card_id_from_listing)
                raise HTTPException(
                    status_code=500, detail="Card not found for card_id"
                )

        item_unit_amount = (
            Decimal(amount_breakdown["item_total"])
            if amount_breakdown.get("item_total")
            else Decimal(str(seller_earnings))
        )

        items = [
            {
                "name": item_name[:127],
                "quantity": "1",
                "unit_amount": {
                    "currency_code": buyer_currency,
                    "value": f"{item_unit_amount}",
                },
            }
        ]

        # create order with PayPal
        paypal_resp = await paypal_create_order(
            amount=str(final_price),
            currency=buyer_currency,
            return_url=return_url,
            cancel_url=cancel_url,
            amount_breakdown=amount_breakdown,
            items=items,
            brand_name="Chaamo",
            user_action="PAY_NOW",
        )

        # Update order with PayPal data (checkout url + id) — always overwrite to refresh stale links
        update_resp = supabase_mutate_order(
            "update",
            {
                "gateway_order_id": paypal_resp["paypal_order_id"],
                "gateway_checkout_url": paypal_resp["paypal_checkout_url"],
            },
            {"id": order_id},
        )
        if not getattr(update_resp, "data", None) or len(update_resp.data) == 0:
            api_logger.error("Order update returned no data for order %s", order_id)
            # best-effort cleanup
            try:
                supabase_delete_order({"id": order_id})
                api_logger.info(
                    "Deleted order %s after gateway update failure", order_id
                )
            except Exception:
                api_logger.exception(
                    "Failed to delete order after gateway update failure"
                )
            raise HTTPException(
                status_code=500, detail="Failed to persist PayPal gateway fields"
            )

    except HTTPException as he:
        api_logger.warning(
            "PayPal order creation HTTP error (status=%s): %s",
            getattr(he, "status_code", None),
            getattr(he, "detail", None),
        )
        # Clean up order if it was created
        if "order_id" in locals():
            try:
                supabase_delete_order({"id": order_id})
                api_logger.info("Deleted order %s after PayPal HTTP error", order_id)
            except Exception:
                api_logger.exception("Failed to clean up order after PayPal HTTP error")
        raise
    except Exception as e:
        api_logger.exception(
            "create paypal approve url failed; cleaning up order: %s", e
        )
        # Clean up order if it was created
        if "order_id" in locals():
            try:
                supabase_delete_order({"id": order_id})
                api_logger.info("Deleted order %s after PayPal error", order_id)
            except Exception:
                api_logger.exception("Failed to clean up order after PayPal error")
        raise HTTPException(status_code=500, detail="Create paypal approve url error")

    log_success(api_logger, f"Created order {order_id} awaiting payment; checkout_url stored")
    return paypal_resp


# ===============================================================
# /paypal/order/return
# ===============================================================
async def paypal_order_return_handler(
    request: Optional[Request] = None,
    redirect: Optional[str] = None,
    token: Optional[str] = None,
    **kwargs,
) -> RedirectResponse:
    try:
        params = {}
        if request is not None:
            params = dict(request.query_params)

        merged = {}
        merged.update(params)
        merged.update(kwargs)
        api_logger.debug(
            "paypal_order_return_handler: params=%s kwargs=%s", params, kwargs
        )

        redirect_from_req = (
            merged.get("redirect")
            or merged.get("return")
            or merged.get("redirectUrl")
            or None
        )
        token_from_req = (
            merged.get("token")
            or merged.get("orderId")
            or merged.get("orderID")
            or merged.get("PayerID")
            or merged.get("PayerId")
            or merged.get("payerID")
            or None
        )

        final_redirect = (
            redirect if redirect is not None else (redirect_from_req or "/")
        )
        final_token = token if token is not None else token_from_req
        api_logger.info(
            "Return handler resolved redirect=%s token=%s", final_redirect, final_token
        )

        # If the user returned with a PayPal order token, ensure there's a pending payment row now
        if final_token:
            # Load the order by gateway_order_id
            order_row = supabase_get_order({"gateway_order_id": final_token})
            if not order_row:
                log_failure(
                    api_logger,
                    f"Order not found for gateway_order_id {final_token} on return",
                )
            else:
                order_status = (order_row.get("status") or "").lower()
                order_id_val = order_row.get("id")
                if order_status == "awaiting_payment":
                    # Check if a payment already exists for this order
                    existing_payment = supabase_get_payment({"order_id": order_id_val})
                    if existing_payment:
                        api_logger.info(
                            "Pending payment already exists on return for order %s",
                            order_id_val,
                        )
                    else:
                        # Insert a pending payment record tied to this PayPal order
                        supabase_mutate_payment(
                            "insert",
                            {
                                "order_id": order_id_val,
                                "user_id": order_row.get("buyer_id"),
                                "gateway": "paypal",
                                "gateway_order_id": final_token,
                                "amount": order_row.get("final_price"),
                                "currency": (order_row.get("currency") or "USD").upper(),
                                "status": "pending",
                            },
                        )
                        log_success(api_logger, f"Inserted pending payment on return for order {order_id_val}")
                else:
                    api_logger.info(
                        "Order %s status is %s — skipping pending payment insert on return",
                        order_id_val,
                        order_status,
                    )

        status = "success" if final_token else "cancel"
        return_params = {"status": status, "orderId": final_token or ""}

        sep = "&" if "?" in final_redirect else "?"
        target_url = f"{final_redirect}{sep}{urlencode(return_params)}"

        return RedirectResponse(url=target_url, status_code=302)
    except Exception as e:
        api_logger.exception("Error returning PayPal order (robust handler): %s", e)
        try:
            fallback_redirect = redirect or (merged.get("redirect") if merged else "/")
            return RedirectResponse(
                url=f"{fallback_redirect}{'&' if '?' in fallback_redirect else '?'}{urlencode({'status':'cancel','orderId':''})}",
                status_code=302,
            )
        except Exception:
            raise HTTPException(status_code=400, detail="Error handling PayPal return")


# ===============================================================
# /paypal/webhook/order
# ===============================================================
"""
PayPal Orders + Payments + Shippo — Webhook Flow (authoritative summary)

Events handled
- CHECKOUT.ORDER.APPROVED (Orders v2)
  For AUTHORIZE intent we do, in order:
  1) Ensure a pending payment row exists (idempotent)
  2) Authorize the PayPal order → authorization_id
  3) Capture the authorization → capture_id
  4) Finalize locally (create Shippo label, mark payment succeeded, set order to awaiting_shipment, cancel siblings)
  The capture webhook below is also accepted as a backup if immediate finalize fails.

- CHECKOUT.ORDER.COMPLETED (Orders v2)
  Some gateways emit COMPLETED in Orders v2; we mirror the same APPROVED flow (ensure pending → authorize → capture → finalize). All steps are idempotent.

- PAYMENT.CAPTURE.COMPLETED (Payments v2)
  PayPal confirms a capture. We finalize locally using only the capture id (idempotent) as a safety net.

Supabase data contract (public.orders)
- Insert at checkout creation (outside webhook):
  - id (uuid), listing_id (uuid), buyer_id (uuid), seller_id (uuid)
  - currency (varchar), final_price (numeric), seller_earnings (numeric)
  - shipping_fee (numeric, nullable), insurance_fee (numeric, nullable)
  - status = 'awaiting_payment'
  - shipping_address (jsonb)
  - shipping_rate_id (varchar, selected server-side during checkout)
  - gateway_order_id (text), gateway_checkout_url (text)
  - created_at (timestamptz auto)

- Updated during webhook finalize-after-capture (success path):
  - status → 'awaiting_shipment'
  - paid_at → now()
  - shipping_transaction_id (text) from Shippo
  - shipping_tracking_number (varchar) from Shippo
  - shipping_tracking_url (text) from Shippo
  - shipping_label_url (text) from Shippo
  - gateway_authorization_id (text) → set if we have it in the flow

- Updated during webhook on shipping failure (Shippo buy label failed):
  - status → 'cancelled'
  - order_failed_info (text) → first Shippo error message or exception text

Supabase data contract (public.payments)
- Insert pending (idempotent) in APPROVED/COMPLETED branch if none exists yet:
  - order_id (uuid), user_id (uuid → buyer_id)
  - gateway = 'paypal'
  - gateway_order_id (text)
  - gateway_transaction_id (text) → any txn id present in the webhook resource (auth or capture)
  - gateway_account_info (jsonb) → payer email and optional card tail/brand when available
  - amount (numeric) → orders.final_price
  - currency (text) → orders.currency
  - status = 'pending'

- Update to succeeded during finalize-after-capture (success path):
  - status → 'succeeded'
  - gateway_transaction_id → capture_id
  - gateway_capture_id → capture_id

- Update to failed/cancelled on error/competition:
  - If Shippo failed: status → 'failed' for the payment(s) of this order
  - If sibling orders cancelled (same listing): status → 'cancelled' for those orders' payments

Idempotency and short-circuit
- Before processing, if any payment for the order already has status = 'succeeded', the handler returns early.
- Finalization can be triggered either immediately (after capture) or by the capture webhook; both paths are idempotent.

Notes
- We currently set orders.gateway_authorization_id during finalize-after-capture when available. This avoids extra writes during APPROVED; behavior is safe and idempotent. If needed, we can set it immediately after authorization as a future enhancement.
- We rely on orders.shipping_rate_id captured at checkout creation; finalize will fail (400) if it's missing.
- All DB mutations are funneled through small helpers: paypal_mutate_* and paypal_order_*.
"""


async def paypal_webhook_order_handler(request: Request) -> dict:
    log_api_request(
        api_logger, "POST", "/paypal/webhook/order", {"headers": dict(request.headers)}
    )
    try:
        raw_body = await request.body()
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            body = {}
            api_logger.error("Failed to parse webhook body as JSON")

        event_type = body.get("event_type")
        api_logger.info("PayPal webhook event_type: %s", event_type)

        # ----------------------------------------------------------
        # CHECKOUT.ORDER.APPROVED — ensure pending payment (idempotent)
        # ----------------------------------------------------------
        if event_type == "CHECKOUT.ORDER.APPROVED":
            resource = body.get("resource")
            if not isinstance(resource, dict):
                log_failure(api_logger, "Webhook APPROVED: missing resource body")
                return {"status": "ignored"}
            paypal_order_id = resource.get("id") or resource.get("order_id")
            if not paypal_order_id:
                api_logger.error("Webhook approval event missing order id")
                return {"status": "ignored"}

            # Extract PayPal transaction id (authorization/capture) and payer/card info
            paypal_txn_id = None
            gateway_account_info = {}
            try:
                pu = resource.get("purchase_units") or []
                if pu:
                    payments = pu[0].get("payments") if isinstance(pu[0], dict) else None
                    payments = payments if isinstance(payments, dict) else {}
                    captures = payments.get("captures") or []
                    if captures and captures[0]:
                        paypal_txn_id = captures[0].get("id")
                    if not paypal_txn_id:
                        auths = payments.get("authorizations") or []
                        if auths and auths[0]:
                            paypal_txn_id = auths[0].get("id")
                payer = resource.get("payer")
                if isinstance(payer, dict):
                    payer_info = payer.get("payer_info")
                    email = payer.get("email_address")
                    if not email and isinstance(payer_info, dict):
                        email = payer_info.get("email")
                    if email:
                        gateway_account_info["email"] = email
                else:
                    log_failure(api_logger, "Webhook APPROVED: missing payer info in resource")

                payment_source = resource.get("payment_source")
                card = payment_source.get("card") if isinstance(payment_source, dict) else None
                if isinstance(card, dict):
                    cc = {}
                    last4 = (
                        card.get("last_digits")
                        or card.get("last4")
                        or card.get("number")
                    )
                    if last4:
                        cc["number"] = last4
                    ctype = card.get("brand") or card.get("type")
                    if ctype:
                        cc["type"] = ctype
                    expiry = card.get("expiry")
                    if not expiry:
                        month = card.get("expiry_month")
                        year = card.get("expiry_year")
                        if month and year:
                            expiry = f"{month}/{year}"
                    if expiry:
                        cc["expire"] = expiry
                    if cc:
                        gateway_account_info["credit_card"] = cc
            except Exception as e:
                api_logger.debug("Could not extract PayPal txn/account info: %s", e)

            # load order
            try:
                order = supabase_get_order(
                    {"gateway_order_id": paypal_order_id},
                    columns="id, listing_id, buyer_id, shipping_rate_id, shipping_transaction_id, final_price, currency, status, shipping_address",
                )
                if not order:
                    api_logger.error(
                        "Order not found for gateway_order_id %s", paypal_order_id
                    )
                    return {"status": "ok"}
            except Exception as e:
                api_logger.exception(
                    "DB error fetching order for gateway_order_id %s: %s",
                    paypal_order_id,
                    e,
                )
                raise HTTPException(
                    status_code=500, detail="DB error while fetching order"
                )

            order_id = order.get("id")

            # Step 1: idempotency check — skip only if an existing payment is already succeeded
            payments_row = None
            try:
                payments_list = supabase_get_payments(
                    {"order_id": order_id}, columns="id,status"
                )
                if payments_list:
                    statuses = [
                        (row.get("status") or "").lower() for row in payments_list
                    ]
                    if any(s == "succeeded" for s in statuses):
                        api_logger.info(
                            "Payment already succeeded for order %s — skipping further processing",
                            order_id,
                        )
                        return {"status": "ok"}
                    payments_row = payments_list[0]
            except Exception as e:
                api_logger.exception(
                    "Failed to check existing payments for order %s: %s", order_id, e
                )
                raise HTTPException(
                    status_code=500, detail="DB error while checking payments"
                )

            # Only handle if awaiting_payment
            if order.get("status") != "awaiting_payment":
                api_logger.info(
                    "Order %s status is %s — not awaiting_payment; skipping",
                    order_id,
                    order.get("status"),
                )
                return {"status": "ok"}

            # If a pending payment already exists, enrich it with gateway_account_info (email/card)
            try:
                if payments_row is not None and gateway_account_info:
                    supabase_mutate_payment(
                        "update",
                        {"gateway_account_info": gateway_account_info},
                        {"order_id": order_id, "status": "pending"},
                    )
                    api_logger.info(
                        "Updated gateway_account_info for pending payment on order %s",
                        order_id,
                    )
            except Exception:
                api_logger.exception(
                    "Failed to update gateway_account_info for pending payment on order %s",
                    order_id,
                )

            # Ensure a pending payment exists (webhook fallback for out-of-order events)
            try:
                if payments_row is None:
                    supabase_mutate_payment(
                        "insert",
                        {
                            "order_id": order_id,
                            "user_id": order.get("buyer_id"),
                            "gateway": "paypal",
                            "gateway_order_id": paypal_order_id,
                            "gateway_transaction_id": paypal_txn_id,
                            "amount": order.get("final_price"),
                            "currency": (order.get("currency") or "USD").upper(),
                            "gateway_account_info": gateway_account_info,
                            "status": "pending",
                        },
                    )
                    api_logger.info(
                        "Inserted pending payment via webhook fallback for order %s",
                        order_id,
                    )
            except Exception:
                api_logger.exception(
                    "Webhook fallback: failed to insert pending payment for order %s",
                    order_id,
                )

            # Attempt to move funds forward: authorize then capture to trigger capture webhook
            try:
                auth_resp = await paypal_authorize_order(paypal_order_id)
                # Extract authorization ids
                auth_ids: list[str] = []
                try:
                    pus = auth_resp.get("purchase_units")
                    if pus is None:
                        log_failure(api_logger, "Authorize response missing purchase_units")
                        return {"status": "ignored"}
                    if not isinstance(pus, list):
                        pus = [pus]
                    for pu in pus:
                        if not isinstance(pu, dict):
                            continue
                        payments = pu.get("payments")
                        if payments is None:
                            log_failure(api_logger, "Authorize response missing payments in purchase_units")
                            return {"status": "ignored"}
                        if not isinstance(payments, dict):
                            payments = {}
                        auths = payments.get("authorizations")
                        if auths is None:
                            auths = []
                        for a in auths:
                            if isinstance(a, dict):
                                aid = a.get("id")
                                if aid:
                                    auth_ids.append(aid)
                except Exception:
                    # Non-fatal parse issue
                    pass
                if auth_ids:
                    for aid in auth_ids:
                        try:
                            cap_resp = await paypal_capture_authorization(aid)
                            cap_id = cap_resp.get("id") if isinstance(cap_resp, dict) else None
                            api_logger.info(
                                "Triggered capture for authorization %s (capture id: %s)",
                                aid,
                                cap_id,
                            )
                        except httpx.HTTPStatusError as he:
                            api_logger.error(
                                "Capture failed for authorization %s: %s",
                                aid,
                                getattr(he.response, "text", str(he)),
                            )
                        except Exception:
                            api_logger.exception(
                                "Unexpected error capturing authorization %s",
                                aid,
                            )
                else:
                    api_logger.info(
                        "No authorization ids found in authorize response for order %s",
                        paypal_order_id,
                    )
            except httpx.HTTPStatusError as he:
                api_logger.error(
                    "Authorize call failed on APPROVED webhook for order %s: %s",
                    paypal_order_id,
                    getattr(he.response, "text", str(he)),
                )
            except Exception:
                api_logger.exception(
                    "Unexpected error while authorizing/capturing for order %s",
                    paypal_order_id,
                )

            # Early exit on APPROVED: do not finalize here; wait for PAYMENT.CAPTURE.COMPLETED
            return {"status": "ok"}

        # ----------------------------------------------------------
        # PAYMENT.CAPTURE.COMPLETED — finalize on payment capture completed (Payments v2) (idempotent)
        # ----------------------------------------------------------
        if event_type == "PAYMENT.CAPTURE.COMPLETED":
            resource = body.get("resource")
            if not isinstance(resource, dict):
                log_failure(api_logger, "Capture webhook: missing resource body")
                return {"status": "ignored"}
            # Resolve order id from capture resource
            capture_id = resource.get("id")
            supplementary_data = resource.get("supplementary_data")
            related_ids = (
                supplementary_data.get("related_ids")
                if isinstance(supplementary_data, dict)
                else None
            )
            paypal_order_id = related_ids.get("order_id") if isinstance(related_ids, dict) else None
            related_auth_id = related_ids.get("authorization_id") if isinstance(related_ids, dict) else None
            if not paypal_order_id:
                try:
                    for l in resource.get("links", []) or []:
                        if l.get("rel") == "up" and "/checkout/orders/" in (
                            l.get("href") or ""
                        ):
                            paypal_order_id = (
                                l.get("href").rstrip("/").split("/") or []
                            )[-1]
                            break
                except Exception:
                    pass
            if not paypal_order_id:
                api_logger.error(
                    "Capture completed but missing related order id; capture_id=%s",
                    capture_id,
                )
                return {"status": "ignored"}

            # load order
            try:
                order = supabase_get_order(
                    {"gateway_order_id": paypal_order_id},
                    columns="id, listing_id, buyer_id, shipping_rate_id, shipping_transaction_id, final_price, currency, status, shipping_address",
                )
                if not order:
                    api_logger.error(
                        "Order not found for gateway_order_id %s", paypal_order_id
                    )
                    return {"status": "ok"}
            except Exception as e:
                api_logger.exception(
                    "DB error fetching order for gateway_order_id %s: %s",
                    paypal_order_id,
                    e,
                )
                raise HTTPException(
                    status_code=500, detail="DB error while fetching order"
                )

            order_id = order.get("id")
            # for downstream updates
            first_auth_id = related_auth_id
            last_capture_id = capture_id

            # Build gateway_account_info reliably from PayPal order details
            gateway_account_info = {}
            try:
                pp_order = await paypal_get_order(paypal_order_id)
                if not isinstance(pp_order, dict):
                    log_failure(api_logger, "Capture webhook: PayPal order response is not a JSON object")
                else:
                    payer = pp_order.get("payer")
                    if isinstance(payer, dict):
                        payer_info = payer.get("payer_info")
                        email = payer.get("email_address")
                        if not email and isinstance(payer_info, dict):
                            email = payer_info.get("email")
                        if email:
                            gateway_account_info["email"] = email
                    else:
                        log_failure(api_logger, "Capture webhook: missing payer info in PayPal order")

                    payment_source = pp_order.get("payment_source")
                    card = payment_source.get("card") if isinstance(payment_source, dict) else None
                    if isinstance(card, dict):
                        cc = {}
                        last4 = (
                            card.get("last_digits")
                            or card.get("last4")
                            or card.get("number")
                        )
                        if last4:
                            cc["number"] = last4
                        ctype = card.get("brand") or card.get("type")
                        if ctype:
                            cc["type"] = ctype
                        expiry = card.get("expiry") or None
                        if not expiry:
                            month = card.get("expiry_month")
                            year = card.get("expiry_year")
                            if month and year:
                                expiry = f"{month}/{year}"
                        if expiry:
                            cc["expire"] = expiry
                        if cc:
                            gateway_account_info["credit_card"] = cc
            except Exception:
                api_logger.debug(
                    "Could not fetch/parse PayPal order details for gateway_account_info on capture"
                )

            # Step 1: idempotency check — skip only if an existing payment is already succeeded
            payments_row = None
            try:
                payments_list = supabase_get_payments(
                    {"order_id": order_id}, columns="id,status"
                )
                if payments_list:
                    # If any succeeded, short-circuit; otherwise keep a reference and continue to process
                    statuses = [
                        (row.get("status") or "").lower() for row in payments_list
                    ]
                    if any(s == "succeeded" for s in statuses):
                        api_logger.info(
                            "Payment already succeeded for order %s — skipping further processing",
                            order_id,
                        )
                        return {"status": "ok"}
                    payments_row = payments_list[0]
            except Exception as e:
                api_logger.exception(
                    "Failed to check existing payments for order %s: %s", order_id, e
                )
                raise HTTPException(
                    status_code=500, detail="DB error while checking payments"
                )

            # Ensure a pending payment exists (webhook fallback for out-of-order events)
            try:
                if payments_row is None:
                    supabase_mutate_payment(
                        "insert",
                        {
                            "order_id": order_id,
                            "user_id": order.get("buyer_id"),
                            "gateway": "paypal",
                            "gateway_order_id": paypal_order_id,
                            "gateway_transaction_id": capture_id,
                            "amount": order.get("final_price"),
                            "currency": (order.get("currency") or "USD").upper(),
                            "gateway_account_info": gateway_account_info or None,
                            "status": "pending",
                        },
                    )
                    api_logger.info(
                        "Inserted pending payment via capture webhook for order %s",
                        order_id,
                    )
            except Exception:
                api_logger.exception(
                    "Capture webhook: failed to insert pending payment for order %s",
                    order_id,
                )

            # Create shipping transaction directly with stored rate
            try:
                final_rate_id = order.get("shipping_rate_id")
                if not final_rate_id:
                    raise HTTPException(
                        status_code=400, detail="Order missing shipping_rate_id"
                    )

                transaction = shippo_sdk.transactions.create(
                    components.TransactionCreateRequest(
                        rate=final_rate_id,
                        label_file_type="PDF",
                        async_=False,
                        metadata=f"order:{order_id} buyer:{order.get('buyer_id')}",
                    )
                )

                api_logger.debug("Shippo transaction: %s", transaction)

                if not transaction.object_id:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Create shippo transaction failed: {transaction}",
                    )

                # Check transaction status
                tx_status = getattr(transaction, "status", "")
                if tx_status != "SUCCESS":
                    messages = getattr(transaction, "messages", [])
                    error_msg = (
                        messages[0].text if messages else "Shipping service error"
                    )

                    # Mark order as cancelled and record failure info
                    try:
                        supabase_mutate_order(
                            "update",
                            {"status": "cancelled", "order_failed_info": error_msg},
                            {"id": order_id},
                        )
                        api_logger.info(
                            "Marked order %s as cancelled with failure info: %s",
                            order_id,
                            error_msg,
                        )
                    except Exception:
                        api_logger.exception(
                            "Failed to update order %s as cancelled after Shippo failure; continuing to void PayPal",
                            order_id,
                        )

                    # Mark related payment(s) as failed (include gateway_account_info when available)
                    try:
                        fail_payload = {"status": "failed"}
                        if gateway_account_info:
                            fail_payload["gateway_account_info"] = gateway_account_info
                        supabase_mutate_payment(
                            "update", fail_payload, {"order_id": order_id}
                        )
                        api_logger.info(
                            "Marked payments for order %s as failed after Shippo failure",
                            order_id,
                        )
                    except Exception:
                        api_logger.exception(
                            "Failed to mark payments failed for order %s after Shippo failure",
                            order_id,
                        )

                    # Step 4: On Shippo failure, void PayPal checkout/authorization (best-effort)
                    try:
                        api_logger.info(
                            "Voiding PayPal checkout for order %s", order_id
                        )
                        await paypal_void_checkout(paypal_order_id)
                    except Exception as ve:
                        api_logger.exception(
                            "Failed to void PayPal checkout for order %s: %s",
                            order_id,
                            ve,
                        )
                    try:
                        # If we already parsed authorization ids from resource above, void them explicitly
                        pus_void = resource.get("purchase_units") if isinstance(resource, dict) else []
                        if pus_void and isinstance(pus_void, list) and pus_void:
                            p0 = pus_void[0]
                            payments_void = p0.get("payments") if isinstance(p0, dict) else None
                            if isinstance(payments_void, dict):
                                auths_void = payments_void.get("authorizations") or []
                                for a in auths_void:
                                    aid = a.get("id") if isinstance(a, dict) else None
                                    if aid:
                                        try:
                                            await paypal_void_authorization(aid)
                                        except Exception:
                                            api_logger.exception(
                                                "Failed to void authorization %s", aid
                                            )
                    except Exception:
                        # ignore parsing issues
                        pass

                    # Build structured error detail for client and retries
                    _emsg = (error_msg or "").lower()
                    if ("multiple records" in _emsg) or ("multiple addresses" in _emsg):
                        detail_payload = {
                            "error": "Address validation failed",
                            "code": "ADDRESS_MULTIPLE_MATCHES",
                            "message": "Shipping address matched multiple records. Please provide more specific details.",
                            "suggestions": [
                                "Add apartment/suite number if applicable",
                                "Double-check street name spelling",
                                "Verify city and state/province",
                                "Ensure postal/ZIP code is complete and correct",
                            ],
                        }
                    elif ("failed_address_validation" in _emsg) or (
                        "address" in _emsg and "valid" in _emsg
                    ):
                        detail_payload = {
                            "error": "Address validation failed",
                            "code": "ADDRESS_VALIDATION_FAILED",
                            "message": "The shipping address could not be verified. Please check the address and try again.",
                        }
                    else:
                        detail_payload = {
                            "error": "Shipping service error",
                            "code": "SHIPPING_TRANSACTION_FAILED",
                            "message": error_msg,
                        }

                    raise HTTPException(status_code=400, detail=detail_payload)
            except HTTPException:
                raise
            except Exception as e:
                error_msg = str(e)
                api_logger.error(f"Shippo transaction failed: {error_msg}")

                # Best-effort: mark order cancelled and record failure info
                try:
                    supabase_mutate_order(
                        "update",
                        {"status": "cancelled", "order_failed_info": error_msg},
                        {"id": order_id},
                    )
                except Exception:
                    api_logger.exception(
                        "Failed to update order %s as cancelled after generic Shippo failure",
                        order_id,
                    )

                # Mark payment(s) failed as well (include gateway_account_info when available)
                try:
                    fail_payload = {"status": "failed"}
                    if gateway_account_info:
                        fail_payload["gateway_account_info"] = gateway_account_info
                    supabase_mutate_payment(
                        "update", fail_payload, {"order_id": order_id}
                    )
                except Exception:
                    api_logger.exception(
                        "Failed to mark payments failed for order %s after generic Shippo failure",
                        order_id,
                    )

                # Ensure webhook reflects failure so provider may retry – with structured detail
                _emsg = (error_msg or "").lower()
                if ("multiple records" in _emsg) or ("multiple addresses" in _emsg):
                    detail_payload = {
                        "error": "Address validation failed",
                        "code": "ADDRESS_MULTIPLE_MATCHES",
                        "message": "Shipping address matched multiple records. Please provide more specific details.",
                        "suggestions": [
                            "Add apartment/suite number if applicable",
                            "Double-check street name spelling",
                            "Verify city and state/province",
                            "Ensure postal/ZIP code is complete and correct",
                        ],
                    }
                elif ("failed_address_validation" in _emsg) or (
                    "address" in _emsg and "valid" in _emsg
                ):
                    detail_payload = {
                        "error": "Address validation failed",
                        "code": "ADDRESS_VALIDATION_FAILED",
                        "message": "The shipping address could not be verified. Please check the address and try again.",
                    }
                else:
                    detail_payload = {
                        "error": "Shipping service error",
                        "code": "SHIPPING_TRANSACTION_FAILED",
                        "message": error_msg,
                    }

                raise HTTPException(status_code=400, detail=detail_payload)

            # Payment already captured by PayPal; mark payment succeeded using capture_id
            try:
                update_payload = {
                    "status": "succeeded",
                    "gateway_transaction_id": capture_id,
                    "gateway_capture_id": capture_id,
                }
                if gateway_account_info:
                    update_payload["gateway_account_info"] = gateway_account_info
                upd_resp = supabase_mutate_payment(
                    "update", update_payload, {"order_id": order_id}
                )
                updated_count = len(getattr(upd_resp, "data", []) or [])
                api_logger.info(
                    "Payments update for order %s affected %s row(s)",
                    order_id,
                    updated_count,
                )
                if updated_count == 0:
                    # Fallback: update the most recent payment row by id
                    try:
                        existing = supabase_get_payments(
                            {"order_id": order_id}, columns="id,created_at"
                        )
                        if existing:
                            # choose latest by created_at when available
                            latest = max(
                                existing,
                                key=lambda r: (r.get("created_at") or ""),
                            )
                            pid = latest.get("id")
                            if pid:
                                supabase_mutate_payment(
                                    "update", update_payload, {"id": pid}
                                )
                                api_logger.info(
                                    "Fallback payment update by id %s for order %s applied",
                                    pid,
                                    order_id,
                                )
                    except Exception:
                        api_logger.exception(
                            "Fallback payment update failed for order %s",
                            order_id,
                        )
            except Exception as e:
                api_logger.exception(
                    "Failed to mark payment succeeded for order %s: %s", order_id, e
                )

            # 3) Update order status to awaiting_shipment (also set paid_at and gateway_authorization_id if available)
            try:
                update_data = {
                    "status": "awaiting_shipment",
                    "shipping_transaction_id": transaction.object_id,
                    "shipping_tracking_number": transaction.tracking_number,
                    "shipping_tracking_url": transaction.tracking_url_provider,
                    "shipping_label_url": transaction.label_url,
                    "paid_at": datetime.now().isoformat(),
                }
                if first_auth_id:
                    update_data["gateway_authorization_id"] = first_auth_id
                supabase_mutate_order("update", update_data, {"id": order_id})
            except Exception as e:
                api_logger.exception(
                    "Failed to update order %s status: %s", order_id, e
                )
                raise HTTPException(
                    status_code=500, detail="Failed to update order status"
                )

            # 4) Cancel sibling orders for the same listing and their payments (now that this order succeeded)
            try:
                listing_id = order.get("listing_id")
                if listing_id:
                    # Cancel other awaiting_payment orders for same listing
                    siblings = supabase_get_orders(
                        {
                            "listing_id": listing_id,
                            "status": "awaiting_payment",
                            "id": {"neq": order_id},
                        },
                        columns="id",
                    )
                    other_ids = [
                        row.get("id") for row in (siblings or []) if row.get("id")
                    ]
                    if other_ids:
                        supabase_mutate_order(
                            "update", {"status": "cancelled"}, {"id": {"in": other_ids}}
                        )
                        api_logger.info(
                            "Cancelled %d sibling order(s) for listing %s",
                            len(other_ids),
                            listing_id,
                        )
                        # Cancel payments for those orders (if not already succeeded)
                        try:
                            supabase_mutate_payment(
                                "update",
                                {"status": "cancelled"},
                                {"order_id": {"in": other_ids}},
                            )
                            api_logger.info(
                                "Marked payments as cancelled for sibling orders: %s",
                                other_ids,
                            )
                        except Exception:
                            api_logger.exception(
                                "Failed to cancel payments for sibling orders: %s",
                                other_ids,
                            )
            except Exception:
                api_logger.exception(
                    "Sibling order/payment cancellation step failed (non-fatal)"
                )

            # 5) Simple success response

            return {"status": "ok"}

        # Skip other webhook types for now
        api_logger.info("Webhook event_type %s ignored", event_type)
        return {"status": "ignored"}

    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception(
            "Unhandled exception in paypal_webhook_order_handler: %s", e
        )
        raise HTTPException(status_code=500, detail="Unhandled webhook error")


# ===============================================================
# /paypal/cancel
# ===============================================================
async def paypal_cancel_handler(
    redirect: str,
    token: Optional[str],
) -> RedirectResponse:
    log_api_request(api_logger, "GET", "/paypal/cancel", {"token": token})
    api_logger.info(f"PayPal payment cancelled: {token or 'no-token'}")

    # Best-effort: mark the local order as cancelled and try voiding any authorization
    try:
        if token:
            try:
                supabase_mutate_order(
                    "update", {"status": "cancelled"}, {"gateway_order_id": token}
                )
                api_logger.info(
                    "Marked order with gateway_order_id %s as cancelled", token
                )
            except Exception:
                api_logger.exception(
                    "Failed to mark order cancelled for token %s", token
                )

            try:
                await paypal_void_checkout(token)
            except Exception:
                api_logger.exception(
                    "Failed to void checkout for token %s (non-fatal)", token
                )
    except Exception as e:
        api_logger.exception("Error handling PayPal cancel: %s", e)

    params = urlencode({"status": "cancel", "orderId": token or ""})
    return RedirectResponse(
        url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
    )

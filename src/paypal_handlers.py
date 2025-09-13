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
from src.utils.supabase import supabase
from src.utils.currency import format_price
from src.utils.shippo import (
    shippo_get_rate_details,
    shippo_validate_address,
    shippo_build_shipment_and_select_rate,
)
from src.utils.paypal import (
    get_access_token,
    get_base_url,
    get_api_base,
    paypal_create_subscription,
    paypal_create_order,
    paypal_void_checkout,
    paypal_void_authorization,
    paypal_authorize_order,
    paypal_capture_authorization,
    paypal_capture_order,
    paypal_cancel_subscription,
    paypal_fetch_plan_fixed_price,
    paypal_fetch_price_by_subscription_id,
    paypal_compute_adjusted_start,
    paypal_ensure_linked_pending_payment,
    paypal_parse_iso_dt,
)
from src.utils.logger import (
    api_logger,
    log_api_request,
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
        # 1. ambil paypal_plan_id
        plan_resp = (
            supabase.table("membership_plans")
            .select("id, paypal_plan_id")
            .eq("id", plan_id)
            .execute()
        )
        if not plan_resp.data:
            raise HTTPException(status_code=404, detail="Plan not found")
        row = plan_resp.data[0]
        paypal_plan_id = row.get("paypal_plan_id")
        if not paypal_plan_id:
            raise HTTPException(
                status_code=400, detail="Plan does not have PayPal plan id"
            )

        # Guard: prevent duplicate active subscription period for same user/plan
        try:
            now_dt = datetime.now()
            existing_active = (
                supabase.table("subscriptions")
                .select("id, start_date, end_date, status")
                .eq("user_id", user_id)
                .eq("plan_id", plan_id)
                .eq("status", "active")
                .execute()
            )
            if getattr(existing_active, "data", None):
                for row in existing_active.data:
                    start_dt = paypal_parse_iso_dt(row.get("start_date"))
                    end_dt = paypal_parse_iso_dt(row.get("end_date"))
                    # Block only if now is within [start_date, end_date)
                    if start_dt and end_dt and start_dt <= now_dt < end_dt:
                        # Already in an active period; block new purchase
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
        except Exception:
            api_logger.exception("Active subscription guard check failed (continuing)")

        # 2. build return/cancel urls (include internal plan_id so return handler can use it)
        base = get_base_url(request)
        return_url = f"{base}/api/v1/paypal/subscription/return?{urlencode({'redirect': redirect, 'user_id': user_id, 'plan_id': plan_id})}"
        cancel_url = f"{base}/api/v1/paypal/cancel?{urlencode({'redirect': redirect})}"

        subscription_resp = await paypal_create_subscription(
            plan_id=paypal_plan_id,
            return_url=return_url,
            cancel_url=cancel_url,
            user_details=None,
        )

        approval_url = subscription_resp.get("approval_url")
        subscription_id = subscription_resp.get("subscription_id")
        if not approval_url or not subscription_id:
            api_logger.error(
                "PayPal create-subscription missing approval_url or subscription_id"
            )
            raise HTTPException(status_code=500, detail="Payment provider error")

        # 3. Persist or reuse a single 'pending' subscription per user/plan
        try:
            # Reuse or create: prefer updating an existing pending row for this user+plan
            pending_res = (
                supabase.table("subscriptions")
                .select("id, payment_id, paypal_subscription_id, status")
                .eq("user_id", user_id)
                .eq("plan_id", plan_id)
                .eq("status", "pending")
                .limit(1)
                .execute()
            )
            primary_id = None
            if getattr(pending_res, "data", None):
                row0 = pending_res.data[0]
                primary_id = row0.get("id")
                # Cancel any linked pending payment and unlink
                try:
                    pid = row0.get("payment_id")
                    if pid:
                        supabase.table("payments").update({"status": "cancelled"}).eq(
                            "id", pid
                        ).execute()
                        supabase.table("subscriptions").update({"payment_id": None}).eq(
                            "id", primary_id
                        ).execute()
                except Exception:
                    api_logger.exception(
                        "Failed to cancel/unlink previous pending payment for subscription reuse"
                    )
                # Update existing pending row with the latest PayPal subscription id
                supabase.table("subscriptions").update(
                    {"paypal_subscription_id": subscription_id}
                ).eq("id", primary_id).execute()
                # Cancel any other pending duplicates for safety
                try:
                    supabase.table("subscriptions").update({"status": "cancelled"}).eq(
                        "user_id", user_id
                    ).eq("plan_id", plan_id).eq("status", "pending").neq(
                        "id", primary_id
                    ).execute()
                except Exception:
                    api_logger.exception(
                        "Failed to cancel duplicate pending subscriptions for user/plan"
                    )
            else:
                ins = (
                    supabase.table("subscriptions")
                    .insert(
                        {
                            "user_id": user_id,
                            "plan_id": plan_id,
                            "status": "pending",
                            "paypal_subscription_id": subscription_id,
                            "start_date": None,
                            "end_date": None,
                        }
                    )
                    .execute()
                )
                primary_id = ins.data[0]["id"] if getattr(ins, "data", None) else None
        except Exception:
            api_logger.exception(
                "Failed to persist/reuse pending subscription; continuing to redirect to PayPal"
            )

        # 4. redirect user to PayPal approval page
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
    subscription_id: Optional[str] = None,
) -> RedirectResponse:
    log_api_request(
        api_logger,
        "GET",
        "/paypal/subscription/cancel",
        {"user_id": user_id, "plan_id": plan_id, "subscription_id": subscription_id},
    )
    try:
        sub_row = None
        # Resolve local subscription row
        if subscription_id:
            res = (
                supabase.table("subscriptions")
                .select("id, paypal_subscription_id")
                .eq("paypal_subscription_id", subscription_id)
                .limit(1)
                .execute()
            )
            if getattr(res, "data", None):
                sub_row = res.data[0]
        elif user_id and plan_id:
            # Prefer the latest pending/active row for this user+plan
            res = (
                supabase.table("subscriptions")
                .select("id, paypal_subscription_id")
                .eq("user_id", user_id)
                .eq("plan_id", plan_id)
                .in_("status", ["pending", "active"])
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if getattr(res, "data", None):
                sub_row = res.data[0]

        # Attempt remote cancellation if we have a PayPal id
        pp_id = subscription_id or (
            sub_row.get("paypal_subscription_id") if sub_row else None
        )
        if pp_id:
            try:
                await paypal_cancel_subscription(
                    pp_id, reason="User cancelled at approval"
                )
            except Exception:
                api_logger.exception(
                    "PayPal remote subscription cancel failed (non-fatal)"
                )

        # Update local status to cancelled
        try:
            if sub_row and sub_row.get("id"):
                supabase.table("subscriptions").update({"status": "cancelled"}).eq(
                    "id", sub_row.get("id")
                ).execute()
            elif pp_id:
                supabase.table("subscriptions").update({"status": "cancelled"}).eq(
                    "paypal_subscription_id", pp_id
                ).execute()
        except Exception:
            api_logger.exception("Failed to update local subscription to cancelled")

        params = urlencode(
            {
                "status": "cancel",
                "subscriptionId": pp_id or "",
            }
        )
        return RedirectResponse(
            url=f"{redirect}{'&' if '?' in redirect else '?'}{params}",
            status_code=302,
        )
    except Exception as e:
        api_logger.exception("Error handling PayPal subscription cancel: %s", e)
        params = urlencode(
            {"status": "cancel", "subscriptionId": subscription_id or ""}
        )
        return RedirectResponse(
            url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
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

        token = await get_access_token()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{get_api_base().rstrip('/')}/v1/billing/subscriptions/{subscription_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()

        # We do NOT activate locally here; leave subscription as pending until webhook.
        # Ensure internal subscription mapping exists (idempotent) without creating duplicates
        sub_id = None
        existing = (
            supabase.table("subscriptions")
            .select("id, user_id, plan_id, status, payment_id")
            .eq("paypal_subscription_id", subscription_id)
            .limit(1)
            .execute()
        )
        if getattr(existing, "data", None):
            sub_row = existing.data[0]
            sub_id = sub_row["id"]
            try:
                supabase.table("subscriptions").update(
                    {
                        "status": sub_row.get("status") or "pending",
                    }
                ).eq("id", sub_id).execute()
            except Exception:
                api_logger.exception("Failed to keep subscription pending on return")
        else:
            # Reuse any existing pending for this user+plan to avoid duplicates
            try:
                fallback = (
                    supabase.table("subscriptions")
                    .select("id")
                    .eq("user_id", user_id)
                    .eq("plan_id", plan_id)
                    .eq("status", "pending")
                    .limit(1)
                    .execute()
                )
                if getattr(fallback, "data", None):
                    sub_id = fallback.data[0]["id"]
                    supabase.table("subscriptions").update(
                        {
                            "paypal_subscription_id": subscription_id,
                        }
                    ).eq("id", sub_id).execute()
                else:
                    insert_resp = (
                        supabase.table("subscriptions")
                        .insert(
                            {
                                "user_id": user_id,
                                "plan_id": plan_id,
                                "status": "pending",
                                "paypal_subscription_id": subscription_id,
                                "start_date": None,
                                "end_date": None,
                            }
                        )
                        .execute()
                    )
                    sub_id = insert_resp.data[0]["id"]
            except Exception:
                api_logger.exception("Failed to upsert pending subscription on return")

        # Insert a pending payment and link it to subscription via subscriptions.payment_id (idempotent)
        try:
            if sub_id:
                # Parse subscriber email for gateway_account_info if available
                subscriber = data.get("subscriber") or {}
                gai = {}
                email = subscriber.get("email_address")
                if email:
                    gai["email"] = email
                plan_amount, plan_currency = await paypal_fetch_plan_fixed_price(
                    data.get("plan_id")
                )
                await paypal_ensure_linked_pending_payment(
                    sub_id=sub_id,
                    user_id=user_id,
                    gateway_order_id=subscription_id,
                    amount=plan_amount,
                    currency=plan_currency,
                    gateway_account_info=gai,
                )
        except Exception:
            api_logger.exception(
                "Failed to insert pending subscription payment on return"
            )

        # Redirect to client
        params = urlencode({"status": "success", "subscriptionId": subscription_id})
        return RedirectResponse(
            f"{redirect}{'&' if '?' in redirect else '?'}{params}",
            status_code=302,
        )

    except Exception as e:
        api_logger.exception(
            "Error checking PayPal subscription %s: %s", subscription_id, e
        )
        redirect_out = redirect or "/"
        params = urlencode(
            {
                "status": "error",
                "subscriptionId": subscription_id or "",
                "reason": "internal_error",
            }
        )
        return RedirectResponse(
            f"{redirect_out}{'&' if '?' in redirect_out else '?'}{params}",
            status_code=302,
        )


# ===============================================================
# /paypal/webhook/subscription
# ===============================================================
async def paypal_webhook_subscription_handler(request: Request) -> dict:
    try:
        body = await request.json()
        event_type = body.get("event_type")
        resource = body.get("resource", {})
        log_api_request(
            api_logger,
            "POST",
            "/paypal/webhook/subscription triggered",
            {"event_type": event_type},
        )

        subscription_id = resource.get("id")
        if not subscription_id:
            api_logger.error("PayPal webhook missing subscription ID")
            return {"status": "error", "message": "Missing subscription ID"}

        # -------------------------
        # Subscription Activated
        # -------------------------
        if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
            # Activate locally using plan duration (start now, end now + subscription_days)
            sub_row = (
                supabase.table("subscriptions")
                .select("id, user_id, plan_id, payment_id")
                .eq("paypal_subscription_id", subscription_id)
                .limit(1)
                .execute()
            )
            if getattr(sub_row, "data", None):
                sub = sub_row.data[0]
                plan_days = 30
                try:
                    plan_resp = (
                        supabase.table("membership_plans")
                        .select("subscription_days")
                        .eq("id", sub.get("plan_id"))
                        .limit(1)
                        .execute()
                    )
                    if getattr(plan_resp, "data", None):
                        plan_days = int(
                            plan_resp.data[0].get("subscription_days") or 30
                        )
                except Exception:
                    api_logger.exception(
                        "Failed to read plan subscription_days; defaulting to 30"
                    )
                adj_start = paypal_compute_adjusted_start(
                    sub.get("user_id"), sub.get("plan_id"), sub.get("id")
                )
                end_dt = adj_start + timedelta(days=plan_days)
                supabase.table("subscriptions").update(
                    {
                        "status": "active",
                        "start_date": adj_start.isoformat(),
                        "end_date": end_dt.isoformat(),
                    }
                ).eq("id", sub.get("id")).execute()

                # Ensure a pending payment exists and is linked (webhook fallback)
                try:
                    if not sub.get("payment_id"):
                        plan_amount, plan_currency = (
                            await paypal_fetch_price_by_subscription_id(subscription_id)
                        )
                        await paypal_ensure_linked_pending_payment(
                            sub_id=sub.get("id"),
                            user_id=sub.get("user_id"),
                            gateway_order_id=subscription_id,
                            amount=plan_amount,
                            currency=plan_currency,
                        )
                except Exception:
                    api_logger.exception(
                        "Failed to ensure pending payment on ACTIVATED event"
                    )
            else:
                api_logger.warning(
                    f"Subscription {subscription_id} ACTIVATED but not found in DB. Skipping activation."
                )

        # -------------------------
        # Subscription Cancelled
        # -------------------------
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            supabase.table("subscriptions").update({"status": "cancelled"}).eq(
                "paypal_subscription_id", subscription_id
            ).execute()

        # -------------------------
        # Payment Completed
        # -------------------------
        elif event_type == "PAYMENT.SALE.COMPLETED":
            paypal_sub_id = resource.get("billing_agreement_id")
            txn_id = resource.get("id")
            amt_obj = resource.get("amount", {}) or {}
            if not paypal_sub_id:
                api_logger.error(
                    "Missing billing_agreement_id in PAYMENT.SALE.COMPLETED"
                )
            else:
                sub_res = (
                    supabase.table("subscriptions")
                    .select(
                        "id, user_id, plan_id, status, start_date, end_date, payment_id"
                    )
                    .eq("paypal_subscription_id", paypal_sub_id)
                    .limit(1)
                    .execute()
                )
                if getattr(sub_res, "data", None):
                    sub = sub_res.data[0]
                    # One payment per subscription cycle:
                    # 1) Prefer updating the linked pending payment via subscriptions.payment_id
                    # 2) Else, idempotent upsert by gateway_transaction_id and link it
                    try:
                        updated = False
                        pid = sub.get("payment_id")
                        if pid:
                            supabase.table("payments").update(
                                {
                                    "status": "succeeded",
                                    "gateway_transaction_id": txn_id,
                                    "gateway_capture_id": txn_id,
                                    "gateway_order_id": paypal_sub_id,
                                    "amount": amt_obj.get("total"),
                                    "currency": amt_obj.get("currency"),
                                }
                            ).eq("id", pid).execute()
                            updated = True
                        if not updated:
                            # Idempotent by gateway_transaction_id
                            existing_txn = (
                                supabase.table("payments")
                                .select("id")
                                .eq("gateway_transaction_id", txn_id)
                                .limit(1)
                                .execute()
                            )
                            if getattr(existing_txn, "data", None):
                                pid = existing_txn.data[0]["id"]
                                supabase.table("payments").update(
                                    {
                                        "user_id": sub.get("user_id"),
                                        "status": "succeeded",
                                        "gateway_capture_id": txn_id,
                                        "gateway_order_id": paypal_sub_id,
                                        "amount": amt_obj.get("total"),
                                        "currency": amt_obj.get("currency"),
                                    }
                                ).eq("id", pid).execute()
                                # Link payment to subscription if not linked yet
                                if not sub.get("payment_id"):
                                    supabase.table("subscriptions").update(
                                        {"payment_id": pid}
                                    ).eq("id", sub.get("id")).execute()
                            else:
                                ins2 = (
                                    supabase.table("payments")
                                    .insert(
                                        {
                                            "user_id": sub.get("user_id"),
                                            "gateway": "paypal",
                                            "gateway_transaction_id": txn_id,
                                            "gateway_capture_id": txn_id,
                                            "gateway_order_id": paypal_sub_id,
                                            "amount": amt_obj.get("total"),
                                            "currency": amt_obj.get("currency"),
                                            "status": "succeeded",
                                        }
                                    )
                                    .execute()
                                )
                                if getattr(ins2, "data", None):
                                    new_pid = ins2.data[0].get("id")
                                    supabase.table("subscriptions").update(
                                        {"payment_id": new_pid}
                                    ).eq("id", sub.get("id")).execute()
                    except Exception:
                        api_logger.exception(
                            "Failed to upsert succeeded subscription payment"
                        )

                    # Ensure subscription is active with correct dates (fallback if ACTIVATED not processed)
                    try:
                        if (sub.get("status") or "").lower() != "active":
                            plan_days = 30
                            plan_resp = (
                                supabase.table("membership_plans")
                                .select("subscription_days")
                                .eq("id", sub.get("plan_id"))
                                .limit(1)
                                .execute()
                            )
                            if getattr(plan_resp, "data", None):
                                plan_days = int(
                                    plan_resp.data[0].get("subscription_days") or 30
                                )
                            adj_start = paypal_compute_adjusted_start(
                                sub.get("user_id"), sub.get("plan_id"), sub.get("id")
                            )
                            end_dt = adj_start + timedelta(days=plan_days)
                            supabase.table("subscriptions").update(
                                {
                                    "status": "active",
                                    "start_date": adj_start.isoformat(),
                                    "end_date": end_dt.isoformat(),
                                }
                            ).eq("id", sub.get("id")).execute()
                    except Exception:
                        api_logger.exception(
                            "Failed to set subscription active on PAYMENT.SALE.COMPLETED fallback"
                        )
                else:
                    api_logger.error(
                        f"No subscription found for PayPal ID {paypal_sub_id}"
                    )

        # -------------------------
        # Subscription Updated
        # -------------------------
        elif event_type == "BILLING.SUBSCRIPTION.UPDATED":
            supabase.table("subscriptions").update(
                {
                    "status": resource["status"].lower(),
                    "end_date": resource.get("billing_info", {}).get(
                        "next_billing_time"
                    ),
                }
            ).eq("paypal_subscription_id", subscription_id).execute()

        api_logger.info(f"Handled PayPal webhook: {event_type} for {subscription_id}")
        return {"status": "ok"}

    except Exception as e:
        api_logger.exception("PayPal webhook error: %s", e)
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
    try:
        listing_resp = (
            supabase.table("vw_listing_cards")
            .select("*")
            .eq("id", listing_id)
            .execute()
        )
        if not getattr(listing_resp, "data", None):
            api_logger.warning("Listing not found: %s", listing_id)
            raise HTTPException(status_code=404, detail="Listing not found")
    except HTTPException:
        # will be logged by safe_handler but also log here for immediate context
        api_logger.warning("Raising HTTPException for missing listing: %s", listing_id)
        raise
    except Exception as e:
        api_logger.exception("Listing lookup failed: %s", e)
        raise HTTPException(status_code=500, detail="Listing lookup failed")

    listing = listing_resp.data[0]
    selected_listing_price = (
        listing.get("highest_bid_price", 0)
        if listing.get("listing_type") == "auction"
        else listing.get("start_price", 0)
    )

    # -------------------------
    # Sold guard: if listing already has a paid order, stop and clean pending ones
    # -------------------------
    try:
        sold_statuses = {"awaiting_shipment", "shipped", "delivered", "completed"}
        orders_res = (
            supabase.table("orders")
            .select("id, status")
            .eq("listing_id", listing_id)
            .execute()
        )
        has_sold = False
        if getattr(orders_res, "data", None):
            for row in orders_res.data:
                st = (row.get("status") or "").lower()
                if st in sold_statuses:
                    has_sold = True
                    break
        if has_sold:
            # Remove any stale awaiting_payment orders for this listing
            try:
                supabase.table("orders").delete().eq("listing_id", listing_id).eq(
                    "status", "awaiting_payment"
                ).execute()
            except Exception:
                api_logger.exception(
                    "Failed to cleanup pending orders for sold listing %s (continuing)",
                    listing_id,
                )
            return {"status": "sold"}
    except Exception as e:
        api_logger.exception("Sold guard check failed: %s", e)
        # Continue flow; better to proceed than hard-fail here

    # -------------------------
    # Load Buyer Currency
    # -------------------------
    try:
        buyer_resp = (
            supabase.table("profiles").select("currency").eq("id", buyer_id).execute()
        )
        if not getattr(buyer_resp, "data", None) or len(buyer_resp.data) == 0:
            api_logger.error("Buyer profile not found for id: %s", buyer_id)
            raise HTTPException(status_code=500, detail="Buyer currency lookup failed")
        buyer = buyer_resp.data[0]
        buyer_currency = (
            buyer.get("currency")
            or insurance_currency
            or selected_rate_currency
            or "USD"
        ).upper()
    except HTTPException:
        # already logged above; re-raise
        raise
    except Exception as e:
        api_logger.exception("Buyer currency lookup failed unexpectedly: %s", e)
        raise HTTPException(status_code=500, detail="Buyer currency lookup failed")

    # -------------------------
    # Check for existing pending order (we will UPDATE it instead of returning stale gateway URLs)
    # -------------------------
    reuse_order = False
    order_id = None
    old_gateway_order_id = None
    try:
        existing_pending = (
            supabase.table("orders")
            .select("id, gateway_checkout_url, gateway_order_id, status")
            .eq("listing_id", listing_id)
            .eq("buyer_id", buyer_id)
            .eq("status", "awaiting_payment")
            .execute()
        )
        if getattr(existing_pending, "data", None) and len(existing_pending.data) > 0:
            existing = existing_pending.data[0]
            order_id = existing.get("id")
            old_gateway_order_id = existing.get("gateway_order_id")
            reuse_order = bool(order_id)
            api_logger.info(
                "Reusing existing awaiting_payment order %s for buyer=%s listing=%s (will refresh amounts and PayPal order)",
                order_id,
                buyer_id,
                listing_id,
            )
    except Exception as e:
        api_logger.exception(
            "Failed to check existing pending orders (non-fatal): %s", e
        )

    # -------------------------
    # Load and Validate Seller Address (address_from)
    # -------------------------
    try:
        seller_id = listing.get("seller_id")
        if not seller_id:
            raise HTTPException(
                status_code=500, detail="Listing missing seller reference"
            )

        seller_addr_resp = (
            supabase.table("user_addresses")
            .select("*")
            .eq("user_id", seller_id)
            .execute()
        )
        if (
            not getattr(seller_addr_resp, "data", None)
            or len(seller_addr_resp.data) == 0
        ):
            api_logger.warning("No address found for seller %s", seller_id)
            raise HTTPException(
                status_code=400, detail="No shipping origin address found for seller"
            )
        seller_addr = seller_addr_resp.data[0]

        seller_profile_resp = (
            supabase.table("profiles")
            .select("username, email, phone_number")
            .eq("id", seller_id)
            .execute()
        )
        if (
            not getattr(seller_profile_resp, "data", None)
            or len(seller_profile_resp.data) == 0
        ):
            api_logger.warning("No seller profile found for %s", seller_id)
            seller_profile = {}
        else:
            seller_profile = seller_profile_resp.data[0]

        seller_address = {
            "name": (seller_profile.get("username") or "").strip() or "Seller",
            "street1": seller_addr.get("address_line_1", ""),
            "street2": seller_addr.get("address_line_2", "") or "",
            "city": seller_addr.get("city", ""),
            "state": seller_addr.get("state_province", ""),
            "zip": seller_addr.get("postal_code", ""),
            "country": (seller_addr.get("country", "") or "").upper(),
            "phone": seller_profile.get("phone_number") or None,
            "email": seller_profile.get("email") or None,
        }

        # Validate and normalize seller address
        try:
            seller_validated = await shippo_validate_address(seller_address)
        except HTTPException:
            raise
        except Exception as e:
            api_logger.exception("Shippo seller address validation failed: %s", e)
            raise HTTPException(
                status_code=500, detail="Unable to validate seller address"
            )

        try:
            vrd = (
                getattr(seller_validated, "validation_results", None)
                if not isinstance(seller_validated, dict)
                else seller_validated.get("validation_results")
            )
            if vrd and not (
                vrd.get("is_valid")
                if isinstance(vrd, dict)
                else getattr(vrd, "is_valid", False)
            ):
                msgs = (
                    vrd.get("messages", [])
                    if isinstance(vrd, dict)
                    else getattr(vrd, "messages", [])
                )
                short_msgs = (
                    [getattr(m, "text", str(m)) for m in msgs][:3] if msgs else []
                )
                raise HTTPException(
                    status_code=400,
                    detail={"error": "Invalid origin address", "messages": short_msgs},
                )

            # Accept normalized fields if present
            def _gf(obj, field):
                if not obj:
                    return None
                if isinstance(obj, dict):
                    return obj.get(field)
                return getattr(obj, field, None)

            norm_street1 = _gf(seller_validated, "street1")
            if norm_street1:
                seller_address.update(
                    {
                        "street1": norm_street1,
                        "street2": _gf(seller_validated, "street2")
                        or seller_address.get("street2", ""),
                        "city": _gf(seller_validated, "city")
                        or seller_address.get("city", ""),
                        "state": _gf(seller_validated, "state")
                        or seller_address.get("state", ""),
                        "zip": _gf(seller_validated, "zip")
                        or seller_address.get("zip", ""),
                        "country": _gf(seller_validated, "country")
                        or seller_address.get("country", ""),
                        "object_id": _gf(seller_validated, "object_id")
                        or seller_address.get("object_id"),
                        "validation_results": {"is_valid": True},
                    }
                )
        except HTTPException:
            raise
        except Exception as e:
            api_logger.exception(
                "Unexpected issue inspecting validated seller address: %s", e
            )
            raise HTTPException(
                status_code=500, detail="Error processing seller address"
            )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("Seller address processing failed: %s", e)
        raise HTTPException(status_code=500, detail="Error processing seller address")

    # -------------------------
    # Load and Validate Buyer Address
    # -------------------------
    try:
        buyer_address_resp = (
            supabase.table("user_addresses")
            .select("*")
            .eq("user_id", buyer_id)
            .execute()
        )
        if (
            not getattr(buyer_address_resp, "data", None)
            or len(buyer_address_resp.data) == 0
        ):
            api_logger.warning("No address found for buyer %s", buyer_id)
            raise HTTPException(
                status_code=400, detail="No shipping address found for buyer"
            )
        buyer_address = buyer_address_resp.data[0]

        # Get buyer profile for name, phone, email (since user_addresses table only has address fields)
        buyer_profile_resp = (
            supabase.table("profiles")
            .select("username, email, phone_number")
            .eq("id", buyer_id)
            .execute()
        )
        if (
            not getattr(buyer_profile_resp, "data", None)
            or len(buyer_profile_resp.data) == 0
        ):
            api_logger.warning("No buyer profile found for %s", buyer_id)
            buyer_profile = {}
        else:
            buyer_profile = buyer_profile_resp.data[0]

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

        # Validate address with Shippo
        try:
            validated_address = await shippo_validate_address(shipping_address)
        except HTTPException:
            # Re-raise HTTP exceptions from address validation
            raise
        except Exception as e:
            # Log and handle other validation failures
            api_logger.exception("Shippo address validation call failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail="Unable to validate shipping address. Please check the address and try again.",
            )

        # Inspect validation results defensively
        try:
            is_address_valid = False
            vr = getattr(validated_address, "validation_results", None)
            if vr is not None:
                is_address_valid = bool(getattr(vr, "is_valid", False))
                messages = getattr(vr, "messages", []) or []
                if not is_address_valid:
                    short_msgs = [getattr(m, "text", str(m)) for m in messages][:3]
                    api_logger.warning(
                        "Shippo returned invalid address for buyer %s: %s",
                        buyer_id,
                        short_msgs,
                    )
                    raise HTTPException(
                        status_code=400,
                        detail={"error": "Invalid address", "messages": short_msgs},
                    )
            else:
                # dict-like handling if helper returns dict
                vrd = (
                    validated_address.get("validation_results")
                    if isinstance(validated_address, dict)
                    else None
                )
                if vrd:
                    is_address_valid = bool(vrd.get("is_valid"))
                    if not is_address_valid:
                        msgs = vrd.get("messages", [])
                        short_msgs = [m.get("text", str(m)) for m in msgs][:3]
                        api_logger.warning(
                            "Shippo invalid address for buyer %s: %s",
                            buyer_id,
                            short_msgs,
                        )
                        raise HTTPException(
                            status_code=400,
                            detail={"error": "Invalid address", "messages": short_msgs},
                        )
                else:
                    # assume ok if no validation metadata
                    is_address_valid = True

            # Accept normalized fields if present
            def _get_field(obj, field):
                if not obj:
                    return None
                if isinstance(obj, dict):
                    return obj.get(field)
                return getattr(obj, field, None)

            normalized_street1 = _get_field(validated_address, "street1")
            if normalized_street1:
                shipping_address.update(
                    {
                        "street1": normalized_street1,
                        "street2": _get_field(validated_address, "street2")
                        or shipping_address.get("street2", ""),
                        "city": _get_field(validated_address, "city")
                        or shipping_address.get("city", ""),
                        "state": _get_field(validated_address, "state")
                        or shipping_address.get("state", ""),
                        "zip": _get_field(validated_address, "zip")
                        or shipping_address.get("zip", ""),
                        "country": _get_field(validated_address, "country")
                        or shipping_address.get("country", ""),
                        "validation_results": {"is_valid": True},
                    }
                )

        except HTTPException:
            raise
        except Exception as e:
            api_logger.exception("Unexpected issue inspecting validated address: %s", e)
            raise HTTPException(
                status_code=500, detail="Error processing validated address"
            )

    except HTTPException:
        # re-raise; safe_handler will log it too
        raise
    except Exception as e:
        api_logger.exception("Buyer address processing failed: %s", e)
        raise HTTPException(status_code=500, detail="Error processing shipping address")

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
            supabase.table("orders").update(update_fields).eq("id", order_id).execute()
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
            insert_resp = supabase.table("orders").insert(order_data).execute()
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
                api_logger.info(
                    "Voiding previous PayPal order id %s for reused order %s",
                    old_gateway_order_id,
                    order_id,
                )
                await paypal_void_checkout(old_gateway_order_id)
            except Exception as ve:
                api_logger.exception(
                    "Failed to void previous PayPal order %s (continuing): %s",
                    old_gateway_order_id,
                    ve,
                )

        base = get_base_url(request)
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

        try:
            # Prefer cards.canonical_title via card_id, then view's title, then name
            canonical_title = None
            try:
                card_id_from_listing = listing.get("card_id")
                if card_id_from_listing:
                    resp_card = (
                        supabase.table("cards")
                        .select("canonical_title")
                        .eq("id", card_id_from_listing)
                        .limit(1)
                        .execute()
                    )
                    if getattr(resp_card, "data", None):
                        canonical_title = (resp_card.data[0] or {}).get(
                            "canonical_title"
                        )
            except Exception:
                api_logger.exception(
                    "Failed to fetch canonical_title for card_id %s", listing.get("card_id")
                )

            item_name = (
                (canonical_title or "").strip()
                or (listing.get("title") or "").strip()
                or (listing.get("name") or "").strip()
                or "Unknown Card"
            )
        except Exception:
            api_logger.exception(
                "Failed to read item name from listing; falling back to Unknown Card"
            )
            item_name = "Unknown Card"

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
        update_resp = (
            supabase.table("orders")
            .update(
                {
                    "gateway_order_id": paypal_resp["paypal_order_id"],
                    "gateway_checkout_url": paypal_resp["paypal_checkout_url"],
                }
            )
            .eq("id", order_id)
            .execute()
        )
        if not getattr(update_resp, "data", None) or len(update_resp.data) == 0:
            api_logger.error("Order update returned no data for order %s", order_id)
            # best-effort cleanup
            try:
                supabase.table("orders").delete().eq("id", order_id).execute()
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
                supabase.table("orders").delete().eq("id", order_id).execute()
                api_logger.info("Cleaned up order %s after PayPal HTTP error", order_id)
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
                supabase.table("orders").delete().eq("id", order_id).execute()
                api_logger.info("Cleaned up order %s after PayPal error", order_id)
            except Exception:
                api_logger.exception("Failed to clean up order after PayPal error")
        raise HTTPException(status_code=500, detail="Create paypal approve url error")

    api_logger.info("Created order %s awaiting payment, checkout_url stored", order_id)
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
            try:
                params = dict(request.query_params)
            except Exception:
                params = {}

        merged = {}
        merged.update(params)
        merged.update(kwargs)

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

        # If the user returned with a PayPal order token, ensure there's a pending payment row now
        try:
            if final_token:
                # Load the order by gateway_order_id
                order_resp = (
                    supabase.table("orders")
                    .select("id, buyer_id, final_price, currency, status")
                    .eq("gateway_order_id", final_token)
                    .limit(1)
                    .execute()
                )
                if getattr(order_resp, "data", None):
                    order_row = order_resp.data[0]
                    if (order_row.get("status") or "").lower() == "awaiting_payment":
                        # Check if a payment already exists
                        pay_check = (
                            supabase.table("payments")
                            .select("id")
                            .eq("order_id", order_row.get("id"))
                            .limit(1)
                            .execute()
                        )
                        if not getattr(pay_check, "data", None):
                            # Insert a pending payment record tied to this PayPal order
                            try:
                                supabase.table("payments").insert(
                                    {
                                        "order_id": order_row.get("id"),
                                        "user_id": order_row.get("buyer_id"),
                                        "gateway": "paypal",
                                        "gateway_order_id": final_token,
                                        "amount": order_row.get("final_price"),
                                        "currency": (
                                            order_row.get("currency") or "USD"
                                        ).upper(),
                                        "status": "pending",
                                    }
                                ).execute()
                                api_logger.info(
                                    "Inserted pending payment on return for order %s",
                                    order_row.get("id"),
                                )
                            except Exception:
                                api_logger.exception(
                                    "Failed to insert pending payment on return for order %s",
                                    order_row.get("id"),
                                )
        except Exception:
            api_logger.exception(
                "Return handler: best-effort pending payment insert failed"
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

        # ---- handle approval events (Orders v2) ----
        # Valid Orders events include CHECKOUT.ORDER.APPROVED and CHECKOUT.ORDER.COMPLETED.
        # There is no CHECKOUT.ORDER.AUTHORIZED in Orders v2 webhooks.
        if event_type in ("CHECKOUT.ORDER.APPROVED",):
            resource = body.get("resource", {}) or {}
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
                    payments = pu[0].get("payments") or {}
                    captures = payments.get("captures") or []
                    if captures and captures[0]:
                        paypal_txn_id = captures[0].get("id")
                    if not paypal_txn_id:
                        auths = payments.get("authorizations") or []
                        if auths and auths[0]:
                            paypal_txn_id = auths[0].get("id")

                payer = resource.get("payer") or {}
                email = payer.get("email_address") or (
                    payer.get("payer_info", {}) or {}
                ).get("email")
                if email:
                    gateway_account_info["email"] = email

                card = (resource.get("payment_source") or {}).get("card") or {}
                if card:
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
                order_resp = (
                    supabase.table("orders")
                    .select(
                        "id, listing_id, buyer_id, shipping_rate_id, shipping_transaction_id, final_price, currency, status, shipping_address"
                    )
                    .eq("gateway_order_id", paypal_order_id)
                    .execute()
                )
                api_logger.debug(
                    "orders.select resp: %s", getattr(order_resp, "data", None)
                )
                if not getattr(order_resp, "data", None):
                    api_logger.error(
                        "Order not found for gateway_order_id %s", paypal_order_id
                    )
                    return {"status": "ok"}
                order = order_resp.data[0]
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
                payments_check = (
                    supabase.table("payments")
                    .select("id, status")
                    .eq("order_id", order_id)
                    .execute()
                )
                api_logger.debug(
                    "payments.check resp: %s", getattr(payments_check, "data", None)
                )
                if (
                    getattr(payments_check, "data", None)
                    and len(payments_check.data) > 0
                ):
                    # If any succeeded, short-circuit; otherwise keep a reference and continue to process
                    statuses = [
                        (row.get("status") or "").lower() for row in payments_check.data
                    ]
                    if any(s == "succeeded" for s in statuses):
                        api_logger.info(
                            "Payment already succeeded for order %s — skipping further processing",
                            order_id,
                        )
                        return {"status": "ok"}
                    payments_row = payments_check.data[0]
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

            # Ensure a pending payment exists (webhook fallback for out-of-order events)
            try:
                if payments_row is None:
                    supabase.table("payments").insert(
                        {
                            "order_id": order_id,
                            "user_id": order.get("buyer_id"),
                            "gateway": "paypal",
                            "gateway_order_id": paypal_order_id,
                            "gateway_transaction_id": paypal_txn_id,
                            "amount": order.get("final_price"),
                            "currency": (order.get("currency") or "USD").upper(),
                            "gateway_account_info": gateway_account_info or None,
                            "status": "pending",
                        }
                    ).execute()
                    api_logger.info(
                        "Inserted pending payment via webhook fallback for order %s",
                        order_id,
                    )
            except Exception:
                api_logger.exception(
                    "Webhook fallback: failed to insert pending payment for order %s",
                    order_id,
                )

            # Early exit on APPROVED: do not finalize here; wait for PAYMENT.CAPTURE.COMPLETED
            return {"status": "ok"}

        # ---- finalize on payment capture completed (Payments v2) ----
        if event_type == "PAYMENT.CAPTURE.COMPLETED":
            resource = body.get("resource", {}) or {}
            # Resolve order id from capture resource
            capture_id = resource.get("id")
            related = (resource.get("supplementary_data", {}) or {}).get(
                "related_ids", {}
            )
            paypal_order_id = related.get("order_id")
            related_auth_id = related.get("authorization_id")
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
                order_resp = (
                    supabase.table("orders")
                    .select(
                        "id, listing_id, buyer_id, shipping_rate_id, shipping_transaction_id, final_price, currency, status, shipping_address"
                    )
                    .eq("gateway_order_id", paypal_order_id)
                    .execute()
                )
                api_logger.debug(
                    "orders.select resp: %s", getattr(order_resp, "data", None)
                )
                if not getattr(order_resp, "data", None):
                    api_logger.error(
                        "Order not found for gateway_order_id %s", paypal_order_id
                    )
                    return {"status": "ok"}
                order = order_resp.data[0]
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

            # Step 1: idempotency check — skip only if an existing payment is already succeeded
            payments_row = None
            try:
                payments_check = (
                    supabase.table("payments")
                    .select("id, status")
                    .eq("order_id", order_id)
                    .execute()
                )
                api_logger.debug(
                    "payments.check resp: %s", getattr(payments_check, "data", None)
                )
                if (
                    getattr(payments_check, "data", None)
                    and len(payments_check.data) > 0
                ):
                    # If any succeeded, short-circuit; otherwise keep a reference and continue to process
                    statuses = [
                        (row.get("status") or "").lower() for row in payments_check.data
                    ]
                    if any(s == "succeeded" for s in statuses):
                        api_logger.info(
                            "Payment already succeeded for order %s — skipping further processing",
                            order_id,
                        )
                        return {"status": "ok"}
                    payments_row = payments_check.data[0]
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
                    supabase.table("payments").insert(
                        {
                            "order_id": order_id,
                            "user_id": order.get("buyer_id"),
                            "gateway": "paypal",
                            "gateway_order_id": paypal_order_id,
                            "gateway_transaction_id": capture_id,
                            "amount": order.get("final_price"),
                            "currency": (order.get("currency") or "USD").upper(),
                            "status": "pending",
                        }
                    ).execute()
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
                        supabase.table("orders").update(
                            {"status": "cancelled", "order_failed_info": error_msg}
                        ).eq("id", order_id).execute()
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

                    # Mark related payment(s) as failed
                    try:
                        supabase.table("payments").update({"status": "failed"}).eq(
                            "order_id", order_id
                        ).execute()
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
                        pus_void = (
                            (resource.get("purchase_units") or [])
                            if isinstance(resource, dict)
                            else []
                        )
                        if pus_void:
                            payments_void = pus_void[0].get("payments") or {}
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
                    supabase.table("orders").update(
                        {"status": "cancelled", "order_failed_info": error_msg}
                    ).eq("id", order_id).execute()
                except Exception:
                    api_logger.exception(
                        "Failed to update order %s as cancelled after generic Shippo failure",
                        order_id,
                    )

                # Mark payment(s) failed as well
                try:
                    supabase.table("payments").update({"status": "failed"}).eq(
                        "order_id", order_id
                    ).execute()
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
                upd_resp = (
                    supabase.table("payments")
                    .update(update_payload)
                    .eq("order_id", order_id)
                    .execute()
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
                        pr2 = (
                            supabase.table("payments")
                            .select("id")
                            .eq("order_id", order_id)
                            .order("created_at", desc=True)
                            .limit(1)
                            .execute()
                        )
                        if getattr(pr2, "data", None):
                            pid = pr2.data[0].get("id")
                            if pid:
                                supabase.table("payments").update(update_payload).eq(
                                    "id", pid
                                ).execute()
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
                supabase.table("orders").update(update_data).eq(
                    "id", order_id
                ).execute()
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
                    other_orders = (
                        supabase.table("orders")
                        .select("id")
                        .eq("listing_id", listing_id)
                        .neq("id", order_id)
                        .eq("status", "awaiting_payment")
                        .execute()
                    )
                    other_ids = [
                        row.get("id")
                        for row in (getattr(other_orders, "data", []) or [])
                        if row.get("id")
                    ]
                    if other_ids:
                        supabase.table("orders").update({"status": "cancelled"}).in_(
                            "id", other_ids
                        ).execute()
                        api_logger.info(
                            "Cancelled %d sibling order(s) for listing %s",
                            len(other_ids),
                            listing_id,
                        )
                        # Cancel payments for those orders (if not already succeeded)
                        try:
                            supabase.table("payments").update(
                                {"status": "cancelled"}
                            ).in_("order_id", other_ids).execute()
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
                supabase.table("orders").update({"status": "cancelled"}).eq(
                    "gateway_order_id", token
                ).execute()
                api_logger.info(
                    "Marked order with gateway_order_id %s as cancelled", token
                )
            except Exception:
                api_logger.exception(
                    "Failed to mark order cancelled for token %s", token
                )

            try:
                # Attempt to void any existing auth related to this order (safe if none)
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

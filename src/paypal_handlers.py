import httpx
import json
from typing import Optional
from shippo import components
from urllib.parse import urlencode
from datetime import datetime, timezone
from fastapi import HTTPException, Request
from starlette.responses import RedirectResponse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from src.shippo_handlers import shippo_sdk
from src.models.shippo import TransactionPayload
from src.utils.supabase import supabase
from src.utils.currency import format_price
from src.utils.shippo import shippo_get_rate_details
from src.utils.paypal import (
    get_access_token,
    get_base_url,
    get_api_base,
    paypal_create_subscription,
    paypal_create_order,
    paypal_authorize_order,
    paypal_capture_authorization,
    paypal_void_authorization,
    paypal_void_checkout,
)
from src.utils.logger import (
    api_logger,
    log_api_request,
    log_error_with_context,
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

        # 3. Persist a 'pending' subscription mapping (idempotent)
        try:
            existing = (
                supabase.table("subscriptions")
                .select("id")
                .eq("paypal_subscription_id", subscription_id)
                .execute()
            )
            if not existing.data:
                supabase.table("subscriptions").insert(
                    {
                        "user_id": user_id,
                        "plan_id": plan_id,  # internal plan id
                        "status": "pending",
                        "paypal_subscription_id": subscription_id,
                        "start_date": None,
                        "end_date": None,
                    }
                ).execute()
            else:
                api_logger.info("Subscription already exists (pending entry).")
        except Exception:
            api_logger.exception(
                "Failed to insert 'pending' subscription - continuing to redirect to PayPal."
            )

        # 4. redirect user to PayPal approval page
        return RedirectResponse(url=approval_url, status_code=302)

    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(api_logger, e, "PayPal checkout")
        raise HTTPException(status_code=500, detail=f"Checkout error: {e}")


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

        status = (data.get("status") or "").lower()
        if status != "active":
            api_logger.warning(
                f"Subscription {subscription_id} status is {status} on return"
            )
            params = urlencode(
                {
                    "status": "error",
                    "reason": "not_active",
                    "subscriptionId": subscription_id,
                }
            )
            return RedirectResponse(
                f"{redirect}{'&' if '?' in redirect else '?'}{params}",
                status_code=302,
            )

        # Find or create internal subscription (idempotent)
        existing = (
            supabase.table("subscriptions")
            .select("id")
            .eq("paypal_subscription_id", subscription_id)
            .execute()
        )
        if existing.data:
            sub_id = existing.data[0]["id"]
            api_logger.info(f"Found existing subscription id: {sub_id}")
            # Update status/dates
            supabase.table("subscriptions").update(
                {
                    "status": "active",
                    "start_date": data.get("start_time") or datetime.now().isoformat(),
                    "end_date": data.get("billing_info", {}).get("next_billing_time"),
                }
            ).eq("paypal_subscription_id", subscription_id).execute()
        else:
            # Insert new subscription (use plan_id from query param)
            insert_resp = (
                supabase.table("subscriptions")
                .insert(
                    {
                        "user_id": user_id,
                        "plan_id": plan_id,
                        "status": "active",
                        "paypal_subscription_id": subscription_id,
                        "start_date": data.get("start_time")
                        or datetime.now().isoformat(),
                        "end_date": data.get("billing_info", {}).get(
                            "next_billing_time"
                        ),
                    }
                )
                .execute()
            )
            sub_id = insert_resp.data[0]["id"]

        # ✅ Remove all payment insertion here
        params = urlencode({"status": "success", "subscriptionId": subscription_id})
        return RedirectResponse(
            f"{redirect}{'&' if '?' in redirect else '?'}{params}",
            status_code=302,
        )

    except Exception as e:
        log_error_with_context(
            api_logger, e, f"checking PayPal subscription {subscription_id}"
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
            api_logger.error("❌ PayPal webhook missing subscription ID")
            return {"status": "error", "message": "Missing subscription ID"}

        # -------------------------
        # Subscription Activated
        # -------------------------
        if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
            existing = (
                supabase.table("subscriptions")
                .select("id")
                .eq("paypal_subscription_id", subscription_id)
                .execute()
            )
            if existing.data:
                supabase.table("subscriptions").update(
                    {
                        "status": "active",
                        "start_date": resource.get("start_time"),
                        "end_date": resource.get("billing_info", {}).get(
                            "next_billing_time"
                        ),
                    }
                ).eq("paypal_subscription_id", subscription_id).execute()
            else:
                api_logger.warning(
                    f"Subscription {subscription_id} ACTIVATED but not found in DB. Skipping insert because user_id unknown."
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
            if not paypal_sub_id:
                api_logger.error(
                    "❌ Missing billing_agreement_id in PAYMENT.SALE.COMPLETED"
                )
            else:
                sub_res = (
                    supabase.table("subscriptions")
                    .select("id, user_id")
                    .eq("paypal_subscription_id", paypal_sub_id)
                    .execute()
                )

                if sub_res.data and len(sub_res.data) > 0:
                    subscription = sub_res.data[0]
                    # Idempotent insert
                    existing_payment = (
                        supabase.table("payments")
                        .select("id")
                        .eq("gateway_transaction_id", txn_id)
                        .execute()
                    )

                    if not existing_payment.data:
                        supabase.table("payments").insert(
                            {
                                "user_id": subscription["user_id"],
                                "subscription_id": subscription["id"],
                                "gateway": "paypal",
                                "gateway_transaction_id": txn_id,
                                "amount": resource["amount"]["total"],
                                "currency": resource["amount"]["currency"],
                                "status": "succeeded",
                            }
                        ).execute()
                    else:
                        api_logger.info(
                            f"Payment {txn_id} already exists, skipping insert"
                        )
                else:
                    api_logger.error(
                        f"❌ No subscription found for PayPal ID {paypal_sub_id}"
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

        api_logger.info(
            f"✅ Handled PayPal webhook: {event_type} for {subscription_id}"
        )
        return {"status": "ok"}

    except Exception as e:
        log_error_with_context(api_logger, e, "Webhook error")
        raise HTTPException(status_code=400, detail="Invalid webhook payload")


# ===============================================================
# /paypal/order
# ===============================================================
async def paypal_order_handler(request: Request, payload: TransactionPayload):
    log_api_request(api_logger, "POST", "/paypal/order", {"payload": payload})
    # -------------------------
    # Parameter Extraction & Initial Validation
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

    if not listing_id or not buyer_id:
        api_logger.error("Missing listing_id or buyer_id in payload")
        raise HTTPException(status_code=400, detail="listing_id and buyer_id required")

    # Use Decimal for money parsing
    try:
        selected_rate_amount = (
            Decimal(str(selected_rate_amount))
            if selected_rate_amount is not None
            else Decimal("0.00")
        )
        insurance_amount = (
            Decimal(str(insurance_amount))
            if insurance_amount is not None
            else Decimal("0.00")
        )
    except (InvalidOperation, TypeError) as e:
        api_logger.error("Invalid numeric inputs in payload: %s", e)
        raise HTTPException(status_code=400, detail="Invalid numeric values provided")

    # -------------------------
    # Load Buyer Currency
    # -------------------------
    try:
        buyer_resp = (
            supabase.table("profiles").select("currency").eq("id", buyer_id).execute()
        )
        api_logger.debug(
            "supabase.profiles.select currency resp: %s",
            getattr(buyer_resp, "data", None),
        )
        if not getattr(buyer_resp, "data", None):
            api_logger.error("Buyer not found: %s", buyer_id)
            raise HTTPException(status_code=404, detail="Buyer not found")
        buyer = buyer_resp.data[0]
        buyer_currency = (
            buyer.get("currency") or selected_rate_currency or "USD"
        ).upper()
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("Buyer currency lookup failed: %s", e)
        raise HTTPException(status_code=500, detail="Buyer currency lookup failed")

    # -------------------------
    # Load Buyer Address
    # -------------------------
    try:
        buyer_address_resp = (
            supabase.table("user_addresses")
            .select("*")
            .eq("user_id", buyer_id)
            .execute()
        )
        api_logger.debug(
            "supabase.user_addresses resp: %s",
            getattr(buyer_address_resp, "data", None),
        )
        if not getattr(buyer_address_resp, "data", None):
            api_logger.error("Buyer address not found for user_id %s", buyer_id)
            raise HTTPException(status_code=404, detail="Buyer address not found")
        buyer_address = buyer_address_resp.data[0]
        shipping_address = {
            "street1": buyer_address.get("address_line_1", ""),
            "city": buyer_address.get("city", ""),
            "state": buyer_address.get("state_province", ""),
            "zip": buyer_address.get("postal_code", ""),
            "country": buyer_address.get("country", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("Buyer address lookup failed: %s", e)
        raise HTTPException(status_code=500, detail="Buyer address lookup failed")

    # -------------------------
    # Load Listing Data
    # -------------------------
    try:
        listing_resp = (
            supabase.table("vw_chaamo_cards").select("*").eq("id", listing_id).execute()
        )
        api_logger.debug("listing resp: %s", getattr(listing_resp, "data", None))
        if not getattr(listing_resp, "data", None):
            api_logger.error("Listing not found: %s", listing_id)
            raise HTTPException(status_code=404, detail="Listing not found")
        listing = listing_resp.data[0]
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("Listing lookup failed: %s", e)
        raise HTTPException(status_code=500, detail="Listing lookup failed")

    # -------------------------
    # Prevent duplicate pending order for same buyer+listing
    # -------------------------
    try:
        existing_pending = (
            supabase.table("orders")
            .select("id, gateway_checkout_url, gateway_order_id, status")
            .eq("listing_id", listing_id)
            .eq("buyer_id", buyer_id)
            .eq("status", "awaiting_payment")
            .execute()
        )
        api_logger.debug(
            "existing_pending resp: %s", getattr(existing_pending, "data", None)
        )
        if getattr(existing_pending, "data", None) and len(existing_pending.data) > 0:
            existing = existing_pending.data[0]
            api_logger.info(
                "Returning existing pending order for buyer %s listing %s",
                buyer_id,
                listing_id,
            )
            return {
                "paypal_order_id": existing.get("gateway_order_id"),
                "paypal_checkout_url": existing.get("gateway_checkout_url"),
            }
    except Exception as e:
        api_logger.exception(
            "Failed to check existing pending orders (non-fatal): %s", e
        )

    # -------------------------
    # Verify selected rate against Shippo server-side
    # -------------------------
    try:
        rate_details = shippo_get_rate_details(selected_rate_id)
        server_rate_amount = rate_details["amount"]
        server_rate_currency = rate_details["currency"].upper()
    except Exception as e:
        api_logger.error("Shippo rate validation failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid shipping rate: {e}")

    # -------------------------
    # Seller Earnings Conversion
    # -------------------------
    try:
        listing_currency = listing["currency"]
        selected_price = (
            listing.get("highest_bid_price", 0)
            if listing["listing_type"] == "auction"
            else listing.get("start_price", 0)
        )
        seller_earnings = Decimal(
            str(
                format_price(
                    from_currency=listing_currency,
                    to_currency=buyer_currency,
                    amount=float(selected_price),
                )
            )
        )
    except Exception as e:
        api_logger.exception("Currency conversion failed: %s", e)
        raise HTTPException(status_code=500, detail="Currency conversion error")

    # -------------------------
    # Shipping Fee Conversion (use server_rate_amount)
    # -------------------------
    try:
        shipping_fee = Decimal(
            str(
                format_price(
                    from_currency=server_rate_currency,
                    to_currency=buyer_currency,
                    amount=float(server_rate_amount),
                )
            )
        )
    except Exception as e:
        api_logger.exception("Shipping fee conversion failed: %s", e)
        raise HTTPException(status_code=500, detail="Shipping fee conversion error")

    # -------------------------
    # Insurance Handling (if applicable)
    # -------------------------
    insurance_fee = Decimal("0.00")
    if insurance and insurance_amount > 0:
        try:
            insurance_fee = Decimal(
                str(
                    format_price(
                        from_currency=insurance_currency,
                        to_currency=buyer_currency,
                        amount=float(insurance_amount),
                    )
                )
            )
        except Exception as e:
            api_logger.exception("Insurance conversion failed: %s", e)
            raise HTTPException(status_code=500, detail="Insurance conversion error")

    # -------------------------
    # Final Price Calculation
    # -------------------------
    try:
        # Ensure all components are precisely two decimals to avoid PayPal sum mismatches
        seller_earnings = seller_earnings.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        shipping_fee = shipping_fee.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        insurance_fee = insurance_fee.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        final_price = (seller_earnings + shipping_fee + insurance_fee).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    except Exception as e:
        api_logger.exception("Final price calc failed: %s", e)
        raise HTTPException(status_code=500, detail="Final price calculation error")

    # -------------------------
    # Create initial order record (awaiting_payment) - DB authoritative
    # -------------------------
    try:
        order_data = {
            "listing_id": listing_id,
            "seller_id": listing["seller_id"],
            "buyer_id": buyer_id,
            "currency": buyer_currency,
            "final_price": str(final_price),
            "seller_earnings": str(seller_earnings),
            "shipping_fee": str(shipping_fee),
            "insurance_fee": str(insurance_fee),
            "status": "awaiting_payment",
            "shipping_address": shipping_address,
            "shipping_rate_id": selected_rate_id,
        }
        insert_resp = supabase.table("orders").insert(order_data).execute()
        api_logger.debug("orders.insert resp: %s", getattr(insert_resp, "data", None))
        if not getattr(insert_resp, "data", None) or len(insert_resp.data) == 0:
            api_logger.error(
                "DB insert returned empty for order creation: %s", insert_resp
            )
            raise Exception("DB insert returned empty")
        order_row = insert_resp.data[0]
        order_id = order_row.get("id")
    except Exception as e:
        log_error_with_context(api_logger, e, "create order failed (initial insert)")
        raise HTTPException(status_code=500, detail=f"Create order error: {e}")

    # -------------------------
    # Create Paypal Approve URL and update order with gateway fields
    # -------------------------
    try:
        base = get_base_url(request)
        return_url = (
            f"{base}/api/v1/paypal/order/return?{urlencode({'redirect': redirect})}"
        )
        cancel_url = f"{base}/api/v1/paypal/cancel?{urlencode({'redirect': redirect})}"
        # Provide amount breakdown so PayPal shows item + shipping (+ handling used for insurance) clearly
        # Build with strict 2-decimal strings and ensure the sum equals final_price exactly.
        amount_breakdown = {
            "item_total": f"{seller_earnings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}",
            "shipping": f"{shipping_fee.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}",
        }
        if insurance_fee and insurance_fee > 0:
            # PayPal v2 supports breakdown fields like handling; use it to represent insurance if applicable
            amount_breakdown["handling"] = f"{insurance_fee.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"

        # Adjust breakdown to guarantee exact equality with final_price if rounding created a minor delta
        bd_item = Decimal(amount_breakdown["item_total"]) if amount_breakdown.get("item_total") else Decimal("0.00")
        bd_ship = Decimal(amount_breakdown["shipping"]) if amount_breakdown.get("shipping") else Decimal("0.00")
        bd_hand = Decimal(amount_breakdown.get("handling", "0.00"))
        bd_sum = (bd_item + bd_ship + bd_hand).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if bd_sum != final_price:
            delta = (final_price - bd_sum).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            # Adjust item_total by the delta to make the sum exact
            bd_item = (bd_item + delta).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            # Safeguard against negative item_total
            if bd_item < Decimal("0.00"):
                # If negative (extremely unlikely), push into handling instead
                bd_hand = (bd_hand + bd_item).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                bd_item = Decimal("0.00")
            amount_breakdown["item_total"] = f"{bd_item}"
            if bd_hand > Decimal("0.00"):
                amount_breakdown["handling"] = f"{bd_hand}"

        # Use the finalized item_total for the line item unit_amount to match PayPal UI exactly
        item_unit_amount = Decimal(amount_breakdown["item_total"]) if amount_breakdown.get("item_total") else seller_earnings

        # Build a single line item for the card
        try:
            item_name = (
                (listing.get("title") or listing.get("name") or "Trading Card").strip()
            )
        except Exception:
            item_name = "Trading Card"

        items = [
            {
                "name": item_name[:127],  # PayPal item name length limits
                "quantity": "1",
                "unit_amount": {
                    "currency_code": buyer_currency,
                    "value": f"{item_unit_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}",
                },
            }
        ]

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
        api_logger.debug("paypal_create_order resp: %s", paypal_resp)

        # Update order with PayPal data (checkout url + id)
        try:
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
            api_logger.debug(
                "orders.update (gateway fields) resp: %s",
                getattr(update_resp, "data", None),
            )
        except Exception as e:
            api_logger.exception(
                "Failed to update order with gateway fields for order %s: %s",
                order_id,
                e,
            )
            # best-effort cleanup: delete order if we cannot update with gateway fields
            try:
                supabase.table("orders").delete().eq("id", order_id).execute()
                api_logger.info(
                    "Deleted order %s after gateway update failure", order_id
                )
            except Exception:
                api_logger.exception(
                    "Failed to delete order after gateway update failure: %s", order_id
                )
            raise HTTPException(
                status_code=500, detail="Failed to persist PayPal gateway fields"
            )
    except Exception as e:
        log_error_with_context(
            api_logger, e, "create paypal approve url failed; cleaning up order"
        )
        raise HTTPException(
            status_code=500, detail=f"Create paypal approve url error: {e}"
        )

    api_logger.info("Created order %s awaiting payment, checkout_url stored", order_id)
    return paypal_resp


# ===============================================================
# /paypal/order/return
# ===============================================================
# ------------------------------------------------------------
# /paypal/order/return  (robust: accepts request or kwargs)
# ------------------------------------------------------------
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

        status = "approved" if final_token else "cancelled"
        return_params = {"status": status, "orderId": final_token or ""}

        sep = "&" if "?" in final_redirect else "?"
        target_url = f"{final_redirect}{sep}{urlencode(return_params)}"

        api_logger.info(
            "Redirecting user to %s after PayPal return (merged params: %s)",
            target_url,
            merged,
        )
        return RedirectResponse(url=target_url, status_code=302)
    except Exception as e:
        log_error_with_context(api_logger, e, "returning PayPal order (robust handler)")
        try:
            fallback_redirect = redirect or (merged.get("redirect") if merged else "/")
            return RedirectResponse(
                url=f"{fallback_redirect}{'&' if '?' in fallback_redirect else '?'}{urlencode({'status':'error','orderId':''})}",
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

        # ---- handle approval events ----
        if event_type in ("CHECKOUT.ORDER.APPROVED", "CHECKOUT.ORDER.AUTHORIZED"):
            resource = body.get("resource", {}) or {}
            paypal_order_id = resource.get("id") or resource.get("order_id")
            if not paypal_order_id:
                api_logger.error("Webhook approval event missing order id")
                return {"status": "ignored"}

            # load order
            try:
                order_resp = (
                    supabase.table("orders")
                    .select(
                        "id, buyer_id, shipping_rate_id, shipping_transaction_id, final_price, currency, status"
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

            # idempotency checks
            try:
                payments_check = (
                    supabase.table("payments")
                    .select("id")
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
                    api_logger.info(
                        "Payment already exists for order %s — skipping", order_id
                    )
                    return {"status": "ok"}
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

            # 1) authorize PayPal order (server-side)
            try:
                auth_resp = await paypal_authorize_order(paypal_order_id)
                api_logger.debug("paypal_authorize_order resp: %s", auth_resp)
                auth_id = None
                for pu in auth_resp.get("purchase_units", []) or []:
                    payments = pu.get("payments", {}) or {}
                    authorizations = payments.get("authorizations", []) or []
                    if authorizations:
                        auth_id = authorizations[0].get("id")
                        break
                if not auth_id:
                    api_logger.error(
                        "No authorization id returned from authorize response: %s",
                        auth_resp,
                    )
                    # mark manual and return
                    try:
                        supabase.table("orders").update(
                            {"status": "awaiting_shipment_manual_intervention"}
                        ).eq("id", order_id).execute()
                    except Exception:
                        api_logger.exception(
                            "Failed to set awaiting_shipment_manual_intervention on order %s",
                            order_id,
                        )
                    return {"status": "ok"}
                # persist auth id
                try:
                    supabase.table("orders").update(
                        {"gateway_authorization_id": auth_id}
                    ).eq("id", order_id).execute()
                    api_logger.info(
                        "Stored gateway_authorization_id for order %s", order_id
                    )
                except Exception:
                    api_logger.exception(
                        "Failed to persist gateway_authorization_id for order %s",
                        order_id,
                    )
            except Exception as e:
                api_logger.exception(
                    "PayPal authorize failed for order %s: %s", paypal_order_id, e
                )
                raise HTTPException(status_code=500, detail="PayPal authorize failed")

            # 2) create Shippo transaction
            try:
                if not order.get("shipping_rate_id"):
                    api_logger.error(
                        "Order %s missing shipping_rate_id — cannot create Shippo transaction",
                        order_id,
                    )
                    # void auth then cancel
                    try:
                        await paypal_void_authorization(auth_id)
                        api_logger.info(
                            "Voided authorization %s for order %s (no rate)",
                            auth_id,
                            order_id,
                        )
                    except Exception:
                        api_logger.exception(
                            "Failed to void authorization %s for order %s",
                            auth_id,
                            order_id,
                        )
                    try:
                        supabase.table("orders").update({"status": "cancelled"}).eq(
                            "id", order_id
                        ).execute()
                    except Exception:
                        api_logger.exception(
                            "Failed to mark order %s cancelled", order_id
                        )
                    return {"status": "ok"}

                api_logger.info(
                    "Creating Shippo transaction for order %s using rate %s",
                    order_id,
                    order.get("shipping_rate_id"),
                )
                tx_req = components.TransactionCreateRequest(
                    rate=order.get("shipping_rate_id"),
                    label_file_type=components.LabelFileTypeEnum.PDF,
                    async_=False,
                )
                transaction = shippo_sdk.transactions.create(tx_req)
                api_logger.debug("Shippo transaction resp: %s", transaction)
                tx_status = (
                    transaction.get("status")
                    if isinstance(transaction, dict)
                    else getattr(transaction, "status", None)
                )
                if not (tx_status and tx_status.upper() in ("SUCCESS", "SUCCESSFUL")):
                    api_logger.error(
                        "Shippo transaction failed for order %s: %s",
                        order_id,
                        transaction,
                    )
                    try:
                        await paypal_void_authorization(auth_id)
                        api_logger.info(
                            "Voided authorization %s for order %s after Shippo failure",
                            auth_id,
                            order_id,
                        )
                    except Exception:
                        api_logger.exception(
                            "Failed to void authorization after Shippo failure for order %s",
                            order_id,
                        )
                    try:
                        supabase.table("orders").update({"status": "cancelled"}).eq(
                            "id", order_id
                        ).execute()
                    except Exception:
                        api_logger.exception(
                            "Failed to mark order %s cancelled after Shippo failure",
                            order_id,
                        )
                    raise HTTPException(
                        status_code=500, detail="Shippo transaction failed"
                    )
                # extract shippo fields
                object_id = transaction.get("object_id") or getattr(
                    transaction, "object_id", None
                )
                tracking_number = transaction.get("tracking_number") or getattr(
                    transaction, "tracking_number", None
                )
                tracking_url = (
                    transaction.get("tracking_url_provider")
                    or transaction.get("tracking_url")
                    or getattr(transaction, "tracking_url_provider", None)
                )
                label_url = transaction.get("label_url") or getattr(
                    transaction, "label_url", None
                )
            except HTTPException:
                raise
            except Exception as e:
                api_logger.exception(
                    "Shippo transaction creation error for order %s: %s", order_id, e
                )
                try:
                    await paypal_void_authorization(auth_id)
                except Exception:
                    api_logger.exception(
                        "Failed to void authorization after Shippo exception for order %s",
                        order_id,
                    )
                raise HTTPException(
                    status_code=500, detail="Shippo transaction creation error"
                )

            # 3) capture authorization
            try:
                capture_resp = await paypal_capture_authorization(auth_id)
                api_logger.debug("paypal_capture_authorization resp: %s", capture_resp)
                capture_ids = []
                for pu in capture_resp.get("purchase_units", []) or []:
                    payments = pu.get("payments", {}) or {}
                    for cap in payments.get("captures", []) or []:
                        if cap.get("id"):
                            capture_ids.append(
                                {
                                    "id": cap.get("id"),
                                    "status": cap.get("status"),
                                    "amount": cap.get("amount"),
                                }
                            )
                if not capture_ids:
                    api_logger.error(
                        "No capture ids found for auth %s (order %s): %s",
                        auth_id,
                        order_id,
                        capture_resp,
                    )
                    supabase.table("orders").update(
                        {"status": "awaiting_shipment_manual_intervention"}
                    ).eq("id", order_id).execute()
                    raise HTTPException(status_code=500, detail="No capture ids found")
                primary = capture_ids[0]
                gateway_transaction_id = primary.get("id")
                payment_amount = (
                    primary.get("amount", {}).get("value")
                    if primary.get("amount")
                    else None
                )
                payment_currency = (
                    primary.get("amount", {}).get("currency_code")
                    if primary.get("amount")
                    else None
                )
            except HTTPException:
                raise
            except Exception as e:
                api_logger.exception(
                    "PayPal capture failed for auth %s (order %s): %s",
                    auth_id,
                    order_id,
                    e,
                )
                supabase.table("orders").update(
                    {"status": "awaiting_shipment_manual_intervention"}
                ).eq("id", order_id).execute()
                raise HTTPException(status_code=500, detail="PayPal capture failed")

            # 4) insert payment
            try:
                payment_data = {
                    "order_id": order_id,
                    "user_id": order.get("buyer_id"),
                    "gateway": "paypal",
                    "gateway_transaction_id": gateway_transaction_id,
                    "amount": (
                        str(payment_amount)
                        if payment_amount is not None
                        else str(order.get("final_price"))
                    ),
                    "currency": payment_currency or order.get("currency"),
                    "status": "succeeded",
                }
                payment_insert_resp = (
                    supabase.table("payments").insert(payment_data).execute()
                )
                api_logger.debug(
                    "payments.insert resp: %s",
                    getattr(payment_insert_resp, "data", None),
                )
                if not getattr(payment_insert_resp, "data", None):
                    api_logger.error(
                        "Payment insert returned no data for order %s: %s",
                        order_id,
                        payment_insert_resp,
                    )
                    raise Exception("Payment insertion returned no data")
            except Exception as e:
                api_logger.exception(
                    "Failed to insert payment for order %s: %s", order_id, e
                )
                # Leave order in manual intervention so ops can reconcile
                try:
                    supabase.table("orders").update(
                        {"status": "awaiting_shipment_manual_intervention"}
                    ).eq("id", order_id).execute()
                except Exception:
                    api_logger.exception(
                        "Failed to set awaiting_shipment_manual_intervention after payment insert failure for order %s",
                        order_id,
                    )
                raise HTTPException(status_code=500, detail="Payment insertion failed")

            # 5) update order with shipping & paid fields
            try:
                supabase.table("orders").update(
                    {
                        "status": "awaiting_shipment",
                        "paid_at": datetime.now(timezone.utc).isoformat(),
                        "shipping_transaction_id": object_id,
                        "shipping_tracking_number": tracking_number,
                        "shipping_tracking_url": tracking_url,
                        "shipping_label_url": label_url,
                    }
                ).eq("id", order_id).execute()
                api_logger.info(
                    "Order %s updated to awaiting_shipment and payment recorded",
                    order_id,
                )
            except Exception as e:
                api_logger.exception(
                    "Failed to update order %s after payment: %s", order_id, e
                )
                # order already has payment inserted; mark manual
                try:
                    supabase.table("orders").update(
                        {"status": "awaiting_shipment_manual_intervention"}
                    ).eq("id", order_id).execute()
                except Exception:
                    api_logger.exception(
                        "Additionally failed to mark order %s manual", order_id
                    )
                raise HTTPException(
                    status_code=500,
                    detail="Order update failed after payment insertion",
                )

            return {"status": "ok"}

        # ---- handle capture webhook (fallback) ----
        if event_type == "PAYMENT.CAPTURE.COMPLETED":
            resource = body.get("resource", {}) or {}
            paypal_order_id = resource.get("supplementary_data", {}).get(
                "related_ids", {}
            ).get("order_id") or resource.get("invoice_id")
            paypal_transaction_id = resource.get("id")
            paypal_amount_paid = resource.get("amount", {}).get("value")
            paypal_currency = resource.get("amount", {}).get("currency_code")

            if not paypal_order_id:
                api_logger.error("PAYMENT.CAPTURE.COMPLETED missing order id")
                return {"status": "ok"}

            try:
                order_resp = (
                    supabase.table("orders")
                    .select("id, buyer_id, status")
                    .eq("gateway_order_id", paypal_order_id)
                    .execute()
                )
                api_logger.debug(
                    "orders.select for capture webhook: %s",
                    getattr(order_resp, "data", None),
                )
                if not getattr(order_resp, "data", None):
                    api_logger.error(
                        "Order not found for gateway_order_id %s in capture webhook",
                        paypal_order_id,
                    )
                    return {"status": "ok"}
                order = order_resp.data[0]
                order_id = order.get("id")
            except Exception as e:
                api_logger.exception(
                    "DB error fetching order for capture webhook: %s", e
                )
                raise HTTPException(
                    status_code=500,
                    detail="DB error while fetching order for capture webhook",
                )

            try:
                existing_payments = (
                    supabase.table("payments")
                    .select("id")
                    .eq("order_id", order_id)
                    .execute()
                )
                api_logger.debug(
                    "payments.select existing: %s",
                    getattr(existing_payments, "data", None),
                )
                if (
                    getattr(existing_payments, "data", None)
                    and len(existing_payments.data) > 0
                ):
                    api_logger.info(
                        "Payment already exists for order %s; skipping capture webhook handling",
                        order_id,
                    )
                    return {"status": "ok"}

                payment_data = {
                    "order_id": order_id,
                    "user_id": order.get("buyer_id"),
                    "gateway": "paypal",
                    "gateway_transaction_id": paypal_transaction_id,
                    "amount": str(paypal_amount_paid),
                    "currency": paypal_currency,
                    "status": "succeeded",
                }
                insert_resp = supabase.table("payments").insert(payment_data).execute()
                api_logger.debug(
                    "payments.insert (capture webhook) resp: %s",
                    getattr(insert_resp, "data", None),
                )
                if not getattr(insert_resp, "data", None):
                    api_logger.error(
                        "Payment insert returned no data on capture webhook for order %s: %s",
                        order_id,
                        insert_resp,
                    )
                    raise Exception("Payment insert returned no data")

                supabase.table("orders").update(
                    {
                        "status": "awaiting_shipment",
                        "paid_at": datetime.now(timezone.utc).isoformat(),
                    }
                ).eq("id", order_id).execute()
                api_logger.info(
                    "Capture webhook: order %s updated to awaiting_shipment", order_id
                )
            except Exception as e:
                api_logger.exception(
                    "Failed to process PAYMENT.CAPTURE.COMPLETED for order %s: %s",
                    order_id if "order_id" in locals() else "unknown",
                    e,
                )
                raise HTTPException(
                    status_code=500, detail="Capture webhook processing failed"
                )

            return {"status": "ok"}

        # other events -> ignore
        api_logger.info("Webhook ignored: event_type %s", event_type)
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
    api_logger.info(f"🚫 PayPal payment cancelled: {token or 'no-token'}")
    
    # Best-effort: mark the local order as cancelled and try voiding any authorization
    try:
        if token:
            try:
                supabase.table("orders").update({"status": "cancelled"}).eq(
                    "gateway_order_id", token
                ).execute()
                api_logger.info("Marked order with gateway_order_id %s as cancelled", token)
            except Exception:
                api_logger.exception("Failed to mark order cancelled for token %s", token)

            try:
                # Attempt to void any existing auth related to this order (safe if none)
                await paypal_void_checkout(token)
            except Exception:
                api_logger.exception("Failed to void checkout for token %s (non-fatal)", token)
    except Exception as e:
        log_error_with_context(api_logger, e, "handling PayPal cancel")

    params = urlencode({"status": "cancel", "orderId": token or ""})
    return RedirectResponse(
        url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
    )

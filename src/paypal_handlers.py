import httpx
import json
from typing import Optional, Dict, Any, List
from shippo.models import components
from urllib.parse import urlencode
from datetime import datetime, timezone
from fastapi import HTTPException, Request
from starlette.responses import RedirectResponse
from decimal import Decimal, InvalidOperation

from src.shippo_handlers import shippo_sdk
from src.models.shippo import TransactionPayload
from src.utils.supabase import supabase
from src.utils.currency import format_price
from src.utils.shippo import (
    shippo_get_rate_details,
    shippo_suggestion_format,
    shippo_validate_address,
    MultipleAddressesException,
)
from src.utils.paypal import (
    get_access_token,
    get_base_url,
    get_api_base,
    paypal_create_subscription,
    paypal_create_order,
    paypal_void_checkout,
)
from src.utils.logger import (
    api_logger,
    log_api_request,
    log_error_with_context,
)
from src.utils.safe_handler import safe_handler


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

        # Remove all payment insertion here
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
            api_logger.error("PayPal webhook missing subscription ID")
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
                    "Missing billing_agreement_id in PAYMENT.SALE.COMPLETED"
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
        log_error_with_context(api_logger, e, "Webhook error")
        raise HTTPException(status_code=400, detail="Invalid webhook payload")


# ===============================================================
# /paypal/order
# ===============================================================


# -------------------------
# Main handler (refactored + logging everywhere)
# -------------------------
@safe_handler(default_status=500, default_detail="PayPal order processing failed")
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
            supabase.table("vw_chaamo_cards")
            .select("*, user_card: user_cards(custom_name)")
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
    # Prevent duplicate pending order for same buyer+listing (non-fatal)
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
        if getattr(existing_pending, "data", None) and len(existing_pending.data) > 0:
            existing = existing_pending.data[0]
            api_logger.info(
                "Returning existing awaiting_payment order for buyer=%s listing=%s",
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
        except MultipleAddressesException as e:
            # Shippo found multiple address matches but doesn't provide suggestions
            # Guide user to provide more specific address details
            api_logger.info(
                "Shippo returned multiple address matches for buyer %s - asking for more specific details",
                buyer_id,
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Address validation failed",
                    "message": "Your address matched multiple records. Please provide more specific details.",
                    "suggestions": [
                        "Add apartment/suite number if applicable",
                        "Double-check street name spelling", 
                        "Verify city and state/province",
                        "Ensure postal/ZIP code is complete and correct"
                    ],
                    "address_provided": {
                        "street1": shipping_address.get("street1"),
                        "street2": shipping_address.get("street2"),
                        "city": shipping_address.get("city"),
                        "state": shipping_address.get("state"),
                        "zip": shipping_address.get("zip"),
                        "country": shipping_address.get("country")
                    }
                },
            )
        except HTTPException:
            # Re-raise HTTP exceptions (including those from MultipleAddressesException handling above)
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
    # Verify selected rate against Shippo server-side
    # -------------------------
    try:
        rate_details = shippo_get_rate_details(selected_rate_id)
        if not rate_details:
            api_logger.warning(
                "Rate details not found for rate id: %s", selected_rate_id
            )
            raise HTTPException(status_code=400, detail="Rate details not found")

        server_rate_amount = rate_details["amount"]
        server_rate_currency = rate_details["currency"].upper()

        if selected_rate_currency.upper() != server_rate_currency:
            converted_server_rate = format_price(
                from_currency=server_rate_currency,
                to_currency=selected_rate_currency,
                amount=float(server_rate_amount),
            )
            if converted_server_rate is None:
                api_logger.error(
                    "Failed to convert server rate currency from %s to %s",
                    server_rate_currency,
                    selected_rate_currency,
                )
                raise HTTPException(
                    status_code=500, detail="Rate currency conversion failed"
                )
            server_rate_decimal = Decimal(str(converted_server_rate))
        else:
            server_rate_decimal = Decimal(str(server_rate_amount))

        client_rate_decimal = Decimal(str(selected_rate_amount))
        tolerance = Decimal("0.01")
        if abs(client_rate_decimal - server_rate_decimal) > tolerance:
            api_logger.warning(
                "Client rate (%s) differs from server rate (%s) for rate id %s",
                client_rate_decimal,
                server_rate_decimal,
                selected_rate_id,
            )
            raise HTTPException(
                status_code=400,
                detail="Selected shipping rate has changed. Please refresh and select a new rate.",
            )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception("Rate validation failed: %s", e)
        raise HTTPException(status_code=400, detail="Invalid shipping rate")

    # -------------------------
    # Format all amounts into buyer currency
    # -------------------------
    try:
        shipping_fee = format_price(
            from_currency=selected_rate_currency,
            to_currency=buyer_currency,
            amount=selected_rate_amount,
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
    # Create initial order record (awaiting_payment) - DB authoritative
    # -------------------------
    try:
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
            "shipping_rate_id": selected_rate_id,
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
        api_logger.exception("Create order failed (initial insert): %s", e)
        raise HTTPException(status_code=500, detail="Create order error")

    # -------------------------
    # Create Paypal Approve URL and update order with gateway fields
    # -------------------------
    try:
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
            item_name = (
                listing.get("user_card", {}).get("custom_name", "").strip()
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

        # Update order with PayPal data (checkout url + id)
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

        status = "success" if final_token else "cancel"
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

            # Create shipping transaction
            try:
                # Get the shipping address from the order
                shipping_address = order.get("shipping_address")
                if not shipping_address:
                    raise HTTPException(
                        status_code=400, detail="No shipping address found in order"
                    )

                # Create transaction with the address
                transaction = shippo_sdk.transactions.create(
                    components.TransactionCreateRequest(
                        rate=order["shipping_rate_id"],
                        label_file_type="PDF",
                        async_=False,
                        metadata={"order_id": order_id, "buyer_id": order["buyer_id"]},
                    )
                )

                # Check transaction status
                tx_status = getattr(transaction, "status", "")
                if tx_status != "SUCCESS":
                    messages = getattr(transaction, "messages", [])
                    error_msg = (
                        messages[0].text if messages else "Shipping service error"
                    )

                    # Update order with error details (using 'cancelled' status as schema doesn't have 'shipping_failed')
                    supabase.table("orders").update({"status": "cancelled"}).eq(
                        "id", order_id
                    ).execute()

                    raise HTTPException(
                        status_code=400, detail=f"Shipping failed: {error_msg}"
                    )
            except HTTPException:
                raise
            except Exception as e:
                error_msg = str(e)
                api_logger.error(f"Shippo transaction failed: {error_msg}")

                # Check for specific error messages
                if "failed_address_validation" in error_msg.lower():
                    raise HTTPException(
                        status_code=400,
                        detail="The shipping address could not be verified. Please check the address and try again.",
                    )
                # Handle multiple address matches consistently with main handler
                elif "multiple records" in error_msg.lower() or "multiple addresses" in error_msg.lower():
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Address validation failed",
                            "message": "Shipping address matched multiple records. Please provide more specific details.",
                            "suggestions": [
                                "Add apartment/suite number if applicable",
                                "Double-check street name spelling", 
                                "Verify city and state/province",
                                "Ensure postal/ZIP code is complete and correct"
                            ]
                        },
                    )

                # Generic error for other cases
                raise HTTPException(
                    status_code=500,
                    detail="Shipping service temporarily unavailable. Please try again later.",
                )

            # 3) Update order status to awaiting_shipment
            try:
                update_data = {
                    "status": "awaiting_shipment",
                    "shipping_transaction_id": transaction.object_id,
                    "shipping_tracking_number": transaction.tracking_number,
                    "shipping_tracking_url": transaction.tracking_url_provider,
                    "shipping_label_url": transaction.label_url,
                }
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

            # 4) Simple success response

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
        log_error_with_context(api_logger, e, "handling PayPal cancel")

    params = urlencode({"status": "cancel", "orderId": token or ""})
    return RedirectResponse(
        url=f"{redirect}{'&' if '?' in redirect else '?'}{params}", status_code=302
    )

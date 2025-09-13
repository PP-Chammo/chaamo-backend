import os
import uuid
import httpx
from decimal import Decimal
from fastapi import HTTPException, Request
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from src.utils.logger import api_logger
from src.utils.supabase import supabase

PAYPAL_ENV = "sandbox"


# ===============================================================
# Environment helpers
# ===============================================================
def paypal_get_client_id() -> Optional[str]:
    return os.getenv("PAYPAL_CLIENT_ID")


def paypal_get_client_secret() -> Optional[str]:
    return os.getenv("PAYPAL_SECRET")


def paypal_get_env() -> str:
    return (os.getenv("PAYPAL_ENV") or PAYPAL_ENV).lower()


def get_base_url(request: Request) -> str:
    # e.g. https://chaamo-backend.fly.dev
    return str(request.base_url).rstrip("/")


def get_api_base() -> str:
    env = paypal_get_env()
    return (
        "https://api-m.paypal.com"
        if env == "live"
        else "https://api-m.sandbox.paypal.com"
    )


def get_credentials() -> Tuple[str, str]:
    client_id: Optional[str] = paypal_get_client_id()
    secret: Optional[str] = paypal_get_client_secret()

    api_logger.info(
        f"PayPal credentials check - Client ID present: {bool(client_id)}, Secret present: {bool(secret)}"
    )
    api_logger.info(f"PayPal environment: {paypal_get_env()}")
    api_logger.info(f"PayPal API base: {get_api_base()}")

    if not client_id or not secret:
        raise ValueError("PAYPAL_CLIENT_ID and PAYPAL_SECRET are required")
    return client_id, secret


# ===============================================================
# Get Access Token
# ===============================================================
async def get_access_token() -> str:
    client_id, secret = get_credentials()
    token_url = f"{get_api_base()}/v1/oauth2/token"
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            token_url,
            auth=(client_id, secret),
            headers={
                "Accept": "application/json",
                "Accept-Language": "en_US",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
        )
        if resp.status_code >= 400:
            # Surface PayPal error details to aid debugging (e.g., credential/env mismatch)
            try:
                err = resp.json()
            except Exception:
                err = {"error": resp.text[:500]}
            raise httpx.HTTPStatusError(
                f"PayPal token error {resp.status_code}: {err}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()
        return data["access_token"]


# ===============================================================
# Create Subscription
# ===============================================================


async def paypal_create_subscription(
    plan_id: str,
    return_url: str,
    cancel_url: str,
    user_details: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    try:
        token = await get_access_token()
        payload: Dict[str, Any] = {
            "plan_id": plan_id,
            "application_context": {
                "brand_name": "Chaamo Gold Membership",
                "locale": "en-US",
                "shipping_preference": "NO_SHIPPING",
                "user_action": "SUBSCRIBE_NOW",
                "return_url": return_url,
                "cancel_url": cancel_url,
            },
        }

        # optional subscriber info (helps PayPal prefill)
        if user_details:
            payload["subscriber"] = {
                "name": {
                    "given_name": user_details.get("first_name", ""),
                    "surname": user_details.get("last_name", ""),
                },
                "email_address": user_details.get("email", ""),
            }

        request_id = str(uuid.uuid4())  # PayPal-Request-Id for idempotency
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "PayPal-Request-Id": request_id,
        }

        api_logger.info("Creating PayPal subscription (create-subscription).")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{get_api_base().rstrip('/')}/v1/billing/subscriptions",
                headers=headers,
                json=payload,
            )
            api_logger.info(f"PayPal create-sub resp: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()

        approval_url = None
        for link in data.get("links", []):
            if link.get("rel") == "approve":
                approval_url = link.get("href")
                break

        return {
            "subscription_id": data.get("id"),
            "approval_url": approval_url,
            "raw_response": data,
            "request_id": request_id,
        }
    except Exception as e:
        api_logger.exception("PayPal subscription create failed: %s", e)
        # bubble up meaningful HTTPException if desired by caller
        raise HTTPException(
            status_code=500, detail=f"Failed creating PayPal subscription: {e}"
        )


# ===============================================================
# Create Order
# ===============================================================
async def paypal_create_order(
    amount: str,
    currency: str,
    return_url: str,
    cancel_url: str,
    *,
    amount_breakdown: Optional[Dict[str, Any]] = None,
    items: Optional[list] = None,
    brand_name: Optional[str] = "Chaamo",
    user_action: Optional[str] = "PAY_NOW",
) -> Dict[str, Any]:
    """
    Create PayPal order with intent=AUTHORIZE (so we can authorize -> void/capture later).
    Returns dict { "paypal_order_id": <id>, "paypal_checkout_url": <approval_url>, "raw": <resp.json()> }
    """
    try:
        token = await get_access_token()
        url = f"{get_api_base().rstrip('/')}/v2/checkout/orders"
        api_logger.info(f"Creating PayPal order: {amount} {currency} at {url}")
    except Exception as e:
        api_logger.error(f"Failed to get PayPal access token: {e}")
        raise
    payload = {
        "intent": "AUTHORIZE",
        "purchase_units": [
            {
                "amount": {
                    "currency_code": currency,
                    "value": f"{Decimal(amount).quantize(Decimal('0.01'))}",
                },
            }
        ],
        "application_context": {
            "return_url": return_url,
            "cancel_url": cancel_url,
            "brand_name": brand_name or "Chaamo",
            "user_action": user_action or "PAY_NOW",
        },
    }

    # Enrich purchase unit with amount breakdown and items when provided
    try:
        pu = payload["purchase_units"][0]
        if amount_breakdown:
            # Ensure values are strings with 2 decimals as PayPal expects
            bd = {}
            for k, v in amount_breakdown.items():
                if isinstance(v, dict):
                    # Already structured: {value, currency_code}
                    bd[k] = v
                else:
                    bd[k] = {
                        "value": f"{Decimal(str(v)).quantize(Decimal('0.01'))}",
                        "currency_code": currency,
                    }
            pu["amount"]["breakdown"] = bd
        if items:
            pu["items"] = items
    except Exception:
        # non-fatal: keep minimal payload if enrichment fails
        api_logger.exception(
            "Failed to enrich PayPal purchase unit; using minimal payload"
        )

    api_logger.debug(f"PayPal order payload: {payload}")
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    api_logger.info(f"PayPal order creation response: {resp.status_code}")

    if resp.status_code >= 400:
        try:
            error_data = resp.json()
            api_logger.error(f"PayPal order creation failed: {error_data}")
        except Exception:
            api_logger.error(f"PayPal order creation failed: {resp.text}")
        resp.raise_for_status()

    data = resp.json()
    api_logger.debug(f"PayPal order creation success: {data}")

    # Find approve/checkout link
    approve_link = None
    for l in data.get("links", []) or []:
        if l.get("rel") == "approve":
            approve_link = l.get("href")
            break

    if not approve_link:
        api_logger.error(f"No approval link found in PayPal response: {data}")
        raise HTTPException(
            status_code=500, detail="PayPal order created but no approval link found"
        )

    return {
        "paypal_order_id": data.get("id"),
        "paypal_checkout_url": approve_link,
        "raw": data,
    }


# ===============================================================
# Capture Order
# ===============================================================
async def paypal_capture_order(paypal_order_id: str) -> Dict[str, Any]:
    access_token = await get_access_token()
    capture_url = (
        f"{get_api_base().rstrip('/')}/v2/checkout/orders/{paypal_order_id}/capture"
    )
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            capture_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={},  # PayPal capture endpoint expects empty JSON body
        )
    resp.raise_for_status()
    return resp.json()


# ===============================================================
# Authorize Order
# ===============================================================
async def paypal_authorize_order(order_id: str) -> Dict[str, Any]:
    """
    POST /v2/checkout/orders/{order_id}/authorize
    Returns the parsed JSON response which includes authorizations.
    """
    token = await get_access_token()
    url = f"{get_api_base().rstrip('/')}/v2/checkout/orders/{order_id}/authorize"
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={},
        )
    resp.raise_for_status()
    return resp.json()


# ===============================================================
# Void Authorization
# ===============================================================
async def paypal_void_authorization(authorization_id: str) -> bool:
    """
    Void an authorization:
    POST /v2/payments/authorizations/{authorization_id}/void
    Returns True if void succeeded (204 expected), raises on non-2xx.
    """
    token = await get_access_token()
    url = f"{get_api_base().rstrip('/')}/v2/payments/authorizations/{authorization_id}/void"
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={},
        )
    if resp.status_code in (200, 204):
        return True
    resp.raise_for_status()
    return True


# ===============================================================
# Capture Authorization
# ===============================================================
async def paypal_capture_authorization(
    authorization_id: str,
    amount: Optional[Decimal] = None,
    currency: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Capture an authorization:
    POST /v2/payments/authorizations/{authorization_id}/capture
    If amount provided, pass it; otherwise let PayPal do full capture.
    """
    token = await get_access_token()
    url = f"{get_api_base().rstrip('/')}/v2/payments/authorizations/{authorization_id}/capture"
    payload: Dict[str, Any] = {}
    if amount is not None and currency:
        payload["amount"] = {
            "value": f"{amount.quantize(Decimal('0.01'))}",
            "currency_code": currency,
        }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    resp.raise_for_status()
    return resp.json()


# ===============================================================
# Void Checkout
# ===============================================================
async def paypal_void_checkout(order_id: str) -> bool:
    """
    Backwards-compatible wrapper: if your orders table stores gateway_authorization_id
    (or similar) use it; otherwise this will attempt to call /v2/checkout/orders/{id}/authorize
    to get an authorization and then void it (safe approach).
    NOTE: if you never authorized the order, there may be nothing to void.
    """
    # Try to fetch our local order to find authorization id (adjust field name if different)
    try:
        order_resp = (
            supabase.table("orders")
            .select("id, gateway_authorization_id")
            .eq("gateway_order_id", order_id)
            .execute()
        )
        if order_resp.data and len(order_resp.data) > 0:
            order = order_resp.data[0]
            auth_id = order.get("gateway_authorization_id") or order.get(
                "authorization_id"
            )
            if auth_id:
                return await paypal_void_authorization(auth_id)
    except Exception:
        api_logger.exception(
            "Failed to lookup local order for authorization_id (non-fatal)"
        )

    # As fallback: try to create an authorization for the order and void it immediately.
    # Only works if order intent=AUTHORIZE and the authorize call returns authorizations.
    try:
        auth_resp = await paypal_authorize_order(order_id)
        # navigate to authorizations array
        pus = auth_resp.get("purchase_units", []) or []
        for pu in pus:
            payments = pu.get("payments", {}) or {}
            authorizations = payments.get("authorizations", []) or []
            if authorizations:
                # void all found authorizations (best-effort)
                success_all = True
                for a in authorizations:
                    aid = a.get("id")
                    if aid:
                        try:
                            await paypal_void_authorization(aid)
                        except Exception:
                            api_logger.exception(f"Failed to void authorization {aid}")
                            success_all = False
                return success_all
        # If no authorizations present, nothing to void
        api_logger.info(
            f"No authorizations found when authorizing order {order_id}; nothing to void."
        )
        return False
    except httpx.HTTPStatusError as he:
        api_logger.exception(f"Failed to call authorize for order {order_id}: {he}")
        return False
    except Exception:
        api_logger.exception("Unexpected error in paypal_void_checkout fallback")
        return False


# ===============================================================
# Cancel Subscription
# ===============================================================
async def paypal_cancel_subscription(subscription_id: str, reason: str | None = None) -> bool:
    """
    Cancel a PayPal billing subscription.

    POST /v1/billing/subscriptions/{id}/cancel
    Returns True if cancellation succeeded (204 expected), raises on non-2xx otherwise.
    """
    token = await get_access_token()
    url = f"{get_api_base().rstrip('/')}/v1/billing/subscriptions/{subscription_id}/cancel"
    payload = {"reason": (reason or "User requested cancellation")}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if resp.status_code in (200, 202, 204):
        return True
    try:
        err = resp.json()
    except Exception:
        err = {"error": resp.text[:500]}
    api_logger.error("PayPal cancel subscription failed %s: %s", resp.status_code, err)
    resp.raise_for_status()
    return False


# ===============================================================
# Helper utilities (subscription flow) — migrated from handlers
# ===============================================================
def paypal_parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except Exception:
        return None


async def paypal_fetch_plan_fixed_price(plan_id_pp: Optional[str]) -> tuple:
    """Return (amount, currency) for a PayPal billing plan's first billing cycle.
    Defaults to (0, 'USD') on failure."""
    if not plan_id_pp:
        return 0, "USD"
    try:
        token2 = await get_access_token()
        async with httpx.AsyncClient(timeout=20) as client:
            pres = await client.get(
                f"{get_api_base().rstrip('/')}/v1/billing/plans/{plan_id_pp}",
                headers={"Authorization": f"Bearer {token2}"},
            )
            if pres.status_code == 200:
                pjson = pres.json() or {}
                cycles = pjson.get("billing_cycles") or []
                if cycles:
                    fixed = (
                        (cycles[0].get("pricing_scheme") or {}).get("fixed_price")
                        if isinstance(cycles[0], dict)
                        else None
                    )
                    if fixed:
                        return (
                            fixed.get("value") or 0,
                            fixed.get("currency_code") or "USD",
                        )
    except Exception:
        api_logger.exception("Failed to fetch PayPal plan price; defaulting")
    return 0, "USD"


async def paypal_fetch_price_by_subscription_id(paypal_subscription_id: str) -> tuple:
    """Fetch (amount, currency) for the plan behind a PayPal subscription id.
    Defaults to (0, 'USD') on failure."""
    try:
        token2 = await get_access_token()
        async with httpx.AsyncClient(timeout=20) as client:
            sres = await client.get(
                f"{get_api_base().rstrip('/')}/v1/billing/subscriptions/{paypal_subscription_id}",
                headers={"Authorization": f"Bearer {token2}"},
            )
            if sres.status_code == 200:
                sjson = sres.json() or {}
                plan_id_pp = sjson.get("plan_id")
                return await paypal_fetch_plan_fixed_price(plan_id_pp)
    except Exception:
        api_logger.exception("Failed to fetch price by subscription id; defaulting")
    return 0, "USD"


def paypal_compute_adjusted_start(user_id: str, plan_id: str, exclude_sub_id: Optional[str]) -> datetime:
    """Return a start datetime that does not overlap any existing active periods for (user, plan)."""
    now = datetime.now()
    try:
        overlap_rows = (
            supabase.table("subscriptions")
            .select("id, start_date, end_date")
            .eq("user_id", user_id)
            .eq("plan_id", plan_id)
            .eq("status", "active")
            .execute()
        )
        latest_end = None
        for r in getattr(overlap_rows, "data", None) or []:
            if exclude_sub_id and r.get("id") == exclude_sub_id:
                continue
            e = paypal_parse_iso_dt(r.get("end_date"))
            if e and e > now and (latest_end is None or e > latest_end):
                latest_end = e
        if latest_end and latest_end > now:
            return latest_end
    except Exception:
        api_logger.exception("Failed to compute adjusted start; using now")
    return now


async def paypal_ensure_linked_pending_payment(
    sub_id: str,
    user_id: str,
    gateway_order_id: str,
    amount,
    currency: str,
    gateway_account_info: Optional[dict] = None,
):
    """Ensure subscriptions.payment_id is set by creating a pending payment if needed."""
    try:
        sr = (
            supabase.table("subscriptions")
            .select("payment_id")
            .eq("id", sub_id)
            .limit(1)
            .execute()
        )
        if getattr(sr, "data", None) and sr.data[0].get("payment_id"):
            return
        ins = (
            supabase.table("payments")
            .insert(
                {
                    "user_id": user_id,
                    "gateway": "paypal",
                    "gateway_order_id": gateway_order_id,
                    "amount": amount,
                    "currency": (currency or "USD"),
                    "gateway_account_info": gateway_account_info or None,
                    "status": "pending",
                }
            )
            .execute()
        )
        if getattr(ins, "data", None):
            pid = ins.data[0].get("id")
            supabase.table("subscriptions").update({"payment_id": pid}).eq(
                "id", sub_id
            ).execute()
    except Exception:
        api_logger.exception("Failed to ensure linked pending payment")

import base64
import os
from typing import Optional, Tuple

import httpx


PAYPAL_ENV = os.environ.get("PAYPAL_ENV", "sandbox").lower()


def _api_base() -> str:
    return "https://api-m.paypal.com" if PAYPAL_ENV == "live" else "https://api-m.sandbox.paypal.com"


def _get_credentials() -> Tuple[str, str]:
    client_id: Optional[str] = os.environ.get("PAYPAL_CLIENT_ID")
    secret: Optional[str] = os.environ.get("PAYPAL_SECRET")
    if not client_id or not secret:
        raise ValueError("PAYPAL_CLIENT_ID and PAYPAL_SECRET are required")
    return client_id, secret


async def get_access_token() -> str:
    client_id, secret = _get_credentials()
    token_url = f"{_api_base()}/v1/oauth2/token"
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
                f"PayPal token error {resp.status_code}: {err}", request=resp.request, response=resp
            )
        data = resp.json()
        return data["access_token"]


async def create_order(
    amount: str,
    currency: str,
    return_url: str,
    cancel_url: str,
) -> dict:
    token = await get_access_token()
    url = f"{_api_base()}/v2/checkout/orders"

    payload = {
        "intent": "CAPTURE",
        "purchase_units": [
            {
                "amount": {
                    "currency_code": currency,
                    "value": amount,
                }
            }
        ],
        "application_context": {
            "brand_name": "Chaamo",
            "user_action": "PAY_NOW",
            "return_url": return_url,
            "cancel_url": cancel_url,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
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


async def capture_order(order_id: str) -> dict:
    token = await get_access_token()
    url = f"{_api_base()}/v2/checkout/orders/{order_id}/capture"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()

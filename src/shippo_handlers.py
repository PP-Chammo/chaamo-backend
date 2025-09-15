from typing import Optional, List
from datetime import datetime
from fastapi import HTTPException, Request
from shippo.models import components

from src.models.shippo import RateOption, RateResponse
from src.utils.supabase import supabase
from src.utils.logger import (
    api_logger,
)
from src.utils.shippo import (
    shippo_validate_address,
    shippo_sdk,
    shippo_is_test,
    SHIPPO_API_KEY,
)


# ===============================================================
# /shippo/rates
# ===============================================================
async def shippo_rates_handler(
    seller_id: str,
    buyer_id: str,
    insurance: Optional[bool] = False,
    insurance_currency: Optional[str] = None,
    insurance_amount: Optional[float] = 0.0,
) -> RateResponse:
    if not SHIPPO_API_KEY:
        raise HTTPException(status_code=500, detail="SHIPPO_API_KEY not set")

    # -------------------------
    # Fetch seller & buyer data (defensive: check .data exists)
    # -------------------------
    try:
        seller_resp = (
            supabase.table("profiles")
            .select("username, email, phone_number")
            .eq("id", seller_id)
            .execute()
        )
        if not seller_resp or not getattr(seller_resp, "data", None):
            raise HTTPException(
                status_code=404, detail=f"Seller profile not found: {seller_id}"
            )
        seller = seller_resp.data[0]
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception(f"Seller lookup failed: {e}")
        raise HTTPException(status_code=500, detail="Seller lookup failed")

    try:
        seller_addr_resp = (
            supabase.table("user_addresses")
            .select(
                "address_line_1, address_line_2, city, state_province, postal_code, country"
            )
            .eq("user_id", seller_id)
            .execute()
        )
        if not seller_addr_resp or not getattr(seller_addr_resp, "data", None):
            raise HTTPException(
                status_code=404, detail=f"Seller address not found, please ask seller to add it in their profile."
            )
        seller_addr = seller_addr_resp.data[0]
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception(f"Seller address lookup failed: {e}")
        raise HTTPException(status_code=500, detail="Seller address lookup failed")

    try:
        buyer_resp = (
            supabase.table("profiles")
            .select("username, email, phone_number")
            .eq("id", buyer_id)
            .execute()
        )
        if not buyer_resp or not getattr(buyer_resp, "data", None):
            raise HTTPException(
                status_code=404, detail=f"Buyer profile not found: {buyer_id}"
            )
        buyer = buyer_resp.data[0]
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception(f"Buyer lookup failed: {e}")
        raise HTTPException(status_code=500, detail="Buyer lookup failed")

    try:
        buyer_addr_resp = (
            supabase.table("user_addresses")
            .select(
                "address_line_1, address_line_2, city, state_province, postal_code, country"
            )
            .eq("user_id", buyer_id)
            .execute()
        )
        if not buyer_addr_resp or not getattr(buyer_addr_resp, "data", None):
            raise HTTPException(
                status_code=404, detail=f"Your address not found, please add it in your profile."
            )
        buyer_addr = buyer_addr_resp.data[0]
    except HTTPException:
        raise
    except Exception as e:
        api_logger.exception(f"Buyer address lookup failed: {e}")
        raise HTTPException(status_code=500, detail="Buyer address lookup failed")

    # -------------------------
    # Build, validate and normalize addresses via Shippo helper
    # -------------------------
    def _raw_address_payload(user, addr):
        return {
            "name": (user.get("username") or "").strip(),
            "email": user.get("email") or None,
            "phone": user.get("phone_number") or None,
            "street1": (addr.get("address_line_1") or "").strip(),
            "street2": (addr.get("address_line_2") or "").strip() or "",
            "city": addr.get("city") or "",
            "state": addr.get("state_province") or "",
            "zip": addr.get("postal_code") or "",
            "country": (addr.get("country") or "").upper(),
        }

    try:
        validated_seller = await shippo_validate_address(
            _raw_address_payload(seller, seller_addr)
        )
        validated_buyer = await shippo_validate_address(
            _raw_address_payload(buyer, buyer_addr)
        )
    except HTTPException:
        # Bubble up specific HTTP errors (auth, rate limits, validation, etc.)
        raise
    except Exception as e:
        api_logger.exception(f"Shippo address validation failed: {e}")
        raise HTTPException(
            status_code=500, detail="Shipping address validation failed"
        )

    # Helper to safely extract fields from dict/object
    def _get_field(obj, field, default=None):
        try:
            if isinstance(obj, dict):
                return obj.get(field, default)
            return getattr(obj, field, default)
        except Exception:
            return default

    # Convert normalized results back into AddressCreateRequest for shipment creation
    seller_address = components.AddressCreateRequest(
        name=_get_field(validated_seller, "name")
        or (seller.get("username") or "").strip(),
        email=_get_field(validated_seller, "email") or (seller.get("email") or ""),
        phone=_get_field(validated_seller, "phone")
        or (seller.get("phone_number") or ""),
        street1=_get_field(validated_seller, "street1")
        or (seller_addr.get("address_line_1") or "").strip(),
        street2=_get_field(validated_seller, "street2")
        or (seller_addr.get("address_line_2") or "").strip()
        or None,
        city=_get_field(validated_seller, "city") or (seller_addr.get("city") or ""),
        state=_get_field(validated_seller, "state")
        or (seller_addr.get("state_province") or ""),
        zip=_get_field(validated_seller, "zip")
        or (seller_addr.get("postal_code") or ""),
        country=_get_field(validated_seller, "country")
        or (seller_addr.get("country") or "").upper(),
    )
    buyer_address = components.AddressCreateRequest(
        name=_get_field(validated_buyer, "name")
        or (buyer.get("username") or "").strip(),
        email=_get_field(validated_buyer, "email") or (buyer.get("email") or ""),
        phone=_get_field(validated_buyer, "phone") or (buyer.get("phone_number") or ""),
        street1=_get_field(validated_buyer, "street1")
        or (buyer_addr.get("address_line_1") or "").strip(),
        street2=_get_field(validated_buyer, "street2")
        or (buyer_addr.get("address_line_2") or "").strip()
        or None,
        city=_get_field(validated_buyer, "city") or (buyer_addr.get("city") or ""),
        state=_get_field(validated_buyer, "state")
        or (buyer_addr.get("state_province") or ""),
        zip=_get_field(validated_buyer, "zip") or (buyer_addr.get("postal_code") or ""),
        country=_get_field(validated_buyer, "country")
        or (buyer_addr.get("country") or "").upper(),
    )

    # -------------------------
    # Create static parcel payload for card item
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
    # Insurance Logic
    # -------------------------
    shippo_insurance_amount = None
    shippo_insurance_currency = None
    if insurance and insurance_amount and float(insurance_amount) > 0:
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
        test=shippo_is_test,
    )

    try:
        shipment = shippo_sdk.shipments.create(shipment_request)
    except Exception as e:
        api_logger.exception(f"Error creating shipment: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error creating shipment: {str(e)}"
        )

    # -------------------------
    # Check if shipment has rates
    # -------------------------
    # Shipment may be dict-like or object-like depending on SDK version
    def get_attr(obj, key, default=None):
        try:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        except Exception:
            return default

    raw_rates = get_attr(shipment, "rates", None) or []
    if not raw_rates:
        api_logger.error(f"No rates found in shipment: {shipment}")
        raise HTTPException(status_code=502, detail="No shipping rates available")

    # -------------------------
    # Improved validation & logging
    # -------------------------
    shipment_msgs = get_attr(shipment, "messages", []) or []
    normalized_msgs = [((get_attr(m, "text", "") or "").lower()) for m in shipment_msgs]

    # Fail fast if carrier reports ambiguous or invalid address at shipment stage
    if any(
        (
            ("failed_address_validation" in msg)
            or ("multiple addresses" in msg)
            or ("multiple records" in msg)
        )
        for msg in normalized_msgs
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Address validation failed",
                "message": "The address matched multiple records. Please provide more specific details.",
                "suggestions": [
                    "Add apartment/suite number if applicable",
                    "Double-check street name spelling",
                    "Verify city and state/province",
                    "Ensure postal/ZIP code is complete and correct",
                ],
            },
        )
    api_logger.debug("Shipment messages: %s", normalized_msgs)

    # Log available raw rates for debugging
    for r in raw_rates:
        try:
            api_logger.debug(
                "Raw rate: id=%s provider=%s service=%s amount=%s msgs=%s carrier_account=%s test=%s",
                get_attr(r, "object_id"),
                get_attr(r, "provider"),
                get_attr(get_attr(r, "servicelevel", None), "name", None),
                get_attr(r, "amount_local") or get_attr(r, "amount"),
                get_attr(r, "messages"),
                get_attr(r, "carrier_account"),
                get_attr(r, "test"),
            )
        except Exception:
            pass

    # Keep only rates that do NOT have per-rate messages (explicit failures)
    filtered_rates = []
    for r in raw_rates:
        messages = get_attr(r, "messages", None)
        # treat non-empty messages as failure for that rate
        if messages:
            api_logger.debug(
                "Excluding rate due to messages: id=%s msgs=%s",
                get_attr(r, "object_id"),
                messages,
            )
            continue
        filtered_rates.append(r)

    if not filtered_rates:
        api_logger.error(
            "No supported rates after filtering per-rate messages. shipment_messages=%s",
            normalized_msgs,
        )
        raise HTTPException(status_code=502, detail="No shipping rates available")

    # -------------------------
    # Parse rates (from filtered list)
    # -------------------------
    parsed: List[RateOption] = []
    for r in filtered_rates:
        try:
            # prefer amount_local, fallback to amount; handle dict/object cases
            raw_amount = get_attr(r, "amount_local", None) or get_attr(
                r, "amount", None
            )
            # If amount is an object like {'amount': '1.00', 'currency': 'USD'} adjust accordingly
            if isinstance(raw_amount, dict):
                raw_amount_val = raw_amount.get("amount") or raw_amount.get("value")
            else:
                raw_amount_val = raw_amount

            amount_float = float(raw_amount_val)
            currency = get_attr(r, "currency_local") or get_attr(r, "currency") or ""
            parsed.append(
                RateOption(
                    id=get_attr(r, "object_id"),
                    service=(
                        get_attr(get_attr(r, "servicelevel", None), "name", "") or ""
                    ),
                    courier=get_attr(r, "provider") or "",
                    amount=amount_float,
                    currency=currency,
                    estimated_days=get_attr(r, "estimated_days"),
                )
            )
        except Exception as ex:
            api_logger.debug("Skipping malformed rate: %s; error: %s", r, ex)

    parsed.sort(key=lambda x: x.amount)
    if not parsed:
        api_logger.error("No valid rates parsed")
        raise HTTPException(status_code=502, detail="No shipping rates available")

    # -------------------------
    # Return structured RateResponse
    # -------------------------
    shipment_id = get_attr(shipment, "object_id") or get_attr(
        shipment, "object_id", None
    )
    return RateResponse(shipment_id=shipment_id, rates=parsed)


# ===============================================================
# /shippo/webhooks
# ===============================================================
async def shippo_webhook_handler(request: Request):
    body = await request.json()
    api_logger.info("Shippo webhook received: %s", body)
    if body.get("event") == "transaction.updated":
        data = body.get("data", {})
        if data.get("status") == "SUCCESS":
            supabase.table("orders").update(
                {
                    "status": "awaiting_shipment",
                    "tracking_number": data.get("tracking_number"),
                    "tracking_url": data.get("tracking_url_provider"),
                    "shipping_label_url": data.get("label_url"),
                    "paid_at": datetime.now().isoformat(),
                }
            ).eq("shipping_transaction_id", data.get("object_id")).execute()

    return {"ok": True}

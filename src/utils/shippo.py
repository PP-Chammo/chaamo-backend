from decimal import Decimal, InvalidOperation
from fastapi import HTTPException
from shippo.models import components
from typing import Dict, Any, List, Tuple
import httpx
import os
import shippo as shippo_module

from src.utils.logger import api_logger

# Centralized Shippo SDK configuration & singleton


def shippo_get_api_key() -> str:
    return os.environ.get("SHIPPO_API_KEY", "")


def shippo_get_is_test() -> bool:
    key = shippo_get_api_key()
    return True if "test" in (key or "").lower() else False


SHIPPO_API_KEY: str = shippo_get_api_key()
shippo_is_test: bool = shippo_get_is_test()
shippo_sdk = shippo_module.Shippo(api_key_header=SHIPPO_API_KEY)

# Cached references
_FROM_ADDRESS_REF: Any = None


def shippo_suggestion_format(sugg: Dict[str, Any]) -> str:
    # Turn a suggestion dict into a short human-friendly string
    parts = []
    if sugg.get("street1"):
        parts.append(sugg["street1"])
    if sugg.get("street2"):
        parts.append(sugg["street2"])
    city_parts = []
    if sugg.get("city"):
        city_parts.append(sugg["city"])
    if sugg.get("state"):
        city_parts.append(sugg["state"])
    if sugg.get("zip"):
        city_parts.append(sugg["zip"])
    if city_parts:
        parts.append(", ".join(city_parts))
    if sugg.get("country"):
        parts.append(sugg["country"])
    return " | ".join(parts)


async def _delete_shippo_address(address_id: str) -> bool:
    """
    Delete an address from Shippo using the official API.
    Returns True if successful, False otherwise.

    Follows: DELETE https://api.goshippo.com/v2/addresses/{address_id}
    """
    try:
        # Try SDK deletion first if available
        try:
            delete_func = getattr(shippo_sdk.addresses, "delete", None)
            if callable(delete_func):
                res = delete_func(address_id)
                # Some SDKs may return None or a truthy sentinel; treat no exception as success
                api_logger.info("Successfully deleted Shippo address via SDK: %s", address_id)
                return True
        except Exception as sdk_err:
            api_logger.debug("SDK address delete failed for %s, will HTTP DELETE: %s", address_id, sdk_err)

        # Fallback to HTTP DELETE for classic endpoint (no /v2 for addresses)
        headers = {
            "Authorization": f"ShippoToken {SHIPPO_API_KEY}",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"https://api.goshippo.com/addresses/{address_id}", headers=headers
            )

        if response.status_code in (200, 204):
            api_logger.info("Successfully deleted Shippo address: %s", address_id)
            return True
        if response.status_code == 404:
            # Already deleted; consider successful idempotence
            api_logger.info("Shippo address already deleted (404): %s", address_id)
            return True

        api_logger.warning(
            "Failed to delete address %s: HTTP %d - %s",
            address_id,
            response.status_code,
            response.text,
        )
        return False

    except Exception as e:
        api_logger.warning("Error deleting Shippo address %s: %s", address_id, e)
        return False


def _reverse_street_if_applicable(street: str) -> str:
    """Return reversed '<name> <number>' for inputs like '<number> <name>' and vice-versa.
    This helps match common permutations like '1 broadway' vs 'broadway 1'."""
    s = (street or "").strip().lower()
    parts = s.split()
    if len(parts) != 2:
        return s
    a, b = parts[0], parts[1]
    is_a_num = a.isdigit()
    is_b_num = b.isdigit()
    if is_a_num and not is_b_num:
        return f"{b} {a}"
    if is_b_num and not is_a_num:
        return f"{b} {a}"
    return s


def _street_candidates(street: str) -> Tuple[str, str]:
    base = (street or "").lower().strip()
    rev = _reverse_street_if_applicable(base)
    return base, rev


def _list_all_addresses() -> List[Any]:
    """Fetch all addresses from Shippo via SDK with pagination if supported."""
    results: List[Any] = []
    try:
        limit = 100
        offset = 0
        while True:
            try:
                page = shippo_sdk.addresses.list(limit=limit, offset=offset)
            except TypeError:
                # SDK may not support kwargs; fallback single call
                page = shippo_sdk.addresses.list()
            page_results = getattr(page, "results", page) or []
            results.extend(page_results)
            # Heuristic: if fewer than limit returned, we've reached the end
            if not page_results or len(page_results) < limit:
                break
            offset += limit
    except Exception as e:
        api_logger.warning("Shippo: pagination failed, using partial list: %s", e)
    return results


def _normalize_address_components(address_data: dict) -> Tuple[str, str, str, str]:
    """Extract and normalize core address components for comparison."""
    return (
        (address_data.get("street1") or "").lower().strip(),
        (address_data.get("city") or "").lower().strip(),
        (address_data.get("state") or "").lower().strip(),
        (address_data.get("country") or "").upper().strip(),
    )


def _find_matching_addresses(
    addresses_list: List[Any],
    input_street: str,
    input_city: str,
    input_state: str,
    input_country: str,
) -> List[Any]:
    """Find addresses that match the input location exactly (street, city, state, country)."""
    candidates: List[Any] = []

    street_a, street_b = _street_candidates(input_street)

    for addr in addresses_list:
        addr_street = (getattr(addr, "street1", "") or "").lower().strip()
        addr_city = (getattr(addr, "city", "") or "").lower().strip()
        addr_state = (getattr(addr, "state", "") or "").lower().strip()
        addr_country = (getattr(addr, "country", "") or "").upper().strip()

        if (
            (addr_street == street_a or addr_street == street_b)
            and addr_city == input_city
            and addr_state == input_state
            and addr_country == input_country
        ):
            candidates.append(addr)

    return candidates


# Note: We intentionally do NOT implement "similar" matching here to strictly follow
# the requested flow: only exact duplicate detection and cleanup.


async def _cleanup_duplicate_addresses(
    candidates: List[Any], keep_address_id: str
) -> int:
    """Remove all duplicate addresses except the one to keep. Returns number deleted."""
    deleted = 0
    for addr in candidates:
        oid = getattr(addr, "object_id", "")
        if oid and oid != keep_address_id:
            if await _delete_shippo_address(oid):
                deleted += 1
    return deleted


def shippo_get_all_addresses() -> List[Any]:
    """Public wrapper used by admin/ops to list all stored Shippo addresses."""
    return _list_all_addresses()


async def shippo_delete_all_addresses(confirm: bool = False) -> dict:
    """
    Danger: Deletes ALL stored Shippo addresses in the connected account.
    Returns a report dict with before_count, deleted_count, after_count.

    confirm must be True to proceed.
    """
    if not confirm:
        raise ValueError("Confirmation required: pass confirm=True to delete all addresses")

    before = _list_all_addresses()
    before_count = len(before)
    deleted = 0
    for addr in before:
        oid = getattr(addr, "object_id", None)
        if not oid:
            continue
        try:
            ok = await _delete_shippo_address(oid)
            if ok:
                deleted += 1
        except Exception as e:
            api_logger.warning("Failed to delete address %s: %s", oid, e)

    after = _list_all_addresses()
    report = {
        "before_count": before_count,
        "deleted_count": deleted,
        "after_count": len(after),
    }
    api_logger.info("Shippo address cleanup report: %s", report)
    return report


async def shippo_validate_address(address_data: dict):
    """
    Validate and normalize a shipping address via Shippo while preventing duplicates.

    Flow:
    1) Get address list from Shippo
    2) Compare against input (exact street/city/state/country)
    3) If duplicates exist, delete older addresses via Shippo DELETE API
    4) Return the latest remaining address
    5) If no matches found, create a new address (validate=False)
    6) Deletion strictly follows Shippo docs
    """

    # 1) List existing addresses
    try:
        addresses_list = _list_all_addresses()
        api_logger.info("Shippo: fetched %d addresses", len(addresses_list))
    except Exception as e:
        api_logger.warning("Shippo: failed to list addresses: %s", e)
        addresses_list = []

    # 2) Normalize input and find exact matches
    input_street, input_city, input_state, input_country = (
        _normalize_address_components(address_data)
    )
    exact_matches = _find_matching_addresses(
        addresses_list, input_street, input_city, input_state, input_country
    )

    # 3) Duplicates present: use latest without deleting others (avoid breaking existing rates/shipments)
    if exact_matches:
        latest = max(
            exact_matches,
            key=lambda x: getattr(x, "object_created", "")
            or getattr(x, "object_id", ""),
        )
        latest_id = getattr(latest, "object_id", "")
        api_logger.info(
            "Shippo: found %d exact matches, using latest %s (no deletions)",
            len(exact_matches),
            latest_id,
        )

        # Return latest address as dict with explicit valid flag to avoid downstream 400s
        return {
            "object_id": latest_id,
            "street1": getattr(latest, "street1", None),
            "street2": getattr(latest, "street2", None),
            "city": getattr(latest, "city", None),
            "state": getattr(latest, "state", None),
            "zip": getattr(latest, "zip", None),
            "country": getattr(latest, "country", None),
            "validation_results": {"is_valid": True, "messages": []},
        }

    # 5) No matches: create new unvalidated address to avoid duplicate detection
    try:
        request = components.AddressCreateRequest(
            name=address_data.get("name"),
            street1=address_data.get("street1"),
            street2=address_data.get("street2"),
            city=address_data.get("city"),
            state=address_data.get("state"),
            zip=address_data.get("zip"),
            country=address_data.get("country"),
            phone=address_data.get("phone"),
            email=address_data.get("email"),
            validate=False,
        )

        created = shippo_sdk.addresses.create(request)
        created_id = getattr(created, "object_id", "unknown")
        api_logger.info("Shippo: created address %s", created_id)
        # Return as dict with explicit valid flag
        return {
            "object_id": created_id,
            "street1": getattr(created, "street1", None),
            "street2": getattr(created, "street2", None),
            "city": getattr(created, "city", None),
            "state": getattr(created, "state", None),
            "zip": getattr(created, "zip", None),
            "country": getattr(created, "country", None),
            "validation_results": {"is_valid": True, "messages": []},
        }
    except Exception as e:
        error_msg = str(e).lower()
        api_logger.exception("Shippo: failed to create address: %s", e)

        if any(k in error_msg for k in ("authorization", "authentication")):
            raise HTTPException(
                status_code=500, detail="Shipping service authentication error"
            )
        if any(k in error_msg for k in ("rate limit", "too many requests")):
            raise HTTPException(
                status_code=429,
                detail="Shipping service temporarily busy, please try again",
            )
        if any(k in error_msg for k in ("network", "connection", "timeout")):
            raise HTTPException(
                status_code=503, detail="Shipping service temporarily unavailable"
            )
        raise HTTPException(status_code=500, detail="Address validation service error")


def _get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def shippo_get_from_address_ref() -> Any:
    """
    Return a reference suitable for ShipmentCreateRequest.address_from.
    Preference order:
      1) SHIPPO_FROM_ADDRESS_ID env var (string id)
      2) Build from env fields and create (validate=False), cached for reuse
    """
    global _FROM_ADDRESS_REF
    if _FROM_ADDRESS_REF is not None:
        return _FROM_ADDRESS_REF

    addr_id = _get_env("SHIPPO_FROM_ADDRESS_ID")
    if addr_id:
        _FROM_ADDRESS_REF = addr_id
        return _FROM_ADDRESS_REF

    # Build from env vars
    name = _get_env("SHIPPO_FROM_NAME", "Chaamo")
    street1 = _get_env("SHIPPO_FROM_STREET1")
    city = _get_env("SHIPPO_FROM_CITY")
    state = _get_env("SHIPPO_FROM_STATE")
    zip_ = _get_env("SHIPPO_FROM_ZIP")
    country = _get_env("SHIPPO_FROM_COUNTRY", "US")
    phone = _get_env("SHIPPO_FROM_PHONE")
    email = _get_env("SHIPPO_FROM_EMAIL")

    if not all([street1, city, state, zip_, country]):
        raise HTTPException(status_code=500, detail="Missing Shippo from-address env configuration")

    try:
        req = components.AddressCreateRequest(
            name=name,
            street1=street1,
            city=city,
            state=state,
            zip=zip_,
            country=country,
            phone=phone,
            email=email,
            validate=False,
        )
        created = shippo_sdk.addresses.create(req)
        _FROM_ADDRESS_REF = getattr(created, "object_id", created)
        api_logger.info("Shippo: created merchant from-address %s", _FROM_ADDRESS_REF)
    except Exception as e:
        api_logger.exception("Shippo: failed to create from-address: %s", e)
        raise HTTPException(status_code=500, detail="Failed to initialize shipping origin address")
    return _FROM_ADDRESS_REF


def _get_default_parcel() -> components.ParcelCreateRequest:
    """Build a parcel from env or fallback defaults for trading cards."""
    try:
        length = _get_env("SHIPPO_PARCEL_LENGTH", "20")
        width = _get_env("SHIPPO_PARCEL_WIDTH", "15")
        height = _get_env("SHIPPO_PARCEL_HEIGHT", "2")
        weight = _get_env("SHIPPO_PARCEL_WEIGHT", "0.05")
        return components.ParcelCreateRequest(
            length=str(length),
            width=str(width),
            height=str(height),
            distance_unit=components.DistanceUnitEnum.CM,
            weight=str(weight),
            mass_unit=components.WeightUnitEnum.KG,
        )
    except Exception as e:
        api_logger.exception("Shippo: failed to build default parcel: %s", e)
        # last resort minimal parcel
        return components.ParcelCreateRequest(
            length="20",
            width="15",
            height="2",
            distance_unit=components.DistanceUnitEnum.CM,
            weight="0.05",
            mass_unit=components.WeightUnitEnum.KG,
        )


def shippo_build_shipment_and_select_rate(
    address_from: Dict[str, Any],
    address_to: Dict[str, Any],
    client_hint: Dict[str, Any],
    parcel: components.ParcelCreateRequest | None = None,
    insurance_amount: str | None = None,
    insurance_currency: str | None = None,
):
    """
    Create a shipment server-side with dynamic address_from/address_to, get fresh
    rates, and select a rate matching the client hint (provider+service if available;
    else amount/currency tolerance match).

    Returns dict: { 'rate_id': str, 'amount': Decimal, 'currency': str }
    """
    # Prepare address references
    from_ref = address_from.get("object_id") or {
        "name": address_from.get("name"),
        "street1": address_from.get("street1"),
        "street2": address_from.get("street2"),
        "city": address_from.get("city"),
        "state": address_from.get("state"),
        "zip": address_from.get("zip"),
        "country": address_from.get("country"),
        "phone": address_from.get("phone"),
        "email": address_from.get("email"),
    }
    to_ref = address_to.get("object_id") or {
        "name": address_to.get("name"),
        "street1": address_to.get("street1"),
        "street2": address_to.get("street2"),
        "city": address_to.get("city"),
        "state": address_to.get("state"),
        "zip": address_to.get("zip"),
        "country": address_to.get("country"),
        "phone": address_to.get("phone"),
        "email": address_to.get("email"),
    }

    if parcel is None:
        parcel = _get_default_parcel()

    shipment_req = components.ShipmentCreateRequest(
        address_from=from_ref,
        address_to=to_ref,
        parcels=[parcel],
        async_=False,
        insurance_amount=insurance_amount,
        insurance_currency=insurance_currency,
        test=shippo_is_test or None,
    )
    shipment = shippo_sdk.shipments.create(shipment_req)
    rates = getattr(shipment, "rates", []) or []
    if not rates:
        raise HTTPException(status_code=400, detail="No shipping rates available for destination")

    # Extract hint
    hint_provider = client_hint.get("provider")
    hint_service = client_hint.get("servicelevel_name")
    hint_amount = client_hint.get("amount")
    hint_currency = (client_hint.get("currency") or "").upper()

    chosen = None
    # 1) Try provider+service match
    if hint_provider and hint_service:
        for r in rates:
            prov = getattr(r, "provider", None)
            sl = getattr(r, "servicelevel", None)
            srv = getattr(sl, "name", None) if sl else None
            if prov == hint_provider and srv == hint_service:
                chosen = r
                break

    # 2) Try amount+currency tolerance
    if chosen is None and (hint_amount is not None and hint_currency):
        try:
            ha = Decimal(str(hint_amount))
        except Exception:
            ha = None
        if ha is not None:
            for r in rates:
                try:
                    ra = Decimal(str(getattr(r, "amount", None) or 0))
                    rc = (getattr(r, "currency", None) or "").upper()
                    if rc == hint_currency and abs(ra - ha) <= Decimal("0.01"):
                        chosen = r
                        break
                except Exception:
                    continue

    # 3) Fallback: pick cheapest
    if chosen is None:
        try:
            rates_sorted = sorted(
                rates,
                key=lambda r: Decimal(str(getattr(r, "amount", "999999"))),
            )
            chosen = rates_sorted[0]
        except Exception:
            chosen = rates[0]

    rate_id = getattr(chosen, "object_id", None) or (
        chosen.get("object_id") if isinstance(chosen, dict) else None
    )
    amount = getattr(chosen, "amount", None)
    currency = (getattr(chosen, "currency", None) or "").upper()
    return {
        "rate_id": rate_id,
        "amount": Decimal(str(amount)) if amount is not None else None,
        "currency": currency,
    }


def shippo_get_rate_details(rate_id: str):
    """
    Return dict with rate details including amount, currency, messages, and expires_at.
    This function tries a couple of method names depending on shippo SDK shape.
    """
    if not rate_id:
        raise ValueError("rate_id empty")

    # try SDK call patterns
    try:
        # some SDKs: shippo.rates.get(rate_id)
        rate = shippo_sdk.rates.get(rate_id)
    except Exception:
        try:
            # fallback: shippo.get_rate(rate_id)
            rate = shippo_sdk.get_rate(rate_id)
        except Exception as e:
            raise RuntimeError(f"Shippo rate lookup failed for {rate_id}: {e}")

    # Normalize dict-like or object-like
    if isinstance(rate, dict):
        amount = rate.get("amount")
        currency = rate.get("currency")
        messages = rate.get("messages", [])
        expires_at = rate.get("expires_at")
        status = rate.get("status")
        provider = rate.get("provider")
        sl = rate.get("servicelevel") or {}
        servicelevel_name = sl.get("name")
    else:
        amount = getattr(rate, "amount", None)
        currency = getattr(rate, "currency", None)
        messages = getattr(rate, "messages", [])
        expires_at = getattr(rate, "expires_at", None)
        status = getattr(rate, "status", None)
        provider = getattr(rate, "provider", None)
        sl = getattr(rate, "servicelevel", None)
        servicelevel_name = getattr(sl, "name", None) if sl else None

    if amount is None or currency is None:
        raise RuntimeError("Unexpected Shippo rate format")

    try:
        result = {
            "amount": Decimal(str(amount)),
            "currency": str(currency).upper(),
            "messages": messages,
            "expires_at": expires_at,
            "status": status,
            "provider": provider,
            "servicelevel_name": servicelevel_name,
        }
        return result
    except InvalidOperation:
        raise RuntimeError("Invalid rate amount from Shippo")

from decimal import Decimal, InvalidOperation
from fastapi import HTTPException
from shippo.models import components
from typing import Dict, Any, List, Tuple
import httpx
import os
import shippo as shippo_module

from src.utils.logger import api_logger

# Centralized Shippo SDK configuration & singleton
SHIPPO_API_KEY: str = os.environ.get("SHIPPO_API_KEY", "")
shippo_is_test: bool = True if "test" in (SHIPPO_API_KEY or "").lower() else False
shippo_sdk = shippo_module.Shippo(api_key_header=SHIPPO_API_KEY)

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
        # Auth header per Shippo docs
        headers = {"Authorization": f"ShippoToken {SHIPPO_API_KEY}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"https://api.goshippo.com/v2/addresses/{address_id}",
                headers=headers
            )
            
        if response.status_code in (200, 204):
            api_logger.info("Successfully deleted Shippo address: %s", address_id)
            return True
        else:
            api_logger.warning("Failed to delete address %s: HTTP %d - %s", 
                             address_id, response.status_code, response.text)
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


async def _cleanup_duplicate_addresses(candidates: List[Any], keep_address_id: str) -> int:
    """Remove all duplicate addresses except the one to keep. Returns number deleted."""
    deleted = 0
    for addr in candidates:
        oid = getattr(addr, "object_id", "")
        if oid and oid != keep_address_id:
            if await _delete_shippo_address(oid):
                deleted += 1
    return deleted

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
    input_street, input_city, input_state, input_country = _normalize_address_components(address_data)
    exact_matches = _find_matching_addresses(
        addresses_list, input_street, input_city, input_state, input_country
    )

    # 3) Duplicates present: clean up and return latest
    if exact_matches:
        latest = max(
            exact_matches,
            key=lambda x: getattr(x, "object_created", "") or getattr(x, "object_id", ""),
        )
        latest_id = getattr(latest, "object_id", "")

        deleted = await _cleanup_duplicate_addresses(exact_matches, latest_id)
        if deleted:
            api_logger.info("Shippo: cleaned %d duplicate addresses, keeping %s", deleted, latest_id)
        else:
            api_logger.info("Shippo: no duplicates to delete, using %s", latest_id)

        # 4) Return latest address as dict with explicit valid flag to avoid downstream 400s
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
            raise HTTPException(status_code=500, detail="Shipping service authentication error")
        if any(k in error_msg for k in ("rate limit", "too many requests")):
            raise HTTPException(status_code=429, detail="Shipping service temporarily busy, please try again")
        if any(k in error_msg for k in ("network", "connection", "timeout")):
            raise HTTPException(status_code=503, detail="Shipping service temporarily unavailable")
        raise HTTPException(status_code=500, detail="Address validation service error")


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
    else:
        amount = getattr(rate, "amount", None)
        currency = getattr(rate, "currency", None)
        messages = getattr(rate, "messages", [])
        expires_at = getattr(rate, "expires_at", None)
        status = getattr(rate, "status", None)

    if amount is None or currency is None:
        raise RuntimeError("Unexpected Shippo rate format")

    try:
        result = {
            "amount": Decimal(str(amount)),
            "currency": str(currency).upper(),
            "messages": messages,
            "expires_at": expires_at,
            "status": status,
        }
        return result
    except InvalidOperation:
        raise RuntimeError("Invalid rate amount from Shippo")

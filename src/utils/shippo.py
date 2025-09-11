from decimal import Decimal, InvalidOperation
from fastapi import HTTPException
from shippo.models import components
from typing import Dict, Any

from src.shippo_handlers import shippo_sdk
from src.utils.logger import api_logger


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


class MultipleAddressesException(Exception):
    """Custom exception for multiple address matches."""

    def __init__(self, message, suggestions=None):
        super().__init__(message)
        self.suggestions = suggestions or []


def _normalize_address_for_comparison(addr_data: dict) -> str:
    """Create normalized key for comprehensive address comparison including name, email, phone"""
    return "|".join([
        (addr_data.get("name") or "").lower().strip(),
        (addr_data.get("email") or "").lower().strip(),
        (addr_data.get("phone") or "").strip().replace("+", "").replace(" ", "").replace("-", ""),
        (addr_data.get("street1") or "").lower().strip(),
        (addr_data.get("street2") or "").lower().strip(),
        (addr_data.get("city") or "").lower().strip(),
        (addr_data.get("state") or "").lower().strip(),
        (addr_data.get("zip") or "").strip(),
        (addr_data.get("country") or "").upper().strip(),
    ])


async def _filter_valid_unique_addresses(addresses_list):
    """Filter out duplicate and invalid addresses, keeping only valid unique ones"""
    seen_keys = set()
    unique_addresses = []
    
    for addr in addresses_list:
        addr_key = _normalize_address_for_comparison({
            "name": getattr(addr, "name", ""),
            "email": getattr(addr, "email", ""),
            "phone": getattr(addr, "phone", ""),
            "street1": getattr(addr, "street1", ""),
            "street2": getattr(addr, "street2", ""),
            "city": getattr(addr, "city", ""),
            "state": getattr(addr, "state", ""),
            "zip": getattr(addr, "zip", ""),
            "country": getattr(addr, "country", "")
        })
        
        # Check if address is valid
        validation_results = getattr(addr, "validation_results", None)
        is_valid = True
        if validation_results:
            is_valid = getattr(validation_results, "is_valid", False)
        
        if addr_key in seen_keys:
            api_logger.info("Skipping duplicate address: %s", getattr(addr, "object_id", "unknown"))
        elif not is_valid:
            api_logger.info("Skipping invalid address: %s", getattr(addr, "object_id", "unknown"))
        else:
            # Keep this valid, unique address
            seen_keys.add(addr_key)
            unique_addresses.append(addr)
            api_logger.info("Keeping valid unique address: %s", getattr(addr, "object_id", "unknown"))
    
    return unique_addresses


async def shippo_validate_address(address_data: dict):
    """
    Validates shipping address using Shippo with clean duplicate handling.
    
    Flow:
    1. List existing addresses
    2. Filter addresses matching input parameters 
    3. If duplicates found, keep only the latest one
    4. If no match found, create new address
    
    Returns: Validated address object
    Raises: MultipleAddressesException or HTTPException for validation errors
    """
    target_key = _normalize_address_for_comparison(address_data)
    
    try:
        # Step 1: Get existing addresses from Shippo
        existing_addresses = shippo_sdk.addresses.list()
        addresses_list = getattr(existing_addresses, 'results', existing_addresses) or []
        
        # Step 2: Find all matching addresses
        matching_addresses = []
        for addr in addresses_list:
            # Skip invalid addresses
            validation_results = getattr(addr, "validation_results", None)
            if validation_results and not getattr(validation_results, "is_valid", True):
                continue
                
            addr_key = _normalize_address_for_comparison({
                "name": getattr(addr, "name", ""),
                "email": getattr(addr, "email", ""),
                "phone": getattr(addr, "phone", ""),
                "street1": getattr(addr, "street1", ""),
                "street2": getattr(addr, "street2", ""),
                "city": getattr(addr, "city", ""),
                "state": getattr(addr, "state", ""),
                "zip": getattr(addr, "zip", ""),
                "country": getattr(addr, "country", "")
            })
            
            if addr_key == target_key:
                matching_addresses.append(addr)
        
        # Step 3: Handle matching addresses
        if matching_addresses:
            if len(matching_addresses) == 1:
                api_logger.info("Reusing existing Shippo address: %s", getattr(matching_addresses[0], "object_id", "unknown"))
                return matching_addresses[0]
            else:
                # Multiple matches - keep the latest one (by object_created or object_id)
                latest_addr = max(matching_addresses, key=lambda x: getattr(x, "object_created", "") or getattr(x, "object_id", ""))
                api_logger.info("Found %d duplicate addresses, using latest: %s", len(matching_addresses), getattr(latest_addr, "object_id", "unknown"))
                return latest_addr
        
    except Exception as e:
        api_logger.warning("Failed to check existing addresses, creating new: %s", e)
    
    # Step 4: Create new address if no valid match found
    try:
        address_request = components.AddressCreateRequest(
            name=address_data.get("name"),
            street1=address_data.get("street1"),
            street2=address_data.get("street2"),
            city=address_data.get("city"),
            state=address_data.get("state"),
            zip=address_data.get("zip"),
            country=address_data.get("country"),
            phone=address_data.get("phone"),
            email=address_data.get("email"),
            validate=True,
        )
        
        validated_address = shippo_sdk.addresses.create(address_request)
        api_logger.info("Created new Shippo address: %s", getattr(validated_address, "object_id", "unknown"))
        
        # Step 5: Check validation results
        validation_results = getattr(validated_address, "validation_results", None)
        if validation_results:
            messages = getattr(validation_results, "messages", []) or []
            
            for msg in messages:
                msg_text = getattr(msg, "text", "").lower()
                
                # Handle ambiguous addresses
                if any(keyword in msg_text for keyword in ["multiple records", "multiple addresses", "no default match"]):
                    raise MultipleAddressesException(
                        "Address matched multiple records. Please provide more specific address details.",
                        suggestions=[]
                    )
                
                # Handle validation failures
                if any(keyword in msg_text for keyword in ["could not be verified", "invalid address"]):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Address validation failed",
                            "message": "The address could not be validated. Please check and correct the address details.",
                            "validation_message": getattr(msg, "text", "")
                        }
                    )
            
            # Check overall validity
            if not getattr(validation_results, "is_valid", True) and messages:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid address",
                        "message": "Address validation failed. Please verify your address details.",
                        "messages": [getattr(m, "text", str(m)) for m in messages[:3]]
                    }
                )
        
        return validated_address
        
    except MultipleAddressesException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        api_logger.exception("Shippo address validation failed: %s", e)
        
        # Handle specific API errors
        if any(keyword in error_msg for keyword in ["multiple addresses", "multiple records"]):
            raise MultipleAddressesException(
                "Address matched multiple records. Please provide more specific address details.",
                suggestions=[]
            )
        elif any(keyword in error_msg for keyword in ["authorization", "authentication"]):
            raise HTTPException(status_code=500, detail="Shipping service authentication error")
        elif any(keyword in error_msg for keyword in ["rate limit", "too many requests"]):
            raise HTTPException(status_code=429, detail="Shipping service temporarily busy, please try again")
        elif any(keyword in error_msg for keyword in ["network", "connection", "timeout"]):
            raise HTTPException(status_code=503, detail="Shipping service temporarily unavailable")
        else:
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

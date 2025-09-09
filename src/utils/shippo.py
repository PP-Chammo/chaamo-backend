import shippo
import asyncio
from fastapi import HTTPException, status

CURRENCY_SYMBOL = {
    "GBP": "£",
    "USD": "$",
    "EUR": "€",
    "AUD": "A$",
    "CAD": "CA$"
}

def fmt_amount(amount: float, currency: str) -> str:
    """Format amount with currency symbol if known, else 'amount CURRENCY'."""
    cur = (currency or "GBP").upper()
    sym = CURRENCY_SYMBOL.get(cur)
    # Format with 2 decimals, use rounding
    amt = f"{amount:,.2f}"
    if sym:
        # Put symbol before amount for common currencies
        return f"{sym}{amt}"
    return f"{amt} {cur}"


# Blocking Shippo SDK call wrapped for to_thread
def _create_shippo_shipment_sync(ship_payload: dict):
    shippo.config.api_key = SHIPPO_API_KEY
    # Shippo SDK expects named params; passing dict with **
    return shippo.Shipment.create(**ship_payload)


async def _get_rates_with_retry(shipment_payload: dict, attempts: int = 3, base_delay: float = 0.4):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            resp = await asyncio.to_thread(_create_shippo_shipment_sync, shipment_payload)
            # Shippo SDK returns dict-like object; 'rates' key contains list
            return resp.get("rates", [])
        except Exception as e:
            last_exc = e
            logger.warning("Shippo attempt %d failed: %s", attempt, e)
            if attempt < attempts:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
    logger.error("Shippo all attempts failed: %s", last_exc)
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to fetch shipping rates from Shippo")

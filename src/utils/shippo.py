from decimal import Decimal, InvalidOperation

from src.shippo_handlers import shippo_sdk

CURRENCY_SYMBOL = {
    "GBP": "£",
    "USD": "$",
    "EUR": "€",
    "AUD": "A$",
    "CAD": "CA$"
}

def shippo_get_rate_details(rate_id: str):
    """
    Return dict with at least {'amount': Decimal, 'currency': 'USD'} or raise.
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
    else:
        amount = getattr(rate, "amount", None)
        currency = getattr(rate, "currency", None)

    if amount is None or currency is None:
        raise RuntimeError("Unexpected Shippo rate format")

    try:
        return {"amount": Decimal(str(amount)), "currency": str(currency).upper()}
    except InvalidOperation:
        raise RuntimeError("Invalid rate amount from Shippo")
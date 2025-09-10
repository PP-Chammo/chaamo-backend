from decimal import Decimal, getcontext, ROUND_HALF_EVEN
import httpx
import logging
import time
from typing import Optional, Dict

# set a high-enough precision for intermediate calcs (default 28 is fine)
getcontext().prec = 28

# simple logger
logger = logging.getLogger("currency")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# rates cache & settings
_RATES_URL = "https://open.er-api.com/v6/latest/USD"
_RATES_CACHE: Dict[str, Dict] = {"rates": {}, "fetched_at": 0.0}
_RATES_TTL_SECONDS = 60 * 5  # cache for 5 minutes
_HTTPX_TIMEOUT = 10.0


def _fetch_rates() -> Optional[Dict[str, Decimal]]:
    """
    Fetch latest rates from public API and return a mapping currency_code -> Decimal(rate).
    Uses an in-memory cache to minimize requests.
    """
    now = time.time()
    if _RATES_CACHE["rates"] and (now - _RATES_CACHE["fetched_at"] < _RATES_TTL_SECONDS):
        return _RATES_CACHE["rates"]

    try:
        resp = httpx.get(_RATES_URL, timeout=_HTTPX_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch exchange rates: %s", exc)
        return None

    if not data or data.get("result") != "success" or "rates" not in data:
        logger.error("Exchange rates API returned non-success: %s", data)
        return None

    raw_rates = data["rates"]
    rates: Dict[str, Decimal] = {}
    try:
        for k, v in raw_rates.items():
            rates[k.upper()] = Decimal(str(v))
    except Exception as exc:
        logger.exception("Failed to parse rates into Decimal: %s", exc)
        return None

    _RATES_CACHE["rates"] = rates
    _RATES_CACHE["fetched_at"] = now
    return rates


def format_price(
    from_currency: str, amount: float, to_currency: str
) -> Optional[float]:
    """
    Convert `amount` from `from_currency` to `to_currency` using USD as base.
    Algorithm:
        amount_in_target = amount * (rate_to / rate_from)
    where rates are taken from endpoint: https://open.er-api.com/v6/latest/USD
    (rates[c] means: 1 USD = rates[c] units of currency c)

    Returns:
        float rounded to 2 decimal places (bankers rounding / ROUND_HALF_EVEN),
        or None on error.
    """
    if from_currency is None or to_currency is None:
        logger.error("from_currency and to_currency must be provided")
        return None

    # normalize currency codes
    src = from_currency.upper()
    dst = to_currency.upper()

    # trivial case
    if src == dst:
        try:
            rounded = Decimal(str(amount)).quantize(
                Decimal("0.00"), rounding=ROUND_HALF_EVEN
            )
            return float(rounded)
        except Exception:
            logger.exception("Failed to quantize same-currency amount")
            return None

    rates = _fetch_rates()
    if not rates:
        logger.error("No exchange rates available")
        return None

    # ensure the currencies exist in the rates map
    if src not in rates:
        logger.error("Source currency not available in rates: %s", src)
        return None
    if dst not in rates:
        logger.error("Target currency not available in rates: %s", dst)
        return None

    try:
        decimal_amount = Decimal(str(amount))
        rate_from = rates[src]  # 1 USD = rate_from units of src currency
        rate_to = rates[dst]  # 1 USD = rate_to units of dst currency

        # compute factor = rate_to / rate_from
        factor = rate_to / rate_from

        # amount_in_dst = decimal_amount * factor
        result = (decimal_amount * factor).quantize(
            Decimal("0.00"), rounding=ROUND_HALF_EVEN
        )

        return float(result)
    except Exception as exc:
        logger.exception("Conversion failed: %s", exc)
        return None

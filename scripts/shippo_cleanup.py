#!/usr/bin/env python3
import asyncio
import os
from typing import Any

# Load .env before importing Shippo utils so SHIPPO_API_KEY is set
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv()  # also load from current working directory as fallback
except Exception:
    pass

from src.utils.logger import api_logger
from src.utils.shippo import (
    shippo_get_all_addresses,
    shippo_delete_all_addresses,
    SHIPPO_API_KEY,
)


def require_env() -> None:
    if not SHIPPO_API_KEY:
        raise SystemExit("SHIPPO_API_KEY is not set in environment")


def print_addresses(title: str) -> int:
    addrs = shippo_get_all_addresses()
    count = len(addrs)
    print(f"{title}: {count} address(es)")
    # Show first few IDs for visibility
    show = min(5, count)
    for i in range(show):
        a: Any = addrs[i]
        oid = getattr(a, "object_id", None) or (a.get("object_id") if isinstance(a, dict) else None)
        street1 = getattr(a, "street1", None) or (a.get("street1") if isinstance(a, dict) else None)
        city = getattr(a, "city", None) or (a.get("city") if isinstance(a, dict) else None)
        state = getattr(a, "state", None) or (a.get("state") if isinstance(a, dict) else None)
        country = getattr(a, "country", None) or (a.get("country") if isinstance(a, dict) else None)
        print(f"  - {i+1}. id={oid} street1={street1} city={city} state={state} country={country}")
    return count


async def main() -> None:
    require_env()
    before = print_addresses("Before cleanup")
    if before == 0:
        print("Nothing to delete.")
        return
    print("Deleting all Shippo addresses...")
    report = await shippo_delete_all_addresses(confirm=True)
    print(f"Cleanup report: {report}")
    print_addresses("After cleanup")


if __name__ == "__main__":
    asyncio.run(main())

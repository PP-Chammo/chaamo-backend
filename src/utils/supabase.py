"""Clean Supabase client configuration and connection management."""

import os
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi import HTTPException
from src.utils.logger import setup_logger

# Load environment variables from .env file
load_dotenv()


def _get_supabase_credentials() -> tuple[str, str]:
    """Get and validate Supabase credentials from environment."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required"
        )

    return url, key


def _create_supabase_client() -> Client:
    """Create and test Supabase client connection."""
    url, key = _get_supabase_credentials()

    from src.utils.logger import setup_logger

    logger = setup_logger("chaamo.supabase")

    logger.info("🔧 Initializing Supabase connection...")
    logger.info(f"   🌐 URL: {url}")
    logger.info(f"   🔑 Key: {key[:20]}...")

    try:
        client = create_client(url, key)

        # Test connection
        test_query = client.table("categories").select("id").limit(1).execute()
        if len(test_query.data) > 0:
            logger.info("✅ Supabase connection test successful")
        else:
            logger.info("❌ Supabase connection test failed")

        return client

    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        raise


# Global Supabase client instance
supabase: Client = _create_supabase_client()
sb_logger = setup_logger("chaamo.supabase")


# ===============================================================
# query helpers
# ===============================================================
def supabase_apply_filter(query, filters: dict | None):
    if not filters:
        return query
    for k, v in filters.items():
        # Skip None values and log for debugging
        if v is None:
            sb_logger.debug(f"supabase_apply_filter: skipping filter {k}=None")
            continue

        if isinstance(v, dict):
            # IN operator handling
            if "in" in v:
                in_val = v.get("in")
                if in_val is None or not isinstance(in_val, (list, tuple)) or len(in_val) == 0:
                    sb_logger.debug(
                        f"supabase_apply_filter: skipping filter {k} IN {in_val!r} (None/invalid/empty)"
                    )
                else:
                    query = query.in_(k, in_val)  # basic IN support
                continue

            # NEQ operator handling
            if "neq" in v:
                neq_val = v.get("neq")
                if neq_val is None:
                    sb_logger.debug(
                        f"supabase_apply_filter: skipping filter {k} NEQ None"
                    )
                else:
                    query = query.neq(k, neq_val)  # not equal support
                continue

            # Unknown structured filter object
            sb_logger.debug(
                f"supabase_apply_filter: unrecognized filter object for key={k}: {v!r} (skipping)"
            )
            continue

        # Default equality filter
        query = query.eq(k, v)
    return query


# ===============================================================
# getters (unified)
# ===============================================================
def supabase_get_subscription(filters: dict, columns: str = "*") -> dict | None:
    """Get a single subscription matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("subscriptions").select(columns), filters)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_subscription failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Database fetch error (subscriptions)"
        )


def supabase_get_subscriptions(filters: dict, columns: str = "*") -> list[dict]:
    """Get list of subscriptions matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("subscriptions").select(columns), filters)
        res = q.execute()
        return list(getattr(res, "data", []) or [])
    except Exception as e:
        sb_logger.exception("supabase_get_subscriptions failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Database fetch error (subscriptions)"
        )


def supabase_get_order(filters: dict, columns: str = "*") -> dict | None:
    """Get a single order matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("orders").select(columns), filters)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_order failed: %s", e)
        raise HTTPException(status_code=500, detail="Database fetch error (orders)")


def supabase_get_orders(filters: dict, columns: str = "*") -> list[dict]:
    """Get list of orders matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("orders").select(columns), filters)
        res = q.execute()
        return list(getattr(res, "data", []) or [])
    except Exception as e:
        sb_logger.exception("supabase_get_orders failed: %s", e)
        raise HTTPException(status_code=500, detail="Database fetch error (orders)")


def supabase_get_payment(filters: dict, columns: str = "*") -> dict | None:
    """Get a single payment matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("payments").select(columns), filters)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_payment failed: %s", e)
        raise HTTPException(status_code=500, detail="Database fetch error (payments)")


def supabase_get_payments(filters: dict, columns: str = "*") -> list[dict]:
    """Get list of payments matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("payments").select(columns), filters)
        res = q.execute()
        return list(getattr(res, "data", []) or [])
    except Exception as e:
        sb_logger.exception("supabase_get_payments failed: %s", e)
        raise HTTPException(status_code=500, detail="Database fetch error (payments)")


def supabase_get_membership_plan(filters: dict, columns: str = "*") -> dict | None:
    """Get a single membership plan matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("membership_plans").select(columns), filters)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_membership_plan failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Database fetch error (membership_plans)"
        )


def supabase_get_boost_plan(filters: dict, columns: str = "*") -> dict | None:
    """Get a single boost plan matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("boost_plans").select(columns), filters)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_boost_plan failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Database fetch error (boost_plans)"
        )


def supabase_get_profile(filters: dict, columns: str = "*") -> dict | None:
    """Get a single profile matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("profiles").select(columns), filters)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_profile failed: %s", e)
        raise HTTPException(status_code=500, detail="Database fetch error (profiles)")


def supabase_get_user_address(filters: dict, columns: str = "*") -> dict | None:
    """Get a single user address matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("user_addresses").select(columns), filters)
        # Prefer default if present
        q = q.order("is_default", desc=True)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_user_address failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Database fetch error (user_addresses)"
        )


def supabase_get_card(filters: dict, columns: str = "*") -> dict | None:
    """Get a single card matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("cards").select(columns), filters)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_card failed: %s", e)
        raise HTTPException(status_code=500, detail="Database fetch error (cards)")


def supabase_get_listing_card(filters: dict, columns: str = "*") -> dict | None:
    """Get a single listing card matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("vw_listing_cards").select(columns), filters)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_listing_card failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Database fetch error (vw_listing_cards)"
        )


def supabase_get_listing(filters: dict, columns: str = "*") -> dict | None:
    """Get a single listing matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("listings").select(columns), filters)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_listing failed: %s", e)
        raise HTTPException(status_code=500, detail="Database fetch error (listings)")


def supabase_get_boost_listing(filters: dict, columns: str = "*") -> dict | None:
    """Get a single boost listing matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("boost_listings").select(columns), filters)
        res = q.limit(1).execute()
        if getattr(res, "data", None):
            return res.data[0]
        return None
    except Exception as e:
        sb_logger.exception("supabase_get_boost_listing failed: %s", e)
        raise HTTPException(status_code=500, detail="Database fetch error (boost_listings)")


def supabase_get_boost_listings(filters: dict, columns: str = "*") -> list[dict]:
    """Get list of boost listings matching filters."""
    try:
        q = supabase_apply_filter(supabase.table("boost_listings").select(columns), filters)
        res = q.execute()
        return list(getattr(res, "data", []) or [])
    except Exception as e:
        sb_logger.exception("supabase_get_boost_listings failed: %s", e)
        raise HTTPException(status_code=500, detail="Database fetch error (boost_listings)")


# ===============================================================
# insert / update mutate
# ===============================================================
def supabase_mutate_subscription(
    mutate_type: str, payload: dict, filters: dict | None = None
):
    """Insert or update subscriptions. Returns Supabase response."""
    try:
        table = supabase.table("subscriptions")
        if mutate_type == "insert":
            return table.insert(payload).execute()
        if mutate_type == "update":
            q = supabase_apply_filter(table.update(payload), filters)
            return q.execute()
        raise ValueError("mutate_type must be 'insert' or 'update'")
    except Exception as e:
        sb_logger.exception("supabase_mutate_subscription failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Database mutate error (subscriptions)"
        )


def supabase_mutate_order(mutate_type: str, payload: dict, filters: dict | None = None):
    """Insert or update orders. Returns Supabase response."""
    try:
        table = supabase.table("orders")
        if mutate_type == "insert":
            return table.insert(payload).execute()
        if mutate_type == "update":
            q = supabase_apply_filter(table.update(payload), filters)
            return q.execute()
        raise ValueError("mutate_type must be 'insert' or 'update'")
    except Exception as e:
        sb_logger.exception("supabase_mutate_order failed: %s", e)
        raise HTTPException(status_code=500, detail="Database mutate error (orders)")


def supabase_mutate_payment(mutate_type: str, payload: dict, filters: dict | None = None):
    """Insert or update payments. Returns Supabase response."""
    try:
        table = supabase.table("payments")
        if mutate_type == "insert":
            return table.insert(payload).execute()
        if mutate_type == "update":
            q = supabase_apply_filter(table.update(payload), filters)
            return q.execute()
        raise ValueError("mutate_type must be 'insert' or 'update'")
    except Exception as e:
        sb_logger.exception("supabase_mutate_payment failed: %s", e)
        raise HTTPException(status_code=500, detail="Database mutate error (payments)")


def supabase_mutate_boost_listing(
    mutate_type: str, payload: dict, filters: dict | None = None
):
    """Insert or update boost_listings. Returns Supabase response."""
    try:
        table = supabase.table("boost_listings")
        if mutate_type == "insert":
            return table.insert(payload).execute()
        if mutate_type == "update":
            q = supabase_apply_filter(table.update(payload), filters)
            return q.execute()
        raise ValueError("mutate_type must be 'insert' or 'update'")
    except Exception as e:
        sb_logger.exception("supabase_mutate_boost_listing failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Database mutate error (boost_listings)"
        )


# ===============================================================
# delete
# ===============================================================
def supabase_delete_subscription(filters: dict):
    try:
        q = supabase_apply_filter(supabase.table("subscriptions").delete(), filters)
        return q.execute()
    except Exception as e:
        sb_logger.exception("supabase_delete_subscription failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Database delete error (subscriptions)"
        )


def supabase_delete_order(filters: dict):
    try:
        q = supabase_apply_filter(supabase.table("orders").delete(), filters)
        return q.execute()
    except Exception as e:
        sb_logger.exception("supabase_delete_order failed: %s", e)
        raise HTTPException(status_code=500, detail="Database delete error (orders)")


def supabase_delete_payment(filters: dict):
    try:
        q = supabase_apply_filter(supabase.table("payments").delete(), filters)
        return q.execute()
    except Exception as e:
        sb_logger.exception("supabase_delete_payment failed: %s", e)
        raise HTTPException(status_code=500, detail="Database delete error (payments)")

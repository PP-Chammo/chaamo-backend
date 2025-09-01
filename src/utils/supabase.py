"""Clean Supabase client configuration and connection management."""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

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

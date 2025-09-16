from datetime import datetime

from src.utils.logger import api_logger
from src.utils.supabase import supabase


async def create_subscription_record(
    user_id: str,
    paypal_subscription_id: str,
    membership_plan_id: str,
    status: str = "pending",
) -> dict:
    """
    Create a new subscription record in the database.

    Args:
        user_id: The user ID (UUID) who owns this subscription
        paypal_subscription_id: The PayPal subscription ID
        plan_id: The membership plan ID (UUID)
        status: Initial status (pending, active, canceled, expired)

    Returns:
        dict: The created subscription record

    Raises:
        Exception: If database insertion fails
    """
    try:
        subscription_data = {
            "user_id": user_id,
            "paypal_subscription_id": paypal_subscription_id,
            "plan_id": membership_plan_id,
            "status": status,
        }

        response = supabase.table("subscriptions").insert(subscription_data).execute()
        if response.data:
            api_logger.info(
                f"✅ Created subscription record: {paypal_subscription_id} for user: {user_id}"
            )
            return response.data[0]
        else:
            raise Exception("No data returned from subscription creation")

    except Exception as e:
        api_logger.error(f"❌ Failed to create subscription record: {str(e)}")
        raise


def update_subscription_status(
    paypal_subscription_id: str, status: str, start_date: datetime, end_date: datetime
) -> dict:
    """
    Update subscription status and end date.

    Args:
        paypal_subscription_id: The PayPal subscription ID
        status: New status (active, pending, canceled, expired)
        start_date: Subscription start date
        end_date: Subscription end date

    Returns:
        dict: The updated subscription record
    """
    try:
        update_data = {
            "status": status,
            "start_date": start_date,
            "end_date": end_date,
        }

        response = (
            supabase.table("subscriptions")
            .update(update_data)
            .eq("paypal_subscription_id", paypal_subscription_id)
            .execute()
        )

        if response.data:
            api_logger.info(
                f"✅ Updated subscription {paypal_subscription_id} status to: {status}"
            )
            return response.data[0]
        else:
            raise Exception("No data returned from subscription update")

    except Exception as e:
        api_logger.error(f"❌ Failed to update subscription status: {str(e)}")
        raise


def get_subscription_record(user_id: str, plan_id: str) -> dict | None:
    try:
        response = (
            supabase.table("subscriptions")
            .select("*")
            .eq("user_id", user_id)
            .eq("plan_id", plan_id)
            .execute()
        )
        if response.data:
            return response.data[0]
        return None

    except Exception as e:
        api_logger.error(f"❌ Failed to get subscription record: {str(e)}")
        return None

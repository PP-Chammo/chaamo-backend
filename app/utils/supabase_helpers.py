from app.services.supabase import supabase
from app.models.card import FinalGroupedCard


async def save_to_supabase(grouped_cards: list[FinalGroupedCard]):
    """
    Iterates through the list of grouped cards and calls the Supabase RPC
    function 'upsert_cards' for each one.
    """
    if not supabase:
        print("Supabase client not available. Skipping save to database.")
        return

    print("Saving results to Supabase...")
    for card_group in grouped_cards:
        try:
            # Convert the list of Pydantic models to a list of dicts for the JSONB column
            sold_cards_json = [sc.model_dump() for sc in card_group.sold_cards]

            params_to_send = {
                'card_title': card_group.title,
                'card_image_url': card_group.image_url,
                'card_sales_data': sold_cards_json
            }

            # Call the RPC function with the new, unambiguous parameters
            response = supabase.rpc('upsert_cards', params_to_send).execute()

            # Optional: check for errors in the Supabase response
            if hasattr(response, 'error') and response.error is not None:
                print(f"Error calling Supabase RPC for '{card_group.title}': {response.error.message}")

        except Exception as e:
            print(f"An unexpected error occurred while saving to Supabase: {e}")
    print("Finished saving results to Supabase.")

from app.services.supabase import supabase
from app.models.card import FinalGroupedCard


async def save_to_supabase(grouped_cards: list[FinalGroupedCard]):
    """
    Iterates through the list of grouped cards and calls the Supabase RPC
    function 'upsert_cards' for each one.
    """
    if not supabase:
        print("❌ ERROR: Supabase client not available. Skipping save to database.")
        return

    print(f"🚀 Starting Supabase save operation for {len(grouped_cards)} card groups...")
    
    success_count = 0
    error_count = 0
    
    for i, card_group in enumerate(grouped_cards, 1):
        print(f"\n📦 Processing card group {i}/{len(grouped_cards)}: '{card_group.title}'")
        
        try:
            # Validate the card group data
            if not card_group.title or not card_group.title.strip():
                print(f"❌ ERROR: Card group {i} has empty or invalid title")
                error_count += 1
                continue
                
            if not card_group.sold_cards:
                print(f"⚠️  WARNING: Card group '{card_group.title}' has no sold cards data")
            
            # Convert the list of Pydantic models to a list of dicts for the JSONB column
            sold_cards_json = [sc.model_dump() for sc in card_group.sold_cards]
            
            print(f"   📊 Sold cards count: {len(sold_cards_json)}")
            print(f"   🖼️  Image URL: {card_group.image_url[:50]}..." if len(card_group.image_url) > 50 else f"   🖼️  Image URL: {card_group.image_url}")

            params_to_send = {
                'card_title': card_group.title,
                'card_image_url': card_group.image_url,
                'card_sales_data': sold_cards_json
            }

            print(f"   🔄 Calling Supabase RPC 'upsert_cards' with parameters:")
            print(f"      - card_title: '{card_group.title}'")
            print(f"      - card_image_url: {card_group.image_url[:50]}..." if len(card_group.image_url) > 50 else f"      - card_image_url: {card_group.image_url}")
            print(f"      - card_sales_data: {len(sold_cards_json)} items")

            # Call the RPC function with the new, unambiguous parameters
            response = supabase.rpc('upsert_cards', params_to_send).execute()

            print(f"   ✅ RPC call completed for '{card_group.title}'")
            
            # Detailed response analysis
            if hasattr(response, 'data'):
                print(f"   📋 Response data: {response.data}")
            else:
                print(f"   📋 Response data: None")
                
            if hasattr(response, 'error') and response.error is not None:
                print(f"   ❌ ERROR in Supabase response for '{card_group.title}': {response.error}")
                if hasattr(response.error, 'message'):
                    print(f"      Error message: {response.error.message}")
                if hasattr(response.error, 'details'):
                    print(f"      Error details: {response.error.details}")
                error_count += 1
            else:
                print(f"   ✅ SUCCESS: Card group '{card_group.title}' saved to Supabase")
                success_count += 1

        except Exception as e:
            print(f"   ❌ EXCEPTION occurred while saving '{card_group.title}' to Supabase:")
            print(f"      Exception type: {type(e).__name__}")
            print(f"      Exception message: {str(e)}")
            error_count += 1
            
            # Print more details for debugging
            import traceback
            print(f"      Full traceback:")
            traceback.print_exc()
    
    print(f"\n🎯 Supabase save operation completed:")
    print(f"   ✅ Successful saves: {success_count}")
    print(f"   ❌ Failed saves: {error_count}")
    print(f"   📊 Total processed: {len(grouped_cards)}")
    
    if error_count > 0:
        print(f"⚠️  WARNING: {error_count} card groups failed to save. Check the logs above for details.")
    else:
        print(f"🎉 All card groups saved successfully!")

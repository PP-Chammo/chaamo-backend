import re
from typing import Any, Dict, List
from app.services.supabase import supabase

from app.services.supabase import supabase
from app.models.tcdb import ScrapeTarget
from app.handlers.tcdb_scrape_handler import discover_cards


async def discover_bulk(category: ScrapeTarget) -> List[Dict[str, Any]]:
    selected_category = supabase.table("categories").select("id, name").eq("name", category).execute()
    result = []
    if selected_category.data and len(selected_category.data) > 0:
        category_id = selected_category.data[0]["id"]
        card_sets = supabase.table("card_sets").select("platform_set_id, category_id, name").eq("category_id", category_id).execute()
        print(len(card_sets.data))
        if card_sets.data and len(card_sets.data) > 0:
            for card_set in card_sets.data:
                print(f"scraping sets {card_set}")
                cards = await discover_cards(card_set["platform_set_id"])
                result.append({
                    "set_id": card_set["platform_set_id"],
                    "set_name": card_set["name"],
                    "cards_scraped": len(cards)
                })
    return result

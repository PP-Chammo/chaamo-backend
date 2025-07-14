import asyncio
from typing import List

from app.models.card import FinalGroupedCard
from app.utils.embedding_extractor import get_image_embedding
from app.utils.scrape_helpers import parse_raw_cards_data, group_cards


async def process_scrape_cards(html_text: str) -> List[FinalGroupedCard]:
    """
    Processes HTML and groups cards with detailed output per card.
    """
    print("🔍 Starting HTML processing and card grouping...")
    
    # Parse raw cards data
    print("📄 Parsing raw cards data from HTML...")
    raw_cards_data = parse_raw_cards_data(html_text)
    print(f"   📊 Found {len(raw_cards_data)} raw cards in HTML")
    
    if not raw_cards_data:
        print("❌ ERROR: No raw cards data found in HTML")
        return []
    
    # Get embeddings for each card
    print("🖼️  Extracting image embeddings...")
    tasks = [get_image_embedding(card['image_url']) for card in raw_cards_data]
    embeddings = await asyncio.gather(*tasks)
    
    # Count successful embeddings
    successful_embeddings = sum(1 for emb in embeddings if emb is not None)
    print(f"   📊 Successful embeddings: {successful_embeddings}/{len(raw_cards_data)}")

    # Combine cards with features
    cards_with_features = [
        {**card, 'embedding': emb} for card, emb in zip(raw_cards_data, embeddings) if emb is not None
    ]
    print(f"   📊 Cards with valid features: {len(cards_with_features)}")

    if not cards_with_features:
        print("❌ ERROR: No cards with valid features after embedding extraction")
        return []

    # Group cards
    print("🔗 Grouping cards by similarity...")
    groups = group_cards(cards_with_features)
    print(f"   📊 Created {len(groups)} card groups")

    if not groups:
        print("❌ ERROR: No card groups created")
        return []

    # Convert to FinalGroupedCard models
    print("🏗️  Converting to FinalGroupedCard models...")
    final_result = []
    for i, g in enumerate(groups, 1):
        try:
            card_group = FinalGroupedCard(
                title=g['title'], 
                image_url=g['image_url'], 
                sold_cards=g['sold_cards']
            )
            final_result.append(card_group)
            print(f"   ✅ Group {i}: '{card_group.title}' with {len(card_group.sold_cards)} sold cards")
        except Exception as e:
            print(f"   ❌ ERROR creating FinalGroupedCard for group {i}: {e}")
            print(f"      Group data: {g}")

    print(f"\n🎯 Processing completed:")
    print(f"   📊 Raw cards found: {len(raw_cards_data)}")
    print(f"   📊 Cards with features: {len(cards_with_features)}")
    print(f"   📊 Groups created: {len(groups)}")
    print(f"   📊 Final result: {len(final_result)} card groups")
    
    if final_result:
        print("✅ Successfully processed cards, ready for Supabase save")
    else:
        print("❌ ERROR: No final results to save to Supabase")

    return final_result


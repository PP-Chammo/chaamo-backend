import asyncio
from typing import List

from app.models.card import FinalGroupedCard
from app.utils.embedding_extractor import get_image_embedding
from app.utils.scrape_helpers import parse_raw_cards_data, group_cards


async def process_scrape_cards(html_text: str) -> List[FinalGroupedCard]:
    """
    Processes HTML and groups cards with detailed output per card.
    """
    raw_cards_data = parse_raw_cards_data(html_text)
    tasks = [get_image_embedding(card['image_url']) for card in raw_cards_data]
    embeddings = await asyncio.gather(*tasks)

    cards_with_features = [
        {**card, 'embedding': emb} for card, emb in zip(raw_cards_data, embeddings) if emb is not None
    ]

    groups = group_cards(cards_with_features)

    print(f"result: {len(groups)} cards")

    final_result = [
        FinalGroupedCard(title=g['title'], image_url=g['image_url'], sold_cards=g['sold_cards'])
        for g in groups
    ]

    return final_result


import re
from thefuzz import fuzz
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from sentence_transformers import util

from app.models.card import SoldCard


def save_debug_html(html_text: str) -> str:
    """Save raw HTML to a file for debugging purposes."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"error_dump_{timestamp}.html"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_text)
        print(f"Raw HTML for debugging has been saved to file: {filename}")
        return filename
    except Exception as e:
        print(f"Failed to save debug HTML file: {e}")
        return ""


def normalize_title(title: str) -> str:
    """Clean and standardize the card title using regex."""
    if not title:
        return ""
    title = title.lower()
    title = re.sub(r'\b(psa|bgs|sgc|cgc|beckett|topps|panini)\s*\d*(\.\d+)?\b', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\b(20\d{2}(-\d{2})?)\b', '', title)
    title = re.sub(r'\b(mint|gem|near|nm|auto|patch|rookie|rc|refractor|prizm|holo|graded|rare|sp|ssp|🔥)\b', '', title, flags=re.IGNORECASE)
    title = re.sub(r'#\w+', '', title)
    title = re.sub(r'/\d+', '', title)
    title = re.sub(r'[^a-z0-9 ]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def parse_price(price_str: str) -> tuple[float, str]:
    """Parse the price string to get the float value and currency."""
    price_str = price_str.replace(',', '')
    match = re.search(r'([£$€])?(\d+\.\d{2})', price_str)

    if match:
        currency = match.group(1) or '$'
        price = float(match.group(2))
        return price, currency
    return 0.0, 'N/A'


def parse_raw_cards_data(html_text: str) -> List[Dict[str, Any]]:
    """Parse the HTML and extract the raw card data."""
    soup = BeautifulSoup(html_text, 'lxml')
    items = soup.select('li.s-item, li.s-card')
    raw_cards_data = []

    for item in items:
        try:
            title_element = item.select_one('.s-item__title span, .s-card__title span')
            if not title_element or title_element.get_text() == "Shop on eBay":
                continue

            link_element = item.select_one('.s-item__link, .su-link')
            img_element = item.select_one('.s-item__image-wrapper img, .s-card__image-wrapper img, .image-treatment img')
            price_element = item.select_one('.s-item__price, .s-card__price')
            sold_date_element = item.select_one('.s-item__caption .POSITIVE, .s-card__caption .POSITIVE, .s-item__caption .positive, .s-card__caption .positive')

            if not all([title_element, img_element, link_element, price_element, sold_date_element]):
                continue

            raw_cards_data.append({
                "title": title_element.get_text(),
                "normalized_title": normalize_title(title_element.get_text()),
                "image_url": img_element.get('src') if img_element else '',
                "item_id": link_element.get('href').split('?')[0].split('/')[-1],
                "price_str": price_element.get_text().replace(',', ''),
                "sold_date_str": sold_date_element.get_text().replace('Sold', '').strip(),
                "link_url": link_element.get('href'),
            })
        except Exception as e:
            print(f"Skipping item due to parsing error: {e}")
            continue
    return raw_cards_data


def group_cards(cards_with_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group the cards based on image and text similarity."""
    IMAGE_WEIGHT = 0.70
    TEXT_WEIGHT = 0.30
    HYBRID_THRESHOLD = 0.90
    groups: List[Dict[str, Any]] = []

    for card in cards_with_features:
        price, currency = parse_price(card['price_str'])
        if price <= 0:
            continue

        sold_card_info = SoldCard(
            item_id=card['item_id'],
            sold_date=card['sold_date_str'],
            price=price,
            currency=currency,
            link_url=card['link_url'],
        )

        found_a_group = False
        for group in groups:
            image_sim = util.cos_sim(card['embedding'], group['representative_embedding']).item()
            text_sim = fuzz.ratio(card['normalized_title'], group['representative_title']) / 100.0
            hybrid_score = (IMAGE_WEIGHT * image_sim) + (TEXT_WEIGHT * text_sim)

            if hybrid_score >= HYBRID_THRESHOLD:
                group['sold_cards'].append(sold_card_info)
                found_a_group = True
                break

        if not found_a_group:
            groups.append({
                'representative_embedding': card['embedding'],
                'representative_title': card['normalized_title'],
                'title': card['title'],
                'image_url': card['image_url'],
                'sold_cards': [sold_card_info]
            })
    return groups


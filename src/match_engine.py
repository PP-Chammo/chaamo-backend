"""
Deterministic matching engine for eBay scraped items with smart query tokenization,
phrase detection, match scoring, and bucket classification.
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import dateutil.parser as date_parser
from src.utils.logger import get_logger
from src.utils.supabase import supabase

match_logger = get_logger("match_engine", "INFO")


def _normalize_text(text: str) -> str:
    """Normalize text: trim, collapse whitespace, lowercase, keep alphanumerics + dashes + plus."""
    if not text:
        return ""

    # Convert to lowercase and collapse whitespace
    normalized = re.sub(r"\s+", " ", text.lower().strip())

    # Remove punctuation except dashes and plus (keep alphanumerics, dashes, plus, spaces)
    normalized = re.sub(r"[^\w\s\-\+]", " ", normalized)

    # Collapse whitespace again after punctuation removal
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def _tokenize_query(query: str) -> List[str]:
    """Split normalized query into tokens by whitespace."""
    normalized = _normalize_text(query)
    if not normalized:
        return []

    return normalized.split()


def _detect_phrases_in_title(
    query_tokens: List[str], normalized_title: str
) -> Dict[str, int]:
    """
    Detect contiguous multi-word phrases from query that appear in title.
    Returns dict mapping detected phrases to their word count.
    Prioritizes longer phrases to avoid overlapping matches.
    """
    phrases = {}
    covered_positions = set()

    # Check all possible contiguous substrings of 2+ words, starting with longest
    for length in range(
        len(query_tokens), 1, -1
    ):  # From longest to shortest (min 2 words)
        for start_idx in range(len(query_tokens) - length + 1):
            # Skip if any position is already covered by a longer phrase
            positions = set(range(start_idx, start_idx + length))
            if positions.intersection(covered_positions):
                continue

            phrase_tokens = query_tokens[start_idx : start_idx + length]
            phrase = " ".join(phrase_tokens)

            # Check if this exact phrase appears in the normalized title
            if phrase in normalized_title:
                phrases[phrase] = length
                covered_positions.update(positions)

    return phrases


def _count_matched_words(query_tokens: List[str], normalized_title: str) -> int:
    """
    Count matched words with phrase detection.
    First detect phrases, then count individual words not covered by phrases.
    """
    if not query_tokens or not normalized_title:
        return 0

    # Detect phrases first
    detected_phrases = _detect_phrases_in_title(query_tokens, normalized_title)

    # Track which query token positions are covered by phrases
    covered_positions = set()
    total_phrase_words = 0

    # Process detected phrases (prioritize longer phrases)
    for phrase, word_count in sorted(
        detected_phrases.items(), key=lambda x: x[1], reverse=True
    ):
        phrase_tokens = phrase.split()

        # Find where this phrase starts in the query
        for start_idx in range(len(query_tokens) - len(phrase_tokens) + 1):
            if (
                query_tokens[start_idx : start_idx + len(phrase_tokens)]
                == phrase_tokens
            ):
                # Check if any positions are already covered
                positions = set(range(start_idx, start_idx + len(phrase_tokens)))
                if not positions.intersection(covered_positions):
                    covered_positions.update(positions)
                    total_phrase_words += word_count
                    break

    # Count individual words not covered by phrases
    individual_matches = 0
    for i, token in enumerate(query_tokens):
        if i not in covered_positions:
            # Check if word appears in title (as whole word or part of hyphenated token)
            if (
                re.search(rf"\b{re.escape(token)}\b", normalized_title)
                or re.search(rf"{re.escape(token)}(?=\-)", normalized_title)
                or re.search(rf"(?<=\-){re.escape(token)}", normalized_title)
            ):
                individual_matches += 1

    return total_phrase_words + individual_matches


def _compute_match_score(
    query_tokens: List[str], normalized_title: str
) -> Tuple[int, int]:
    """
    Compute match score for a scraped item.
    Returns: (matched_word_count, match_percentage)
    """
    if not query_tokens:
        return 0, 0

    matched_words = _count_matched_words(query_tokens, normalized_title)
    match_ratio = matched_words / len(query_tokens)
    match_percentage = math.floor(match_ratio * 100)

    return matched_words, match_percentage


def _get_bucket_from_percentage(match_percentage: int) -> Optional[str]:
    """Convert match_percentage to bucket, clipping to allowed buckets."""
    allowed_buckets = [100, 90, 80, 70, 60, 50]

    if match_percentage < 50:
        return None

    # Round down to nearest 10, then find the allowed bucket
    rounded = (match_percentage // 10) * 10

    # Find the highest allowed bucket that is <= rounded
    for bucket in allowed_buckets:
        if bucket <= rounded:
            return str(bucket)

    return "50"  # Fallback


def _safe_parse_datetime(date_str: str) -> Optional[datetime]:
    """Safely parse ISO 8601 datetime string."""
    if not date_str:
        return None

    try:
        return date_parser.parse(date_str)
    except Exception:
        return None


def _safe_parse_float(value) -> float:
    """Safely convert value to float."""
    if value is None:
        return 0.0

    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _select_best_item_from_bucket(items: List[Dict]) -> Dict:
    """
    Select the best item from a bucket using tie-breaking rules:
    1. Most recent sold_at datetime
    2. Highest numeric price
    3. Lexicographically highest post_url
    """

    def sort_key(item):
        # Parse sold_at (missing/invalid sorts last)
        sold_at = _safe_parse_datetime(item.get("sold_at"))
        sold_at_timestamp = sold_at.timestamp() if sold_at else -1

        # Parse price (missing/invalid treated as 0)
        price = _safe_parse_float(item.get("price"))

        # Get post_url (missing treated as empty string)
        post_url = item.get("post_url") or ""

        # Sort by: sold_at DESC, price DESC, post_url DESC
        return (sold_at_timestamp, price, post_url)

    return max(items, key=sort_key)


def _update_user_cards_database(user_card_id: str, user_card_update: Dict):
    """Update user_cards table with new last_sold_* values."""
    try:
        supabase.table("user_cards").update(user_card_update).eq(
            "id", user_card_id
        ).execute()
        match_logger.info(f"✅ Updated user_cards for {user_card_id}")
    except Exception as e:
        match_logger.error(f"❌ Failed to update user_cards for {user_card_id}: {e}")
        raise


def deterministic_match_and_update(
    user_card_id: str, query: str, scraped_items: List[Dict]
) -> Dict:
    """
    Deterministic matching function for eBay scraped items.

    Args:
        user_card_id: User card ID to update
        query: Search query string
        scraped_items: List of scraped eBay items

    Returns:
        Dict containing match results and update information
    """
    # Initialize result structure
    matched = {"100": [], "90": [], "80": [], "70": [], "60": [], "50": []}

    # Tokenize query
    query_tokens = _tokenize_query(query)

    if not query_tokens:
        # Empty query - set fallback values
        user_card_update = {
            "last_sold_price": 0,
            "last_sold_post_url": None,
            "last_sold_currency": None,
        }

        _update_user_cards_database(user_card_id, user_card_update)

        return {
            "user_card_id": user_card_id,
            "query": query,
            "matched_counts": {k: 0 for k in matched.keys()},
            "matched": matched,
            "selected_for_update": None,
            "user_card_update": user_card_update,
        }

    # Process each scraped item
    for item in scraped_items:
        title = item.get("name") or item.get("title") or ""
        normalized_title = _normalize_text(title)

        if not normalized_title:
            continue

        # Compute match score
        matched_words, match_percentage = _compute_match_score(
            query_tokens, normalized_title
        )

        # Get bucket
        bucket = _get_bucket_from_percentage(match_percentage)

        if bucket:
            matched[bucket].append(item)

    # Count items in each bucket
    matched_counts = {bucket: len(items) for bucket, items in matched.items()}

    # Log individual bucket counts
    for bucket in ["100", "90", "80", "70", "60", "50"]:
        count = matched_counts[bucket]
        match_logger.info(f"📊 matched {bucket}% - count: {count}")

    # Select item for update
    selected_for_update = None
    user_card_update = {
        "last_sold_price": 0,
        "last_sold_post_url": None,
        "last_sold_currency": None,
    }

    # Examine buckets in descending order
    for bucket in ["100", "90", "80", "70", "60", "50"]:
        if matched_counts[bucket] > 0:
            best_item = _select_best_item_from_bucket(matched[bucket])

            selected_for_update = {"bucket": bucket, "item": best_item}

            user_card_update = {
                "last_sold_price": _safe_parse_float(best_item.get("price")),
                "last_sold_post_url": best_item.get("post_url"),
                "last_sold_currency": best_item.get("currency"),
            }
            
            # Log the selection
            ebay_post_id = best_item.get("id", "unknown")
            sold_at = best_item.get("sold_at", "unknown")
            match_logger.info(
                f"🎯 selected ebay_posts id {ebay_post_id} for update user_cards id \"{user_card_id}\" "
                f"with last_sold_price \"{user_card_update['last_sold_price']}\" sold at \"{sold_at}\""
            )
            break

    # Update database
    _update_user_cards_database(user_card_id, user_card_update)

    return {
        "user_card_id": user_card_id,
        "query": query,
        "matched_counts": matched_counts,
        "matched": matched,
        "selected_for_update": selected_for_update,
        "user_card_update": user_card_update,
    }

"""High-precision trading card matching pipeline for eBay listings.

This module implements a hybrid multi-stage filtering pipeline designed to achieve
97-100% accuracy in matching trading cards from unstructured eBay titles.
"""

import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from dateutil import parser as date_parser
import numpy as np

# Third-party imports
try:
    from rapidfuzz import fuzz
except ImportError:
    from fuzzywuzzy import fuzz

from src.utils.logger import get_logger

matching_logger = get_logger("matching")

# ============================================================================
# KNOWLEDGE BASE & CONFIGURATION
# ============================================================================


@dataclass
class CardFeatures:
    """Structured representation of extracted card features."""

    year: Optional[str] = None
    manufacturer: Optional[str] = None
    set_name: Optional[str] = None
    card_number: Optional[str] = None
    card_name: Optional[str] = None
    variants: List[str] = None
    rarity: Optional[str] = None
    edition: Optional[str] = None
    grade: Optional[str] = None

    def __post_init__(self):
        if self.variants is None:
            self.variants = []


# Comprehensive card manufacturer sets (sorted by length desc for matching)
CARD_DATA = {
    "topps": {
        "sets": [
            "topps chrome sapphire edition",
            "topps chrome black finite",
            "topps chrome platinum anniversary",
            "topps finest flashbacks",
            "topps gallery masterpiece",
            "topps stadium club chrome",
            "topps chrome update series",
            "topps sterling signatures",
            "topps luminaries masters",
            "topps definitive collection",
            "topps tier one limited",
            "topps inception signatures",
            "topps dynasty collection",
            "topps museum collection",
            "topps tribute edition",
            "topps gypsy queen chrome",
            "topps archives signature",
            "topps allen & ginter chrome",
            "topps fire platinum",
            "topps chrome rookie",
            "topps stadium club",
            "topps chrome refractor",
            "topps finest refractor",
            "topps heritage chrome",
            "topps bowman chrome",
            "topps opening day",
            "topps holiday mega",
            "topps series 1",
            "topps series 2",
            "topps update series",
            "topps chrome",
            "topps finest",
            "topps heritage",
            "topps merlin heritage 97",
            "topps archives",
            "topps fire",
            "topps bowman",
            "topps flagship",
            "topps base",
        ],
        "variants": [
            "superfractor",
            "refractor",
            "xfractor",
            "atomic refractor",
            "prism refractor",
            "gold refractor",
            "orange refractor",
            "red refractor",
            "blue refractor",
            "green refractor",
            "purple refractor",
            "black refractor",
            "sepia refractor",
            "negative refractor",
            "wave refractor",
            "shimmer",
        ],
    },
    "panini": {
        "sets": [
            "panini flawless collegiate",
            "panini national treasures collegiate",
            "panini immaculate collection collegiate",
            "panini black gold white box",
            "panini one and one timeless",
            "panini flawless finishes",
            "panini national treasures timeline",
            "panini immaculate collection vintage",
            "panini spectra aspiring signatures",
            "panini limited team trademarks",
            "panini contenders championship ticket",
            "panini obsidian volcanic material",
            "panini phoenix rising signatures",
            "panini select courtside chrome",
            "panini mosaic reactive prizm",
            "panini prizm draft picks chrome",
            "panini noir spotlight signatures",
            "panini one and one",
            "panini national treasures",
            "panini immaculate collection",
            "panini flawless gems",
            "panini black gold",
            "panini gold standard",
            "panini spectra neon",
            "panini limited patches",
            "panini obsidian electric",
            "panini phoenix fanatics",
            "panini contenders optic",
            "panini select premier",
            "panini mosaic genesis",
            "panini prizm emergent",
            "panini noir vintage",
            "panini certified cuts",
            "panini donruss optic",
            "panini chronicles draft",
            "panini flawless",
            "panini spectra",
            "panini limited",
            "panini obsidian",
            "panini phoenix",
            "panini contenders",
            "panini select",
            "panini mosaic",
            "panini prizm",
            "panini noir",
            "panini certified",
            "panini donruss",
            "panini chronicles",
        ],
        "variants": [
            "prizm",
            "holo prizm",
            "silver prizm",
            "gold prizm",
            "black prizm",
            "white sparkle",
            "reactive blue",
            "reactive orange",
            "genesis",
            "nebula",
            "cosmic",
            "astral",
            "fluorescent",
            "mojo prizm",
            "fast break prizm",
            "hyper prizm",
            "cracked ice",
            "shimmer prizm",
            "disco prizm",
        ],
    },
    "upper deck": {
        "sets": [
            "upper deck sp authentic signatures",
            "upper deck the cup championship",
            "upper deck ultimate collection signatures",
            "upper deck black diamond championship",
            "upper deck ice premieres rookies",
            "upper deck artifacts treasured",
            "upper deck trilogy signature pucks",
            "upper deck premier collection limited",
            "upper deck spx winning materials",
            "upper deck spa future watch",
            "upper deck mvp silver script",
            "upper deck series 1 young guns",
            "upper deck series 2 young guns",
            "upper deck sp authentic",
            "upper deck the cup",
            "upper deck ultimate",
            "upper deck black diamond",
            "upper deck ice premieres",
            "upper deck artifacts",
            "upper deck trilogy",
            "upper deck premier",
            "upper deck spx",
            "upper deck spa",
            "upper deck mvp",
            "upper deck series 1",
            "upper deck series 2",
            "upper deck ice",
            "upper deck",
        ],
        "variants": [
            "young guns",
            "future watch",
            "signature pucks",
            "high gloss",
            "exclusives",
            "spectrum",
            "clear cut",
            "acetate",
            "canvas",
            "retro",
            "silver script",
        ],
    },
    "bowman": {
        "sets": [
            "bowman chrome prospect autographs",
            "bowman sterling prospect autographs",
            "bowman draft chrome autographs",
            "bowman sapphire edition chrome",
            "bowman platinum prospect pipeline",
            "bowman inception prospect showcase",
            "bowman best of chrome",
            "bowman chrome mega box",
            "bowman chrome prospects",
            "bowman sterling continuity",
            "bowman draft picks chrome",
            "bowman sapphire chrome",
            "bowman platinum prospects",
            "bowman inception rookies",
            "bowman chrome refractor",
            "bowman chrome",
            "bowman sterling",
            "bowman draft",
            "bowman sapphire",
            "bowman platinum",
            "bowman inception",
            "bowman best",
            "bowman",
        ],
        "variants": [
            "refractor",
            "atomic refractor",
            "gold refractor",
            "orange refractor",
            "red refractor",
            "blue refractor",
            "green refractor",
            "purple refractor",
            "superfractor",
        ],
    },
    "leaf": {
        "sets": [
            "leaf metal draft prismatic",
            "leaf ultimate draft signatures",
            "leaf trinity patch autographs",
            "leaf valiant gridiron gems",
            "leaf metal draft",
            "leaf ultimate draft",
            "leaf trinity inscriptions",
            "leaf valiant recruits",
            "leaf certified materials",
            "leaf rookie retro",
            "leaf metal",
            "leaf ultimate",
            "leaf trinity",
            "leaf valiant",
            "leaf certified",
            "leaf",
        ],
        "variants": [
            "prismatic",
            "crystal",
            "gold prismatic",
            "red prismatic",
            "blue crystal",
            "wave",
            "clear",
        ],
    },
    "pokemon": {
        "sets": [
            "base set 1st edition shadowless",
            "neo genesis 1st edition",
            "skyridge crystal guardians",
            "hidden fates shiny vault",
            "champion's path rainbow rare",
            "vivid voltage amazing rare",
            "shining fates shiny vault",
            "evolving skies alternate art",
            "brilliant stars trainer gallery",
            "astral radiance trainer gallery",
            "lost origin trainer gallery",
            "silver tempest trainer gallery",
            "crown zenith galarian gallery",
            "scarlet violet 151 special",
            "paldea evolved illustration",
            "obsidian flames special",
            "paradox rift illustration",
            "base set unlimited",
            "base set shadowless",
            "base set 1st edition",
            "jungle 1st edition",
            "fossil 1st edition",
            "team rocket 1st edition",
            "gym heroes 1st edition",
            "gym challenge 1st edition",
            "neo genesis",
            "neo discovery",
            "neo revelation",
            "neo destiny",
            "expedition base",
            "aquapolis e-card",
            "skyridge e-card",
            "hidden fates",
            "champion's path",
            "vivid voltage",
            "shining fates",
            "evolving skies",
            "brilliant stars",
            "astral radiance",
            "lost origin",
            "silver tempest",
            "crown zenith",
            "scarlet violet",
            "paldea evolved",
            "obsidian flames",
            "paradox rift",
            "temporal forces",
            "twilight masquerade",
        ],
        "variants": [
            "holo",
            "reverse holo",
            "full art",
            "rainbow rare",
            "gold secret rare",
            "shiny vault",
            "amazing rare",
            "radiant",
            "vstar",
            "vmax",
            "v",
            "gx",
            "ex",
            "trainer gallery",
            "alternate art",
            "special illustration",
        ],
    },
}

# Lexicons for normalization and extraction
JUNK_WORDS = {
    "card",
    "cards",
    "trading",
    "tcg",
    "ccg",
    "collectible",
    "authentic",
    "genuine",
    "original",
    "official",
    "licensed",
    "mint",
    "near mint",
    "nm",
    "excellent",
    "ex",
    "good",
    "gd",
    "poor",
    "damaged",
    "played",
    "lp",
    "mp",
    "hp",
    "new",
    "sealed",
    "unopened",
    "pack fresh",
    "pulled",
}

ABBREVIATIONS = {
    "rc": "rookie card",
    "rpa": "rookie patch auto",
    "sp": "short print",
    "ssp": "super short print",
    "1/1": "one of one",
    "psa": "professional sports authenticator",
    "bgs": "beckett grading services",
    "sgc": "sports card guaranty",
    "cgc": "certified guaranty company",
}

RARITY_TERMS = {
    "common",
    "uncommon",
    "rare",
    "super rare",
    "ultra rare",
    "secret rare",
    "hyper rare",
    "rainbow rare",
    "gold rare",
    "short print",
    "super short print",
    "case hit",
    "ssp",
    "sp",
}

EDITION_TERMS = {
    "1st edition",
    "first edition",
    "unlimited edition",
    "limited edition",
    "special edition",
    "collector's edition",
}

# Regex patterns
YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
CARD_NUMBER_PATTERNS = [
    re.compile(r"#(\d+[a-zA-Z]?(?:/\d+)?)", re.IGNORECASE),
    re.compile(r"\b(\d{1,4}[a-zA-Z]?(?:/\d+)?)\s+(?:of|/)?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,4}[a-zA-Z]?)\b(?=\s+[A-Z][a-z]+)", re.IGNORECASE),
]
GRADE_PATTERN = re.compile(r"\b(?:psa|bgs|sgc|cgc)\s*(\d+(?:\.\d)?)\b", re.IGNORECASE)

# ============================================================================
# STAGE 1: PREPROCESSING & NORMALIZATION
# ============================================================================


def normalize_title(title: str) -> str:
    """Normalize and clean card title for processing."""
    if not title:
        return ""

    normalized = title.lower().strip()
    normalized = re.sub(r"(\d{4})[-/](\d{2,4})", r"\1", normalized)
    normalized = re.sub(r"[^a-z0-9 ]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    matching_logger.debug(f"Normalized: \"{title}\" -> \"{normalized}\"")
    return normalized


def remove_junk_words(text: str) -> str:
    """Remove common junk words while preserving important terms."""
    words = text.split()
    filtered = []

    for word in words:
        # Keep numbers, manufacturer names, set names
        if word.isdigit() or word in CARD_DATA or len(word) > 2:
            filtered.append(word)
        elif word not in JUNK_WORDS:
            filtered.append(word)

    return " ".join(filtered)


# ============================================================================
# STAGE 2: DETERMINISTIC FEATURE EXTRACTION
# ============================================================================


def extract_features(title: str) -> CardFeatures:
    """Extract structured features from normalized title."""
    features = CardFeatures()
    normalized = normalize_title(title)

    # Extract year
    year_match = YEAR_PATTERN.search(title)
    if year_match:
        features.year = year_match.group(1)

    # Extract manufacturer and set
    for manufacturer, data in CARD_DATA.items():
        if manufacturer in normalized:
            features.manufacturer = manufacturer

            # Find longest matching set name
            for set_name in data["sets"]:
                if set_name in normalized:
                    features.set_name = set_name
                    break

            # Extract variants
            for variant in data["variants"]:
                if variant in normalized:
                    features.variants.append(variant)
            break

    # Extract card number
    for pattern in CARD_NUMBER_PATTERNS:
        match = pattern.search(title)
        if match:
            features.card_number = match.group(1)
            break

    # Extract grade
    grade_match = GRADE_PATTERN.search(title)
    if grade_match:
        features.grade = grade_match.group(1)

    # Extract rarity
    for rarity in RARITY_TERMS:
        if rarity in normalized:
            features.rarity = rarity
            break

    # Extract edition
    for edition in EDITION_TERMS:
        if edition in normalized:
            features.edition = edition
            break

    # Extract player/card name (remaining significant words)
    cleaned = remove_junk_words(normalized)
    # Remove already extracted features
    if features.manufacturer:
        cleaned = cleaned.replace(features.manufacturer, "")
    if features.set_name:
        cleaned = cleaned.replace(features.set_name, "")
    if features.card_number:
        cleaned = re.sub(r"#?" + re.escape(features.card_number), "", cleaned)

    # Remaining words likely to be player/card name
    name_words = cleaned.split()
    name_candidates = [w for w in name_words if len(w) > 2 and not w.isdigit()]
    if name_candidates:
        features.card_name = " ".join(
            name_candidates[:3]
        )  # Take first 3 significant words

    matching_logger.debug(f"Extracted features: {features}")
    return features


# ============================================================================
# STAGE 3: WEIGHTED SCORING SYSTEM
# ============================================================================


class ScoringWeights:
    """Configurable weights for feature matching."""

    CARD_NUMBER = 0.30
    SET_NAME = 0.20
    CARD_NAME = 0.20
    VARIANTS = 0.10
    YEAR = 0.10
    EDITION = 0.05
    RARITY = 0.05


def calculate_feature_score(
    user_features: CardFeatures, post_features: CardFeatures
) -> float:
    """Calculate weighted similarity score between two sets of features."""
    score = 0.0

    # Card number match (highest priority)
    if user_features.card_number and post_features.card_number:
        if user_features.card_number == post_features.card_number:
            score += ScoringWeights.CARD_NUMBER

    # Set name match
    if user_features.set_name and post_features.set_name:
        if user_features.set_name == post_features.set_name:
            score += ScoringWeights.SET_NAME
        elif (
            post_features.set_name and user_features.set_name in post_features.set_name
        ):
            score += ScoringWeights.SET_NAME * 0.7

    # Card name similarity
    if user_features.card_name and post_features.card_name:
        name_similarity = (
            fuzz.token_sort_ratio(user_features.card_name, post_features.card_name)
            / 100.0
        )
        score += ScoringWeights.CARD_NAME * name_similarity

    # Variant matches
    if user_features.variants and post_features.variants:
        variant_overlap = len(set(user_features.variants) & set(post_features.variants))
        if variant_overlap > 0:
            score += ScoringWeights.VARIANTS * (
                variant_overlap / len(user_features.variants)
            )

    # Year match
    if user_features.year and post_features.year:
        if user_features.year == post_features.year:
            score += ScoringWeights.YEAR

    # Edition match
    if user_features.edition and post_features.edition:
        if user_features.edition == post_features.edition:
            score += ScoringWeights.EDITION

    # Rarity match
    if user_features.rarity and post_features.rarity:
        if user_features.rarity == post_features.rarity:
            score += ScoringWeights.RARITY

    return score


# ============================================================================
# STAGE 4: SEMANTIC TIE-BREAKING (OPTIONAL)
# ============================================================================

try:
    from sentence_transformers import SentenceTransformer

    SEMANTIC_MODEL_AVAILABLE = True
except ImportError:
    SEMANTIC_MODEL_AVAILABLE = False
    matching_logger.warning(
        "sentence-transformers not available, semantic tie-breaking disabled"
    )


class SemanticMatcher:
    """Handles semantic similarity for tie-breaking."""

    def __init__(self):
        self.model = None
        if SEMANTIC_MODEL_AVAILABLE:
            try:
                self.model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                matching_logger.info("Semantic model loaded successfully")
            except Exception as e:
                matching_logger.error(f"Failed to load semantic model: {e}")

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if not self.model:
            return 0.0

        try:
            embeddings = self.model.encode([text1, text2])
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            matching_logger.error(f"Semantic similarity computation failed: {e}")
            return 0.0


# ============================================================================
# MAIN MATCHING PIPELINE
# ============================================================================


class TradingCardMatcher:
    """High-precision trading card matching pipeline."""

    def __init__(self, enable_semantic: bool = False):
        """Initialize the matcher.

        Args:
            enable_semantic: Whether to enable semantic tie-breaking
        """
        self.enable_semantic = enable_semantic and SEMANTIC_MODEL_AVAILABLE
        self.semantic_matcher = SemanticMatcher() if self.enable_semantic else None
        matching_logger.info(
            f"TradingCardMatcher initialized (semantic={'enabled' if self.enable_semantic else 'disabled'})"
        )

    async def retrieve_candidates(
        self, user_query: str, posts: List[Dict]
    ) -> List[Dict]:
        """Stage 3: Retrieve and rank candidates.

        Args:
            user_query: User's card description
            posts: List of eBay post dictionaries

        Returns:
            Filtered and ranked list of candidate posts
        """
        if not posts:
            return []

        user_features = extract_features(user_query)
        user_normalized = normalize_title(user_query)
        user_words = set(user_normalized.split())

        candidates = []

        for post in posts:
            title = post.get("title", "")
            if not title:
                continue

            post_normalized = normalize_title(title)
            post_words = set(post_normalized.split())

            # Quick filter: require at least 40% word overlap OR key feature match
            word_overlap = len(user_words & post_words) / max(len(user_words), 1)

            # Check for key feature matches
            has_manufacturer = (
                user_features.manufacturer
                and user_features.manufacturer in post_normalized
            )
            has_card_number = (
                user_features.card_number and user_features.card_number in title
            )
            has_player_name = user_features.card_name and any(
                name in post_normalized for name in user_features.card_name.split()
            )

            if (
                word_overlap >= 0.4
                or has_manufacturer
                or has_card_number
                or has_player_name
            ):
                post_features = extract_features(title)
                score = calculate_feature_score(user_features, post_features)

                candidates.append({**post, "_score": score, "_features": post_features})

        # Sort by score
        candidates.sort(key=lambda x: x["_score"], reverse=True)

        matching_logger.info(
            f"Retrieved {len(candidates)} candidates from {len(posts)} posts"
        )
        return candidates

    async def select_best_match(
        self, user_query: str, posts: List[Dict], threshold: float = 0.4
    ) -> Optional[Dict]:
        """Select the best matching post for a user query.

        Args:
            user_query: User's card description
            posts: List of eBay post dictionaries
            threshold: Minimum score threshold for a match

        Returns:
            Best matching post or None if no good match found
        """
        candidates = await self.retrieve_candidates(user_query, posts)

        if not candidates:
            matching_logger.info("No candidates found")
            return None

        # Get top candidates
        top_candidates = candidates[:5]  # Consider top 5 for tie-breaking

        # Apply semantic tie-breaking if enabled and scores are close
        if self.enable_semantic and len(top_candidates) > 1:
            if top_candidates[0]["_score"] - top_candidates[1]["_score"] < 0.1:
                matching_logger.info("Applying semantic tie-breaking")

                for candidate in top_candidates:
                    semantic_score = self.semantic_matcher.compute_similarity(
                        user_query, candidate.get("title", "")
                    )
                    # Blend scores (70% feature, 30% semantic)
                    candidate["_final_score"] = (
                        0.7 * candidate["_score"] + 0.3 * semantic_score
                    )

                # Re-sort by final score
                top_candidates.sort(
                    key=lambda x: x.get("_final_score", x["_score"]), reverse=True
                )

        best_match = top_candidates[0]
        best_score = best_match.get("_final_score", best_match["_score"])

        if best_score < threshold:
            matching_logger.info(
                f"Best score {best_score:.3f} below threshold {threshold}"
            )
            return None

        # Clean up internal fields
        clean_match = {k: v for k, v in best_match.items() if not k.startswith("_")}

        matching_logger.info(
            f"✅ Best match found: score={best_score:.3f}, "
            f"title='{clean_match.get('title', '')[:50]}...'"
        )

        return clean_match


# ============================================================================
# ASYNC INTEGRATION WITH EBAY SCRAPER
# ============================================================================


async def match_ebay_posts(
    user_card_title: str, ebay_posts: List[Dict], enable_semantic: bool = False
) -> Optional[Dict]:
    """Main entry point for matching eBay posts with a user card.

    Args:
        user_card_title: User's card description
        ebay_posts: List of eBay posts from the scraper
        enable_semantic: Whether to enable semantic matching

    Returns:
        Best matching eBay post or None
    """
    matcher = TradingCardMatcher(enable_semantic=enable_semantic)
    stage1_result = normalize_title(user_card_title, ebay_posts)
    print(stage1_result)
    return await matcher.select_best_match(user_card_title, ebay_posts)


# ============================================================================
# COMPATIBILITY WRAPPER FOR EXISTING CODEBASE
# ============================================================================


def deterministic_match_and_update(
    user_card_title: str, ebay_posts: List[Dict]
) -> Optional[Dict]:
    """Synchronous wrapper for backwards compatibility.

    Args:
        user_card_title: User's card description
        ebay_posts: List of eBay posts

    Returns:
        Best matching eBay post or None
    """
    # Run async function in sync context
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # If we're already in an async context, create a task
        task = asyncio.create_task(
            match_ebay_posts(user_card_title, ebay_posts, enable_semantic=False)
        )
        return asyncio.run_coroutine_threadsafe(task, loop).result()
    else:
        # Run normally
        return loop.run_until_complete(
            match_ebay_posts(user_card_title, ebay_posts, enable_semantic=False)
        )


# ============================================================================
# GLOBAL INSTANCE FOR IMPORT COMPATIBILITY
# ============================================================================

# Create global matcher instance for backwards compatibility with existing imports
card_matcher = TradingCardMatcher(enable_semantic=False)

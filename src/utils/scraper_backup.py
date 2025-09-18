import os
import asyncio
import math
import random
import re
import unicodedata
import difflib
import json
from typing import Optional, Any, Dict, List

import dateutil.parser as date_parser
import pytz
from bs4 import BeautifulSoup
import httpx

from urllib.parse import urlencode
from src.models.category import CategoryId
from src.utils.logger import scraper_logger
from src.models.ebay import Region, base_target_url
from src.utils.httpx import httpx_get_content
from src.utils.supabase import supabase

from typing import Optional, Any, Dict, List

import json
import os
import re
import difflib
import httpx
from rapidfuzz import fuzz
import dateutil.parser as date_parser

from src.utils.supabase import supabase
from src.utils.logger import scraper_logger

# use global scraper_logger

try:
    from rapidfuzz import process, fuzz  # type: ignore

    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

DEFAULT_SET_NAMES = [
    "Panini Prizm",
    "Topps Chrome",
    "Evolving Skies",
    "Fusion Strike",
    "Base Set",
    "Jungle",
    "Fossil",
    "Modern Horizons",
    "Shadowmoor",
    "Topps Deco",
    "Topps Finest",
    "Topps Museum",
    "Topps Merlin",
    "Topps Stadium Club",
    "Topps Superstars",
    "Topps MLS",
    "Topps Now",
    "Topps Bundesliga",
    "Topps Living Set",
    "Topps UEFA",
    "Topps FC Barcelona",
    "Topps Liverpool",
    "Topps Arsenal",
    "Topps Manchester City",
    "Topps PSG",
    "Topps Bayern Munich",
    "Topps Real Madrid",
    "Topps Argentina",
]

# Canonical mapping for competition synonyms
COMPETITION_CANON = {
    "uefa champions league": "ucl",
    "champions league": "ucl",
    "ucl": "ucl",
    "uefa europa league": "europa league",
    "europa league": "europa league",
    "premier league": "premier league",
    "epl": "premier league",
    "la liga": "la liga",
    "serie a": "serie a",
    "bundesliga": "bundesliga",
    "mls": "mls",
    "nba": "nba",
    "nfl": "nfl",
    "mlb": "mlb",
    "nhl": "nhl",
    "fifa": "fifa",
    "world cup": "world cup",
}


def _normalize_competition(c: str | None) -> str | None:
    if not c:
        return None
    return COMPETITION_CANON.get(c.lower().strip(), c.lower().strip())


RARITY_KEYWORDS = [
    "common",
    "uncommon",
    "rare",
    "ultra rare",
    "super rare",
    "secret rare",
    "holo",
    "holographic",
    "holofoil",
    "foil",
    "parallel",
    "first edition",
    "1st edition",
    "promo",
    "limited",
    "shiny",
    "vmax",
    "vstar",
    "ex",
    "gx",
    "ultra",
    "reverse holo",
    "reverse-holo",
]

PARALLEL_KEYWORDS = [
    "silver",
    "gold",
    "prizm",
    "optic",
    "rainbow",
    "chromium",
    "chrome",
    "green parallel",
    "blue parallel",
]

AUTOGRAPH_KEYWORDS = ["auto", "autograph", "signed", "signature", "certified autograph"]
RELIC_KEYWORDS = ["relic", "patch", "jersey", "memorabilia", "game used"]
LANGUAGE_MAP = {
    "jpn": "Japanese",
    "japanese": "Japanese",
    "eng": "English",
    "english": "English",
    "kor": "Korean",
    "korean": "Korean",
    "ger": "German",
    "german": "German",
    "spa": "Spanish",
    "spanish": "Spanish",
}

# Competition/league keywords (helps separate set vs tournament)
COMPETITION_KEYWORDS = [
    "uefa champions league",
    "champions league",
    "ucl",
    "uefa europa league",
    "europa league",
    "premier league",
    "epl",
    "la liga",
    "serie a",
    "bundesliga",
    "mls",
    "nba",
    "nfl",
    "mlb",
    "nhl",
    "fifa",
    "world cup",
]

# Regex patterns
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_CARD_NUM_RE = re.compile(r"#\s*([0-9]{1,4}[A-Za-z]?)", re.IGNORECASE)
_CARD_NUM_ALT_RE = re.compile(
    r"\b(?:no\.?|card)\s*#?\s*([0-9]{1,4}[A-Za-z-]?)\b",
    re.IGNORECASE,
)
_CARD_NUM_ALNUM_HASH_RE = re.compile(
    r"#\s*([A-Za-z]{1,6}[0-9]{1,4}[A-Za-z]?)\b",
    re.IGNORECASE,
)
_SERIAL_RE = re.compile(r"\b([0-9]{1,4}\s*/\s*[0-9]{1,4})\b")
_GRADING_RE = re.compile(
    r"\b(psa|bgs|cgc|sgc|csg|hga|gma)\s*\.?\s*#?\s*([0-9]{1,2}(?:\.\d)?)\b",
    re.IGNORECASE,
)
_GRADING_ONLY_RE = re.compile(r"\b(psa|bgs|cgc|sgc|csg|hga|gma)\b", re.IGNORECASE)


# ===============================================================
# Normalizer functions (moved from normalizer.py)
# ===============================================================


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    # remove accents, lower, keep slash for serials
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # replace weird punctuation with spaces except slash and numbers
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ===============================================================
# Fuzzy matching helper
# ===============================================================


def _fuzzy_best_match(candidate: str, choices: list[str]) -> tuple[str | None, float]:
    if not candidate or not choices:
        return None, 0.0
    if _HAS_RAPIDFUZZ:
        # Use token_set_ratio for better handling of word order variations
        best = process.extractOne(candidate, choices, scorer=fuzz.token_set_ratio)
        if best:
            name, score, _ = best
            return name, float(score)
        return None, 0.0
    else:
        # difflib fallback (ratio 0..1), convert to 0..100
        match = difflib.get_close_matches(candidate, choices, n=1, cutoff=0.0)
        if not match:
            return None, 0.0
        best = match[0]
        score = difflib.SequenceMatcher(a=candidate, b=best).ratio() * 100.0
        return best, float(score)


# ===============================================================
# Main normalize function
# ===============================================================


def normalize_card_title(
    title: str, set_names: list[str] | None = None, fuzzy_threshold: float = 80.0
) -> dict[str, any]:
    """
    Parse a raw title and return a dict with fixed keys:
    {
      year, set, set_score, rarity, parallel, card_number, serial_number,
      player_or_character, grading, grade, autograph, relic_patch, language,
      confidence: {field: score 0..1}, raw_title, normalized_title
    }
    """
    if set_names is None:
        set_names = DEFAULT_SET_NAMES

    raw = title or ""
    norm = _normalize_text(raw)

    # initialize result with defaults
    result = {
        "year": None,
        "set": None,
        "set_score": 0.0,
        "rarity": None,
        "parallel": None,
        "card_number": None,
        "serial_number": None,
        "player_or_character": None,
        "grading": None,
        "grade": None,
        "autograph": False,
        "relic_patch": False,
        "language": None,
        "competition": None,
        "confidence": {},
        "raw_title": raw,
        "normalized_title": norm,
    }

    # Year
    y = _YEAR_RE.search(norm)
    if y:
        try:
            result["year"] = int(y.group())
            result["confidence"]["year"] = 1.0
        except Exception:
            result["confidence"]["year"] = 0.0
    else:
        result["confidence"]["year"] = 0.0

    # Serial number (xx/yyy, /25, #123, etc.)
    serial = _extract_serial_number(raw)
    if serial:
        result["serial_number"] = serial
        result["confidence"]["serial_number"] = 1.0
        # Remove serial pattern from normalized text for cleaner matching
        serial_patterns_to_remove = [
            f"/{serial}",
            f"#{serial}",
            f"{serial}/",
            f" {serial}/",
        ]
        for pattern in serial_patterns_to_remove:
            norm = norm.replace(pattern.lower(), "")
    else:
        result["confidence"]["serial_number"] = 0.0

    # Card number (#123)
    cn = _CARD_NUM_RE.search(raw)
    if cn:
        result["card_number"] = cn.group(1)
        result["confidence"]["card_number"] = 1.0
        norm = norm.replace(cn.group(0).lower(), "")
    else:
        # Try alternative card number patterns, e.g., "No. 10", "Card #10", alphanumeric
        alt = _CARD_NUM_ALT_RE.search(raw) or _CARD_NUM_ALT_RE.search(norm)
        if alt:
            result["card_number"] = alt.group(1)
            result["confidence"]["card_number"] = 0.9
            norm = norm.replace(alt.group(0).lower(), "")
        else:
            # Fallback: hash-prefixed alphanumeric like #MMB30
            alnum = _CARD_NUM_ALNUM_HASH_RE.search(
                raw
            ) or _CARD_NUM_ALNUM_HASH_RE.search(norm)
            if alnum:
                result["card_number"] = alnum.group(1)
                result["confidence"]["card_number"] = 0.85
                norm = norm.replace(alnum.group(0).lower(), "")
            else:
                result["confidence"]["card_number"] = 0.0

    # Grading (PSA 10, BGS 9.5 etc)
    g = _GRADING_RE.search(raw)
    if g:
        result["grading"] = g.group(1).upper()
        try:
            result["grade"] = float(g.group(2))
        except:
            result["grade"] = None
        result["confidence"]["grading"] = 1.0
        norm = norm.replace(g.group(0).lower(), "")
    else:
        g2 = _GRADING_ONLY_RE.search(raw)
        if g2:
            result["grading"] = g2.group(1).upper()
            result["confidence"]["grading"] = 0.8
            norm = norm.replace(g2.group(0).lower(), "")
        else:
            result["confidence"]["grading"] = 0.0

    # Autograph
    is_auto = any(tok in norm for tok in AUTOGRAPH_KEYWORDS)
    result["autograph"] = bool(is_auto)
    result["confidence"]["autograph"] = 0.95 if is_auto else 0.0
    if is_auto:
        for tok in AUTOGRAPH_KEYWORDS:
            norm = norm.replace(tok, "")

    # Relic/patch
    is_relic = any(tok in norm for tok in RELIC_KEYWORDS)
    result["relic_patch"] = bool(is_relic)
    result["confidence"]["relic_patch"] = 0.95 if is_relic else 0.0
    if is_relic:
        for tok in RELIC_KEYWORDS:
            norm = norm.replace(tok, "")

    # Language
    found_lang = None
    for k, v in LANGUAGE_MAP.items():
        if k in norm:
            found_lang = v
            norm = norm.replace(k, "")
            break
    result["language"] = found_lang
    result["confidence"]["language"] = 0.9 if found_lang else 0.0

    # Competition / league
    found_comp = None
    for comp in COMPETITION_KEYWORDS:
        if comp in norm:
            found_comp = comp
            norm = norm.replace(comp, "")
            break
    result["competition"] = found_comp
    result["confidence"]["competition"] = 0.9 if found_comp else 0.0

    # Rarity & Parallel
    found_rarity = None
    for r in RARITY_KEYWORDS:
        if r in norm:
            found_rarity = r
            norm = norm.replace(r, "")
            break
    result["rarity"] = found_rarity
    result["confidence"]["rarity"] = 0.95 if found_rarity else 0.0

    found_parallel = None
    for p in PARALLEL_KEYWORDS:
        if p in norm:
            found_parallel = p
            norm = norm.replace(p, "")
            break
    result["parallel"] = found_parallel
    result["confidence"]["parallel"] = 0.95 if found_parallel else 0.0

    # Set detection: first exact substring match, then fuzzy match
    found_set = None
    set_score = 0.0

    # Try direct substring matching first (more precise)
    for set_name in set_names:
        if set_name.lower() in norm:
            found_set = set_name
            set_score = 1.0
            norm = norm.replace(set_name.lower(), "")
            break

    # If no direct match, try fuzzy matching with lower threshold
    if not found_set:
        best_name, score = _fuzzy_best_match(norm, set_names)
        if best_name and score >= fuzzy_threshold:
            found_set = best_name
            set_score = score / 100.0
            norm = norm.replace(best_name.lower(), "")
        else:
            set_score = score / 100.0 if best_name else 0.0

    result["set"] = found_set
    result["set_score"] = set_score
    result["confidence"]["set"] = set_score

    # Player/character extraction: leftover tokens after removals
    leftovers = re.sub(
        r"\b(card|cards|pokemon|pokémon|mtg|yugioh|yu gi oh|yugi)\b", " ", norm
    )
    leftovers = re.sub(r"[^a-z0-9 ]", " ", leftovers)
    leftovers = re.sub(r"\s+", " ", leftovers).strip()
    result["player_or_character"] = leftovers.title() if leftovers else None
    result["confidence"]["player_or_character"] = 0.9 if leftovers else 0.0

    # ensure normalized_title updated
    result["normalized_title"] = _normalize_text(result["raw_title"])

    return result


# ===============================================================
# OpenAI config
# ===============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
RERANK_MODEL = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "128"))
# Disable embeddings by default (no paid embedding) unless explicitly enabled
EMBED_ENABLED = os.getenv("OPENAI_EMBED_ENABLED", "0") == "1"
GPT_MODEL = RERANK_MODEL

# ===============================================================
# Selection config (deterministic + optional rerank)
# ===============================================================

# Weights (configurable)
WEIGHT_NAME = float(os.getenv("SEL_W_NAME", "0.35"))
WEIGHT_SET = float(os.getenv("SEL_W_SET", "0.25"))
WEIGHT_NUMBER = float(os.getenv("SEL_W_NUMBER", "0.10"))
WEIGHT_YEAR = float(os.getenv("SEL_W_YEAR", "0.15"))
WEIGHT_VARIATION = float(os.getenv("SEL_W_VARIATION", "0.05"))
WEIGHT_SERIAL = float(os.getenv("SEL_W_SERIAL", "0.05"))
WEIGHT_GRADING = float(os.getenv("SEL_W_GRADING", "0.05"))

SCORE_THRESHOLD = float(os.getenv("SEL_SCORE_THRESHOLD", "0.75"))
RERANK_IF_TOP_LT = float(os.getenv("SEL_RERANK_IF_TOP_LT", "0.98"))
RERANK_IF_TIE_DELTA = float(os.getenv("SEL_RERANK_IF_TIE_DELTA", "0.05"))
TOP_K_FOR_RERANK = int(os.getenv("SEL_TOP_K_FOR_RERANK", "5"))


# ===============================================================
# Helper: compute cosine similarity
# ===============================================================


def _cosine_sim(a: list[float], b: list[float]) -> float:
    # returns similarity in [-1..1]
    if not a or not b or len(a) != len(b):
        return -1.0
    # compute dot / (||a|| * ||b||)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ===============================================================
# OpenAI helpers (async)
# ===============================================================


async def _openai_post(
    path: str, payload: dict[str, any], timeout: int = 30
) -> dict[str, any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    url = OPENAI_BASE.rstrip("/") + path
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


# ===============================================================
# Embed texts using OpenAI
# ===============================================================


async def embed_texts(texts: list[str], model: str = None) -> list[list[float]]:
    model = model or EMBED_MODEL
    if not texts:
        return []
    # OpenAI embeddings endpoint
    # Batch sizes handled by caller
    payload = {"model": model, "input": texts}
    data = await _openai_post("/v1/embeddings", payload, timeout=60)
    # data['data'] -> list of dicts {embedding: [...], index: i}
    return [item["embedding"] for item in data.get("data", [])]


# ===============================================================
# GPT rerank prompt
# ===============================================================


async def gpt_rerank_prompt(
    prompt: str, model: str = None, max_tokens: int = 512, json_response: bool = True
) -> str:
    model = model or RERANK_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    # Prefer structured JSON to make parsing robust
    if json_response:
        payload["response_format"] = {"type": "json_object"}
    data = await _openai_post("/v1/chat/completions", payload, timeout=60)
    return data["choices"][0]["message"]["content"]


# ===============================================================
# Card lookup + matching helpers (moved from utils/ebay.py)
# ===============================================================


async def get_card(card_id: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("cards")
            .select(
                "canonical_title, years, card_set, name, variation, serial_number, number, grading_company, grade_number"
            )
            .eq("id", card_id)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]
    except Exception as e:
        scraper_logger.info(f"⚠️ Failed to resolve card: {e}")
    return None


def normalized_text(val: Any) -> Optional[str]:
    if not val or val == "":
        return None
    return str(val).strip().lower()


def word_match(
    title: Optional[str],
    filter: Optional[str],
    word_threshold: Optional[int] = None,
) -> bool:
    """Return True if title matches filter using token and fuzzy checks with count-based threshold.

    - title: text to search within (can be None; treated as empty string)
    - filter: space-delimited string of tokens to check (None/empty => False)
    - word_threshold: minimum number of tokens/phrases that must match (>= 1). If None or > token count, require all tokens.

    Special rules:
    - Tokens are split on whitespace and evaluated independently (even single-character tokens like "1").
    - Use RapidFuzz per token as a flexible fallback (no static alias expansions).
    """
    if not filter:
        return True
    # Normalize title
    title_text = (title or "").lower().strip()
    filter_text = str(filter).lower().strip()
    # Build collapsed variants to catch cases like 'logo fractor' vs 'logofractor' or 'logo-fractor'
    title_collapsed = title_text.replace(" ", "")
    filter_collapsed = filter_text.replace(" ", "")
    # Alphanumeric-only squeeze (remove all non a-z0-9)
    title_squeezed = re.sub(r"[^a-z0-9]+", "", title_text)
    filter_squeezed = re.sub(r"[^a-z0-9]+", "", filter_text)
    # Tokenize the filter on whitespace; do not merge short tokens
    tokens = [t for t in filter_text.split() if t]
    if not tokens:
        return False

    token_count = len(tokens)
    # Compute required matches based on word_threshold (default: require all)
    required = (
        token_count
        if not word_threshold or word_threshold <= 0
        else min(word_threshold, token_count)
    )

    # 1) If requiring all tokens, allow an early perfect whole-string equality
    if required == token_count:
        if fuzz.token_set_ratio(title_text, filter_text) >= 100:
            return True
        if fuzz.token_set_ratio(title_collapsed, filter_collapsed) >= 100:
            return True
        if fuzz.token_set_ratio(title_squeezed, filter_squeezed) >= 100:
            return True

    # 2) Count token coverage (exact or fuzzy) for tokens
    matched = 0
    fuzzy_token_min_ratio = 80  # fixed per-token fuzzy similarity threshold
    for t in tokens:
        tc = t.replace(" ", "")
        ts = re.sub(r"[^a-z0-9]+", "", t)
        if t in title_text or tc in title_collapsed or ts in title_squeezed:
            matched += 1
            continue
        # last resort fuzzy per-token when available
        if (
            fuzz.partial_ratio(t, title_text) >= fuzzy_token_min_ratio
            or fuzz.partial_ratio(tc, title_collapsed) >= fuzzy_token_min_ratio
            or fuzz.partial_ratio(ts, title_squeezed) >= fuzzy_token_min_ratio
        ):
            matched += 1

    return matched >= required


def all_present_matched(
    attribute_filters: List[str], req: List[str], matched: Dict[str, bool]
) -> bool:
    filtered_req = [k for k in req if k in attribute_filters]
    return all(matched.get(k, False) for k in filtered_req)


def build_strict_promisable_candidates(
    attribute_filters: Dict[str, Any],
    posts: List[Dict],
) -> Dict[str, List[Dict]]:
    low_req = ["years", "name"]
    med_req = ["years", "card_set", "name"]
    high_req = ["years", "card_set", "name", "variation"]

    # -----------------------------------------------------------
    # Normalize card-side values once
    # -----------------------------------------------------------
    unmatch_posts: List[Dict] = []
    match_posts: List[Dict] = []
    # Debug counters
    low_cnt = med_cnt = high_cnt = unmatch_cnt = fallback_cnt = 0

    # Log incoming filters once for debugging
    try:
        dbg_filters = {
            k: v
            for k, v in attribute_filters.items()
            if k
            in ["years", "card_set", "name", "variation", "serial_number", "number"]
        }
        scraper_logger.info(
            f"[build_candidates] filters: {json.dumps(dbg_filters, ensure_ascii=False)}"
        )
    except Exception:
        pass

    card_years = normalized_text(attribute_filters.get("years", ""))
    card_set = normalized_text(attribute_filters.get("card_set", ""))
    card_name = normalized_text(attribute_filters.get("name", ""))
    card_variation = normalized_text(attribute_filters.get("variation", ""))
    card_serial = normalized_text(attribute_filters.get("serial_number", ""))
    card_number = normalized_text(attribute_filters.get("number", ""))

    for ebay_post in posts:
        # -----------------------------------------------------------
        # Post normalization helpers
        # -----------------------------------------------------------
        title = ebay_post.get("normalized_title")
        matched = {
            # Years: match any of the provided years (threshold=1)
            "years": word_match(title, card_years, 1),
            "card_set": word_match(title, card_set, 2),
            "name": word_match(title, card_name),
            "variation": word_match(title, card_variation, 0),
            "serial_number": word_match(title, card_serial),
            "number": word_match(title, card_number),
        }

        if all_present_matched(attribute_filters, low_req, matched):
            matched["candidate_score"] = "low"
            low_cnt += 1
        elif all_present_matched(attribute_filters, med_req, matched):
            matched["candidate_score"] = "medium"
            med_cnt += 1
        elif all_present_matched(attribute_filters, high_req, matched):
            matched["candidate_score"] = "high"
            high_cnt += 1
        else:
            matched["candidate_score"] = "unmatch"
            unmatch_cnt += 1

        ebay_post["matched"] = matched

        if matched["candidate_score"] == "unmatch":
            # Fallback: allow GPT to refine when set+name look good, even if year is missing
            if matched.get("card_set") and matched.get("name"):
                matched["candidate_score"] = "fallback"
                match_posts.append(ebay_post)
                fallback_cnt += 1
            else:
                unmatch_posts.append(ebay_post)
        else:
            match_posts.append(ebay_post)

    # Log summary counts and up to 5 sample unmatches for diagnosis
    try:
        scraper_logger.info(
            f"[build_candidates] counts -> low={low_cnt}, med={med_cnt}, high={high_cnt}, fallback={fallback_cnt}, unmatch={unmatch_cnt}"
        )
    except Exception:
        pass

    return {"unmatch_posts": unmatch_posts, "match_posts": match_posts}


# ===============================================================
# GPT selection (Step 4 for card_id mode)
# ===============================================================


async def select_best_candidate_with_gpt(
    canonical_title: Optional[str],
    match_posts: List[Dict],
    use_gpt: bool = True,
) -> Optional[Dict]:
    """Return the single best candidate (full post) chosen by GPT and finalized by most recent sold_date.

    Flow:
    - Build a compact list from ALL `match_posts` for GPT (index + id + titles).
    - GPT returns indices of likely matches.
    - Map indices back to `match_posts` to obtain the candidate list.
    - Print the compact entries for the selected indices for debugging.
    - Choose the most recent by sold_date and return that single post.
    """
    if len(match_posts) == 0:
        scraper_logger.info("❌ No match posts found; selection returns None")
        return None

    if not use_gpt or not OPENAI_API_KEY:
        scraper_logger.info(
            "❌ GPT disabled or missing API key; selection returns None"
        )
        try:
            scraper_logger.info(f"gpt_disabled_compact_size: {len(match_posts)}")
        except Exception:
            pass
        return None

    compact: List[Dict] = []
    for idx, match_post in enumerate(match_posts):
        compact.append(
            {
                "i": idx,  # index used for selection
                "id": match_post.get("id"),
                "raw": str(match_post.get("title")),
                "normalized": str(match_post.get("normalized_title")),
            }
        )

    system_prompt = (
        "You are an EXPERT trading-card matcher. Decide if each candidate describes the EXACT same card as the canonical title. Rules: "
        "1) Mandatory: YEAR and PLAYER/CHARACTER NAME must match (allow obvious synonyms/spelling/case/hyphen/space variants). "
        "2) SET/BRAND/SERIES (card_set) must match conceptually (same product line/family). "
        "3) VARIATION/PARALLEL/SUBSET handling: If the canonical specifies a variation (e.g., 'Gold Logofractor /50'), then: "
        "   • If a candidate explicitly states a DIFFERENT variation (e.g., 'Sapphire', 'Speckle', 'Auto/Autograph'), EXCLUDE it. "
        "   • If a candidate omits variation tokens (no explicit variant) but YEAR, NAME, and SET match, ALLOW it. "
        "4) Bonus: SERIAL NUMBER and CARD NUMBER help but are not required. Prefer matches that also align on these when present. "
        "5) Super bonus: GRADING COMPANY and GRADE NUMBER are informative but never required; never exclude solely due to grading differences. "
        "Normalize case/hyphens/spaces. Allowed synonyms: 'f1' == 'formula 1' == 'formula one'; 'logofractor' == 'logo fractor'. "
        "Set equivalence: 'Topps Chrome F1' == 'Topps Chrome Formula 1'. Different product line: 'Topps Turbo Attax' ≠ 'Topps Chrome F1'. "
        "Explicit different variation keywords to EXCLUDE when canonical specifies a different one: 'sapphire', 'speckle', 'speckle refractor', 'auto', 'autograph', 'refractor', 'xfractor', 'x-fractor', 'prism', 'prizm'. "
        "Examples: Canonical '2024 Topps Chrome F1 Kimi Antonelli /50 Gold Logofractor' → INCLUDE '2024 Topps Chrome F1 #94 Andrea Kimi Antonelli' (no variation stated); EXCLUDE '2024 Topps Chrome Sapphire F1 ...', '... Auto ...', '... Speckle ...', 'Topps Turbo Attax ...'. "
        "Output a JSON object with keys: 'matches' (array of indices) and 'reason' (string). Include only indices with ≥96% confidence; if none qualify, return an empty list and explain why in 'reason'."
    )

    user_prompt = {
        "canonical_title": canonical_title or "",
        "candidates": compact,
        "instructions": (
            "Compare 'canonical_title' with each candidate using both 'raw' and 'normalized' titles. "
            "Include an index ONLY IF: YEAR matches and PLAYER/CHARACTER NAME matches. SET/BRAND/SERIES must also match conceptually (e.g., Chrome F1 == Chrome Formula 1; Turbo Attax is a different set). "
            "If the canonical specifies a variation, exclude candidates that explicitly state a different variation (e.g., 'Sapphire', 'Speckle', 'Auto'). "
            "If a candidate omits variation tokens but YEAR, NAME, and SET match, include it. "
            "CARD NUMBER and SERIAL NUMBER are bonuses (prefer but do not require). GRADING COMPANY and GRADE NUMBER are super bonuses (never require). "
            "Return JSON: {\"matches\": [i1, i2, ...], \"reason\": <string>}. Include only indices with ≥96% confidence; if none qualify, return an empty list and set 'reason' to a concise explanation (e.g., 'year mismatch', 'name mismatch', 'set mismatch', or 'explicit different variation')."
        ),
    }

    try:
        url = OPENAI_BASE.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GPT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        idxs = parsed.get("matches", [])
        reason = parsed.get("reason")
        if not isinstance(idxs, list):
            try:
                scraper_logger.info("gpt_reason:", reason or "invalid_response_format")
                scraper_logger.info(
                    "gpt_compact_all:", json.dumps(compact, ensure_ascii=False)
                )
            except Exception:
                scraper_logger.info("gpt_reason:", reason or "invalid_response_format")
                scraper_logger.info("gpt_compact_all:", compact)
            return None

        selected_indices: List[int] = [
            i for i in idxs if isinstance(i, int) and 0 <= i < len(match_posts)
        ]
        if not selected_indices:
            try:
                scraper_logger.info("gpt_reason:", reason or "no_matches")
                scraper_logger.info(
                    "gpt_compact_all:", json.dumps(compact, ensure_ascii=False)
                )
            except Exception:
                scraper_logger.info("gpt_reason:", reason or "no_matches")
                scraper_logger.info("gpt_compact_all:", compact)
            return None

        selected_posts: List[Dict] = [match_posts[i] for i in selected_indices]
        selected_compact: List[Dict] = [compact[i] for i in selected_indices]

        try:
            scraper_logger.info(
                "gpt_selected_compact:",
                json.dumps(selected_compact, ensure_ascii=False),
            )
        except Exception:
            scraper_logger.info("gpt_selected_compact:", selected_compact)

        def sold_ts(p: Dict) -> float:
            d = p.get("sold_date") or p.get("sold_at") or ""
            try:
                return date_parser.parse(d, fuzzy=True).timestamp()
            except Exception:
                return 0.0

        most_recent_post = max(selected_posts, key=sold_ts)
        return most_recent_post
    except Exception as e:
        scraper_logger.info(f"⚠️ GPT selection failed: {e}")
        try:
            scraper_logger.info(f"gpt_error: {str(e)}")
        except Exception:
            pass
        return None


# ===============================================================
# Get HTML Content
# ===============================================================


async def scrape_ebay_html(
    region: Region,
    query: str | None = None,
    category_id: CategoryId | None = None,
    card_id: str | None = None,
    max_pages: int = 50,
    page_retries: int = 3,
    disable_proxy: bool = False,
) -> list[str]:
    scraper_logger.info(
        f"[scrape_ebay_html] params: query={query}, category_id={category_id}, card_id={card_id}"
    )
    # (same logic as before) - build config, request first page to determine total pages, loop pages
    final_query = query
    final_category_id = category_id

    # -----------------------------------------------
    # Mode: card_id
    # -----------------------------------------------
    if card_id:
        try:
            response = (
                supabase.table("cards")
                .select("canonical_title, category_id")
                .eq("id", card_id)
                .limit(1)
                .execute()
            )
            if response.data:
                card_data = response.data[0]
                final_query = card_data.get("canonical_title")
                final_category_id = (
                    CategoryId(card_data.get("category_id"))
                    if card_data.get("category_id")
                    else None
                )
                scraper_logger.info(
                    f"✅ [Scraping Mode: card_id] ------------------------------------"
                )
        except Exception as e:
            scraper_logger.warning(f"⚠️ Error resolving card_id {card_id}: {e}")
    # -----------------------------------------------
    # Mode: query + category_id
    # -----------------------------------------------
    if query and category_id:
        scraper_logger.info(
            f"✅ [Scraping Mode:] ------------------ query + category_id ------------------"
        )

    config = {
        "query": final_query,
        "category_id": final_category_id.value if final_category_id else None,
        "region_str": region.value,
        "base_url": f"{base_target_url[region.value]}/sch/i.html",
    }

    total_pages = 1
    if max_pages > 1:
        try:
            params = {
                "_nkw": config["query"],
                "_sacat": "0",
                "_from": "R40",
                "rt": "nc",
                "LH_Sold": "1",
                "LH_Complete": "1",
                "Country/Region of Manufacture": (
                    "United Kingdom"
                    if config["region_str"] == "uk"
                    else "United States"
                ),
                "_ipg": "240",
                "_pgn": "1",
                "_sop": "12",
            }
            scraper_logger.info(
                f"📡 [Scraping eBay start] with query --- {config['base_url']}?{urlencode(params)}"
            )
            html = await httpx_get_content(
                config["base_url"], params=params, use_proxy=not disable_proxy
            )
            if html:
                soup = BeautifulSoup(html, "html.parser")
                count_element = soup.select_one(
                    ".srp-controls__count-heading, .result-count__count-heading"
                )
                if count_element:
                    bold = count_element.find("span", class_="BOLD")
                    text = (
                        bold.get_text(strip=True)
                        if bold
                        else count_element.get_text(" ", strip=True)
                    )
                    match = re.search(r"(\d[\d,]*)", text)
                    if match:
                        total_results = int(match.group(1).replace(",", ""))
                        total_pages = max(1, math.ceil(total_results / 240))
        except Exception as e:
            scraper_logger.warning(f"⚠️ Error getting page count: {e}")

    total_pages = min(max_pages, total_pages)
    scraper_logger.info(f"🚀 [Scraping {total_pages} pages] ---- '{final_query}'")

    html_pages = []
    for page in range(1, total_pages + 1):
        if page > 1:
            delay = random.uniform(1.0, 3.5) + (page - 1) * random.uniform(0.2, 0.8)
            await asyncio.sleep(delay)

        page_html = None
        for attempt in range(1, page_retries + 1):
            try:
                params = {
                    "_nkw": config["query"],
                    "_sacat": (
                        str(config["category_id"]) if config["category_id"] else "0"
                    ),
                    "_from": "R40",
                    "rt": "nc",
                    "LH_Sold": "1",
                    "LH_Complete": "1",
                    "Country/Region of Manufacture": (
                        "United Kingdom"
                        if config["region_str"] == "uk"
                        else "United States"
                    ),
                    "_ipg": "240",
                    "_pgn": str(page),
                    "_sop": "12",
                }
                scraper_logger.info(
                    f"📡 [eBay URL] --- {config['base_url']}?{urlencode(params)}"
                )
                timeout = random.uniform(12.0, 18.0)
                jitter_min = random.uniform(0.5, 1.2)
                jitter_max = random.uniform(1.5, 3.0)
                page_html = await httpx_get_content(
                    config["base_url"],
                    params=params,
                    attempts=2,
                    request_timeout=timeout,
                    jitter_min=jitter_min,
                    jitter_max=jitter_max,
                    use_proxy=not disable_proxy,
                )
                if page_html:
                    break
            except Exception as e:
                scraper_logger.warning(
                    f"⚠️ Error scraping page {page} attempt {attempt}: {e}"
                )
                if attempt < page_retries:
                    await asyncio.sleep(random.uniform(2.0, 5.0))
        if page_html:
            html_pages.append(page_html)

    scraper_logger.info(f"📄 Successfully scraped {len(html_pages)} HTML pages")
    return html_pages


# ===============================================================
# Extract Ebay HTML content to data
# ===============================================================


async def extract_ebay_post_data(
    html_pages: list[str], region: str, category_id: int | None = None
) -> list[dict[str, any]]:
    all_posts: list[dict[str, any]] = []
    for html in html_pages:
        soup = BeautifulSoup(html, "html.parser")
        items = soup.select("li[data-listingid]")
        for item in items:
            try:
                listing_id = item.get("data-listingid")
                if not listing_id:
                    continue

                # title
                title = None
                title_el = item.select_one(".s-card__title .su-styled-text")
                if title_el and title_el.get_text().strip():
                    title = title_el.get_text(strip=True)
                if not title or title.lower() == "shop on ebay":
                    continue

                # price / currency
                price = None
                currency = None
                price_el = item.select_one(".su-styled-text.s-card__price")
                if price_el:
                    price_text = price_el.get_text(strip=True)
                    if "£" in price_text:
                        currency = "GBP"
                        price_match = re.search(
                            r"£([\d,]+\.?\d*)", price_text.replace(",", "")
                        )
                    elif "$" in price_text:
                        currency = "USD"
                        price_match = re.search(
                            r"\$([\d,]+\.?\d*)", price_text.replace(",", "")
                        )
                    elif "€" in price_text:
                        currency = "EUR"
                        price_match = re.search(
                            r"€([\d,]+\.?\d*)", price_text.replace(",", "")
                        )
                    else:
                        price_match = re.search(
                            r"([\d,]+\.?\d*)", price_text.replace(",", "")
                        )
                        currency = "GBP" if region == "uk" else "USD"
                    if price_match:
                        price = price_match.group(1)
                if not price:
                    continue
                if not currency:
                    currency = "GBP" if region == "uk" else "USD"

                # url
                url = ""
                link_el = item.select_one("a[href*='/itm/']")
                if link_el:
                    url = link_el.get("href")
                if not url:
                    continue

                # images
                image_url = ""
                image_hd_url = ""
                img_el = item.select_one(".s-card__image") or item.select_one(
                    "img[src*='ebayimg.com']"
                )
                if img_el:
                    src = (
                        img_el.get("src")
                        or img_el.get("data-src")
                        or img_el.get("data-defer-load")
                    )
                    if src and "ebayimg.com" in src:
                        if "s-l140" in src:
                            image_url = src
                            image_hd_url = src.replace("s-l140", "s-l1600")
                        elif "s-l500" in src:
                            image_url = src.replace("s-l500", "s-l140")
                            image_hd_url = src.replace("s-l500", "s-l1600")
                        else:
                            image_url = src
                            if src and "i.ebayimg.com" in src:
                                image_hd_url = re.sub(
                                    r"/s-l[0-9]{2,4}", "/s-l1600", src
                                )
                            else:
                                image_hd_url = src

                # sold date
                sold_date = None
                sold_el = item.select_one(".s-card__caption .su-styled-text")
                if sold_el:
                    sold_text = sold_el.get_text(strip=True)
                    if "sold" in sold_text.lower():
                        try:
                            dt = date_parser.parse(sold_text, fuzzy=True, dayfirst=True)
                            tz = pytz.timezone(
                                "Europe/London"
                                if region == "uk"
                                else "America/New_York"
                            )
                            dt = dt if dt.tzinfo else tz.localize(dt)
                            sold_date = dt.isoformat()
                        except Exception:
                            sold_date = datetime.utcnow().isoformat()
                if not sold_date:
                    sold_date = datetime.utcnow().isoformat()

                # --- Step 1: Normalize title (deterministic + fuzzy) ---
                norm_attrs = normalize_card_title(title)

                # build tags & top-level compat fields
                condition = "graded" if norm_attrs.get("grading") else "raw"
                grading_company = norm_attrs.get("grading")
                grade = norm_attrs.get("grade")
                normalized_name = norm_attrs.get("normalized_title") or ""

                # tags: set, player tokens, rarity, parallel
                tags = []
                if norm_attrs.get("set"):
                    tags.append(norm_attrs["set"].lower())
                if norm_attrs.get("player_or_character"):
                    tags.extend(
                        [w.lower() for w in norm_attrs["player_or_character"].split()][
                            :6
                        ]
                    )
                if norm_attrs.get("rarity"):
                    tags.append(norm_attrs["rarity"].lower())
                if norm_attrs.get("parallel"):
                    tags.append(norm_attrs["parallel"].lower())
                # dedupe & limit
                seen = set()
                clean_tags = []
                for t in tags:
                    if not t or t in seen:
                        continue
                    seen.add(t)
                    clean_tags.append(t)
                    if len(clean_tags) >= 8:
                        break
                tags = clean_tags

                post_data: dict[str, any] = {
                    "id": listing_id,
                    "title": title,
                    "image_url": image_url,
                    "image_hd_url": image_hd_url,
                    # top-level compatibility
                    "condition": condition,
                    "grading_company": grading_company,
                    "grade": grade,
                    "normalized_title": normalized_name,
                    # nested metadata
                    # "metadata": {
                    #     "condition": condition,
                    #     "grading_company": grading_company,
                    #     "grade": grade,
                    #     "region": region,
                    #     "category_id": category_id,
                    #     "normalized_name": normalized_name,
                    #     "tags": tags,
                    #     "normalized_attributes": norm_attrs,
                    # },
                    "sold_date": sold_date,
                    "sold_price": price,
                    "sold_currency": currency,
                    "sold_post_url": url,
                    "embedding": None,  # Will be filled in next step
                }

                all_posts.append(post_data)

            except Exception as e:
                scraper_logger.warning(f"⚠️ Failed to extract item data: {e}")
                continue

    # dedupe by url
    seen_urls = set()
    unique_posts: list[dict[str, any]] = []
    for post in all_posts:
        url = post.get("sold_post_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_posts.append(post)

    scraper_logger.info(
        f"📦 Extracted {len(unique_posts)} unique eBay post items (after normalization)"
    )
    return unique_posts


# ===============================================================
# Add Embeddings to Posts
# ===============================================================


async def embed_posts(
    posts: list[dict[str, any]], batch_size: int = EMBED_BATCH
) -> None:
    if not posts:
        return

    if not EMBED_ENABLED:
        # Explicitly skip embedding to avoid paid embedding usage
        for p in posts:
            p["embedding"] = None
        scraper_logger.info("🧠 Embeddings disabled; skipping embedding stage")
        return posts

    model = EMBED_MODEL
    texts = [p["metadata"]["normalized_attributes"]["normalized_title"] for p in posts]
    embeddings: list[list[float]] = []
    # batch
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            embs = await embed_texts(batch, model=model)
            embeddings.extend(embs)
        except Exception as e:
            scraper_logger.error(f"❌ Embedding batch failed: {e}")
            # fallback: fill batch with None
            embeddings.extend([None] * len(batch))
    # attach embeddings to posts
    for p, emb in zip(posts, embeddings):
        p["embedding"] = emb

    return posts


# ===============================================================
# Selection Last Sold By AI (Post Ranking)
# ===============================================================


async def select_best_ebay_post(
    posts: list[dict[str, any]],
    user_query: str,
    top_k: int = 5,
    rerank_with_gpt: bool = True,
    strict: bool = False,
) -> dict[str, any] | None:
    """Select best eBay post using precise attribute matching and GPT-4o-mini rerank (no embeddings)."""
    if not posts:
        return None

    scraper_logger.info(f"🎯 Starting selection from {len(posts)} posts...")

    # Normalize user query to extract attributes
    user_attrs = normalize_card_title(user_query)
    # scraper_logger.info(f"🔍 Extracted user attributes: {user_attrs}")

    # Extract exact match criteria from user_attrs
    user_year = user_attrs.get("year")
    user_set = user_attrs.get("set")
    user_player = user_attrs.get("player_or_character")
    # Prefer structured card_number then regex fallback
    user_card_no = user_attrs.get("card_number")
    if not user_card_no:
        m = _CARD_NUM_ALT_RE.search(user_attrs.get("normalized_title", ""))
        user_card_no = m.group(1) if m else None
    # Competition
    user_comp = _normalize_competition(user_attrs.get("competition"))

    # Filter posts that match exactly on year, set, and player (if they are specified)
    filtered_posts = []

    def _norm_card_no(s: str | None) -> str | None:
        if not s:
            return None
        return re.sub(r"[^a-z0-9]", "", s.lower())

    for post in posts:
        post_attrs = post.get("metadata", {}).get("normalized_attributes", {})
        # Check year
        if user_year is not None and post_attrs.get("year") != user_year:
            continue
        # Check set
        if user_set is not None and post_attrs.get("set") != user_set:
            continue
        # Check player/character
        if (
            user_player is not None
            and post_attrs.get("player_or_character") != user_player
        ):
            continue
        # Strict-only requirements
        if strict:
            # Card number must match if present in canonical title
            if user_card_no is not None:
                post_card_no = post_attrs.get("card_number")
                if not post_card_no:
                    m2 = _CARD_NUM_ALT_RE.search(post_attrs.get("normalized_title", ""))
                    post_card_no = m2.group(1) if m2 else None
                if _norm_card_no(user_card_no) != _norm_card_no(post_card_no):
                    continue
            # Competition must match if present in canonical title
            post_comp = _normalize_competition(post_attrs.get("competition"))
            if user_comp and (post_comp != user_comp):
                continue
        filtered_posts.append(post)

    # If no posts match the exact criteria, fall back to all posts
    if not filtered_posts:
        scraper_logger.info(
            "No exact matches found for year, set, and player. Falling back to all posts."
        )
        filtered_posts = posts
    else:
        scraper_logger.info(
            f"Filtered to {len(filtered_posts)} posts with exact matches for year, set, and player."
        )

    # Remove obviously irrelevant listings only in strict mode (card_id flow)
    if strict:

        def _is_disqualified(p: dict[str, any]) -> bool:
            t = (p.get("title") or "").lower()
            user_norm = (user_attrs.get("normalized_title") or "").lower()
            banned = [
                "lot of",
                "lot",
                "bundle",
                "case",
                "box",
                "pack",
                "sealed box",
                "sealed pack",
                "break",
                "proxy",
                "reprint",
                "custom card",
                "digital",
            ]
            for b in banned:
                if b in t and b not in user_norm:
                    return True
            return False

        filtered_posts = [p for p in filtered_posts if not _is_disqualified(p)]

    # Limit posts to prevent timeout (take first 100 for efficiency)
    if len(filtered_posts) > 100:
        scraper_logger.info(f"⚡ Limiting to first 100 posts for efficiency")
        filtered_posts = filtered_posts[:100]

    # Score posts using attribute matching only (no embeddings)
    scored = []
    for i, post in enumerate(filtered_posts):
        post_attrs = post.get("metadata", {}).get("normalized_attributes", {})

        # Calculate attribute match score (0-1)
        attr_score = _calculate_attribute_match_score(user_attrs, post_attrs)

        # Combined score is purely attribute-based now
        final_score = attr_score

        scored.append((post, final_score, attr_score))

    if not scored:
        return None

    # Sort by combined score and take top-k
    scored.sort(key=lambda x: x[1], reverse=True)
    top_posts = scored[:top_k]

    # Optional GPT reranking for close matches
    best_post, best_score, best_attr = top_posts[0]
    gpt_debug = None
    if rerank_with_gpt and len(top_posts) > 1 and best_score < 0.9:
        # Prepare structured candidate data for LLM
        candidates = []
        for idx, (p, score, attr) in enumerate(top_posts, start=1):
            pa = p.get("metadata", {}).get("normalized_attributes", {})
            candidates.append(
                {
                    "idx": idx,
                    "title": p.get("title", ""),
                    "price": p.get("sold_price"),
                    "currency": p.get("sold_currency"),
                    "url": p.get("sold_post_url"),
                    "attrs": {
                        "year": pa.get("year"),
                        "set": pa.get("set"),
                        "player_or_character": pa.get("player_or_character"),
                        "card_number": pa.get("card_number"),
                        "serial_number": pa.get("serial_number"),
                        "grading": pa.get("grading"),
                        "grade": pa.get("grade"),
                        "parallel": pa.get("parallel"),
                        "rarity": pa.get("rarity"),
                        "language": pa.get("language"),
                        "competition": pa.get("competition"),
                    },
                    "attr_score": round(float(attr), 4),
                }
            )

        prompt = (
            "You are evaluating which eBay sold listing best matches a TCG card title.\n"
            "Rules (in order of importance):\n"
            "1) Player/Character name must match (first+last if present).\n"
            "2) Set name must match (allow close synonyms but prefer exact).\n"
            "3) Year should match; minor season offsets (e.g., 2024 vs 2024-25) acceptable.\n"
            "4) If the user title includes a serial number or card number, it must match.\n"
            "5) Respect grading/grade if mentioned; otherwise ignore grading.\n"
            "6) Ignore lots or unrelated items even if similar.\n"
            "Given the user query and candidates, choose the single best match.\n"
            'Return STRICT JSON with fields: {"choice": <int idx>, "reason": <string>, "confidence": <0..1>}\n\n'
            f"User query: {user_query}\n\n"
            f"Candidates: {json.dumps(candidates, ensure_ascii=False)}"
        )

        try:
            gpt_content = await gpt_rerank_prompt(prompt, json_response=True)
            choice_data = json.loads(gpt_content)
            choice_idx = int(choice_data.get("choice", 1)) - 1
            gpt_debug = choice_data
            if 0 <= choice_idx < len(top_posts):
                best_post, best_score, best_attr = top_posts[choice_idx]
        except Exception as e:
            scraper_logger.warning(
                f"⚠️ GPT rerank failed ({e}), using attribute-based selection"
            )

    return {
        "found": True,
        "match": {
            "title": best_post.get("title", ""),
            "sold_price": best_post.get("sold_price"),
            "sold_currency": best_post.get("sold_currency"),
            "sold_post_url": best_post.get("sold_post_url"),
            "sold_date": best_post.get("sold_date"),
            "sold_at": best_post.get("sold_date"),
            "similarity": best_score,
            "attribute_score": best_attr,
            "metadata": best_post.get("metadata", {}),
        },
        "debug_info": {
            "top_k_scores": [(p["title"], score, attr) for p, score, attr in top_posts],
            "total_candidates": len(scored),
            "user_attributes": user_attrs,
            "gpt_choice": gpt_debug,
        },
    }


# ===============================================================
# Deterministic selector using card_row (source of truth)
# ===============================================================


def _normalize_hyphens_and_spaces(s: str) -> str:
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_for_compare(s: str) -> str:
    if not s:
        return ""
    s = _normalize_hyphens_and_spaces(s)
    s = unicodedata.normalize("NFKD", s).lower()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # keep a-z0-9, space, #, / and -
    s = re.sub(r"[^a-z0-9#/_\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_number_token(s: str) -> str:
    if not s:
        return ""
    s = _normalize_for_compare(s)
    # strip prefixes like no., card
    s = re.sub(r"\b(?:no|no\.|card)\b", "", s).strip()
    # keep first #?alnum(-alnum) token (allow alnum dashes)
    m = re.search(r"#?[a-z0-9]+(?:-[a-z0-9]+)*", s)
    return m.group(0) if m else s


def _tokenize(s: str) -> list[str]:
    s = _normalize_for_compare(s)
    return [t for t in s.split() if t]


def _token_overlap_ratio(user_s: str, post_s: str) -> float:
    utoks = set(_tokenize(user_s))
    ptoks = set(_tokenize(post_s))
    if not utoks:
        return 0.0
    return len(utoks & ptoks) / float(len(utoks))


def _fuzzy_ratio(a: str, b: str) -> float:
    """Return fuzzy token_set_ratio in 0..1 if rapidfuzz is available, else difflib ratio."""
    a_n = _normalize_for_compare(a)
    b_n = _normalize_for_compare(b)
    if not a_n or not b_n:
        return 0.0
    if _HAS_RAPIDFUZZ:
        try:
            return float(fuzz.token_set_ratio(a_n, b_n)) / 100.0
        except Exception:
            pass
    # fallback
    try:
        return difflib.SequenceMatcher(a=a_n, b=b_n).ratio()
    except Exception:
        return 0.0


def _score_attr(user_val: str | None, post_val: str | None) -> float:
    if not user_val or not post_val:
        return 0.0
    u = _normalize_for_compare(str(user_val))
    p = _normalize_for_compare(str(post_val))
    if not u or not p:
        return 0.0
    if u == p:
        return 1.0
    r = _token_overlap_ratio(u, p)
    if r >= 0.75:
        return 0.8
    if r >= 0.4:
        return 0.5
    return 0.0


def _start_year_from_card_row(card_row: dict) -> int | None:
    years = card_row.get("years")
    if isinstance(years, list) and years:
        y0 = years[0]
        if isinstance(y0, int):
            return y0
        if isinstance(y0, str):
            m = re.match(r"(\d{4})", y0)
            return int(m.group(1)) if m else None
    y = card_row.get("year")
    if isinstance(y, int):
        return y
    if isinstance(y, str):
        m = re.match(r"(\d{4})", y)
        return int(m.group(1)) if m else None
    return None


def _grading_equal(
    user_company: str | None, user_grade: any, post_company: str | None, post_grade: any
) -> bool:
    if not user_company or user_grade is None:
        return False
    uc = str(user_company).upper().strip()
    pc = str(post_company).upper().strip() if post_company else ""
    try:
        ug = float(user_grade)
        pg = float(post_grade) if post_grade is not None else None
    except Exception:
        return False
    return bool(uc and uc == pc and pg is not None and abs(ug - pg) < 1e-6)


def _post_field(post: dict, *path, default=None):
    cur = post
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _years_match(
    card_years: list | int | str | None, post_year: int | str | None
) -> bool:
    if post_year is None:
        return False
    try:
        py = int(post_year)
    except Exception:
        return False
    if card_years is None:
        return False
    if isinstance(card_years, list):
        for y in card_years:
            try:
                if int(y) == py:
                    return True
            except Exception:
                continue
        return False
    try:
        return int(card_years) == py
    except Exception:
        return False


def build_promisable_candidates(
    posts: list[dict], card_attrs: dict, limit: int = 20
) -> list[dict]:
    """Filter posts by strictly matching all non-null card attributes.

    Medium-proof fields (must match if provided): name, card_set, year, variation.
    High-proof fields (must match if provided): number, serial_number, grading_company+grade_number.
    """
    result = []
    if not posts:
        return result

    target_name = card_attrs.get("name")
    target_set = card_attrs.get("card_set") or card_attrs.get("set")
    target_years = card_attrs.get("years")
    target_year = card_attrs.get("year")
    target_variation = card_attrs.get("variation")
    target_number = card_attrs.get("number")
    target_serial = card_attrs.get("serial_number")
    target_grade_co = (card_attrs.get("grading_company") or "").upper() or None
    target_grade_num = card_attrs.get("grade_number")

    # unify years to list if present
    if target_years is None and target_year is not None:
        target_years = [target_year]

    for p in posts:
        pa = _post_field(p, "metadata", "normalized_attributes", default={}) or {}
        p_name = pa.get("player_or_character") or _post_field(
            p, "metadata", "normalized_name"
        )
        p_set = pa.get("set")
        p_year = pa.get("year")
        p_number = pa.get("card_number")
        p_serial = pa.get("serial_number")
        p_grade_co = pa.get("grading")
        p_grade_num = pa.get("grade")
        p_variation = pa.get("parallel") or pa.get("rarity")

        # Try parsing title for missing attrs
        if (
            not p_number
            or not p_grade_co
            or not p_grade_num
            or not p_variation
            or not p_set
            or not p_name
        ):
            try:
                parsed = normalize_card_title(p.get("title") or "")
            except Exception:
                parsed = {}
            p_name = p_name or parsed.get("player_or_character")
            p_set = p_set or parsed.get("set")
            p_year = p_year or parsed.get("year")
            p_number = p_number or parsed.get("card_number")
            p_serial = p_serial or parsed.get("serial_number")
            p_grade_co = p_grade_co or parsed.get("grading")
            p_grade_num = p_grade_num or parsed.get("grade")
            p_variation = p_variation or parsed.get("parallel")

        # Medium checks
        proofs = {}
        # name
        if target_name:
            ok = (_token_overlap_ratio(target_name, p_name or "") >= 0.6) or (
                _fuzzy_ratio(target_name, p_name or "") >= 0.8
            )
            if not ok:
                continue
            proofs["name"] = "medium"
        # set
        if target_set:
            ok = (_token_overlap_ratio(target_set, p_set or "") >= 0.6) or (
                _fuzzy_ratio(target_set, p_set or "") >= 0.8
            )
            if not ok:
                continue
            proofs["card_set"] = "medium"
        # year(s)
        if target_years:
            if not _years_match(target_years, p_year):
                continue
            proofs["year"] = "medium"
        # variation
        if target_variation:
            ok = (_token_overlap_ratio(target_variation, p_variation or "") >= 0.6) or (
                _fuzzy_ratio(target_variation, p_variation or "") >= 0.8
            )
            if not ok:
                continue
            proofs["variation"] = "medium"

        # High checks
        if target_number:
            if _normalize_number_token(target_number) != _normalize_number_token(
                p_number or ""
            ):
                continue
            proofs["number"] = "high"
        if target_serial:
            if _normalize_for_compare(target_serial) != _normalize_for_compare(
                p_serial or ""
            ):
                continue
            proofs["serial_number"] = "high"
        if target_grade_co and target_grade_num is not None:
            try:
                ok = (
                    str(target_grade_co).upper() == str(p_grade_co or "").upper()
                ) and (abs(float(target_grade_num) - float(p_grade_num or 0)) < 1e-6)
            except Exception:
                ok = False
            if not ok:
                continue
            proofs["grading_company"] = "high"
            proofs["grade_number"] = "high"

        # If we reach here, all provided attributes matched
        result.append(
            {
                "idx": len(result) + 1,
                "id": p.get("id"),
                "title": p.get("title", ""),
                "sold_price": p.get("sold_price"),
                "sold_currency": p.get("sold_currency"),
                "sold_post_url": p.get("sold_post_url"),
                "sold_date": p.get("sold_date"),
                "attrs": {
                    "name": p_name,
                    "set": p_set,
                    "year": p_year,
                    "number": p_number,
                    "serial_number": p_serial,
                    "grading_company": p_grade_co,
                    "grade_number": p_grade_num,
                    "variation": p_variation,
                },
                "proofs": proofs,
            }
        )

        if len(result) >= limit:
            break

    return result


async def gpt_pick_strict_candidate(card_attrs: dict, candidates: list[dict]) -> dict:
    """Ask GPT-4o-mini (JSON-only) to pick candidates that strictly satisfy all non-null fields, then choose the most recent by sold_date.

    Returns a dict with keys: {matches: [int], primary: int|null, reason: str}
    """
    try:
        target = {k: v for k, v in card_attrs.items() if v not in (None, "", [])}
        payload = {
            "target": target,
            "candidates": [
                {
                    "idx": c.get("idx"),
                    "title": c.get("title"),
                    "sold_date": c.get("sold_date"),
                    "attrs": c.get("attrs"),
                }
                for c in candidates
            ],
        }
        prompt = (
            "You are an expert verifier for trading cards."
            " Given a TARGET card attributes and a list of CANDIDATES, pick those that strictly satisfy ALL non-null fields in TARGET."
            " If multiple satisfy, select the most recent (largest sold_date)."
            ' Respond with ONLY a single JSON object: {"matches":[int],"primary":int|null,"reason":string}.'
            f" TARGET: {json.dumps(payload['target'], ensure_ascii=False)}\n"
            f" CANDIDATES: {json.dumps(payload['candidates'], ensure_ascii=False)}"
        )
        content = await gpt_rerank_prompt(prompt, json_response=True)
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("invalid JSON response")
        return data
    except Exception as e:
        scraper_logger.warning(f"⚠️ GPT strict pick failed: {e}")
        return {"matches": [], "primary": None, "reason": str(e)}


def _derive_user_attrs_from_card_row(card_row: dict) -> dict:
    """Build reliable attribute dict from card_row; fallback to canonical_title parsing when fields are missing."""
    cr = card_row or {}
    cano = cr.get("canonical_title") or ""
    parsed = normalize_card_title(cano) if cano else {}
    # name fields: allow multiple possible keys
    name = (
        cr.get("name")
        or cr.get("player")
        or cr.get("player_name")
        or (parsed.get("player_or_character") if parsed else None)
    )
    card_set = (
        cr.get("card_set")
        or cr.get("set")
        or cr.get("set_name")
        or (parsed.get("set") if parsed else None)
    )
    # year: from years array, then explicit year, then parsed
    year = _start_year_from_card_row(cr)
    if year is None and parsed:
        try:
            year = int(parsed.get("year")) if parsed.get("year") is not None else None
        except Exception:
            year = None
    # number: prefer explicit, else parsed
    number = (
        cr.get("number")
        or cr.get("card_number")
        or (parsed.get("card_number") if parsed else None)
    )
    variation = cr.get("variation") or (parsed.get("parallel") if parsed else None)
    serial_number = cr.get("serial_number") or (
        parsed.get("serial_number") if parsed else None
    )
    grading_company = cr.get("grading_company") or (
        parsed.get("grading") if parsed else None
    )
    grade_number = cr.get("grade_number") or (parsed.get("grade") if parsed else None)

    return {
        "name": name,
        "set": card_set,
        "year": year,
        "number": number,
        "variation": variation,
        "serial_number": serial_number,
        "grading_company": (grading_company.upper() if grading_company else None),
        "grade_number": grade_number,
        "canonical_title": cano,
    }


def select_best_candidate(
    posts: list[dict[str, any]], card_row: dict
) -> dict[str, any]:
    """Deterministic selection using card_row as the truthful source of user_attrs.

    Returns:
      {
        "found": bool,
        "match": {...} | None,
        "debug_info": {
          "top_k_scores": [(title, score, attr_obj)],
          "top_k_ids": [str],
          "top_k_struct": [ {idx,title,price,currency,url,attrs,soft_score} ],
          "total_candidates": int,
          "user_attributes": dict,
          "needs_rerank": bool,
          "top_score": float
        }
      }
    """
    if not posts:
        return {"found": False, "match": None, "debug_info": {"reason": "no posts"}}

    user_attrs = _derive_user_attrs_from_card_row(card_row)

    banned = [
        "lot",
        "bulk",
        "case",
        "box",
        "break",
        "reprint",
        "proxy",
        "digital",
        "cardboard",
    ]
    ct_lower = (user_attrs["canonical_title"] or "").lower()

    candidates: list[tuple[dict, float, dict]] = []
    for p in posts:
        title = p.get("title", "")
        tnorm = title.lower()
        pa = _post_field(p, "metadata", "normalized_attributes", default={}) or {}
        post_name = pa.get("player_or_character") or _post_field(
            p, "metadata", "normalized_name"
        )
        post_set = pa.get("set")
        post_year = pa.get("year")
        post_number = pa.get("card_number")
        post_serial = pa.get("serial_number")
        post_company = pa.get("grading")
        post_grade = pa.get("grade")
        post_variation = pa.get("parallel") or pa.get("rarity")

        # Hard filters (order): name -> set -> year -> number -> banned
        if user_attrs.get("name"):
            if not post_name:
                continue
            # Accept either sufficient token overlap or strong fuzzy ratio (typo tolerant)
            name_overlap = _token_overlap_ratio(user_attrs["name"], post_name)
            name_fuzzy = _fuzzy_ratio(user_attrs["name"], post_name)
            if name_overlap < 0.6 and name_fuzzy < 0.8:
                continue

        if user_attrs.get("set"):
            if not post_set:
                continue
            set_overlap = _token_overlap_ratio(user_attrs["set"], post_set)
            set_fuzzy = _fuzzy_ratio(user_attrs["set"], post_set)
            if set_overlap < 0.6 and set_fuzzy < 0.8:
                continue

        if user_attrs.get("year") is not None:
            if post_year is None or int(post_year) != int(user_attrs["year"]):
                continue

        if user_attrs.get("number"):
            if not post_number:
                continue
            if _normalize_number_token(user_attrs["number"]) != _normalize_number_token(
                post_number
            ):
                continue

        bad = False
        for b in banned:
            if b in tnorm and b not in ct_lower:
                bad = True
                break
        if bad:
            continue

        # Soft scoring
        s_name = _score_attr(user_attrs.get("name"), post_name) * WEIGHT_NAME
        s_set = _score_attr(user_attrs.get("set"), post_set) * WEIGHT_SET
        s_number = (
            1.0
            if (
                user_attrs.get("number")
                and _normalize_number_token(user_attrs.get("number"))
                == _normalize_number_token(post_number)
            )
            else 0.0
        )
        s_year = (
            1.0
            if (
                user_attrs.get("year") is not None
                and post_year is not None
                and int(post_year) == int(user_attrs["year"])
            )
            else 0.0
        )
        s_variation = _score_attr(user_attrs.get("variation"), post_variation)
        s_serial = (
            1.0
            if (
                user_attrs.get("serial_number")
                and post_serial
                and _normalize_for_compare(user_attrs["serial_number"])
                == _normalize_for_compare(post_serial)
            )
            else 0.0
        )
        s_grading = (
            1.0
            if _grading_equal(
                user_attrs.get("grading_company"),
                user_attrs.get("grade_number"),
                post_company,
                post_grade,
            )
            else 0.0
        )

        final_score = (
            s_name
            + s_set
            + s_number * WEIGHT_NUMBER
            + s_year * WEIGHT_YEAR
            + s_variation * WEIGHT_VARIATION
            + s_serial * WEIGHT_SERIAL
            + s_grading * WEIGHT_GRADING
        )

        attr_obj = {
            "name": (s_name / WEIGHT_NAME) if WEIGHT_NAME else 0.0,
            "set": (s_set / WEIGHT_SET) if WEIGHT_SET else 0.0,
            "number": 1.0 if s_number > 0 else 0.0,
            "year": 1.0 if s_year > 0 else 0.0,
            "variation": s_variation,
            "serial": 1.0 if s_serial > 0 else 0.0,
            "grading": 1.0 if s_grading > 0 else 0.0,
        }

        candidates.append((p, final_score, attr_obj))

    if not candidates:
        return {
            "found": False,
            "match": None,
            "debug_info": {
                "reason": "no candidates after hard filters",
                "user_attributes": user_attrs,
            },
        }

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_score = candidates[0][1]

    needs_rerank = False
    if top_score >= SCORE_THRESHOLD and top_score < RERANK_IF_TOP_LT:
        needs_rerank = True
    elif (
        len(candidates) > 1 and abs(top_score - candidates[1][1]) <= RERANK_IF_TIE_DELTA
    ):
        needs_rerank = True

    top_k = candidates[:TOP_K_FOR_RERANK]
    best_post, best_score, best_attrs = top_k[0]
    found = bool(best_score >= SCORE_THRESHOLD)

    debug_info = {
        "top_k_scores": [(p.get("title", ""), sc, a) for p, sc, a in top_k],
        "top_k_ids": [p.get("id") for p, _, _ in top_k],
        "top_k_struct": [
            {
                "idx": i + 1,
                "title": p.get("title", ""),
                "price": p.get("sold_price"),
                "currency": p.get("sold_currency"),
                "url": p.get("sold_post_url"),
                "attrs": {
                    "name": _post_field(
                        p, "metadata", "normalized_attributes", default={}
                    ).get("player_or_character"),
                    "set": _post_field(
                        p, "metadata", "normalized_attributes", default={}
                    ).get("set"),
                    "year": _post_field(
                        p, "metadata", "normalized_attributes", default={}
                    ).get("year"),
                    "number": _post_field(
                        p, "metadata", "normalized_attributes", default={}
                    ).get("card_number"),
                    "variation": (
                        _post_field(
                            p, "metadata", "normalized_attributes", default={}
                        ).get("parallel")
                        or _post_field(
                            p, "metadata", "normalized_attributes", default={}
                        ).get("rarity")
                    ),
                    "serial": _post_field(
                        p, "metadata", "normalized_attributes", default={}
                    ).get("serial_number"),
                    "grading_company": _post_field(
                        p, "metadata", "normalized_attributes", default={}
                    ).get("grading"),
                    "grade_number": _post_field(
                        p, "metadata", "normalized_attributes", default={}
                    ).get("grade"),
                },
                "soft_score": round(float(sc), 4),
            }
            for i, (p, sc, a) in enumerate(top_k)
        ],
        "total_candidates": len(candidates),
        "user_attributes": {
            "name": user_attrs.get("name"),
            "set": user_attrs.get("set"),
            "year": user_attrs.get("year"),
            "number": user_attrs.get("number"),
            "variation": user_attrs.get("variation"),
            "serial_number": user_attrs.get("serial_number"),
            "grading_company": user_attrs.get("grading_company"),
            "grade_number": user_attrs.get("grade_number"),
        },
        "needs_rerank": needs_rerank,
        "top_score": top_score,
    }

    result = {
        "found": found,
        "match": None,
        "debug_info": debug_info,
    }

    if found:
        result["match"] = {
            "title": best_post.get("title", ""),
            "sold_price": best_post.get("sold_price"),
            "sold_currency": best_post.get("sold_currency"),
            "sold_post_url": best_post.get("sold_post_url"),
            "sold_date": best_post.get("sold_date"),
            "sold_at": best_post.get("sold_date"),
            "similarity": best_score,
            "attribute_score": best_attrs,
            "metadata": best_post.get("metadata", {}),
        }

    return result


# ===============================================================
# Attribute Matching
# ===============================================================


def _calculate_attribute_match_score(
    user_attrs: dict[str, any], post_attrs: dict[str, any]
) -> float:
    """Calculate precise attribute match score between user query and post."""
    score = 0.0
    total_weight = 0.0

    # Player/Character matching - 50% weight (HIGHEST PRIORITY - must match player)
    if user_attrs.get("player_or_character") and post_attrs.get("player_or_character"):
        user_player = user_attrs["player_or_character"].lower()
        post_player = post_attrs["player_or_character"].lower()

        # Extract key names (ignoring numbers/grades) - minimum 3 characters
        user_names = [w for w in user_player.split() if w.isalpha() and len(w) >= 3]
        post_names = [w for w in post_player.split() if w.isalpha() and len(w) >= 3]

        if user_names and post_names:
            # Require exact name matches (not substring matches)
            name_matches = sum(1 for name in user_names if name in post_names)

            # Only give score if we have significant name overlap
            if len(user_names) >= 2:  # Full names (first + last)
                # Require both first and last name to match for full score
                if name_matches >= 2:
                    score += 0.5  # Full player name match
                elif name_matches == 1:
                    score += 0.2  # Partial match for one name
                # No score if no names match
            else:  # Single name
                if name_matches > 0:
                    score += 0.5
        total_weight += 0.5

    # Set matching - 25% weight
    if user_attrs.get("set") and post_attrs.get("set"):
        if user_attrs["set"].lower() == post_attrs["set"].lower():
            score += 0.25
        total_weight += 0.25
    elif user_attrs.get("set"):  # User has set but post doesn't
        total_weight += 0.25  # Penalty for missing set

    # Year/Date matching - 15% weight
    user_year = _extract_year(user_attrs.get("normalized_title", ""))
    post_year = _extract_year(post_attrs.get("normalized_title", ""))
    if user_year and post_year:
        if user_year == post_year:
            score += 0.15
        elif abs(int(user_year) - int(post_year)) <= 1:  # Allow 1 year difference
            score += 0.075
        total_weight += 0.15

    # Serial number/Card number matching - 15% weight
    user_serial = _extract_serial_number(user_attrs.get("normalized_title", ""))
    post_serial = _extract_serial_number(post_attrs.get("normalized_title", ""))
    if user_serial and post_serial:
        if user_serial.lower() == post_serial.lower():
            score += 0.15
        elif user_serial.replace("/", "").replace("#", "") == post_serial.replace(
            "/", ""
        ).replace("#", ""):
            score += 0.1  # Partial match for format differences
        total_weight += 0.15

    # Return normalized score
    return score / total_weight if total_weight > 0 else 0.0


# ===============================================================
# Helper Functions
# ===============================================================


def _extract_year(text: str) -> str | None:
    """Extract year from card title, handling season ranges like 2024-25."""
    import re

    # Look for 4-digit years or year ranges like 2024, 2023/24, 2022-23
    # Always return the first (main) year from ranges
    year_match = re.search(r"(20\d{2})(?:[/-]\d{2})?", text)
    return year_match.group(1) if year_match else None


def _extract_year_range(text: str) -> list[str]:
    # extract year pattern at the start, can be 1994 or 1994-96
    m = re.match(r"(\d{4})(?:-(\d{2,4}))?", text)
    if not m:
        return []

    start = int(m.group(1))
    end_str = m.group(2)

    if end_str:
        if len(end_str) == 2:
            end = int(str(start)[:2] + end_str)
        else:
            end = int(end_str)
        return [str(y) for y in range(start, end + 1)]
    else:
        return [str(start)]


def _extract_serial_number(text: str) -> str | None:
    """Extract serial number like /25, #123, 048/299, avoiding year ranges like 2024-25."""
    import re

    # Look for patterns like /25, #123, 048/299, etc.
    # But avoid year ranges like 2024-25, 2023-24
    serial_patterns = [
        r"(?<!20\d{2}[-/])/(\d+)",  # /25 but not after year like 2024-25
        r"#(\d+)",  # #123
        r"(\d{3}/\d+)",  # 048/299
        r"(?<!20\d{2})(\d{1,2}/\d+)",  # 12/50 but not after 4-digit year
    ]

    for pattern in serial_patterns:
        match = re.search(pattern, text)
        if match:
            captured = match.group(1)
            # Additional check: if it looks like a year range (specifically XX-YY or XX/YY where XX is a 4-digit year), skip it
            # But don't skip legitimate serial numbers like /25, #25 etc.
            year_range_context = re.search(
                r"(20\d{2})[-/]" + re.escape(captured) + r"\b", text
            )
            if year_range_context:
                continue
            return captured

    return None


# ===============================================================
# Step 4: upsert ebay_posts (supabase) + update cards if card_id mode
# ===============================================================


async def upsert_ebay_listings(
    posts: list[dict[str, any]], card_id: str | None = None
) -> dict[str, any]:
    """
    Upsert posts into supabase table `ebay_posts`. If embedding present, store it in embedding column.
    """
    results = {"upserted_count": 0, "errors": []}
    if not posts:
        return results
    try:
        db_records = []
        for post in posts:
            # generate deterministic-ish id if not present
            db_id = post.get("title", "")[:50] + "_" + str(random.randint(1000, 9999))
            db_record = {
                "id": db_id,
                "title": post.get("title", ""),
                "image_url": post.get("image_url", ""),
                "image_hd_url": post.get("image_hd_url", ""),
                "region": post.get("metadata", {}).get("region", ""),
                "sold_at": post.get("sold_date", ""),
                "currency": post.get("sold_currency", "USD"),
                "price": post.get("sold_price", ""),
                "condition": post.get("condition", "raw"),
                "grading_company": post.get("grading_company"),
                "grade": post.get("grade"),
                "post_url": post.get("sold_post_url", ""),
                "blocked": False,
                "user_card_ids": [],
                "normalised_name": post.get("normalized_title", ""),
                "category_id": post.get("metadata", {}).get("category_id"),
            }
            db_records.append(db_record)
        # upsert in batches with robust fallback for missing columns
        BATCH = 500
        for i in range(0, len(db_records), BATCH):
            batch = db_records[i : i + BATCH]

            def _upsert_safe(recs: list[dict[str, any]]):
                while True:
                    try:
                        supabase.table("ebay_posts").upsert(
                            recs, on_conflict="id", returning="minimal"
                        ).execute()
                        return True
                    except Exception as e:
                        msg = str(e)
                        # Example: Could not find the 'normalized_attributes' column of 'ebay_posts'
                        m = re.search(r"Could not find the '([^']+)' column", msg)
                        if m:
                            missing_col = m.group(1)
                            for r in recs:
                                r.pop(missing_col, None)
                            scraper_logger.warning(
                                f"⚠️ Upsert retry: removed missing column '{missing_col}' from batch"
                            )
                            continue
                        # Some Supabase clients return JSON dict-like errors; attempt to parse column name
                        if "normalized_attributes" in msg:
                            for r in recs:
                                r.pop("normalized_attributes", None)
                            scraper_logger.warning(
                                "⚠️ Upsert retry: removed 'normalized_attributes' from batch"
                            )
                            continue
                        if "embedding" in msg:
                            for r in recs:
                                r.pop("embedding", None)
                            scraper_logger.warning(
                                "⚠️ Upsert retry: removed 'embedding' from batch"
                            )
                            continue
                        # If we cannot recover, re-raise
                        raise

            _upsert_safe(batch)
            results["upserted_count"] += len(batch)
        scraper_logger.info(
            f"✅ Upserted {results['upserted_count']} listings to ebay_posts"
        )
    except Exception as e:
        results["errors"].append(str(e))
        scraper_logger.error(f"❌ Failed to upsert listings: {e}")
    return results


# ===============================================================
# Update user card with selected post
# ===============================================================


async def update_card(
    selected_post: dict[str, any], card_id: str, debug_info: dict | None = None
) -> dict[str, any]:
    results = {"card_updated": False, "errors": []}
    try:
        update_data = {
            "last_sold_post_url": selected_post.get("sold_post_url"),
            "last_sold_price": selected_post.get("sold_price"),
            "last_sold_currency": selected_post.get("sold_currency"),
            "last_sold_at": selected_post.get("sold_date"),
        }
        if debug_info:
            try:
                trimmed = {
                    "top_k_scores": (debug_info.get("top_k_scores") or [])[:5],
                    "gpt_choice": debug_info.get("gpt_choice"),
                    "gpt_rerank_failed": debug_info.get("gpt_rerank_failed", False),
                    "needs_rerank": debug_info.get("needs_rerank", False),
                }
                update_data["last_sold_debug_info"] = trimmed
            except Exception:
                pass
        supabase.table("cards").update(update_data).eq("id", card_id).execute()
        results["card_updated"] = True
        scraper_logger.info(
            f"✅ Updated card {card_id}: price={selected_post.get('sold_price')} {selected_post.get('sold_currency')}"
        )
    except Exception as e:
        results["errors"].append(str(e))
        scraper_logger.error(f"❌ Failed update cards: {e}")
    return results


# ===============================================================
# Upsert card_sets (supabase)
# ===============================================================
# NOTE: this module expects `supabase` client to be available (same as project).
# Upsert functions now return a dict with explicit 'count' and 'records'.
async def upsert_card_sets(card_sets: list[dict[str, any]]) -> dict[str, any]:
    """
    Upsert card_sets into supabase table `card_sets`.
    Return:
      {
        "count": int,            # number of records returned by supabase (accumulated)
        "records": List[Dict],   # list of returned representations
        "errors": List[str]      # errors, if any
      }
    """
    results = {"count": 0, "records": [], "errors": []}
    if not card_sets:
        return results

    try:
        db_records = []
        for card_set in card_sets:
            db_record = {
                "name": card_set.get("name", ""),
                "category_id": card_set.get("category_id", ""),
                "platform": card_set.get("platform", ""),
                "platform_set_id": card_set.get("platform_set_id", ""),
                "years": card_set.get("years", []),
                "link": card_set.get("link", ""),
                "browse_type": card_set.get("browse_type", ""),
            }
            db_records.append(db_record)

        # upsert in batches, accumulate returned representations
        BATCH = 500
        for i in range(0, len(db_records), BATCH):
            batch = db_records[i : i + BATCH]
            resp = (
                supabase.table("card_sets")
                .upsert(
                    batch, on_conflict="platform_set_id", returning="representation"
                )
                .execute()
            )
            batch_records = resp.data or []
            results["records"].extend(batch_records)
            results["count"] += len(batch_records)

        scraper_logger.info(f"✅ Upserted {results['count']} sets to card_sets")
        return results
    except Exception as e:
        results["errors"].append(str(e))
        scraper_logger.error(f"❌ Failed to upsert card_sets: {e}")
        return results


# ===============================================================
# Upsert master_cards (supabase)
# ===============================================================
async def upsert_master_cards(cards: list[dict[str, any]]) -> dict[str, any]:
    """
    Upsert cards into supabase table `master_cards`.
    Return structure same as upsert_card_sets: {count, records, errors}
    """
    results = {"count": 0, "records": [], "errors": []}
    if not cards:
        return results

    try:
        db_records = []
        for card in cards:
            years = card.get("years", []) or []
            db_record = {
                "category_id": card.get("category_id", ""),
                "set_id": card.get("set_id", ""),
                "platform": card.get("platform", ""),
                "platform_card_id": card.get("platform_card_id", ""),
                "name": card.get("name", ""),
                "card_number": card.get("card_number", ""),
                "canonical_image_url": card.get("canonical_image_url", ""),
                "link": card.get("link", ""),
                "attributes": card.get("attributes", {}),
                "years": years,
                "year": years[0] if years else None,
            }
            db_records.append(db_record)

        # upsert in batches, accumulate returned representations
        BATCH = 500
        for i in range(0, len(db_records), BATCH):
            batch = db_records[i : i + BATCH]
            resp = (
                supabase.table("master_cards")
                .upsert(
                    batch, on_conflict="platform_card_id", returning="representation"
                )
                .execute()
            )
            batch_records = resp.data or []
            results["records"].extend(batch_records)
            results["count"] += len(batch_records)

        scraper_logger.info(f"✅ Upserted {results['count']} cards to master_cards")
        return results
    except Exception as e:
        results["errors"].append(str(e))
        scraper_logger.error(f"❌ Failed to upsert master_cards: {e}")
        return results


async def get_card(card_id: str) -> Optional[dict]:
    try:
        resp = (
            supabase.table("cards")
            .select(
                "canonical_title, years, card_set, name, variation, serial_number, number, grading_company, grade_number"
            )
            .eq("id", card_id)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]
    except Exception as e:
        scraper_logger.info(f"⚠️ Failed to resolve card: {e}")
    return None


def normalized_text(val: Any) -> Optional[str]:
    if not val or val == "":
        return None
    return str(val).strip().lower()


def word_match(
    title: Optional[str],
    filter: Optional[str],
    word_threshold: Optional[int] = None,
) -> bool:
    """Return True if title matches filter using token and fuzzy checks with count-based threshold.

    - title: text to search within (can be None; treated as empty string)
    - filter: space-delimited string of tokens to check (None/empty => False)
    - word_threshold: minimum number of tokens/phrases that must match (>= 1). If None or > token count, require all tokens.

    Special rules:
    - Tokens are split on whitespace and evaluated independently (even single-character tokens like "1").
    - Use RapidFuzz per token as a flexible fallback (no static alias expansions).
    """
    if not filter:
        return True
    # Normalize title
    title_text = (title or "").lower().strip()
    filter_text = str(filter).lower().strip()
    # Build collapsed variants to catch cases like 'logo fractor' vs 'logofractor' or 'logo-fractor'
    title_collapsed = title_text.replace(" ", "")
    filter_collapsed = filter_text.replace(" ", "")
    # Alphanumeric-only squeeze (remove all non a-z0-9)
    title_squeezed = re.sub(r"[^a-z0-9]+", "", title_text)
    filter_squeezed = re.sub(r"[^a-z0-9]+", "", filter_text)
    # Tokenize the filter on whitespace; do not merge short tokens
    tokens = [t for t in filter_text.split() if t]
    if not tokens:
        return False

    token_count = len(tokens)
    # Compute required matches based on word_threshold (default: require all)
    required = (
        token_count
        if not word_threshold or word_threshold <= 0
        else min(word_threshold, token_count)
    )

    # 1) If requiring all tokens, allow an early perfect whole-string equality
    if required == token_count:
        if fuzz.token_set_ratio(title_text, filter_text) >= 100:
            return True
        if fuzz.token_set_ratio(title_collapsed, filter_collapsed) >= 100:
            return True
        if fuzz.token_set_ratio(title_squeezed, filter_squeezed) >= 100:
            return True

    # 2) Count token coverage (exact or fuzzy) for tokens
    matched = 0
    fuzzy_token_min_ratio = 80  # fixed per-token fuzzy similarity threshold
    for t in tokens:
        tc = t.replace(" ", "")
        ts = re.sub(r"[^a-z0-9]+", "", t)
        if t in title_text or tc in title_collapsed or ts in title_squeezed:
            matched += 1
            continue
        # last resort fuzzy per-token when available
        if (
            fuzz.partial_ratio(t, title_text) >= fuzzy_token_min_ratio
            or fuzz.partial_ratio(tc, title_collapsed) >= fuzzy_token_min_ratio
            or fuzz.partial_ratio(ts, title_squeezed) >= fuzzy_token_min_ratio
        ):
            matched += 1

    return matched >= required


def all_present_matched(
    attribute_filters: List[str], req: List[str], matched: Dict[str, bool]
) -> bool:
    filtered_req = [k for k in req if k in attribute_filters]
    return all(matched.get(k, False) for k in filtered_req)


def build_strict_promisable_candidates(
    attribute_filters: Dict[str, Any],
    posts: List[Dict],
) -> Dict[str, List[Dict]]:
    low_req = ["years", "name"]
    med_req = ["years", "card_set", "name"]
    high_req = ["years", "card_set", "name", "variation"]

    # -----------------------------------------------------------
    # Normalize card-side values once
    # -----------------------------------------------------------
    unmatch_posts: List[Dict] = []
    match_posts: List[Dict] = []
    # Debug counters
    low_cnt = med_cnt = high_cnt = unmatch_cnt = fallback_cnt = 0

    # Log incoming filters once for debugging
    try:
        dbg_filters = {
            k: v
            for k, v in attribute_filters.items()
            if k
            in ["years", "card_set", "name", "variation", "serial_number", "number"]
        }
        scraper_logger.info(
            f"[build_candidates] filters: {json.dumps(dbg_filters, ensure_ascii=False)}"
        )
    except Exception:
        pass

    card_years = normalized_text(attribute_filters.get("years", ""))
    card_set = normalized_text(attribute_filters.get("card_set", ""))
    card_name = normalized_text(attribute_filters.get("name", ""))
    card_variation = normalized_text(attribute_filters.get("variation", ""))
    card_serial = normalized_text(attribute_filters.get("serial_number", ""))
    card_number = normalized_text(attribute_filters.get("number", ""))
    # card_grading = normalized_text(attribute_filters.get("grading_company", ""))
    # card_grade = normalized_text(attribute_filters.get("grade_number", ""))

    # print(f"card_years: {card_years}")
    # print(f"card_set: {card_set}")
    # print(f"card_name: {card_name}")
    # print(f"card_variation: {card_variation}")
    # print(f"card_serial: {card_serial}")
    # print(f"card_number: {card_number}")
    # print(f"card_grading: {card_grading}")
    # print(f"card_grade: {card_grade}")

    for ebay_post in posts:
        # -----------------------------------------------------------
        # Post normalization helpers
        # -----------------------------------------------------------
        title = ebay_post.get("normalized_title")
        matched = {
            # Years: match any of the provided years (threshold=1)
            "years": word_match(title, card_years, 1),
            "card_set": word_match(title, card_set, 2),
            "name": word_match(title, card_name),
            "variation": word_match(title, card_variation, 0),
            "serial_number": word_match(title, card_serial),
            "number": word_match(title, card_number),
            # "grading_company": word_match(title, card_grading),
            # "grade_number": word_match(title, card_grade),
        }

        if all_present_matched(attribute_filters, low_req, matched):
            matched["candidate_score"] = "low"
            low_cnt += 1
        elif all_present_matched(attribute_filters, med_req, matched):
            matched["candidate_score"] = "medium"
            med_cnt += 1
        elif all_present_matched(attribute_filters, high_req, matched):
            matched["candidate_score"] = "high"
            high_cnt += 1
        else:
            matched["candidate_score"] = "unmatch"
            unmatch_cnt += 1

        ebay_post["matched"] = matched

        if matched["candidate_score"] == "unmatch":
            # Fallback: allow GPT to refine when set+name look good, even if year is missing
            if matched.get("card_set") and matched.get("name"):
                matched["candidate_score"] = "fallback"
                match_posts.append(ebay_post)
                fallback_cnt += 1
            else:
                unmatch_posts.append(ebay_post)
        else:
            match_posts.append(ebay_post)

    # Log summary counts and up to 5 sample unmatches for diagnosis
    try:
        scraper_logger.info(
            f"[build_candidates] counts -> low={low_cnt}, med={med_cnt}, high={high_cnt}, fallback={fallback_cnt}, unmatch={unmatch_cnt}"
        )
        if unmatch_cnt and len(unmatch_posts) > 0:
            sample = []
            for p in unmatch_posts[:5]:
                sample.append(
                    {
                        "title": p.get("normalized_title") or p.get("title"),
                        "matched": p.get("matched"),
                    }
                )
            # scraper_logger.info(
            #     f"[build_candidates] unmatch_sample: {json.dumps(sample, ensure_ascii=False)}"
            # )
    except Exception:
        pass

    return {"unmatch_posts": unmatch_posts, "match_posts": match_posts}


# ===============================================================
# GPT selection (Step 4 for card_id mode)
# ===============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com")
GPT_MODEL = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")


async def select_best_candidate_with_gpt(
    canonical_title: Optional[str],
    match_posts: List[Dict],
) -> Optional[Dict]:
    """Return the single best candidate (full post) chosen by GPT and finalized by most recent sold_date.

    Flow:
    - Build a compact list from ALL `match_posts` for GPT (index + id + titles).
    - GPT returns indices of likely matches.
    - Map indices back to `match_posts` to obtain the candidate list.
    - Print the compact entries for the selected indices for debugging.
    - Choose the most recent by sold_date and return that single post.
    """
    if len(match_posts) == 0:
        scraper_logger.info("❌ No match posts found; selection returns None")
        return None

    compact: List[Dict] = []
    for idx, match_post in enumerate(match_posts):
        compact.append(
            {
                "i": idx,  # index used for selection
                "id": match_post.get("id"),
                "raw": str(match_post.get("title")),
                "normalized": str(match_post.get("normalized_title")),
            }
        )

    scraper_logger.info(f"[compact count] --- {len(compact)}")
    scraper_logger.info("--------------------------------------")

    system_prompt = (
        "You are an EXPERT trading-card matcher. Decide if each candidate describes the EXACT same card as the canonical title. Rules: "
        "1) Mandatory: YEAR and PLAYER/CHARACTER NAME must match (allow obvious synonyms/spelling/case/hyphen/space variants). "
        "2) SET/BRAND/SERIES (card_set) must match conceptually (same product line/family). "
        "3) VARIATION/PARALLEL/SUBSET handling: If the canonical specifies a variation (e.g., 'Gold Logofractor /50'), then: "
        "   • If a candidate explicitly states a DIFFERENT variation (e.g., 'Sapphire', 'Speckle', 'Auto/Autograph'), EXCLUDE it. "
        "   • If a candidate omits variation tokens (no explicit variant) but YEAR, NAME, and SET match, ALLOW it. "
        "4) Bonus: SERIAL NUMBER and CARD NUMBER help but are not required. Prefer matches that also align on these when present. "
        "5) Super bonus: GRADING COMPANY and GRADE NUMBER are informative but never required; never exclude solely due to grading differences. "
        "Normalize case/hyphens/spaces. Allowed synonyms: 'f1' == 'formula 1' == 'formula one'; 'logofractor' == 'logo fractor'. "
        "Set equivalence: 'Topps Chrome F1' == 'Topps Chrome Formula 1'. Different product line: 'Topps Turbo Attax' ≠ 'Topps Chrome F1'. "
        "Explicit different variation keywords to EXCLUDE when canonical specifies a different one: 'sapphire', 'speckle', 'speckle refractor', 'auto', 'autograph', 'refractor', 'xfractor', 'x-fractor', 'prism', 'prizm'. "
        "Examples: Canonical '2024 Topps Chrome F1 Kimi Antonelli /50 Gold Logofractor' → INCLUDE '2024 Topps Chrome F1 #94 Andrea Kimi Antonelli' (no variation stated); EXCLUDE '2024 Topps Chrome Sapphire F1 ...', '... Auto ...', '... Speckle ...', 'Topps Turbo Attax ...'. "
        "Output a JSON object with keys: 'matches' (array of indices) and 'reason' (string). Include only indices with ≥96% confidence; if none qualify, return an empty list and explain why in 'reason'."
    )

    user_prompt = {
        "canonical_title": canonical_title or "",
        "candidates": compact,
        "instructions": (
            "Compare 'canonical_title' with each candidate using both 'raw' and 'normalized' titles. "
            "Include an index ONLY IF: YEAR matches and PLAYER/CHARACTER NAME matches. SET/BRAND/SERIES must also match conceptually (e.g., Chrome F1 == Chrome Formula 1; Turbo Attax is a different set). "
            "If the canonical specifies a variation, exclude candidates that explicitly state a different variation (e.g., 'Sapphire', 'Speckle', 'Auto'). "
            "If a candidate omits variation tokens but YEAR, NAME, and SET match, include it. "
            "CARD NUMBER and SERIAL NUMBER are bonuses (prefer but do not require). GRADING COMPANY and GRADE NUMBER are super bonuses (never require). "
            "Return JSON: {\"matches\": [i1, i2, ...], \"reason\": <string>}. Include only indices with ≥96% confidence; if none qualify, return an empty list and set 'reason' to a concise explanation (e.g., 'year mismatch', 'name mismatch', 'set mismatch', or 'explicit different variation')."
        ),
    }

    try:
        url = OPENAI_BASE.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GPT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        idxs = parsed.get("matches", [])
        reason = parsed.get("reason")
        # if not isinstance(idxs, list):
        #     try:
        #         scraper_logger.info("gpt_reason:", reason or "invalid_response_format")
        #     except Exception:
        #         scraper_logger.info("gpt_reason:", reason or "invalid_response_format")
        #     return None

        selected_indices: List[int] = [
            i for i in idxs if isinstance(i, int) and 0 <= i < len(match_posts)
        ]
        # if not selected_indices:
        #     try:
        #         scraper_logger.info(f"gpt_reason: {reason or 'no_matches'}")
        #     except Exception:
        #         scraper_logger.info(f"gpt_reason: {reason or 'no_matches'}")
        #     return None

        selected_posts: List[Dict] = [match_posts[i] for i in selected_indices]
        selected_compact: List[Dict] = [compact[i] for i in selected_indices]

        def sold_ts(p: Dict) -> float:
            d = p.get("sold_date") or p.get("sold_at") or ""
            try:
                return date_parser.parse(d, fuzzy=True).timestamp()
            except Exception:
                return 0.0

        scraper_logger.info(f"[GPT reason] --- {reason}")
        scraper_logger.info("--------------------------------------")
        scraper_logger.info(f"[compact candidates] --- {selected_compact}")
        scraper_logger.info("--------------------------------------")
        scraper_logger.info(f"[candidates] --- {selected_posts}")

        most_recent_post = max(selected_posts, key=sold_ts)
        return most_recent_post
    except Exception as e:
        scraper_logger.info(f"⚠️ GPT selection failed: {e}")
        try:
            scraper_logger.info(f"gpt_error: {str(e)}")
        except Exception:
            pass
        return None

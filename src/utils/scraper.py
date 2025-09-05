import os
import asyncio
import math
import random
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlencode
import json
import unicodedata
import difflib

import dateutil.parser as date_parser
import pytz
from bs4 import BeautifulSoup
import httpx

from src.utils.logger import get_logger
from src.models.category import CategoryId
from src.models.ebay import Region, base_target_url
from src.utils.httpx import httpx_get_content
from src.utils.supabase import supabase

scraper_logger = get_logger("scraper")

# Try to import rapidfuzz (optional). If available, use it for better fuzzy matching.
try:
    from rapidfuzz import process, fuzz  # type: ignore

    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Starter lookups (extend externally if perlu)
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

# Regex patterns
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_CARD_NUM_RE = re.compile(r"#\s*([0-9]{1,4}[A-Za-z]?)", re.IGNORECASE)
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
    # replace weird punctuation with spaces except slash
    s = re.sub(r"[^a-z0-9/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ===============================================================
# Fuzzy matching helper
# ===============================================================


def _fuzzy_best_match(
    candidate: str, choices: List[str]
) -> Tuple[Optional[str], float]:
    if not candidate or not choices:
        return None, 0.0
    if _HAS_RAPIDFUZZ:
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
    title: str, set_names: Optional[List[str]] = None, fuzzy_threshold: float = 75.0
) -> Dict[str, Any]:
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
        if best_name and score >= 50.0:  # Lowered from 75% to 50%
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


# ===============================================================
# Helper: compute cosine similarity
# ===============================================================


def _cosine_sim(a: List[float], b: List[float]) -> float:
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
    path: str, payload: Dict[str, Any], timeout: int = 30
) -> Dict[str, Any]:
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


async def embed_texts(texts: List[str], model: str = None) -> List[List[float]]:
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
    prompt: str, model: str = None, max_tokens: int = 512
) -> str:
    model = model or RERANK_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    data = await _openai_post("/v1/chat/completions", payload, timeout=60)
    return data["choices"][0]["message"]["content"]


# ===============================================================
# Get HTML Content
# ===============================================================


async def scrape_ebay_html(
    region: Region,
    query: Optional[str] = None,
    category_id: Optional[CategoryId] = None,
    user_card_id: Optional[str] = None,
    max_pages: int = 50,
    page_retries: int = 3,
    disable_proxy: bool = False,
) -> List[str]:
    # (same logic as before) - build config, request first page to determine total pages, loop pages
    final_query = query or "card"
    final_category_id = category_id

    if user_card_id:
        try:
            response = (
                supabase.table("user_cards")
                .select("custom_name, category_id")
                .eq("id", user_card_id)
                .limit(1)
                .execute()
            )
            if response.data:
                card_data = response.data[0]
                final_query = card_data.get("custom_name") or "card"
                final_category_id = (
                    CategoryId(card_data.get("category_id"))
                    if card_data.get("category_id")
                    else None
                )
                scraper_logger.info(
                    f"✅ Mode 2: Using user_card_id - query: '{final_query}'"
                )
        except Exception as e:
            scraper_logger.warning(
                f"⚠️ Error resolving user_card_id {user_card_id}: {e}"
            )

    if query and category_id:
        scraper_logger.info(f"✅ Mode 1: Using direct query: '{query}'")

    config = {
        "query": final_query,
        "category_id": final_category_id.value if final_category_id else None,
        "region_str": region.value,
        "base_url": f"{base_target_url[region.value]}/sch/i.html",
    }

    total_pages = 1
    try:
        params = {
            "_nkw": config["query"],
            "_sacat": "0",
            "_from": "R40",
            "rt": "nc",
            "LH_Sold": "1",
            "LH_Complete": "1",
            "Country/Region of Manufacture": (
                "United Kingdom" if config["region_str"] == "uk" else "United States"
            ),
            "_ipg": "240",
            "_pgn": "1",
            "_sop": "12",
        }
        scraper_logger.info(
            f"📡 Scraping eBay URL: {config['base_url']}?{urlencode(params)}"
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
    scraper_logger.info(f"🚀 Scraping {total_pages} pages for query: '{final_query}'")

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
                    "_sop": "13",
                }
                timeout = random.uniform(12.0, 18.0)
                jitter_min = random.uniform(0.5, 1.2)
                jitter_max = random.uniform(1.5, 3.0)
                page_html = await httpx_get_content(
                    config["base_url"],
                    params=params,
                    attempts=5,
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
    html_pages: List[str], region: str, category_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    all_posts: List[Dict[str, Any]] = []
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

                post_data: Dict[str, Any] = {
                    "id": listing_id,
                    "title": title,
                    "image_url": image_url,
                    "image_hd_url": image_hd_url,
                    # top-level compatibility
                    "condition": condition,
                    "grading_company": grading_company,
                    "grade": grade,
                    "normalised_name": normalized_name,
                    # nested metadata
                    "metadata": {
                        "condition": condition,
                        "grading_company": grading_company,
                        "grade": grade,
                        "region": region,
                        "category_id": category_id,
                        "normalized_name": normalized_name,
                        "tags": tags,
                        "normalized_attributes": norm_attrs,
                    },
                    "sold_date": sold_date,
                    "sold_price": price,
                    "sold_currency": currency,
                    "sold_post_url": url,
                    "embedding": None,  # Will be filled in Step 2
                }

                all_posts.append(post_data)

            except Exception as e:
                scraper_logger.warning(f"⚠️ Failed to extract item data: {e}")
                continue

    # dedupe by url
    seen_urls = set()
    unique_posts: List[Dict[str, Any]] = []
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
    posts: List[Dict[str, Any]], batch_size: int = EMBED_BATCH
) -> None:
    if not posts:
        return

    model = EMBED_MODEL
    texts = [p["metadata"]["normalized_attributes"]["normalized_title"] for p in posts]
    embeddings: List[List[float]] = []
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
# Step 3: NN search (top-k) + optional GPT rerank
# ===============================================================


async def select_best_ebay_post(
    posts: List[Dict[str, Any]],
    user_query: str,
    top_k: int = 5,
    rerank_with_gpt: bool = True,
) -> Optional[Dict[str, Any]]:
    """Select best eBay post using precise attribute matching + embedding similarity."""
    if not posts:
        return None

    scraper_logger.info(f"🎯 Starting selection from {len(posts)} posts...")

    # Limit posts to prevent timeout (take first 100 for efficiency)
    if len(posts) > 100:
        scraper_logger.info(f"⚡ Limiting to first 100 posts for efficiency")
        posts = posts[:100]

    # Normalize user query to extract attributes
    user_attrs = normalize_card_title(user_query)
    # scraper_logger.info(f"🔍 Extracted user attributes: {user_attrs}")

    # Score posts using attribute matching + embedding similarity
    scored = []
    for i, post in enumerate(posts):
        post_attrs = post.get("metadata", {}).get("normalized_attributes", {})

        # Calculate attribute match score (0-1)
        attr_score = _calculate_attribute_match_score(user_attrs, post_attrs)

        # Calculate embedding similarity (0-1)
        embed_score = 0.0
        if post.get("embedding"):
            q_norm = user_attrs.get("normalized_title", user_query)
            q_emb = (await embed_texts([q_norm]))[0]
            embed_score = _cosine_sim(q_emb, post["embedding"])

        # Combined score: 80% attribute matching + 20% embedding similarity
        final_score = (attr_score * 0.8) + (embed_score * 0.2)

        scored.append((post, final_score, attr_score, embed_score))

    if not scored:
        return None

    # Sort by combined score and take top-k
    scored.sort(key=lambda x: x[1], reverse=True)
    top_posts = scored[:top_k]

    # Optional GPT reranking for close matches
    best_post, best_score, best_attr, best_embed = top_posts[0]
    if rerank_with_gpt and len(top_posts) > 1 and best_score < 0.9:
        candidates_text = "\n".join(
            [
                f"{i+1}. {post['title']} - {post['sold_price']} {post['sold_currency']} (attr:{attr:.2f}, embed:{embed:.2f})"
                for i, (post, _, attr, embed) in enumerate(top_posts)
            ]
        )

        prompt = f"""
        User is looking for: {user_query}
        
        Top candidates with scores:
        {candidates_text}
        
        Return the number (1-{len(top_posts)}) of the best match as JSON: {{"choice": N}}
        """

        try:
            gpt_out = await gpt_rerank_prompt(prompt)
            if isinstance(gpt_out, dict) and "choice" in gpt_out:
                choice_idx = int(gpt_out["choice"]) - 1
                if 0 <= choice_idx < len(top_posts):
                    best_post, best_score, best_attr, best_embed = top_posts[choice_idx]
        except:
            scraper_logger.warning(
                "⚠️ GPT rerank failed, using attribute-based selection"
            )

    return {
        "found": True,
        "match": {
            "title": best_post.get("title", ""),
            "sold_price": best_post.get("sold_price", 0),
            "sold_currency": best_post.get("sold_currency", "USD"),
            "sold_post_url": best_post.get("sold_post_url", ""),
            "sold_at": best_post.get("sold_at", ""),
            "similarity": best_score,
            "attribute_score": best_attr,
            "embedding_score": best_embed,
            "metadata": best_post.get("metadata", {}),
        },
        "debug_info": {
            "top_k_scores": [
                (p["title"], score, attr, embed) for p, score, attr, embed in top_posts
            ],
            "total_candidates": len(scored),
            "user_attributes": user_attrs,
        },
    }


def _calculate_attribute_match_score(
    user_attrs: Dict[str, Any], post_attrs: Dict[str, Any]
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


def _extract_year(text: str) -> Optional[str]:
    """Extract year from card title, handling season ranges like 2024-25."""
    import re

    # Look for 4-digit years or year ranges like 2024, 2023/24, 2022-23
    # Always return the first (main) year from ranges
    year_match = re.search(r"(20\d{2})(?:[/-]\d{2})?", text)
    return year_match.group(1) if year_match else None


def _extract_serial_number(text: str) -> Optional[str]:
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


# ===============================================================
# Step 4: store (supabase) + update user_cards if needed
# ===============================================================


async def upsert_ebay_listings(
    posts: List[Dict[str, Any]], user_card_id: Optional[str] = None
) -> Dict[str, Any]:
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
                "name": post.get("title", ""),
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
                "normalised_name": post.get("normalised_name", ""),
                "category_id": post.get("metadata", {}).get("category_id"),
                # embed vector into record (Supabase/Postgres vector column should accept array)
                "embedding": post.get("embedding"),
                # store structured attributes
                "normalized_attributes": post.get("metadata", {}).get(
                    "normalized_attributes", {}
                ),
            }
            db_records.append(db_record)
        # upsert in batches
        BATCH = 500
        for i in range(0, len(db_records), BATCH):
            batch = db_records[i : i + BATCH]
            supabase.table("ebay_posts").upsert(
                batch, on_conflict="id", returning="minimal"
            ).execute()
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


async def update_user_card(
    selected_post: Dict[str, Any], user_card_id: str
) -> Dict[str, Any]:
    results = {"user_card_updated": False, "errors": []}
    try:
        update_data = {
            "last_sold_price": selected_post.get("sold_price"),
            "last_sold_currency": selected_post.get("sold_currency"),
            "last_sold_at": selected_post.get("sold_date"),
            "last_sold_post_url": selected_post.get("sold_post_url"),
        }
        supabase.table("user_cards").update(update_data).eq(
            "id", user_card_id
        ).execute()
        results["user_card_updated"] = True
        scraper_logger.info(
            f"✅ Updated user_card {user_card_id}: price={selected_post.get('sold_price')} {selected_post.get('sold_currency')}"
        )
    except Exception as e:
        results["errors"].append(str(e))
        scraper_logger.error(f"❌ Failed update user_cards: {e}")
    return results

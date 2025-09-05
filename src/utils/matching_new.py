# matcher_updated.py
import os
import re
import unicodedata
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

# optional OpenAI usage (gpt selector / embeddings) - only used if key present
try:
    import openai
except Exception:
    openai = None

import numpy as np

# ---------------------------
# Improved cleaner & extractor
# ---------------------------
def clean_text_for_tokens(s: str) -> str:
    """
    Normalize unicode, remove emoji, keep / # - . : because they are meaningful
    for card numbers and serials. Collapse whitespace.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # remove emoji / non-BMP
    s = re.sub(r"[\U00010000-\U0010ffff]", " ", s)
    # remove quotes and some punctuation but keep slash, hash, dash, dot, colon
    s = re.sub(r"[,_·•\(\)\[\]:;\"`']", " ", s)
    # remove other weird chars but preserve / # - . :
    s = re.sub(r"[^\w\s\/#\-\.\:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_numbers_and_serials(text: str) -> Dict[str, Optional[str]]:
    """
    Robust extractor:
      - card_number: '238/191' or '12/25' (numerator/denominator)
      - serial_full: e.g. 'SV08/25' or 'SVO8/25' (code + run)
      - serial_code: e.g. 'SV08', 'CR7' (code without run)
      - serial_run: denominator-only like '25' if found as '/25' token
    It returns last/most-likely occurrences (prefer trailing tokens).
    """
    out = {"card_number": None, "serial_full": None, "serial_code": None, "serial_run": None}
    if not text:
        return out
    s = text

    # 1) Find tokens like "#12/25" or "12/25" or "SV08/25"
    for m in re.finditer(r"(?<!\S)(#?\s*[A-Za-z0-9\-]{1,10})\s*/\s*(\d{1,4})(?!\S)", s):
        left_raw = m.group(1).replace(" ", "")
        denom = m.group(2)
        left = left_raw.lstrip("#")
        # if left is pure digits => card number numerator/denominator
        if re.fullmatch(r"\d{1,4}", left):
            out["card_number"] = f"{left}/{denom}"
        else:
            # left contains letters or letters+digits -> treat as serial_full OR serial_code+run
            out["serial_full"] = f"{left}/{denom}"
            code_match = re.match(r"([A-Za-z0-9\-]+)", left)
            if code_match:
                out["serial_code"] = code_match.group(1).upper()
            out["serial_run"] = denom
        # continue looping to keep last occurrence (prefer trailing)

    # 2) Denominator-only like '/25' (standalone)
    if not out["card_number"] and not out["serial_full"]:
        m_denom = re.search(r"(?<!\S)/\s*(\d{1,4})(?!\S)", s)
        if m_denom:
            out["serial_run"] = m_denom.group(1)

    # 3) Orphan serial codes (trailing tokens like 'SV08' or 'CR7')
    if not out["serial_code"]:
        tokens = re.findall(r"[A-Za-z0-9\-]{2,12}", s)
        for t in reversed(tokens):
            if re.fullmatch(r"\d{1,4}", t):
                continue
            if re.search(r"[A-Za-z]", t) and re.search(r"\d", t):
                if len(t) >= 3:
                    out["serial_code"] = t.upper()
                    break

    # 4) If card_number still not found, try pattern with leading '#'
    if not out["card_number"]:
        m = re.search(r"#\s*(\d{1,4})\s*/\s*(\d{1,4})", s)
        if m:
            out["card_number"] = f"{m.group(1)}/{m.group(2)}"

    # cleanup formats
    if out["card_number"]:
        out["card_number"] = out["card_number"].replace(" ", "")
    if out["serial_full"]:
        out["serial_full"] = out["serial_full"].replace(" ", "")
        if not out["serial_code"]:
            left = out["serial_full"].split("/")[0]
            out["serial_code"] = left.upper()
    if out["serial_run"]:
        out["serial_run"] = out["serial_run"].strip()

    return out

# ---------------------------
# Set/name detection
# ---------------------------
def detect_set_and_name(tokens: List[str], brand: Optional[str], year_token: Optional[str],
                        card_number_token: Optional[str], serial_token: Optional[str]) -> (Optional[str], Optional[str]):
    cleaned = []
    excludes = set()
    if year_token:
        excludes.add(year_token)
    if card_number_token:
        excludes.add(card_number_token)
        excludes.add("#" + card_number_token)
    if serial_token:
        excludes.add(serial_token)
    if brand:
        excludes.add(brand.lower()); excludes.add(brand)
    for t in tokens:
        if not t:
            continue
        if t.lower() in excludes:
            continue
        cleaned.append(t)
    set_keywords = {"set", "series", "collection", "pack", "edition", "evolution", "prismatic", "classic", "foil", "holo", "premium", "deco", "topps", "chrome"}
    idx_set = None
    for i, t in enumerate(cleaned):
        if t.lower() in set_keywords:
            idx_set = i
            break
    set_name = None; name = None
    if idx_set is not None:
        left = cleaned[max(0, idx_set - 3): idx_set + 1]
        right = []
        j = idx_set + 1
        while j < len(cleaned) and len(right) < 4:
            if re.fullmatch(r"\d{1,4}", cleaned[j]):
                break
            right.append(cleaned[j]); j += 1
        set_name = " ".join(left + right).strip()
        if j < len(cleaned):
            name_tokens = cleaned[j:j+3]
            if name_tokens:
                name = " ".join(name_tokens).strip()
    else:
        if len(cleaned) >= 4:
            set_name = " ".join(cleaned[:3])
            name = " ".join(cleaned[3:6])
        elif len(cleaned) >= 2:
            name = " ".join(cleaned[-2:])
            set_name = " ".join(cleaned[:-2]) if len(cleaned) > 2 else None
        elif len(cleaned) == 1:
            name = cleaned[0]; set_name = None
    if name:
        name = " ".join([w.capitalize() for w in re.findall(r"[A-Za-z0-9]+", name)])
    if set_name:
        set_name = " ".join([w.capitalize() for w in re.findall(r"[A-Za-z0-9]+", set_name)])
    return set_name or None, name or None

# ---------------------------
# Build normalized canonical + metadata
# ---------------------------
def build_normalized_and_metadata(title: str) -> Dict[str, Any]:
    t = title or ""
    cleaned = clean_text_for_tokens(t)
    nums = extract_numbers_and_serials(t)  # use raw title for best symbol capture

    # year detection from cleaned text
    year = None
    ym = re.search(r"\b(19|20)\d{2}\b", cleaned)
    if ym:
        year = ym.group(0)

    # tokens keep slashes/hashes/dots preserved
    tokens = re.findall(r"[A-Za-z0-9\/#\-\.\:]{1,20}", cleaned)

    # brand detection (first alpha token not in noise)
    brand = None
    noise_start = {"the", "authentic", "official", "new", "rare", "lot", "sealed"}
    for tok in tokens:
        tl = tok.lower()
        if tl in noise_start:
            continue
        if re.search(r"[A-Za-z]", tok):
            brand = tok.capitalize()
            break

    set_name, name = detect_set_and_name(tokens, brand, year, nums.get("card_number"), nums.get("serial_code"))

    # fallback name if missing
    if not name:
        excludes = set()
        if brand:
            excludes.add(brand.lower()); excludes.add(brand)
        if year:
            excludes.add(year)
        if nums.get("card_number"):
            excludes.add(nums["card_number"])
        if nums.get("serial_full"):
            excludes.add(nums["serial_full"])
        if nums.get("serial_run"):
            excludes.add("/" + nums["serial_run"])
        remaining = [tk for tk in tokens if tk and tk not in excludes and re.search(r"[A-Za-z]", tk)]
        if remaining:
            name = " ".join([w.capitalize() for w in remaining[-2:]])

    # compose canonical normalized string: Year Brand Set Name Serial CardNumber SerialRun (if no cardnum)
    parts = []
    if year: parts.append(str(year))
    if brand: parts.append(brand)
    if set_name: parts.append(set_name)
    if name: parts.append(name)
    serial_code = nums.get("serial_code")
    if serial_code:
        parts.append(serial_code.upper())
    card_number = nums.get("card_number")
    if card_number:
        parts.append(card_number)
    elif nums.get("serial_run") and not serial_code:
        parts.append("/" + nums.get("serial_run"))

    normalized = " ".join(parts).strip()

    metadata = {
        "Year": str(year) if year else None,
        "Brand": brand if brand else None,
        "Set": set_name if set_name else None,
        "Name": name if name else None,
        "Serial Numbered": serial_code.upper() if serial_code else None,
        "Card Number": card_number if card_number else None,
        "Serial Run": nums.get("serial_run") if nums.get("serial_run") else None,
        "Serial Full": nums.get("serial_full") if nums.get("serial_full") else None,
    }
    return {"raw": title, "normalized": normalized, "metadata": metadata}

# ---------------------------
# Compare metadata scoring (weights include serial run/full)
# ---------------------------
def compare_metadata_score(user_meta: Dict[str, Any], post_meta: Dict[str, Any]) -> float:
    """
    Returns raw score (not normalized). We define weights for:
      - card number exact
      - serial full exact
      - serial run denominator match
      - serial code match
      - name exact / partial
      - set match
      - year & brand
    """
    # weights (tunable)
    W_CARD = 5.0            # exact numerator/denominator card number
    W_SERIAL_FULL = 4.5    # serial code + /run exact
    W_SERIAL_RUN = 3.5     # denominator-only match (/25)
    W_SERIAL_CODE = 3.0    # serial code only (SV08)
    W_NAME_EXACT = 2.0
    W_NAME_PART = 1.0
    W_SET = 1.0
    W_YEAR = 0.6
    W_BRAND = 0.4

    def norm(x): return "" if not x else re.sub(r"\s+", " ", re.sub(r"[^\w\/#\-\:]"," ", str(x))).strip().lower()

    u_card = norm(user_meta.get("Card Number"))
    u_serial_full = norm(user_meta.get("Serial Full"))
    u_serial_code = norm(user_meta.get("Serial Numbered"))
    u_serial_run = norm(user_meta.get("Serial Run"))
    u_name = norm(user_meta.get("Name"))
    u_set = norm(user_meta.get("Set"))
    u_year = norm(user_meta.get("Year"))
    u_brand = norm(user_meta.get("Brand"))

    p_card = norm(post_meta.get("Card Number"))
    p_serial_full = norm(post_meta.get("Serial Full"))
    p_serial_code = norm(post_meta.get("Serial Numbered"))
    p_serial_run = norm(post_meta.get("Serial Run"))
    p_name = norm(post_meta.get("Name"))
    p_set = norm(post_meta.get("Set"))
    p_year = norm(post_meta.get("Year"))
    p_brand = norm(post_meta.get("Brand"))

    score = 0.0

    # exact card number
    if u_card and p_card and u_card.replace("#","") == p_card.replace("#",""):
        score += W_CARD

    # serial full exact (code/run)
    if u_serial_full and p_serial_full and (u_serial_full == p_serial_full):
        score += W_SERIAL_FULL

    # serial code exact
    if u_serial_code and p_serial_code and u_serial_code == p_serial_code:
        score += W_SERIAL_CODE

    # serial run denominator match (e.g., user has /25 and post has /25 in either card_number or serial_run)
    if u_serial_run:
        # compare denominators
        if p_serial_run and u_serial_run == p_serial_run:
            score += W_SERIAL_RUN
        else:
            # maybe post card_number endswith same denom
            if p_card and "/" in p_card:
                denom = p_card.split("/")[-1]
                if denom == u_serial_run:
                    score += W_SERIAL_RUN * 0.9  # slightly lower than exact serial_run field
    # Name exact / partial
    if u_name and p_name and u_name == p_name:
        score += W_NAME_EXACT
    else:
        if u_name and p_name:
            utoks = set(u_name.split()); ptoks = set(p_name.split())
            if utoks and ptoks:
                overlap = len(utoks & ptoks); denom = max(len(utoks), len(ptoks))
                score += W_NAME_PART * (overlap / denom)
    # Set matching (exact or partial)
    if u_set and p_set:
        if u_set == p_set:
            score += W_SET
        else:
            us = set(u_set.split()); ps = set(p_set.split())
            if us & ps:
                score += W_SET * (len(us & ps) / max(1, len(us)))

    # year & brand
    if u_year and p_year and u_year == p_year:
        score += W_YEAR
    if u_brand and p_brand and u_brand == p_brand:
        score += W_BRAND

    return score

# ---------------------------
# Optional: GPT selector (compact metadata) - synchronous wrapper
# ---------------------------
def call_gpt_selector_sync(user_meta: Dict[str, Any], compact_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Calls gpt-4o-mini to select best candidate.
    Returns dict: {"chosen_index": int|null, "confidence": float, "reason": str} or None if not available.
    """
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if openai is None or not openai_key:
        return None

    try:
        openai.api_key = openai_key
        candidates_json = json.dumps(compact_candidates, ensure_ascii=False, separators=(",", ":"))
        user_json = json.dumps(user_meta, ensure_ascii=False, separators=(",", ":"))

        system_msg = (
            "You are a precise assistant. Given a user's card metadata and a short list of candidate postings (each with metadata), "
            "select the single BEST MATCH. Prioritize keys in this order: Card Number (exact), Serial Full (code/run), "
            "Serial Run (denominator), Serial Code, Name, Set, Year, Brand. Return JSON ONLY: "
            "{\"chosen_index\": <int|null>, \"confidence\": <0.0-1.0>, \"reason\": \"short explanation\"}."
        )
        user_msg = (
            f"User metadata: {user_json}\n\n"
            f"Candidates (index,title,metadata): {candidates_json}\n\n"
            "Return JSON only. If no candidate matches sufficiently, return chosen_index:null and confidence:0.0."
        )

        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.0,
            max_tokens=200,
        )
        text = ""
        try:
            text = resp["choices"][0]["message"]["content"]
        except Exception:
            text = str(resp)

        text_str = str(text).strip()
        try:
            parsed = json.loads(text_str)
            return parsed
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text_str)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None
    except Exception:
        return None

# ---------------------------
# match_user_card fallback deterministic (sync)
# ---------------------------
def match_user_card(user_card_title: str, ebay_posts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Synchronous deterministic comparator:
     - normalizes all posts and user
     - tries direct exact matches: card_number, serial_full, serial_run (denominator)
     - otherwise scores per compare_metadata_score and picks best (tie-break by sold_at)
     - returns matched original post dict or None
    """
    # prepare posts normalized
    posts = []
    for p in ebay_posts or []:
        post = dict(p)
        raw_title = post.get("name") or post.get("title") or ""
        nm = build_normalized_and_metadata(raw_title)
        post["_normalized"] = nm["normalized"]
        post["_metadata"] = nm["metadata"]
        sold = post.get("sold_at")
        post["_sold_at_parsed"] = None
        if sold:
            try:
                post["_sold_at_parsed"] = datetime.fromisoformat(str(sold).replace("Z", "+00:00"))
            except Exception:
                try:
                    post["_sold_at_parsed"] = datetime.strptime(str(sold), "%Y-%m-%d")
                except Exception:
                    post["_sold_at_parsed"] = None
        posts.append(post)

    user_norm = build_normalized_and_metadata(user_card_title)
    user_meta = user_norm["metadata"]

    # require at least some metadata
    if not any(user_meta.values()):
        return None

    # 1) direct deterministic checks
    u_card = (user_meta.get("Card Number") or "").strip()
    u_serial_full = (user_meta.get("Serial Full") or "").strip()
    u_serial_run = (user_meta.get("Serial Run") or "").strip()
    u_serial_code = (user_meta.get("Serial Numbered") or "").strip()

    # if exact card_number in user -> return exact match if exists
    if u_card:
        for p in posts:
            p_card = (p.get("_metadata") or {}).get("Card Number")
            if p_card and p_card.replace(" ", "") == u_card.replace(" ", ""):
                return p

    # if full serial present -> exact match
    if u_serial_full:
        for p in posts:
            if (p.get("_metadata") or {}).get("Serial Full") == u_serial_full:
                return p

    # if serial_run present (denominator only) -> filter posts with matching denom either in Card Number or Serial Run
    if u_serial_run:
        candidates = []
        for p in posts:
            pm = (p.get("_metadata") or {})
            p_serial_run = pm.get("Serial Run")
            p_card = pm.get("Card Number")
            if p_serial_run and p_serial_run == u_serial_run:
                candidates.append(p)
            elif p_card and "/" in p_card and p_card.split("/")[-1] == u_serial_run:
                candidates.append(p)
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            # score among candidates
            best = None; best_score = -1
            for c in candidates:
                s = compare_metadata_score(user_meta, c.get("_metadata") or {})
                if s > best_score:
                    best_score = s; best = c
                elif s == best_score:
                    # tie-break by sold_at (most recent)
                    d_best = best.get("_sold_at_parsed")
                    d_c = c.get("_sold_at_parsed")
                    if d_c and (not d_best or d_c > d_best):
                        best = c
            # require reasonable score
            if best and best_score > 0:
                return best

    # 2) fallback scoring across all posts
    scored = []
    for p in posts:
        s = compare_metadata_score(user_meta, p.get("_metadata") or {})
        scored.append({"post": p, "score": s})
    scored.sort(key=lambda x: x["score"], reverse=True)
    if not scored:
        return None

    top = scored[0]
    # compute SUM_POSSIBLE to normalize
    SUM_POSSIBLE = 5.0 + 4.5 + 3.5 + 3.0 + 2.0 + 1.0 + 0.6 + 0.4
    threshold_raw = 0.55 * SUM_POSSIBLE  # require >55% of max or card/serial exact
    if top["score"] >= threshold_raw:
        # if multiple similarly high, choose most recent sold_at
        close = [c for c in scored if (top["score"] - c["score"]) <= 0.03 * SUM_POSSIBLE]
        if len(close) > 1:
            best = None; bestd = None
            for c in close:
                d = c["post"].get("_sold_at_parsed")
                if d and (bestd is None or d > bestd):
                    best = c["post"]; bestd = d
            return best or close[0]["post"]
        return top["post"]
    # if not meet threshold but top has exact card or serial, accept
    pmeta = top["post"].get("_metadata") or {}
    if u_card and pmeta.get("Card Number") and u_card.replace(" ", "") == pmeta.get("Card Number").replace(" ", ""):
        return top["post"]
    if u_serial_full and pmeta.get("Serial Full") and u_serial_full == pmeta.get("Serial Full"):
        return top["post"]
    # else no match
    return None

# ---------------------------
# get_last_sold_item (async main function required)
# ---------------------------
async def get_last_sold_item(
    user_card_title: str,
    normalized_posts: List[Dict[str, Any]],
    *,
    top_k: int = 500,
    shortlist_k: int = 12,
    required_confidence: float = 0.97,
) -> Dict[str, Any]:
    """
    Async wrapper compatible with prior signature.
    Returns dict: {found(bool), match(post|None), score(float 0..1), reason(str), candidates(list)}
    """
    # limit posts
    posts = list(normalized_posts or [])[:top_k]

    # ensure posts normalized
    posts_prepared = []
    for p in posts:
        post = dict(p)
        raw_title = post.get("name") or post.get("title") or ""
        if not post.get("_metadata") or not post.get("_normalized"):
            nm = build_normalized_and_metadata(raw_title)
            post["_normalized"] = nm["normalized"]
            post["_metadata"] = nm["metadata"]
        sold = post.get("sold_at")
        post["_sold_at_parsed"] = None
        if sold:
            try:
                post["_sold_at_parsed"] = datetime.fromisoformat(str(sold).replace("Z", "+00:00"))
            except Exception:
                try:
                    post["_sold_at_parsed"] = datetime.strptime(str(sold), "%Y-%m-%d")
                except Exception:
                    post["_sold_at_parsed"] = None
        posts_prepared.append(post)

    # normalize user
    user_nm = build_normalized_and_metadata(user_card_title)
    user_meta = user_nm["metadata"]

    # if user metadata empty -> cannot match
    if not any(user_meta.values()):
        return {"found": False, "match": None, "score": 0.0, "reason": "insufficient_user_metadata", "candidates": []}

    # Try deterministic fast-path (no LLM) via match_user_card
    try:
        maybe = match_user_card(user_card_title, posts_prepared)
        if asyncio.iscoroutine(maybe):
            maybe = await maybe
        if maybe:
            # compute score
            final_score_raw = compare_metadata_score(user_meta, (maybe.get("_metadata") or {}))
            SUM_POSSIBLE = 5.0 + 4.5 + 3.5 + 3.0 + 2.0 + 1.0 + 0.6 + 0.4
            normalized_final = float(max(0.0, min(1.0, final_score_raw / SUM_POSSIBLE)))
            # Deterministic match found - should always be True regardless of confidence
            return {
                "found": True,
                "match": maybe,
                "score": round(normalized_final, 4),
                "reason": "deterministic_match",
                "candidates": []
            }
    except Exception:
        # ignore and fallback to scoring+LLM selection below
        pass

    # Build compact candidates (metadata only) to optionally send to LLM
    compact_candidates = []
    for idx, p in enumerate(posts_prepared):
        compact_candidates.append({
            "index": idx,
            "title": p.get("name") or p.get("title") or "",
            "metadata": p.get("_metadata") or {}
        })

    # Try LLM selector (synchronous wrapper) - runs in thread to avoid blocking
    llm_choice = None
    try:
        loop = asyncio.get_event_loop()
        llm_choice = await loop.run_in_executor(None, call_gpt_selector_sync, user_meta, compact_candidates)
    except Exception:
        llm_choice = None

    chosen_post = None
    final_score = 0.0
    reason = ""

    if llm_choice and isinstance(llm_choice, dict):
        ci = llm_choice.get("chosen_index")
        conf = llm_choice.get("confidence") or 0.0
        if ci is not None:
            try:
                ci_int = int(ci)
                if 0 <= ci_int < len(posts_prepared):
                    chosen_post = posts_prepared[ci_int]
                    final_score_raw = compare_metadata_score(user_meta, (chosen_post.get("_metadata") or {}))
                    final_score = final_score_raw
                    reason = f"llm_choice_conf_{conf}"
            except Exception:
                chosen_post = None

    # If no LLM choice or low confidence, do deterministic scoring across all posts
    if not chosen_post:
        scored = []
        for p in posts_prepared:
            s = compare_metadata_score(user_meta, p.get("_metadata") or {})
            scored.append({"post": p, "score": s})
        scored.sort(key=lambda x: x["score"], reverse=True)
        if not scored:
            return {"found": False, "match": None, "score": 0.0, "reason": "no_candidates", "candidates": []}

        # Take top shortlist_k
        shortlist = scored[:shortlist_k]
        top_score_raw = shortlist[0]["score"]
        # raw threshold on SUM_POSSIBLE scale
        SUM_POSSIBLE = 5.0 + 4.5 + 3.5 + 3.0 + 2.0 + 1.0 + 0.6 + 0.4
        raw_threshold = required_confidence * SUM_POSSIBLE
        high = [s for s in shortlist if s["score"] >= raw_threshold]

        if len(high) == 1:
            chosen_post = high[0]["post"]
            final_score = high[0]["score"]
            reason = "heuristic_high"
        elif len(high) > 1:
            # pick most recent sold_at among high
            best = None; bestd = None
            for h in high:
                d = h["post"].get("_sold_at_parsed")
                if d and (bestd is None or d > bestd):
                    best = h["post"]; bestd = d
            chosen_post = best or high[0]["post"]
            final_score = max([h["score"] for h in high])
            reason = "heuristic_tiebreak_sold_at"
        else:
            # maybe accept top if exact card or serial present
            top = shortlist[0]
            top_meta = top["post"].get("_metadata") or {}
            def n(x): return "" if not x else re.sub(r"\s+"," ",re.sub(r"[^\w\/#\-]"," ",str(x))).strip().lower()
            u_card = n(user_meta.get("Card Number")); p_card = n(top_meta.get("Card Number"))
            u_serial_full = n(user_meta.get("Serial Full")); p_serial_full = n(top_meta.get("Serial Full"))
            u_serial_code = n(user_meta.get("Serial Numbered")); p_serial_code = n(top_meta.get("Serial Numbered"))
            if u_card and p_card and u_card.replace(" ", "") == p_card.replace(" ", ""):
                chosen_post = top["post"]; final_score = top["score"]; reason = "card_number_exact"
            elif u_serial_full and p_serial_full and u_serial_full == p_serial_full:
                chosen_post = top["post"]; final_score = top["score"]; reason = "serial_full_exact"
            elif u_serial_run and ( (top_meta.get("Serial Run") and top_meta.get("Serial Run") == u_serial_run) or (top_meta.get("Card Number") and "/" in top_meta.get("Card Number") and top_meta.get("Card Number").split("/")[-1] == u_serial_run) ):
                chosen_post = top["post"]; final_score = top["score"]; reason = "serial_run_match"
            else:
                # no suitable match
                # prepare compact candidates output for debugging
                out_candidates = []
                for c in shortlist:
                    out_candidates.append({
                        "title": c["post"].get("name") or c["post"].get("title"),
                        "normalized": c["post"].get("_normalized"),
                        "metadata": c["post"].get("_metadata"),
                        "score_raw": round(float(c["score"]), 4)
                    })
                return {"found": False, "match": None, "score": round(float(top["score"]/SUM_POSSIBLE),4), "reason": "below_threshold", "candidates": out_candidates}

    # normalize final_score to 0..1
    SUM_POSSIBLE = 5.0 + 4.5 + 3.5 + 3.0 + 2.0 + 1.0 + 0.6 + 0.4
    normalized_final_score = float(max(0.0, min(1.0, final_score / SUM_POSSIBLE)))

    # if multiple posts have normalized score >= required_confidence -> pick most recent among them
    all_scored = []
    for p in posts_prepared:
        s = compare_metadata_score(user_meta, p.get("_metadata") or {})
        all_scored.append({"post": p, "score": s})
    multi_high = [s for s in all_scored if (s["score"] / SUM_POSSIBLE) >= required_confidence]
    if len(multi_high) > 1:
        best = None; bestd = None
        for s in multi_high:
            d = s["post"].get("_sold_at_parsed")
            if d and (bestd is None or d > bestd):
                best = s["post"]; bestd = d
        if best:
            chosen_post = best

    # found should be True if we have a chosen_post, regardless of confidence score
    # because exact matches (card number, serial) can override confidence requirements
    found = chosen_post is not None

    # prepare top candidates list (compact)
    final_candidates = []
    scored_all_sorted = sorted(all_scored, key=lambda x: x["score"], reverse=True)
    for c in scored_all_sorted[:shortlist_k]:
        final_candidates.append({
            "title": c["post"].get("name") or c["post"].get("title"),
            "normalized": c["post"].get("_normalized"),
            "metadata": c["post"].get("_metadata"),
            "score_raw": round(float(c["score"]), 4)
        })

    return {
        "found": False if not chosen_post else True,
        "match": chosen_post,
        "score": round(normalized_final_score, 4),
        "reason": reason or ("llm_selected" if llm_choice else "heuristic"),
        "candidates": final_candidates
    }

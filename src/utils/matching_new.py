# matching_new.py
import os
import re
import unicodedata
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import numpy as np

from src.utils.logger import get_logger

matching_logger = get_logger("matching", level="DEBUG")


def normalize_title(title: str) -> Dict[str, Any]:
    """
    Normalize a raw eBay title and extract structured fields.
    Returns dict with keys:
      raw, normalized, year, brand, setName, subset, player_name,
      serial_numbered, card_number, grading, variant, confidence
    """
    if not title:
        return {
            "raw": title,
            "normalized": "",
            "year": None,
            "brand": None,
            "setName": None,
            "subset": None,
            "player_name": None,
            "serial_numbered": None,
            "card_number": None,
            "grading": None,
            "variant": None,
            "confidence": 0.0,
        }

    raw = title
    s = unicodedata.normalize("NFKC", raw)
    s = re.sub(r"[\U00010000-\U0010ffff]", " ", s)  # remove emoji / non-BMP
    s = s.strip()
    orig = s[:]
    s_lower = s.lower()

    # remove irrelevant punctuation but keep #, /, -
    s_lower = re.sub(r"[,_·•\(\)\[\]:;\"`']", " ", s_lower)
    s_lower = re.sub(r"\s+", " ", s_lower).strip()

    # grading
    grading = None
    gm = re.search(r"\b(psa|bgs|sgc|hga|cas)\s*([0-9](?:\.[05])?)\b", s_lower, flags=re.IGNORECASE)
    if gm:
        grading = f"{gm.group(1).upper()} {gm.group(2)}"
        s_lower = s_lower[:gm.start()] + s_lower[gm.end():]
        s_lower = s_lower.strip()

    # explicit # token
    explicit_hash = None
    hm = re.search(r"#\s*([a-z0-9\-\/]+)", s_lower, flags=re.IGNORECASE)
    if hm:
        explicit_hash = hm.group(1).strip()
        s_lower = s_lower[:hm.start()] + s_lower[hm.end():]
        s_lower = s_lower.strip()

    # fractions (a/b)
    fracs = [(m.group(0), m.start(), m.end(), m.group(1), m.group(2))
             for m in re.finditer(r"\b(\d{1,4})\s*/\s*(\d{1,4})\b", s_lower)]

    # year (YYYY) or season-derived
    year = None
    ym = re.search(r"\b(19|20)\d{2}\b", s_lower)
    if ym:
        year = ym.group(0)
        s_lower = s_lower[:ym.start()] + s_lower[ym.end():]
        s_lower = s_lower.strip()
    else:
        chosen = None
        for i, (f, start, end, a, b) in enumerate(fracs):
            if len(a) == 4:
                chosen = i
                break
        if chosen is None:
            for i, (f, start, end, a, b) in enumerate(fracs):
                left = s_lower[max(0, start - 20):start]
                if re.search(r"( set| season| series| collection| pack| fan| ucl| champions| match )", left):
                    chosen = i
                    break
        if chosen is not None:
            _, start, end, a, b = fracs[chosen]
            if len(a) == 2:
                year = "20" + a
            else:
                year = a
            s_lower = s_lower[:start] + s_lower[end:]
            s_lower = s_lower.strip()
            fracs.pop(chosen)

    # card_number (from fraction) prefer remaining fractions
    card_number = None
    if fracs:
        for (f, start, end, a, b) in fracs:
            card_number = f"{a}/{b}"
            s_lower = s_lower[:start] + s_lower[end:]
            s_lower = s_lower.strip()
            break

    # prefer explicit_hash with slash as card number
    if explicit_hash and "/" in explicit_hash and re.search(r"\d", explicit_hash):
        card_number = explicit_hash.replace(" ", "")
        explicit_hash = None

    # detect serial code (letters+digits, e.g., SVO8, BA-CR1, CR7)
    serial_code = None
    tokens_orig = re.findall(r"[A-Za-z0-9\-\/]+", orig)
    tokens_lower = [t.lower() for t in tokens_orig]
    for t in reversed(tokens_orig):
        tl = t.strip()
        if "/" in tl or "#" in tl:
            continue
        if re.fullmatch(r"\d{1,4}", tl):
            continue
        # pattern: letters + digits (optionally hyphen)
        if re.search(r"[A-Za-z]{1,4}\-?[A-Za-z0-9]{1,6}\d", tl):
            if re.search(r"[A-Za-z]", tl) and re.search(r"\d", tl):
                serial_code = tl.upper()
                # remove first occurrence in s_lower
                try:
                    s_lower = re.sub(re.escape(tl.lower()), " ", s_lower, count=1)
                    s_lower = re.sub(r"\s+", " ", s_lower).strip()
                except Exception:
                    pass
                break

    # explicit_hash that is alnum+digit but not slash -> serial
    if not serial_code and explicit_hash and re.search(r"[A-Za-z]", explicit_hash) and re.search(r"\d", explicit_hash) and "/" not in explicit_hash:
        serial_code = explicit_hash.upper()
        explicit_hash = None

    # if explicit_hash still digits -> treat as simple card_number (collector)
    if explicit_hash and re.search(r"\d", explicit_hash) and not card_number:
        card_number = explicit_hash.upper()
        explicit_hash = None

    # cleanup
    s_lower = re.sub(r"[^\w\s\-\/]", " ", s_lower)
    s_lower = re.sub(r"\s+", " ", s_lower).strip()
    tokens = s_lower.split()

    # strip noise at end
    noise_end = {"e", "en", "eng", "ebay", "lot", "cards", "card", "for", "from"}
    while tokens and tokens[-1] in noise_end:
        tokens.pop()

    # brand heuristic (first token)
    brand = None
    noise_start = {"the", "authentic", "official", "new", "rare", "lot"}
    if tokens and tokens[0] not in noise_start:
        brand = tokens.pop(0).capitalize()

    # set detection
    setName = None
    subset = None
    idx_set = None
    for i, t in enumerate(tokens):
        if t in ("set", "series", "collection", "pack", "fan", "edition",
                 "evolution", "prismatic", "premier", "classic"):
            idx_set = i
            break

    if idx_set is not None:
        left_start = max(0, idx_set - 3)
        set_tokens = tokens[left_start:idx_set + 1]
        right = []
        j = idx_set + 1
        while j < len(tokens) and len(right) < 4 and not re.match(r"\d{1,4}$", tokens[j]):
            right.append(tokens[j])
            j += 1
        if set_tokens:
            setName = " ".join(set_tokens)
        if right:
            subset = " ".join(right)
        tokens = tokens[:left_start] + tokens[j:]
    else:
        if tokens:
            take = min(3, len(tokens))
            setName = " ".join(tokens[:take])
            tokens = tokens[take:]

    # avoid set duplicating brand
    if setName and brand and setName.lower().startswith(brand.lower()):
        parts = setName.split()
        if len(parts) > 1:
            setName = " ".join(parts[1:])
        else:
            setName = None

    # player/card name extraction
    player_name = None
    extras = []
    if tokens:
        alpha_tokens = [t for t in tokens if not re.search(r"\d", t)]
        if alpha_tokens:
            take = min(len(alpha_tokens), 3)
            name_tokens = []
            removed = 0
            for t in tokens:
                if not re.search(r"\d", t) and len(name_tokens) < take:
                    name_tokens.append(t)
                    removed += 1
                else:
                    break
            player_name = " ".join([w.capitalize() for w in name_tokens])
            tokens = tokens[removed:]
        extras.extend(tokens)

    # include serial and grading in extras
    if serial_code and serial_code not in extras:
        extras.insert(0, serial_code)
    if grading:
        extras.append(grading)

    # build canonical normalized string
    parts = []
    if year:
        parts.append(str(year))
    if brand:
        parts.append(brand)
    if setName:
        set_name_clean = " ".join([w.capitalize() if w.isalpha() else w for w in setName.split()])
        parts.append(set_name_clean)
    if player_name:
        parts.append(player_name)
    if serial_code:
        parts.append(serial_code.upper())
    if card_number:
        parts.append("#" + str(card_number).upper())
    if extras:
        ex_formatted = []
        for e in extras:
            if re.search(r"[A-Za-z]", e) and re.search(r"\d", e):
                ex_formatted.append(e.upper())
            else:
                ex_formatted.append(e.capitalize())
        parts.append(" ".join(ex_formatted))

    normalized = " ".join(parts).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", normalized, flags=re.IGNORECASE)

    # confidence scoring (simple heuristic)
    score = 0.0
    weight = 0.0
    weight += 1; score += 1.0 if year else 0.0
    weight += 1; score += 1.0 if brand else 0.0
    weight += 1; score += 1.0 if setName else 0.0
    weight += 1
    if card_number:
        score += 1.0
    elif serial_code:
        score += 0.6
    else:
        score += 0.0
    weight += 1; score += 1.0 if player_name else 0.5 if extras else 0.0
    confidence = round((score / weight) if weight > 0 else 0.0, 3)

    serial_numbered = serial_code.upper() if serial_code else None
    card_number_formatted = card_number if card_number else None

    matching_logger.debug(
        f'Ebay Post: "{raw}"\n'
        f'Normalized: "{normalized}" | year={year} brand={brand} set={setName} subset={subset} '
        f'player={player_name} serial_numbered={serial_numbered} card_number={card_number_formatted} grading={grading} confidence={confidence}'
    )

    return {
        "raw": raw,
        "normalized": normalized,
        "year": year,
        "brand": brand,
        "setName": setName,
        "subset": subset,
        "player_name": player_name,
        "serial_numbered": serial_numbered,
        "card_number": card_number_formatted,   # e.g. "238/191"
        "grading": grading,
        "variant": " ".join(extras) if extras else None,
        "confidence": confidence
    }


# ------------------------------
# Matching function: single pipeline
# ------------------------------
async def get_last_sold_item(
    user_card_title: str,
    normalized_posts: List[Dict[str, Any]],
    *,
    top_k: int = 500,
    shortlist_k: int = 12,
    required_confidence: float = 0.97,
) -> Dict[str, Any]:
    """
    Inputs:
      - user_card_title: string
      - normalized_posts: list of dicts (each should contain at least 'id','title'/'name','sold_at')
        optionally may already contain 'normalized' field (dict from normalize_title)
    Returns:
      dict with keys: found(bool), match(dict|None), score(float), reason(str), candidates(list)
    """

    # helper: parse sold_at robustly
    def parse_date(d: Optional[Any]) -> Optional[datetime]:
        if not d:
            return None
        if isinstance(d, datetime):
            return d
        s = str(d).strip()
        # try iso
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            pass
        fmts = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%b %d %Y", "%d %b %Y"]
        for f in fmts:
            try:
                return datetime.strptime(s, f)
            except Exception:
                continue
        m = re.search(r"(20\d{2}|19\d{2})", s)
        if m:
            try:
                return datetime(int(m.group(0)), 1, 1)
            except Exception:
                return None
        return None

    # helper: cosine similarity
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # helper: extract price from post dict robustly
    def extract_price(post: Dict[str, Any]) -> Optional[float]:
        if not post:
            return None
        for key in ("price", "sold_price", "sale_price", "final_price", "amount", "soldAmount"):
            if key in post and post[key] is not None:
                v = post[key]
                # if dict with value/amount
                if isinstance(v, dict):
                    for subk in ("value", "amount", "price"):
                        if subk in v and v[subk] is not None:
                            try:
                                return float(v[subk])
                            except Exception:
                                continue
                else:
                    try:
                        return float(v)
                    except Exception:
                        # maybe string with currency symbol
                        s = str(v)
                        m = re.search(r"[\d\.,]+", s)
                        if m:
                            try:
                                # normalize thousand/comma
                                val = m.group(0).replace(",", "")
                                return float(val)
                            except Exception:
                                continue
        # try nested 'metadata' or 'sale' keys
        if "metadata" in post and isinstance(post["metadata"], dict):
            return extract_price(post["metadata"])
        if "sale" in post and isinstance(post["sale"], dict):
            return extract_price(post["sale"])
        return None

    # LLM selector: ask gpt-4o-mini to pick best candidate (returns dict)
    async def call_gpt_selector(user_norm: Dict[str, Any], shortlist: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if not openai_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)

            # build prompt
            user_summary = {
                "normalized": user_norm.get("normalized"),
                "year": user_norm.get("year"),
                "brand": user_norm.get("brand"),
                "player_name": user_norm.get("player_name"),
                "card_number": user_norm.get("card_number"),
                "serial_numbered": user_norm.get("serial_numbered"),
            }

            candidate_lines = []
            for idx, c in enumerate(shortlist):
                p = c["post"]
                price = extract_price(p)
                candidate_lines.append({
                    "index": idx,
                    "title": p.get("title") or p.get("name"),
                    "normalized": (p.get("_norm") or {}).get("normalized") or (p.get("_canonical")),
                    "sold_at": p.get("sold_at"),
                    "price": price,
                    "score": round(c.get("score", 0.0), 4)
                })

            # messages: instruct to return JSON only
            system_msg = (
                "You are a concise assistant that selects which past sold listing is the BEST match "
                "for the provided user card description. Output JSON ONLY with keys: "
                "\"chosen_index\" (integer index into candidates array or null), "
                "\"confidence\" (number 0.0-1.0), and \"reason\" (short explanation). "
                "If multiple candidates are equally plausible, choose the one with the most recent sold_at. "
                "Do NOT output anything else."
            )

            user_msg = (
                f"User card normalized: {json.dumps(user_summary, ensure_ascii=False)}\n\n"
                f"Candidates (index, title, normalized, sold_at, price, score):\n"
                f"{json.dumps(candidate_lines, ensure_ascii=False, indent=2)}\n\n"
                "Return JSON only as described."
            )

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=256,
                temperature=0.0,
            )
            # extract text
            text = ""
            try:
                text = resp.choices[0].message.content
            except Exception:
                text = str(resp)

            # try to find JSON substring
            text_str = str(text).strip()
            # attempt direct json parse
            try:
                parsed = json.loads(text_str)
                return parsed
            except Exception:
                # try to find first {...} block
                m = re.search(r"\{[\s\S]*\}", text_str)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        return parsed
                    except Exception:
                        matching_logger.debug("LLM returned non-JSON or unparseable JSON: %s", text_str)
                        return None
                else:
                    matching_logger.debug("LLM returned no JSON block: %s", text_str)
                    return None
        except Exception as e:
            matching_logger.exception("Error calling gpt selector: %s", e)
            return None

    # prepare posts: ensure normalized dict exists
    posts = []
    for p in normalized_posts:
        post = dict(p)  # shallow copy
        raw_title = post.get("title") or post.get("name") or post.get("title_raw") or ""
        norm = post.get("normalized")
        if isinstance(norm, dict):
            norm_dict = norm
        elif isinstance(norm, str) and norm.strip():
            # wrap string to minimal normalized dict
            norm_dict = {"raw": raw_title, "normalized": norm}
        else:
            # compute normalization
            try:
                norm_dict = normalize_title(raw_title)
            except Exception:
                norm_dict = {"raw": raw_title, "normalized": raw_title}
        post["_norm"] = norm_dict
        post["_canonical"] = norm_dict.get("normalized") or raw_title
        post["_sold_at_parsed"] = parse_date(post.get("sold_at"))
        posts.append(post)

    # normalize user title
    try:
        user_norm = normalize_title(user_card_title)
    except Exception:
        user_norm = {"raw": user_card_title, "normalized": user_card_title}

    user_text = user_norm.get("normalized") or user_card_title

    # build texts to embed: user + top_k candidate canonical texts (we'll limit posts for embedding to top_k by naive heuristic)
    # NOTE: if you have precomputed embeddings, replace this block with vector DB kNN call.
    texts = [user_text]
    candidates_for_embedding = posts[:top_k]
    texts.extend([p["_canonical"] for p in candidates_for_embedding])

    # embedding backend: try OpenAI then fallback to sentence-transformers
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    embeddings: List[List[float]] = []

    async def embed_with_openai(ts: List[str]) -> List[List[float]]:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        resp = client.embeddings.create(model="text-embedding-3-small", input=ts)
        return [r.embedding for r in resp.data]

    async def embed_with_st(ts: List[str]) -> List[List[float]]:
        # runs in thread to avoid blocking event loop
        from sentence_transformers import SentenceTransformer
        def _encode(texts):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
        return await asyncio.to_thread(_encode, ts)

    # try embedding
    try:
        if openai_key:
            embeddings = await embed_with_openai(texts)
        else:
            embeddings = await embed_with_st(texts)
    except Exception:
        # fallback
        embeddings = await embed_with_st(texts)

    user_emb = np.array(embeddings[0], dtype=np.float32)
    cand_embs = [np.array(e, dtype=np.float32) for e in embeddings[1:]]

    # compute base cosine
    results = []
    for i, post in enumerate(candidates_for_embedding):
        base_sim = cosine(user_emb, cand_embs[i]) if i < len(cand_embs) else 0.0
        results.append({
            "post": post,
            "base_sim": base_sim,
            "score": base_sim,
            "norm": post["_norm"]
        })

    # heuristics boosting weights (tunable)
    BOOST_CARDNUMBER = 0.38
    BOOST_SERIAL = 0.30
    BOOST_PLAYER = 0.15
    BOOST_YEAR = 0.06
    BOOST_BRAND = 0.04

    def norm_str(x):
        return None if x is None else str(x).strip().lower()

    u_card = norm_str(user_norm.get("card_number") or user_norm.get("card_number"))
    u_serial = norm_str(user_norm.get("serial_numbered") or user_norm.get("serial_numbered"))
    u_player = norm_str(user_norm.get("player_name"))
    u_year = norm_str(user_norm.get("year"))
    u_brand = norm_str(user_norm.get("brand"))

    for r in results:
        pn = r["norm"] or {}
        p_card = norm_str(pn.get("card_number"))
        p_serial = norm_str(pn.get("serial_numbered"))
        p_player = norm_str(pn.get("player_name"))
        p_year = norm_str(pn.get("year"))
        p_brand = norm_str(pn.get("brand"))

        score = r["base_sim"]
        if u_card and p_card and u_card.replace("#", "") == p_card.replace("#", ""):
            score += BOOST_CARDNUMBER
        if u_serial and p_serial and u_serial == p_serial:
            score += BOOST_SERIAL
        if u_player and p_player:
            if u_player == p_player:
                score += BOOST_PLAYER
            elif u_player in p_player or p_player in u_player:
                score += BOOST_PLAYER * 0.8
            else:
                up_tokens = set(u_player.split())
                pp_tokens = set(p_player.split())
                overlap = len(up_tokens & pp_tokens)
                if overlap > 0:
                    score += (BOOST_PLAYER * 0.4) * (overlap / max(len(up_tokens), 1))
        if u_year and p_year and u_year == p_year:
            score += BOOST_YEAR
        if u_brand and p_brand and u_brand == p_brand:
            score += BOOST_BRAND

        r["score"] = float(score)

    # shortlist
    results.sort(key=lambda x: x["score"], reverse=True)
    shortlist = results[:shortlist_k] if len(results) > 0 else []

    if not shortlist:
        return {"found": False, "match": None, "score": 0.0, "reason": "no_candidates", "candidates": []}

    # Attempt LLM selection (gpt-4o-mini) to disambiguate among shortlist
    llm_choice = None
    try:
        llm_response = await call_gpt_selector(user_norm, shortlist)
        if llm_response and isinstance(llm_response, dict):
            # expect keys: chosen_index, confidence, reason
            chosen_index = llm_response.get("chosen_index")
            conf = llm_response.get("confidence")
            reason_text = llm_response.get("reason")
            if chosen_index is not None and isinstance(chosen_index, int) and 0 <= chosen_index < len(shortlist):
                llm_choice = {
                    "index": int(chosen_index),
                    "confidence": float(conf) if conf is not None else None,
                    "reason": reason_text
                }
                matching_logger.debug("LLM choice: %s", llm_choice)
            else:
                matching_logger.debug("LLM returned no valid chosen_index: %s", llm_response)
        else:
            matching_logger.debug("LLM did not return a usable response")
    except Exception:
        matching_logger.exception("Error while using LLM selector")

    # tie-breaker / chosen selection
    chosen = None
    if llm_choice:
        chosen = shortlist[llm_choice["index"]]
        # we still set final_score based on chosen["score"] but factor in model confidence if available
        final_score = float(chosen["score"])
        if llm_choice.get("confidence") is not None:
            # combine heuristic score and model confidence conservatively (simple average)
            try:
                conf = float(llm_choice["confidence"])
                final_score = float(max(final_score, min(0.99, (final_score * 0.5) + (conf * 0.5))))
            except Exception:
                pass
        chosen_source = "llm"
    else:
        # fallback to original heuristic tie-breaker: most recent sold_at among close candidates
        top_score = shortlist[0]["score"]
        close_candidates = [c for c in shortlist if (top_score - c["score"]) <= 0.03]  # 0.03 tolerance
        if len(close_candidates) > 1:
            best = None
            best_date = None
            for c in close_candidates:
                d = c["post"].get("_sold_at_parsed")
                if d:
                    if best_date is None or d > best_date:
                        best = c
                        best_date = d
            if best is not None:
                chosen = best
            else:
                chosen = shortlist[0]
        else:
            chosen = shortlist[0]
        final_score = chosen["score"]
        chosen_source = "heuristic"

    # final score adjustments: if exact strong signals, force high confidence
    pn = chosen["norm"] or {}
    p_card = norm_str(pn.get("card_number"))
    p_player = norm_str(pn.get("player_name"))
    if u_card and p_card and u_card.replace("#", "") == p_card.replace("#", "") and u_player and p_player and (u_player == p_player or u_player in p_player or p_player in u_player):
        final_score = max(final_score, 0.98)

    final_score = float(max(0.0, min(final_score, 1.0)))
    found = final_score >= required_confidence

    reason_parts = []
    reason_parts.append("accepted" if found else "below_threshold")
    reason_parts.append("selection_source=" + chosen_source)
    if chosen_source == "llm" and llm_choice:
        reason_parts.append(f"llm_confidence={llm_choice.get('confidence')}")
        if llm_choice.get("reason"):
            reason_parts.append("llm_reason=" + llm_choice.get("reason"))

    heuristics = []
    if u_card and p_card and u_card.replace("#", "") == p_card.replace("#", ""):
        heuristics.append("card_number_match")
    if u_serial and p_serial and u_serial == p_serial:
        heuristics.append("serial_match")
    if u_player and p_player and (u_player == p_player or u_player in p_player or p_player in u_player):
        heuristics.append("player_match")
    if u_year and p_year and u_year == p_year:
        heuristics.append("year_match")
    if heuristics:
        reason_parts.append("heuristics=" + ",".join(heuristics))

    # prepare candidates list for output
    out_candidates = []
    for c in shortlist:
        out_candidates.append({
            "id": c["post"].get("id"),
            "title": c["post"].get("title") or c["post"].get("name"),
            "sold_at": c["post"].get("sold_at"),
            "score": round(c["score"], 4),
            "normalized": c["norm"],
            "price": extract_price(c["post"])
        })

    chosen_post = chosen["post"]
    matching_logger.debug("Found 1 candidate most likely: %s", user_card_title)

    matching_logger.debug(
        f'Found 1 candidate most likely: "{user_card_title}"\n'
        f'Candidate: {chosen_post}'
    )
    return {
        "found": found,
        "match": chosen_post,
        "score": round(final_score, 4),
        "reason": "; ".join(reason_parts),
        "candidates": out_candidates
    }

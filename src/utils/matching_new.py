import re
import unicodedata
from typing import Dict, List, Optional

from src.utils.logger import get_logger

matching_logger = get_logger("matching", level="DEBUG")

def normalize_title(title: str) -> str:
    if not title:
        return ""

    s = unicodedata.normalize("NFKC", title)
    s = re.sub(r"[\U00010000-\U0010ffff]", " ", s)
    s = s.strip()
    orig = s[:]
    s = s.lower()

    s = re.sub(r"[,_·•\(\)\[\]:;]", " ", s)

    grading = None
    m = re.search(r"\b(psa|bgs|sgc)\s*([0-9]\.?\d?)\b", s, flags=re.IGNORECASE)
    if m:
        grading = f"{m.group(1).upper()} {m.group(2)}"
        s = s[:m.start()] + s[m.end():]

    hash_token = None
    m = re.search(r"#\s*([a-z0-9\-]+)", s, flags=re.IGNORECASE)
    if m:
        hash_token = m.group(1).strip()
        s = s[:m.start()] + s[m.end():]

    fracs = [(m.group(0), m.start(), m.end(), m.group(1), m.group(2))
             for m in re.finditer(r"\b(\d{1,4})\s*/\s*(\d{1,4})\b", s)]

    year = None
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if m:
        year = m.group(0)
        s = s[:m.start()] + s[m.end():]
    else:
        chosen_frac_idx = None
        for i, (_, start, end, a, b) in enumerate(fracs):
            if len(a) == 4:
                chosen_frac_idx = i; break
        if chosen_frac_idx is None:
            for i, (f, start, end, a, b) in enumerate(fracs):
                left = s[max(0, start-20):start]
                if re.search(r"( set| season| series| collection| pack| fan| ucl| champions| match )", left):
                    chosen_frac_idx = i; break
        if chosen_frac_idx is None and fracs:
            for i, (_, start, end, a, b) in enumerate(fracs):
                try:
                    if int(b) <= 31:
                        chosen_frac_idx = i; break
                except:
                    pass
            if chosen_frac_idx is None:
                chosen_frac_idx = 0
        if chosen_frac_idx is not None:
            frac_str, start, end, a, b = fracs[chosen_frac_idx]
            if len(a) == 2:
                year = "20" + a
            else:
                year = a
            s = s[:start] + s[end:]
            fracs.pop(chosen_frac_idx)

    collector = None
    variant_from_frac = None
    for f, start, end, a, b in fracs:
        left = s[max(0, start-15):start]
        right = s[end:end+15]
        near_end = (end >= len(s) - 6) or (orig.strip().endswith(f))
        denom_large = False
        try:
            denom_large = int(b) > 31
        except:
            pass
        if re.search(r"( set| season| series| collection| pack| fan| ucl| champions )", left) and not denom_large:
            s = s[:start] + s[end:]; continue
        if denom_large or near_end:
            collector = f"#{b}"
            variant_from_frac = a
            s = s[:start] + s[end:]; break
        s = s[:start] + s[end:]

    if not collector and hash_token:
        if re.search(r"\d", hash_token):
            collector = "#" + hash_token.upper()
            hash_token = None
        else:
            hash_token = hash_token.upper()

    m = re.search(r"\b(?:no\.|no|number)\s*([0-9]{1,4})\b", s, flags=re.IGNORECASE)
    if m and not collector:
        collector = f"#{m.group(1)}"
        s = s[:m.start()] + s[m.end():]

    if not collector:
        m = re.search(r"(\d{1,4})\s*$", s)
        if m and (len(s.split()) <= 3 or orig.strip().endswith(m.group(1))):
            collector = f"#{m.group(1)}"
            s = s[:m.start()]

    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()

    noise_end = {"e","en","eng","ebay","lot","cards","card","for","from"}
    while tokens and tokens[-1] in noise_end:
        tokens.pop()

    noise_start = {"the","authentic","official","new","rare","lot"}
    brand = None
    if tokens and tokens[0] not in noise_start:
        brand = tokens.pop(0).capitalize()

    set_name = None
    subset = None
    idx_set = None
    for i, t in enumerate(tokens):
        if t in ("set","series","collection","pack","fan","edition"):
            idx_set = i; break

    if idx_set is not None:
        left_start = max(0, idx_set-3)
        set_tokens = tokens[left_start:idx_set+1]
        right = []
        j = idx_set+1
        while j < len(tokens) and len(right) < 3 and not re.match(r"\d{1,4}$", tokens[j]):
            right.append(tokens[j]); j += 1
        if set_tokens:
            set_name = " ".join(set_tokens)
        if right:
            subset = " ".join(right)
        tokens = tokens[:left_start] + tokens[j:]
    else:
        if tokens:
            take = min(3, len(tokens))
            set_name = " ".join(tokens[:take])
            tokens = tokens[take:]

    if subset and ("set" in subset.lower() or (set_name and subset.lower().strip() == set_name.lower().strip())):
        subset = None

    if set_name and brand and set_name.lower().startswith(brand.lower()):
        parts = set_name.split()
        if len(parts) > 1:
            set_name = " ".join(parts[1:])
        else:
            set_name = None

    card_name = None
    extras = []
    if tokens:
        if len(tokens) <= 2:
            card_name = " ".join(tokens); tokens = []
        else:
            card_name = " ".join(tokens[:2])
            extras.append(" ".join(tokens[2:])); tokens = []

    if variant_from_frac:
        if year and variant_from_frac == str(year)[-2:]:
            variant_from_frac = None
        if collector and variant_from_frac and variant_from_frac == collector.lstrip('#'):
            variant_from_frac = None

    if variant_from_frac:
        extras.insert(0, variant_from_frac)

    if hash_token:
        extras.append(hash_token.upper())
    if grading:
        extras.append(grading)

    parts = []
    if year: parts.append(str(year))
    if brand: parts.append(brand)
    if set_name:
        set_name_clean = " ".join([w.capitalize() if w.isalpha() else w for w in set_name.split()])
        parts.append(set_name_clean)
    if subset:
        if set_name and subset.lower() not in set_name.lower():
            parts[-1] = parts[-1] + " - " + " ".join([w.capitalize() for w in subset.split()])
    if collector: parts.append(collector)
    if card_name: parts.append(" ".join([w.capitalize() if w.isalpha() else w for w in card_name.split()]))
    if extras:
        flat = " ".join([e for e in extras if e and e.strip()])
        if flat:
            parts.append(" ".join([w.capitalize() if w.isalpha() else w for w in flat.split()]))

    normalized = " ".join(parts).strip()
    normalized = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"#\s*([a-z0-9\-]+)", lambda m: "#" + m.group(1).upper(), normalized)

    print('------------------')
    matching_logger.debug(f'Ebay Post: "{title}"')
    matching_logger.debug(f'Normalized: "{normalized}"')
    return normalized


async def get_last_sold_item(
    user_card_title: str, ebay_posts: List[Dict]
) -> Optional[Dict]:

    # Stage 1: Normalize titles
    for post in ebay_posts:
        post["normalised_title"] = normalize_title(post["title"])

    # print(ebay_posts)
    return {}

import re
from bs4 import BeautifulSoup
from typing import Optional, List, Tuple, Any
from src.models.category import CategoryId
from src.models.tcdb import BrowseDropdown
from src.utils.playwright import playwright_get_content
from src.utils.logger import get_logger
from src.utils.scraper import _extract_year_range, upsert_card_sets, upsert_master_cards

logger = get_logger("tcdb_scraper")

ignore_set_keywords = [
    "autograph",
    "magazine",
    "signatures",
    "printing",
    "ink",
    "logo",
    "stamped",
    "tribute",
]

minimum_set_year = 1980


def _normalize_upsert_result(raw: Any) -> Tuple[int, List[dict], List[str]]:
    """
    Normalize different possible return shapes from upsert helpers into a tuple:
      (count, records_list, errors_list)
    Acceptable raw shapes:
      - dict with keys 'count','records','errors'  (preferred)
      - list of records (legacy) -> count = len(list)
      - anything else -> count=0, records=[]
    """
    if isinstance(raw, dict):
        count = int(raw.get("count", 0))
        records = raw.get("records", []) or []
        errors = raw.get("errors", []) or []
        return count, records, errors
    if isinstance(raw, list):
        return len(raw), raw, []
    return 0, [], ["unexpected upsert return type"]


async def tcdb_scrape_handler(
    category_id: CategoryId,
    browse_categories: Optional[List[BrowseDropdown]],
    search_mode: bool = True,
) -> Tuple[int, int]:
    """
    Scrape TCDB for sets and cards.

    Returns:
        Tuple[int, int]: (upserted_set_count, upserted_card_count)
    """
    if not category_id:
        logger.warning(
            f"No TCDB browse categories found for category_id: {category_id}"
        )
        return 0, 0

    if not browse_categories:
        logger.warning(f"No browse_categories provided for category_id: {category_id}")
        return 0, 0

    base_url = "https://www.tcdb.com"

    # totals across all browse categories
    total_upserted_sets = 0
    total_upserted_cards = 0

    # Compile regexes once for performance
    _ws_re = re.compile(r"\s+")
    _hyphen_year_re = re.compile(r"(\b\d{4}) (\d{2}\b)")
    _card_number_re = re.compile(r"\b\d{1,3}/\d{1,3}\b")

    for browse_category in browse_categories:
        # per-browse-category containers
        category_name = CategoryId.get_name(category_id.value)
        brand_keyword = category_name.lower()
        brand_sets: List[str] = []
        sets: List[dict] = []
        cards: List[dict] = []
        upserted_sets: List[dict] = []
        upserted_cards: List[dict] = []

        # dedupe helpers per category
        seen_platform_set_ids = set()
        seen_brand_urls = set()
        processed_card_ids = set()

        if search_mode:
            # Search mode: iterate pages until no more sets found
            for page_number in range(1, 600):
                found_sets_on_page = False
                scrape_url = f"{base_url}/ViewResults.cfm?q={category_name}&Type={browse_category}&PageIndex={page_number}"
                logger.info(f"Scraping TCDB URL: {scrape_url}")
                try:
                    html_content = await playwright_get_content(scrape_url)
                except Exception as e:
                    logger.error(
                        f"Failed to retrieve HTML content from {scrape_url}: {e}"
                    )
                    break

                if not html_content:
                    logger.debug(f"No HTML content for {scrape_url}")
                    break

                soup = BeautifulSoup(html_content, "lxml")

                # Search Mode: Pull all sets
                sets_tags = soup.find_all(
                    "a", href=lambda x: x and "/ViewSet.cfm/sid/" in x
                )
                if sets_tags:
                    found_sets_on_page = True
                    for set_tag in sets_tags:
                        name = set_tag.get_text()
                        name = name.replace("\xa0", " ")
                        name = _ws_re.sub(" ", name).strip()

                        href = set_tag.get("href", "")
                        parts = href.split("/")
                        # ensure expected path structure
                        if len(parts) <= 3:
                            continue
                        platform_set_id = parts[3]
                        if (
                            not platform_set_id
                            or platform_set_id in seen_platform_set_ids
                        ):
                            continue
                        if any(sub in name.lower() for sub in ignore_set_keywords):
                            continue

                        seen_platform_set_ids.add(platform_set_id)

                        years = _extract_year_range(name)

                        if len(years) > 0 and years[0].isdigit():
                            year = int(years[0])
                            if year < minimum_set_year:
                                continue

                        link = f"{base_url}/Checklist.cfm/sid/{platform_set_id}"
                        sets.append(
                            {
                                "category_id": category_id.value,
                                "platform": base_url,
                                "platform_set_id": platform_set_id,
                                "name": name,
                                "years": years,
                                "link": link,
                                "browse_type": browse_category,
                            }
                        )

                if not found_sets_on_page:
                    # no more pages with sets
                    break

            if sets:
                try:
                    raw_upsert = await upsert_card_sets(sets)
                    count, records, errors = _normalize_upsert_result(raw_upsert)
                    if errors:
                        logger.warning(f"upsert_card_sets returned errors: {errors}")
                    if records:
                        upserted_sets.extend(records)
                        total_upserted_sets += count
                except Exception as e:
                    logger.exception(f"Error upserting sets for {brand_keyword}: {e}")

        else:
            # Non-search mode: find brand listing and then sets under each brand
            scrape_url = f"{base_url}/ViewAll.cfm/sp/{browse_category}?Let={category_name[0]}&MODE=Years"
            logger.info(f"Scraping TCDB URL: {scrape_url}")
            try:
                html_content = await playwright_get_content(scrape_url)
            except Exception as e:
                logger.error(f"Failed to retrieve HTML content from {scrape_url}: {e}")
                continue

            if not html_content:
                logger.debug(f"No HTML content for {scrape_url}")
                continue

            soup = BeautifulSoup(html_content, "lxml")

            # Find "Browse by brand:" paragraph, then its sibling ul
            jumper_card_sets = soup.find(
                "p", string=lambda s: s and "Browse by brand" in s
            )
            if jumper_card_sets:
                _ul = jumper_card_sets.find_next_sibling("ul")
                if _ul:
                    for card_set_link in _ul.find_all("li"):
                        a = card_set_link.find("a")
                        if not a or not a.has_attr("href"):
                            continue
                        href = a["href"].strip()
                        text = a.get_text(strip=True).lower()
                        if "ViewAll.cfm/" in href and brand_keyword in text:
                            full_url = f"{base_url}{href}"
                            if full_url not in seen_brand_urls:
                                seen_brand_urls.add(full_url)
                                brand_sets.append(full_url)

            if not brand_sets:
                # nothing found for this browse_category
                continue

            # For each brand set page, collect individual sets
            for brand_set_link in brand_sets:
                try:
                    html_content = await playwright_get_content(brand_set_link)
                except Exception as e:
                    logger.warning(
                        f"Failed to retrieve brand page {brand_set_link}, skipping: {e}"
                    )
                    continue

                if not html_content:
                    logger.debug(f"No HTML content for brand page {brand_set_link}")
                    continue

                soup_sets = BeautifulSoup(html_content, "lxml")
                jumper_brand_sets = soup_sets.find(
                    "h3", string=lambda s: s and "Select a set" in s
                )
                if not jumper_brand_sets:
                    continue
                _ul = jumper_brand_sets.find_next_sibling("ul")
                if not _ul:
                    continue

                for set_link in _ul.find_all("li"):
                    a = set_link.find("a")
                    if not a or not a.has_attr("href"):
                        continue
                    href = a["href"].strip()
                    parts = href.split("/")
                    if len(parts) <= 3:
                        continue
                    name = a.get_text()
                    name = name.replace("\xa0", " ")
                    name = _ws_re.sub(" ", name).strip()
                    if any(sub in name.lower() for sub in ignore_set_keywords):
                        continue
                    platform_set_id = parts[3]
                    full_url = f"{base_url}{href}"
                    # Avoid duplicates by platform_set_id
                    if platform_set_id in seen_platform_set_ids:
                        continue
                    seen_platform_set_ids.add(platform_set_id)

                    years = _extract_year_range(name)

                    if len(years) > 0 and years[0].isdigit():
                        year = int(years[0])
                        if year < minimum_set_year:
                            continue

                    link = f"{base_url}/Checklist.cfm/sid/{platform_set_id}"
                    sets.append(
                        {
                            "category_id": category_id.value,
                            "platform": base_url,
                            "platform_set_id": platform_set_id,
                            "name": name,
                            "years": years,
                            "link": link,
                            "browse_type": browse_category,
                        }
                    )

            if sets:
                try:
                    raw_upsert = await upsert_card_sets(sets)
                    count, records, errors = _normalize_upsert_result(raw_upsert)
                    if errors:
                        logger.warning(f"upsert_card_sets returned errors: {errors}")
                    if records:
                        upserted_sets.extend(records)
                        total_upserted_sets += count
                except Exception as e:
                    logger.exception(f"Error upserting sets for {brand_keyword}: {e}")

        # Last Step: Pull all cards for upserted sets
        if upserted_sets:
            set_last_number = 1
            set_last_name = ""
            for card_set in upserted_sets:
                # Some defensive checks
                card_set_link = card_set.get("link")
                if not card_set_link:
                    continue

                # iterate card pages; break when no rows found
                for page_number in range(1, 30):
                    if card_set["name"] != set_last_name:
                        print(
                            "--------------------------------------------------------------------------------"
                        )
                        logger.info(
                            f"[{browse_category}] Processing set \"{card_set['name']}\" - {set_last_number}/{len(upserted_sets)}"
                        )
                        set_last_name = card_set["name"]
                        set_last_number += 1
                    page_url = f"{card_set_link}?PageIndex={page_number}"
                    try:
                        html_content = await playwright_get_content(page_url)
                    except Exception as e:
                        logger.debug(
                            f"No content for {page_url}, breaking card pages loop: {e}"
                        )
                        break

                    if not html_content:
                        break

                    soup_cards = BeautifulSoup(html_content, "lxml")

                    tables = soup_cards.find_all("table")
                    found_rows_on_page = False

                    for table in tables:
                        # Only rows with specific bgcolor (as original logic)
                        rows = table.find_all("tr", {"bgcolor": ["#F7F9F9", "#EAEEEE"]})
                        if not rows:
                            continue

                        found_rows_on_page = True
                        for row in rows:
                            tds = row.find_all("td")
                            if not tds:
                                continue
                            for td in tds:
                                a_tags = td.find_all("a", href=True)
                                img = td.find("img")
                                if not a_tags or not img:
                                    continue

                                # find the correct anchor(s) that point to ViewCard
                                for a in a_tags:
                                    href = a.get("href", "")
                                    if not href.startswith("/ViewCard.cfm"):
                                        continue
                                    parts = href.split("/")
                                    platform_card_id = (
                                        parts[5] if len(parts) > 5 else None
                                    )
                                    if (
                                        not platform_card_id
                                        or platform_card_id in processed_card_ids
                                    ):
                                        continue
                                    processed_card_ids.add(platform_card_id)

                                    # Process card name parsing
                                    parsed_name = (
                                        "/".join(parts[6:])
                                        .replace("-", " ")
                                        .split("?")[0]
                                    )
                                    parsed_name = _hyphen_year_re.sub(
                                        r"\1-\2", parsed_name
                                    )

                                    # try to find card number in the part after first space, fallback to whole name
                                    after_space = (
                                        parsed_name.partition(" ")[2] or parsed_name
                                    )
                                    match_card_number = _card_number_re.search(
                                        after_space
                                    )

                                    # Construct image URL if possible
                                    img_src = (
                                        img.get("data-original") or img.get("src") or ""
                                    )
                                    img_url = (
                                        img_src.replace("/Thumbs/", "/Cards/")
                                        .replace("_", "-")
                                        .replace("Thumb.jpg", "Fr.jpg")
                                    )
                                    full_img_url = (
                                        f"{base_url}{img_url}"
                                        if (
                                            len(parts) > 3
                                            and str(parts[3]) in img_url
                                            and str(platform_card_id) in img_url
                                        )
                                        else None
                                    )

                                    card_href = href.split("?")[0]
                                    cards.append(
                                        {
                                            "category_id": category_id.value,
                                            "set_id": card_set.get("id"),
                                            "platform": base_url,
                                            "platform_card_id": platform_card_id,
                                            "name": parsed_name,
                                            "card_number": (
                                                match_card_number.group()
                                                if match_card_number
                                                else None
                                            ),
                                            "years": card_set.get("years"),
                                            "canonical_image_url": full_img_url,
                                            "link": f"{base_url}{card_href}",
                                        }
                                    )

                    if not found_rows_on_page:
                        # no more rows on subsequent pages
                        break

            # upsert cards if any
            if cards:
                try:
                    raw_upsert = await upsert_master_cards(cards)
                    count, records, errors = _normalize_upsert_result(raw_upsert)
                    if errors:
                        logger.warning(f"upsert_master_cards returned errors: {errors}")
                    if records:
                        upserted_cards.extend(records)
                        total_upserted_cards += count
                except Exception as e:
                    logger.exception(f"Error upserting cards for {brand_keyword}: {e}")

        logger.info(
            f"------------------------> Finished upserting {len(upserted_sets)} unique '{brand_keyword}' sets in {browse_category}."
        )
        logger.info(
            f"------------------------> Finished upserting {len(upserted_cards)} unique '{brand_keyword}' cards in {browse_category}."
        )

    logger.info(f"Total upserted sets: {total_upserted_sets}")
    logger.info(f"Total upserted cards: {total_upserted_cards}")
    return total_upserted_sets, total_upserted_cards

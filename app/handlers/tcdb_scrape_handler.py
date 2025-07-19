import re
from typing import List
from bs4 import BeautifulSoup, Tag

from app.models.tcdb import ScrapeTarget
from app.utils.tcdb_fetcher import get_random_headers, fetch_html
from app.services.supabase import supabase

base_target_url = "https://www.tcdb.com"

ignored_title_words = ["sticker", "japanese"]

sets = {
    "Topps": {
        "urls": [
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Chrome",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Museum%20Collection",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Finest%20Flashbacks",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Merlin%20Heritage%2095",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Premier%20Club",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Premier%20Gold",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Premier%20Stars"
        ],
        "sets": [
            # "2023",
            # "2024",
            # "UEFA",
            # "Merlin",
        ]
    },
    "Panini": {
        "urls": [
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Prizm",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Select",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Obsidian",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Immaculate%20Collection",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Chronicles",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20UEFA%20Champions%20League",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20UEFA%20Euro",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Revolution",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20National%20Treasures",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Eminence",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Mosaic",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Eminence",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Eminence",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Eminence",
            # "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Eminence",
        ],
        "sets": [
        ]
    },
    "Pokemon": {
        "urls": [
            "https://www.tcdb.com/ViewAll.cfm/sp/Gaming/brand/Pokemon",
            "https://www.tcdb.com/ViewAll.cfm/sp/Gaming/brand/Pokemon%20Scarlet%20&%20Violet%20Shrouded%20Fable%20(Italian)",
            "https://www.tcdb.com/ViewAll.cfm/sp/Gaming/brand/Pokemon%20Sword%20&%20Shield%20Brilliant%20Stars",
            "https://www.tcdb.com/ViewAll.cfm/sp/Gaming/brand/Pokemon%20Sword%20&%20Shield%20Darkness%20Ablaze",
            "https://www.tcdb.com/ViewAll.cfm/sp/Gaming/brand/Pokemon%20Sword%20&%20Shield%20Lost%20Origin",
            "https://www.tcdb.com/ViewAll.cfm/sp/Gaming/brand/Pokemon%20Sword%20&%20Shield%20Rebel%20Clash"
        ],
        "sets": [
            "Surging Spark",
            "Prismatic Evolutions",
            "Journey Together",
            "Stellar Crown",
            "Shrouded Fable",
            "Destined Rivals",
            "2024 Trick or Trade",
            "2023 Trick or Trade",
            "Destined Rivals",
            "Twilight Masquerade",
            "Temporal Forces",
            "Paldean Fates",
            "Paradox Rift",
            "151",
            "Obsidian Flames",
            "Paldea Evolved",
            "Energize",
            "2023 Scarlet & Violet",
            "Crown Zenith",
            "Sword & Shield Silver",
            "Sword & Shield Celebrations",
            "Sword & Shield Evolving Skies",
            "Sword & Shield Chilling Reign",
            "Sword & Shield Battle Styles",
            "Sword & Shield Shining Fates",
            "McDonald's 25th Anniversary",
            "Sword & Shield Vivid Voltage",
            "Sword & Shield Champion's Path",
            "Pokemon Futsal Promos",
            "Sword & Shield",
        ]
    }
}

async def discover_sets(target: ScrapeTarget) -> List[dict]:
    selected_category = sets[target]
    category_index = list(ScrapeTarget).index(target)
    result = []
    for url in selected_category["urls"]:
        html_text = await fetch_html(url, headers=get_random_headers())
        soup = BeautifulSoup(html_text, "lxml")
        blocks = soup.find_all("div", class_="block1")
        block = blocks[1] if len(blocks) > 1 and isinstance(blocks[1], Tag) else None
        if block:
            for ul in block.find_all("ul"):
                for li in ul.find_all("li", recursive=False):
                    a = li.find("a")
                    if a:
                        a_text = a.get_text(strip=True)
                        if any(word in a_text.lower() for word in ignored_title_words):
                            continue
                        for set_name in selected_category["sets"]:
                            words = set_name.lower().split()
                            if all(word in a_text.lower() for word in words):
                                # Extract all year patterns (single or range)
                                # Match 4-digit year or 4-digit year followed by - and 2 or 4 digits
                                year_matches = re.findall(r'(18|19|20)\d{2}(?:-(\d{2,4}))?', a_text)
                                years = []
                                for match in year_matches:
                                    start_year_str = match[0] + a_text[a_text.find(match[0])+2:a_text.find(match[0])+4]
                                    start_year = int(start_year_str)
                                    if match[1]:
                                        end_part = match[1]
                                        if len(end_part) == 2:
                                            end_year = int(str(start_year)[:2] + end_part)
                                        else:
                                            end_year = int(end_part)
                                        # Create full range array
                                        years.extend(range(start_year, end_year + 1))
                                    else:
                                        years.append(start_year)
                                years = [int(y) for y in years] if years else None
                                if years:
                                    seen = set()
                                    years = [x for x in years if not (x in seen or seen.add(x))]
                                # Extract sid from link
                                link_val = a.get("href", "")
                                set_id_match = re.search(r"/sid/(\d+)/", link_val)
                                platform_set_id = set_id_match.group(1) if set_id_match else None
                                result.append({
                                    "category_id": category_index + 1,
                                    "platform": base_target_url,
                                    "platform_set_id": platform_set_id,
                                    "name": a_text,
                                    "years": years,
                                    "link": f"{base_target_url}/Checklist.cfm/sid/{platform_set_id}" if platform_set_id else None,
                                })
                                break
    if result:
        upserts = []
        for set_obj in result:
            upserts.append(set_obj)
        try:
            res = (supabase.table("card_sets").upsert(upserts, on_conflict="platform_set_id").execute())
            print(f"Upsert card_sets {len(res.data)} rows")
        except Exception as e:
            print(f"Supabase upsert card_sets error: {e}")
    return result

# -----------------------------------------------------------------------------------------------------------------------

async def discover_cards(platform_set_id: int) -> List[dict]:
    response = (supabase.table("card_sets").select("id, category_id").eq("platform_set_id", platform_set_id).execute())
    set_id = response.data[0]["id"]
    category_id = response.data[0]["category_id"]
    url = f"{base_target_url}/Checklist.cfm/sid/{platform_set_id}"
    result = []
    existing_cards = set()
    for page_number in range(1, 5):
        html_text = await fetch_html(f"{url}?PageIndex={page_number}", headers=get_random_headers())
        soup = BeautifulSoup(html_text, "lxml")
        tables = soup.find_all("table")
        for table in tables:
            target_colors = ["#F7F9F9", "#EAEEEE"]
            rows = table.find_all("tr", attrs={"bgcolor": lambda c, colors=target_colors: c in colors})
            for row in rows:
                tds = row.find_all("td")
                for td in tds:
                    a_tags = td.find_all("a", href=True)
                    img = td.find("img") if td.find("img") else None
                    if (a_tags and img):
                        for a in a_tags:
                            href = a.get("href").split("?")[0]
                            if href.startswith("/ViewCard.cfm"):
                                _, _, _, platform_set_id, _, platform_card_id, *name_arr = href.split("/")
                                img_url = img.get("data-original").replace("/Thumbs/", "/Cards/").replace("_", "-").replace("Thumb.jpg", "Fr.jpg")
                                full_img_url = f"{base_target_url}{img_url}" if (str(platform_set_id) in img_url and str(platform_card_id) in img_url) else None
                                if platform_card_id in existing_cards:
                                    continue
                                existing_cards.add(platform_card_id)
                                parsed_name = "/".join(name_arr).replace("-", " ")
                                match_year = re.match(r'^(\d{4})\b', parsed_name)
                                match_card_number = re.search(r'\b\d{1,3}/\d{1,3}\b', parsed_name.partition(' ')[2])
                                result.append({
                                    "category_id": category_id,
                                    "set_id": set_id,
                                    "platform": base_target_url,
                                    "platform_card_id": platform_card_id,
                                    "name": parsed_name,
                                    "card_number": match_card_number.group() if match_card_number else None,
                                    "year":  match_year.group(1) if match_year else None,
                                    "link": f"{base_target_url}{href}",
                                    "canonical_image_url": full_img_url
                                })
                                break
    if result:
        upserts = []
        for card_obj in result:
            upserts.append(card_obj)
        try:
            res = (supabase.table("master_cards").upsert(upserts, on_conflict="platform_card_id").execute())
            print(f"Upsert master_cards {len(res.data)} rows")
        except Exception as e:
            print(f"Supabase upsert master_cards error: {e}")
    print(f"Found {len(result)} cards")
    return result

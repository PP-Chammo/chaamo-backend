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
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Chrome",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Museum%20Collection",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Finest%20Flashbacks",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Merlin%20Heritage%2095",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Premier%20Club",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Premier%20Gold",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Topps%20Premier%20Stars"
        ],
        "sets": [
            "2023",
            "2024",
            "UEFA",
            "Merlin",
        ]
    },
    "Panini": {
        "urls": [
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Prizm",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Select",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Obsidian",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Immaculate%20Collection",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Chronicles",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20UEFA%20Champions%20League",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20UEFA%20Euro",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Revolution",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20National%20Treasures",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Eminence",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Mosaic",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Eminence",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Eminence",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Eminence",
            "https://www.tcdb.com/ViewAll.cfm/sp/Soccer/brand/Panini%20Eminence",
        ],
        "sets": [
            "Prizm",
            "Select",
            "Obsidian",
            "Immaculate",
            "Chronicles",
            "UEFA",
            "Revolution",
            "National Treasures",
            
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
            "Trick or Trade",
            "Twilight Masquerade",
            "Temporal Forces",
            "Paldean Fates",
            "Paradox Rift",
            "151",
            "Obsidian Flames",
            "Paldea Evolved",
            "Energize",
            "Scarlet & Violet",
            "Sword & Shield",
            "Lost",
            "Origin",
            "Astral Radiance"
        ]
    }
}

async def discover_sets(target: ScrapeTarget) -> List[dict]:
    selected_category = sets[target]
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
                                        years.extend([start_year, end_year])
                                    else:
                                        years.append(start_year)
                                years = [int(y) for y in years] if years else None
                                if years:
                                    seen = set()
                                    years = [x for x in years if not (x in seen or seen.add(x))]
                                # Extract sid from link
                                link_val = a.get("href", "")
                                set_id_match = re.search(r"/sid/(\d+)/", link_val)
                                set_id = set_id_match.group(1) if set_id_match else None
                                result.append({
                                    "name": a_text,
                                    "years": years,
                                    "set_id": set_id,
                                    "link": f"{base_target_url}/Checklist.cfm/sid/{set_id}" if set_id else None,
                                })
                                break
    if result:
        category_id = list(ScrapeTarget).index(target)
        upserts = []
        for set_obj in result:
            upserts.append({
                'category_id': category_id + 1,
                'name': set_obj['name'],
                'years': set_obj['years'],
                'set_id': set_obj['set_id'],
                'link': set_obj['link'],
            })
        try:
            response = (supabase.table('card_sets').upsert(upserts, on_conflict='set_id').execute())
            print(response)
        except Exception as e:
            print(f"Supabase upsert error: {e}")
    return result

# -----------------------------------------------------------------------------------------------------------------------

async def discover_cards(set_id: int) -> List[dict]:
    url = f"{base_target_url}/Checklist.cfm/sid/{set_id}"
    result = []
    html_text = await fetch_html(url, headers=get_random_headers())
    soup = BeautifulSoup(html_text, "lxml")
    tables = soup.find_all("table")
    if tables:
        table = tables[-1] if isinstance(tables[-1], Tag) else None
        if table:
            for row in table.find_all("tr"):
                first_td = row.find('td')
                if first_td:
                    print("-----------------------------")
                    a_tags = first_td.find_all('a', href=True)
                    img = first_td.find('img')
                    for a in a_tags:
                        href = a.get("href").split('?')[0]
                        img_url = img.get("data-original").replace("/Thumbs/", "/Cards/").replace("_", "-").replace("RepThumb.jpg", "RepFr.jpg")
                        if href.startswith("/ViewCard.cfm"):
                            _, _, _, set_id, _, card_id, *rest_name = href.split("/")
                            parsed_name = "/".join(rest_name).replace("-", " ")
                            result.append({
                                "name": parsed_name,
                                "set_id": set_id,
                                "card_id": card_id,
                                "link": f"{href}",
                                "image_url": f"{base_target_url}{img_url}"
                            })
                            break
    return result

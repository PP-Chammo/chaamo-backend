from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class SetData(BaseModel):
    category_id: int
    platform: str
    platform_set_id: str
    name: str
    years: Optional[List[int]] = None
    link: str

class CardData(BaseModel):
    category_id: int
    set_id: str
    platform: str
    platform_card_id: str
    name: str
    card_number: Optional[str] = None
    year: Optional[int] = None
    link: str
    canonical_image_url: Optional[str] = None

class ScrapeTarget(str, Enum):
    topps = "Topps"
    panini = "Panini"
    futera = "Futera"
    pokemon = "Pokemon"
    dc = "DC"
    fortnite = "Fortnite"
    marvel = "Marvel"
    garbage_pail_kids = "Garbage Pail Kids"
    digimon = "Digimon"
    poker = "Poker"
    wrestling = "Wrestling"
    yu_gi_oh = "Yu-Gi-Oh!"

class DiscoverSetsRequest(BaseModel):
    targets: List[ScrapeTarget] = Field(..., description="A list of categories/brands to discover sets for.")

class SetsResponse(BaseModel):
    sets: List[SetData]

class CardsResponse(BaseModel):
    cards: List[CardData]

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class CardData(BaseModel):
    name: str
    set_id: int
    card_id: int
    card_number: Optional[str] = None
    # manufacturer: Optional[str] = None
    # year: Optional[int] = None
    # team: Optional[str] = None
    link: str
    image_url: Optional[str] = None

class SetInfo(BaseModel):
    name: str
    set_id: Optional[str] = None
    link: str
    years: Optional[List[int]] = None

# Enum for selectable categories/brands
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
    sets: List[SetInfo]

class CardsResponse(BaseModel):
    cards: List[CardData]

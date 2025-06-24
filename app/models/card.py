from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional


class Region(str, Enum):
    us = "us"
    uk = "uk"

class SearchPage(str, Enum):
    one = "1"
    two = "2"
    three = "3"
    four = "4"
    five = "5"


class CardScrapeRequest(BaseModel):
    query: str = Field(
        ...,
        description="Keyword or name of the card to search for on eBay.",
        json_schema_extra={"format": "textarea"}
    )
    region: Region


class SoldCard(BaseModel):
    """
    Represents a single sold card instance, complete with its image URL.
    """
    item_id: str
    sold_date: str
    price: float
    currency: str
    link_url: str


class FinalGroupedCard(BaseModel):
    """
    Represents a grouped card with the new structure.
    """
    title: str
    image_url: str
    sold_cards: List[SoldCard]


class ScrapeResponse(BaseModel):
    """
    Top-level response model, updated to use the new grouped structure.
    """
    result: List[FinalGroupedCard]

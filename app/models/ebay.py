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


class EbaySold(BaseModel):
    """
    Represents a sold card from eBay.
    """
    id: str
    master_card_id: str
    source_image_url: str
    region: str
    sold_at: str
    price: float
    currency: str
    condition: str
    grading_company: Optional[str]
    grade: Optional[str]
    source_listing_url: str


class EbaySoldsResponse(BaseModel):
    """
    Response model for eBay card listings.
    """
    solds: List[EbaySold]

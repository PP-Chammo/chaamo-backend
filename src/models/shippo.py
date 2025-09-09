from pydantic import BaseModel, Field
from typing import Optional

class Address(BaseModel):
    name: str
    street1: str
    city: str
    zip: str
    country: str
    email: Optional[str] = None
    phone: Optional[str] = None

class RateRequest(BaseModel):
    listing_id: str = Field(..., example="listing_abc123")
    seller_id: str = Field(..., example="seller_abc123")
    buyer_id: str = Field(..., example="buyer_xyz789")
    delivery_option: Optional[str] = Field("home", example="home")  # "home" | "pickup" default to "home" for now
    insurance: Optional[bool] = False
    insurance_amount: Optional[float] = 0.0
    currency: Optional[str] = "GBP"

class RateOption(BaseModel):
    id: str
    value: str
    label: str
    courier: str
    service: str
    amount: float
    currency: str
    estimated_days: Optional[int] = None
    shippo_rate_id: str

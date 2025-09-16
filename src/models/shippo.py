from pydantic import BaseModel, Field
from typing import List, Optional


class Address(BaseModel):
    name: str
    street1: str
    city: str
    zip: str
    country: str
    email: Optional[str] = None
    phone: Optional[str] = None

class RateOption(BaseModel):
    id: str
    service: str
    courier: str
    amount: float
    currency: str
    estimated_days: Optional[int] = None

class RateResponse(BaseModel):
    shipment_id: str
    rates: List[RateOption]


class TransactionPayload(BaseModel):
    listing_id: str
    buyer_id: str
    selected_rate_id: str
    selected_rate_amount: float = 0.0
    selected_rate_currency: str
    insurance: Optional[bool] = False
    insurance_amount: Optional[float] = 0.0
    insurance_currency: Optional[str] = "USD"
    redirect: Optional[str] = None


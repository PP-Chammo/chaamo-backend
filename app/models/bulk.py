from typing import List
from pydantic import BaseModel

class BulkData(BaseModel):
    set_id: str
    set_name: str
    cards_scraped: float

class BulkResponse(BaseModel):
    status: str
    data: List[BulkData]



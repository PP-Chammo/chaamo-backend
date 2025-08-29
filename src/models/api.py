"""
Pydantic models for API request/response validation and documentation.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from src.models.ebay import Region
from src.models.category import CategoryId


# Request Models
class ScrapeRequest(BaseModel):
    """Request model for starting eBay scraping worker."""

    region: Region = Field(Region.uk, description="Choose a region from 'us' or 'uk'")
    category_id: CategoryId = Field(..., description="Category filter - REQUIRED")
    query: Optional[str] = Field(
        None, description="Search keyword (e.g., '2023 Topps Merlin Lamine Yamal')"
    )
    user_card_id: Optional[str] = Field(
        None, description="Alternative to query: Use existing user card ID"
    )
    max_pages: int = Field(
        50, description="Max pages to scrape (lower is faster)", ge=1, le=100
    )

    class Config:
        json_schema_extra = {
            "example": {
                "region": "us",
                "category_id": 2,
                "query": "2023 Topps Merlin Lamine Yamal",
                "max_pages": 10,
            }
        }


# Response Models
class WorkerTaskResponse(BaseModel):
    """Response model for a single worker task."""

    id: str = Field(..., description="Unique task identifier")
    status: str = Field(
        ..., description="Task status (pending/running/completed/failed)"
    )
    query: str = Field(..., description="Search query used")
    region: str = Field(..., description="Region scraped")
    category_id: int = Field(..., description="Category ID used")
    user_card_id: Optional[str] = Field(None, description="User card ID if used")
    max_pages: int = Field(..., description="Maximum pages to scrape")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Task completion timestamp"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    results_count: int = Field(0, description="Number of results found")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "task_abc123def456",
                "status": "completed",
                "query": "pokemon card",
                "region": "us",
                "category_id": 4,
                "max_pages": 10,
                "created_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:35:00Z",
                "results_count": 245,
            }
        }


class WorkerListResponse(BaseModel):
    """Response model for listing all worker tasks."""

    total_tasks: int = Field(..., description="Total number of tasks")
    tasks: List[WorkerTaskResponse] = Field(..., description="List of worker tasks")

    class Config:
        json_schema_extra = {
            "example": {
                "total_tasks": 2,
                "tasks": [
                    {
                        "id": "task_abc123",
                        "status": "completed",
                        "query": "pokemon card",
                        "region": "us",
                        "category_id": 4,
                        "results_count": 245,
                    }
                ],
            }
        }


class ScrapeStartResponse(BaseModel):
    """Response model for starting a scrape worker."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Initial task status")
    message: str = Field(..., description="Success message")
    query: str = Field(..., description="Search query being used")
    region: str = Field(..., description="Region being scraped")
    category_id: int = Field(..., description="Category ID being used")
    estimated_duration: str = Field(
        "2-5 minutes", description="Estimated completion time"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_abc123def456",
                "status": "pending",
                "message": "eBay scraping worker started successfully",
                "query": "pokemon card",
                "region": "us",
                "category_id": 4,
                "estimated_duration": "2-5 minutes",
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""

    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(
        None, description="Error code for programmatic handling"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Error timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Task not found",
                "error_code": "TASK_NOT_FOUND",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


# PayPal Models
class PayPalCheckoutRequest(BaseModel):
    """Request model for PayPal checkout."""

    amount: str = Field("1.00", description="Payment amount as string")
    currency: str = Field("USD", description="Currency code")
    redirect: str = Field(..., description="App redirect deep link")

    class Config:
        json_schema_extra = {
            "example": {
                "amount": "9.99",
                "currency": "USD",
                "redirect": "chaamo://payment-success",
            }
        }

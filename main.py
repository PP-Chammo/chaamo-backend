from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import scrape_cards as api_v1

# Create an instance of the FastAPI application
app = FastAPI(
    title="eBay Card Scraper API",
    description="API for scraping data of sold cards on eBay with grouping.",
    version="1.0.0"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow origins in the list
    allow_credentials=True, # Allow cookies (if needed)
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)


# Include the router from v1 with a prefix
app.include_router(api_v1.router, prefix="/api/v1", tags=["Scraping v1"])

# Root endpoint for health check
@app.get("/", tags=["Root"])
def read_root():
    return {"status": "ok", "message": "Welcome to the eBay Scraper API!"}

# To run this application, use the following command:
# uvicorn main:app --reload


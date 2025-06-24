from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import scrape_cards as api_v1

# Create an instance of the FastAPI application
app = FastAPI(
    title="eBay Card Scraper API",
    description="API for scraping data of sold cards on eBay with grouping.",
    version="1.0.0"
)

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Izinkan origin yang ada di daftar
    allow_credentials=True, # Izinkan cookies (jika dibutuhkan)
    allow_methods=["*"],    # Izinkan semua metode (GET, POST, etc.)
    allow_headers=["*"],    # Izinkan semua header
)


# Include the router from v1 with a prefix
app.include_router(api_v1.router, prefix="/api/v1", tags=["Scraping v1"])

# Root endpoint for health check
@app.get("/", tags=["Root"])
def read_root():
    return {"status": "ok", "message": "Welcome to the eBay Scraper API!"}

# To run this application, use the following command:
# uvicorn main:app --reload


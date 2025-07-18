import subprocess
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import router as api_v1
from app.api.v2 import router as api_v2

# Create an instance of the FastAPI application
app = FastAPI(
    title="Chaamo Card Scraper API",
    description="API for scraping data of sold cards on eBay and tcdb.com with grouping.",
    version="1.0.0"
)

@app.on_event("startup")
async def ensure_playwright_installed():
    try:
        import playwright
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
    # Always ensure browser binaries are present
    subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow origins in the list
    allow_credentials=True, # Allow cookies (if needed)
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)


# Include the router from v1 with a prefix
app.include_router(api_v1, prefix="/api/v1", tags=["Scraping v1"])
app.include_router(api_v2, prefix="/api/v2", tags=["Scraping v2"])

# Root endpoint for health check
@app.get("/", tags=["Root"])
def read_root():
    return {"status": "ok", "message": "Welcome to the Chaamo Scraper API!"}

# To run this application, use the following command:
# uvicorn main:app --reload


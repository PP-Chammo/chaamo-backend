import subprocess
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ebay import router as api_ebay
from app.api.tcdb import router as api_tcdb
from app.api.bulk import router as api_bulk
from app.utils.compare_image import setup_models

app = FastAPI(
    title="Chaamo Card Scraper API",
    description="API for scraping data of sold cards on eBay and tcdb.com with grouping.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_tasks():
    try:
        import playwright
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
    subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)

    setup_models()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow origins in the list
    allow_credentials=True, # Allow cookies (if needed)
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)


# Include the router from v1 with a prefix
app.include_router(api_ebay, prefix="/api/ebay", tags=["EBAY Endpoint"])
app.include_router(api_tcdb, prefix="/api/tcdb", tags=["TCDB Endpoint"])
app.include_router(api_bulk, prefix="/api/bulk", tags=["Bulk Endpoint"])

# Root endpoint for health check
@app.get("/", tags=["Root"])
def read_root():
    return {"status": "ok", "message": "Welcome to the Chaamo Scraper API!"}

# To run this application, use the following command:
# uvicorn main:app --reload


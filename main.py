from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1 import router as v1_endpoint

app = FastAPI(
    title="API",
    description="Chaamo API",
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


app.include_router(v1_endpoint, prefix="/api/v1", tags=["Scraping v1"])

@app.get("/", tags=["Root"])
def read_root():
    return {"status": "ok", "message": "Welcome Chaamo API!"}

# debug purpose, because in render.com swagger cant be load
try:
    app.openapi()
    print("✅ OpenAPI schema valid 🎉")
except Exception as e:
    print("❌ Failed to generate OpenAPI schema:", e)

# To run this application for development:
# uvicorn main:app --reload


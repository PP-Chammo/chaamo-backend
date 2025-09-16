from fastapi import APIRouter
from . import endpoint

router = APIRouter()
router.include_router(endpoint.router)
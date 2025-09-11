from fastapi import APIRouter
from . import endpoint
from . import paypal

router = APIRouter()
router.include_router(endpoint.router)
router.include_router(paypal.router, prefix="/paypal")
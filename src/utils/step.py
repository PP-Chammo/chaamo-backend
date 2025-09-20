from contextlib import asynccontextmanager
import traceback

from src.utils.logger import step_logger
from fastapi import HTTPException

@asynccontextmanager
async def step(label: str, logger=step_logger):
    try:
        if logger:
            logger.info(f"------------- Step: {label} -------------")
        yield
    except Exception as e:
        if logger:
            logger.error(f"❌ Error in step -------'{label}'------- : {type(e).__name__} - {e}")
            logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed in step: {label}")

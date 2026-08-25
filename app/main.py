import logging

from fastapi import FastAPI

from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="elevation-mask",
    version="0.1.0",
    description="CAD elevation segmentation API: detect parts, paint masks, measure area.",
)
app.include_router(router)

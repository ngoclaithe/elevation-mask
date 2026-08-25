from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="elevation-mask",
    version="0.1.0",
    description="CAD elevation segmentation API: detect parts, paint masks, measure area.",
)
app.include_router(router)

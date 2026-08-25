import logging
import threading

from fastapi import FastAPI

from app.api.routes import router
from app.settings import settings

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


@app.on_event("startup")
def _warmup() -> None:
    def load() -> None:
        if settings.enable_florence:
            from app.pipeline.florence import _load

            _load()
        if settings.enable_sam:
            from app.pipeline.sam_seg import _load as load_sam

            load_sam()

    threading.Thread(target=load, daemon=True).start()

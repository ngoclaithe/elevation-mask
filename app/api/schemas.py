from typing import Literal

from pydantic import BaseModel, Field


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error"]
    created_at: str | None = None
    error: str | None = None
    areas: dict | None = None
    meta: dict | None = None
    trace: list[dict] | None = None
    overlay_url: str | None = None
    mask_url: str | None = None
    source_url: str | None = None
    masks_url: str | None = None


class SegmentOptions(BaseModel):
    max_iters: int = Field(default=6, ge=1, le=12)
    scale_mm_per_px: float | None = Field(default=None, gt=0)

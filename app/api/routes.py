from __future__ import annotations

import logging
import threading
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.schemas import JobStatus
from app.jobs import store
from app.pipeline.agent import run_agent
from app.pipeline.classes import CLASSES
from app.settings import settings

log = logging.getLogger(__name__)
router = APIRouter()
_run_lock = threading.Lock()


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return img


def _run_job(
    job_id: str,
    bgr: np.ndarray,
    max_iters: int,
    enable_florence: bool | None,
    enable_sam: bool | None,
    enable_vl_critic: bool | None,
    scale_mm_per_px: float | None,
) -> None:
    store.write_meta(job_id, {"status": "running"})
    try:
        with _run_lock:
            if enable_florence is not None:
                settings.enable_florence = enable_florence
            if enable_sam is not None:
                settings.enable_sam = enable_sam
            if enable_vl_critic is not None:
                settings.enable_vl_critic = enable_vl_critic
            result = run_agent(bgr, max_iters=max_iters)
        if scale_mm_per_px:
            from app.pipeline.area import compute_areas
            from app.pipeline.geometry import perceive

            env = perceive(bgr).envelope
            result["areas"] = compute_areas(result["masks"], env, scale_mm_per_px)
        store.save_image(job_id, "source.png", bgr)
        store.save_image(job_id, "overlay.png", result["overlay"])
        store.save_image(job_id, "mask.png", result["mask_layer"])
        store.save_masks(job_id, result["masks"])
        store.write_meta(
            job_id,
            {
                "status": "done",
                "areas": result["areas"],
                "meta": result["meta"],
                "trace": result["trace"],
                "source_url": f"/v1/jobs/{job_id}/source",
                "overlay_url": f"/v1/jobs/{job_id}/overlay",
                "mask_url": f"/v1/jobs/{job_id}/mask",
                "masks_url": f"/v1/jobs/{job_id}/masks",
            },
        )
    except Exception as exc:
        log.exception("job %s failed", job_id)
        store.write_meta(job_id, {"status": "error", "error": str(exc)})


@router.get("/health")
def health() -> dict:
    return {"ok": True, "service": "elevation-mask"}


@router.get("/v1/classes")
def list_classes() -> dict:
    return {
        name: {"bgr": list(cls.bgr), "priority": cls.priority, "count_area": cls.count_area}
        for name, cls in CLASSES.items()
    }


@router.post("/v1/segment", status_code=202)
async def segment(
    image: Annotated[UploadFile, File()],
    max_iters: Annotated[int, Form()] = 6,
    enable_florence: Annotated[bool | None, Form()] = None,
    enable_sam: Annotated[bool | None, Form()] = None,
    enable_vl_critic: Annotated[bool | None, Form()] = None,
    scale_mm_per_px: Annotated[float | None, Form()] = None,
) -> dict:
    data = await image.read()
    if not data:
        raise HTTPException(400, "Empty file")
    bgr = _decode(data)
    job_id = store.create()
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, bgr, max_iters, enable_florence, enable_sam, enable_vl_critic, scale_mm_per_px),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued", "poll": f"/v1/jobs/{job_id}"}


@router.get("/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> dict:
    meta = store.get(job_id)
    if not meta:
        raise HTTPException(404, "Job not found")
    return meta


@router.get("/v1/jobs/{job_id}/overlay")
def get_overlay(job_id: str) -> FileResponse:
    path = store.file(job_id, "overlay.png")
    if path is None:
        raise HTTPException(404, "Overlay not ready")
    return FileResponse(path, media_type="image/png")


@router.get("/v1/jobs/{job_id}/mask")
def get_mask_layer(job_id: str) -> FileResponse:
    path = store.file(job_id, "mask.png")
    if path is None:
        raise HTTPException(404, "Mask not ready")
    return FileResponse(path, media_type="image/png")


@router.get("/v1/jobs/{job_id}/source")
def get_source(job_id: str) -> FileResponse:
    path = store.file(job_id, "source.png")
    if path is None:
        raise HTTPException(404, "Source not ready")
    return FileResponse(path, media_type="image/png")


@router.get("/v1/jobs/{job_id}/masks/{name}")
def get_mask(job_id: str, name: str) -> FileResponse:
    if name not in CLASSES:
        raise HTTPException(404, "Unknown class")
    path = store.file(job_id, f"masks/{name}.png")
    if path is None:
        raise HTTPException(404, "Mask not ready")
    return FileResponse(path, media_type="image/png")

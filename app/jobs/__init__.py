from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.settings import settings


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        settings.job_dir.mkdir(parents=True, exist_ok=True)

    def _dir(self, job_id: str) -> Path:
        return settings.job_dir / job_id

    def create(self) -> str:
        job_id = uuid.uuid4().hex
        path = self._dir(job_id)
        path.mkdir(parents=True, exist_ok=True)
        self.write_meta(
            job_id,
            {
                "id": job_id,
                "status": "queued",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return job_id

    def write_meta(self, job_id: str, data: dict) -> None:
        with self._lock:
            meta_path = self._dir(job_id) / "job.json"
            current = {}
            if meta_path.exists():
                current = json.loads(meta_path.read_text(encoding="utf-8"))
            current.update(data)
            meta_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    def get(self, job_id: str) -> dict | None:
        path = self._dir(job_id) / "job.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_image(self, job_id: str, name: str, bgr: np.ndarray) -> Path:
        path = self._dir(job_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), bgr)
        return path

    def save_masks(self, job_id: str, masks: dict[str, np.ndarray]) -> None:
        mask_dir = self._dir(job_id) / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for name, mask in masks.items():
            cv2.imwrite(str(mask_dir / f"{name}.png"), mask)

    def file(self, job_id: str, relative: str) -> Path | None:
        path = self._dir(job_id) / relative
        if not path.exists() or not path.is_file():
            return None
        return path


store = JobStore()

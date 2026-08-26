"""Upload the three local elevations to the live API and save results."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://100.104.138.69:8787"
SAMPLES_DIR = ROOT / "samples"
IMAGES = sorted([
    p for p in SAMPLES_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]) if SAMPLES_DIR.exists() else []
OUT = ROOT / "output" / "api-test"
POLL_SEC = 3
TIMEOUT_SEC = 600


UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) elevation-mask-test/0.1",
    "Accept": "*/*",
}


def _request(url: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
    merged = {**UA, **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=merged, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _multipart(path: Path) -> tuple[bytes, str]:
    boundary = "----ElevationMaskBoundary"
    name = path.name
    payload = path.read_bytes()
    ctype = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(path.suffix.lower(), "application/octet-stream")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def submit(path: Path) -> str:
    body, content_type = _multipart(path)
    raw = _request(f"{API}/v1/segment", body, {"Content-Type": content_type})
    data = json.loads(raw)
    job_id = data["job_id"]
    print(f"{path.name}: queued {job_id}")
    return job_id


def poll(job_id: str) -> dict:
    deadline = time.time() + TIMEOUT_SEC
    while time.time() < deadline:
        job = json.loads(_request(f"{API}/v1/jobs/{job_id}"))
        status = job.get("status")
        print(f"  {job_id[:8]}… {status}")
        if status in {"done", "error"}:
            return job
        time.sleep(POLL_SEC)
    raise TimeoutError(f"job {job_id} timed out")


def save(path: Path, job: dict, elapsed_s: float) -> None:
    dest = OUT / path.stem
    dest.mkdir(parents=True, exist_ok=True)
    job = dict(job)
    job["client_elapsed_s"] = round(elapsed_s, 2)
    (dest / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    if job.get("status") != "done":
        print(f"  FAILED {job.get('error')}")
        return
    overlay = _request(f"{API}/v1/jobs/{job['id']}/overlay")
    (dest / "overlay.png").write_bytes(overlay)
    mask = _request(f"{API}/v1/jobs/{job['id']}/mask")
    (dest / "mask.png").write_bytes(mask)
    source = _request(f"{API}/v1/jobs/{job['id']}/source")
    (dest / "source.png").write_bytes(source)
    timing = (job.get("meta") or {}).get("timing") or {}
    print(f"  TIME {path.name}: {elapsed_s:.1f}s client  server={timing}")
    print(f"  source {dest / 'source.png'}")
    print(f"  mask   {dest / 'mask.png'}")
    print(f"  overlay {dest / 'overlay.png'}")
    print(json.dumps(job.get("areas") or {}, indent=2))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    health = json.loads(_request(f"{API}/health"))
    print("health", health)
    missing = [p for p in IMAGES if not p.exists()]
    if missing:
        print("missing", missing, file=sys.stderr)
        return 1
    times: dict[str, float] = {}
    for path in IMAGES:
        t0 = time.perf_counter()
        job_id = submit(path)
        job = poll(job_id)
        elapsed = time.perf_counter() - t0
        times[path.name] = round(elapsed, 2)
        save(path, job, elapsed)
    print("TIME_SUMMARY", json.dumps(times, indent=2))
    (OUT / "timing.json").write_text(json.dumps(times, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

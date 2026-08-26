# elevation-mask

API that segments a CAD **elevation** drawing into materials, paints masks, and measures area.

Not a chatbot. Pipeline:

1. **Florence-2-base** — vision grounding → boxes + labels (`roof`, `window`, `vent`, …)
2. **SAM 2.1 Tiny** — box → mask
3. **OpenCV** — snap masks to CAD ink, split walls by floor line
4. **Agent loop** — geometric critic (optional **Qwen2.5-VL** overlay critic) → fixer → repeat

Palette matches the colored elevation example: roof red, wall L1 green, wall L2 cyan, foundation magenta, window yellow, vent brown, pipe gray.

## API

```
POST /v1/segment          multipart image → { job_id }
GET  /v1/jobs/{id}        status, areas, trace
GET  /v1/jobs/{id}/overlay
GET  /v1/jobs/{id}/mask
GET  /v1/jobs/{id}/masks
GET  /v1/jobs/{id}/masks/{class}
GET  /v1/classes
GET  /health
```

```bash
curl -F image=@samples/image_1.jpg http://127.0.0.1:8787/v1/segment
curl http://127.0.0.1:8787/v1/jobs/<job_id>
```

Form fields: `max_iters`, `enable_yolo_world`, `enable_florence`, `enable_sam`, `enable_vl_critic`, `scale_mm_per_px`.

Default: YOLO-World on, SAM & Florence off, VL critic **off** (CPU 7B/3B is minutes per loop). Geometry critic always runs.

## Run

```bash
docker compose up --build
```

Host port **8787** (8080 inside the container) so it does not collide with other services.

Without Docker (CPU torch):

```bash
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/bin/pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Geometry-only (no model download):

```bash
python -m app.cli samples/image_1.jpg --no-yolo --no-florence --no-sam
```

## Why Python

Florence-2, SAM2, and Qwen2.5-VL are PyTorch. Request time is model inference, not the web framework. FastAPI is the shell.

## Hardware

Built for a CPU box (no NVIDIA). First request downloads weights into the HuggingFace / Ultralytics cache. Set `ENABLE_VL_CRITIC=true` only if you accept a slow vision-language pass.

## Classes

| class | color | area |
|---|---|---|
| roof | red | yes |
| wall_l1 | green | yes |
| wall_l2 | cyan | yes |
| foundation | magenta | yes |
| window | yellow | yes |
| vent | brown | yes |
| pipe | gray | no |

"""CLI: python -m app.cli image_1.jpg --no-yolo --no-florence --no-sam"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from app.pipeline.agent import run_agent
from app.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment a CAD elevation drawing")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--scale-mm-per-px", type=float, default=None)

    yolo_group = parser.add_mutually_exclusive_group()
    yolo_group.add_argument("--yolo", dest="enable_yolo", action="store_true", default=None)
    yolo_group.add_argument("--no-yolo", dest="enable_yolo", action="store_false")

    florence_group = parser.add_mutually_exclusive_group()
    florence_group.add_argument("--florence", dest="enable_florence", action="store_true", default=None)
    florence_group.add_argument("--no-florence", dest="enable_florence", action="store_false")

    sam_group = parser.add_mutually_exclusive_group()
    sam_group.add_argument("--sam", dest="enable_sam", action="store_true", default=None)
    sam_group.add_argument("--no-sam", dest="enable_sam", action="store_false")

    vl_group = parser.add_mutually_exclusive_group()
    vl_group.add_argument("--vl-critic", dest="enable_vl_critic", action="store_true", default=None)
    vl_group.add_argument("--no-vl-critic", dest="enable_vl_critic", action="store_false")

    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")

    result = run_agent(
        bgr,
        max_iters=args.max_iters,
        enable_florence=args.enable_florence,
        enable_yolo_world=args.enable_yolo,
        enable_sam=args.enable_sam,
        enable_vl_critic=args.enable_vl_critic,
    )
    if args.scale_mm_per_px:
        from app.pipeline.area import compute_areas
        from app.pipeline.geometry import perceive

        env = perceive(bgr).envelope
        result["areas"] = compute_areas(result["masks"], env, args.scale_mm_per_px)

    out = args.out / args.image.stem
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "overlay.png"), result["overlay"])
    cv2.imwrite(str(out / "mask.png"), result["mask_layer"])
    cv2.imwrite(str(out / "source.png"), bgr)
    for name, mask in result["masks"].items():
        cv2.imwrite(str(out / f"{name}.png"), mask)
    (out / "areas.json").write_text(
        json.dumps({"areas": result["areas"], "meta": result["meta"], "trace": result["trace"]}, indent=2),
        encoding="utf-8",
    )
    print(out / "overlay.png")
    print(json.dumps(result["areas"], indent=2))


if __name__ == "__main__":
    main()


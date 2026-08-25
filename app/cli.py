"""CLI: python -m app.cli samples/image_1.jpg --no-florence --no-sam"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.pipeline.agent import run_agent
from app.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment a CAD elevation drawing")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--no-florence", action="store_true")
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--vl-critic", action="store_true")
    args = parser.parse_args()

    if args.no_florence:
        settings.enable_florence = False
    if args.no_sam:
        settings.enable_sam = False
    if args.vl_critic:
        settings.enable_vl_critic = True

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    result = run_agent(bgr, max_iters=args.max_iters)
    out = args.out / args.image.stem
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "overlay.png"), result["overlay"])
    cv2.imwrite(str(out / "mask.png"), result["mask_layer"])
    cv2.imwrite(str(out / "source.png"), bgr)
    for name, mask in result["masks"].items():
        cv2.imwrite(str(out / f"{name}.png"), mask)
    (out / "areas.json").write_text(
        __import__("json").dumps({"areas": result["areas"], "meta": result["meta"], "trace": result["trace"]}, indent=2),
        encoding="utf-8",
    )
    print(out / "overlay.png")
    print(__import__("json").dumps(result["areas"], indent=2))


if __name__ == "__main__":
    main()

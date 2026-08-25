from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.pipeline.agent import run_agent
from app.pipeline.geometry import perceive
from app.settings import settings

settings.enable_florence = False
settings.enable_sam = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "local-silhouette"
IMAGES = [ROOT / "image_1.jpg", ROOT / "image_2.jpeg", ROOT / "image_3.png"]


def main() -> None:
    for path in IMAGES:
        bgr = cv2.imread(str(path))
        p = perceive(bgr)
        print("====", path.name, "env%", round(100 * p.envelope.mean() / 255, 1), "eave", p.eave_y, "floor", p.floor_y)
        print("  hatch", 0 if p.hatch is None else int((p.hatch > 0).sum()), "faces", len(p.faces))
        for k, v in p.geometry.items():
            ys, xs = np.where(v > 0)
            if ys.size == 0:
                print(" ", k, "EMPTY")
                continue
            print(f"  {k} y {ys.min()}-{ys.max()} x {xs.min()}-{xs.max()} px {ys.size}")
        for face in p.faces:
            x1, y1, x2, y2 = face.box
            print(" ", face.label, face.box, "wh", x2 - x1, y2 - y1)
        result = run_agent(bgr, max_iters=3)
        dest = OUT / path.stem
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest / "overlay.png"), result["overlay"])
        cv2.imwrite(str(dest / "mask.png"), result["mask_layer"])
        print("  areas", {k: v["percent_of_envelope"] for k, v in result["areas"].items() if k != "_envelope" and v["pixels"]})
        print("  iters", result["meta"]["iters"], "issues", result["meta"]["open_issues"])
        print("  wrote", dest / "overlay.png")


if __name__ == "__main__":
    main()

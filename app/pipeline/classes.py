from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialClass:
    name: str
    bgr: tuple[int, int, int]
    priority: int
    count_area: bool = True


# BGR for OpenCV. Priority: higher wins on overlap.
CLASSES: dict[str, MaterialClass] = {
    "window": MaterialClass("window", (0, 220, 255), 90),
    "vent": MaterialClass("vent", (40, 140, 196), 85),
    "foundation": MaterialClass("foundation", (180, 40, 220), 70),
    "roof": MaterialClass("roof", (40, 40, 220), 60),
    "wall_l2": MaterialClass("wall_l2", (220, 220, 0), 40),
    "wall_l1": MaterialClass("wall_l1", (70, 180, 40), 40),
    "pipe": MaterialClass("pipe", (180, 180, 180), 20, count_area=False),
}

SYNONYMS: dict[str, str] = {
    "roof": "roof",
    "roofs": "roof",
    "gable": "roof",
    "eave": "roof",
    "eaves": "roof",
    "gutter": "roof",
    "gutters": "roof",
    "shingle": "roof",
    "tile roof": "roof",
    "window": "window",
    "windows": "window",
    "sliding window": "window",
    "door": "window",
    "doors": "window",
    "glazing": "window",
    "vent": "vent",
    "vents": "vent",
    "louver": "vent",
    "louvers": "vent",
    "louvre": "vent",
    "wall": "wall_l1",
    "walls": "wall_l1",
    "siding": "wall_l1",
    "facade": "wall_l1",
    "cladding": "wall_l1",
    "second floor wall": "wall_l2",
    "upper wall": "wall_l2",
    "first floor wall": "wall_l1",
    "foundation": "foundation",
    "plinth": "foundation",
    "base": "foundation",
    "slab": "foundation",
    "pipe": "pipe",
    "downspout": "pipe",
    "drain pipe": "pipe",
    "gutter pipe": "pipe",
}


def canonical_label(raw: str) -> str | None:
    key = raw.strip().lower()
    if key in CLASSES:
        return key
    return SYNONYMS.get(key)

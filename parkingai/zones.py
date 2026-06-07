"""Parking-spot zones and occupancy geometry.

A zone is a labelled polygon. A zone is "occupied" when a vehicle bounding box
covers at least ``coverage_threshold`` of the zone's area.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

Point = Tuple[int, int]


@dataclass
class Zone:
    id: str
    points: List[Point]
    polygon: Polygon = field(init=False, repr=False)
    area: float = field(init=False)

    def __post_init__(self) -> None:
        self.polygon = Polygon(self.points)
        # buffer(0) repairs minor self-intersections from hand-drawn polygons.
        if not self.polygon.is_valid:
            self.polygon = self.polygon.buffer(0)
        self.area = float(self.polygon.area)


def load_zones(path: str | Path) -> List[Zone]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    zones: List[Zone] = []
    for item in data.get("zones", []):
        pts = [(int(x), int(y)) for x, y in item["points"]]
        zones.append(Zone(id=str(item["id"]), points=pts))
    return zones


def save_zones(path: str | Path, zones: Sequence[Zone]) -> None:
    data = {"zones": [{"id": z.id, "points": [list(p) for p in z.points]} for z in zones]}
    Path(path).write_text(json.dumps(data, indent=2))


def zone_coverage(zone: Zone, boxes: Sequence[Sequence[float]]) -> float:
    """Largest fraction of the zone area covered by any single vehicle box."""
    if zone.area <= 0:
        return 0.0
    best = 0.0
    for b in boxes:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        rect = shapely_box(x1, y1, x2, y2)
        inter = zone.polygon.intersection(rect).area
        best = max(best, inter / zone.area)
    return best

"""Bounding-box geometry helpers in xyxy coordinates."""

from __future__ import annotations

import math

from helmet_rider.detection.types import BBox


def area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: BBox, b: BBox) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def iou(a: BBox, b: BBox) -> float:
    overlap = intersection(a, b)
    union = area(a) + area(b) - overlap
    return overlap / union if union > 0.0 else 0.0


def ioa(subject: BBox, region: BBox) -> float:
    """Intersection over the subject area."""
    subject_area = area(subject)
    return intersection(subject, region) / subject_area if subject_area > 0.0 else 0.0


def expand(box: BBox, x_ratio: float, y_ratio: float) -> BBox:
    width, height = box[2] - box[0], box[3] - box[1]
    dx, dy = width * x_ratio, height * y_ratio
    return box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy


def bottom_center(box: BBox) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, box[3]


def point_box_distance(point: tuple[float, float], box: BBox) -> float:
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


def clip(box: BBox, width: int, height: int) -> BBox:
    return (
        min(max(box[0], 0.0), float(width)),
        min(max(box[1], 0.0), float(height)),
        min(max(box[2], 0.0), float(width)),
        min(max(box[3], 0.0), float(height)),
    )

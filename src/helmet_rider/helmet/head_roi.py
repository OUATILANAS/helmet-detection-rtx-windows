"""Configurable, clipped head/upper-head crop estimation."""

from __future__ import annotations

import numpy as np

from helmet_rider.config import HeadROIConfig
from helmet_rider.detection.types import BBox


class HeadROIExtractor:
    def __init__(self, config: HeadROIConfig | None = None) -> None:
        self.config = config or HeadROIConfig()

    def extract_box(self, frame: np.ndarray, bbox: BBox) -> tuple[np.ndarray, BBox] | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        margin_x = (x2-x1) * self.config.matched_context
        margin_y = (y2-y1) * self.config.matched_context
        left, top = max(0, int(np.floor(x1-margin_x))), max(0, int(np.floor(y1-margin_y)))
        right, bottom = min(width, int(np.ceil(x2+margin_x))), min(height, int(np.ceil(y2+margin_y)))
        if min(right-left, bottom-top) < self.config.minimum_size:
            return None
        return frame[top:bottom, left:right], (float(left), float(top), float(right), float(bottom))

    def extract(self, frame: np.ndarray, person_box: BBox) -> tuple[np.ndarray, BBox] | None:
        if frame is None or frame.ndim < 2 or frame.size == 0:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = person_box
        person_width = max(0.0, x2 - x1)
        person_height = max(0.0, y2 - y1)
        if person_width <= 0.0 or person_height <= 0.0:
            return None

        roi_height = person_height * self.config.height_ratio
        roi_width = min(person_width, roi_height * self.config.width_expansion)
        center_x = (x1 + x2) / 2.0
        left = max(0, int(round(center_x - roi_width / 2)))
        right = min(width, int(round(center_x + roi_width / 2)))
        top = max(0, int(round(y1)))
        bottom = min(height, int(round(y1 + roi_height)))
        if right - left < self.config.minimum_size or bottom - top < self.config.minimum_size:
            return None
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        return crop, (float(left), float(top), float(right), float(bottom))

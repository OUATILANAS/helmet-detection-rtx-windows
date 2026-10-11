"""Typed detection and frame result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

BBox = tuple


class HelmetStatus(str, Enum):
    HELMET = "helmet"
    NO_HELMET = "no_helmet"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: BBox
    track_id: int | None = None


@dataclass(frozen=True)
class HelmetPrediction:
    status: HelmetStatus = HelmetStatus.UNKNOWN
    confidence: float = 0.0
    helmet_probability: float = 0.0
    no_helmet_probability: float = 0.0


@dataclass(frozen=True)
class Association:
    person_index: int
    vehicle_index: int
    score: float
    lower_body_overlap: float
    person_overlap: float
    horizontal_alignment: float
    bottom_proximity: float


@dataclass(frozen=True)
class RiderResult:
    track_id: int
    bbox: BBox
    vehicle_id: int
    vehicle_type: str
    head_bbox: BBox | None
    helmet: HelmetPrediction
    association_score: float
    helmet_bbox: BBox | None = None
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VehicleResult:
    track_id: int
    vehicle_type: str
    bbox: BBox
    confidence: float
    riders: tuple[RiderResult, ...] = ()


@dataclass(frozen=True)
class PersonResult:
    track_id: int
    bbox: BBox
    confidence: float
    motorcycle_status: str
    vehicle_id: int | None = None


@dataclass(frozen=True)
class HeadResult:
    """An observed head; a body or motorcycle association may be unavailable."""
    track_id: int
    head_bbox: BBox
    helmet: HelmetPrediction
    confidence: float
    person_id: int | None = None
    vehicle_id: int | None = None
    motorcycle_status: str = "uncertain"
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FrameTimings:
    capture_ms: float = 0.0
    detection_ms: float = 0.0
    tracking_ms: float = 0.0
    association_ms: float = 0.0
    classification_ms: float = 0.0
    drawing_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class FrameResult:
    width: int
    height: int
    detections: tuple[Detection, ...]
    vehicles: tuple[VehicleResult, ...]
    riders: tuple[RiderResult, ...]
    associations: tuple[Association, ...] = ()
    timings: FrameTimings = field(default_factory=FrameTimings)
    fps: float = 0.0
    persons: tuple[PersonResult, ...] = ()
    detector_ran: bool = True
    heads: tuple[HeadResult, ...] | None = None

"""JSON-safe result serialization shared by the API and CLI."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum

from helmet_rider.detection.types import FrameResult


def result_to_dict(result: FrameResult) -> dict:
    data = asdict(result)

    def convert(value):
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    payload = convert(data)
    payload["motorcycle_presence"] = (
        "detected" if any(v.vehicle_type == "motorcycle" for v in result.vehicles)
        else "not_detected" if result.detector_ran else "uncertain"
    )
    return payload

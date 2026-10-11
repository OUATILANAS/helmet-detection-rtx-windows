"""Attach capture and drawing costs to immutable frame results."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from helmet_rider.config import AppConfig
from helmet_rider.detection.types import FrameResult
from helmet_rider.visualization.draw import draw_result


def add_capture_time(result: FrameResult, capture_ms: float) -> FrameResult:
    timings = replace(
        result.timings,
        capture_ms=max(0.0, capture_ms),
        total_ms=result.timings.total_ms + max(0.0, capture_ms),
    )
    return replace(
        result,
        timings=timings,
        fps=1000.0 / timings.total_ms if timings.total_ms > 0 else 0.0,
    )


def draw_timed(
    frame: np.ndarray, result: FrameResult, config: AppConfig
) -> tuple[np.ndarray, FrameResult]:
    started = perf_counter()
    annotated = draw_result(frame, result, config)
    drawing_ms = (perf_counter() - started) * 1000.0
    timings = replace(
        result.timings,
        drawing_ms=drawing_ms,
        total_ms=result.timings.total_ms + drawing_ms,
    )
    return annotated, replace(
        result,
        timings=timings,
        fps=1000.0 / timings.total_ms if timings.total_ms > 0 else 0.0,
    )

"""Class-aware IoU tracking with elapsed-time motion and bounded track age."""

from __future__ import annotations

from dataclasses import dataclass, replace

from helmet_rider.association.geometry import iou
from helmet_rider.config import TrackingConfig
from helmet_rider.detection.types import BBox, Detection


@dataclass
class _TrackState:
    detection: Detection
    velocity: BBox = (0.0, 0.0, 0.0, 0.0)
    age: int = 0
    observed_at: float = 0.0

    def predicted_bbox(self, now: float) -> BBox:
        elapsed = max(0.0, now - self.observed_at)
        return tuple(
            c + v * elapsed for c, v in zip(self.detection.bbox, self.velocity, strict=True)
        )


class IoUTracker:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}
        self._clock = -1.0
        self._timed = False

    def _advance(self, timestamp_seconds: float | None) -> float:
        self._timed = timestamp_seconds is not None
        self._clock = float(timestamp_seconds) if self._timed else self._clock + 1
        self._prune()
        return self._clock

    def update(
        self, detections: list[Detection], *, timestamp_seconds: float | None = None
    ) -> list[Detection]:
        now = self._advance(timestamp_seconds)
        unmatched = set(self._tracks)
        output: list[Detection | None] = [None] * len(detections)
        for index in sorted(
            range(len(detections)), key=lambda i: detections[i].confidence, reverse=True
        ):
            detection = detections[index]
            candidates = [
                (iou(detection.bbox, s.predicted_bbox(now)), tid)
                for tid, s in self._tracks.items()
                if tid in unmatched and s.detection.class_name == detection.class_name
            ]
            best_iou, best_id = max(candidates, default=(0.0, None))
            if best_id is not None and best_iou >= self.config.iou_threshold:
                track_id = best_id
                unmatched.remove(track_id)
                previous = self._tracks[track_id]
                elapsed = max(1e-6, now - previous.observed_at)
                velocity = tuple(
                    0.5 * old_v + 0.5 * (c - old) / elapsed
                    for c, old, old_v in zip(
                        detection.bbox, previous.detection.bbox, previous.velocity, strict=True
                    )
                )
                weight = self.config.bbox_smoothing
                bbox = tuple(
                    weight * c + (1 - weight) * prior
                    for c, prior in zip(detection.bbox, previous.predicted_bbox(now), strict=True)
                )
            else:
                track_id = self._next_id
                self._next_id += 1
                velocity, bbox = (0.0, 0.0, 0.0, 0.0), detection.bbox
            tracked = replace(detection, bbox=bbox, track_id=track_id)
            self._tracks[track_id] = _TrackState(tracked, velocity, 0, now)
            output[index] = tracked
        for track_id in unmatched:
            self._tracks[track_id].age += 1
        self._prune()
        return [item for item in output if item is not None]

    def predict(self, *, timestamp_seconds: float | None = None) -> list[Detection]:
        now = self._advance(timestamp_seconds)
        predictions = []
        for track_id, state in self._tracks.items():
            if state.age >= self.config.max_age:
                continue
            state.age += 1
            predictions.append(
                replace(state.detection, bbox=state.predicted_bbox(now), track_id=track_id)
            )
        self._prune()
        return predictions

    def _prune(self) -> None:
        self._tracks = {
            tid: s
            for tid, s in self._tracks.items()
            if s.age <= self.config.max_age
            and (not self._timed or self._clock - s.observed_at <= self.config.max_age_seconds)
        }

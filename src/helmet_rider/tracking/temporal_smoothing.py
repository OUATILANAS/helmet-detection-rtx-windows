"""Confidence-weighted short-window helmet prediction smoothing."""

from __future__ import annotations

from collections import defaultdict, deque

from helmet_rider.config import ClassifierConfig
from helmet_rider.detection.types import HelmetPrediction, HelmetStatus


class TemporalSmoother:
    def __init__(self, window_size: int = 5, classifier: ClassifierConfig | None = None):
        self.window_size = max(1, int(window_size))
        self.classifier = classifier or ClassifierConfig()
        self._history: dict[int, deque[HelmetPrediction]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._last_seen_frame: dict[int, int] = {}

    def update(
        self, track_id: int, prediction: HelmetPrediction, frame_number: int | None = None
    ) -> HelmetPrediction:
        if frame_number is None:
            frame_number = max(self._last_seen_frame.values(), default=0) + 1
        self._last_seen_frame[track_id] = frame_number
        history = self._history[track_id]
        history.append(prediction)
        known = [item for item in history if item.status != HelmetStatus.UNKNOWN]
        if not known:
            return prediction
        weights = [max(item.confidence, 0.01) for item in known]
        total = sum(weights)
        helmet_probability = sum(
            item.helmet_probability * weight
            for item, weight in zip(known, weights, strict=True)
        ) / total
        no_helmet_probability = 1.0 - helmet_probability
        confidence = max(helmet_probability, no_helmet_probability)
        if confidence < self.classifier.min_confidence:
            status = HelmetStatus.UNKNOWN
        elif helmet_probability >= self.classifier.helmet_positive_threshold:
            status = HelmetStatus.HELMET
        elif no_helmet_probability >= self.classifier.no_helmet_negative_threshold:
            status = HelmetStatus.NO_HELMET
        else:
            status = HelmetStatus.UNKNOWN
        return HelmetPrediction(status, confidence, helmet_probability, no_helmet_probability)

    def forget(self, track_id: int) -> None:
        self._history.pop(track_id, None)
        self._last_seen_frame.pop(track_id, None)

    def forget_except(
        self,
        active_track_ids: set[int],
        frame_number: int | None = None,
        max_age: int = 0,
    ) -> None:
        if frame_number is None:
            frame_number = max(self._last_seen_frame.values(), default=0)
        for track_id in tuple(self._history):
            if track_id in active_track_ids:
                self._last_seen_frame[track_id] = frame_number
            elif frame_number - self._last_seen_frame.get(track_id, frame_number) > max_age:
                del self._history[track_id]
                self._last_seen_frame.pop(track_id, None)

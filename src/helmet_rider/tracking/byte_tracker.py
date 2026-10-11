"""Optional class-isolated ByteTrack candidate; time expiry stays with the adapter."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from helmet_rider.config import TrackingConfig


class ByteTrackAdapter:
    CLASSES = ("person", "motorcycle", "bicycle")

    def __init__(self, config: TrackingConfig):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.config, self.Boxes = config, Boxes
        args = SimpleNamespace(
            track_high_thresh=0.1,
            track_low_thresh=0.02,
            new_track_thresh=0.1,
            track_buffer=config.max_age,
            match_thresh=0.8,
            fuse_score=True,
        )
        self.trackers = [BYTETracker(args) for _ in self.CLASSES]
        self._last_timestamp = None
        self._generation = 0

    def update(self, detections, *, timestamp_seconds=None):
        if (
            timestamp_seconds is not None
            and self._last_timestamp is not None
            and timestamp_seconds - self._last_timestamp > self.config.max_age_seconds
        ):
            for tracker in self.trackers:
                tracker.reset()
            self._generation += 1
        self._last_timestamp = timestamp_seconds
        output = []
        for class_index, (name, tracker) in enumerate(
            zip(self.CLASSES, self.trackers, strict=True)
        ):
            subset = [d for d in detections if d.class_name == name]
            array = np.array(
                [list(d.bbox) + [d.confidence, class_index] for d in subset], dtype=np.float32
            ).reshape(-1, 6)
            boxes = self.Boxes(array, orig_shape=(1, 1))
            tracks = tracker.update(boxes)
            for track in tracks:
                source = subset[int(track[-1])]
                identifier = self._generation * 100000000 + class_index * 1000000 + int(track[4])
                output.append(
                    replace(source, bbox=tuple(float(v) for v in track[:4]), track_id=identifier)
                )
        return output

    def predict(self, *, timestamp_seconds=None):
        # ByteTrack's Kalman clock advances per update; do not present stale boxes as detections.
        return self.update([], timestamp_seconds=timestamp_seconds)


def create_tracker(config):
    if config.backend == "bytetrack":
        return ByteTrackAdapter(config)
    from helmet_rider.tracking.tracker import IoUTracker

    return IoUTracker(config)

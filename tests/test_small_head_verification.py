"""Small-head recovery must retain abstention, ownership, and stream gates."""
from dataclasses import replace

import numpy as np

from helmet_rider.config import AppConfig
from helmet_rider.detection.types import Detection, HelmetPrediction, HelmetStatus, RiderResult
from helmet_rider.pipeline.pipeline import HelmetRiderPipeline

VEHICLES = [Detection("motorcycle", .9, (0, 90, 80, 190), track_id=2)]


class Detector:
    def detect(self, frame):
        return []


class Classifier:
    def __init__(self, scores=(.91, .89)):
        self.scores = scores
        self.calls = 0

    def classify_batch(self, crops):
        self.calls += 1
        return [HelmetPrediction(HelmetStatus.NO_HELMET, p, 1-p, p)
                for p in self.scores]


def setup(scores=(.91, .89)):
    classifier = Classifier(scores)
    pipe = HelmetRiderPipeline(AppConfig(), detector=Detector(), classifier=classifier)
    pipe.frame_number = 1
    rider = RiderResult(1, (10, 10, 60, 180), 2, "motorcycle", (20, 10, 40, 30),
                        HelmetPrediction(), .9,
                        evidence={"classifier_updated": True,
                                  "classifier_no_helmet_probability": .95})
    head = Detection("without_helmet", .70, (20, 10, 38, 29))
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    return pipe, classifier, rider, head, frame


def test_localized_small_head_requires_both_views_and_real_source_detail():
    pipe, _, rider, head, frame = setup()
    result = pipe._recheck_small_no_helmet(frame, [rider], [head], vehicles=VEHICLES)[0]
    assert result.helmet.status == HelmetStatus.NO_HELMET
    assert result.head_bbox == head.bbox
    assert result.evidence["small_head_verified"]
    assert result.evidence["negative_confirmations"] == 1

    pipe, _, rider, head, frame = setup((.98, .80))
    assert pipe._recheck_small_no_helmet(frame, [rider], [head], vehicles=VEHICLES)[0].helmet.status == HelmetStatus.UNKNOWN
    tiny = replace(head, bbox=(20, 10, 35, 29))
    assert pipe._recheck_small_no_helmet(frame, [rider], [tiny], vehicles=VEHICLES)[0].helmet.status == HelmetStatus.UNKNOWN
    assert pipe._recheck_small_no_helmet(frame, [rider], [replace(head, confidence=.49)], vehicles=VEHICLES)[0].helmet.status == HelmetStatus.UNKNOWN
    assert pipe._recheck_small_no_helmet(frame, [rider], [replace(head, class_name="with_helmet")], vehicles=VEHICLES)[0].helmet.status == HelmetStatus.UNKNOWN


def test_small_negative_never_overrides_helmet_or_positive_recheck():
    pipe, classifier, rider, head, frame = setup()
    known = replace(rider, helmet=HelmetPrediction(HelmetStatus.HELMET, .9, .9, .1))
    assert pipe._recheck_small_no_helmet(frame, [known], [head], vehicles=VEHICLES)[0] == known
    conflicted = replace(rider, evidence={**rider.evidence, "head_recheck_helmet_probability": .60})
    assert pipe._recheck_small_no_helmet(frame, [conflicted], [head], vehicles=VEHICLES)[0] == conflicted
    assert classifier.calls == 0


def test_stream_requires_two_fresh_passes_and_discards_expired_evidence():
    pipe, classifier, rider, head, frame = setup()
    def process(r, heads, t):
        return pipe._recheck_small_no_helmet(
            frame, [r], heads, vehicles=VEHICLES, temporal_consistency=True, timestamp_seconds=t)[0]
    assert process(rider, [head], 0).helmet.status == HelmetStatus.UNKNOWN
    cached = replace(rider, evidence={**rider.evidence, "classifier_updated": False})
    pipe.frame_number = 2
    assert process(cached, [head], .1).helmet.status == HelmetStatus.UNKNOWN
    assert classifier.calls == 1
    pipe.frame_number = 4
    result = process(rider, [head], .3)
    assert result.helmet.status == HelmetStatus.NO_HELMET
    assert result.evidence["negative_confirmations"] == 2
    pipe.frame_number = 5
    assert process(cached, [head], .4).helmet.status == HelmetStatus.NO_HELMET
    assert classifier.calls == 2
    pipe.frame_number = 6
    assert process(cached, [head], 20).helmet.status == HelmetStatus.UNKNOWN
    assert not pipe._small_no_helmet_by_track
    pipe.frame_number = 7
    process(rider, [head], 21)
    assert process(cached, [], 21.1).helmet.status == HelmetStatus.UNKNOWN
    assert not pipe._small_no_helmet_by_track


def test_two_overlapping_riders_cannot_share_one_negative_detection():
    pipe, _, rider, head, frame = setup()
    duplicate = replace(rider, track_id=3, head_bbox=(22, 10, 42, 30))
    result = pipe._recheck_small_no_helmet(frame, [rider, duplicate], [head], vehicles=VEHICLES)
    assert sum(r.helmet.status == HelmetStatus.NO_HELMET for r in result) == 1
    pipe.reset()
    assert not pipe._small_no_helmet_by_track


def test_person_with_motorcycle_box_on_upper_body_cannot_gain_negative_label():
    pipe, classifier, rider, head, frame = setup()
    false_vehicle = Detection("motorcycle", .9, (12, 30, 58, 140), track_id=2)
    result = pipe._recheck_small_no_helmet(frame, [rider], [head], vehicles=[false_vehicle])[0]
    assert result.helmet.status == HelmetStatus.UNKNOWN
    assert classifier.calls == 0

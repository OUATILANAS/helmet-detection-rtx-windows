"""Head refinement must preserve reviewed decisions and fresh stream evidence."""
from dataclasses import replace

import numpy as np

from helmet_rider.detection.types import Detection, HelmetPrediction, HelmetStatus, RiderResult
from helmet_rider.helmet.head_roi import HeadROIExtractor
from helmet_rider.helmet.localized import LocalizedHeadRefiner, assign_heads


class Classifier:
    def __init__(self):
        self.p = .8
        self.calls = 0

    def classify_batch(self, crops):
        self.calls += 1
        return [HelmetPrediction(HelmetStatus.NO_HELMET, self.p, 1-self.p, self.p) for _ in crops]


def setup():
    classifier = Classifier()
    refiner = LocalizedHeadRefiner(classifier, HeadROIExtractor(), max_age=1)
    rider = RiderResult(1, (10, 10, 60, 180), 2, "motorcycle", (20, 10, 40, 30),
                        HelmetPrediction(), .9, evidence={"classifier_updated": True})
    head = Detection("with_helmet", .7, (20, 10, 38, 29))
    return refiner, classifier, rider, head, np.zeros((200, 100, 3), dtype=np.uint8)


def test_verified_rider_reserves_head_without_being_overridden():
    refiner, classifier, rider, head, frame = setup()
    known = replace(rider, helmet=HelmetPrediction(HelmetStatus.HELMET, .95, .95, .05))
    duplicate = replace(rider, track_id=3, head_bbox=(22, 10, 42, 30))
    output = refiner.refine(frame, [known, duplicate], [head])
    assert output[0] == known
    assert output[1] == duplicate
    assert classifier.calls == 0


def test_stream_needs_fresh_negative_confirmations_and_discards_stale_or_absent_heads():
    refiner, classifier, rider, head, frame = setup()
    def process(r, heads, timestamp):
        return refiner.refine(frame, [r], heads, temporal=True, timestamp=timestamp)[0]
    assert process(rider, [head], 0).helmet.status == HelmetStatus.UNKNOWN
    cached = replace(rider, evidence={"classifier_updated": False})
    assert process(cached, [head], .1).helmet.status == HelmetStatus.UNKNOWN
    assert classifier.calls == 1
    assert process(rider, [head], .3).helmet.status == HelmetStatus.NO_HELMET
    assert process(cached, [head], .4).helmet.status == HelmetStatus.NO_HELMET
    assert classifier.calls == 2
    assert process(cached, [head], 2).helmet.status == HelmetStatus.UNKNOWN
    assert not refiner.cache
    process(rider, [head], 2.1)
    assert process(cached, [], 2.2).helmet.status == HelmetStatus.UNKNOWN
    assert not refiner.cache
    process(rider, [head], 2.3)
    refiner.reset()
    assert not refiner.cache


def test_classifier_conflict_and_head_reassignment_do_not_keep_old_negative():
    refiner, classifier, rider, head, frame = setup()
    refiner.refine(frame, [rider], [head], temporal=True, timestamp=0)
    classifier.p = .15
    output = refiner.refine(frame, [rider], [head], temporal=True, timestamp=.2)[0]
    assert output.helmet.status == HelmetStatus.HELMET
    classifier.p = .51
    output = refiner.refine(frame, [rider], [head], temporal=True, timestamp=.4)[0]
    assert output.helmet.status == HelmetStatus.UNKNOWN
    assert output.evidence["localized_refinement_confirmations"] == 1


def test_native_model_relocalizes_only_agreeing_known_heads_and_keeps_stream_gate():
    refiner, classifier, rider, head, frame = setup()
    refiner.native_threshold = .5
    refiner.relocalize_known = True
    known = replace(rider, helmet=HelmetPrediction(HelmetStatus.HELMET, .95, .95, .05))
    output = refiner.refine(frame, [known], [head])[0]
    assert output.helmet == known.helmet
    assert output.head_bbox == head.bbox
    assert classifier.calls == 0
    bare = replace(head, class_name="without_helmet")
    assert refiner.refine(frame, [known], [bare])[0] == known
    refiner.relocalize_all_known = True
    moved = refiner.refine(frame, [known], [bare])[0]
    assert moved.head_bbox == bare.bbox
    assert moved.helmet == known.helmet
    classifier.p = .1  # The native negative disagrees with the crop classifier.
    output = refiner.refine(frame, [rider], [bare], temporal=True, timestamp=0)[0]
    assert output.helmet.status == HelmetStatus.UNKNOWN
    output = refiner.refine(frame, [rider], [bare], temporal=True, timestamp=.2)[0]
    assert output.helmet.status == HelmetStatus.NO_HELMET


def test_two_negative_classifiers_veto_native_helmet_and_expire_positive_cache():
    refiner, classifier, rider, head, frame = setup()
    refiner.native_threshold = .5
    classifier.p = .1
    verifier = Classifier()
    refiner.verifier = verifier
    rider = replace(rider, evidence={"classifier_updated": True,
                                    "classifier_no_helmet_probability": .8})
    first = refiner.refine(frame, [rider], [head], temporal=True, timestamp=0)[0]
    assert first.helmet.status == HelmetStatus.UNKNOWN
    assert refiner.cache[rider.track_id]["prediction"].status == HelmetStatus.UNKNOWN
    stale = replace(rider, evidence={**rider.evidence, "classifier_updated": False})
    assert refiner.refine(frame, [stale], [head], temporal=True, timestamp=.1)[0].helmet.status == HelmetStatus.UNKNOWN
    verifier.p = .1
    assert refiner.refine(frame, [rider], [head], temporal=True, timestamp=.2)[0].helmet.status == HelmetStatus.HELMET


def test_strong_classifier_consensus_overrides_native_class_but_needs_fresh_negatives():
    refiner, classifier, rider, head, frame = setup()
    refiner.native_threshold = .5
    refiner.prefer_consensus = True
    classifier.p = .95
    verifier = Classifier()
    verifier.p = .90
    refiner.verifier = verifier
    first = refiner.refine(frame,[rider],[head],temporal=True,timestamp=0)[0]
    assert first.helmet.status == HelmetStatus.UNKNOWN
    assert first.evidence["localized_classifier_consensus"]
    second = refiner.refine(frame,[rider],[head],temporal=True,timestamp=.2)[0]
    assert second.helmet.status == HelmetStatus.NO_HELMET
    classifier.p = .05
    verifier.p = .05
    bare = replace(head,class_name="without_helmet")
    positive = refiner.refine(frame,[rider],[bare],temporal=True,timestamp=.4)[0]
    assert positive.helmet.status == HelmetStatus.HELMET


def test_known_helmet_requires_both_contexts_and_crown_coverage_to_be_overridden():
    refiner, _, rider, head, _ = setup()
    refiner.verify_known = True
    known = replace(rider, helmet=HelmetPrediction(HelmetStatus.HELMET, .95, .95, .05),
                    helmet_bbox=(20, 5, 40, 30))
    negative = HelmetPrediction(HelmetStatus.NO_HELMET, .98, .02, .98)
    # A face-only crop below a helmet cannot prove absence of a helmet.
    face = replace(head, bbox=(20, 20, 38, 39))
    assert refiner.decide_prediction(known, face, negative, negative, negative)[0] is None
    weak_wide = HelmetPrediction(HelmetStatus.NO_HELMET, .80, .20, .80)
    assert refiner.decide_prediction(known, head, negative, negative, weak_wide)[0] is None
    decision, consensus = refiner.decide_prediction(known, head, negative, negative, negative)
    assert decision.status == HelmetStatus.NO_HELMET
    assert consensus
    bare = replace(known, helmet=negative)
    positive = HelmetPrediction(HelmetStatus.HELMET, .98, .98, .02)
    assert refiner.decide_prediction(bare, head, positive, positive, positive)[0].status == HelmetStatus.HELMET
    assert refiner.decide_prediction(bare, head, positive, positive, None)[0] is None


def test_occluded_body_can_match_head_above_body_and_keeps_one_owner():
    _, _, rider, head, _ = setup()
    rider = replace(rider, bbox=(10, 40, 60, 180))
    assert assign_heads([rider], [head]) == []
    assert assign_heads([rider], [head], occlusion_tolerant=True) == [(0, 0)]
    duplicate = replace(rider, track_id=2)
    assert len(assign_heads([rider, duplicate], [head], occlusion_tolerant=True)) == 1
    unrelated = replace(head, bbox=(70, 10, 88, 29))
    assert assign_heads([rider], [unrelated], occlusion_tolerant=True) == []


def test_missing_legacy_probability_does_not_crash_conflict_verifier():
    refiner, _, rider, head, _ = setup()
    rider = replace(rider, evidence={"classifier_no_helmet_probability": None})
    positive = HelmetPrediction(HelmetStatus.HELMET, .9, .9, .1)
    negative = HelmetPrediction(HelmetStatus.NO_HELMET, .9, .1, .9)
    assert refiner.decide_prediction(rider, head, positive, negative)[0].status == HelmetStatus.HELMET


def test_low_confidence_head_requires_explicit_recovery_profile():
    refiner, classifier, rider, head, frame = setup()
    weak = replace(head, confidence=.07)
    assert refiner.refine(frame, [rider], [weak])[0] == rider
    assert classifier.calls == 0
    refiner.head_min_confidence = .05
    assert refiner.refine(frame, [rider], [weak])[0].helmet.status == HelmetStatus.NO_HELMET
    assert classifier.calls == 1

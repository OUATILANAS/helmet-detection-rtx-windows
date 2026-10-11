import numpy as np

from helmet_rider.config import AppConfig, DetectorConfig
from helmet_rider.detection.types import Detection, HelmetPrediction, HelmetStatus
from helmet_rider.pipeline.pipeline import HelmetRiderPipeline


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return list(self.detections)


def test_primary_head_model_replaces_old_heads_and_is_reused_for_refinement():
    from dataclasses import replace

    from helmet_rider.config import HeadROIConfig

    class Heads(FakeDetector):
        calls = 0
        def detect(self, frame):
            self.calls += 1
            return super().detect(frame)

    class Refiner:
        seen = None
        def refine(self, frame, riders, heads, **kwargs):
            self.seen = heads
            return riders

    base = AppConfig(head_roi=HeadROIConfig(strategy="verified"))
    config = replace(base, classifier=replace(base.classifier, localized_use_as_primary=True))
    old_head = Detection("with_helmet", .75, (380, 80, 420, 140))
    new_head = Detection("with_helmet", .85, (385, 90, 430, 150))
    pipeline = HelmetRiderPipeline(config, detector=FakeDetector([
        Detection("person", .9, (350, 80, 460, 340)),
        Detection("motorcycle", .9, (300, 250, 600, 400)), old_head,
    ]), classifier=CountingClassifier())
    heads = Heads([new_head])
    pipeline.localized_head_detector = heads
    pipeline.localized_refiner = Refiner()
    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert heads.calls == 1
    assert pipeline.localized_refiner.seen == [new_head]
    assert old_head not in result.detections
    assert new_head in result.detections
    assert result.riders[0].evidence["detector_confidence"] == new_head.confidence


def test_large_frame_recovery_deduplicates_bikes_and_keeps_bystander_unassociated():
    from dataclasses import replace

    class Recovery(FakeDetector):
        calls = 0
        def detect(self, frame):
            self.calls += 1
            return super().detect(frame)

    base = AppConfig()
    config = replace(base, classifier=replace(base.classifier, localized_recover_vehicles=True))
    bike = Detection("motorcycle", .9, (790, 390, 890, 470))
    pipeline = HelmetRiderPipeline(config, detector=FakeDetector([bike]), classifier=CountingClassifier())
    recovery = Recovery([Detection("person", .9, (720, 240, 775, 455)),
                         Detection("motorcycle", .7, (791, 391, 889, 469))])
    pipeline.localized_person_detector = recovery
    small = pipeline.process(np.zeros((480, 1000, 3), dtype=np.uint8))
    assert recovery.calls == 0
    assert not small.persons
    large = pipeline.process(np.zeros((480, 1920, 3), dtype=np.uint8))
    assert recovery.calls == 1
    assert len([d for d in large.detections if d.class_name == "motorcycle"]) == 1
    assert not large.riders
    assert large.persons[0].motorcycle_status == "no_motorcycle_match"


def test_verified_localization_confirms_head_and_reuses_fresh_track_evidence():
    from helmet_rider.config import HeadROIConfig

    class TwoViews:
        def __init__(self): self.calls = 0
        def classify_batch(self, crops):
            self.calls += 1
            p = (HelmetPrediction(HelmetStatus.UNKNOWN, .63, .37, .63)
                 if self.calls == 1 else HelmetPrediction(HelmetStatus.HELMET, .68, .68, .32))
            return [p for _ in crops]

    pipeline = HelmetRiderPipeline(
        AppConfig(head_roi=HeadROIConfig(strategy="verified")),
        detector=FakeDetector([
            Detection("person", .9, (350, 80, 460, 340)),
            Detection("motorcycle", .9, (300, 250, 600, 400)),
            Detection("with_helmet", .75, (385, 90, 430, 150)),
        ]), classifier=TwoViews(),
    )
    for _ in range(2):
        r = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8)).riders[0]
        assert r.helmet.status == HelmetStatus.HELMET
        assert r.head_bbox == (385, 90, 430, 150)
        assert r.helmet.helmet_probability == .68


def test_verified_localization_requires_source_detail_and_agreement():
    from helmet_rider.config import HeadROIConfig

    pipeline = HelmetRiderPipeline(AppConfig(head_roi=HeadROIConfig(strategy="verified")),
                                   detector=FakeDetector([]), classifier=CountingClassifier())
    primary = HelmetPrediction(HelmetStatus.UNKNOWN, .63, .37, .63)
    positive = HelmetPrediction(HelmetStatus.HELMET, .68, .68, .32)
    evidence = (HelmetPrediction(HelmetStatus.HELMET, .75, 1, 0), (10, 10, 40, 50))
    assert pipeline._verified_localized_helmet(primary, positive, evidence)
    assert not pipeline._verified_localized_helmet(primary, positive, (evidence[0], (10, 10, 30, 50)))
    assert not pipeline._verified_localized_helmet(primary, positive,
        (HelmetPrediction(HelmetStatus.HELMET, .59, 1, 0), evidence[1]))
    assert not pipeline._verified_localized_helmet(
        HelmetPrediction(HelmetStatus.NO_HELMET, .95, .05, .95), positive, evidence)


class CountingClassifier:
    def __init__(self):
        self.calls = 0

    def classify(self, crop):
        self.calls += 1
        return HelmetPrediction(HelmetStatus.HELMET, 0.94, 0.94, 0.06)


class BatchCountingClassifier(CountingClassifier):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def classify(self, crop):
        raise AssertionError("Pipeline should use the batch classifier path")

    def classify_batch(self, crops):
        self.batch_sizes.append(len(crops))
        self.calls += len(crops)
        return [HelmetPrediction(HelmetStatus.HELMET, 0.94, 0.94, 0.06) for _ in crops]


class AlternatingClassifier(CountingClassifier):
    def classify(self, crop):
        self.calls += 1
        if self.calls == 1:
            return HelmetPrediction(HelmetStatus.HELMET, 0.94, 0.94, 0.06)
        if self.calls == 2:
            return HelmetPrediction(HelmetStatus.NO_HELMET, 0.91, 0.09, 0.91)
        return HelmetPrediction(HelmetStatus.UNKNOWN, 0.50, 0.50, 0.50)


class StrongNoHelmetClassifier(CountingClassifier):
    def classify(self, crop):
        self.calls += 1
        return HelmetPrediction(HelmetStatus.NO_HELMET, 0.96, 0.04, 0.96)


class BorderlineHelmetClassifier(CountingClassifier):
    def classify(self, crop):
        self.calls += 1
        return HelmetPrediction(HelmetStatus.HELMET, 0.65, 0.65, 0.35)


class BorderlineNoHelmetClassifier(CountingClassifier):
    def classify(self, crop):
        self.calls += 1
        return HelmetPrediction(HelmetStatus.UNKNOWN, 0.84, 0.16, 0.84)


class FocusedHelmetClassifier(CountingClassifier):
    def classify(self, crop):
        self.calls += 1
        if self.calls == 1:
            return HelmetPrediction(HelmetStatus.UNKNOWN, 0.84, 0.16, 0.84)
        return HelmetPrediction(HelmetStatus.HELMET, 0.77, 0.77, 0.23)


class StrongNoThenFocusedHelmetBatchClassifier(BatchCountingClassifier):
    def classify_batch(self, crops):
        self.batch_sizes.append(len(crops))
        self.calls += len(crops)
        if len(self.batch_sizes) == 1:
            return [HelmetPrediction(HelmetStatus.NO_HELMET, 0.96, 0.04, 0.96)]
        return [HelmetPrediction(HelmetStatus.HELMET, 0.84, 0.84, 0.16)]


def test_only_associated_persons_reach_classifier():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("person", 0.9, (20, 70, 90, 300)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
    ]
    classifier = CountingClassifier()
    config = AppConfig(detector=DetectorConfig(model="unused"))
    pipeline = HelmetRiderPipeline(
        config, detector=FakeDetector(detections), classifier=classifier
    )
    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(result.riders) == 1
    assert classifier.calls == 1
    assert result.riders[0].helmet.status == HelmetStatus.HELMET
    assert len(result.vehicles) == 1
    assert len(result.vehicles[0].riders) == 1


def test_weak_unassociated_motorcycles_are_hidden_but_strong_ones_remain():
    detections = [
        Detection("motorcycle", 0.59, (20, 200, 100, 400)),
        Detection("motorcycle", 0.60, (180, 200, 280, 400)),
    ]
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=CountingClassifier(),
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(result.vehicles) == 1
    assert result.vehicles[0].confidence == 0.60
    motorcycles = [item for item in result.detections if item.class_name == "motorcycle"]
    assert len(motorcycles) == 1
    assert motorcycles[0].track_id == 1


def test_weak_motorcycle_with_a_geometrically_associated_rider_is_kept():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.05, (300, 250, 600, 400)),
    ]
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=CountingClassifier(),
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(result.vehicles) == 1
    assert result.riders[0].vehicle_type == "motorcycle"


def test_batch_classifier_only_receives_associated_riders():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("person", 0.9, (20, 70, 90, 300)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
    ]
    classifier = BatchCountingClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert classifier.batch_sizes == [1]
    assert classifier.calls == 1
    assert result.riders[0].helmet.status == HelmetStatus.HELMET


def test_tiny_associated_rider_without_extractable_head_crop_does_not_crash_batching():
    detections = [
        Detection("person", 0.9, (300, 100, 400, 120)),
        Detection("motorcycle", 0.9, (250, 100, 500, 240)),
    ]
    classifier = BatchCountingClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(result.riders) == 1
    assert result.riders[0].head_bbox is None
    assert result.riders[0].helmet.status == HelmetStatus.UNKNOWN
    assert classifier.batch_sizes == []


def test_no_vehicle_means_classifier_is_not_called():
    detections = [Detection("person", 0.9, (350, 80, 460, 340))]
    classifier = CountingClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )
    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result.riders == ()
    assert classifier.calls == 0


def test_helmet_detector_evidence_is_assigned_only_to_an_associated_rider():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("person", 0.9, (20, 70, 90, 300)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("with_helmet", 0.62, (385, 90, 410, 120)),
        Detection("with_helmet", 0.88, (40, 80, 60, 100)),
    ]
    classifier = CountingClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(result.riders) == 1
    assert result.riders[0].helmet.status == HelmetStatus.HELMET
    assert result.riders[0].helmet.confidence == 0.62
    assert result.riders[0].helmet_bbox == (385, 90, 410, 120)
    assert classifier.calls == 1


def test_direct_no_helmet_call_is_overturned_by_strong_head_classifier_helmet_evidence():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("without_helmet", 0.93, (385, 90, 410, 120)),
    ]
    classifier = CountingClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert classifier.calls == 1
    assert result.riders[0].helmet.status == HelmetStatus.HELMET
    assert result.riders[0].helmet_bbox is None


def test_moderate_helmet_probability_overturns_a_conflicting_detector_negative():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("without_helmet", 0.93, (385, 90, 410, 120)),
    ]
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=BorderlineHelmetClassifier(),
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result.riders[0].helmet.status == HelmetStatus.HELMET
    assert result.riders[0].helmet.helmet_probability == 0.65


def test_direct_no_helmet_call_becomes_unknown_without_classifier_confirmation():
    from helmet_rider.helmet.classifier import UnknownHelmetClassifier

    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("without_helmet", 0.93, (385, 90, 410, 120)),
    ]
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=UnknownHelmetClassifier(),
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result.riders[0].helmet.status == HelmetStatus.UNKNOWN
    assert result.riders[0].helmet_bbox is None


def test_direct_no_helmet_call_needs_strong_independent_confirmation():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("without_helmet", 0.93, (385, 90, 410, 120)),
    ]
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=StrongNoHelmetClassifier(),
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result.riders[0].helmet.status == HelmetStatus.NO_HELMET
    assert result.riders[0].helmet.confidence == 0.96


def test_tiny_head_crop_cannot_trigger_a_no_helmet_label():
    detections = [
        Detection("person", 0.9, (100, 100, 126, 154)),
        Detection("motorcycle", 0.9, (80, 130, 180, 240)),
    ]
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=StrongNoHelmetClassifier(),
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result.riders[0].helmet.status == HelmetStatus.UNKNOWN


def test_borderline_no_helmet_classifier_stays_unknown_without_confirmation():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("without_helmet", 0.93, (385, 90, 410, 120)),
    ]
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=BorderlineNoHelmetClassifier(),
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result.riders[0].helmet.status == HelmetStatus.UNKNOWN


def test_focused_helmet_crop_can_correct_a_borderline_no_helmet_prediction():
    helmet_box = (385, 90, 430, 160)
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("without_helmet", 0.93, helmet_box),
    ]
    classifier = FocusedHelmetClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert classifier.calls == 2
    assert result.riders[0].helmet.status == HelmetStatus.HELMET
    assert result.riders[0].helmet_bbox is None


def test_focused_head_crop_can_correct_a_classifier_only_no_helmet_prediction():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
    ]
    classifier = StrongNoThenFocusedHelmetBatchClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert classifier.batch_sizes == [1, 1]
    assert result.riders[0].helmet.status == HelmetStatus.HELMET


def test_focused_head_crop_overrides_a_strong_but_conflicting_detector_negative():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("without_helmet", 0.93, (385, 90, 430, 160)),
    ]
    classifier = StrongNoThenFocusedHelmetBatchClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert classifier.batch_sizes == [1, 1]
    assert result.riders[0].helmet.status == HelmetStatus.HELMET
    assert result.riders[0].helmet_bbox is None


def test_live_no_helmet_label_requires_two_fresh_classifier_observations():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("without_helmet", 0.93, (385, 90, 410, 120)),
    ]
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=StrongNoHelmetClassifier(),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    first = pipeline.process(frame, temporal_consistency=True)
    second = pipeline.process(frame, temporal_consistency=True)

    assert first.riders[0].helmet.status == HelmetStatus.UNKNOWN
    assert second.riders[0].helmet.status == HelmetStatus.UNKNOWN
    assert second.riders[0].evidence["classifier_updated"] is False
    third = pipeline.process(frame, temporal_consistency=True)
    fourth = pipeline.process(frame, temporal_consistency=True)
    assert third.riders[0].helmet.status == HelmetStatus.UNKNOWN
    assert fourth.riders[0].helmet.status == HelmetStatus.NO_HELMET
    assert fourth.riders[0].evidence["classifier_updated"] is True


def test_helmet_evidence_is_ignored_for_low_confidence_person_boxes():
    detections = [
        Detection("person", 0.09, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("with_helmet", 0.0056, (385, 90, 410, 120)),
    ]
    classifier = CountingClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert result.riders == ()
    assert classifier.calls == 0


def test_strong_helmet_evidence_is_kept_for_low_confidence_person_boxes():
    detections = [
        Detection("person", 0.09, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("with_helmet", 0.5, (385, 90, 410, 120)),
    ]
    classifier = CountingClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(result.riders) == 1
    assert result.riders[0].helmet_bbox == (385, 90, 410, 120)
    assert result.riders[0].helmet.confidence == 0.5
    assert classifier.calls == 1


def test_filtered_low_confidence_person_does_not_shift_association_indices():
    detections = [
        Detection("person", 0.09, (350, 80, 460, 340)),
        Detection("person", 0.9, (455, 85, 555, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
    ]
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=CountingClassifier(),
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(result.riders) == 1
    assert result.riders[0].bbox == detections[1].bbox


def test_helmet_matching_prefers_confidence_over_small_overlap_advantage():
    head_rois = {0: (np.zeros((100, 100, 3), dtype=np.uint8), (0, 0, 100, 100))}
    persons = [Detection("person", 0.9, (0, 0, 100, 200))]
    weaker_box = Detection("with_helmet", 0.005, (30, 10, 70, 70))
    clear_box = Detection("with_helmet", 0.93, (10, -10, 70, 70))

    matched = HelmetRiderPipeline._match_helmet_detections(
        head_rois, [weaker_box, clear_box], persons
    )

    assert matched[0][1] == clear_box.bbox
    assert matched[0][0].confidence == clear_box.confidence


def test_weak_helmet_positive_is_not_committed_as_a_helmet():
    head_rois = {0: (np.zeros((100, 100, 3), dtype=np.uint8), (0, 0, 100, 100))}
    persons = [Detection("person", 0.9, (0, 0, 100, 200))]
    weak_box = Detection("with_helmet", 0.20, (30, 10, 70, 70))

    matched = HelmetRiderPipeline._match_helmet_detections(
        head_rois, [weak_box], persons, minimum_positive_confidence=0.35
    )

    assert matched == {}


def test_helmet_box_is_not_assigned_to_adjacent_rider_by_edge_overlap():
    head_rois = {
        0: (np.zeros((60, 50, 3), dtype=np.uint8), (0, 0, 50, 60)),
        1: (np.zeros((60, 50, 3), dtype=np.uint8), (40, 0, 90, 60)),
    }
    persons = [
        Detection("person", 0.9, (0, 0, 50, 200)),
        Detection("person", 0.9, (40, 0, 90, 200)),
    ]
    helmet_box = Detection("with_helmet", 0.80, (10, 5, 55, 40))

    matched = HelmetRiderPipeline._match_helmet_detections(
        head_rois, [helmet_box], persons
    )

    assert set(matched) == {0}


def test_nested_person_boxes_are_deduplicated_without_merging_adjacent_riders():
    detections = [
        Detection("person", 0.91, (100, 20, 180, 260)),
        Detection("person", 0.83, (102, 21, 178, 125)),
        Detection("person", 0.79, (170, 23, 245, 240)),
        Detection("motorcycle", 0.92, (90, 180, 250, 390)),
    ]

    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=CountingClassifier(),
    )

    deduplicated = pipeline._deduplicate_person_detections(detections)

    assert [item.bbox for item in deduplicated if item.class_name == "person"] == [
        detections[0].bbox,
        detections[2].bbox,
    ]
    assert deduplicated[-1] == detections[-1]


def test_conflicting_overlapping_helmet_classes_are_sent_to_the_classifier():
    detections = [
        Detection("person", 0.9, (350, 80, 460, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
        Detection("without_helmet", 0.93, (385, 90, 430, 160)),
        Detection("with_helmet", 0.72, (386, 91, 429, 159)),
    ]
    classifier = CountingClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )

    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))

    assert classifier.calls == 1
    assert result.riders[0].helmet.status == HelmetStatus.HELMET
    assert result.riders[0].helmet_bbox is None


def test_each_rider_on_one_motorcycle_gets_an_independent_prediction():
    detections = [
        Detection("person", 0.9, (350, 80, 450, 340)),
        Detection("person", 0.9, (455, 85, 555, 340)),
        Detection("motorcycle", 0.9, (300, 250, 600, 400)),
    ]
    classifier = AlternatingClassifier()
    pipeline = HelmetRiderPipeline(
        AppConfig(detector=DetectorConfig(model="unused")),
        detector=FakeDetector(detections),
        classifier=classifier,
    )
    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(result.vehicles[0].riders) == 2
    assert classifier.calls == 3
    assert {rider.helmet.status for rider in result.riders} == {
        HelmetStatus.HELMET,
        HelmetStatus.NO_HELMET,
    }


def test_untrained_classifier_unknown_is_a_valid_prediction():
    from helmet_rider.helmet.classifier import UnknownHelmetClassifier

    prediction = UnknownHelmetClassifier().classify(np.zeros((20, 20, 3), dtype=np.uint8))
    assert prediction.status == HelmetStatus.UNKNOWN


def test_source_seek_and_resolution_change_clear_stream_confirmation():
    detections = [Detection("person", .9, (350,80,460,340)),
                  Detection("motorcycle", .9, (300,250,600,400)),
                  Detection("without_helmet", .93, (385,90,410,120))]
    pipeline = HelmetRiderPipeline(AppConfig(detector=DetectorConfig(model="unused")),
                                   detector=FakeDetector(detections), classifier=StrongNoHelmetClassifier())
    frame = np.zeros((480,640,3), dtype=np.uint8)
    for timestamp in (1.,2.,3.,4.):
        result = pipeline.process(frame, temporal_consistency=True, timestamp_seconds=timestamp)
    assert result.riders[0].helmet.status == HelmetStatus.NO_HELMET
    sought = pipeline.process(frame, temporal_consistency=True, timestamp_seconds=.5)
    assert sought.riders[0].helmet.status == HelmetStatus.UNKNOWN
    changed = pipeline.process(np.zeros((600,800,3),dtype=np.uint8), temporal_consistency=True, timestamp_seconds=.6)
    assert changed.riders[0].helmet.status == HelmetStatus.UNKNOWN


def test_contradictory_detector_observation_clears_confirmation():
    detector = FakeDetector([Detection("person", .9, (350,80,460,340)),
                             Detection("motorcycle", .9, (300,250,600,400)),
                             Detection("without_helmet", .93, (385,90,410,120))])
    pipeline = HelmetRiderPipeline(AppConfig(detector=DetectorConfig(model="unused")),
                                   detector=detector, classifier=StrongNoHelmetClassifier())
    frame = np.zeros((480,640,3),dtype=np.uint8)
    pipeline.process(frame, temporal_consistency=True)
    detector.detections[-1] = Detection("with_helmet", .93, (385,90,410,120))
    pipeline.process(frame, temporal_consistency=True)
    assert not pipeline._no_helmet_hits_by_track
    detector.detections[-1] = Detection("without_helmet", .93, (385,90,410,120))
    pipeline.process(frame, temporal_consistency=True)
    result = pipeline.process(frame, temporal_consistency=True)
    assert result.riders[0].helmet.status == HelmetStatus.UNKNOWN
    assert result.riders[0].evidence["negative_confirmations"] == 1


def test_cap_detector_positive_is_overruled_by_independent_head_classifier():
    detector = FakeDetector([
        Detection("person", .9, (350, 80, 460, 340)),
        Detection("motorcycle", .9, (300, 250, 600, 400)),
        Detection("with_helmet", .97, (385, 90, 410, 120)),
    ])
    classifier = StrongNoHelmetClassifier()
    pipeline = HelmetRiderPipeline(AppConfig(), detector=detector, classifier=classifier)
    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert classifier.calls == 2
    assert result.riders[0].helmet.status == HelmetStatus.NO_HELMET
    assert result.riders[0].helmet_bbox is None
    assert result.riders[0].evidence["detector_status"] == "helmet"
    assert result.riders[0].evidence["classifier_updated"]


def test_detector_positive_without_independent_confirmation_is_unknown():
    from helmet_rider.helmet.classifier import UnknownHelmetClassifier

    detector = FakeDetector([
        Detection("person", .9, (350, 80, 460, 340)),
        Detection("motorcycle", .9, (300, 250, 600, 400)),
        Detection("with_helmet", .97, (385, 90, 410, 120)),
    ])
    pipeline = HelmetRiderPipeline(AppConfig(), detector=detector,
                                   classifier=UnknownHelmetClassifier())
    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result.riders[0].helmet.status == HelmetStatus.UNKNOWN
    assert result.riders[0].helmet_bbox is None


def test_positive_detector_uses_batch_classifier_and_cached_evidence():
    detector = FakeDetector([
        Detection("person", .9, (350, 80, 460, 340)),
        Detection("motorcycle", .9, (300, 250, 600, 400)),
        Detection("with_helmet", .97, (385, 90, 410, 120)),
    ])
    classifier = BatchCountingClassifier()
    pipeline = HelmetRiderPipeline(AppConfig(), detector=detector, classifier=classifier)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    first = pipeline.process(frame)
    second = pipeline.process(frame)
    assert classifier.batch_sizes == [1]
    assert first.riders[0].helmet.status == HelmetStatus.HELMET
    assert second.riders[0].helmet.status == HelmetStatus.HELMET
    assert first.riders[0].evidence["classifier_updated"]
    assert not second.riders[0].evidence["classifier_updated"]


def test_complete_matched_head_can_confirm_shifted_person_crop():
    detections = [
        Detection("person", .9, (350, 80, 460, 340)),
        Detection("motorcycle", .9, (300, 250, 600, 400)),
        Detection("with_helmet", .90, (385, 90, 430, 160)),
    ]
    classifier = FocusedHelmetClassifier()
    pipeline = HelmetRiderPipeline(AppConfig(), detector=FakeDetector(detections),
                                   classifier=classifier)
    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert classifier.calls == 2
    assert result.riders[0].helmet.status == HelmetStatus.HELMET
    assert result.riders[0].evidence["head_recheck_helmet_probability"] == .77


def test_occluded_vehicle_is_kept_even_when_rider_selects_overlapping_front_vehicle():
    detections = [
        Detection("person", .81, (173.87, 26.83, 291.07, 321.33)),
        Detection("motorcycle", .52, (3.94, 201.5, 264.76, 396.27)),
        Detection("motorcycle", .043, (234, 176.58, 282.15, 320.14)),
    ]
    pipeline = HelmetRiderPipeline(AppConfig(), detector=FakeDetector(detections),
                                   classifier=CountingClassifier())
    result = pipeline.process(np.zeros((400, 700, 3), dtype=np.uint8))
    assert len(result.vehicles) == 2
    assert len(result.riders) == 1


def test_matched_helmet_verification_is_batched():
    detector = FakeDetector([
        Detection("person", .9, (350, 80, 460, 340)),
        Detection("motorcycle", .9, (300, 250, 600, 400)),
        Detection("with_helmet", .90, (385, 90, 430, 160)),
    ])
    classifier = StrongNoThenFocusedHelmetBatchClassifier()
    pipeline = HelmetRiderPipeline(AppConfig(), detector=detector, classifier=classifier)
    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert classifier.batch_sizes == [1, 1]
    assert result.riders[0].helmet.status == HelmetStatus.HELMET
    assert result.riders[0].evidence["head_recheck_helmet_probability"] == .84


def test_small_estimated_head_requires_localized_negative_evidence():
    # An occluded person box can start at the torso and produce a confident
    # classifier negative on a 30px clothing patch, as in the traffic photo.
    detector = FakeDetector([
        Detection("person", .9, (30, 115, 66, 239)),
        Detection("motorcycle", .9, (20, 180, 100, 330)),
        Detection("without_helmet", .003, (34, 115, 62, 140)),
    ])
    pipeline = HelmetRiderPipeline(AppConfig(), detector=detector,
                                   classifier=StrongNoHelmetClassifier())
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    assert pipeline.process(frame).riders[0].helmet.status == HelmetStatus.UNKNOWN
    pipeline.reset()
    detector.detections[-1] = Detection("without_helmet", .9, (34, 115, 62, 140))
    assert pipeline.process(frame).riders[0].helmet.status == HelmetStatus.NO_HELMET


def test_people_and_frame_without_motorcycles_are_reported():
    from helmet_rider.utils.results import result_to_dict
    pipeline = HelmetRiderPipeline(AppConfig(), detector=FakeDetector([
        Detection("person", .9, (30, 30, 100, 200)),
        Detection("person", .2, (200, 30, 260, 200)),
    ]), classifier=CountingClassifier())
    result = pipeline.process(np.zeros((400, 400, 3), dtype=np.uint8))
    assert [p.motorcycle_status for p in result.persons] == ["no_motorcycle_match", "uncertain"]
    assert result_to_dict(result)["motorcycle_presence"] == "not_detected"
    assert not result.riders


def test_rejected_weak_motorcycle_does_not_turn_person_into_no_moto():
    pipeline = HelmetRiderPipeline(AppConfig(), detector=FakeDetector([
        Detection("person", .9, (30, 30, 100, 200)),
        Detection("motorcycle", .1, (60, 170, 160, 260)),
    ]), classifier=CountingClassifier())
    result = pipeline.process(np.zeros((400, 400, 3), dtype=np.uint8))
    assert not result.vehicles
    assert result.persons[0].motorcycle_status == "uncertain"


def test_skipped_detector_frame_cannot_establish_no_moto():
    from dataclasses import replace

    from helmet_rider.utils.results import result_to_dict
    config = AppConfig(detector=replace(DetectorConfig(), every_n_frames=3))
    pipeline = HelmetRiderPipeline(config, detector=FakeDetector([
        Detection("person", .9, (30, 30, 100, 200)),
    ]), classifier=CountingClassifier())
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    pipeline.process(frame)
    result = pipeline.process(frame)
    assert result_to_dict(result)["motorcycle_presence"] == "uncertain"
    assert result.persons[0].motorcycle_status == "uncertain"


def test_uncertain_crop_rechecks_localized_head_in_batch():
    class UncertainThenHelmet:
        def __init__(self): self.calls = 0
        def classify_batch(self, crops):
            self.calls += 1
            prediction = (HelmetPrediction(HelmetStatus.UNKNOWN, .6, .4, .6)
                          if self.calls == 1 else HelmetPrediction(HelmetStatus.HELMET, .9, .9, .1))
            return [prediction for _ in crops]
    classifier = UncertainThenHelmet()
    pipeline = HelmetRiderPipeline(AppConfig(), detector=FakeDetector([
        Detection("person", .9, (350, 80, 460, 340)),
        Detection("motorcycle", .9, (300, 250, 600, 400)),
        Detection("without_helmet", .8, (385, 90, 430, 150)),
    ]), classifier=classifier)
    result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert classifier.calls == 2
    assert result.riders[0].helmet.status == HelmetStatus.HELMET


def test_localized_recheck_rejects_borderline_negative_and_weak_helmet():
    class RecheckClassifier:
        def __init__(self, prediction):
            self.calls = 0
            self.prediction = prediction
        def classify_batch(self, crops):
            self.calls += 1
            prediction = (HelmetPrediction(HelmetStatus.UNKNOWN, .6, .4, .6)
                          if self.calls == 1 else self.prediction)
            return [prediction for _ in crops]
    detections = [
        Detection("person", .9, (350, 80, 460, 340)),
        Detection("motorcycle", .9, (300, 250, 600, 400)),
        Detection("without_helmet", .8, (385, 90, 430, 150)),
    ]
    for prediction in (HelmetPrediction(HelmetStatus.NO_HELMET, .94, .06, .94),
                       HelmetPrediction(HelmetStatus.HELMET, .70, .70, .30)):
        pipeline = HelmetRiderPipeline(AppConfig(), detector=FakeDetector(detections),
                                       classifier=RecheckClassifier(prediction))
        result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result.riders[0].helmet.status == HelmetStatus.UNKNOWN


def test_real_localized_head_can_replace_a_small_misplaced_estimated_crop():
    class LocalizedNoHelmet:
        def __init__(self): self.calls = 0
        def classify_batch(self, crops):
            self.calls += 1
            p = (HelmetPrediction(HelmetStatus.UNKNOWN, .6, .4, .6)
                 if self.calls == 1 else HelmetPrediction(HelmetStatus.NO_HELMET, .98, .02, .98))
            return [p for _ in crops]
    for width, expected in ((21, HelmetStatus.NO_HELMET), (19, HelmetStatus.UNKNOWN)):
        head_box = (356, 230, 356 + width, 258)
        pipeline = HelmetRiderPipeline(AppConfig(), detector=FakeDetector([
            Detection("person", .9, (350, 230, 385, 320)),
            Detection("motorcycle", .9, (330, 280, 410, 370)),
            Detection("without_helmet", .8, head_box),
        ]), classifier=LocalizedNoHelmet())
        result = pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result.riders[0].helmet.status == expected
        if expected == HelmetStatus.NO_HELMET:
            assert result.riders[0].head_bbox == head_box


def test_stream_requires_two_fresh_localized_negative_confirmations():
    class LocalizedNoHelmet:
        def classify_batch(self, crops):
            # The estimated head is wider than the actual detected head.
            return [HelmetPrediction(HelmetStatus.NO_HELMET, .98, .02, .98)
                    if crop.shape[1] < 40 else HelmetPrediction(HelmetStatus.UNKNOWN, .6, .4, .6)
                    for crop in crops]
    pipeline = HelmetRiderPipeline(AppConfig(), detector=FakeDetector([
        Detection("person", .9, (350, 80, 460, 340)),
        Detection("motorcycle", .9, (300, 250, 600, 400)),
        Detection("without_helmet", .8, (385, 90, 406, 120)),
    ]), classifier=LocalizedNoHelmet())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        assert pipeline.process(frame, temporal_consistency=True).riders[0].helmet.status == HelmetStatus.UNKNOWN
    result = pipeline.process(frame, temporal_consistency=True)
    assert result.riders[0].helmet.status == HelmetStatus.NO_HELMET
    assert result.riders[0].evidence["negative_confirmations"] == 2


def test_two_headwear_views_resolve_a_weak_helmet_detection():
    class UncertainPrimary:
        def classify_batch(self, crops):
            return [HelmetPrediction(HelmetStatus.UNKNOWN, .65, .35, .65) for _ in crops]
    class Headwear:
        def __init__(self): self.calls = 0
        def classify_batch(self, crops):
            self.calls += 1
            return [HelmetPrediction(HelmetStatus.NO_HELMET, .82, .18, .82),
                    HelmetPrediction(HelmetStatus.NO_HELMET, .78, .22, .78)]
    headwear = Headwear()
    head_box = (385, 90, 415, 130)
    pipeline = HelmetRiderPipeline(AppConfig(), detector=FakeDetector([
        Detection("person", .9, (350, 80, 460, 340)),
        Detection("motorcycle", .9, (300, 250, 600, 400)),
        Detection("with_helmet", .4, head_box),
    ]), classifier=UncertainPrimary(), headwear_classifier=headwear)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        assert pipeline.process(frame, temporal_consistency=True).riders[0].helmet.status == HelmetStatus.UNKNOWN
    result = pipeline.process(frame, temporal_consistency=True)
    assert headwear.calls == 2  # Fresh classification frames, not cached frames.
    assert result.riders[0].helmet.status == HelmetStatus.NO_HELMET
    assert result.riders[0].head_bbox == head_box
    assert result.riders[0].evidence["headwear_no_helmet_probability"] == .78


def test_headwear_verification_rejects_conflict_and_strong_helmet_evidence():
    class Headwear:
        def __init__(self): self.calls = 0
        def classify_batch(self, crops):
            self.calls += 1
            return [HelmetPrediction(HelmetStatus.NO_HELMET, .9, .1, .9),
                    HelmetPrediction(HelmetStatus.UNKNOWN, .7, .3, .7)]
    headwear = Headwear()
    pipeline = HelmetRiderPipeline(AppConfig(), detector=FakeDetector([]),
                                   classifier=CountingClassifier(), headwear_classifier=headwear)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    uncertain = HelmetPrediction(HelmetStatus.UNKNOWN, .65, .35, .65)
    positive = HelmetPrediction(HelmetStatus.HELMET, .9, .9, .1)
    evidence = (HelmetPrediction(HelmetStatus.HELMET, .4, 1, 0), (385, 90, 415, 130))
    assert pipeline._verify_headwear_negative(frame, frame[:30, :30], uncertain, uncertain, evidence) is None
    assert headwear.calls == 1
    assert pipeline._verify_headwear_negative(frame, frame[:30, :30], positive, uncertain, evidence) is None
    strong_evidence = (HelmetPrediction(HelmetStatus.HELMET, .9, 1, 0), evidence[1])
    assert pipeline._verify_headwear_negative(frame, frame[:30, :30], uncertain, uncertain, strong_evidence) is None
    assert headwear.calls == 1


def test_positive_recheck_requires_supported_views_and_source_pixels():
    verify = HelmetRiderPipeline._positive_view_confidence
    assert verify(.68, 29, .43, .94, .41) == .94  # shell on credible head
    assert verify(.018, 26, .78, .88, .85) == .78  # weak localization, agreement
    assert verify(.32, 29, .78, .53, .55) == .78  # full-face helmet
    assert verify(.64, 18, .40, .76, .77) == .76  # tiny head, two positive views
    assert verify(.40, 29, .34, .56, .36) is None  # red cap
    assert verify(.01, 26, .40, .97, .95) is None  # weak box, no full-head support
    assert verify(.64, 18, .40, .90, .70) is None  # tiny head, disagreement
    assert verify(.99, 11, .99, .99, .99) is None  # insufficient real pixels


def test_positive_recheck_preserves_negative_and_expires_on_fresh_disagreement():
    from dataclasses import replace

    from helmet_rider.config import HeadROIConfig
    from helmet_rider.detection.types import RiderResult

    class Views:
        calls = 0
        positive = True
        def classify_batch(self, crops):
            self.calls += 1
            probability = .9 if self.positive else .4
            return [HelmetPrediction(HelmetStatus.UNKNOWN, probability, probability, 1-probability)
                    for _ in crops]

    classifier = Views()
    pipeline = HelmetRiderPipeline(AppConfig(head_roi=HeadROIConfig(strategy="verified")),
                                   detector=FakeDetector([]), classifier=classifier)
    frame = np.zeros((150, 150, 3), dtype=np.uint8)
    persons = [Detection("person", .9, (10, 10, 80, 140), track_id=1)]
    detections = [Detection("with_helmet", .01, (25, 10, 55, 40))]
    rider = RiderResult(1, persons[0].bbox, 2, "motorcycle", (25, 10, 55, 40),
                       HelmetPrediction(HelmetStatus.UNKNOWN, .5, .5, .5), .9,
                       evidence={"classifier_updated": True})
    assert pipeline._recheck_uncertain_helmets(frame, [rider], persons, detections)[0].helmet.status == HelmetStatus.HELMET
    cached = replace(rider, evidence={"classifier_updated": False})
    assert pipeline._recheck_uncertain_helmets(frame, [cached], persons, detections)[0].helmet.status == HelmetStatus.HELMET
    assert classifier.calls == 1
    classifier.positive = False
    assert pipeline._recheck_uncertain_helmets(frame, [rider], persons, detections)[0].helmet.status == HelmetStatus.UNKNOWN
    negative = replace(rider, helmet=HelmetPrediction(HelmetStatus.NO_HELMET, .95, .05, .95))
    classifier.positive = True
    assert pipeline._recheck_uncertain_helmets(frame, [negative], persons, detections)[0].helmet.status == HelmetStatus.NO_HELMET
    assert classifier.calls == 2
    pipeline.reset()
    assert not pipeline._positive_recheck_by_track

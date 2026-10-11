from helmet_rider.config import TrackingConfig
from helmet_rider.detection.types import Detection, HelmetPrediction, HelmetStatus
from helmet_rider.tracking.temporal_smoothing import TemporalSmoother
from helmet_rider.tracking.tracker import IoUTracker


def test_iou_tracker_keeps_id_for_overlapping_frames():
    tracker = IoUTracker()
    first = tracker.update([Detection("person", 0.9, (10, 10, 80, 160))])[0]
    second = tracker.update([Detection("person", 0.8, (12, 12, 82, 162))])[0]
    assert first.track_id == second.track_id


def test_tracker_predicts_between_detector_frames():
    tracker = IoUTracker()
    first = tracker.update([Detection("person", 0.9, (10, 10, 80, 160))])[0]
    assert tracker.predict()[0].track_id == first.track_id
    updated = tracker.update([Detection("person", 0.9, (14, 10, 84, 160))])[0]
    assert updated.track_id == first.track_id
    predicted = tracker.predict()[0]
    assert predicted.bbox[0] > updated.bbox[0]


def test_tracker_smooths_small_box_jitter_without_changing_identity():
    tracker = IoUTracker(TrackingConfig(bbox_smoothing=0.5))
    first = tracker.update([Detection("motorcycle", 0.9, (10, 10, 80, 120))])[0]
    second = tracker.update([Detection("motorcycle", 0.9, (20, 10, 90, 120))])[0]

    assert second.track_id == first.track_id
    assert 10 < second.bbox[0] < 20


def test_temporal_smoothing_reduces_one_frame_flip():
    smoother = TemporalSmoother(window_size=3)
    first = smoother.update(4, HelmetPrediction(HelmetStatus.HELMET, 0.92, 0.92, 0.08))
    second = smoother.update(4, HelmetPrediction(HelmetStatus.NO_HELMET, 0.62, 0.38, 0.62))
    assert first.status == HelmetStatus.HELMET
    assert second.status == HelmetStatus.HELMET


def test_temporal_history_survives_short_association_gaps():
    smoother = TemporalSmoother(window_size=3)
    smoother.update(
        8, HelmetPrediction(HelmetStatus.HELMET, 0.9, 0.9, 0.1), frame_number=2
    )
    smoother.forget_except(set(), frame_number=8, max_age=10)
    result = smoother.update(
        8, HelmetPrediction(HelmetStatus.NO_HELMET, 0.6, 0.4, 0.6), frame_number=9
    )
    assert result.status == HelmetStatus.HELMET


def test_timestamped_tracker_expires_tracks_by_seconds():
    tracker = IoUTracker(TrackingConfig(max_age_seconds=1.0))
    first = tracker.update([Detection("person",.9,(10,10,80,160))], timestamp_seconds=1.)[0]
    second = tracker.update([Detection("person",.9,(10,10,80,160))], timestamp_seconds=3.)[0]
    assert first.track_id != second.track_id


def test_prediction_scales_with_elapsed_source_time():
    tracker = IoUTracker(TrackingConfig(bbox_smoothing=1., max_age_seconds=5.))
    tracker.update([Detection("person",.9,(10,10,80,160))], timestamp_seconds=0.)
    observed = tracker.update([Detection("person",.9,(20,10,90,160))], timestamp_seconds=1.)[0]
    predicted = tracker.predict(timestamp_seconds=3.)[0]
    assert predicted.track_id == observed.track_id
    assert predicted.bbox[0] == 30.


def test_bytetrack_candidate_keeps_classes_separate_and_expires_by_time():
    import pytest
    pytest.importorskip('lap')
    from helmet_rider.tracking.byte_tracker import ByteTrackAdapter
    tracker = ByteTrackAdapter(TrackingConfig(backend='bytetrack', max_age_seconds=1.))
    detections = [Detection('person',.9,(10,10,80,160)), Detection('motorcycle',.9,(10,10,80,160))]
    first = tracker.update(detections,timestamp_seconds=0.)
    second = tracker.update(detections,timestamp_seconds=.1)
    assert len(first)==len(second)==2
    assert {d.track_id for d in first} == {d.track_id for d in second}
    assert len({d.track_id for d in first}) == 2
    expired = tracker.update(detections,timestamp_seconds=3.)
    assert not {d.track_id for d in first}.intersection(d.track_id for d in expired)

from dataclasses import replace

import numpy as np

from helmet_rider.config import AppConfig, TrackingConfig
from helmet_rider.detection.types import Detection, FrameResult, HelmetStatus, PersonResult
from helmet_rider.helmet.head_outputs import HeadOutputBuilder
from helmet_rider.utils.results import result_to_dict
from helmet_rider.visualization.draw import draw_result, result_summary


def test_two_real_heads_survive_one_body_without_inventing_another_person():
    builder=HeadOutputBuilder(TrackingConfig())
    heads=[Detection('with_helmet',.9,(30,10,48,30)),Detection('without_helmet',.8,(52,12,70,32))]
    person=PersonResult(4,(20,10,85,150),.9,'motorcycle',6)
    results=builder.build(heads,[person])
    assert len(results)==2
    assert {r.person_id for r in results}=={4}
    assert {r.vehicle_id for r in results}=={6}
    assert [r.head_bbox for r in results]==[h.bbox for h in heads]
    assert results[0].helmet.status==HelmetStatus.HELMET
    assert results[1].helmet.status==HelmetStatus.NO_HELMET


def test_physical_head_never_implies_no_helmet_without_classifier():
    builder = HeadOutputBuilder(TrackingConfig())
    head = Detection('physical_head', .99, (10, 10, 30, 30))
    person = PersonResult(1, (0, 0, 40, 100), .9, 'uncertain')
    for timestamp in (0, .1, .2):
        result = builder.build([head], [person], temporal=True, timestamp=timestamp)[0]
        assert result.helmet.status == HelmetStatus.UNKNOWN
        assert result.helmet.helmet_probability == 0
        assert result.helmet.no_helmet_probability == 0
        assert result.helmet.confidence == 0
        assert result.confidence == .99


def test_physical_head_can_be_classified_from_its_actual_crop():
    from helmet_rider.detection.types import HelmetPrediction
    from helmet_rider.helmet.head_roi import HeadROIExtractor

    class Classifier:
        def classify_batch(self, crops):
            assert len(crops) == 1
            return [HelmetPrediction(HelmetStatus.HELMET, .95, .95, .05)]

    builder = HeadOutputBuilder(TrackingConfig(), classifier=Classifier(),
                                extractor=HeadROIExtractor(), verification_threshold=.9)
    head = Detection('physical_head', .99, (10, 10, 30, 30))
    person = PersonResult(1, (0, 0, 40, 100), .9, 'uncertain')
    result = builder.build([head], [person], frame=np.zeros((100, 100, 3), dtype=np.uint8))[0]
    assert result.helmet.status == HelmetStatus.HELMET


def test_confident_background_without_body_cannot_be_reported_as_a_head():
    builder=HeadOutputBuilder(TrackingConfig())
    head=Detection('with_helmet',.99,(10,10,30,30))
    assert builder.build([head],[])==()
    person=PersonResult(1,(0,0,40,100),.9,'uncertain')
    result=builder.build([head],[person])[0]
    assert result.motorcycle_status=='uncertain'
    assert result.vehicle_id is None


def test_arm_and_oversized_box_cannot_be_validated_as_a_head():
    builder=HeadOutputBuilder(TrackingConfig())
    person=PersonResult(1,(0,0,40,100),.9,'uncertain')
    assert builder.build([Detection('with_helmet',.9,(10,50,30,70))],[person])==()
    assert builder.build([Detection('with_helmet',.9,(0,0,80,60))],[person])==()


def test_weak_head_needs_a_supported_person_and_weak_person_cannot_validate_it():
    builder=HeadOutputBuilder(TrackingConfig(),min_confidence=.05)
    head=Detection('with_helmet',.08,(10,10,30,30))
    person=PersonResult(1,(0,0,40,100),.09,'uncertain')
    assert builder.build([head],[])==()
    assert builder.build([head],[person])==()
    assert len(builder.build([head],[replace(person,confidence=.5)]))==1
    negative=replace(head,class_name='without_helmet')
    strong=replace(person,confidence=.5)
    assert builder.build([negative],[strong],temporal=True,timestamp=0)[0].helmet.status==HelmetStatus.UNKNOWN
    assert builder.build([negative],[strong],temporal=True,timestamp=.1)[0].helmet.status==HelmetStatus.NO_HELMET
    assert builder.build([negative],[],temporal=True,timestamp=.2)==()
    assert builder.build([negative],[strong],temporal=True,timestamp=.3)[0].helmet.status==HelmetStatus.UNKNOWN


def test_negative_needs_two_observations_and_resets_after_absence_gap_or_owner_change():
    builder=HeadOutputBuilder(TrackingConfig(bbox_smoothing=1))
    head=Detection('without_helmet',.9,(10,10,30,30))
    person=PersonResult(1,(0,0,40,100),.9,'uncertain')
    def process(timestamp,people=(person,)):
        return builder.build([head],people,temporal=True,timestamp=timestamp)[0].helmet.status
    assert process(0)==HelmetStatus.UNKNOWN
    assert process(.1)==HelmetStatus.NO_HELMET
    assert process(4)==HelmetStatus.UNKNOWN
    assert process(4.1)==HelmetStatus.NO_HELMET
    person=PersonResult(4,(0,0,40,100),.9,'no_motorcycle_match')
    assert process(4.2,[person])==HelmetStatus.UNKNOWN
    assert process(4.3,[person])==HelmetStatus.NO_HELMET
    assert builder.build([],[],temporal=True,timestamp=4.4)==()
    assert process(4.5,[person])==HelmetStatus.UNKNOWN
    builder.reset()
    assert process(4.6,[person])==HelmetStatus.UNKNOWN


def test_independent_head_outputs_render_and_count_without_changing_rider_count():
    heads=HeadOutputBuilder(TrackingConfig()).build([Detection('with_helmet',.9,(10,10,30,30))],[PersonResult(1,(0,0,40,100),.9,'uncertain')])
    result=FrameResult(100,100,(),(),(),heads=heads)
    assert result_summary(result)['helmet']==1
    assert result_summary(result)['riders']==0
    assert result_to_dict(result)['heads'][0]['person_id']==1
    assert draw_result(np.zeros((100,100,3),dtype=np.uint8),result,AppConfig()).any()
    assert result_summary(replace(result,heads=()))['helmet']==0


def test_actual_detected_crops_are_classified_separately_and_negative_votes_survive_detector_label_flip():
    from helmet_rider.detection.types import HelmetPrediction
    from helmet_rider.helmet.head_roi import HeadROIExtractor
    class Classifier:
        def __init__(self):
            self.count=0

        def classify_batch(self,crops):
            self.count+=len(crops)
            return [HelmetPrediction(HelmetStatus.NO_HELMET,.95,.05,.95) for _ in crops]

    classifier=Classifier()
    builder=HeadOutputBuilder(TrackingConfig(bbox_smoothing=1),classifier=classifier,extractor=HeadROIExtractor())
    frame=np.ones((100,100,3),dtype=np.uint8)
    head=Detection('without_helmet',.9,(10,10,30,30))
    person=PersonResult(1,(0,0,40,100),.9,'uncertain')
    first=builder.build([head],[person],frame=frame,temporal=True,timestamp=0)[0]
    second=builder.build([replace(head,class_name='with_helmet')],[person],frame=frame,temporal=True,timestamp=.1)[0]
    assert first.track_id==second.track_id
    assert first.helmet.status==HelmetStatus.UNKNOWN
    assert second.helmet.status==HelmetStatus.NO_HELMET
    assert classifier.count==2
    builder.reset()
    assert builder.build([head],[person],frame=frame,temporal=True,timestamp=.2)[0].helmet.status==HelmetStatus.UNKNOWN


def test_verified_unowned_head_is_retained_without_inventing_motorcycle_absence():
    from helmet_rider.helmet.head_roi import HeadROIExtractor

    class Validator:
        def validate_batch(self,crops):
            return [.1,.9]

    builder=HeadOutputBuilder(TrackingConfig(),head_validator=Validator(),extractor=HeadROIExtractor())
    heads=[Detection('with_helmet',.99,(70,10,90,30)),Detection('without_helmet',.9,(10,10,30,30))]
    result=builder.build(heads,[],frame=np.ones((100,100,3),dtype=np.uint8))
    assert len(result)==1
    assert result[0].head_bbox==heads[1].bbox
    assert result[0].person_id is None
    assert result[0].motorcycle_status=='uncertain'
    assert result[0].helmet.status==HelmetStatus.NO_HELMET


def test_person_support_cannot_override_background_rejection_and_missing_validity_clears_negative_votes():
    from helmet_rider.helmet.head_roi import HeadROIExtractor

    class Validator:
        score=.9

        def validate_batch(self,crops):
            return [self.score for _ in crops]

    validator=Validator()
    builder=HeadOutputBuilder(TrackingConfig(bbox_smoothing=1),head_validator=validator,extractor=HeadROIExtractor())
    head=Detection('without_helmet',.99,(10,10,30,30))
    person=PersonResult(1,(0,0,40,100),.9,'uncertain')
    frame=np.ones((100,100,3),dtype=np.uint8)
    assert builder.build([head],[person],frame=frame,temporal=True,timestamp=0)[0].helmet.status==HelmetStatus.UNKNOWN
    assert builder.build([head],[person],frame=frame,temporal=True,timestamp=.1)[0].helmet.status==HelmetStatus.NO_HELMET
    validator.score=.1
    assert builder.build([head],[person],frame=frame,temporal=True,timestamp=.2)==()
    validator.score=.9
    assert builder.build([head],[person],frame=frame,temporal=True,timestamp=.3)[0].helmet.status==HelmetStatus.UNKNOWN


def test_selective_verification_keeps_weak_disagreement_and_confirms_strong_negative():
    from helmet_rider.detection.types import HelmetPrediction
    from helmet_rider.helmet.head_roi import HeadROIExtractor

    class Classifier:
        probability = .8

        def classify_batch(self, crops):
            return [HelmetPrediction(HelmetStatus.NO_HELMET, self.probability,
                                     1-self.probability, self.probability) for _ in crops]

    classifier = Classifier()
    builder = HeadOutputBuilder(TrackingConfig(bbox_smoothing=1), classifier=classifier,
                                extractor=HeadROIExtractor(), verification_threshold=.9)
    frame = np.ones((100, 100, 3), dtype=np.uint8)
    head = Detection('with_helmet', .95, (10, 10, 30, 30))
    person = PersonResult(1, (0, 0, 40, 100), .9, 'uncertain')
    def status(t):
        return builder.build([head], [person], frame=frame, temporal=True, timestamp=t)[0].helmet.status
    assert status(0) == HelmetStatus.HELMET
    classifier.probability = .95
    assert status(.1) == HelmetStatus.UNKNOWN
    assert status(.2) == HelmetStatus.NO_HELMET
    classifier.probability = .05
    assert status(.3) == HelmetStatus.HELMET
    classifier.probability = .95
    assert status(.4) == HelmetStatus.UNKNOWN

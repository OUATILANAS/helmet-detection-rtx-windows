"""Two-stage detector → association → head crop → classifier pipeline."""

from __future__ import annotations

import math
from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np

from helmet_rider.association.geometry import iou
from helmet_rider.association.rider_vehicle import RiderVehicleAssociator
from helmet_rider.config import AppConfig
from helmet_rider.detection.detector import (
    ObjectDetector,
    UltralyticsHeadDetector,
    UltralyticsPersonDetector,
    create_detector,
)
from helmet_rider.detection.types import (
    BBox,
    Detection,
    FrameResult,
    FrameTimings,
    HelmetPrediction,
    HelmetStatus,
    PersonResult,
    RiderResult,
    VehicleResult,
)
from helmet_rider.helmet.classifier import HelmetClassifier, create_classifier
from helmet_rider.helmet.head_outputs import HeadOutputBuilder
from helmet_rider.helmet.head_roi import HeadROIExtractor
from helmet_rider.helmet.localized import LocalizedHeadRefiner
from helmet_rider.tracking.byte_tracker import create_tracker
from helmet_rider.tracking.temporal_smoothing import TemporalSmoother

HELMET_DETECTION_CLASSES = frozenset({"with_helmet", "without_helmet"})
# Tiny crops can produce highly confident but visibly wrong no-helmet scores.
# Estimated crops need 24 source pixels. Credible localized heads have a
# separate 20-pixel gate and require a 0.95 classifier negative.
MIN_NO_HELMET_HEAD_EDGE_PX = 24
MIN_CLASSIFIER_ONLY_NO_HELMET_HEAD_EDGE_PX = 40
MIN_LOCALIZED_NO_HELMET_HEAD_EDGE_PX = 20


class HelmetRiderPipeline:
    """Classify only people associated with a motorcycle or bicycle."""

    def __init__(
        self,
        config: AppConfig,
        detector: ObjectDetector | None = None,
        classifier: HelmetClassifier | None = None,
        headwear_classifier: HelmetClassifier | None = None,
    ) -> None:
        self.config = config
        thread_count = max(1, config.performance.inference_threads)
        cv2.setNumThreads(thread_count)
        self.detector = detector or create_detector(config.detector, thread_count)
        self.classifier = classifier or create_classifier(config.classifier, thread_count)
        self.headwear_classifier = headwear_classifier
        if self.headwear_classifier is None and config.classifier.headwear_model:
            self.headwear_classifier = create_classifier(
                replace(config.classifier, model=config.classifier.headwear_model,
                        headwear_model=None), thread_count
            )
        self.associator = RiderVehicleAssociator(config.association)
        self.roi_extractor = HeadROIExtractor(config.head_roi)
        self.localized_refiner = None
        self.localized_head_detector = None
        self.localized_person_detector = None
        if config.classifier.localized_person_model:
            self.localized_person_detector = UltralyticsPersonDetector(
                config.classifier.localized_person_model, device=config.detector.device,
                threads=thread_count, tile_grid=config.classifier.localized_person_tile_grid,
                include_vehicles=config.classifier.localized_recover_vehicles,
            )
        if config.classifier.localized_refinement_model:
            localized_classifier = create_classifier(
                replace(config.classifier, model=config.classifier.localized_refinement_model,
                        headwear_model=None), thread_count
            )
            self.localized_refiner = LocalizedHeadRefiner(
                localized_classifier, self.roi_extractor,
                threshold=config.classifier.localized_refinement_threshold,
                max_age=config.classifier.evidence_max_age_seconds,
                native_threshold=config.classifier.localized_native_threshold,
                relocalize_known=config.classifier.localized_relocalize_known,
                relocalize_all_known=config.classifier.localized_relocalize_all_known,
                verifier=self.headwear_classifier if config.classifier.localized_verify_conflicts else None,
                negative_threshold=config.classifier.no_helmet_negative_threshold,
                prefer_consensus=config.classifier.localized_prefer_consensus,
                verify_known=config.classifier.localized_verify_known,
                occlusion_tolerant=config.classifier.localized_occlusion_tolerant,
                head_min_confidence=config.classifier.localized_head_min_confidence,
            )
            if config.classifier.localized_head_model:
                self.localized_head_detector = UltralyticsHeadDetector(
                    config.classifier.localized_head_model, device=config.detector.device,
                    threads=thread_count,
                    tile_grid=config.classifier.localized_head_tile_grid,
                    adaptive_tiles=config.classifier.localized_adaptive_tiles,
                    tile_size=config.classifier.localized_head_tile_size,
                )
        self.tracker = create_tracker(config.tracking)
        head_validator = None
        if config.classifier.localized_head_validity_model:
            from helmet_rider.helmet.head_validity import EfficientNetHeadValidator
            head_validator = EfficientNetHeadValidator(config.classifier.localized_head_validity_model, config.classifier, thread_count)
        independent_classifier = None
        if config.classifier.localized_independent_classify:
            independent_classifier = (create_classifier(
                replace(config.classifier, model=config.classifier.localized_independent_classifier_model,
                        headwear_model=None), thread_count)
                if config.classifier.localized_independent_classifier_model else self.localized_refiner.classifier)
        self.head_output_builder = HeadOutputBuilder(
            config.tracking, min_confidence=config.classifier.localized_head_min_confidence,
            classifier=independent_classifier,
            verification_threshold=config.classifier.localized_independent_verification_threshold,
            extractor=self.roi_extractor,
            classification_threshold=config.classifier.localized_refinement_threshold,
            head_validator=head_validator,
            head_validity_threshold=config.classifier.localized_head_validity_threshold,
            unassociated_validity_threshold=config.classifier.localized_unassociated_head_validity_threshold,
        ) if config.classifier.localized_independent_heads else None
        self.smoother = TemporalSmoother(
            config.tracking.smoothing_window, config.classifier
        )
        self.frame_number = 0
        self._cached_detections: list[Detection] = []
        self._helmet_by_track = {}
        self._classifier_by_track: dict[int, HelmetPrediction] = {}
        self._focused_helmet_by_track: dict[int, HelmetPrediction] = {}
        self._positive_recheck_by_track: dict[int, tuple[HelmetPrediction, BBox, dict]] = {}
        self._small_no_helmet_by_track: dict[int, dict] = {}
        self._localized_negative_by_track: dict[int, tuple[HelmetPrediction, BBox]] = {}
        self._headwear_negative_by_track: dict[int, tuple[HelmetPrediction, BBox]] = {}
        self._last_classification_frame: dict[int, int] = {}
        self._no_helmet_hits_by_track: dict[int, int] = {}
        self._last_timestamp: float | None = None
        self._last_shape: tuple | None = None
        self._classification_time: dict[int, float] = {}
        self._evidence_time: dict[int, float] = {}


    def reset(self) -> None:
        """Clear tracking and temporal labels before a new independent source."""
        if self.head_output_builder is not None:
            self.head_output_builder.reset()
        self.frame_number = 0
        self._cached_detections = []
        self._helmet_by_track.clear()
        self._classifier_by_track.clear()
        self._focused_helmet_by_track.clear()
        self._positive_recheck_by_track.clear()
        self._small_no_helmet_by_track.clear()
        self._localized_negative_by_track.clear()
        self._headwear_negative_by_track.clear()
        self._last_classification_frame.clear()
        self._no_helmet_hits_by_track.clear()
        self._last_timestamp = None
        self._last_shape = None
        self._classification_time.clear()
        self._evidence_time.clear()
        if self.localized_refiner is not None:
            self.localized_refiner.reset()
        self.tracker = create_tracker(self.config.tracking)
        self.smoother = TemporalSmoother(
            self.config.tracking.smoothing_window, self.config.classifier
        )

    def process(
        self, frame: np.ndarray, *, temporal_consistency: bool = False,
        timestamp_seconds: float | None = None
    ) -> FrameResult:
        started = perf_counter()
        if frame is None or frame.ndim < 2 or frame.size == 0:
            raise ValueError("Input frame must be a non-empty image array")
        if timestamp_seconds is not None and (not math.isfinite(timestamp_seconds) or timestamp_seconds < 0):
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if (self._last_shape is not None and frame.shape[:2] != self._last_shape) or (
            timestamp_seconds is not None and self._last_timestamp is not None
            and (timestamp_seconds <= self._last_timestamp or
                 timestamp_seconds - self._last_timestamp > self.config.tracking.reset_gap_seconds)
        ):
            self.reset()
        self._last_timestamp = timestamp_seconds
        self._last_shape = frame.shape[:2]
        if timestamp_seconds is not None:
            for rider_id in tuple(self._evidence_time):
                if timestamp_seconds - min(self._evidence_time[rider_id], self._classification_time.get(rider_id, self._evidence_time[rider_id])) > self.config.classifier.evidence_max_age_seconds:
                    for cache in (self._helmet_by_track, self._classifier_by_track,
                                  self._focused_helmet_by_track, self._positive_recheck_by_track,
                                  self._last_classification_frame,
                                  self._localized_negative_by_track,
                                  self._headwear_negative_by_track,
                                  self._no_helmet_hits_by_track, self._classification_time, self._evidence_time):
                        cache.pop(rider_id, None)
                    self.smoother.forget(rider_id)
        self.frame_number += 1
        height, width = frame.shape[:2]

        stage = perf_counter()
        detector_period = max(1, self.config.detector.every_n_frames)
        detector_ran = (
            not self._cached_detections
            or (self.frame_number - 1) % detector_period == 0
        )
        localized_heads = None
        if detector_ran:
            base_detections = self.detector.detect(frame)
            # Recovery tiles matter when full-frame inference shrinks source
            # pixels. Avoid adding redundant full-frame people on smaller
            # images, where the primary detector already sees source detail.
            if (self.localized_person_detector is not None
                    and max(frame.shape[:2]) > self.config.detector.image_size):
                base_detections = [*base_detections, *self.localized_person_detector.detect(frame)]
                if self.config.classifier.localized_recover_vehicles:
                    kept_vehicles = []
                    for detection in sorted((d for d in base_detections if d.class_name in {"motorcycle", "bicycle"}), key=lambda d: d.confidence, reverse=True):
                        if all(detection.class_name != other.class_name or iou(detection.bbox, other.bbox) < .45 for other in kept_vehicles):
                            kept_vehicles.append(detection)
                    base_detections = [d for d in base_detections if d.class_name not in {"motorcycle", "bicycle"}] + kept_vehicles
            raw_detections = self._deduplicate_person_detections(
                base_detections
            )
            helmet_detections = [
                item for item in raw_detections
                if item.class_name in HELMET_DETECTION_CLASSES
            ]
            if self.config.classifier.localized_use_as_primary and self.localized_head_detector is not None:
                localized_heads = self.localized_head_detector.detect(frame)
                helmet_detections = localized_heads
                raw_detections = [d for d in raw_detections if d.class_name not in HELMET_DETECTION_CLASSES] + localized_heads
            self._cached_detections = self._gate_unassociated_motorcycles([
                item for item in raw_detections
                if item.class_name not in HELMET_DETECTION_CLASSES
            ])
        else:
            raw_detections = self._cached_detections
            helmet_detections = []
        detections = list(self._cached_detections) if detector_ran else []
        detection_ms = (perf_counter() - stage) * 1000.0

        stage = perf_counter()
        if self.config.tracking.enabled:
            detections = (
                self.tracker.update(detections, timestamp_seconds=timestamp_seconds)
                if detector_ran else self.tracker.predict(timestamp_seconds=timestamp_seconds)
            )
        else:
            if not detector_ran:
                detections = self._cached_detections
            detections = [replace(item, track_id=index + 1)
                          for index, item in enumerate(detections)]
        detections.extend(helmet_detections)
        tracking_ms = (perf_counter() - stage) * 1000.0
        detections = self._gate_unassociated_motorcycles(detections)
        persons = [item for item in detections if item.class_name == "person"]
        vehicles = [
            item for item in detections if item.class_name in {"motorcycle", "bicycle"}
        ]

        stage = perf_counter()
        candidate_associations = self.associator.associate(persons, vehicles)
        all_scores = self.associator.diagnostics(persons, vehicles)
        association_ms = (perf_counter() - stage) * 1000.0

        stage = perf_counter()
        candidate_person_indices = {
            association.person_index for association in candidate_associations
        }
        head_rois = {
            person_index: self.roi_extractor.extract(frame, persons[person_index].bbox)
            for person_index in candidate_person_indices
        }
        helmet_predictions = self._match_helmet_detections(
            head_rois,
            helmet_detections,
            persons,
            self.config.detector.helmet_min_person_confidence,
            self.config.detector.helmet_min_confidence_for_weak_person,
            self.config.detector.helmet_min_positive_confidence,
        )
        if self.config.head_roi.strategy == "matched":
            for person_index, (_, box) in helmet_predictions.items():
                crop = self.roi_extractor.extract_box(frame, box)
                if crop is not None:
                    head_rois[person_index] = crop
        minimum_person_confidence = self.config.detector.helmet_min_person_confidence
        associations = [
            association
            for association in candidate_associations
            if (
                persons[association.person_index].confidence >= minimum_person_confidence
                or association.person_index in helmet_predictions
            )
        ]
        riders: list[RiderResult] = []
        active_rider_ids: set[int] = set()
        classifier_period = max(1, self.config.classifier.every_n_frames)
        by_person = {item.person_index: item for item in associations}
        batch_predictions: dict[int, HelmetPrediction] = {}
        classify_batch = getattr(self.classifier, "classify_batch", None)
        if callable(classify_batch):
            batch_indices = []
            for person_index, person in enumerate(persons):
                if (
                    person_index not in by_person
                    or head_rois.get(person_index) is None
                ):
                    continue
                rider_id = person.track_id if person.track_id is not None else person_index + 1
                helmet_evidence = helmet_predictions.get(person_index)
                due_for_classification = (
                    rider_id not in self._last_classification_frame
                    or self.frame_number - self._last_classification_frame[rider_id]
                    >= classifier_period
                )
                if (
                    (person.confidence >= self.config.detector.helmet_min_person_confidence
                     or (helmet_evidence is not None
                         and helmet_evidence[0].status == HelmetStatus.HELMET))
                    and due_for_classification
                ):
                    batch_indices.append(person_index)
            if batch_indices:
                batch_results = classify_batch(
                    [head_rois[index][0] for index in batch_indices]
                )
                batch_predictions = dict(zip(batch_indices, batch_results, strict=True))
        focused_batch_predictions: dict[int, HelmetPrediction] = {}
        focused_indices = []
        focused_crops = []
        negative_threshold = max(
            self.config.classifier.no_helmet_negative_threshold, 0.75
        )
        for person_index, broad_prediction in batch_predictions.items():
            helmet_evidence = helmet_predictions.get(person_index)
            if (
                broad_prediction.status != HelmetStatus.HELMET
                and helmet_evidence is not None
                and helmet_evidence[0].status == HelmetStatus.NO_HELMET
                and helmet_evidence[0].confidence >= 0.20
                and min(helmet_evidence[1][2] - helmet_evidence[1][0],
                        helmet_evidence[1][3] - helmet_evidence[1][1]) >= MIN_LOCALIZED_NO_HELMET_HEAD_EDGE_PX
            ):
                matched = self.roi_extractor.extract_box(frame, helmet_evidence[1])
                if matched is not None:
                    focused_indices.append(person_index)
                    focused_crops.append(matched[0])
                    continue
            # When an estimated crop is uncertain, a credible localized head
            # can recover a helmet obscured by face/clothing in that crop.
            # Only a strong positive from this recheck can change the decision.
            if (
                broad_prediction.status == HelmetStatus.UNKNOWN
                and helmet_evidence is not None
                and helmet_evidence[0].confidence >= 0.20
            ):
                matched = self.roi_extractor.extract_box(frame, helmet_evidence[1])
                if matched is not None and min(matched[0].shape[:2]) >= 24:
                    focused_indices.append(person_index)
                    focused_crops.append(matched[0])
                    continue
            # A helmet can be lost in the full head crop when the face or clothing
            # occupies most of the pixels. Recheck likely negatives using just the
            # upper head. Do this for classifier-only negatives as well as detector
            # negatives; the classifier is often the source of the false NO label.
            if (
                broad_prediction.helmet_probability
                >= self.config.classifier.helmet_positive_threshold
                or broad_prediction.no_helmet_probability < 0.70
            ):
                continue
            helmet_evidence = helmet_predictions.get(person_index)
            if helmet_evidence is not None:
                if helmet_evidence[0].status == HelmetStatus.HELMET:
                    # A broad person box can shift the head crop into a face or
                    # shoulder. Independently verify the complete detected head
                    # before allowing a negative crop to overturn this candidate.
                    matched = self.roi_extractor.extract_box(frame, helmet_evidence[1])
                    focused_crop = matched[0] if matched is not None else None
                    if focused_crop is not None:
                        focused_indices.append(person_index)
                        focused_crops.append(focused_crop)
                    continue
                if helmet_evidence[0].status not in {
                    HelmetStatus.NO_HELMET,
                    HelmetStatus.UNKNOWN,
                }:
                    continue
                focused_crop = self._upper_helmet_crop(frame, helmet_evidence[1])
            else:
                roi = head_rois.get(person_index)
                focused_crop = self._upper_head_crop(roi[0]) if roi is not None else None
            if focused_crop is not None:
                focused_indices.append(person_index)
                focused_crops.append(focused_crop)
        if focused_crops and callable(classify_batch):
            focused_batch_predictions = dict(
                zip(focused_indices, classify_batch(focused_crops), strict=True)
            )
        for person_index, person in enumerate(persons):
            association = by_person.get(person_index)
            if association is None:
                continue
            vehicle = vehicles[association.vehicle_index]
            rider_id = person.track_id if person.track_id is not None else person_index + 1
            vehicle_id = vehicle.track_id if vehicle.track_id is not None else association.vehicle_index + 1
            active_rider_ids.add(rider_id)
            roi = head_rois.get(person_index)
            head_bbox = roi[1] if roi else None
            helmet_evidence = helmet_predictions.get(person_index)
            helmet_bbox = helmet_evidence[1] if helmet_evidence else None
            should_classify = (
                roi is not None
                and (person.confidence >= self.config.detector.helmet_min_person_confidence
                     or (helmet_evidence is not None
                         and helmet_evidence[0].status == HelmetStatus.HELMET))
                and (
                    rider_id not in self._last_classification_frame
                    or self.frame_number - self._last_classification_frame[rider_id]
                    >= classifier_period
                )
            )
            classifier_updated = person_index in batch_predictions
            focused_prediction = focused_batch_predictions.get(person_index)
            if should_classify:
                raw_prediction = (
                    batch_predictions[person_index]
                    if person_index in batch_predictions
                    else self.classifier.classify(roi[0])
                )
                self._classifier_by_track[rider_id] = raw_prediction
                self._last_classification_frame[rider_id] = self.frame_number
                if timestamp_seconds is not None:
                    self._classification_time[rider_id] = timestamp_seconds
                    self._evidence_time[rider_id] = timestamp_seconds
                classifier_updated = True
                self._localized_negative_by_track.pop(rider_id, None)
                self._headwear_negative_by_track.pop(rider_id, None)
                if (
                    raw_prediction.status != HelmetStatus.HELMET
                    and helmet_evidence is not None
                    and helmet_evidence[0].status == HelmetStatus.NO_HELMET
                    and helmet_evidence[0].confidence >= 0.20
                    and min(helmet_evidence[1][2] - helmet_evidence[1][0],
                            helmet_evidence[1][3] - helmet_evidence[1][1]) >= MIN_LOCALIZED_NO_HELMET_HEAD_EDGE_PX
                    and focused_prediction is None
                ):
                    matched = self.roi_extractor.extract_box(frame, helmet_evidence[1])
                    if matched is not None:
                        focused_prediction = self.classifier.classify(matched[0])
                if (
                    raw_prediction.status == HelmetStatus.UNKNOWN
                    and helmet_evidence is not None
                    and helmet_evidence[0].confidence >= 0.20
                    and focused_prediction is None
                    and person_index not in focused_batch_predictions
                ):
                    matched = self.roi_extractor.extract_box(frame, helmet_evidence[1])
                    if matched is not None and min(matched[0].shape[:2]) >= 24:
                        focused_prediction = self.classifier.classify(matched[0])
                if helmet_evidence is None:
                    self._helmet_by_track[rider_id] = self.smoother.update(
                        rider_id, raw_prediction, frame_number=self.frame_number
                    )
                    if (
                        raw_prediction.no_helmet_probability >= 0.70
                        and raw_prediction.helmet_probability < 0.70
                        and person_index not in focused_batch_predictions
                    ):
                        focused_crop = self._upper_head_crop(roi[0])
                        if focused_crop is not None:
                            focused_prediction = self.classifier.classify(focused_crop)
                elif (
                    helmet_evidence[0].status == HelmetStatus.HELMET
                    and focused_prediction is None
                    and raw_prediction.helmet_probability
                    < self.config.classifier.helmet_positive_threshold
                    and raw_prediction.no_helmet_probability >= 0.70
                    and person_index not in focused_batch_predictions
                ):
                    matched = self.roi_extractor.extract_box(frame, helmet_evidence[1])
                    if matched is not None:
                        focused_prediction = self.classifier.classify(matched[0])
                elif (
                    helmet_evidence[0].status == HelmetStatus.NO_HELMET
                    and focused_prediction is None
                    and raw_prediction.helmet_probability < 0.70
                    and (
                        raw_prediction.status != HelmetStatus.NO_HELMET
                        or raw_prediction.no_helmet_probability < negative_threshold
                    )
                    and person_index not in focused_batch_predictions
                ):
                    focused_crop = self._upper_helmet_crop(frame, helmet_evidence[1])
                    if focused_crop is not None:
                        focused_prediction = self.classifier.classify(focused_crop)
            if focused_prediction is not None:
                if (
                    helmet_evidence is not None
                    and helmet_evidence[0].status == HelmetStatus.NO_HELMET
                    and helmet_evidence[0].confidence >= 0.20
                    and focused_prediction.status == HelmetStatus.NO_HELMET
                    and focused_prediction.no_helmet_probability >= max(negative_threshold, 0.95)
                ):
                    self._localized_negative_by_track[rider_id] = (
                        focused_prediction, helmet_evidence[1]
                    )
                if (
                    focused_prediction.helmet_probability
                    >= self.config.classifier.helmet_crop_positive_threshold
                    or self._verified_localized_helmet(
                        raw_prediction, focused_prediction, helmet_evidence
                    )
                ):
                    self._focused_helmet_by_track[rider_id] = focused_prediction
                else:
                    self._focused_helmet_by_track.pop(rider_id, None)
            classifier_prediction = self._classifier_by_track.get(
                rider_id, HelmetPrediction()
            )
            if classifier_updated and roi is not None and focused_prediction is not None:
                verified = self._verify_headwear_negative(
                    frame, roi[0], classifier_prediction, focused_prediction, helmet_evidence
                )
                if verified is not None:
                    self._headwear_negative_by_track[rider_id] = verified
            verified_headwear = False
            if helmet_evidence is not None:
                direct_prediction = helmet_evidence[0]
                if direct_prediction.status == HelmetStatus.NO_HELMET:
                    # The helmet detector produces useful candidate boxes, but its
                    # negative labels are prone to mistaking motorcycle helmets for
                    # bare heads. Only call NO when the independent head classifier
                    # agrees; strong helmet evidence from that classifier wins.
                    focused_helmet = self._focused_helmet_by_track.get(rider_id)
                    localized_negative = self._localized_negative_by_track.get(rider_id)
                    if (
                        classifier_prediction.helmet_probability
                        >= self.config.classifier.helmet_positive_threshold
                    ):
                        prediction = HelmetPrediction(
                            status=HelmetStatus.HELMET,
                            confidence=classifier_prediction.confidence,
                            helmet_probability=classifier_prediction.helmet_probability,
                            no_helmet_probability=classifier_prediction.no_helmet_probability,
                        )
                        helmet_bbox = None
                    elif (
                        focused_helmet is not None
                        and focused_helmet.helmet_probability
                        >= self.config.classifier.helmet_crop_positive_threshold
                    ):
                        # A top-of-head positive takes priority over a broad-crop
                        # negative: helmet shells are localized above the face.
                        prediction = focused_helmet
                        helmet_bbox = None
                    elif localized_negative is not None and min(
                        helmet_evidence[1][2] - helmet_evidence[1][0],
                        helmet_evidence[1][3] - helmet_evidence[1][1],
                        localized_negative[1][2] - localized_negative[1][0],
                        localized_negative[1][3] - localized_negative[1][1],
                    ) >= MIN_LOCALIZED_NO_HELMET_HEAD_EDGE_PX:
                        # The real detector head replaces a torso-derived ROI.
                        # Padding does not count toward the source detail gate.
                        prediction = localized_negative[0]
                        head_bbox = helmet_evidence[1]
                    elif classifier_prediction.no_helmet_probability >= negative_threshold:
                        prediction = classifier_prediction
                    else:
                        prediction = HelmetPrediction(
                            status=HelmetStatus.UNKNOWN,
                            confidence=classifier_prediction.confidence,
                            helmet_probability=classifier_prediction.helmet_probability,
                            no_helmet_probability=classifier_prediction.no_helmet_probability,
                        )
                    # Negative detector boxes often include the face or upper
                    # body. The user should see the rider head region, not that
                    # noisy detector extent.
                    helmet_bbox = None
                elif direct_prediction.status == HelmetStatus.HELMET:
                    # Caps can trigger strong helmet detector scores. Verify the
                    # complete head crop instead of bypassing the classifier.
                    # A missing or uncertain classifier cannot confirm a helmet.
                    localized_helmet = self._focused_helmet_by_track.get(rider_id)
                    if self._verified_localized_helmet(
                        classifier_prediction, localized_helmet, helmet_evidence
                    ):
                        prediction = localized_helmet
                        head_bbox = helmet_evidence[1]
                    elif (
                        classifier_prediction.status == HelmetStatus.HELMET
                        and classifier_prediction.helmet_probability
                        >= self.config.classifier.helmet_positive_threshold
                    ) or (
                        self._focused_helmet_by_track.get(rider_id) is not None
                        and self._focused_helmet_by_track[rider_id].helmet_probability
                        >= self.config.classifier.helmet_crop_positive_threshold
                    ):
                        prediction = direct_prediction
                    elif (
                        rider_id in self._headwear_negative_by_track
                        and direct_prediction.confidence < 0.60
                    ):
                        prediction, head_bbox = self._headwear_negative_by_track[rider_id]
                        helmet_bbox = None
                        verified_headwear = True
                    elif classifier_prediction.no_helmet_probability >= negative_threshold:
                        prediction = classifier_prediction
                        helmet_bbox = None
                    else:
                        prediction = HelmetPrediction(
                            status=HelmetStatus.UNKNOWN,
                            confidence=classifier_prediction.confidence,
                            helmet_probability=classifier_prediction.helmet_probability,
                            no_helmet_probability=classifier_prediction.no_helmet_probability,
                        )
                        helmet_bbox = None
                else:
                    focused_helmet = self._focused_helmet_by_track.get(rider_id)
                    if (
                        focused_helmet is not None
                        and focused_helmet.helmet_probability
                        >= self.config.classifier.helmet_crop_positive_threshold
                    ):
                        prediction = focused_helmet
                    elif (
                        classifier_prediction.helmet_probability
                        >= self.config.classifier.helmet_positive_threshold
                    ):
                        prediction = classifier_prediction
                    elif classifier_prediction.no_helmet_probability >= negative_threshold:
                        prediction = classifier_prediction
                    else:
                        prediction = HelmetPrediction(
                            status=HelmetStatus.UNKNOWN,
                            confidence=classifier_prediction.confidence,
                            helmet_probability=classifier_prediction.helmet_probability,
                            no_helmet_probability=classifier_prediction.no_helmet_probability,
                        )
                    helmet_bbox = None
                self._helmet_by_track[rider_id] = prediction
                if timestamp_seconds is not None:
                    self._evidence_time[rider_id] = timestamp_seconds
            else:
                prediction = self._helmet_by_track.get(rider_id, HelmetPrediction())
                focused_helmet = self._focused_helmet_by_track.get(rider_id)
                if (
                    focused_helmet is not None
                    and focused_helmet.helmet_probability
                    >= self.config.classifier.helmet_crop_positive_threshold
                    and classifier_prediction.no_helmet_probability >= 0.70
                ):
                    prediction = focused_helmet
                    self._helmet_by_track[rider_id] = prediction

            # A very small head crop does not contain enough detail for a reliable
            # violation decision. Keep positive detections, but avoid asserting
            # NO HELMET from a classifier operating on only a few source pixels.
            verified_localized = False
            if prediction.status == HelmetStatus.NO_HELMET and head_bbox is not None:
                head_width = head_bbox[2] - head_bbox[0]
                head_height = head_bbox[3] - head_bbox[1]
                # Small estimated ROIs can land on clothing in an occluded
                # person box. A classifier score alone cannot localize a head.
                localized_negative = (
                    helmet_evidence is not None
                    and helmet_evidence[0].status == HelmetStatus.NO_HELMET
                    and helmet_evidence[0].confidence >= 0.20
                )
                short_edge = min(head_width, head_height)
                verified_localized = (
                    localized_negative
                    and rider_id in self._localized_negative_by_track
                    and prediction.no_helmet_probability >= 0.95
                )
                if (
                    short_edge < (MIN_LOCALIZED_NO_HELMET_HEAD_EDGE_PX
                                  if verified_localized else MIN_NO_HELMET_HEAD_EDGE_PX)
                    or (short_edge < MIN_CLASSIFIER_ONLY_NO_HELMET_HEAD_EDGE_PX
                        and not localized_negative and not verified_headwear)
                ):
                    prediction = HelmetPrediction(
                        status=HelmetStatus.UNKNOWN,
                        confidence=prediction.confidence,
                        helmet_probability=prediction.helmet_probability,
                        no_helmet_probability=prediction.no_helmet_probability,
                    )
                    self._helmet_by_track[rider_id] = prediction
                    self._no_helmet_hits_by_track.pop(rider_id, None)

            # A single bad CCTV frame can make a helmet disappear behind a hand,
            # glare, or a detector box. For streams, require the negative decision
            # to recur before displaying a definitive NO HELMET label. Static image
            # review remains a one-frame decision, but uses the higher confidence
            # threshold above.
            if temporal_consistency:
                if (helmet_evidence is not None
                    and helmet_evidence[0].status != HelmetStatus.NO_HELMET
                    and not verified_headwear):
                    self._no_helmet_hits_by_track.pop(rider_id, None)
                saw_fresh_no_evidence = prediction.status == HelmetStatus.NO_HELMET and (
                    (verified_headwear and classifier_updated)
                    or
                    (
                        helmet_evidence is not None
                        and helmet_evidence[0].status == HelmetStatus.NO_HELMET
                        and classifier_updated
                        and (
                            classifier_prediction.no_helmet_probability
                            >= max(self.config.classifier.no_helmet_negative_threshold, 0.90)
                            or (verified_localized and prediction.no_helmet_probability >= 0.95)
                        )
                    ) or (
                        helmet_evidence is None
                        and classifier_updated
                        and prediction.no_helmet_probability
                        >= max(self.config.classifier.no_helmet_negative_threshold, 0.90)
                    )
                )
                if saw_fresh_no_evidence:
                    self._no_helmet_hits_by_track[rider_id] = (
                        self._no_helmet_hits_by_track.get(rider_id, 0) + 1
                    )
                elif prediction.status != HelmetStatus.NO_HELMET:
                    self._no_helmet_hits_by_track.pop(rider_id, None)
                if (
                    prediction.status == HelmetStatus.NO_HELMET
                    and self._no_helmet_hits_by_track.get(rider_id, 0) < 2
                ):
                    prediction = HelmetPrediction(
                        status=HelmetStatus.UNKNOWN,
                        confidence=prediction.confidence,
                        helmet_probability=prediction.helmet_probability,
                        no_helmet_probability=prediction.no_helmet_probability,
                    )
                self._helmet_by_track[rider_id] = prediction
            riders.append(
                RiderResult(
                    track_id=rider_id,
                    bbox=person.bbox,
                    vehicle_id=vehicle_id,
                    vehicle_type=vehicle.class_name,
                    head_bbox=head_bbox,
                    helmet=prediction,
                    association_score=association.score,
                    helmet_bbox=helmet_bbox,
                    evidence={
                        "source_timestamp_seconds": timestamp_seconds,
                        "classifier_updated": classifier_updated,
                        "classifier_frame": self._last_classification_frame.get(rider_id),
                        "classifier_age_seconds": (
                            timestamp_seconds - self._classification_time[rider_id]
                            if timestamp_seconds is not None and rider_id in self._classification_time else None
                        ),
                        "classifier_helmet_probability": classifier_prediction.helmet_probability,
                        "classifier_no_helmet_probability": classifier_prediction.no_helmet_probability,
                        "head_recheck_helmet_probability": (
                            self._focused_helmet_by_track[rider_id].helmet_probability
                            if rider_id in self._focused_helmet_by_track else None
                        ),
                        "localized_no_helmet_probability": (
                            self._localized_negative_by_track[rider_id][0].no_helmet_probability
                            if rider_id in self._localized_negative_by_track else None
                        ),
                        "headwear_no_helmet_probability": (
                            self._headwear_negative_by_track[rider_id][0].no_helmet_probability
                            if verified_headwear else None
                        ),
                        "detector_status": helmet_evidence[0].status.value if helmet_evidence else None,
                        "detector_confidence": helmet_evidence[0].confidence if helmet_evidence else None,
                        "head_short_edge_px": min(head_bbox[2]-head_bbox[0], head_bbox[3]-head_bbox[1]) if head_bbox else None,
                        "negative_confirmations": self._no_helmet_hits_by_track.get(rider_id, 0),
                        "crop_strategy": self.config.head_roi.strategy,
                    },
                )
            )
        refinement_heads = []
        riders = self._recheck_uncertain_helmets(frame, riders, persons, raw_detections)
        if self.config.classifier.verify_small_heads:
            riders = self._recheck_small_no_helmet(
                frame, riders, raw_detections, vehicles=vehicles,
                temporal_consistency=temporal_consistency,
                timestamp_seconds=timestamp_seconds,
            )
        if self.localized_refiner is not None:
            refinement_heads = localized_heads if localized_heads is not None else raw_detections
            if localized_heads is None and self.localized_head_detector is not None and (self.head_output_builder is not None or self.config.classifier.localized_relocalize_known or self.config.classifier.localized_relocalize_all_known or self.config.classifier.localized_verify_known or any(r.helmet.status == HelmetStatus.UNKNOWN for r in riders)):
                refinement_heads = self.localized_head_detector.detect(frame)
            riders = self.localized_refiner.refine(
                frame, riders, refinement_heads, temporal=temporal_consistency,
                timestamp=timestamp_seconds,
            )
        self.smoother.forget_except(
            active_rider_ids,
            frame_number=self.frame_number,
            max_age=self.config.tracking.max_age,
        )
        self._helmet_by_track = {
            track_id: value
            for track_id, value in self._helmet_by_track.items()
            if track_id in active_rider_ids
        }
        self._classifier_by_track = {
            track_id: value
            for track_id, value in self._classifier_by_track.items()
            if track_id in active_rider_ids
        }
        self._focused_helmet_by_track = {
            track_id: value
            for track_id, value in self._focused_helmet_by_track.items()
            if track_id in active_rider_ids
        }
        self._localized_negative_by_track = {
            track_id: value for track_id, value in self._localized_negative_by_track.items()
            if track_id in active_rider_ids
        }
        self._headwear_negative_by_track = {
            track_id: value for track_id, value in self._headwear_negative_by_track.items()
            if track_id in active_rider_ids
        }
        self._last_classification_frame = {
            track_id: value
            for track_id, value in self._last_classification_frame.items()
            if track_id in active_rider_ids
        }
        self._no_helmet_hits_by_track = {
            track_id: value
            for track_id, value in self._no_helmet_hits_by_track.items()
            if track_id in active_rider_ids
        }
        self._classification_time = {k: v for k, v in self._classification_time.items() if k in active_rider_ids}
        self._evidence_time = {k: v for k, v in self._evidence_time.items() if k in active_rider_ids}
        self._positive_recheck_by_track = {
            k: v for k, v in self._positive_recheck_by_track.items() if k in active_rider_ids
        }
        self._small_no_helmet_by_track = {
            k: v for k, v in self._small_no_helmet_by_track.items() if k in active_rider_ids
        }
        classification_ms = (perf_counter() - stage) * 1000.0

        riders_by_vehicle: dict[int, list[RiderResult]] = {}
        for rider in riders:
            riders_by_vehicle.setdefault(rider.vehicle_id, []).append(rider)
        vehicle_results = tuple(
            VehicleResult(
                track_id=vehicle.track_id or vehicle_index + 1,
                vehicle_type=vehicle.class_name,
                bbox=vehicle.bbox,
                confidence=vehicle.confidence,
                riders=tuple(riders_by_vehicle.get(vehicle.track_id or vehicle_index + 1, ())),
            )
            for vehicle_index, vehicle in enumerate(vehicles)
        )
        # Report people independently of helmet classification. A failed head
        # crop must not make a supported rider look like a pedestrian.
        candidate_by_person = {a.person_index: a for a in candidate_associations}
        person_results = []
        for index, person in enumerate(persons):
            association = candidate_by_person.get(index)
            vehicle_id = None
            if association is not None:
                vehicle = vehicles[association.vehicle_index]
                vehicle_id = vehicle.track_id or association.vehicle_index + 1
                status = "motorcycle" if vehicle.class_name == "motorcycle" else "no_motorcycle_match"
                if index not in by_person:
                    status = "uncertain"
            else:
                # Proximity and weak boxes cannot establish that a person is
                # not riding. Nor can a predicted frame establish absence.
                possible_contacts = (
                    self.associator.score(person.bbox, vehicle.bbox)
                    for vehicle in raw_detections
                    if vehicle.class_name == "motorcycle"
                )
                ambiguous = any(
                    a.person_overlap >= 0.05 and a.horizontal_alignment >= 0.30
                    for a in possible_contacts
                )
                status = (
                    "no_motorcycle_match"
                    if detector_ran and person.confidence >= 0.50 and not ambiguous
                    else "uncertain"
                )
            person_results.append(PersonResult(
                person.track_id or index + 1, person.bbox, person.confidence, status, vehicle_id
            ))
        head_results = None
        if (self.head_output_builder is not None
                and max(frame.shape[:2]) >= self.config.classifier.localized_independent_min_dimension):
            head_results = self.head_output_builder.build(
                refinement_heads, person_results, frame=frame, temporal=temporal_consistency,
                timestamp=timestamp_seconds,
            )
        total_ms = (perf_counter() - started) * 1000.0
        timings = FrameTimings(
            detection_ms=detection_ms,
            tracking_ms=tracking_ms,
            association_ms=association_ms,
            classification_ms=classification_ms,
            total_ms=total_ms,
        )
        return FrameResult(
            width=width,
            height=height,
            detections=tuple(detections),
            vehicles=vehicle_results,
            riders=tuple(riders),
            associations=tuple(all_scores),
            timings=timings,
            fps=1000.0 / total_ms if total_ms > 0 else 0.0,
            persons=tuple(person_results),
            heads=head_results,
            detector_ran=detector_ran,
        )

    @staticmethod
    def _positive_view_confidence(detector_score: float, edge: int,
                                  full: float, upper: float, context: float) -> float | None:
        """Require classifier support for weak heads; never infer a negative."""
        if edge >= 20:
            if detector_score >= 0.60 and upper >= 0.90:
                return upper
            if detector_score >= 0.005 and full >= 0.75 and max(upper, context) >= 0.75:
                return min(full, max(upper, context))
            if detector_score >= 0.20 and full >= 0.75:
                return full
        if edge >= 12 and detector_score >= 0.60 and min(upper, context) >= 0.75:
            return min(upper, context)
        return None

    def _recheck_uncertain_helmets(self, frame: np.ndarray, riders: list[RiderResult],
                                   persons: list[Detection], detections: list[Detection]) -> list[RiderResult]:
        if self.config.head_roi.strategy != "verified":
            return riders
        # Use full person width and the upper third to localize heads that the
        # narrow estimated ROI misses. Match against all people so a neighbouring
        # confirmed rider still owns their head; only UNKNOWN labels may change.
        search_regions = {}
        for index, person in enumerate(persons):
            x1, y1, x2, y2 = person.bbox
            search_regions[index] = (frame, (x1, y1, x2, y1 + .35 * (y2-y1)))
        matches = self._match_helmet_detections(
            search_regions, [d for d in detections if d.class_name == "with_helmet"],
            persons, minimum_positive_confidence=.005,
        )
        by_track = {p.track_id or i+1: i for i, p in enumerate(persons)}
        crops, pending = [], []
        height, width = frame.shape[:2]
        for rider in riders:
            if rider.helmet.status != HelmetStatus.UNKNOWN:
                self._positive_recheck_by_track.pop(rider.track_id, None)
                continue
            if not rider.evidence.get("classifier_updated"):
                continue
            self._positive_recheck_by_track.pop(rider.track_id, None)
            match = matches.get(by_track.get(rider.track_id))
            if match is None:
                continue
            prediction, box = match
            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(width,x2), min(height,y2)
            edge = min(x2-x1, y2-y1)
            if edge < 12:
                continue
            px, py = int((x2-x1)*.15), int((y2-y1)*.15)
            crops.extend((frame[y1:y2,x1:x2],
                          frame[y1:y1+int((y2-y1)*.5),x1:x2],
                          frame[max(0,y1-py):min(height,y2+py),max(0,x1-px):min(width,x2+px)]))
            pending.append((rider.track_id, prediction.confidence, edge, box))
        if crops:
            predictions = self.classifier.classify_batch(crops)
            for i, (track, score, edge, box) in enumerate(pending):
                full, upper, context = [p.helmet_probability for p in predictions[i*3:i*3+3]]
                confidence = self._positive_view_confidence(score, edge, full, upper, context)
                if confidence is not None:
                    verified = HelmetPrediction(HelmetStatus.HELMET, confidence, confidence, 1-confidence)
                    self._positive_recheck_by_track[track] = (verified, box, {
                        "positive_recheck_views": {"full": full, "upper": upper, "context": context},
                        "positive_recheck_detector_confidence": score,
                        "head_short_edge_px": edge,
                    })
        output = []
        for rider in riders:
            verified = self._positive_recheck_by_track.get(rider.track_id)
            if rider.helmet.status == HelmetStatus.UNKNOWN and verified is not None:
                prediction, box, evidence = verified
                rider = replace(rider, helmet=prediction, head_bbox=box, helmet_bbox=box,
                                evidence={**rider.evidence, **evidence})
            output.append(rider)
        return output

    def _recheck_small_no_helmet(
        self, frame: np.ndarray, riders: list[RiderResult],
        detections: list[Detection], *, vehicles: list[Detection],
        temporal_consistency: bool = False,
        timestamp_seconds: float | None = None,
    ) -> list[RiderResult]:
        """Recover abstentions with a localized bare head and two negative views.

        The estimated crop alone cannot waive the small-head gate. Require a
        strong bare-head detection, actual source detail, and agreement on both
        the estimated and localized crops. Keep every existing known decision.
        Streams need two fresh classification passes for the same rider.
        """
        heads = [d for d in detections
                 if d.class_name == "without_helmet" and d.confidence >= .50]
        vehicle_by_id = {v.track_id: v for v in vehicles if v.class_name == "motorcycle"}
        candidates = []
        eligible = set()
        for index, rider in enumerate(riders):
            evidence = rider.evidence
            vehicle = vehicle_by_id.get(rider.vehicle_id)
            if vehicle is not None:
                px1, py1, px2, py2 = rider.bbox
                vx1, vy1, vx2, _ = vehicle.bbox
                # Side views need a vehicle wider than the body; front views
                # need the vehicle to start below the upper torso. A false
                # motorbike box on a pedestrian's torso supports neither view.
                vehicle_supported = (
                    vx2-vx1 >= 1.40*(px2-px1)
                    or vy1 >= py1+.35*(py2-py1)
                )
            else:
                vehicle_supported = False
            if (rider.helmet.status != HelmetStatus.UNKNOWN or rider.head_bbox is None
                or not vehicle_supported
                or evidence.get("classifier_no_helmet_probability", 0) < .90
                or (evidence.get("head_recheck_helmet_probability") or 0) >= .55
                or evidence.get("positive_recheck_views")):
                self._small_no_helmet_by_track.pop(rider.track_id, None)
                continue
            eligible.add(index)
            for hi, head in enumerate(heads):
                overlap = iou(rider.head_bbox, head.bbox)
                if overlap >= .25 and min(head.bbox[2]-head.bbox[0],
                                         head.bbox[3]-head.bbox[1]) >= 16:
                    candidates.append((overlap, index, hi))
        used_riders, used_heads, pending, crops = set(), set(), [], []
        height, width = frame.shape[:2]
        for _, index, hi in sorted(candidates, reverse=True):
            if index in used_riders or hi in used_heads:
                continue
            used_riders.add(index)
            used_heads.add(hi)
            rider, head = riders[index], heads[hi]
            if not rider.evidence.get("classifier_updated"):
                cached = self._small_no_helmet_by_track.get(rider.track_id)
                if cached is not None:
                    cached["bbox"] = head.bbox
                continue
            localized = self.roi_extractor.extract_box(frame, head.bbox)
            x1, y1, x2, y2 = map(int, rider.head_bbox)
            broad = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
            if localized is None or broad.size == 0:
                self._small_no_helmet_by_track.pop(rider.track_id, None)
                continue
            pending.append((index, head))
            crops.extend([broad, localized[0]])
        batch = getattr(self.classifier, "classify_batch", None)
        predictions = (batch(crops) if callable(batch)
                       else [self.classifier.classify(c) for c in crops]) if crops else []
        for j, (index, head) in enumerate(pending):
            rider = riders[index]
            views = predictions[j*2:j*2+2]
            if len(views) != 2 or min(v.no_helmet_probability for v in views) < .85:
                self._small_no_helmet_by_track.pop(rider.track_id, None)
                continue
            old = self._small_no_helmet_by_track.get(rider.track_id)
            fresh_old = old is not None and self.frame_number - old["frame"] <= max(
                self.config.tracking.max_age, 2*self.config.classifier.every_n_frames
            )
            if fresh_old and timestamp_seconds is not None and old["timestamp"] is not None:
                fresh_old = timestamp_seconds-old["timestamp"] <= self.config.classifier.evidence_max_age_seconds
            hits = old["hits"]+1 if fresh_old else 1
            probability = min(v.no_helmet_probability for v in views)
            self._small_no_helmet_by_track[rider.track_id] = {
                "prediction": HelmetPrediction(HelmetStatus.NO_HELMET, probability,
                                               1-probability, probability),
                "bbox": head.bbox, "hits": hits, "frame": self.frame_number,
                "timestamp": timestamp_seconds,
                "views": [v.no_helmet_probability for v in views],
                "detector_confidence": head.confidence,
            }
        for index, rider in enumerate(riders):
            cached = self._small_no_helmet_by_track.get(rider.track_id)
            # Fresh localization/ownership is required even between classifier
            # passes. A lost or conflicting head clears accumulated evidence.
            if index not in eligible or index not in used_riders:
                self._small_no_helmet_by_track.pop(rider.track_id, None)
                continue
            if cached is None or (temporal_consistency and cached["hits"] < 2):
                continue
            expired = self.frame_number-cached["frame"] > max(
                self.config.tracking.max_age, 2*self.config.classifier.every_n_frames
            )
            if timestamp_seconds is not None and cached["timestamp"] is not None:
                expired |= timestamp_seconds-cached["timestamp"] > self.config.classifier.evidence_max_age_seconds
            if expired:
                self._small_no_helmet_by_track.pop(rider.track_id, None)
                continue
            riders[index] = replace(
                rider, helmet=cached["prediction"], head_bbox=cached["bbox"], helmet_bbox=None,
                evidence={**rider.evidence, "small_head_verified": True,
                          "small_head_no_helmet_views": cached["views"],
                          "small_head_detector_confidence": cached["detector_confidence"],
                          "head_short_edge_px": min(cached["bbox"][2]-cached["bbox"][0], cached["bbox"][3]-cached["bbox"][1]),
                          "negative_confirmations": cached["hits"]},
            )
        return riders

    def _verified_localized_helmet(
        self, primary: HelmetPrediction, localized: HelmetPrediction | None,
        evidence: tuple[HelmetPrediction, BBox] | None,
    ) -> bool:
        """Confirm a misplaced uncertain crop with credible localized evidence."""
        return bool(
            self.config.head_roi.strategy == "verified"
            and primary.status == HelmetStatus.UNKNOWN
            and localized is not None
            and localized.status == HelmetStatus.HELMET
            and localized.helmet_probability >= max(
                0.65, self.config.classifier.helmet_positive_threshold
            )
            and evidence is not None
            and evidence[0].status == HelmetStatus.HELMET
            and evidence[0].confidence >= 0.60
            and min(evidence[1][2] - evidence[1][0],
                    evidence[1][3] - evidence[1][1]) >= 24
        )

    def _verify_headwear_negative(
        self, frame: np.ndarray, broad_crop: np.ndarray,
        primary: HelmetPrediction, localized: HelmetPrediction,
        evidence: tuple[HelmetPrediction, BBox] | None,
    ) -> tuple[HelmetPrediction, BBox] | None:
        """Resolve weak helmet detections using two headwear-trained views.

        The original model must abstain and favor no helmet in both crops.
        This avoids replacing confident traffic decisions with the headwear
        checkpoint, which previously regressed helmeted riders.
        """
        if (
            self.headwear_classifier is None or evidence is None
            or evidence[0].status != HelmetStatus.HELMET
            or evidence[0].confidence >= 0.60
            or primary.status != HelmetStatus.UNKNOWN
            or primary.no_helmet_probability < 0.55
            or localized.status == HelmetStatus.HELMET
            or localized.no_helmet_probability < 0.60
        ):
            return None
        box = evidence[1]
        if min(box[2] - box[0], box[3] - box[1]) < MIN_NO_HELMET_HEAD_EDGE_PX:
            return None
        matched = self.roi_extractor.extract_box(frame, box)
        if matched is None:
            return None
        batch = getattr(self.headwear_classifier, "classify_batch", None)
        predictions = (
            batch([broad_crop, matched[0]]) if callable(batch)
            else [self.headwear_classifier.classify(c) for c in (broad_crop, matched[0])]
        )
        threshold = max(0.75, self.config.classifier.no_helmet_negative_threshold)
        if len(predictions) != 2 or any(
            p.status != HelmetStatus.NO_HELMET or p.no_helmet_probability < threshold
            for p in predictions
        ):
            return None
        # The weakest view determines confidence; do not multiply correlated
        # crop scores or reinterpret them as independent probabilities.
        probability = min(p.no_helmet_probability for p in predictions)
        return HelmetPrediction(HelmetStatus.NO_HELMET, probability, 1 - probability,
                                probability), box

    def _gate_unassociated_motorcycles(
        self, detections: list[Detection]
    ) -> list[Detection]:
        """Drop weak bike boxes unless person geometry supports a rider."""
        persons = [item for item in detections if item.class_name == "person"]
        vehicles = [
            item for item in detections
            if item.class_name in {"motorcycle", "bicycle"}
        ]
        supported = {
            item.vehicle_index
            for item in self.associator.associate(persons, vehicles)
            if vehicles[item.vehicle_index].class_name == "motorcycle"
        }
        if len(vehicles) < self.config.detector.motorcycle_gap_max_existing:
            # In low-count scenes, an occluded motorcycle may share a rider's
            # contact region with a stronger foreground box. Dense traffic keeps
            # the conservative single-assignment gate to limit weak subpart hits.
            supported.update(
                index for index, vehicle in enumerate(vehicles)
                if vehicle.class_name == "motorcycle" and any(
                    self.associator.supports(person, vehicle) for person in persons
                )
            )
        cutoff = self.config.detector.motorcycle_unassociated_confidence
        retained = [
            vehicle
            for index, vehicle in enumerate(vehicles)
            if (
                vehicle.class_name != "motorcycle"
                or index in supported
                or vehicle.confidence >= cutoff
            )
        ]
        if len(retained) == len(vehicles):
            return detections
        retained_ids = {id(item) for item in retained}
        return [
            item
            for item in detections
            if item.class_name not in {"motorcycle", "bicycle"}
            or id(item) in retained_ids
        ]

    def _deduplicate_person_detections(
        self, detections: list[Detection],
    ) -> list[Detection]:
        """Collapse nested person boxes that describe the same visible rider.

        Crowded traffic models commonly emit both a full-person box and an
        upper-body box for one rider. Suppress only boxes that cover nearly the
        same area and share a top/head anchor, so adjacent passengers remain
        separate detections.
        """
        person_indices = [
            index for index, item in enumerate(detections)
            if item.class_name == "person"
        ]
        groups: list[list[int]] = []
        for index in person_indices:
            matching_group = next(
                (
                    group for group in groups
                    if any(
                        self._same_person_box(
                            detections[index].bbox, detections[member].bbox
                        )
                        for member in group
                    )
                ),
                None,
            )
            if matching_group is None:
                groups.append([index])
            else:
                matching_group.append(index)

        vehicles = [
            item for item in detections
            if item.class_name in {"motorcycle", "bicycle"}
        ]
        kept_person_indices: set[int] = set()
        for group in groups:
            # Multiple detector boxes for one partially occluded rider can
            # point to different bikes. Pick the box with the strongest valid
            # rider-to-bike match before falling back to detector confidence.
            # This preserves a rear bike when the full-body duplicate overlaps
            # a foreground motorcycle.
            best_index = max(
                group,
                key=lambda candidate_index: (
                    max(
                        (
                            association.score
                            for association in self.associator.associate(
                                [detections[candidate_index]], vehicles
                            )
                        ),
                        default=0.0,
                    ),
                    detections[candidate_index].confidence,
                ),
            )
            kept_person_indices.add(best_index)

        return [
            item for index, item in enumerate(detections)
            if item.class_name != "person" or index in kept_person_indices
        ]

    @staticmethod
    def _same_person_box(first: BBox, second: BBox) -> bool:
        """Return whether two nested person detections share one head anchor."""
        x1, y1, x2, y2 = first
        kx1, ky1, kx2, ky2 = second
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        kept_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
        smaller_area = min(area, kept_area)
        if smaller_area <= 0.0:
            return False
        intersection = max(0.0, min(x2, kx2) - max(x1, kx1)) * max(
            0.0, min(y2, ky2) - max(y1, ky1)
        )
        if intersection / smaller_area < 0.72:
            return False
        width = max(1.0, min(x2 - x1, kx2 - kx1))
        height = max(1.0, min(y2 - y1, ky2 - ky1))
        return (
            abs(y1 - ky1) <= max(6.0, 0.12 * height)
            and abs((x1 + x2) / 2.0 - (kx1 + kx2) / 2.0)
            <= max(8.0, 0.16 * width)
        )

    @staticmethod
    def _upper_helmet_crop(frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        """Return a top-focused crop for an ambiguous no-helmet detection."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(width, int(np.ceil(x2)))
        bottom = min(height, int(np.floor(y1 + (y2 - y1) * 0.65)))
        if right - left < 32 or bottom - top < 40:
            return None
        crop = frame[top:bottom, left:right]
        return crop if crop.size else None

    @staticmethod
    def _upper_head_crop(head_crop: np.ndarray) -> np.ndarray | None:
        """Trim a rider head ROI to emphasize helmet shells above the face."""
        if head_crop is None or head_crop.ndim < 2 or head_crop.size == 0:
            return None
        height, width = head_crop.shape[:2]
        bottom = int(np.floor(height * 0.65))
        if width < 32 or bottom < 40:
            return None
        crop = head_crop[:bottom]
        return crop if crop.size else None

    @staticmethod
    def _match_helmet_detections(
        head_rois: dict[int, tuple[np.ndarray, BBox] | None],
        detections: list[Detection],
        persons: list[Detection],
        minimum_person_confidence: float = 0.10,
        minimum_helmet_confidence_for_weak_person: float = 0.02,
        minimum_positive_confidence: float = 0.35,
    ) -> dict[int, tuple[HelmetPrediction, BBox]]:
        """Require stronger helmet evidence when the rider box is weak."""
        candidates = []
        for detection_index, detection in enumerate(detections):
            if detection.class_name not in HELMET_DETECTION_CLASSES:
                continue
            if (
                detection.class_name == "with_helmet"
                and detection.confidence < minimum_positive_confidence
            ):
                continue
            x1, y1, x2, y2 = detection.bbox
            box_area = max(1.0, (x2 - x1) * (y2 - y1))
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            for person_index, roi in head_rois.items():
                if (
                    roi is None
                    or person_index >= len(persons)
                    or (
                        persons[person_index].confidence < minimum_person_confidence
                        and detection.confidence
                        < minimum_helmet_confidence_for_weak_person
                    )
                ):
                    continue
                hx1, hy1, hx2, hy2 = roi[1]
                intersection = max(0.0, min(x2, hx2) - max(x1, hx1)) * max(
                    0.0, min(y2, hy2) - max(y1, hy1)
                )
                box_coverage = intersection / box_area
                center_inside = hx1 <= center_x <= hx2 and hy1 <= center_y <= hy2
                # Helmet boxes can straddle adjacent heads in crowded frames.
                # Require their center to land in the rider head ROI, or strong
                # containment when the detector box is offset.
                if box_coverage < 0.55 and not center_inside:
                    continue
                match_score = box_coverage + (0.25 if center_inside else 0.0)
                candidates.append(
                    (
                        match_score * (0.5 + detection.confidence),
                        detection.confidence,
                        match_score,
                        person_index,
                        detection_index,
                    )
                )

        assigned_people: set[int] = set()
        assigned_detections: set[int] = set()
        detection_owner: dict[int, int] = {}
        predictions: dict[int, tuple[HelmetPrediction, BBox]] = {}
        status_by_class = {
            "with_helmet": HelmetStatus.HELMET,
            "without_helmet": HelmetStatus.NO_HELMET,
        }
        for _, _, _, person_index, detection_index in sorted(
            candidates, reverse=True
        ):
            if person_index in assigned_people or detection_index in assigned_detections:
                continue
            detection = detections[detection_index]
            status = status_by_class[detection.class_name]
            predictions[person_index] = (
                HelmetPrediction(
                    status=status,
                    confidence=detection.confidence,
                    helmet_probability=float(status == HelmetStatus.HELMET),
                    no_helmet_probability=float(status == HelmetStatus.NO_HELMET),
                ),
                detection.bbox,
            )
            assigned_people.add(person_index)
            assigned_detections.add(detection_index)
            detection_owner[detection_index] = person_index

        # If two credible, overlapping boxes disagree on the same rider, keep
        # the disagreement for the head classifier instead of committing to the
        # higher-scoring class. This is particularly important for red or
        # partially occluded helmets that the detector may call bare heads.
        for person_index, (prediction, bbox) in tuple(predictions.items()):
            selected_class = (
                "with_helmet"
                if prediction.status == HelmetStatus.HELMET
                else "without_helmet"
            )
            for _, confidence, _, candidate_person, detection_index in candidates:
                if candidate_person != person_index or confidence < 0.20:
                    continue
                candidate_detection = detections[detection_index]
                if candidate_detection.class_name == selected_class:
                    continue
                owner = detection_owner.get(detection_index)
                if owner is not None and owner != person_index:
                    continue
                other_bbox = candidate_detection.bbox
                ix1, iy1 = max(bbox[0], other_bbox[0]), max(bbox[1], other_bbox[1])
                ix2, iy2 = min(bbox[2], other_bbox[2]), min(bbox[3], other_bbox[3])
                intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_a = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                area_b = max(1.0, (other_bbox[2] - other_bbox[0]) *
                             (other_bbox[3] - other_bbox[1]))
                if intersection / (area_a + area_b - intersection) < 0.30:
                    continue
                predictions[person_index] = (
                    HelmetPrediction(
                        status=HelmetStatus.UNKNOWN,
                        confidence=max(prediction.confidence, confidence),
                        helmet_probability=0.5,
                        no_helmet_probability=0.5,
                    ),
                    bbox,
                )
                break
        return predictions


def build_pipeline(config: AppConfig) -> HelmetRiderPipeline:
    """Build a pipeline for CLI and web use."""
    return HelmetRiderPipeline(config)

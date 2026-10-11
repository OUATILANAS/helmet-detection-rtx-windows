"""One-to-one detector head crops for riders whose body crop abstained."""
from __future__ import annotations

from dataclasses import replace

from helmet_rider.association.geometry import iou
from helmet_rider.detection.types import HelmetPrediction, HelmetStatus
from helmet_rider.helmet.head_roi import HeadROIExtractor


def assign_heads(riders, heads, *, occlusion_tolerant=False):
    """Match within the upper body; known riders also reserve their own heads."""
    candidates = []
    for ri, rider in enumerate(riders):
        x1, y1, x2, y2 = rider.bbox
        w, h = max(1, x2-x1), max(1, y2-y1)
        for hi, head in enumerate(heads):
            hx1, hy1, hx2, hy2 = head.bbox
            cx, cy = (hx1+hx2)/2, (hy1+hy2)/2
            above, below = (.25, .55) if occlusion_tolerant else (.05, .40)
            if not (x1 <= cx <= x2 and y1-above*h <= cy <= y1+below*h):
                continue
            max_width, max_height = (1.7, .75) if occlusion_tolerant else (1.3, .45)
            if hx2-hx1 > w*max_width or hy2-hy1 > h*max_height:
                continue
            overlap = iou(rider.head_bbox, head.bbox) if rider.head_bbox else 0
            distance = abs(cx-(x1+x2)/2)/w + abs(cy-(y1+.12*h))/h
            confidence_weight = .5 if occlusion_tolerant else .1
            candidates.append((2*overlap-distance+confidence_weight*head.confidence, ri, hi))
    used_riders, used_heads, matches = set(), set(), []
    for _, ri, hi in sorted(candidates, reverse=True):
        if ri not in used_riders and hi not in used_heads:
            used_riders.add(ri)
            used_heads.add(hi)
            matches.append((ri, hi))
    return matches


class LocalizedHeadRefiner:
    def __init__(self, classifier, extractor, *, threshold=.55, max_age=10, native_threshold=0, relocalize_known=False, relocalize_all_known=False, verifier=None, negative_threshold=.75, prefer_consensus=False, verify_known=False, occlusion_tolerant=False, head_min_confidence=.1):
        self.head_min_confidence = head_min_confidence
        self.classifier = classifier
        self.extractor = extractor
        self.threshold = threshold
        self.max_age = max_age
        self.native_threshold = native_threshold
        self.relocalize_known = relocalize_known
        self.relocalize_all_known = relocalize_all_known
        self.verifier = verifier
        self.negative_threshold = negative_threshold
        self.prefer_consensus = prefer_consensus
        self.verify_known = verify_known
        self.occlusion_tolerant = occlusion_tolerant
        self.wide_extractor = HeadROIExtractor(replace(extractor.config,matched_context=.35))
        self.cache = {}

    def reset(self):
        self.cache.clear()

    def refine(self, frame, riders, detections, *, temporal=False, timestamp=None):
        heads = []
        for head in sorted(detections, key=lambda d: d.confidence, reverse=True):
            if head.class_name not in {"with_helmet", "without_helmet"} or head.confidence < self.head_min_confidence:
                continue
            if all(iou(head.bbox, other.bbox) < .45 for other in heads):
                heads.append(head)
        matched = assign_heads(riders, heads,occlusion_tolerant=self.occlusion_tolerant)
        owned = {riders[ri].track_id for ri, _ in matched if riders[ri].helmet.status == HelmetStatus.UNKNOWN or self.verify_known}
        for track in tuple(self.cache):
            if track not in owned:
                self.cache.pop(track)
        pending, crops = [], []
        output = list(riders)
        for ri, hi in matched:
            rider, head = riders[ri], heads[hi]
            if rider.helmet.status != HelmetStatus.UNKNOWN:
                native_status = HelmetStatus.HELMET if head.class_name == "with_helmet" else HelmetStatus.NO_HELMET
                if ((self.relocalize_known or self.relocalize_all_known) and self.native_threshold
                        and head.confidence >= self.native_threshold
                        and (self.relocalize_all_known or native_status == rider.helmet.status)):
                    output[ri] = replace(rider, head_bbox=head.bbox)
                if not self.verify_known:
                    continue
            cached = self.cache.get(rider.track_id)
            if cached and ((timestamp is not None and cached["timestamp"] is not None
                            and timestamp-cached["timestamp"] > self.max_age)
                           or iou(cached["bbox"], head.bbox) < .3):
                self.cache.pop(rider.track_id)
                cached = None
            fresh = not temporal or rider.evidence.get("classifier_updated", False)
            crop = self.extractor.extract_box(frame, head.bbox)
            if crop is None:
                self.cache.pop(rider.track_id, None)
                continue
            if fresh:
                pending.append((ri, head, cached))
                crops.append(crop[0])
            elif cached:
                output[ri] = self._apply(rider, head, cached, temporal)
        predictions = self.classifier.classify_batch(crops) if crops else []
        verified = self.verifier.classify_batch(crops) if crops and self.verifier else [None]*len(crops)
        wide_indices, wide_crops = [], []
        for index,(ri,head,_) in enumerate(pending):
            if riders[ri].helmet.status != HelmetStatus.UNKNOWN:
                wide = self.wide_extractor.extract_box(frame,head.bbox)
                if wide is not None:
                    wide_indices.append(index)
                    wide_crops.append(wide[0])
        wide_values = self.classifier.classify_batch(wide_crops) if wide_crops else []
        wide_predictions = dict(zip(wide_indices,wide_values,strict=True))
        for index, ((ri, head, previous), prediction, verification) in enumerate(zip(pending, predictions, verified, strict=True)):
            prediction, consensus = self.decide_prediction(riders[ri], head, prediction, verification, wide_predictions.get(index))
            if prediction is None:
                self.cache.pop(riders[ri].track_id,None)
                continue
            status = prediction.status
            hits = previous["hits"]+1 if previous and previous["prediction"].status == status else 1
            entry = {"bbox": head.bbox, "timestamp": timestamp, "prediction": prediction, "hits": hits,
                     "consensus":consensus,"native_confidence":head.confidence}
            self.cache[riders[ri].track_id] = entry
            output[ri] = self._apply(riders[ri], head, entry, temporal)
        return output

    def decide_prediction(self, rider, head, prediction, verification, wide=None):
        """Return a validated head decision, or None to preserve a known label."""
        crop_prediction = prediction
        consensus = False
        if rider.helmet.status != HelmetStatus.UNKNOWN:
            if not self.verify_known:
                return None, False
            old = rider.helmet.status
            if wide is None or verification is None:
                return None, False
            h = (prediction.helmet_probability+wide.helmet_probability+verification.helmet_probability)/3
            reference = rider.helmet_bbox or rider.head_bbox
            negative_covers_top = (reference is None or head.bbox[1] <= reference[1]+.25*(reference[3]-reference[1]))
            if (old == HelmetStatus.HELMET and negative_covers_top and prediction.no_helmet_probability >= .95
                    and wide.no_helmet_probability >= .95 and verification.no_helmet_probability >= .90):
                prediction = HelmetPrediction(HelmetStatus.NO_HELMET,1-h,h,1-h)
            elif (old == HelmetStatus.NO_HELMET and prediction.helmet_probability >= .95
                    and wide.helmet_probability >= .95 and verification.helmet_probability >= .90):
                prediction = HelmetPrediction(HelmetStatus.HELMET,h,h,1-h)
            else:
                return None, False
            consensus = True
            status = prediction.status
        status = (HelmetStatus.HELMET if prediction.helmet_probability >= self.threshold
                  else HelmetStatus.NO_HELMET if prediction.no_helmet_probability >= self.threshold
                  else HelmetStatus.UNKNOWN)
        prediction = replace(prediction, status=status) if not consensus else prediction
        if not consensus and self.native_threshold and head.confidence >= self.native_threshold:
            status = HelmetStatus.HELMET if head.class_name == "with_helmet" else HelmetStatus.NO_HELMET
            h = head.confidence if status == HelmetStatus.HELMET else 1-head.confidence
            prediction = HelmetPrediction(status, head.confidence, h, 1-h)
        if not consensus and self.prefer_consensus and verification is not None:
            if crop_prediction.helmet_probability >= .90 and verification.helmet_probability >= .85:
                status = HelmetStatus.HELMET
                h = (crop_prediction.helmet_probability+verification.helmet_probability)/2
                prediction = HelmetPrediction(status,h,h,1-h)
                consensus = True
            elif crop_prediction.no_helmet_probability >= .90 and verification.no_helmet_probability >= .85:
                status = HelmetStatus.NO_HELMET
                n = (crop_prediction.no_helmet_probability+verification.no_helmet_probability)/2
                prediction = HelmetPrediction(status,n,1-n,n)
                consensus = True
        if (not consensus and prediction.status == HelmetStatus.HELMET and verification is not None
                and (rider.evidence.get("classifier_no_helmet_probability") or 0) >= self.negative_threshold
                and verification.no_helmet_probability >= self.negative_threshold):
            prediction = replace(prediction, status=HelmetStatus.UNKNOWN)
            status = HelmetStatus.UNKNOWN
        return prediction, consensus

    @staticmethod
    def _apply(rider, head, entry, temporal):
        prediction = entry["prediction"]
        if temporal and prediction.status == HelmetStatus.NO_HELMET and entry["hits"] < 2:
            prediction = replace(prediction, status=HelmetStatus.UNKNOWN)
        return replace(rider, head_bbox=head.bbox, helmet=prediction, helmet_bbox=None,
                       evidence={**rider.evidence, "localized_refinement": True,
                                 "localized_refinement_probability": entry["prediction"].confidence,
                                 "localized_classifier_consensus":entry["consensus"],
                                 "localized_detector_confidence":entry["native_confidence"],
                                 "localized_refinement_confirmations": entry["hits"]})

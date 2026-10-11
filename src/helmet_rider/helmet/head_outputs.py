"""Head-first outputs retain passengers even when their body box is merged.

Only real detector boxes are emitted. Body and motorcycle matching enrich
these observations independently; a missing body never implies no motorcycle.
"""
from dataclasses import replace
from math import isfinite

from helmet_rider.association.geometry import iou
from helmet_rider.detection.types import HeadResult, HelmetPrediction, HelmetStatus
from helmet_rider.tracking.byte_tracker import create_tracker


class HeadOutputBuilder:
    def __init__(self, tracking, *, min_confidence=.1, negative_confirmations=2,
                 min_person_confidence=.25,
                 classifier=None, extractor=None, classification_threshold=.55, verification_threshold=None,
                 head_validator=None, head_validity_threshold=.5, unassociated_validity_threshold=.8):
        self.tracking = tracking
        self.tracker=create_tracker(replace(tracking,backend='iou'))
        self.min_confidence=min_confidence
        self.negative_confirmations=negative_confirmations
        self.min_person_confidence=min_person_confidence
        self.head_validator=head_validator
        self.head_validity_threshold=head_validity_threshold
        self.unassociated_validity_threshold=unassociated_validity_threshold
        self.classifier=classifier
        self.extractor=extractor
        self.classification_threshold=classification_threshold
        self.verification_threshold=verification_threshold
        self.negatives={}
        self.classified_negatives={}

    def reset(self):
        self.tracker=create_tracker(replace(self.tracking,backend='iou'))
        self.negatives.clear()
        self.classified_negatives.clear()

    def build(self, detections, persons, *, frame=None, temporal=False, timestamp=None):
        heads=[]
        for head in sorted(detections,key=lambda d:d.confidence,reverse=True):
            if head.class_name not in {'with_helmet','without_helmet','physical_head'} or head.confidence<self.min_confidence:
                continue
            if all(iou(head.bbox,h.bbox)<.45 for h in heads):
                heads.append(head)
        validity_scores=[None]*len(heads)
        if self.head_validator is not None:
            if frame is None or self.extractor is None:
                raise ValueError('Head validity requires the actual frame and ROI extractor')
            crops=[]
            valid_heads=[]
            for head in heads:
                extracted=self.extractor.extract_box(frame,head.bbox)
                if extracted is not None:
                    valid_heads.append(head)
                    crops.append(extracted[0])
            scores=self.head_validator.validate_batch(crops) if crops else []
            if len(scores)!=len(valid_heads) or any(not isfinite(s) or not 0<=s<=1 for s in scores):
                raise ValueError('Invalid head validity scores')
            retained=[(h,s) for h,s in zip(valid_heads,scores,strict=True) if s>=self.head_validity_threshold]
            heads=[h for h,_ in retained]
            validity_scores=[s for _,s in retained]
        if temporal:
            # Identity must survive a helmet-label disagreement. The dedicated
            # IoU tracker preserves input order and always returns observations.
            tracked=self.tracker.update([replace(h,class_name='head') for h in heads],timestamp_seconds=timestamp)
            heads=[replace(t,class_name=h.class_name) for t,h in zip(tracked,heads,strict=True)]
        else:
            heads=[replace(h,track_id=i+1) for i,h in enumerate(heads)]
        results=[]
        active=set()
        for index,head in enumerate(heads):
            validity=validity_scores[index]
            x1,y1,x2,y2=head.bbox
            cx,cy=(x1+x2)/2,(y1+y2)/2
            candidates=[]
            for person in persons:
                if person.confidence<self.min_person_confidence:
                    continue
                a,b,c,d=person.bbox
                w,h=max(1,c-a),max(1,d-b)
                if (a-.15*w<=cx<=c+.15*w and b-.25*h<=cy<=b+.4*h
                        and x2-x1<=1.2*w and y2-y1<=.5*h):
                    distance=abs(cx-(a+c)/2)/w+abs(cy-(b+.12*h))/h
                    candidates.append((distance,person.track_id,person))
            owner=min(candidates,key=lambda item:item[:2])[2] if candidates else None
            # High native scores can still be foliage or street furniture.
            # Require a real person before exposing a head classification.
            if owner is None and (validity is None or validity<self.unassociated_validity_threshold):
                continue
            active.add(head.track_id)
            positive=head.class_name=='with_helmet'
            neutral=head.class_name=='physical_head'
            status=HelmetStatus.UNKNOWN if neutral else HelmetStatus.HELMET if positive else HelmetStatus.NO_HELMET
            if temporal and head.class_name=='without_helmet':
                previous=self.negatives.get(head.track_id)
                owner_id=owner.track_id if owner else None
                fresh=(previous is not None and previous['owner']==owner_id
                    and iou(previous['bbox'],head.bbox)>=.3
                    and (timestamp is None or previous['timestamp'] is None
                         or 0<=timestamp-previous['timestamp']<=self.tracking.max_age_seconds))
                count=previous['count']+1 if fresh else 1
                self.negatives[head.track_id]={'count':count,'owner':owner_id,'bbox':head.bbox,'timestamp':timestamp}
                if count<self.negative_confirmations:
                    status=HelmetStatus.UNKNOWN
            else:
                self.negatives.pop(head.track_id,None)
            # These are raw detector class scores, not calibrated probabilities.
            prediction=HelmetPrediction(status,0 if neutral else head.confidence,
                head.confidence if positive else 0,head.confidence if not positive and not neutral else 0)
            results.append(HeadResult(head.track_id,head.bbox,prediction,head.confidence,
                owner.track_id if owner else None,owner.vehicle_id if owner else None,
                owner.motorcycle_status if owner else 'uncertain',
                {'source':'independent_head_detector','person_match_available':owner is not None,
                 'confidence_scope':'raw detector class score; not calibrated probability',
                 **({'head_validity_score':validity,'head_validity_scope':'head/background softmax; calibration not measured'} if validity is not None else {})}))
        if self.classifier is not None:
            if frame is None:
                raise ValueError('Head classification requires the actual frame')
            indices,crops=[],[]
            for index,result in enumerate(results):
                crop=self.extractor.extract_box(frame,result.head_bbox)
                if crop is not None:
                    indices.append(index)
                    crops.append(crop[0])
                else:
                    self.classified_negatives.pop(result.track_id, None)
            classified=self.classifier.classify_batch(crops) if crops else []
            # Replacement mode abstains on invalid crops. Selective verification
            # retains native evidence unless the crop expert is confident.
            output=list(results) if self.verification_threshold is not None else [replace(r,helmet=HelmetPrediction()) for r in results]
            for index,prediction in zip(indices,classified,strict=True):
                threshold=self.verification_threshold or self.classification_threshold
                status=(HelmetStatus.HELMET if prediction.helmet_probability>=threshold
                    else HelmetStatus.NO_HELMET if prediction.no_helmet_probability>=threshold
                    else HelmetStatus.UNKNOWN)
                result=results[index]
                if self.verification_threshold is not None and status==HelmetStatus.UNKNOWN:
                    self.classified_negatives.pop(result.track_id,None)
                    continue
                if status==HelmetStatus.HELMET:
                    self.negatives.pop(result.track_id,None)
                if temporal and status==HelmetStatus.NO_HELMET:
                    previous=self.classified_negatives.get(result.track_id)
                    fresh=(previous is not None and previous['owner']==result.person_id
                        and iou(previous['bbox'],result.head_bbox)>=.3
                        and (timestamp is None or previous['timestamp'] is None
                             or 0<=timestamp-previous['timestamp']<=self.tracking.max_age_seconds))
                    count=previous['count']+1 if fresh else 1
                    self.classified_negatives[result.track_id]={'count':count,'owner':result.person_id,'bbox':result.head_bbox,'timestamp':timestamp}
                    if count<self.negative_confirmations:
                        status=HelmetStatus.UNKNOWN
                else:
                    self.classified_negatives.pop(result.track_id,None)
                output[index]=replace(result,helmet=replace(prediction,status=status),
                    evidence={**result.evidence,'source':'independent_detected_head_crop',
                              'confidence_scope':'crop classifier softmax; calibration not measured'})
            results=output
            self.classified_negatives={k:v for k,v in self.classified_negatives.items() if k in active}
        self.negatives={k:v for k,v in self.negatives.items() if k in active}
        return tuple(results)

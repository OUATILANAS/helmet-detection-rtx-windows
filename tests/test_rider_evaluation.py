import json

import cv2
import numpy as np

from helmet_rider.evaluation.metrics import score_frame, summarize
from helmet_rider.evaluation.reviews import load_reviews


def test_unknown_and_missing_riders_reduce_end_to_end_recall():
    samples = [{'bbox':[0,0,20,20],'label':'no_helmet'}, {'bbox':[30,0,50,20],'label':'no_helmet'}]
    predictions = [{'head_bbox':[0,0,20,20],'helmet':{'status':'unknown'}}]
    metrics = summarize([score_frame(predictions,samples)])
    assert metrics['per_class']['no_helmet']['recall'] == 0
    assert metrics['unknown_rate'] == .5
    assert metrics['missed_rider_rate'] == .5


def test_partial_reviews_do_not_count_unreviewed_heads_as_false_positives():
    pred = [{'head_bbox':[0,0,20,20],'helmet':{'status':'no_helmet'}}]
    assert score_frame(pred,[],fully_reviewed=False)['unmatched_predictions'] == {}
    assert score_frame(pred,[],fully_reviewed=True)['unmatched_predictions'] == {'no_helmet':1}


def test_duplicate_conflicts_are_excluded_instead_of_training_on_arbitrary_copy(tmp_path):
    for identifier,label in [('a','helmet'),('b','no_helmet')]:
        folder=tmp_path/'images'/'all'
        folder.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(folder/f'{identifier}.png'),np.zeros((20,20,3),dtype=np.uint8))
        meta=tmp_path/'metadata'/'all'
        meta.mkdir(parents=True,exist_ok=True)
        (meta/f'{identifier}.json').write_text(json.dumps({'id':identifier,'extension':'.png','sequence':'one',
          'samples':[{'bbox':[0,0,20,20],'label':label}]}))
    records,audit=load_reviews(tmp_path)
    assert records == []
    assert len(audit['conflicts']) == 1


def test_temporal_metrics_use_human_references_and_report_delay():
    from helmet_rider.evaluation.temporal import TemporalMetrics
    metric=TemporalMetrics()
    samples=[{'bbox':[0,0,20,20],'label':'no_helmet','entity_id':'rider-a'}]
    pred=[{'track_id':1,'helmet':{'status':'unknown'}}]
    metric.update('clip',2.,pred,samples,[(0,0,1.)])
    pred=[{'track_id':2,'helmet':{'status':'no_helmet'}}]
    metric.update('clip',4.,pred,samples,[(0,0,1.)])
    summary=metric.summary()
    assert summary['id_switches']==1
    assert summary['mean_confirmation_delay_seconds']==2.


def test_mislocalized_runtime_crops_are_excluded_from_class_supervision(tmp_path):
    from training.finetune_head_classifier import HeadCropDataset
    for split in ('train','val','test'):
        folder=tmp_path/'images'/split
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder/'head.png'),np.zeros((60,80,3),dtype=np.uint8))
    manifest={'rows':[{'path':'images/train/head.png','split':'train','label':'helmet',
                      'bbox':[0,0,20,20],'reviewed_bbox':[40,40,60,60]},
                     {'path':'images/train/head.png','split':'train','label':'no_helmet',
                      'bbox':[10,10,40,40],'reviewed_bbox':[10,10,40,40]}]}
    (tmp_path/'runtime_manifest.json').write_text(json.dumps(manifest))
    dataset=HeadCropDataset(tmp_path,'train',training=False)
    assert len(dataset)==1
    assert dataset.rows[0][1]==1
def test_classifier_training_padding_matches_actual_runtime_input():
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    from helmet_rider.config import ClassifierConfig
    from helmet_rider.helmet.classifier import EfficientNetHelmetClassifier
    from training.finetune_head_classifier import HeadCropDataset

    rgb = np.arange(9*17*3,dtype=np.uint8).reshape(9,17,3)
    runtime = EfficientNetHelmetClassifier.__new__(EfficientNetHelmetClassifier)
    runtime.config = ClassifierConfig(input_size=224)
    runtime.torch = torch
    runtime.mean = torch.tensor([.485,.456,.406])[:,None,None]
    runtime.std = torch.tensor([.229,.224,.225])[:,None,None]
    actual = runtime._preprocess(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    training = transforms.Normalize([.485,.456,.406],[.229,.224,.225])(
        transforms.ToTensor()(HeadCropDataset._pad_resize(Image.fromarray(rgb))))
    assert torch.equal(actual,training)

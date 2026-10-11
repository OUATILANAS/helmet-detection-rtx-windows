"""Strict full-frame comparison; keeps misses, abstentions and false positives."""
import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from helmet_rider.config import load_config
from helmet_rider.pipeline.pipeline import HelmetRiderPipeline
from helmet_rider.utils.results import result_to_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_strict_predictions import summarize_rows  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('evaluation/validation.json'))
    parser.add_argument('--baseline', type=Path, default=Path('models/helmet_localizer_video_20261010.pt'))
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--pipeline-config', type=Path, help='Use the complete app, replacing only its localized head model for the candidate')
    parser.add_argument('--imgsz', type=int, default=768)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--device', default='0')
    parser.add_argument('--output', type=Path, default=Path('data/outputs/windows-comparison'))
    args = parser.parse_args()
    if args.device != 'cpu' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_num_threads(2)
    rows = json.loads(args.manifest.read_text())['rows']
    if any(not r['fully_reviewed'] for r in rows):
        parser.error('Strict metrics require fully reviewed frames, not partial publisher boxes')
    args.output.mkdir(parents=True, exist_ok=False)
    predictions, report = {}, {}
    for name, weights in [('baseline', args.baseline), ('candidate', args.candidate)]:
        output = []
        if args.pipeline_config:
            cfg = load_config(args.pipeline_config)
            cfg = replace(cfg, detector=replace(cfg.detector, device=args.device),
                          classifier=replace(cfg.classifier, localized_head_model=str(weights)))
            model = HelmetRiderPipeline(cfg)
        else:
            model = YOLO(weights)
            normalized = {int(k): str(v).lower().replace('-', '_') for k,v in model.names.items()}
            if normalized != {0: 'helmet', 1: 'no_helmet'}:
                parser.error('Detector comparison expects helmet=0, no_helmet=1. Use the matching baseline weights.')
        for row in rows:
            image = Path(row['image'])
            if not image.is_absolute():
                image = args.manifest.parent/image
            if hashlib.sha256(image.read_bytes()).hexdigest() != row['image_sha256']:
                raise ValueError('Evaluation image checksum changed')
            frame = cv2.imread(str(image))
            if args.pipeline_config:
                result = result_to_dict(model.process(frame, temporal_consistency=False))
            else:
                raw = model.predict(frame, imgsz=args.imgsz, device=args.device,
                                    conf=args.confidence, verbose=False, max_det=1000)[0]
                heads = [{'head_bbox': box, 'confidence': confidence,
                          'helmet': {'status': 'helmet' if int(cls) == 0 else 'no_helmet'}}
                         for box, confidence, cls in zip(raw.boxes.xyxy.tolist(), raw.boxes.conf.tolist(), raw.boxes.cls.tolist(), strict=True)]
                result = {'heads': heads}
            output.append({**row, 'result': result})
        predictions[name] = output
        report[name] = {scope: summarize_rows([r for r in output if scope == 'combined' or
                      r['sequence'].startswith('still-val:') == (scope == 'still')])
                        for scope in ('combined', 'video', 'still')}
        del model
    (args.output/'predictions.json').write_text(json.dumps(predictions, indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps({'results': report,
        'scope': 'Strict complete-frame static head/helmet F1, IoU>=.3 and min edge 8px. Not crop F1, not track/event F1.',
        'pipeline': bool(args.pipeline_config), 'deployment_changed': False}, indent=2)+'\n')
    print(json.dumps({name: {scope: metric['macro_f1'] for scope, metric in values.items()} for name,values in report.items()}, indent=2))


if __name__ == '__main__':
    main()

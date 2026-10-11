"""Validation-only feasibility check of pretrained physical-head proposals."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from helmet_rider.evaluation.metrics import match_heads


def nms(detections, threshold=.45):
    if not detections:
        return []
    boxes = [[d['head_bbox'][0], d['head_bbox'][1], d['head_bbox'][2]-d['head_bbox'][0],
              d['head_bbox'][3]-d['head_bbox'][1]] for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, [d['confidence'] for d in detections], .03, threshold)
    return [detections[int(i)] for i in np.asarray(indices).reshape(-1)]


def detect(session, image, grid, color, tile_size=768):
    height, width = image.shape[:2]
    tw, th = min(width, tile_size), min(height, tile_size)
    input_meta = session.get_inputs()[0]
    ih, iw = input_meta.shape[2:]
    if not isinstance(ih, int) or not isinstance(iw, int):
        raise ValueError('The pilot requires a fixed spatial ONNX input')
    rectangles = [(0, 0, width, height)] if grid == 1 else [
        (int(x), int(y), tw, th) for y in np.linspace(0, height-th, grid)
        for x in np.linspace(0, width-tw, grid)]
    results = []
    for left, top, w, h in rectangles:
        crop = image[top:top+h, left:left+w]
        scale = min(iw/w, ih/h)
        nw, nh = int(w*scale), int(h*scale)
        px, py = (iw-nw)//2, (ih-nh)//2
        canvas = np.full((ih, iw, 3), 114, np.uint8)
        canvas[py:py+nh, px:px+nw] = cv2.resize(crop, (nw, nh))
        if color == 'rgb':
            canvas = canvas[:, :, ::-1]
        tensor = np.transpose(canvas.astype(np.float32)/255, (2, 0, 1))[None]
        outputs = session.run(None, {input_meta.name: tensor})
        if len(outputs) == 3:
            boxes, scores, classes = outputs
            valid = (scores[0, :, 0] >= .03) & (classes[0, :, 0] == 1)
            selected_boxes, selected_scores = boxes[0, valid], scores[0, valid, 0]
        elif len(outputs) == 1 and outputs[0].shape[1] == 5:
            raw = outputs[0][0].T
            valid = raw[:, 4] >= .03
            selected_boxes, selected_scores = raw[valid, :4], raw[valid, 4]
        else:
            raise ValueError('Unsupported ONNX output contract')
        # Verified ONNX graph: boxes are unmodified YOLO cx/cy/w/h, NOT xyxy.
        for box, score in zip(selected_boxes, selected_scores, strict=True):
            cx, cy, bw, bh = map(float, box)
            x1, y1 = (cx-bw/2-px)/scale+left, (cy-bh/2-py)/scale+top
            x2, y2 = (cx+bw/2-px)/scale+left, (cy+bh/2-py)/scale+top
            bbox = [max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2)]
            if min(bbox[2]-bbox[0], bbox[3]-bbox[1]) >= 8:
                results.append({'head_bbox': bbox, 'confidence': float(score)})
    return nms(results)


def physical_score(rows, confidence):
    tp = fp = fn = 0
    for row in rows:
        truth = [s for s in row['samples'] if min(s['bbox'][2]-s['bbox'][0], s['bbox'][3]-s['bbox'][1]) >= 8]
        predictions = [p for p in row['predictions'] if p['confidence'] >= confidence]
        matched, extra, missed = match_heads(predictions, truth, .3)
        tp += len(matched)
        fp += len(extra)
        fn += len(missed)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp/max(1, tp+fp),
            'recall': tp/max(1, tp+fn), 'f1': 2*tp/max(1, 2*tp+fp+fn)}


def main():
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    root = Path('data/outputs/crowdhuman-head-pilot-20261011')
    manifest_path = Path('dataset/strict-head-review-20261011/val.json')
    manifest = json.loads(manifest_path.read_text())
    if any(r['split'] != 'val' for r in manifest['rows']):
        raise ValueError('Only validation media allowed in this pilot')
    plan = {'policy': __doc__, 'grids': [1, 4, 8], 'colors': ['bgr', 'rgb'],
            'confidence_grid': [.1, .25, .4], 'minimum_head_edge': 8, 'iou': .3,
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'input_contract': 'Graph inspected: input NCHW float32, 640px, boxes cx/cy/w/h, scores/classes separate. Publisher example incorrectly assumes a single output.',
            'gate': 'Physical-head F1 >= .65 on complete reviewed validation before classification/integration; no training and no reserved inference.'}
    (root/'PILOT-PLAN.json').write_text(json.dumps(plan, indent=2)+'\n')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession('models/crowdhuman_yolo11s_head_person.onnx', sess_options=options,
                                   providers=['CPUExecutionProvider'])
    started = time.monotonic()
    predictions, results = {}, {}
    for grid in plan['grids']:
        for color in plan['colors']:
            name = f'grid{grid}-{color}'
            rows = []
            for row in manifest['rows'][:args.limit]:
                image = cv2.imread(row['image'])
                rows.append({**row, 'predictions': detect(session, image, grid, color)})
                print(name, len(rows), '/', args.limit, flush=True)
            predictions[name] = rows
            results[name] = {str(c): physical_score(rows, c) for c in plan['confidence_grid']}
            (root/'predictions.json').write_text(json.dumps(predictions, indent=2)+'\n')
            (root/'physical-metrics.json').write_text(json.dumps({'elapsed_seconds': time.monotonic()-started,
                                                                'frames': len(rows), 'results': results}, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

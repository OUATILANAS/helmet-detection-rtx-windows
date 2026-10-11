"""Summarize frozen strict outputs, including physical-head false positives."""
import argparse
import json
from pathlib import Path

from helmet_rider.evaluation.metrics import match_heads, summarize
from helmet_rider.evaluation.strict_heads import strict_score_frame


def summarize_rows(rows, threshold=None):
    frames = []
    counts = {'true_positive':0,'false_positive':0,'false_negative':0}
    class_extras = {s:0 for s in ('helmet','no_helmet','unknown')}
    ignored = 0
    for row in rows:
        result = row['result']
        predictions = result['heads'] if result.get('heads') is not None else result['riders']
        if threshold is not None:
            predictions = [p for p in predictions if p['confidence']>=threshold]
        score = strict_score_frame(predictions,row['samples'],fully_reviewed=row['fully_reviewed'])
        frames.append(score)
        ignored += score['strict_scope']['ignored_predictions_on_unresolved_heads']
        for status in class_extras:
            class_extras[status] += score['unmatched_predictions'].get(status,0)
        truth = [s for s in row['samples'] if min(s['bbox'][2]-s['bbox'][0],s['bbox'][3]-s['bbox'][1])>=8]
        physical = [p for p in predictions if p.get('head_bbox') and min(p['head_bbox'][2]-p['head_bbox'][0],p['head_bbox'][3]-p['head_bbox'][1])>=8]
        matched,extra,miss = match_heads(physical,truth,.3)
        counts['true_positive'] += len(matched)
        counts['false_positive'] += len(extra)
        counts['false_negative'] += len(miss)
    metrics = summarize(frames)
    metrics['macro_f1'] = sum(v['f1'] or 0 for v in metrics['per_class'].values())/2
    metrics['false_positive_status_predictions'] = class_extras
    metrics['ignored_on_explicit_unknown_heads'] = ignored
    tp,fp,fn = (counts[k] for k in ('true_positive','false_positive','false_negative'))
    metrics['physical_head_detection'] = {**counts,'precision':tp/max(1,tp+fp),
        'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)}
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-sweep',action='store_true')
    args = p.parse_args()
    data = json.loads(args.predictions.read_text())
    if args.validation_sweep and any(r['split']!='val' for rows in data.values() for r in rows):
        raise ValueError('Confidence selection is forbidden on test frames')
    report = {'policy':__doc__,'results':{name:summarize_rows(rows) for name,rows in data.items()}}
    if args.validation_sweep:
        report['confidence_sweep'] = {name:{str(t):summarize_rows(rows,t) for t in (.05,.1,.2)}
            for name,rows in data.items() if name!='baseline'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{'macro_f1':m['macro_f1'],'head_detection_f1':m['physical_head_detection']['f1'],'false_positives':m['false_positive_status_predictions']} for name,m in report['results'].items()},indent=2))


if __name__=='__main__':
    main()

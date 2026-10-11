#!/usr/bin/env python3
"""Export reviewed UNKNOWN/errors/misses for head-box and training triage."""

import argparse
import json
from pathlib import Path

from helmet_rider.evaluation.reviews import load_reviews


def export_errors(metrics: dict, records: list[dict]) -> dict:
    by_id = {record['id']: record for record in records}
    errors = []
    for frame in metrics['frames']:
        record = by_id[frame['id']]
        for trace in frame['traces']:
            if trace['truth'] == trace['predicted']:
                continue
            sample = record['samples'][trace['sample_index']]
            prediction = frame['predictions'][trace['prediction_index']]
            errors.append({
                'record_id': record['id'], 'image': record['image'],
                'sequence': record['sequence'], 'kind': trace['predicted'],
                'reviewed_label': sample['label'], 'reviewed_bbox': sample['bbox'],
                'predicted_bbox': prediction['head_bbox'], 'iou': trace['iou'],
                'evidence': prediction.get('evidence', {}),
            })
        for index in frame['missed_sample_indices']:
            sample = record['samples'][index]
            if sample['label'] not in {'helmet', 'no_helmet'}:
                continue
            errors.append({
                'record_id': record['id'], 'image': record['image'],
                'sequence': record['sequence'], 'kind': 'missed_match',
                'reviewed_label': sample['label'], 'reviewed_bbox': sample['bbox'],
                'note': 'Check head localization and review-box geometry before treating as an absent rider.',
            })
    return {
        'scope': 'Clean human reviews only. A model mismatch is not a new training label.',
        'errors': errors, 'count': len(errors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--review-data', type=Path, default=Path('dataset/helmet_status_review'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, audit = load_reviews(args.review_data)
    report = export_errors(json.loads(args.metrics.read_text()), records)
    report['audit'] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'count': report['count'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()

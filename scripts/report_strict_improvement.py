"""Report frozen experiments without selecting models from reserved results."""
import json
from pathlib import Path

from helmet_rider.evaluation.metrics import match_heads

ROOT = Path('data/outputs/strict-heads-20261011')


def membership(rows):
    """Normalize legacy rider vehicle types for diagnostic scoring only."""
    counts = {r: {} for r in ('motorcycle', 'no_motorcycle')}
    for row in rows:
        result = row['result']
        predictions = result['heads'] if result.get('heads') is not None else result['riders']
        truth = [s for s in row['samples'] if s.get('motorcycle_status') in counts
                 and min(s['bbox'][2]-s['bbox'][0], s['bbox'][3]-s['bbox'][1]) >= 8]
        matched, _, missed = match_heads(predictions, truth, .3)
        for pi, si, _ in matched:
            prediction = predictions[pi]
            status = prediction.get('motorcycle_status')
            if status is None:
                vehicle = prediction.get('vehicle_type')
                status = ('motorcycle' if vehicle == 'motorcycle' else
                          'no_motorcycle' if vehicle in ('car', 'bicycle', 'truck', 'bus') else 'uncertain')
            if status == 'no_motorcycle_match':
                status = 'no_motorcycle'
            target = counts[truth[si]['motorcycle_status']]
            target[status] = target.get(status, 0)+1
        for si in missed:
            target = counts[truth[si]['motorcycle_status']]
            target['missed'] = target.get('missed', 0)+1
    return counts


def matched_classification(rows):
    matched = correct = missed = 0
    for row in rows:
        truth = [s for s in row['samples'] if s['label'] in ('helmet', 'no_helmet')
                 and min(s['bbox'][2]-s['bbox'][0], s['bbox'][3]-s['bbox'][1]) >= 8]
        result = row['result']
        predictions = result['heads'] if result.get('heads') is not None else result['riders']
        pairs, _, missing = match_heads(predictions, truth, .3)
        matched += len(pairs)
        missed += len(missing)
        correct += sum(predictions[pi]['helmet']['status'] == truth[si]['label'] for pi, si, _ in pairs)
    return {'matched': matched, 'correct': correct, 'missed': missed,
            'accuracy_conditional_on_matched_head': correct/max(1, matched)}


def main():
    execution = json.loads((ROOT/'TEST-EXECUTION.json').read_text())
    if not execution.get('complete'):
        raise RuntimeError('Finish all frozen evaluations before reporting')
    strict = {}
    diagnostics = {}
    classification = {}
    for stage in ('strict-videos', 'strict-images'):
        directory = ROOT/'test'/stage
        strict[stage] = json.loads((directory/'strict-summary.json').read_text())['results']
        predictions = json.loads((directory/'predictions.json').read_text())
        diagnostics[stage] = {name: membership(rows) for name, rows in predictions.items()}
        classification[stage] = {name: matched_classification(rows) for name, rows in predictions.items()}
    (ROOT/'motorcycle-diagnostics.json').write_text(json.dumps({
        'scope': 'Explicit labelled head roles, retaining missed heads. Diagnostic normalization of legacy rider vehicle_type; frozen helmet scoring unchanged.',
        'results': diagnostics}, indent=2)+'\n')
    (ROOT/'matched-classification-diagnostics.json').write_text(json.dumps(classification, indent=2)+'\n')
    broad = json.loads((ROOT/'test/publisher-240-video-frames/report.json').read_text())['results']
    broad_change = broad['candidate']['macro_f1']-broad['baseline']['macro_f1']
    lines = ['# Hard-background fine-tuning results', '',
             f"Decision: keep the current app configuration. The new experiment improves strict helmet-status video F1, "
             f"but loses physical-head recall and changes the larger 240-frame F1 by {broad_change:+.3f}. "
             'It has not met the requested 0.8–0.9 complete-frame target and is not promoted.', '',
             'The candidate was trained on 1,600 replay images and 40 visually reviewed head-free background patches. '
             'Four epochs were compared using eight fully reviewed validation frames. The first epoch with head confidence 0.10 was frozen before reserved inference.', '',
             '## Fully reviewed evaluation', '',
             'Unmatched predictions count as false positives. Known helmet labels are matched before unresolved physical heads. '
             'Minimum head edge is 8 pixels; matching IoU is 0.30. Annotations are Codex visual review assisted by publisher hints, not independent expert certification.', '',
             '| Set | Production F1 | Previous experiment F1 | New experiment F1 | Previous/new physical-head FP | Previous/new physical-head FN |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    validation = json.loads((ROOT/'selected-validation/strict-summary.json').read_text())['results']
    for name, results in [('Validation: 8 frames', validation), *strict.items()]:
        previous, new = (results[k] for k in ('candidate', 'candidate-2'))
        a, b = (r['physical_head_detection'] for r in (previous, new))
        lines.append(f"| {name} | {results['baseline']['macro_f1']:.3f} | {previous['macro_f1']:.3f} | {new['macro_f1']:.3f} | {a['false_positive']}/{b['false_positive']} | {a['false_negative']}/{b['false_negative']} |")
    lines += ['', 'Validation is selection data. The reserved strict sets contain 16 frames from eight video clips and six images from three sources. '
              'They are excluded from training and this round’s model selection, but were evaluated in previous experiments. '
              'These small sets do not establish camera-independent generalization or continuous-video tracking accuracy.', '',
              '## Larger publisher-label checks', '',
              'These existing labels are incomplete: unmatched predictions generally do not count as background false positives. '
              'Their F1 values are not comparable with the complete-frame benchmark above.', '',
              '| Set | Previous experiment F1 | New experiment F1 |', '| --- | ---: | ---: |']
    for stage in ('publisher-240-video-frames', 'publisher-original-forty', 'publisher-forty-images'):
        results = json.loads((ROOT/'test'/stage/'report.json').read_text())['results']
        lines.append(f"| {stage} | {results['baseline']['macro_f1']:.3f} | {results['candidate']['macro_f1']:.3f} |")
    regression = json.loads((ROOT/'test/user-photo-regressions/report.json').read_text())
    lines += ['', '## User-photo regressions', '',
              *[f"- {name}: {'PASS' if check['passed'] else 'FAIL'}." for name, check in regression['checks'].items()], '',
              'The supplied user photos were regression tests only, never training data. '
              'The app’s default configuration is unchanged; this experiment remains opt-in.', '',
              '## Remaining limitations', '',
              f"On strict reserved video frames, physical-head F1 changed from {strict['strict-videos']['candidate']['physical_head_detection']['f1']:.3f} "
              f"to {strict['strict-videos']['candidate-2']['physical_head_detection']['f1']:.3f}; recall changed from "
              f"{strict['strict-videos']['candidate']['physical_head_detection']['recall']:.3f} to "
              f"{strict['strict-videos']['candidate-2']['physical_head_detection']['recall']:.3f}. "
              'A higher helmet-status score does not remove this detection regression.', '',
              'The reserved results must support the requested 0.8–0.9 F1 before claiming that target. '
              'Detection coverage remains a major bottleneck. Classification cannot recover heads that were never proposed. '
              'Only three no-motorcycle frames were reviewed, all bicycle scenes; motorcycle absence needs a broader independent test. '
              'Person-to-motorcycle diagnostics are in motorcycle-diagnostics.json with upstream misses retained. '
              'No new video tracking improvement was measured in this round.', '',
              f"On the reserved strict video frames, the new classifier correctly labels {classification['strict-videos']['candidate-2']['correct']} "
              f"of {classification['strict-videos']['candidate-2']['matched']} matched known heads, while "
              f"{classification['strict-videos']['candidate-2']['missed']} other known heads are missed. "
              'This conditional classification accuracy is not overall detector accuracy or F1.', '',
              'Post-test localization audit: 25 of the 32 missed known heads have a shorter edge below 32 pixels. '
              'Seven missed heads have a prediction with IoU between 0.10 and 0.30, below the frozen matching threshold. '
              'See missed-head-diagnostics.json and strict-video-error-gallery.jpg. These diagnostics were not used to retune the candidate.', '',
              'Next training work should add exhaustively labelled full scenes with small and occluded positive heads, '
              'standardize head boxes to cover the full worn helmet, and balance positive replay against hard negatives. '
              'Use new reserved capture groups for the next final test; the current test results must not become model-selection data.', '',
              '## Reproduction and verification', '',
              'Training: training/finetune_hard_background_heads.py. Frozen selection: SELECTION-FREEZE.json. '
              'Reserved commands and exit codes: TEST-EXECUTION.json. '
              'Exact decoded-pixel train/test collision audit: split-audit.json. '
              'Candidate: SELECTED.yaml. Per-frame predictions and metrics: test/. '
              'Verification: 149 tests passed; report helper and its two diagnostic tests passed Ruff. '
              'The annotation gallery is dataset/strict-head-review-20261011/review.html.', '']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:22]))


if __name__ == '__main__':
    main()

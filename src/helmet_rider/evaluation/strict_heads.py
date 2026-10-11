"""Complete-frame metrics with explicit small/ambiguous-head scope.

Known labels are matched first. Remaining predictions overlapping an unresolved
physical head are ignored for helmet-status precision; they cannot consume a
known head match. All other unmatched helmet/no-helmet predictions count as FP.
"""
from helmet_rider.association.geometry import iou
from helmet_rider.evaluation.metrics import LABELS, match_heads, score_frame


def strict_score_frame(predictions, samples, *, fully_reviewed=False, threshold=.3,
                       minimum_edge=8):
    if not fully_reviewed:
        raise ValueError("Strict scoring requires complete visual annotation")
    known = [s for s in samples if s['label'] in LABELS and
             min(s['bbox'][2]-s['bbox'][0], s['bbox'][3]-s['bbox'][1]) >= minimum_edge]
    uncertain = [s for s in samples if s not in known]
    eligible = [p for p in predictions if p.get('head_bbox') and
                min(p['head_bbox'][2]-p['head_bbox'][0], p['head_bbox'][3]-p['head_bbox'][1]) >= minimum_edge]
    _, extras, _ = match_heads(eligible, known, threshold)
    ignored = {pi for pi in extras if any(iou(eligible[pi]['head_bbox'],s['bbox']) >= threshold for s in uncertain)}
    # Only one prediction per uncertain physical head is exempt. Duplicate
    # detections still count against precision, as they do on known heads.
    if ignored:
        unknown_predictions = [eligible[i] for i in sorted(ignored)]
        matches, _, _ = match_heads(unknown_predictions, uncertain, threshold)
        index = sorted(ignored)
        ignored = {index[pi] for pi, _, _ in matches}
    scored = score_frame([p for i,p in enumerate(eligible) if i not in ignored], known,
                         fully_reviewed=True, threshold=threshold)
    scored['strict_scope'] = {'minimum_head_edge_pixels':minimum_edge,
        'known_heads':len(known), 'unresolved_or_subminimum_heads':len(uncertain),
        'ignored_predictions_on_unresolved_heads':len(ignored),
        'subminimum_predictions':len(predictions)-len(eligible)}
    return scored


def motorcycle_membership_score(predictions, samples, *, threshold=.3):
    """Score explicit head-to-motorcycle roles, retaining upstream misses."""
    roles = ('motorcycle','no_motorcycle')
    truth = [s for s in samples if s.get('motorcycle_status') in roles and
             min(s['bbox'][2]-s['bbox'][0],s['bbox'][3]-s['bbox'][1]) >= 8]
    matches, _, missed = match_heads(predictions,truth,threshold)
    confusion = {role:{} for role in roles}
    for pi,si,_ in matches:
        status = predictions[pi].get('motorcycle_status','uncertain')
        if status == 'no_motorcycle_match':
            status = 'no_motorcycle'
        row = confusion[truth[si]['motorcycle_status']]
        row[status] = row.get(status,0)+1
    for si in missed:
        row = confusion[truth[si]['motorcycle_status']]
        row['missed'] = row.get('missed',0)+1
    return confusion

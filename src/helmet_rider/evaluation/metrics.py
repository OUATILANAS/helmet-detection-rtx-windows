"""Rider metrics include abstentions and upstream misses in recall."""

from __future__ import annotations

from collections import Counter

from helmet_rider.association.geometry import iou

LABELS = ("helmet", "no_helmet")


def match_heads(predictions: list[dict], samples: list[dict], threshold: float = 0.3):
    candidates = sorted(
        (
            (iou(p["head_bbox"], s["bbox"]), pi, si)
            for pi, p in enumerate(predictions)
            if p.get("head_bbox")
            for si, s in enumerate(samples)
        ),
        reverse=True,
    )
    used_p, used_s, matches = set(), set(), []
    for overlap, pi, si in candidates:
        if overlap < threshold:
            break
        if pi not in used_p and si not in used_s:
            used_p.add(pi)
            used_s.add(si)
            matches.append((pi, si, overlap))
    return (
        matches,
        sorted(set(range(len(predictions))) - used_p),
        sorted(set(range(len(samples))) - used_s),
    )


def score_frame(
    predictions: list[dict],
    samples: list[dict],
    *,
    fully_reviewed: bool = False,
    threshold: float = 0.3,
):
    matches, extras, missed = match_heads(predictions, samples, threshold)
    confusion = {label: Counter() for label in LABELS}
    traces = []
    sizes = {
        name: {label: Counter() for label in LABELS} for name in ("below24", "24to47", "48plus")
    }
    associations = Counter()
    for pi, si, overlap in matches:
        truth, pred = samples[si], predictions[pi]
        if truth["label"] not in LABELS:
            continue
        status = pred["helmet"]["status"]
        confusion[truth["label"]][status] += 1
        edge = min(truth["bbox"][2] - truth["bbox"][0], truth["bbox"][3] - truth["bbox"][1])
        sizes["below24" if edge < 24 else "24to47" if edge < 48 else "48plus"][truth["label"]][
            status
        ] += 1
        traces.append(
            {
                "truth": truth["label"],
                "predicted": status,
                "iou": overlap,
                "sample_index": si,
                "prediction_index": pi,
                "evidence": pred.get("evidence", {}),
            }
        )
        if truth.get("vehicle_bbox"):
            associations["reviewed"] += 1
            associations["correct"] += int(
                bool(pred.get("vehicle_bbox"))
                and iou(truth["vehicle_bbox"], pred["vehicle_bbox"]) >= 0.5
            )
    for si in missed:
        truth = samples[si]
        if truth["label"] in LABELS:
            confusion[truth["label"]]["missed"] += 1
            edge = min(truth["bbox"][2] - truth["bbox"][0], truth["bbox"][3] - truth["bbox"][1])
            sizes["below24" if edge < 24 else "24to47" if edge < 48 else "48plus"][truth["label"]][
                "missed"
            ] += 1
    unmatched = (
        Counter(predictions[i]["helmet"]["status"] for i in extras) if fully_reviewed else Counter()
    )
    return {
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "unmatched_predictions": dict(unmatched),
        "unmatched_predictions_scored": fully_reviewed,
        "traces": traces,
        "associations": dict(associations),
        "head_size_confusion": {
            size: {k: dict(v) for k, v in rows.items()} for size, rows in sizes.items()
        },
        "missed_sample_indices": missed,
    }


def summarize(frames: list[dict]) -> dict:
    rows = {label: Counter() for label in LABELS}
    extras, associations = Counter(), Counter()
    for frame in frames:
        for label in LABELS:
            rows[label].update(frame["confusion"][label])
        extras.update(frame["unmatched_predictions"])
        associations.update(frame["associations"])
    metrics = {}
    for label in LABELS:
        tp = rows[label][label]
        fp = sum(rows[other][label] for other in LABELS if other != label) + extras[label]
        support = sum(rows[label].values())
        recall = tp / support if support else None
        precision = tp / (tp + fp) if tp + fp else None
        metrics[label] = {
            "precision_on_reviewed_scope": precision,
            "recall": recall,
            "support": support,
            "f1": 2 * tp / (2 * tp + fp + support - tp) if support or fp else None,
        }
    support = sum(sum(row.values()) for row in rows.values())
    return {
        "per_class": metrics,
        "confusion": {k: dict(v) for k, v in rows.items()},
        "known_status_support": support,
        "unknown_rate": sum(r["unknown"] for r in rows.values()) / support if support else None,
        "missed_rider_rate": sum(r["missed"] for r in rows.values()) / support if support else None,
        "association_correctness": associations["correct"] / associations["reviewed"]
        if associations["reviewed"]
        else None,
        "exhaustively_reviewed_frames": sum(f["unmatched_predictions_scored"] for f in frames),
        "precision_scope": "Unmatched predictions count only on explicitly fully reviewed frames.",
    }

"""Read local reviews without silently selecting conflicting duplicate labels."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2


def load_reviews(root: Path) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    invalid = []
    for path in sorted((root / "metadata").glob("*/*.json")):
        record = json.loads(path.read_text())
        image = (
            root / "images" / path.parent.name / f"{record['id']}{record.get('extension', '.jpg')}"
        )
        frame = cv2.imread(str(image))
        if frame is None:
            invalid.append(str(path))
            continue
        digest = hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest()
        groups[digest].append({**record, "image": str(image), "image_sha256": digest})
    clean, conflicts, duplicate_count = [], [], 0
    for digest, copies in groups.items():
        fingerprints = set()
        for record in copies:
            labels = sorted(
                (tuple(round(float(x), 2) for x in s["bbox"]), s["label"])
                for s in record.get("samples", [])
            )
            fingerprints.add(json.dumps([labels, bool(record.get("fully_reviewed", False))]))
        if len(fingerprints) > 1 or len({r["sequence"] for r in copies}) > 1:
            conflicts.append(
                {
                    "image_sha256": digest,
                    "ids": [r["id"] for r in copies],
                    "reason": "Conflicting annotations or capture groups; all copies excluded.",
                }
            )
            continue
        duplicate_count += len(copies) - 1
        clean.append(copies[0])
    return clean, {
        "records_read": sum(map(len, groups.values())),
        "unique_clean_images": len(clean),
        "duplicates_skipped": duplicate_count,
        "conflicts": conflicts,
        "invalid_images": invalid,
        "capture_groups": sorted({r["sequence"] for r in clean}),
    }


def split_records_by_sequence(records: list[dict]) -> dict[str, list[dict]]:
    """Create deterministic group-disjoint train/val/test splits."""
    groups = sorted({str(record["sequence"]) for record in records})
    if len(groups) < 2:
        raise ValueError("Save frames from at least two different capture groups to export.")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(group.encode("utf-8")).hexdigest(),
    )
    if len(ordered) >= 3:
        test_count = max(1, round(len(ordered) * 0.15))
        val_count = max(1, round(len(ordered) * 0.15))
        if test_count + val_count >= len(ordered):
            test_count, val_count = 1, 1
        test_groups = set(ordered[-test_count:])
        val_groups = set(ordered[-test_count - val_count:-test_count])
    else:
        test_groups = set()
        val_groups = {ordered[-1]}
    split_records = {"train": [], "val": [], "test": []}
    for record in records:
        group = str(record["sequence"])
        split = "test" if group in test_groups else (
            "val" if group in val_groups else "train"
        )
        split_records[split].append(record)
    return split_records

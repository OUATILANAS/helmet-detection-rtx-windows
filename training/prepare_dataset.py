#!/usr/bin/env python3
"""Validate head crops and split by video/sequence group to prevent leakage."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from helmet_rider.config import load_config
from helmet_rider.helmet.head_roi import HeadROIExtractor

LABELS = {"helmet": "helmet", "no_helmet": "no_helmet", "no helmet": "no_helmet"}
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    sequence: str
    bbox: tuple[float, float, float, float] | None = None


def label_value(value: str) -> str:
    label = LABELS.get(value.strip().lower().replace("-", "_"))
    if label is None:
        raise ValueError(f"Unsupported label {value!r}; use helmet or no_helmet.")
    return label


def discover(source: Path, annotations: Path | None) -> list[Sample]:
    samples = []
    if annotations:
        with annotations.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                path = Path(row["path"])
                if not path.is_absolute():
                    path = source / path
                sequence = (row.get("sequence") or "").strip()
                if not sequence:
                    raise ValueError(
                        "Every annotation needs a sequence value (for example a video ID)."
                    )
                coordinates = [row.get(key, "").strip() for key in ("x1", "y1", "x2", "y2")]
                bbox = None
                if any(coordinates):
                    if not all(coordinates):
                        raise ValueError(
                            "Provide all x1,y1,x2,y2 columns for head extraction."
                        )
                    bbox = tuple(float(value) for value in coordinates)
                samples.append(Sample(path, label_value(row["label"]), sequence, bbox))
        return samples

    for folder in sorted(source.iterdir()):
        if not folder.is_dir() or folder.name.lower().replace("-", "_") not in LABELS:
            continue
        label = label_value(folder.name)
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            relative_parent = path.parent.relative_to(folder)
            # A named subfolder can represent a video/sequence. Flat folders
            # use each file as its own group; pass an annotation CSV to prevent
            # adjacent video frames leaking between splits.
            sequence = str(relative_parent) if str(relative_parent) != "." else path.stem
            samples.append(Sample(path, label, sequence))
    return samples


def split_groups(samples: list[Sample], seed: int) -> dict[str, str]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.sequence].append(sample)
    names = list(groups)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda name: len(groups[name]), reverse=True)
    target = {name: max(1.0, ratio * len(samples)) for name, ratio in SPLITS}
    assigned: dict[str, str] = {}
    counts = Counter()
    for name in names:
        split = min(
            SPLITS,
            key=lambda item: counts[item[0]] / target[item[0]],
        )[0]
        assigned[name] = split
        counts[split] += len(groups[name])
    return assigned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Folder containing helmet/ and no_helmet/")
    parser.add_argument("--annotations", help="CSV columns: path,label,sequence")
    parser.add_argument("--output", default="dataset")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--size", type=int, default=128, help="Output crop width and height")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"),
        help="YAML profile providing the head ROI settings",
    )
    parser.add_argument("--clean", action="store_true", help="Clear the destination split folders first")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if args.size < 16:
        parser.error("size must be at least 16 pixels")
    annotations = Path(args.annotations).resolve() if args.annotations else None
    samples = discover(source, annotations)
    if not samples:
        raise SystemExit("No helmet/no_helmet images were found.")
    output.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for split, _ in SPLITS:
            shutil.rmtree(output / split, ignore_errors=True)
    group_to_split = split_groups(samples, args.seed)
    roi_extractor = HeadROIExtractor(load_config(args.config).head_roi)
    stats: dict[str, Counter] = {split: Counter() for split, _ in SPLITS}
    invalid = []
    seen_output = set()
    for sample_index, sample in enumerate(samples):
        split = group_to_split[sample.sequence]
        try:
            with Image.open(sample.path) as image:
                image.verify()
            with Image.open(sample.path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
            if sample.bbox is not None:
                extracted = roi_extractor.extract(
                    np.asarray(image), sample.bbox
                )
                if extracted is None:
                    raise ValueError("Person box is too small for a head ROI.")
                image = Image.fromarray(extracted[0])
        except (OSError, ValueError) as exc:
            invalid.append({"path": str(sample.path), "error": str(exc)})
            continue
        destination_dir = output / split / sample.label
        destination_dir.mkdir(parents=True, exist_ok=True)
        image = image.resize((args.size, args.size), Image.Resampling.BILINEAR)
        candidate = destination_dir / f"{sample.path.stem}_{sample_index:06d}.jpg"
        if candidate in seen_output or candidate.exists():
            candidate = destination_dir / f"{sample.path.stem}_{len(seen_output)}.jpg"
        seen_output.add(candidate)
        image.save(candidate, format="JPEG", quality=95)
        stats[split][sample.label] += 1
    report = {
        "seed": args.seed,
        "sequence_groups": len(group_to_split),
        "invalid_files": invalid,
        "splits": {
            split: {
                "helmet": stats[split]["helmet"],
                "no_helmet": stats[split]["no_helmet"],
                "total": sum(stats[split].values()),
            }
            for split, _ in SPLITS
        },
        "warning": (
            "Fewer than three independent sequences were found; val or test may be empty."
            if len(group_to_split) < 3 else None
        ),
    }
    (output / "dataset_stats.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

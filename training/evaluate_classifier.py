#!/usr/bin/env python3
"""Evaluate the classifier with accuracy, precision, recall, F1 and confusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from helmet_rider.helmet.tiny_model import build_tiny_helmet_net

CLASSES = ("helmet", "no_helmet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--checkpoint", default="models/training/best.pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to evaluate a classifier checkpoint.") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_tiny_helmet_net().to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()
    confusion = np.zeros((2, 2), dtype=np.int64)
    for label, name in enumerate(CLASSES):
        paths = sorted((Path(args.dataset) / args.split / name).glob("*"))
        for path in paths:
            if not path.is_file():
                continue
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = image.resize((args.size, args.size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.inference_mode():
                prediction = int(model(tensor).argmax(1).item())
            confusion[label, prediction] += 1
    total = int(confusion.sum())
    metrics = {}
    for index, name in enumerate(CLASSES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "support": int(confusion[index, :].sum()),
        }
    report = {
        "split": args.split,
        "accuracy": float(np.trace(confusion) / max(total, 1)),
        "precision_macro": float(np.mean([value["precision"] for value in metrics.values()])),
        "recall_macro": float(np.mean([value["recall"] for value in metrics.values()])),
        "f1_macro": float(np.mean([value["f1"] for value in metrics.values()])),
        "classes": metrics,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": list(CLASSES),
        "samples": total,
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

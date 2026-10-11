#!/usr/bin/env python3
"""Fit temperature and abstention thresholds on validation crops only."""

import argparse
import json
from pathlib import Path

import numpy as np


def probabilities(logits, temperature):
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def calibrate(logits, labels, minimum_negative_precision=0.97):
    labels = np.asarray(labels, dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Calibration requires both human-labeled classes.")
    temperatures = np.geomspace(0.25, 4.0, 121)
    losses = [
        float(
            -np.log(
                np.clip(probabilities(logits, t)[np.arange(len(labels)), labels], 1e-12, 1)
            ).mean()
        )
        for t in temperatures
    ]
    temperature = float(temperatures[int(np.argmin(losses))])
    probs = probabilities(logits, temperature)
    best = None
    for positive in np.arange(0.55, 0.96, 0.05):
        for negative in np.arange(0.55, 0.96, 0.05):
            predictions = np.full(len(labels), -1)
            predictions[probs[:, 0] >= positive] = 0
            predictions[probs[:, 1] >= negative] = 1
            class_metrics = []
            for label in (0, 1):
                tp = int(((predictions == label) & (labels == label)).sum())
                fp = int(((predictions == label) & (labels != label)).sum())
                fn = int(((predictions != label) & (labels == label)).sum())
                class_metrics.append(
                    {
                        "precision": tp / (tp + fp) if tp + fp else 0.0,
                        "recall": tp / (tp + fn),
                        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                        "support": int((labels == label).sum()),
                    }
                )
            if class_metrics[1]["precision"] < minimum_negative_precision:
                continue
            row = {
                "helmet_positive_threshold": round(float(positive), 2),
                "no_helmet_negative_threshold": round(float(negative), 2),
                "macro_f1": float(np.mean([m["f1"] for m in class_metrics])),
                "unknown_rate": float((predictions == -1).mean()),
                "classes": class_metrics,
            }
            if best is None or (row["macro_f1"], -row["unknown_rate"]) > (
                best["macro_f1"],
                -best["unknown_rate"],
            ):
                best = row
    raw = probabilities(logits, 1.0)
    return {
        "temperature": temperature,
        "nll_before": float(-np.log(raw[np.arange(len(labels)), labels].clip(1e-12)).mean()),
        "nll_after": min(losses),
        "threshold_selection": best,
        "minimum_negative_precision": minimum_negative_precision,
        "scope": "Crop-only validation. Final pipeline rules and new-camera accuracy are not calibrated by this score.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/helmet_head_edgevision.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/outputs/classifier-calibration"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--minimum-negative-precision", type=float, default=0.97)
    args = parser.parse_args()
    import torch
    from finetune_head_classifier import HeadCropDataset
    from torch.utils.data import DataLoader
    from torchvision import models

    torch.set_num_threads(4)
    dataset = HeadCropDataset(args.dataset, "val", training=False)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, 2)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("state_dict", state))
    model.to(args.device).eval()
    logits, labels = [], []
    with torch.inference_mode():
        for images, truth in DataLoader(dataset, batch_size=args.batch_size):
            logits.extend(model(images.to(args.device)).cpu().tolist())
            labels.extend(truth.tolist())
    result = calibrate(logits, labels, args.minimum_negative_precision)
    result.update(
        dataset=str(args.dataset.resolve()),
        checkpoint=str(args.checkpoint.resolve()),
        split="val",
        support=len(labels),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez(args.output / "validation_logits.npz", logits=logits, labels=labels)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

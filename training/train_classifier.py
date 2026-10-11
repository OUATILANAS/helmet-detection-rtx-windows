#!/usr/bin/env python3
"""Train the small helmet/no-helmet classifier on head crops."""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from helmet_rider.helmet.tiny_model import build_tiny_helmet_net

CLASSES = ("helmet", "no_helmet")


class HeadCropDataset:
    def __init__(self, root: Path, split: str, size: int, augment: bool, torch):
        self.samples = [
            (path, label)
            for label, class_name in enumerate(CLASSES)
            for path in sorted((root / split / class_name).glob("*"))
            if path.is_file()
        ]
        self.size = size
        self.augment = augment
        self.torch = torch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.25:
                width, height = image.size
                margin_x = max(1, int(width * 0.04))
                margin_y = max(1, int(height * 0.04))
                left = random.randint(0, margin_x)
                top = random.randint(0, margin_y)
                right = random.randint(width - margin_x, width)
                bottom = random.randint(height - margin_y, height)
                image = image.crop((left, top, right, bottom))
            if random.random() < 0.45:
                image = image.rotate(random.uniform(-7, 7), resample=Image.Resampling.BILINEAR)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.72, 1.28))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.78, 1.22))
            if random.random() < 0.10:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.8)))
            if random.random() < 0.12:
                compressed = io.BytesIO()
                image.save(compressed, format="JPEG", quality=random.randint(48, 88))
                compressed.seek(0)
                with Image.open(compressed) as image:
                    image = image.convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(pixels.copy()).permute(2, 0, 1)
        return tensor, label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="models/training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Install a development PyTorch build to train the classifier.") from exc
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device
    root, output = Path(args.dataset), Path(args.output)
    train_set = HeadCropDataset(root, "train", args.size, True, torch)
    val_set = HeadCropDataset(root, "val", args.size, False, torch)
    if not train_set.samples or not val_set.samples:
        raise SystemExit("Both train and val need images in helmet/ and no_helmet/.")
    output.mkdir(parents=True, exist_ok=True)
    counts = [sum(label == index for _, label in train_set.samples) for index in range(2)]
    if any(count == 0 for count in counts):
        raise SystemExit("Training data must contain both helmet and no_helmet crops.")
    validation_counts = [
        sum(label == index for _, label in val_set.samples) for index in range(2)
    ]
    if any(count == 0 for count in validation_counts):
        raise SystemExit("Validation data must contain both helmet and no_helmet crops.")
    class_weights = torch.tensor(
        [len(train_set) / max(2 * count, 1) for count in counts],
        dtype=torch.float32,
        device=device,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    model = build_tiny_helmet_net().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            seen += len(labels)
        model.eval()
        val_loss, val_seen, correct = 0.0, 0, 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * len(labels)
                correct += int((logits.argmax(1) == labels).sum().item())
                val_seen += len(labels)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss / max(seen, 1),
            "val_loss": val_loss / max(val_seen, 1),
            "val_accuracy": correct / max(val_seen, 1),
        }
        history.append(metrics)
        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "input_size": args.size,
            "classes": list(CLASSES),
            "seed": args.seed,
        }
        torch.save(checkpoint, output / "last.pt")
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            torch.save(checkpoint, output / "best.pt")
        print(
            f"epoch {epoch:03d}/{args.epochs}: train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} val_accuracy={metrics['val_accuracy']:.3f}"
        )
    (output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    training_config = vars(args) | {
        "device_used": device,
        "class_order": list(CLASSES),
        "class_counts": dict(zip(CLASSES, counts, strict=True)),
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "validation_class_counts": dict(zip(CLASSES, validation_counts, strict=True)),
    }
    (output / "training_config.json").write_text(
        json.dumps(training_config, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

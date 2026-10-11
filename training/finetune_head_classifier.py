#!/usr/bin/env python3
"""Fine-tune a traffic head-crop EfficientNet on the prepared EdgeVision split.

The external checkpoint must be a plain EfficientNet-B0 state dict or a safe
weights-only checkpoint containing ``state_dict``. Class order is helmet=0,
no_helmet=1, matching the source training project's alphabetical ImageFolder
order and the EdgeVision-prepared YOLO labels.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class HeadCropDataset(Dataset):
    def __init__(self, root: Path, split: str, training: bool) -> None:
        self.root = root
        self.training = training
        self.rows: list[tuple[Path, int, tuple[float, float, float, float]]] = []
        # Accept both the EdgeVision layout (images/train, labels/train) and
        # the common split-local YOLO layout (train/images, train/labels).
        split_names = ("val", "valid", "validation") if split == "val" else (split,)
        layouts = [
            (root / "labels" / name, root / "images" / name)
            for name in split_names
        ] + [
            (root / name / "labels", root / name / "images")
            for name in split_names
        ]
        label_dir, image_dir = next(
            ((labels, images) for labels, images in layouts
             if labels.is_dir() and images.is_dir()),
            layouts[0],
        )
        manifest_path = root / "runtime_manifest.json"
        if not manifest_path.is_file() and (not label_dir.is_dir() or not image_dir.is_dir()):
            raise FileNotFoundError(
                f"Could not find {split!r} image/label folders under {root}; "
                f"checked {[str(path) for pair in layouts for path in pair]}"
            )
        for label_path in sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []:
            image_path = next((p for p in sorted(image_dir.glob(f"{label_path.stem}.*"))
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}), image_dir / f"{label_path.stem}.jpg")
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            image_height, image_width = image.shape[:2]
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5:
                    continue
                class_id, cx, cy, width, height = map(float, fields)
                left = (cx - width / 2) * image_width
                right = (cx + width / 2) * image_width
                top = (cy - height / 2) * image_height
                bottom = (cy + height / 2) * image_height
                if (
                    right <= left
                    or bottom <= top
                    or right <= 0
                    or left >= image_width
                    or bottom <= 0
                    or top >= image_height
                ):
                    continue
                self.rows.append((image_path, int(class_id), (cx, cy, width, height)))

        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            self.rows = []
            for row in manifest["rows"]:
                if row["split"] != split:
                    continue
                image_path = root / row["path"]
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Cannot read runtime image {image_path}")
                height, width = image.shape[:2]
                x1,y1,x2,y2 = row["bbox"]
                hx1,hy1,hx2,hy2 = row.get("reviewed_bbox", row["bbox"])
                head_area = max(1., (hx2-hx1)*(hy2-hy1))
                coverage = max(0.,min(x2,hx2)-max(x1,hx1))*max(0.,min(y2,hy2)-max(y1,hy1))/head_area
                if coverage < .5 or min(x2-x1,y2-y1) < 8:
                    continue
                box = ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)
                self.rows.append((image_path, ("helmet", "no_helmet").index(row["label"]), box))
        if not self.rows:
            raise ValueError(f"No labeled crops in {split} under {root}")
        augments = [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10, fill=0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            transforms.RandomApply([self._degrade], p=0.35),
        ] if training else []
        self.transform = transforms.Compose([
            self._pad_resize,
            *augments,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _degrade(image):
        # Reproduce low-resolution/JPEG camera appearance without changing labels.
        import io
        edge = random.choice((24, 32, 48, 64, 96, 128))
        small = image.resize((edge, edge), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        small.save(buffer, format="JPEG", quality=random.randint(55, 95))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").resize(image.size, Image.Resampling.BILINEAR)

    @staticmethod
    def _pad_resize(image: Image.Image, size: int = 224) -> Image.Image:
        width, height = image.size
        scale = size / max(width, height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        resized = cv2.resize(np.asarray(image), (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        left = (size - new_width) // 2
        top = (size - new_height) // 2
        padded = np.zeros((size,size,3),dtype=np.uint8)
        padded[top:top+new_height,left:left+new_width] = resized
        return Image.fromarray(padded)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label, box = self.rows[index]
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Could not decode image: {image_path}")
        image_height, image_width = bgr.shape[:2]
        cx, cy, box_width, box_height = box
        scale = random.uniform(1.0, 1.35) if self.training else 1.0
        half_width = box_width * image_width * scale / 2
        half_height = box_height * image_height * scale / 2
        if self.training:
            cx += random.uniform(-.08,.08)*box_width
            cy += random.uniform(-.08,.08)*box_height
        x1 = max(0, min(image_width, int(cx * image_width - half_width)))
        x2 = max(0, min(image_width, int(cx * image_width + half_width)))
        y1 = max(0, min(image_height, int(cy * image_height - half_height)))
        y2 = max(0, min(image_height, int(cy * image_height + half_height)))
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Empty head crop for {image_path}")
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(rgb)), label


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    confusion = [[0, 0], [0, 0]]
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            for truth, pred in zip(labels.tolist(), predictions.tolist(), strict=True):
                confusion[truth][pred] += 1

    metrics = {}
    f1_values = []
    for class_id, name in enumerate(("helmet", "no_helmet")):
        tp = confusion[class_id][class_id]
        fn = sum(confusion[class_id]) - tp
        fp = sum(confusion[row][class_id] for row in range(2)) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics[name] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    metrics["accuracy"] = sum(confusion[i][i] for i in range(2)) / max(
        sum(map(sum, confusion)), 1
    )
    metrics["macro_f1"] = sum(f1_values) / 2
    metrics["confusion"] = confusion
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/edgevision/yolo"))
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/edgevision-training/head-classifier"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Train only the classification head; keep feature weights and BatchNorm fixed.")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0,
                        help="Train only the last N feature blocks plus the classifier.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Keep test images unopened during development.")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto",
        help="Training device. CPU avoids the Metal graph cache on low-disk systems.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was selected, but it is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was selected, but it is unavailable.")
    torch.set_num_threads(args.threads)

    train_data = HeadCropDataset(args.dataset, "train", training=True)
    val_data = HeadCropDataset(args.dataset, "val", training=False)
    test_data = None if args.skip_test else HeadCropDataset(args.dataset, "test", training=False)
    counts = Counter(label for _, label, _ in train_data.rows)
    if any(counts[c] == 0 for c in (0,1)):
        raise SystemExit("Both classes must be present in training.")
    weights = torch.tensor(
        [len(train_data) / (2 * counts[class_id]) for class_id in (0, 1)],
        dtype=torch.float32,
        device=device,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0) if test_data else None

    checkpoint = torch.load(args.initial, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 2)
    model.load_state_dict(state, strict=True)
    model.to(device)
    if args.freeze_backbone:
        model.features.requires_grad_(False)
    if args.unfreeze_last_blocks:
        if args.freeze_backbone or not 1 <= args.unfreeze_last_blocks <= len(model.features):
            parser.error("Choose either --freeze-backbone or a valid --unfreeze-last-blocks count")
        model.features.requires_grad_(False)
        model.features[-args.unfreeze_last_blocks:].requires_grad_(True)

    baseline_metrics = {
        "val": evaluate(model, val_loader, device),

    }
    print(json.dumps({"baseline": baseline_metrics}), flush=True)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "helmet_head_finetuned.pt"
    # Keep the starting model as the fallback. A candidate must beat the
    # validation baseline before it can replace the saved checkpoint.
    best_macro_f1 = baseline_metrics["val"]["macro_f1"]
    stale = 0
    history = []
    torch.save(model.cpu().state_dict(), checkpoint_path)
    model.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.features.eval()
        if args.unfreeze_last_blocks:
            model.features.eval()
            for block in model.features[-args.unfreeze_last_blocks:]:
                block.train()
        batch_losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite classifier loss; checkpoint rejected")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=10, error_if_nonfinite=True,
            )
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "val": val_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            stale = 0
            torch.save(model.cpu().state_dict(), checkpoint_path)
            model.to(device)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after epoch {epoch}.", flush=True)
                break

    best_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device) if test_loader else None
    result = {
        "source_checkpoint": str(args.initial),
        "dataset": str(args.dataset.resolve()),
        "runtime_manifest": str(args.dataset / "runtime_manifest.json") if (args.dataset / "runtime_manifest.json").is_file() else None,
        "training_settings": vars(args) | {"dataset": str(args.dataset), "initial": str(args.initial), "output": str(args.output)},
        "class_order": ["helmet", "no_helmet"],
        "class_counts_train": {"helmet": counts[0], "no_helmet": counts[1]},
        "baseline": baseline_metrics,
        "best_validation_macro_f1": best_macro_f1,
        "test": test_metrics,
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint_path), "test": test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()

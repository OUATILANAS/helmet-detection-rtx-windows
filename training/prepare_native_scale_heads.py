"""Prepare TRAIN-only physical-head augmentation at deployment-like scales.

Small reviewed context patches are pasted without enlargement. Broad reviewed
head chips supplement appearance diversity. Constructed images never evaluate F1.
"""

import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path("dataset/head-union-native-scale-20261011")
EXPORT = Path("dataset/all-head-training-review-20261011/tiles-all30-audited")
CROPS = Path("dataset/headwear-finetune-2026-10-10-clean")


def pixels(path):
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def main():
    rng = random.Random(113)
    output = ROOT / "prepared"
    if output.exists():
        raise ValueError("Prepared dataset exists; refusing overwrite")
    manifest = json.loads((EXPORT / "export.json").read_text())
    review = Path("dataset/all-head-training-review-20261011/reviewed-training-regions.json")
    if hashlib.sha256(review.read_bytes()).hexdigest() != manifest["source_manifest_sha256"]:
        raise ValueError("Stale reviewed training tiles")
    broad = [r for r in json.loads((ROOT / "BROAD-POSITIVE-REVIEW.json").read_text())["rows"]
             if r["physical_head_review"] == "confirmed_physical_head"]
    if any(r["split"] != "train" for r in broad):
        raise ValueError("Broad positives must be TRAIN only")
    reserved_paths = set()
    for name in ("strict-head-review-20261011/reviewed-manifest.json",
                 "video-gap-final-holdout-20261010/manifest.json",
                 "ridesafe-final-holdout-20261010/manifest.json",
                 "video-gap-broad-image-regression-20261010/manifest.json"):
        reserved_paths.update(r["image"] for r in json.loads((Path("dataset") / name).read_text())["rows"])
    reserved_paths.update(r["image"] for r in json.loads((ROOT / "IMAGE-VALIDATION-QUEUE.json").read_text())["rows"])
    training_paths = {t["source"] for t in manifest["tiles"]} | {r["source_image"] for r in broad}
    reserved_pixels = {pixels(Path(p)) for p in reserved_paths}
    training_hashes = {p: pixels(Path(p)) for p in training_paths}
    if reserved_pixels & set(training_hashes.values()):
        raise ValueError("Decoded-pixel train/reserved or new validation overlap")
    images = output / "images/train"
    labels = output / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    positives = [t for t in manifest["tiles"] if t["heads"]]
    negatives = sorted([t for t in manifest["tiles"] if not t["heads"]],
                       key=lambda t: hashlib.sha256(t["name"].encode()).hexdigest())[:59]
    records, head_edges = [], []

    def load_tile(tile):
        image = Image.open(EXPORT / "images/train" / (tile["name"] + ".jpg")).convert("RGB")
        values = [[float(v) for v in line.split()] for line in
                  (EXPORT / "labels/train" / (tile["name"] + ".txt")).read_text().splitlines()]
        return image, values

    def background(index):
        image, values = load_tile(negatives[index % len(negatives)])
        if values:
            raise ValueError("Nonempty background rejected")
        if min(image.size) < 384:
            raise ValueError("Native background smaller than composite")
        x, y = rng.randrange(image.width - 383), rng.randrange(image.height - 383)
        return image.crop((x, y, x + 384, y + 384))

    def save(name, image, values, sources, kind):
        if any(v[0] != 0 or min(v[1:]) < 0 or max(v[1:]) > 1 for v in values):
            raise ValueError("Invalid remapped target")
        image.save(images / (name + ".jpg"), quality=95)
        (labels / (name + ".txt")).write_text(
            "".join(" ".join(f"{v:.8f}" for v in row) + "\n" for row in values))
        head_edges.extend(min(row[3] * image.width, row[4] * image.height) * 640 / max(image.size)
                          for row in values)
        records.append({"name": name, "kind": kind, "sources": sources, "heads": len(values)})

    # Retain larger original-context crops without artificially huge tiny heads.
    for tile in positives:
        image, values = load_tile(tile)
        if min(image.size) >= 384:
            save("context-" + tile["name"], image, values, [tile["source"]], "native_context")
    small = []
    for tile in positives:
        image, values = load_tile(tile)
        if max(image.size) <= 192:
            small.append((tile, image, values))
    rng.shuffle(small)
    for start in range(0, len(small), 4):
        canvas, remapped, sources = background(start), [], []
        for quadrant, (tile, patch, values) in enumerate(small[start:start + 4]):
            x = (quadrant % 2) * 192 + rng.randrange(193 - patch.width)
            y = (quadrant // 2) * 192 + rng.randrange(193 - patch.height)
            canvas.paste(patch, (x, y))
            remapped.extend([0, (v[1] * patch.width + x) / 384, (v[2] * patch.height + y) / 384,
                             v[3] * patch.width / 384, v[4] * patch.height / 384] for v in values)
            sources.append(tile["source"])
        save(f"native-small-{start:04d}", canvas, remapped, sources, "native_patch_composite")
    excluded_small = []
    for index, row in enumerate(broad):
        with Image.open(CROPS / row["path"]) as source:
            chip = source.convert("RGB")
        if min(chip.size) < 12:
            excluded_small.append(row["review_id"])
            continue
        edge = (16, 24, 32, 48, 64)[index % 5]
        scale = edge / min(chip.size)
        width, height = round(chip.width * scale), round(chip.height * scale)
        if max(width, height) > 192:
            raise ValueError("Reviewed chip aspect ratio incompatible with bounded augmentation")
        chip = chip.resize((width, height), Image.Resampling.LANCZOS)
        canvas = background(index)
        x, y = rng.randrange(385 - width), rng.randrange(385 - height)
        canvas.paste(chip, (x, y))
        save(f"broad-{row['review_id']:03d}", canvas,
             [[0, (x + width / 2) / 384, (y + height / 2) / 384, width / 384, height / 384]],
             [row["source_image"]], "reviewed_broad_chip")
    positive_count = len(records)
    for index in range(positive_count // 4):
        save(f"background-{index:03d}", background(index), [],
             [negatives[index % len(negatives)]["source"]], "reviewed_empty")
    histogram = {"under16": 0, "16to32": 0, "32to64": 0, "64plus": 0}
    for edge in head_edges:
        histogram["under16" if edge < 16 else "16to32" if edge < 32 else
                  "32to64" if edge < 64 else "64plus"] += 1
    report = {"policy": __doc__, "training_imgsz": 640, "decoded_pixel_collisions": 0,
              "source_pixel_hashes": training_hashes, "positive_images": positive_count,
              "empty_images": len(records) - positive_count, "head_instances": len(head_edges),
              "head_min_edge_after_resize": histogram, "excluded_tiny_broad_ids": excluded_small,
              "records": records, "validation_queue_sha256": hashlib.sha256(
                  (ROOT / "IMAGE-VALIDATION-QUEUE.json").read_bytes()).hexdigest(),
              "status": "Prepared augmentation; visual inspection required before training"}
    (output / "PREPARATION.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "train.txt").write_text("".join(str((images / (r["name"] + ".jpg")).resolve()) + "\n"
                                              for r in records))
    print(json.dumps({k: v for k, v in report.items() if k not in {"records", "source_pixel_hashes", "policy"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()

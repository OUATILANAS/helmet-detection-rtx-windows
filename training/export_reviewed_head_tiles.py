"""Export only audited TRAIN regions for class-agnostic physical-head learning."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def tile_labels(row, tile):
    """None means unsafe, [] means an explicitly reviewed empty tile."""
    if any(intersection(tile, box) > 0 for box in row.get("ignore_regions", [])):
        return None
    labels = []
    width, height = tile[2] - tile[0], tile[3] - tile[1]
    for sample in row["samples"]:
        box = sample["bbox"]
        overlap = intersection(tile, box)
        if overlap == 0:
            continue
        # Never silently turn a small fragment of a physical head into background.
        area = (box[2] - box[0]) * (box[3] - box[1])
        if overlap / area < 0.8:
            return None
        a, b = max(box[0], tile[0]), max(box[1], tile[1])
        c, d = min(box[2], tile[2]), min(box[3], tile[3])
        labels.append([0, ((a + c) / 2 - tile[0]) / width,
                       ((b + d) / 2 - tile[1]) / height,
                       (c - a) / width, (d - b) / height])
    return labels


def preflight(data, project):
    train_groups = set(data["training_groups"])
    if train_groups & set(data["evaluation_groups"]):
        raise ValueError("Train and evaluation capture groups overlap")
    for row in data["rows"]:
        if row["split"] != "train" or row["capture_group"] not in train_groups:
            raise ValueError("Non-training image rejected")
        if row.get("training_review_status") != "usable_with_ignore_regions":
            raise ValueError("Pending review rejected")
        path = project / row["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError("Reviewed image changed")
        with Image.open(path) as image:
            if list(image.size) != row["image_size"]:
                raise ValueError("Image dimensions changed")
        width, height = row["image_size"]
        for sample in row["samples"]:
            if sample.get("physical_head") is not True:
                raise ValueError("Unverified physical-head label rejected")
        for box in [s["bbox"] for s in row["samples"]] + row.get("ignore_regions", []):
            a, b, c, d = box
            if not 0 <= a < c <= width or not 0 <= b < d <= height:
                raise ValueError("Invalid native image box")


def origins(length, edge):
    stop = max(0, length - edge)
    return sorted(set([*range(0, stop + 1, edge // 2), stop]))


def reviewed_tiles(row, edge):
    width, height = row["image_size"]
    tw, th = min(edge, width), min(edge, height)
    boxes = {(left, top, left + tw, top + th)
             for top in origins(height, th) for left in origins(width, tw)}
    # Smaller head-centred crops preserve audited actors near ignored vehicles.
    # They never override exclusions; keep uniform scene tiles for background.
    for sample in row["samples"]:
        a, b, c, d = sample["bbox"]
        sizes = sorted({edge // divisor for divisor in (1, 2, 4, 8, 16)
                        if edge // divisor >= 32})
        found = False
        for size in sizes:
            if size < max(c - a, d - b) * 2:
                continue
            cw, ch = min(size, width), min(size, height)
            # Try centring first, then place the actor near an edge to avoid
            # unresolved neighbours without removing them from the source frame.
            for x in ((a + c - cw) / 2, a - 2, c + 2 - cw):
                for y in ((b + d - ch) / 2, b - 2, d + 2 - ch):
                    left = max(0, min(width - cw, int(x)))
                    top = max(0, min(height - ch, int(y)))
                    box = (left, top, left + cw, top + ch)
                    labels = tile_labels(row, box)
                    if labels:
                        boxes.add(box)
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return sorted(boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--edge", type=int, default=768)
    args = parser.parse_args()
    if args.edge < 32:
        raise ValueError("Tile edge must be at least 32 pixels")
    project = Path.cwd()
    data = json.loads(args.manifest.read_text())
    preflight(data, project)  # Finish all source checks before creating output.
    if args.output.exists():
        raise ValueError("Output exists; refusing to overwrite an earlier dataset")
    image_root = args.output / "images/train"
    label_root = args.output / "labels/train"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    exported, skipped = [], 0
    for row in data["rows"]:
        with Image.open(project / row["image"]) as source:
            image = source.convert("RGB")
        for box in reviewed_tiles(row, args.edge):
            labels = tile_labels(row, box)
            if labels is None:
                skipped += 1
                continue
            left, top, right, bottom = box
            name = f"review-{row['review_id']:02d}-{left}-{top}-{right-left}x{bottom-top}"
            image.crop(box).save(image_root / f"{name}.jpg", quality=95)
            (label_root / f"{name}.txt").write_text(
                "".join(" ".join(f"{v:.8f}" for v in label) + "\n" for label in labels)
            )
            exported.append({"name": name, "source": row["image"],
                             "native_tile": box, "heads": len(labels)})
    report = {"source_manifest": str(args.manifest),
              "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "class_names": ["physical_head"], "helmet_status_training": False,
              "tiles": exported, "skipped_unsafe_tiles": skipped,
              "positive_tiles": sum(x["heads"] > 0 for x in exported),
              "empty_tiles": sum(x["heads"] == 0 for x in exported)}
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tiles"}, indent=2))


if __name__ == "__main__":
    main()

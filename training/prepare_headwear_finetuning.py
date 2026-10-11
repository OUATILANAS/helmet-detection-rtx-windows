#!/usr/bin/env python3
"""Combine traffic heads and publisher-labeled worn hats for candidate fine-tuning.

The new labels remain publisher annotations with automated geometric filtering,
not human-reviewed ground truth. All source/crop geometry and filtering reasons
are saved. Mixed bicycle helmets, balaclavas, rider boxes and invalid labels are
excluded. Existing traffic evaluation splits remain fixed.
"""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dataset/headwear-finetune-2026-10-10"
HATS = {"Baseball cap", "Beanie", "Bucket Hat", "cap"}


def phash(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32)).astype(np.float32)
    hashes = []
    for flip in (small, np.fliplr(small)):
        for rotation in range(4):
            coeff = cv2.dct(np.ascontiguousarray(np.rot90(flip, rotation)))[:8, :8].ravel()
            median = np.median(coeff[1:])
            hashes.append(int("".join("1" if c > median else "0" for c in coeff), 2))
    return min(hashes)


def original_key(name):
    return re.split(r"\.rf\.", Path(name).stem)[0]


def worn_head_box(box, persons, width, height):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1
    candidates = []
    for px1, py1, px2, py2, confidence in persons:
        pw, ph = px2-px1, py2-py1
        overlap = max(0, min(x2, px2)-max(x1, px1))*max(0, min(y2, py2+ph*.35)-max(y1, py1))
        coverage = overlap/max(bw*bh, 1)
        if (coverage >= .65 and px1 <= cx <= px2 and py1 <= cy <= py1+ph*.30
                and bw <= pw*.9 and bh <= ph*.45):
            bottom = min(height, py1+ph*.45, y1+max(bh*2.5, bw*1.3))
            head = [max(0, x1-bw*.08), max(0, y1-bh*.08),
                    min(width, x2+bw*.08), bottom]
            if head[3] >= y2 and min(head[2]-head[0], head[3]-head[1]) >= 16:
                candidates.append((coverage*confidence, head))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    rows, base_hashes = [], []
    image_links = {}

    def add_image(path, prefix):
        path = path.resolve()
        if path not in image_links:
            target = OUTPUT / "images" / f"{prefix}-{path.name}"
            target.parent.mkdir(exist_ok=True)
            target.symlink_to(path)
            image_links[path] = str(target.relative_to(OUTPUT))
        return image_links[path]

    # Copy the established split's source annotation geometry exactly.
    edge = ROOT / "dataset/edgevision/yolo"
    for split in ("train", "val", "test"):
        for label in sorted((edge / "labels" / split).glob("*.txt")):
            path = next((edge / "images" / split).glob(label.stem+".*"))
            frame = cv2.imread(str(path))
            height, width = frame.shape[:2]
            base_hashes.append(phash(frame))
            for line in label.read_text().splitlines():
                cls, cx, cy, bw, bh = map(float, line.split())
                if cls not in (0, 1):
                    raise ValueError("Prepared EdgeVision must use binary class IDs")
                box = [(cx-bw/2)*width, (cy-bh/2)*height,
                       (cx+bw/2)*width, (cy+bh/2)*height]
                if min(box[2]-box[0], box[3]-box[1]) < 8:
                    continue  # match the runtime crop loader's minimum size
                rows.append({"path": add_image(path, "edgevision"), "bbox": box,
                             "reviewed_bbox": box, "label": ("helmet", "no_helmet")[int(cls)],
                             "split": split, "sequence": f"edgevision:{path.stem}",
                             "source": "edgevision", "label_source": "publisher head annotations"})
    print(json.dumps({"traffic_rows": len(rows)}), flush=True)

    detector = YOLO(str(ROOT / "models/yolo26n.pt"))
    rejection = Counter()
    new_items = []
    for source in ("hat-2405", "baseball-cap", "cs173"):
        groups = defaultdict(list)
        for path in sorted((ROOT / f"dataset/headwear-review/{source}/metadata/all").glob("*.json")):
            record = json.loads(path.read_text())
            if source == "cs173" and record["source_split"] == "train":
                rejection["cs173_distorted_training_export"] += 1
                continue
            groups[original_key(record["original_filename"])].append(record)
        selected = []
        for _key, copies in sorted(groups.items()):
            # Evaluation export examples have no train-time synthetic variants.
            copies.sort(key=lambda r: (r["source_split"] == "train", r["original_filename"]))
            record = copies[0]
            eligible = [s for s in record["samples"] if
                        (source != "cs173" and s["source_class"] in HATS) or
                        (source == "cs173" and s["source_class"] in
                         {"full-faced", "half-faced", "not wearing helmet"})]
            if not eligible:
                rejection["excluded_classes"] += 1
                continue
            path = ROOT / f"dataset/headwear-review/{source}/images/all/{record['id']}{record['extension']}"
            selected.append((record, eligible, path))
        for index, (record, eligible, path) in enumerate(selected):
            frame = cv2.imread(str(path))
            height, width = frame.shape[:2]
            frame_hash = phash(frame)
            if any((frame_hash ^ value).bit_count() <= 4 for value in base_hashes):
                rejection["existing_traffic_duplicate"] += 1
                continue
            persons = []
            if source != "cs173":
                pred = detector.predict(frame, classes=[0], conf=.35, imgsz=640,
                                        device="mps", verbose=False)[0]
                persons = np.column_stack((pred.boxes.xyxy.cpu().numpy(),
                                           pred.boxes.conf.cpu().numpy())).tolist()
            samples = []
            for sample in eligible:
                box = sample["bbox"]
                if source != "cs173":
                    head = worn_head_box(box, persons, width, height)
                    label = "no_helmet"
                    if head is None:
                        rejection["no_reliable_worn_hat_head"] += 1
                        continue
                else:
                    head = box
                    label = "no_helmet" if sample["source_class"] == "not wearing helmet" else "helmet"
                if min(head[2]-head[0], head[3]-head[1]) < 16:
                    rejection["head_too_small"] += 1
                    continue
                samples.append({"path": add_image(path, source), "bbox": head,
                                "reviewed_bbox": head, "source_object_bbox": box,
                                "label": label, "source_class": sample["source_class"],
                                "source": source, "source_url": record["source_url"],
                                "source_split": record["source_split"],
                                "label_source": "publisher label with geometric filtering; not human reviewed"})
            if samples:
                # Group all trims from one source video part together.
                sequence = record["sequence"]
                if source == "cs173":
                    sequence = "cs173:" + original_key(record["original_filename"]).split("_")[0]
                new_items.append({"sequence": sequence, "hash": frame_hash, "rows": samples})
            if index % 100 == 0:
                print(json.dumps({"source": source, "processed": index, "total": len(selected),
                                  "accepted_images": len(new_items)}), flush=True)

    # Merge near-duplicate image groups BEFORE assigning splits. Filename
    # grouping alone cannot prevent the same public image crossing sources.
    parents = list(range(len(new_items)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    sequences = {}
    for i, item in enumerate(new_items):
        if item["sequence"] in sequences:
            parents[find(i)] = find(sequences[item["sequence"]])
        sequences[item["sequence"]] = i
        for j in range(i):
            if (item["hash"] ^ new_items[j]["hash"]).bit_count() <= 4:
                parents[find(i)] = find(j)
    seen_hashes = []
    for i, item in enumerate(new_items):
        group_names = sorted({new_items[j]["sequence"] for j in range(len(new_items))
                              if find(j) == find(i)})
        sequence = "|".join(group_names)
        bucket = int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 70 else "val" if bucket < 85 else "test"
        if any((item["hash"] ^ value).bit_count() <= 4 for value in seen_hashes):
            rejection["new_near_duplicate_image"] += 1
            continue
        seen_hashes.append(item["hash"])
        rows.extend({**r, "sequence": sequence, "split": split} for r in item["rows"])
    counts = Counter((r["split"], r["source"], r["label"]) for r in rows)
    manifest = {"version": 1, "class_order": ["helmet", "no_helmet"], "rows": rows,
                "counts": {" / ".join(k): v for k, v in sorted(counts.items())},
                "rejections": dict(rejection), "phash_distance": 4,
                "scope": "Publisher labels and automated head geometry; no claim of full human review"}
    (OUTPUT / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

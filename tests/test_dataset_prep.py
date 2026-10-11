import csv
import json
import sys
from pathlib import Path

from PIL import Image

from training.prepare_dataset import Sample, main, split_groups


def test_frames_from_same_sequence_stay_in_one_split():
    samples = [
        Sample(Path("a-1.jpg"), "helmet", "video-a"),
        Sample(Path("a-2.jpg"), "no_helmet", "video-a"),
        Sample(Path("b-1.jpg"), "helmet", "video-b"),
        Sample(Path("b-2.jpg"), "no_helmet", "video-b"),
        Sample(Path("c-1.jpg"), "helmet", "video-c"),
        Sample(Path("c-2.jpg"), "no_helmet", "video-c"),
        Sample(Path("d-1.jpg"), "helmet", "video-d"),
        Sample(Path("d-2.jpg"), "no_helmet", "video-d"),
    ]
    assigned = split_groups(samples, seed=17)
    assert len(assigned) == 4
    assert len(set(assigned.values())) >= 2


def test_manifest_can_extract_and_resize_a_head_roi(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (100, 120), (80, 120, 160)).save(source / "street.jpg")
    manifest = tmp_path / "annotations.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("path", "label", "sequence", "x1", "y1", "x2", "y2")
        )
        writer.writeheader()
        writer.writerow({
            "path": "street.jpg",
            "label": "helmet",
            "sequence": "video-1",
            "x1": 20,
            "y1": 5,
            "x2": 80,
            "y2": 105,
        })
    destination = tmp_path / "dataset"
    monkeypatch.setattr(sys, "argv", [
        "prepare_dataset.py", "--source", str(source),
        "--annotations", str(manifest), "--output", str(destination), "--clean",
        "--size", "64",
    ])
    main()
    crops = list(destination.glob("*/helmet/*.jpg"))
    assert len(crops) == 1
    with Image.open(crops[0]) as crop:
        assert crop.size == (64, 64)
    report = json.loads((destination / "dataset_stats.json").read_text())
    assert report["splits"]["train"]["helmet"] == 1

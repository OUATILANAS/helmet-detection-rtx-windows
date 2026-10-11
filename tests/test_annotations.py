import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import api
from app.annotations import (
    helmet_yolo_label_text,
    split_records_by_sequence,
    validate_boxes,
    validate_helmet_samples,
    yolo_label_text,
)


def test_validate_boxes_clamps_to_image_and_rejects_bad_values():
    assert validate_boxes([[-4, 5, 25, 40]], 20, 30) == [[0.0, 5.0, 20.0, 30.0]]
    with pytest.raises(ValueError, match="finite"):
        validate_boxes([[0, 0, float("nan"), 8]], 20, 30)
    with pytest.raises(ValueError, match="at least 2 pixels"):
        validate_boxes([[4, 4, 5, 12]], 20, 30)
    with pytest.raises(ValueError, match="numbers"):
        validate_boxes([[True, 0, 10, 10]], 20, 30)


def test_yolo_labels_use_normalized_center_xywh():
    labels = yolo_label_text([[10, 5, 30, 25], [0, 0, 100, 50]], 100, 50)
    assert labels == (
        "0 0.200000 0.300000 0.200000 0.400000\n"
        "0 0.500000 0.500000 1.000000 1.000000\n"
    )
    assert yolo_label_text([], 100, 50) == ""


def test_helmet_review_labels_keep_uncertain_out_of_training_labels():
    samples = validate_helmet_samples([
        {"bbox": [10, 5, 30, 25], "label": "helmet", "track_id": 3,
         "model_confidence": 0.91},
        {"bbox": [40, 5, 60, 25], "label": "no_helmet", "track_id": 4},
        {"bbox": [70, 5, 90, 25], "label": "unknown", "track_id": 5},
    ], 100, 50)
    assert helmet_yolo_label_text(samples, 100, 50) == (
        "0 0.200000 0.300000 0.200000 0.400000\n"
        "1 0.500000 0.300000 0.200000 0.400000\n"
    )


def test_helmet_review_samples_reject_invalid_labels_and_confidences():
    with pytest.raises(ValueError, match="Helmet labels"):
        validate_helmet_samples([{"bbox": [0, 0, 10, 10], "label": "maybe"}], 20, 20)
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_helmet_samples([
            {"bbox": [0, 0, 10, 10], "label": "helmet", "model_confidence": 1.5}
        ], 20, 20)


def test_capture_groups_do_not_leak_between_splits():
    records = [
        {"id": "a1", "sequence": "video-a"},
        {"id": "a2", "sequence": "video-a"},
        {"id": "b1", "sequence": "video-b"},
        {"id": "c1", "sequence": "video-c"},
    ]
    splits = split_records_by_sequence(records)
    assert splits == split_records_by_sequence(records)
    group_split = {
        record["sequence"]: split_name
        for split_name, split_records in splits.items()
        for record in split_records
    }
    assert group_split["video-a"] in {"train", "val", "test"}
    assert group_split["video-b"] != group_split["video-a"]
    assert group_split["video-c"] not in {group_split["video-a"], group_split["video-b"]}


def test_dataset_export_saves_images_labels_and_split_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ANNOTATION_DIR", tmp_path / "review")
    client = TestClient(api.app)
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", frame)
    assert success
    image_bytes = encoded.tobytes()

    first = client.post(
        "/api/annotations",
        files={"file": ("street-a.jpg", image_bytes, "image/jpeg")},
        data={"annotations": json.dumps([[10, 5, 30, 25]]), "sequence": "street-a"},
    )
    second = client.post(
        "/api/annotations",
        files={"file": ("street-b.jpg", image_bytes, "image/jpeg")},
        data={"annotations": "[]", "sequence": "street-b"},
    )
    assert first.status_code == second.status_code == 200
    assert first.json()["boxes"] == 1
    assert second.json()["export_ready"] is True
    assert client.get("/api/annotations/stats").json() == {
        "images": 2, "boxes": 1, "sequences": 2, "export_ready": True
    }

    exported = client.get("/api/annotations/export")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        names = archive.namelist()
        assert "dataset.yaml" in names
        assert "split_manifest.json" in names
        assert len([name for name in names if name.startswith("images/")]) == 2
        label_paths = [name for name in names if name.startswith("labels/")]
        assert len(label_paths) == 2
        assert any(archive.read(path).decode().strip() == "" for path in label_paths)
        motorcycle_labels = [archive.read(path).decode() for path in label_paths]
        assert any(label == "0 0.200000 0.300000 0.200000 0.400000\n" for label in motorcycle_labels)
        manifest = json.loads(archive.read("split_manifest.json"))
        assert all(len(records) == 1 for records in manifest.values() if records)


def test_helmet_review_endpoint_exports_group_split_training_data(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "HELMET_REVIEW_DIR", tmp_path / "helmet-review")
    client = TestClient(api.app)
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", frame)
    assert success
    image_bytes = encoded.tobytes()
    sample_rows = [
        {"bbox": [10, 5, 30, 25], "label": "helmet", "track_id": 1,
         "model_confidence": 0.8},
        {"bbox": [40, 5, 60, 25], "label": "no_helmet", "track_id": 2,
         "model_confidence": 0.7},
        {"bbox": [70, 5, 90, 25], "label": "unknown", "track_id": 3,
         "model_confidence": 0.5},
    ]
    for index, group in enumerate(("camera-a", "camera-b", "camera-c")):
        frame[:] = 30 * index
        _, encoded = cv2.imencode(".jpg", frame)
        image_bytes = encoded.tobytes()
        saved = client.post(
            "/api/helmet-review",
            files={"file": (f"{group}.jpg", image_bytes, "image/jpeg")},
            data={"samples": json.dumps(sample_rows), "sequence": group},
        )
        assert saved.status_code == 200
    assert saved.json()["export_ready"] is True
    assert client.get("/api/helmet-review/stats").json() == {
        "images": 3, "sequences": 3, "helmet": 3, "no_helmet": 3,
        "unknown": 3, "export_ready": True,
    }

    exported = client.get("/api/helmet-review/export")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        manifest = json.loads(archive.read("split_manifest.json"))
        assert set(manifest["frames"]) == {"train", "val", "test"}
        assert all(len(values) == 1 for values in manifest["frames"].values())
        assert "names:\n  0: helmet\n  1: no_helmet" in archive.read("dataset.yaml").decode()
        labels = [archive.read(name).decode() for name in archive.namelist() if name.startswith("labels/")]
        assert len(labels) == 3
        assert all(len(label.strip().splitlines()) == 2 for label in labels)
        annotations = [
            json.loads(archive.read(name))
            for name in archive.namelist() if name.startswith("annotations/")
        ]
        assert all(any(sample["label"] == "unknown" for sample in item["samples"]) for item in annotations)


def test_export_requires_two_capture_groups():
    with pytest.raises(ValueError, match="two different capture groups"):
        split_records_by_sequence([{"id": "a", "sequence": "one"}])


def test_duplicate_image_conflicts_block_export(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'HELMET_REVIEW_DIR', tmp_path/'reviews')
    client=TestClient(api.app)
    _,encoded=cv2.imencode('.png',np.zeros((50,100,3),dtype=np.uint8))
    for label in ('helmet','no_helmet'):
        response=client.post('/api/helmet-review',files={'file':('same.png',encoded.tobytes(),'image/png')},
          data={'sequence':'one','samples':json.dumps([{'bbox':[10,5,30,25],'label':label}])})
        assert response.status_code==200
    exported=client.get('/api/helmet-review/export')
    assert exported.status_code==409
    assert 'Conflicting duplicate' in exported.json()['detail']


def test_review_keeps_runtime_crop_and_human_temporal_reference():
    rows=validate_helmet_samples([{'bbox':[20,10,40,30],'runtime_bbox':[15,5,45,35],
      'vehicle_bbox':[0,30,60,50],'label':'helmet','entity_id':'rider-a'}],100,100)
    assert rows[0]['runtime_bbox']==[15.,5.,45.,35.]
    assert rows[0]['entity_id']=='rider-a'

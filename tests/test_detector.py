import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ultralytics

from helmet_rider.detection import detector as detector_module
from helmet_rider.detection.detector import UltralyticsDetector
from helmet_rider.detection.types import Detection


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBox:
    def __init__(self, class_id, confidence, coordinates):
        self.cls = FakeScalar(class_id)
        self.conf = FakeScalar(confidence)
        self.xyxy = [SimpleNamespace(to_list=lambda: coordinates,
                                     tolist=lambda: coordinates)]


class FakeYOLO:
    def __init__(self, path, calls):
        self.path = path
        self.calls = calls
        self.names = (
            {0: "motorbike"}
            if Path(path).name == "trafficai_motorbike.pt"
            else (
                {9: "motor"}
                if Path(path).name == "visdrone.pt"
                else (
                    {0: "With helmet", 1: "  Without helmet"}
                    if Path(path).name == "safety_helmet.pt"
                    else {0: "person", 1: "bicycle", 3: "motorcycle"}
                )
            )
        )

    def predict(self, **kwargs):
        self.calls.append((self.path, kwargs))
        if Path(self.path).name == "trafficai_motorbike.pt":
            boxes = [FakeBox(0, 0.91, [30, 40, 100, 120])]
        elif Path(self.path).name == "visdrone.pt":
            boxes = [
                FakeBox(9, 0.90, [35, 50, 70, 90]),
                FakeBox(9, 0.60, [120, 50, 180, 120]),
            ]
        elif Path(self.path).name == "safety_helmet.pt":
            boxes = [
                FakeBox(0, 0.62, [20, 30, 40, 50]),
                FakeBox(1, 0.31, [60, 30, 80, 50]),
            ]
        else:
            boxes = [
                FakeBox(0, 0.82, [10, 20, 55, 140]),
                FakeBox(1, 0.76, [120, 80, 180, 145]),
            ]
        source = kwargs.get("source")
        result_count = len(source) if isinstance(source, list) else 1
        return [SimpleNamespace(boxes=boxes) for _ in range(result_count)]


def test_traffic_head_model_maps_nohelmet_and_ignores_plate(tmp_path, monkeypatch):
    coco = tmp_path / "yolo26n.pt"
    helmet = tmp_path / "traffic_head.pt"
    coco.touch()
    helmet.touch()

    def load(path):
        model = FakeYOLO(path, [])
        if Path(path).name == helmet.name:
            model.names = {0: "helmet", 1: "nohelmet", 2: "licenseplate"}
        return model

    monkeypatch.setattr(ultralytics, "YOLO", load)
    detector = UltralyticsDetector(str(coco), helmet_model_path=str(helmet))
    assert detector.helmet_class_ids == {"with_helmet": 0, "without_helmet": 1}


def test_supplementary_physical_head_model_keeps_status_neutral(tmp_path, monkeypatch):
    from helmet_rider.detection.detector import UltralyticsHeadDetector

    checkpoint = tmp_path / "heads.pt"
    checkpoint.touch()

    def load(path):
        model = FakeYOLO(path, [])
        model.names = {0: "physical_head"}
        return model

    monkeypatch.setattr(ultralytics, "YOLO", load)
    detector = UltralyticsHeadDetector(str(checkpoint), tile_grid=1)
    detections = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))
    assert detector.helmet_class_ids == {"physical_head": 0}
    assert len(detections) == 1
    assert detections[0].class_name == "physical_head"


def test_auto_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    cuda_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", cuda_torch)
    assert detector_module._resolve_device("auto") == "0"

    mps_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", mps_torch)
    monkeypatch.setattr(detector_module.sys, "platform", "darwin")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    assert detector_module._resolve_device("auto") == "mps"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

    cpu_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", cpu_torch)
    assert detector_module._resolve_device("auto") == "cpu"


def test_auto_device_falls_back_to_cpu_when_accelerators_are_unavailable(monkeypatch):
    cpu_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", cpu_torch)
    monkeypatch.setattr(detector_module.sys, "platform", "linux")
    assert detector_module._resolve_device("auto") == "cpu"


def test_specialist_motorbike_model_replaces_coco_motorcycle_and_merges_people(
    tmp_path, monkeypatch
):
    coco_path = tmp_path / "yolo26n.pt"
    motorbike_path = tmp_path / "trafficai_motorbike.pt"
    helmet_path = tmp_path / "safety_helmet.pt"
    coco_path.touch()
    motorbike_path.touch()
    helmet_path.touch()
    calls = []
    monkeypatch.setattr(
        ultralytics,
        "YOLO",
        lambda path: FakeYOLO(str(path), calls),
    )
    detector = UltralyticsDetector(
        str(coco_path),
        confidence=0.08,
        image_size=1280,
        augment=True,
        nms_iou=0.55,
        motorcycle_model_path=str(motorbike_path),
        motorcycle_confidence=0.04,
        motorcycle_nms_iou=0.45,
        helmet_model_path=str(helmet_path),
        helmet_confidence=0.02,
        helmet_nms_iou=0.45,
    )

    detections = detector.detect(np.zeros((160, 200, 3), dtype=np.uint8))

    assert [item.class_name for item in detections] == [
        "person", "bicycle", "motorcycle", "with_helmet", "without_helmet"
    ]
    assert calls[0][1]["classes"] == [0, 1, 3]
    assert calls[1][1]["classes"] == [0]
    assert calls[0][1]["augment"] is True
    assert calls[0][1]["conf"] == 0.04
    assert calls[1][1]["conf"] == 0.04
    assert calls[1][1]["iou"] == 0.45
    assert calls[2][1]["classes"] == [0, 1]
    assert calls[2][1]["conf"] == 0.02


def test_coco_motorcycle_fallback_only_fills_specialist_gaps():
    specialist = Detection("motorcycle", 0.91, (0, 0, 100, 100))
    overlapping_coco = Detection("motorcycle", 0.82, (5, 5, 95, 95))
    missed_coco = Detection("motorcycle", 0.74, (120, 120, 190, 155))
    duplicate_coco = Detection("motorcycle", 0.67, (125, 122, 188, 154))
    weak_coco = Detection("motorcycle", 0.27, (210, 120, 280, 155))

    merged = UltralyticsDetector._merge_motorcycle_fallbacks(
        [overlapping_coco, missed_coco, duplicate_coco, weak_coco], [specialist]
    )

    assert merged == [specialist, missed_coco]


def test_helmet_nms_preserves_overlapping_opposite_class_evidence():
    detector = object.__new__(UltralyticsDetector)
    detector.helmet_nms_iou = 0.45
    detections = [
        Detection("without_helmet", 0.91, (20, 20, 70, 80)),
        Detection("with_helmet", 0.72, (21, 21, 69, 79)),
        Detection("without_helmet", 0.55, (22, 22, 68, 78)),
    ]

    kept = detector._helmet_nms(detections)

    assert [item.class_name for item in kept] == ["without_helmet", "with_helmet"]


def test_motorcycle_gap_model_adds_only_new_full_frame_boxes(tmp_path, monkeypatch):
    coco_path = tmp_path / "yolo26n.pt"
    motorbike_path = tmp_path / "trafficai_motorbike.pt"
    gap_path = tmp_path / "visdrone.pt"
    coco_path.touch()
    motorbike_path.touch()
    gap_path.touch()
    calls = []
    monkeypatch.setattr(
        ultralytics,
        "YOLO",
        lambda path: FakeYOLO(str(path), calls),
    )
    detector = UltralyticsDetector(
        str(coco_path),
        motorcycle_model_path=str(motorbike_path),
        motorcycle_gap_model_path=str(gap_path),
        motorcycle_gap_confidence=0.045,
        motorcycle_gap_min_area=800,
    )

    detections = detector.detect(np.zeros((160, 200, 3), dtype=np.uint8))
    motorcycles = [item for item in detections if item.class_name == "motorcycle"]

    assert len(motorcycles) == 2
    assert motorcycles[0].bbox == (30.0, 40.0, 100.0, 120.0)
    assert motorcycles[1].bbox == (120.0, 50.0, 180.0, 120.0)
    assert calls[-1][1]["source"].shape == (160, 200, 3)
    assert calls[-1][1]["classes"] == [9]
    assert calls[-1][1]["conf"] == 0.045
    assert calls[-1][1]["imgsz"] == 640


def test_motorcycle_gap_merge_rejects_contained_and_tiny_boxes():
    existing = [Detection("motorcycle", 0.9, (0, 0, 100, 100))]
    candidates = [
        Detection("motorcycle", 0.8, (20, 20, 50, 50)),
        Detection("motorcycle", 0.7, (120, 0, 140, 20)),
        Detection("motorcycle", 0.65, (70, 0, 130, 60)),
        Detection("motorcycle", 0.6, (150, 20, 200, 80)),
    ]

    merged = UltralyticsDetector._merge_motorcycle_gap_detections(
        existing, candidates, minimum_area=800
    )

    assert merged == [candidates[3]]


def test_motorcycle_nms_removes_nested_same_bike_boxes_but_keeps_adjacent_bike():
    large = Detection("motorcycle", 0.91, (0, 0, 100, 100))
    nested = Detection("motorcycle", 0.80, (20, 20, 50, 50))
    adjacent = Detection("motorcycle", 0.75, (85, 0, 185, 100))

    kept = UltralyticsDetector._deduplicate_motorcycles(
        [large, nested, adjacent]
    )

    assert kept == [large, adjacent]


def test_helmet_tiling_maps_tile_boxes_to_original_frame(tmp_path, monkeypatch):
    coco_path = tmp_path / "yolo26n.pt"
    helmet_path = tmp_path / "safety_helmet.pt"
    coco_path.touch()
    helmet_path.touch()
    calls = []
    monkeypatch.setattr(
        ultralytics,
        "YOLO",
        lambda path: FakeYOLO(str(path), calls),
    )
    detector = UltralyticsDetector(
        str(coco_path),
        helmet_model_path=str(helmet_path),
        helmet_tile_grid=2,
        helmet_tile_overlap=0.20,
        helmet_tile_confidence=0.005,
    )

    detections = detector.detect(np.zeros((160, 200, 3), dtype=np.uint8))

    tile_call = calls[-1][1]
    assert len(tile_call["source"]) == 4
    assert tile_call["conf"] == 0.005
    assert tile_call["batch"] == 4
    assert any(
        item.class_name == "with_helmet"
        and item.bbox == (108.0, 30.0, 128.0, 50.0)
        for item in detections
    )


def test_tile_source_lists_are_bounded_and_last_chunk_keeps_global_coordinates(tmp_path, monkeypatch):
    coco_path = tmp_path / "yolo26n.pt"
    helmet_path = tmp_path / "safety_helmet.pt"
    coco_path.touch()
    helmet_path.touch()
    calls = []
    monkeypatch.setattr(ultralytics, "YOLO", lambda path: FakeYOLO(str(path), calls))
    detector = UltralyticsDetector(str(coco_path), helmet_model_path=str(helmet_path),
                                  helmet_tile_grid=3, helmet_tile_overlap=.2)
    detections = detector.detect(np.zeros((420, 600, 3), dtype=np.uint8))
    tile_calls = [kwargs for path, kwargs in calls
                  if Path(path).name == helmet_path.name and isinstance(kwargs["source"], list)]
    assert [len(kwargs["source"]) for kwargs in tile_calls] == [4, 4, 1]
    assert all(kwargs["batch"] == len(kwargs["source"]) for kwargs in tile_calls)
    assert any(d.class_name == "with_helmet" and d.bbox == (389., 288., 409., 308.)
               for d in detections)


def test_stronger_motorcycle_removes_only_matching_bicycle_box():
    motor = Detection("motorcycle", .60, (528, 170, 646, 395))
    duplicate_bike = Detection("bicycle", .23, (530, 166, 640, 396))
    true_bike = Detection("bicycle", .75, (10, 200, 100, 390))
    assert UltralyticsDetector._resolve_vehicle_class_conflicts(
        [motor, duplicate_bike, true_bike]
    ) == [motor, true_bike]
    weak_motor = Detection("motorcycle", .08, duplicate_bike.bbox)
    assert UltralyticsDetector._resolve_vehicle_class_conflicts(
        [weak_motor, duplicate_bike]
    ) == [weak_motor, duplicate_bike]


def test_weak_motorcycle_fallback_needs_confident_riding_contact():
    rear = Detection("motorcycle", .043, (234, 176.58, 282.15, 320.14))
    rider = Detection("person", .81, (173.87, 26.83, 291.07, 321.33))
    assert UltralyticsDetector._merge_motorcycle_fallbacks(
        [rear], [], persons=[rider]
    ) == [rear]
    assert UltralyticsDetector._merge_motorcycle_fallbacks(
        [rear], [], persons=[Detection("person", .20, rider.bbox)]
    ) == []
    assert UltralyticsDetector._merge_motorcycle_fallbacks(
        [rear], [], persons=[Detection("person", .90, (300, 20, 350, 310))]
    ) == []


def test_weak_motorcycle_recovery_is_disabled_when_scene_is_dense():
    rear = Detection("motorcycle", .043, (234, 176.58, 282.15, 320.14))
    rider = Detection("person", .81, (173.87, 26.83, 291.07, 321.33))
    foreground = Detection("motorcycle", .80, (400, 200, 500, 390))
    assert UltralyticsDetector._merge_motorcycle_fallbacks(
        [rear], [foreground], persons=[rider], maximum_supported_existing=1
    ) == [foreground]


def test_cross_class_resolution_preserves_stronger_bicycle_prediction():
    bicycle = Detection("bicycle", .90, (10, 20, 100, 150))
    motorcycle = Detection("motorcycle", .60, bicycle.bbox)
    assert UltralyticsDetector._resolve_vehicle_class_conflicts(
        [bicycle, motorcycle]
    ) == [bicycle, motorcycle]


def test_adaptive_head_tiles_preserve_detail_without_overtiling_small_images():
    from helmet_rider.detection.detector import UltralyticsHeadDetector
    detector = object.__new__(UltralyticsHeadDetector)
    detector.helmet_tile_grid = 8
    detector.adaptive_tiles = True
    detector.tile_size = 384
    assert detector._effective_helmet_tile_grid(np.zeros((1440, 2560, 3))) == 7
    assert detector._effective_helmet_tile_grid(np.zeros((416, 832, 3))) == 3
    detector.tile_size = 768
    assert detector._effective_helmet_tile_grid(np.zeros((1440, 2560, 3))) == 4
    assert detector._effective_helmet_tile_grid(np.zeros((404, 612, 3))) == 1
    assert detector._effective_helmet_tile_grid(np.zeros((190, 266, 3))) == 1
    detector.adaptive_tiles = False
    assert detector._effective_helmet_tile_grid(np.zeros((416, 832, 3))) == 8

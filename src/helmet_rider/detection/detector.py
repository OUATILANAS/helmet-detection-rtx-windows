"""Object detector protocol and Ultralytics-compatible YOLO backend."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from helmet_rider.config import DetectorConfig
from helmet_rider.detection.types import Detection

LOGGER = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Resolve ``auto`` to CUDA, Apple MPS, or CPU for the installed runtime."""
    requested = str(device).strip()
    if requested.lower() != "auto":
        if requested.lower() == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return requested

    # Set this before importing torch: torchvision NMS currently falls back to
    # CPU on MPS, while the model layers still run on Apple's GPU.
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
    except ImportError:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
    except (AttributeError, RuntimeError):
        pass
    return "cpu"


class ObjectDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class UltralyticsDetector:
    """Loads a local YOLO checkpoint or exported Ultralytics model directory."""

    COCO_CLASSES = {0: "person", 1: "bicycle", 3: "motorcycle"}

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.08,
        image_size: int = 1280,
        device: str = "cpu",
        threads: int = 4,
        augment: bool = True,
        nms_iou: float = 0.55,
        motorcycle_model_path: str | None = None,
        motorcycle_confidence: float | None = None,
        motorcycle_nms_iou: float = 0.45,
        helmet_model_path: str | None = None,
        helmet_confidence: float = 0.002,
        helmet_nms_iou: float = 0.45,
        helmet_tile_grid: int = 1,
        helmet_tile_overlap: float = 0.20,
        helmet_tile_confidence: float = 0.005,
        motorcycle_gap_model_path: str | None = None,
        motorcycle_gap_confidence: float = 0.045,
        motorcycle_gap_image_size: int = 640,
        motorcycle_gap_min_area: float = 800.0,
        motorcycle_gap_max_existing: int = 24,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Detector model not found: {path}. Place a YOLO26n .pt checkpoint "
                "or exported model directory there; inference does not download weights."
            )
        self.device = _resolve_device(device)
        try:
            import torch

            torch.set_num_threads(max(1, int(threads)))
        except ImportError:
            pass
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Install the detector runtime with pip install -r requirements.txt."
            ) from exc
        self.model_path = str(path)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.augment = bool(augment and path.suffix == ".pt")
        self.nms_iou = float(nms_iou)
        self.motorcycle_confidence = float(
            confidence if motorcycle_confidence is None else motorcycle_confidence
        )
        self.motorcycle_nms_iou = float(motorcycle_nms_iou)
        self.motorcycle_gap_confidence = float(motorcycle_gap_confidence)
        self.motorcycle_gap_image_size = int(motorcycle_gap_image_size)
        self.motorcycle_gap_min_area = float(motorcycle_gap_min_area)
        self.motorcycle_gap_max_existing = int(motorcycle_gap_max_existing)
        self.helmet_confidence = float(helmet_confidence)
        self.helmet_nms_iou = float(helmet_nms_iou)
        self.helmet_tile_grid = int(helmet_tile_grid)
        self.helmet_tile_overlap = float(helmet_tile_overlap)
        self.helmet_tile_confidence = float(helmet_tile_confidence)
        if self.helmet_tile_grid < 1:
            raise ValueError("helmet_tile_grid must be at least 1")
        if not 0.0 <= self.helmet_tile_overlap < 0.5:
            raise ValueError("helmet_tile_overlap must be between 0 and 0.5")
        if not 0.0 <= self.helmet_tile_confidence <= 1.0:
            raise ValueError("helmet_tile_confidence must be between 0 and 1")
        self.model = YOLO(self.model_path)
        self.motorcycle_model = None
        self.motorcycle_class_id: int | None = None
        if motorcycle_model_path:
            motorcycle_path = Path(motorcycle_model_path)
            if not motorcycle_path.exists():
                raise FileNotFoundError(
                    f"Motorcycle detector model not found: {motorcycle_path}. "
                    "Install the configured local traffic checkpoint."
                )
            self.motorcycle_model = YOLO(str(motorcycle_path))
            names = self.motorcycle_model.names
            names = names if isinstance(names, dict) else dict(enumerate(names))
            motorcycle_names = {str(name).strip().lower() for name in names.values()}
            if not motorcycle_names.intersection({"motorbike", "motorcycle"}):
                raise RuntimeError(
                    "The configured motorcycle model must contain a motorbike or "
                    "motorcycle class."
                )
            self.motorcycle_class_id = next(
                int(class_id)
                for class_id, name in names.items()
                if str(name).strip().lower() in {"motorbike", "motorcycle"}
            )
        self.motorcycle_gap_model = None
        self.motorcycle_gap_class_id: int | None = None
        if motorcycle_gap_model_path:
            gap_path = Path(motorcycle_gap_model_path)
            if not gap_path.exists():
                raise FileNotFoundError(
                    f"Motorcycle gap model not found: {gap_path}. "
                    "Install the configured local VisDrone checkpoint."
                )
            self.motorcycle_gap_model = YOLO(str(gap_path))
            names = self.motorcycle_gap_model.names
            names = names if isinstance(names, dict) else dict(enumerate(names))
            self.motorcycle_gap_class_id = next(
                (
                    int(class_id)
                    for class_id, name in names.items()
                    if str(name).strip().lower() in {"motor", "motorbike", "motorcycle"}
                ),
                None,
            )
            if self.motorcycle_gap_class_id is None:
                raise RuntimeError(
                    "The configured motorcycle gap model must contain a motor, "
                    "motorbike, or motorcycle class."
                )
        self.helmet_model = None
        self.helmet_class_ids: dict[str, int] = {}
        if helmet_model_path:
            helmet_path = Path(helmet_model_path)
            if not helmet_path.exists():
                raise FileNotFoundError(
                    f"Helmet detector model not found: {helmet_path}. "
                    "Install the configured local helmet checkpoint."
                )
            self.helmet_model = YOLO(str(helmet_path))
            names = self.helmet_model.names
            names = names if isinstance(names, dict) else dict(enumerate(names))
            for class_id, name in names.items():
                normalized = str(name).strip().lower().replace(" ", "_")
                normalized = normalized.replace("-", "_")
                if normalized in {"with_helmet", "helmet"}:
                    self.helmet_class_ids["with_helmet"] = int(class_id)
                elif normalized in {"without_helmet", "no_helmet", "nohelmet"}:
                    self.helmet_class_ids["without_helmet"] = int(class_id)
            if set(self.helmet_class_ids) != {"with_helmet", "without_helmet"}:
                raise RuntimeError(
                    "The configured helmet model must provide with-helmet and "
                    "without-helmet classes."
                )
        LOGGER.info("detector loaded: backend=ultralytics model=%s", self.model_path)
        if motorcycle_model_path:
            LOGGER.info("motorcycle detector loaded: model=%s", motorcycle_model_path)
        if motorcycle_gap_model_path:
            LOGGER.info("motorcycle gap detector loaded: model=%s", motorcycle_gap_model_path)
        if helmet_model_path:
            LOGGER.info("helmet detector loaded: model=%s", helmet_model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        coco_classes = list(self.COCO_CLASSES)
        results = self.model.predict(
            source=frame,
            conf=min(self.confidence, self.motorcycle_confidence),
            iou=self.nms_iou,
            imgsz=self.image_size,
            device=self.device,
            classes=coco_classes,
            augment=self.augment,
            verbose=False,
        )
        detections: list[Detection] = []
        coco_motorcycles: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                class_name = self.COCO_CLASSES.get(class_id)
                if class_name is None:
                    continue
                if class_name != "motorcycle" and float(box.conf.item()) < self.confidence:
                    continue
                coordinates = box.xyxy[0].tolist()
                detection = Detection(
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    bbox=tuple(float(value) for value in coordinates),
                )
                if class_name == "motorcycle" and self.motorcycle_model is not None:
                    coco_motorcycles.append(detection)
                else:
                    detections.append(detection)
        specialist_motorcycles: list[Detection] = []
        if self.motorcycle_model is not None:
            motorcycle_results = self.motorcycle_model.predict(
                source=frame,
                conf=self.motorcycle_confidence,
                iou=self.motorcycle_nms_iou,
                imgsz=self.image_size,
                device=self.device,
                classes=[self.motorcycle_class_id],
                augment=self.augment,
                verbose=False,
            )
            for result in motorcycle_results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    coordinates = box.xyxy[0].tolist()
                    specialist_motorcycles.append(
                        Detection(
                            class_name="motorcycle",
                            confidence=float(box.conf.item()),
                            bbox=tuple(float(value) for value in coordinates),
                        )
                    )
            detections.extend(
                self._merge_motorcycle_fallbacks(
                    coco_motorcycles,
                    specialist_motorcycles,
                    minimum_confidence=0.50,
                    persons=[item for item in detections if item.class_name == "person"],
                    minimum_supported_confidence=self.motorcycle_confidence,
                    maximum_supported_existing=self.motorcycle_gap_max_existing,
                )
            )
        else:
            detections.extend(coco_motorcycles)
        if (
            self.motorcycle_gap_model is not None
            and sum(item.class_name == "motorcycle" for item in detections)
            < self.motorcycle_gap_max_existing
        ):
            gap_results = self.motorcycle_gap_model.predict(
                source=frame,
                conf=self.motorcycle_gap_confidence,
                iou=self.motorcycle_nms_iou,
                imgsz=self.motorcycle_gap_image_size,
                device=self.device,
                classes=[self.motorcycle_gap_class_id],
                augment=self.augment,
                verbose=False,
            )
            gap_candidates = []
            for result in gap_results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
                    gap_candidates.append(
                        Detection(
                            class_name="motorcycle",
                            confidence=float(box.conf.item()),
                            bbox=coordinates,
                        )
                    )
            detections.extend(
                self._merge_motorcycle_gap_detections(
                    [item for item in detections if item.class_name == "motorcycle"],
                    gap_candidates,
                    minimum_confidence=self.motorcycle_gap_confidence,
                    minimum_area=self.motorcycle_gap_min_area,
                )
            )
        detections = self._deduplicate_motorcycles(detections)
        detections = self._resolve_vehicle_class_conflicts(detections)
        if self.helmet_model is not None:
            detections.extend(self._detect_helmets(frame))
        return detections

    @classmethod
    def _resolve_vehicle_class_conflicts(cls, detections: list[Detection]) -> list[Detection]:
        """Remove weaker bicycle boxes for the same confidently detected motorbike."""
        motorcycles = [item for item in detections if item.class_name == "motorcycle"]
        return [item for item in detections if item.class_name != "bicycle" or not any(
            motor.confidence >= 0.50 and motor.confidence > item.confidence
            and cls._bbox_iou(motor.bbox, item.bbox) >= 0.50
            for motor in motorcycles
        )]

    @classmethod
    def _deduplicate_motorcycles(
        cls, detections: list[Detection], iou_threshold: float = 0.50,
        containment_threshold: float = 0.82, center_distance_ratio: float = 0.22,
    ) -> list[Detection]:
        """Collapse near-identical bike boxes returned by the detector ensemble.

        Cross-model predictions can describe the same motorcycle at different
        scales: ordinary IoU NMS misses a small box nested inside a larger box.
        Require either strong IoU or high containment *and* nearby centers so
        crowded, side-by-side motorcycles are not accidentally merged.
        """
        motorcycles = sorted(
            (item for item in detections if item.class_name == "motorcycle"),
            key=lambda item: item.confidence,
            reverse=True,
        )
        kept: list[Detection] = []
        for candidate in motorcycles:
            duplicate = False
            for existing in kept:
                overlap = cls._bbox_iou(candidate.bbox, existing.bbox)
                containment = max(
                    cls._bbox_coverage(candidate.bbox, existing.bbox),
                    cls._bbox_coverage(existing.bbox, candidate.bbox),
                )
                c = candidate.bbox
                e = existing.bbox
                center_distance = float(np.hypot(
                    (c[0] + c[2] - e[0] - e[2]) * 0.5,
                    (c[1] + c[3] - e[1] - e[3]) * 0.5,
                ))
                scale = max(
                    float(np.hypot(c[2] - c[0], c[3] - c[1])),
                    float(np.hypot(e[2] - e[0], e[3] - e[1])),
                    1.0,
                )
                if overlap >= iou_threshold or (
                    containment >= containment_threshold
                    and center_distance <= center_distance_ratio * scale
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        remaining = [item for item in detections if item.class_name != "motorcycle"]
        return [*remaining, *kept]

    def _detect_helmets(self, frame: np.ndarray) -> list[Detection]:
        if self.helmet_model is None:
            return []
        height, width = frame.shape[:2]
        grid = self._effective_helmet_tile_grid(frame)
        class_name_by_id = {
            class_id: class_name
            for class_name, class_id in self.helmet_class_ids.items()
        }
        shared_args = {
            "iou": self.helmet_nms_iou,
            "imgsz": self.image_size,
            "device": self.device,
            "classes": list(self.helmet_class_ids.values()),
            "augment": self.augment,
            "verbose": False,
        }
        detections: list[Detection] = []
        full_results = self.helmet_model.predict(
            source=frame, conf=self.helmet_confidence, **shared_args
        )
        detections.extend(
            self._helmet_result_detections(full_results, [(0, 0)], width, height,
                                           class_name_by_id)
        )

        if grid > 1:
            tile_width = min(
                width,
                int(np.ceil(width / (grid - (grid - 1) * self.helmet_tile_overlap))),
            )
            tile_height = min(
                height,
                int(np.ceil(height / (grid - (grid - 1) * self.helmet_tile_overlap))),
            )
            x_starts = self._tile_starts(width, tile_width, grid)
            y_starts = self._tile_starts(height, tile_height, grid)
            tiles = [
                frame[y:y + tile_height, x:x + tile_width]
                for y in y_starts
                for x in x_starts
            ]
            offsets = [(x, y) for y in y_starts for x in x_starts]
            # Ultralytics' NumPy-list loader ignores the batch argument and
            # stacks the entire list. Slice the actual source list to bound
            # memory and prevent time-limited NMS from dropping later tiles.
            batch_size = max(1, int(getattr(self, "helmet_batch_size", 4)))
            for start in range(0, len(tiles), batch_size):
                source_tiles = tiles[start:start + batch_size]
                tile_results = self.helmet_model.predict(
                    source=source_tiles,
                    conf=self.helmet_tile_confidence,
                    batch=len(source_tiles),
                    **shared_args,
                )
                detections.extend(
                    self._helmet_result_detections(
                        tile_results, offsets[start:start + batch_size], width, height, class_name_by_id
                    )
                )
        return self._helmet_nms(detections)

    def _effective_helmet_tile_grid(self, frame):
        return self.helmet_tile_grid

    @staticmethod
    def _tile_starts(length: int, tile_length: int, count: int) -> list[int]:
        if count <= 1 or tile_length >= length:
            return [0]
        return [round((length - tile_length) * index / (count - 1))
                for index in range(count)]

    @staticmethod
    def _helmet_result_detections(
        results,
        offsets: list[tuple[int, int]],
        frame_width: int,
        frame_height: int,
        class_name_by_id: dict[int, str],
    ) -> list[Detection]:
        detections = []
        for result, (x_offset, y_offset) in zip(results, offsets, strict=True):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_name = class_name_by_id.get(int(box.cls.item()))
                if class_name is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=float(box.conf.item()),
                        bbox=(
                            max(0.0, min(float(frame_width), x1 + x_offset)),
                            max(0.0, min(float(frame_height), y1 + y_offset)),
                            max(0.0, min(float(frame_width), x2 + x_offset)),
                            max(0.0, min(float(frame_height), y2 + y_offset)),
                        ),
                    )
                )
        return detections

    def _helmet_nms(self, detections: list[Detection]) -> list[Detection]:
        kept = []
        for detection in sorted(detections, key=lambda item: item.confidence,
                                reverse=True):
            # A helmet and a bare-head box can overlap on the same rider. Keep
            # both classes so the rider classifier can resolve the disagreement;
            # suppress duplicate boxes only within the same class.
            if all(
                detection.class_name != item.class_name
                or self._bbox_iou(detection.bbox, item.bbox) <= self.helmet_nms_iou
                for item in kept
            ):
                kept.append(detection)
        return kept

    @staticmethod
    def _bbox_iou(first, second) -> float:
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
        second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / (first_area + second_area - intersection)

    @classmethod
    def _merge_motorcycle_fallbacks(
        cls,
        coco_detections: list[Detection],
        specialist_detections: list[Detection],
        overlap_threshold: float = 0.35,
        minimum_confidence: float = 0.50,
        persons: list[Detection] | None = None,
        minimum_supported_confidence: float = 0.04,
        maximum_supported_existing: int = 24,
    ) -> list[Detection]:
        """Fill specialist gaps with high-confidence, non-duplicate COCO boxes."""
        from helmet_rider.association.rider_vehicle import RiderVehicleAssociator

        associator = RiderVehicleAssociator()
        recovery_allowed = len(specialist_detections) + sum(
            item.confidence >= minimum_confidence for item in coco_detections
        ) < maximum_supported_existing
        fallback = []
        for detection in sorted(
            coco_detections, key=lambda item: item.confidence, reverse=True
        ):
            if detection.confidence < minimum_confidence:
                if not recovery_allowed or detection.confidence < minimum_supported_confidence or not any(
                    person.confidence >= 0.50 and associator.supports(person, detection)
                    for person in persons or []
                ):
                    continue
            if any(
                cls._bbox_iou(detection.bbox, specialist.bbox) >= overlap_threshold
                for specialist in specialist_detections
            ):
                continue
            if any(
                cls._bbox_iou(detection.bbox, kept.bbox) >= overlap_threshold
                for kept in fallback
            ):
                continue
            fallback.append(detection)
        return [*specialist_detections, *fallback]

    @classmethod
    def _merge_motorcycle_gap_detections(
        cls,
        existing: list[Detection],
        candidates: list[Detection],
        minimum_confidence: float = 0.045,
        minimum_area: float = 800.0,
        overlap_threshold: float = 0.10,
        contained_threshold: float = 0.45,
    ) -> list[Detection]:
        """Add low-count background hits without duplicating bike subparts."""
        kept: list[Detection] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            x1, y1, x2, y2 = candidate.bbox
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if candidate.confidence < minimum_confidence or area < minimum_area:
                continue
            if any(
                cls._bbox_iou(candidate.bbox, item.bbox) >= overlap_threshold
                or cls._bbox_coverage(candidate.bbox, item.bbox) >= contained_threshold
                for item in [*existing, *kept]
            ):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _bbox_coverage(inner, outer) -> float:
        """Return the fraction of ``inner`` covered by ``outer``."""
        x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
        x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        inner_area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
        return intersection / inner_area


class ONNXRuntimeDetector:
    """Direct CPU ONNX Runtime for exports created with embedded NMS."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.08,
        image_size: int = 1280,
        device: str = "cpu",
        threads: int = 4,
    ) -> None:
        if device == "auto":
            device = "cpu"
        if device != "cpu":
            raise ValueError("The direct ONNX Runtime backend currently supports CPU only.")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX detector model not found: {path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional ONNX Runtime dependency to use backend=onnxruntime."
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        LOGGER.info("detector loaded: backend=onnxruntime model=%s", path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(
            frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
        left = (self.image_size - resized_w) // 2
        top = (self.image_size - resized_h) // 2
        canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_h, left:left + resized_w] = resized
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        raw = self.session.run(None, {self.input_name: tensor[None]})[0]
        rows = np.asarray(raw)
        if rows.ndim == 3:
            rows = rows[0]
        if rows.ndim != 2:
            raise RuntimeError(f"Unsupported ONNX detector output shape: {raw.shape}")
        if rows.shape[1] < 6 and rows.shape[0] >= 6:
            rows = rows.T
        if rows.shape[1] < 6:
            raise RuntimeError(
                "Expected ONNX output rows in xyxy, confidence, class format. "
                "Export with NMS enabled."
            )
        detections = []
        for x1, y1, x2, y2, confidence, class_id, *_ in rows:
            class_id = int(round(float(class_id)))
            name = UltralyticsDetector.COCO_CLASSES.get(class_id)
            if name is None or float(confidence) < self.confidence:
                continue
            box = (
                max(0.0, min(float(width), (float(x1) - left) / scale)),
                max(0.0, min(float(height), (float(y1) - top) / scale)),
                max(0.0, min(float(width), (float(x2) - left) / scale)),
                max(0.0, min(float(height), (float(y2) - top) / scale)),
            )
            detections.append(Detection(name, float(confidence), box))
        return detections


class UltralyticsPersonDetector(UltralyticsDetector):
    """Supplement real person detections using the shared tile/NMS machinery."""

    def __init__(self, model_path: str, *, device="cpu", threads=2, tile_grid=2, include_vehicles=False):
        import torch
        from ultralytics import YOLO

        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        torch.set_num_threads(max(1, threads))
        self.helmet_model = YOLO(model_path)
        names = self.helmet_model.names
        people = {int(i): n for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if str(n).lower() in {"person", "rider"}}
        if len(people) != 1:
            raise ValueError("Person recovery model must have a person class")
        self.helmet_class_ids = {"person": next(iter(people))}
        if include_vehicles:
            self.helmet_class_ids.update({str(n).lower(): int(i) for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if str(n).lower() in {"motorcycle", "bicycle"}})
        self.device = _resolve_device(device)
        self.image_size = 1280
        self.helmet_batch_size = 1
        self.augment = False
        self.helmet_confidence = .10
        self.helmet_nms_iou = .50
        self.helmet_tile_grid = tile_grid
        self.helmet_tile_overlap = .20
        self.helmet_tile_confidence = .10

    def detect(self, frame):
        return self._detect_helmets(frame)


class UltralyticsHeadDetector(UltralyticsDetector):
    """Load only a supplementary head model, sharing tile and NMS handling."""

    def __init__(self, model_path: str, *, device="cpu", threads=2, tile_grid=2, adaptive_tiles=False, tile_size=768):
        from ultralytics import YOLO

        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        import torch
        torch.set_num_threads(max(1, threads))
        self.helmet_model = YOLO(model_path)
        self.helmet_class_ids = {}
        names = self.helmet_model.names
        for class_id, name in (names.items() if isinstance(names, dict) else enumerate(names)):
            normalized = str(name).strip().lower().replace(" ", "_").replace("-", "_")
            if normalized in {"with_helmet", "helmet"}:
                self.helmet_class_ids["with_helmet"] = int(class_id)
            elif normalized in {"without_helmet", "no_helmet", "nohelmet"}:
                self.helmet_class_ids["without_helmet"] = int(class_id)
            elif normalized == "physical_head":
                self.helmet_class_ids["physical_head"] = int(class_id)
        if set(self.helmet_class_ids) not in ({"with_helmet", "without_helmet"}, {"physical_head"}):
            raise ValueError("Supplementary head detector must have helmet/no-helmet or physical_head classes")
        self.device = _resolve_device(device)
        self.image_size = 768
        self.augment = False
        self.helmet_confidence = .002
        self.helmet_nms_iou = .45
        self.helmet_tile_grid = tile_grid
        self.adaptive_tiles = adaptive_tiles
        self.tile_size = tile_size
        self.helmet_tile_overlap = .20
        self.helmet_tile_confidence = .005

    def detect(self, frame):
        return self._detect_helmets(frame)

    def _effective_helmet_tile_grid(self, frame):
        if not self.adaptive_tiles:
            return self.helmet_tile_grid
        if max(frame.shape[:2]) <= self.tile_size:
            return 1
        return min(self.helmet_tile_grid, max(2, int(np.ceil(max(frame.shape[:2])/self.tile_size))))


def create_detector(config: DetectorConfig, threads: int = 4) -> ObjectDetector:
    if config.backend == "onnxruntime":
        return ONNXRuntimeDetector(
            config.model, config.confidence, config.image_size, config.device, threads
        )
    if config.backend not in {"ultralytics", "ncnn"}:
        raise ValueError(
            f"Unsupported detector backend {config.backend!r}. "
            "Use ultralytics, ncnn, or onnxruntime."
        )
    return UltralyticsDetector(
        model_path=config.model,
        motorcycle_model_path=config.motorcycle_model,
        confidence=config.confidence,
        image_size=config.image_size,
        device=config.device,
        threads=threads,
        augment=config.augment,
        nms_iou=config.nms_iou,
        motorcycle_confidence=config.motorcycle_confidence,
        motorcycle_nms_iou=config.motorcycle_nms_iou,
        helmet_model_path=config.helmet_model,
        helmet_confidence=config.helmet_confidence,
        helmet_nms_iou=config.helmet_nms_iou,
        helmet_tile_grid=config.helmet_tile_grid,
        helmet_tile_overlap=config.helmet_tile_overlap,
        helmet_tile_confidence=config.helmet_tile_confidence,
        motorcycle_gap_model_path=config.motorcycle_gap_model,
        motorcycle_gap_confidence=config.motorcycle_gap_confidence,
        motorcycle_gap_image_size=config.motorcycle_gap_image_size,
        motorcycle_gap_min_area=config.motorcycle_gap_min_area,
        motorcycle_gap_max_existing=config.motorcycle_gap_max_existing,
    )

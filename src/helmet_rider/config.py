"""Validated, dependency-light YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DetectorConfig:
    model: str = "models/yolo26n.pt"
    confidence: float = 0.08
    image_size: int = 1280
    device: str = "cpu"
    backend: str = "ultralytics"
    augment: bool = True
    nms_iou: float = 0.55
    every_n_frames: int = 1
    motorcycle_model: str | None = None
    motorcycle_confidence: float = 0.08
    motorcycle_nms_iou: float = 0.45
    motorcycle_unassociated_confidence: float = 0.60
    motorcycle_gap_model: str | None = None
    motorcycle_gap_confidence: float = 0.045
    motorcycle_gap_image_size: int = 640
    motorcycle_gap_min_area: float = 800.0
    motorcycle_gap_max_existing: int = 24
    helmet_model: str | None = None
    helmet_confidence: float = 0.002
    helmet_min_positive_confidence: float = 0.35
    helmet_min_person_confidence: float = 0.10
    helmet_min_confidence_for_weak_person: float = 0.02
    helmet_nms_iou: float = 0.45
    helmet_tile_grid: int = 1
    helmet_tile_overlap: float = 0.20
    helmet_tile_confidence: float = 0.005


@dataclass(frozen=True)
class ClassifierConfig:
    model: str = "models/helmet_classifier.pt"
    backend: str = "tiny"
    input_size: int = 128
    min_confidence: float = 0.55
    helmet_positive_threshold: float = 0.60
    no_helmet_negative_threshold: float = 0.75
    helmet_crop_positive_threshold: float = 0.75
    every_n_frames: int = 3
    device: str = "cpu"
    temperature: float = 1.0
    evidence_max_age_seconds: float = 10.0
    headwear_model: str | None = None
    verify_small_heads: bool = False
    localized_refinement_model: str | None = None
    localized_refinement_threshold: float = .55
    localized_head_model: str | None = None
    localized_head_tile_grid: int = 2
    localized_head_tile_size: int = 768
    localized_head_min_confidence: float = .1
    localized_adaptive_tiles: bool = False
    localized_use_as_primary: bool = False
    localized_person_model: str | None = None
    localized_person_tile_grid: int = 2
    localized_recover_vehicles: bool = False
    localized_native_threshold: float = 0
    localized_relocalize_known: bool = False
    localized_relocalize_all_known: bool = False
    localized_verify_conflicts: bool = False
    localized_prefer_consensus: bool = False
    localized_verify_known: bool = False
    localized_occlusion_tolerant: bool = False
    localized_independent_heads: bool = False
    localized_independent_classify: bool = False
    localized_independent_classifier_model: str | None = None
    localized_independent_verification_threshold: float | None = None
    localized_independent_min_dimension: int = 0
    localized_head_validity_model: str | None = None
    localized_head_validity_threshold: float = .5
    localized_unassociated_head_validity_threshold: float = .8


@dataclass(frozen=True)
class AssociationConfig:
    threshold: float = 0.34
    lower_body_ratio: float = 0.42
    vehicle_expand_x: float = 0.22
    vehicle_expand_y: float = 0.12
    minimum_lower_body_overlap: float = 0.08


@dataclass(frozen=True)
class HeadROIConfig:
    strategy: str = "person"
    matched_context: float = 0.15
    height_ratio: float = 0.25
    width_expansion: float = 0.95
    minimum_size: int = 8


@dataclass(frozen=True)
class TrackingConfig:
    backend: str = "iou"
    enabled: bool = True
    iou_threshold: float = 0.12
    max_age: int = 12
    smoothing_window: int = 5
    bbox_smoothing: float = 0.62
    max_age_seconds: float = 3.0
    reset_gap_seconds: float = 10.0


@dataclass(frozen=True)
class CameraConfig:
    source: str = "webcam"
    width: int = 640
    height: int = 480
    fps: int = 25
    frame_skip: int = 0


@dataclass(frozen=True)
class VisualizationConfig:
    debug: bool = False
    show_association_lines: bool = False


@dataclass(frozen=True)
class PerformanceConfig:
    inference_threads: int = 4
    measure_memory: bool = False


@dataclass(frozen=True)
class AppConfig:
    detector: DetectorConfig = DetectorConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    association: AssociationConfig = AssociationConfig()
    head_roi: HeadROIConfig = HeadROIConfig()
    tracking: TrackingConfig = TrackingConfig()
    camera: CameraConfig = CameraConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    performance: PerformanceConfig = PerformanceConfig()


def _build(cls: type, values: dict[str, Any] | None) -> Any:
    values = values or {}
    allowed = cls.__dataclass_fields__
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML file and ignore unknown keys for forward compatibility."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    config = AppConfig(
        detector=_build(DetectorConfig, raw.get("detector")),
        classifier=_build(ClassifierConfig, raw.get("classifier")),
        association=_build(AssociationConfig, raw.get("association")),
        head_roi=_build(HeadROIConfig, raw.get("head_roi")),
        tracking=_build(TrackingConfig, raw.get("tracking")),
        camera=_build(CameraConfig, raw.get("camera")),
        visualization=_build(VisualizationConfig, raw.get("visualization")),
        performance=_build(PerformanceConfig, raw.get("performance")),
    )
    if not 0.0 <= config.detector.confidence <= 1.0:
        raise ValueError("detector.confidence must be between 0 and 1")
    if not 0.0 <= config.detector.nms_iou <= 1.0:
        raise ValueError("detector.nms_iou must be between 0 and 1")
    if not 0.0 <= config.detector.motorcycle_confidence <= 1.0:
        raise ValueError("detector.motorcycle_confidence must be between 0 and 1")
    if not 0.0 <= config.detector.motorcycle_nms_iou <= 1.0:
        raise ValueError("detector.motorcycle_nms_iou must be between 0 and 1")
    if not 0.0 <= config.detector.motorcycle_unassociated_confidence <= 1.0:
        raise ValueError(
            "detector.motorcycle_unassociated_confidence must be between 0 and 1"
        )
    if not 0.0 <= config.detector.motorcycle_gap_confidence <= 1.0:
        raise ValueError(
            "detector.motorcycle_gap_confidence must be between 0 and 1"
        )
    if config.detector.motorcycle_gap_image_size < 160:
        raise ValueError("detector.motorcycle_gap_image_size must be at least 160")
    if config.detector.motorcycle_gap_min_area < 0:
        raise ValueError("detector.motorcycle_gap_min_area must be non-negative")
    if config.detector.motorcycle_gap_max_existing < 0:
        raise ValueError("detector.motorcycle_gap_max_existing must be non-negative")
    if not 0.0 <= config.detector.helmet_confidence <= 1.0:
        raise ValueError("detector.helmet_confidence must be between 0 and 1")
    if not 0.0 <= config.detector.helmet_min_positive_confidence <= 1.0:
        raise ValueError(
            "detector.helmet_min_positive_confidence must be between 0 and 1"
        )
    if not 0.0 <= config.detector.helmet_min_person_confidence <= 1.0:
        raise ValueError(
            "detector.helmet_min_person_confidence must be between 0 and 1"
        )
    if not 0.0 <= config.detector.helmet_min_confidence_for_weak_person <= 1.0:
        raise ValueError(
            "detector.helmet_min_confidence_for_weak_person must be between 0 and 1"
        )
    if not 0.0 <= config.detector.helmet_nms_iou <= 1.0:
        raise ValueError("detector.helmet_nms_iou must be between 0 and 1")
    if config.detector.helmet_tile_grid < 1:
        raise ValueError("detector.helmet_tile_grid must be at least 1")
    if not 0.0 <= config.detector.helmet_tile_overlap < 0.5:
        raise ValueError("detector.helmet_tile_overlap must be between 0 and 0.5")
    if not 0.0 <= config.detector.helmet_tile_confidence <= 1.0:
        raise ValueError("detector.helmet_tile_confidence must be between 0 and 1")
    if config.detector.image_size < 160:
        raise ValueError("detector.image_size must be at least 160 pixels")
    if config.detector.every_n_frames < 1:
        raise ValueError("detector.every_n_frames must be positive")
    if config.classifier.device not in {"cpu", "auto", "mps", "cuda", "0"}:
        raise ValueError("Unsupported classifier.device")
    if not 0 < config.classifier.temperature < 100:
        raise ValueError("classifier.temperature must be positive and below 100")
    if not 0 < config.classifier.evidence_max_age_seconds < 3600:
        raise ValueError("classifier.evidence_max_age_seconds must be positive and below 3600")
    if config.head_roi.strategy not in {"person", "matched", "verified"}:
        raise ValueError("head_roi.strategy must be person, matched or verified")
    if not 0 <= config.head_roi.matched_context <= 1:
        raise ValueError("head_roi.matched_context must be between 0 and 1")
    if not 0 < config.tracking.max_age_seconds <= config.tracking.reset_gap_seconds < 3600:
        raise ValueError("tracking time limits must be positive, ordered, and below 3600")
    if config.classifier.input_size < 32:
        raise ValueError("classifier.input_size must be at least 32 pixels")
    if config.classifier.backend not in {"tiny", "efficientnet_b0"}:
        raise ValueError("classifier.backend must be 'tiny' or 'efficientnet_b0'")
    if config.classifier.headwear_model and config.classifier.backend != "efficientnet_b0":
        raise ValueError("classifier.headwear_model requires the efficientnet_b0 backend")
    if config.classifier.localized_head_model and not config.classifier.localized_refinement_model:
        raise ValueError("localized_head_model requires a localized_refinement_model")
    if config.classifier.localized_use_as_primary and not config.classifier.localized_head_model:
        raise ValueError("localized_use_as_primary requires a localized_head_model")
    if config.classifier.localized_person_model and not config.classifier.localized_head_model:
        raise ValueError("localized_person_model requires a localized_head_model")
    if config.classifier.localized_recover_vehicles and not config.classifier.localized_person_model:
        raise ValueError("localized_recover_vehicles requires a localized_person_model")
    if not 1 <= config.classifier.localized_person_tile_grid <= 4:
        raise ValueError("localized_person_tile_grid must be between 1 and 4")
    if not 1 <= config.classifier.localized_head_tile_grid <= 8:
        raise ValueError("localized_head_tile_grid must be between 1 and 8")
    if (config.classifier.localized_native_threshold or config.classifier.localized_relocalize_known or config.classifier.localized_relocalize_all_known) and not config.classifier.localized_head_model:
        raise ValueError("Native head refinement requires a localized_head_model")
    if config.classifier.localized_verify_conflicts and not config.classifier.headwear_model:
        raise ValueError("Localized conflict verification requires a headwear_model")
    if config.classifier.localized_prefer_consensus and not config.classifier.localized_verify_conflicts:
        raise ValueError("Localized classifier consensus requires localized_verify_conflicts")
    if not 128 <= config.classifier.localized_head_tile_size <= 1280:
        raise ValueError("localized_head_tile_size must be between 128 and 1280")
    if config.classifier.localized_independent_heads and not config.classifier.localized_head_model:
        raise ValueError("Independent head outputs require a localized head detector")
    if config.classifier.localized_independent_classify and not config.classifier.localized_independent_heads:
        raise ValueError("Independent head classification requires independent head outputs")
    verification = config.classifier.localized_independent_verification_threshold
    if verification is not None and (not .5 <= verification <= 1 or not config.classifier.localized_independent_classify):
        raise ValueError("Independent verification requires classification and a threshold in [.5, 1]")
    if config.classifier.localized_head_validity_model and not config.classifier.localized_independent_heads:
        raise ValueError("Head validity requires independent head outputs")
    if config.classifier.localized_unassociated_head_validity_threshold < config.classifier.localized_head_validity_threshold:
        raise ValueError("Unassociated head validity must be at least supported head validity")
    if config.classifier.localized_independent_min_dimension < 0:
        raise ValueError("localized_independent_min_dimension must be non-negative")
    if config.classifier.localized_verify_known and not config.classifier.localized_verify_conflicts:
        raise ValueError("Known head verification requires localized_verify_conflicts")
    for key, value in (
        ("classifier.min_confidence", config.classifier.min_confidence),
        ("classifier.helmet_positive_threshold",
         config.classifier.helmet_positive_threshold),
        ("classifier.no_helmet_negative_threshold",
         config.classifier.no_helmet_negative_threshold),
        ("classifier.localized_refinement_threshold",
         config.classifier.localized_refinement_threshold),
        ("classifier.localized_head_min_confidence",
         config.classifier.localized_head_min_confidence),
        ("classifier.localized_head_validity_threshold", config.classifier.localized_head_validity_threshold),
        ("classifier.localized_unassociated_head_validity_threshold", config.classifier.localized_unassociated_head_validity_threshold),
        ("classifier.localized_native_threshold",
         config.classifier.localized_native_threshold),
        ("classifier.helmet_crop_positive_threshold",
         config.classifier.helmet_crop_positive_threshold),
        ("association.threshold", config.association.threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
    if config.classifier.every_n_frames < 1:
        raise ValueError("classifier.every_n_frames must be positive")
    if config.tracking.backend not in {"iou", "bytetrack"}:
        raise ValueError("tracking.backend must be iou or bytetrack")
    if config.tracking.smoothing_window < 1:
        raise ValueError("tracking.smoothing_window must be positive")
    if not 0.0 <= config.tracking.bbox_smoothing <= 1.0:
        raise ValueError("tracking.bbox_smoothing must be between 0 and 1")
    if config.camera.width < 1 or config.camera.height < 1 or config.camera.fps < 1:
        raise ValueError("camera width, height, and FPS must be positive")
    project_root = Path(path).resolve().parent.parent
    detector_model = Path(config.detector.model)
    motorcycle_model = (
        Path(config.detector.motorcycle_model)
        if config.detector.motorcycle_model
        else None
    )
    motorcycle_gap_model = (
        Path(config.detector.motorcycle_gap_model)
        if config.detector.motorcycle_gap_model
        else None
    )
    helmet_model = (
        Path(config.detector.helmet_model)
        if config.detector.helmet_model
        else None
    )
    classifier_model = Path(config.classifier.model)
    headwear_model = Path(config.classifier.headwear_model) if config.classifier.headwear_model else None
    config = replace(
        config,
        detector=replace(
            config.detector,
            model=str(detector_model if detector_model.is_absolute()
                      else (project_root / detector_model).resolve()),
            motorcycle_model=(
                str(motorcycle_model if motorcycle_model.is_absolute()
                    else (project_root / motorcycle_model).resolve())
                if motorcycle_model
                else None
            ),
            motorcycle_gap_model=(
                str(motorcycle_gap_model if motorcycle_gap_model.is_absolute()
                    else (project_root / motorcycle_gap_model).resolve())
                if motorcycle_gap_model
                else None
            ),
            helmet_model=(
                str(helmet_model if helmet_model.is_absolute()
                    else (project_root / helmet_model).resolve())
                if helmet_model
                else None
            ),
        ),
        classifier=replace(
            config.classifier,
            model=str(classifier_model if classifier_model.is_absolute()
                      else (project_root / classifier_model).resolve()),
            headwear_model=(str(headwear_model if headwear_model.is_absolute()
                               else (project_root / headwear_model).resolve())
                           if headwear_model else None),
        ),
    )
    return config

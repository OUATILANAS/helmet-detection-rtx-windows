"""Shared CLI argument/configuration helpers."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from helmet_rider.config import AppConfig, load_config
from helmet_rider.pipeline.pipeline import HelmetRiderPipeline
from helmet_rider.utils.logging import configure_logging

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", help="Local detector checkpoint or exported model path")
    parser.add_argument("--classifier", help="Optional trained helmet classifier checkpoint")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", default=None, help="Inference device (cpu by default)")
    parser.add_argument(
        "--backend",
        choices=("ultralytics", "ncnn", "onnxruntime"),
        help="Inference runtime; NCNN models use the Ultralytics runtime wrapper",
    )
    parser.add_argument("--debug", action="store_true")


def configured(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    detector = config.detector
    classifier = config.classifier
    visualization = config.visualization
    if args.model:
        model = Path(args.model)
        detector = replace(detector, model=str(model.resolve()))
    if args.classifier:
        classifier = replace(classifier, model=str(Path(args.classifier).resolve()))
    if args.confidence is not None:
        detector = replace(detector, confidence=args.confidence)
    if args.imgsz is not None:
        detector = replace(detector, image_size=args.imgsz)
    if args.device is not None:
        detector = replace(detector, device=args.device)
    if args.backend is not None:
        detector = replace(detector, backend=args.backend)
    if args.debug:
        visualization = replace(visualization, debug=True)
    config = replace(config, detector=detector, classifier=classifier,
                     visualization=visualization)
    configure_logging("DEBUG" if args.debug else "INFO")
    return config


def make_pipeline(config: AppConfig) -> HelmetRiderPipeline:
    return HelmetRiderPipeline(config)


def output_path(value: str | None, fallback: str) -> Path | None:
    if value is None:
        return None
    if value:
        return Path(value)
    return ROOT / "data" / "outputs" / fallback

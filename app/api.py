"""FastAPI image upload and optional local camera stream."""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import logging
import os
import secrets
import socket
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.annotations import (
    helmet_yolo_label_text,
    split_records_by_sequence,
    validate_boxes,
    validate_helmet_samples,
    yolo_label_text,
)
from helmet_rider.camera.picamera import PiCameraSource
from helmet_rider.camera.rtsp import RTSPCamera
from helmet_rider.camera.webcam import OpenCVCamera
from helmet_rider.config import load_config
from helmet_rider.evaluation.reviews import load_reviews
from helmet_rider.pipeline.pipeline import HelmetRiderPipeline
from helmet_rider.utils.results import result_to_dict
from helmet_rider.utils.timing import add_capture_time, draw_timed
from helmet_rider.visualization.draw import result_summary

PROJECT_DIR = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = PROJECT_DIR / "dataset" / "motorcycle_review"
HELMET_REVIEW_DIR = PROJECT_DIR / "dataset" / "helmet_status_review"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "configs" / "default.yaml"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
REQUIRED_IMAGE_UPLOAD = File(...)
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Helmet Rider Detection", version="0.1.0")
_pipeline: HelmetRiderPipeline | None = None
_pipeline_lock = threading.Lock()
_inference_lock = threading.Lock()
_summary_lock = threading.Lock()
_annotation_lock = threading.Lock()
_rtsp_lock = threading.Lock()
_camera_stream_lock = threading.Lock()
_active_analysis_session_id: str | None = None
_rtsp_sessions: dict[str, tuple[RTSPCamera, threading.Timer]] = {}
_camera_stream_sessions: dict[str, threading.Event] = {}
_active_camera_stream_session_id: str | None = None
_camera_stats = {
    "active": False,
    "preview_only": False,
    "summary": {
        "motorcycles": 0, "bicycles": 0, "riders": 0,
        "helmet": 0, "no_helmet": 0, "unknown": 0,
    },
}


class RTSPSessionRequest(BaseModel):
    url: str


class CameraStopRequest(BaseModel):
    session_id: str


def config_path() -> Path:
    configured = Path(os.environ.get("HELMET_RIDER_CONFIG", DEFAULT_CONFIG_PATH))
    return configured if configured.is_absolute() else (PROJECT_DIR / configured).resolve()


def get_pipeline() -> HelmetRiderPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                config = load_config(config_path())
                _pipeline = HelmetRiderPipeline(config)
    return _pipeline


@app.get("/")
def index():
    return FileResponse(PROJECT_DIR / "app" / "static" / "index.html")


def _review_queue():
    path = PROJECT_DIR / "dataset" / "review_queue" / "manifest.json"
    return json.loads(path.read_text()).get("frames", []) if path.is_file() else []


@app.get("/api/review-queue")
def review_queue():
    return {"frames": _review_queue()}


@app.get("/api/review-queue/{identifier}")
def review_queue_image(identifier: str):
    record = next((r for r in _review_queue() if r["id"] == identifier), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Review frame not found.")
    root = (PROJECT_DIR / "dataset" / "review_queue").resolve()
    path = (root / record["filename"]).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail="Review frame not found.")
    return FileResponse(path)


@app.get("/api/health")
def health():
    config = yaml.safe_load(config_path().read_text(encoding="utf-8")) or {}
    detector_config = config.get("detector", {})
    model = Path(detector_config.get("model", ""))
    if not model.is_absolute():
        model = PROJECT_DIR / model
    motorcycle_model = detector_config.get("motorcycle_model")
    motorcycle_model_available = None
    if motorcycle_model:
        motorcycle_path = Path(motorcycle_model)
        if not motorcycle_path.is_absolute():
            motorcycle_path = PROJECT_DIR / motorcycle_path
        motorcycle_model_available = motorcycle_path.exists()
    helmet_model = detector_config.get("helmet_model")
    helmet_model_available = None
    if helmet_model:
        helmet_path = Path(helmet_model)
        if not helmet_path.is_absolute():
            helmet_path = PROJECT_DIR / helmet_path
        helmet_model_available = helmet_path.exists()
    classifier = (PROJECT_DIR / config.get("classifier", {}).get("model", "")).resolve()
    return {
        "status": "ok",
        "detector_model_available": model.exists(),
        "motorcycle_detector_model_available": motorcycle_model_available,
        "helmet_detector_model_available": helmet_model_available,
        "classifier_model_available": classifier.exists(),
    }


@app.get("/api/camera/stats")
def camera_stats():
    with _summary_lock:
        return dict(_camera_stats)


def _register_camera_stream(session_id: str, preview_only: bool) -> threading.Event:
    global _active_camera_stream_session_id
    stop_event = threading.Event()
    with _camera_stream_lock:
        for active_event in _camera_stream_sessions.values():
            active_event.set()
        _camera_stream_sessions[session_id] = stop_event
        _active_camera_stream_session_id = session_id
    with _summary_lock:
        _camera_stats.update(active=True, preview_only=preview_only)
    return stop_event


def _finish_camera_stream(session_id: str) -> None:
    global _active_camera_stream_session_id
    with _camera_stream_lock:
        _camera_stream_sessions.pop(session_id, None)
        was_active = _active_camera_stream_session_id == session_id
        if was_active:
            _active_camera_stream_session_id = None
    if was_active:
        with _summary_lock:
            _camera_stats.update(active=False)


@app.post("/api/camera/stop")
def stop_camera_stream(request: CameraStopRequest):
    """Stop a server-side webcam or RTSP stream opened by the web interface."""
    global _active_camera_stream_session_id
    session_id = request.session_id.strip()
    if not session_id or len(session_id) > 128:
        raise HTTPException(status_code=400, detail="Enter a valid camera session ID.")

    pending_rtsp = None
    with _rtsp_lock:
        pending_rtsp = _rtsp_sessions.pop(session_id, None)
    if pending_rtsp:
        pending_rtsp[1].cancel()
        pending_rtsp[0].release()

    with _camera_stream_lock:
        stop_event = _camera_stream_sessions.get(session_id)
        if stop_event:
            stop_event.set()
        was_active = _active_camera_stream_session_id == session_id
        if was_active:
            _active_camera_stream_session_id = None

    if was_active:
        with _summary_lock:
            _camera_stats.update(active=False)
    return {"stopped": bool(pending_rtsp or stop_event), "session_id": session_id}


def _validate_rtsp_url(url: str) -> None:
    if not url or len(url) > 2048:
        raise HTTPException(status_code=400, detail="Enter a valid RTSP URL (up to 2048 characters).")
    try:
        parsed = urlsplit(url)
        port = parsed.port if parsed.port is not None else 554
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="The RTSP URL has an invalid port.") from exc
    if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
        raise HTTPException(
            status_code=400,
            detail="Use an rtsp:// or rtsps:// URL for a network CCTV camera.",
        )
    if not 1 <= port <= 65535:
        raise HTTPException(status_code=400, detail="The RTSP port must be between 1 and 65535.")
    try:
        addresses = {
            ipaddress.ip_address(info[4][0].split("%", 1)[0])
            for info in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        }
    except OSError as exc:
        if parsed.hostname.lower().endswith(".local"):
            return
        raise HTTPException(
            status_code=400,
            detail="Could not resolve the CCTV host. Check the camera name or local network.",
        ) from exc
    if not addresses or any(
        not (address.is_private or address.is_loopback or address.is_link_local)
        or address.is_multicast or address.is_unspecified or address.is_reserved
        for address in addresses
    ):
        raise HTTPException(
            status_code=400,
            detail="For safety, this app accepts RTSP cameras on the local network only.",
        )


def _expire_rtsp_session(session_id: str) -> None:
    with _rtsp_lock:
        entry = _rtsp_sessions.pop(session_id, None)
    if entry:
        entry[0].release()


@app.post("/api/camera/rtsp/session")
async def start_rtsp_session(request: RTSPSessionRequest):
    """Open a local-network RTSP source without putting credentials in a URL path."""
    _validate_rtsp_url(request.url.strip())
    try:
        camera = await run_in_threadpool(RTSPCamera, request.url.strip())
    except (RuntimeError, cv2.error) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not open the RTSP camera. Check its stream path, credentials, and network access.",
        ) from exc
    session_id = secrets.token_urlsafe(24)
    expiry = threading.Timer(45, _expire_rtsp_session, args=(session_id,))
    expiry.daemon = True
    with _rtsp_lock:
        _rtsp_sessions[session_id] = (camera, expiry)
    expiry.start()
    return {"session_id": session_id}


def _saved_annotation_records() -> list[dict]:
    metadata_dir = ANNOTATION_DIR / "metadata" / "all"
    if not metadata_dir.is_dir():
        return []
    records = []
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("id") == path.stem:
            records.append(record)
    return records


def _annotation_stats(records: list[dict]) -> dict[str, int | bool]:
    sequences = {str(record.get("sequence", "")) for record in records}
    return {
        "images": len(records),
        "boxes": sum(int(record.get("box_count", 0)) for record in records),
        "sequences": len(sequences),
        "export_ready": len(records) >= 2 and len(sequences) >= 2,
    }


@app.get("/api/annotations/stats")
def annotation_stats():
    with _annotation_lock:
        return _annotation_stats(_saved_annotation_records())


def _saved_helmet_review_records() -> list[dict]:
    metadata_dir = HELMET_REVIEW_DIR / "metadata" / "all"
    if not metadata_dir.is_dir():
        return []
    records = []
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("id") == path.stem:
            records.append(record)
    return records


def _helmet_review_stats(records: list[dict]) -> dict[str, int | bool]:
    labels = [
        sample.get("label")
        for record in records
        for sample in record.get("samples", [])
        if isinstance(sample, dict)
    ]
    sequences = {str(record.get("sequence", "")) for record in records}
    return {
        "images": len(records),
        "sequences": len(sequences),
        "helmet": labels.count("helmet"),
        "no_helmet": labels.count("no_helmet"),
        "unknown": labels.count("unknown"),
        "export_ready": (
            len(sequences) >= 3
            and labels.count("helmet") > 0
            and labels.count("no_helmet") > 0
        ),
    }


@app.get("/api/helmet-review/stats")
def helmet_review_stats():
    with _annotation_lock:
        return _helmet_review_stats(_saved_helmet_review_records())


@app.post("/api/helmet-review")
async def save_helmet_status_review(
    file: UploadFile = REQUIRED_IMAGE_UPLOAD,
    samples: str = Form(...),
    sequence: str = Form(...),
    fully_reviewed: bool = Form(default=False),
    timestamp_seconds: float | None = Form(default=None),
):
    """Save human-confirmed rider head statuses for target-camera fine-tuning."""
    if timestamp_seconds is not None and (not np.isfinite(timestamp_seconds) or timestamp_seconds < 0):
        raise HTTPException(status_code=400, detail="Timestamp must be finite and non-negative.")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    sequence = sequence.strip()
    if not sequence or len(sequence) > 120:
        raise HTTPException(
            status_code=400, detail="Capture group must contain 1 to 120 characters."
        )
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 20 MB.")
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")
    height, width = frame.shape[:2]
    try:
        parsed = json.loads(samples)
        clean_samples = validate_helmet_samples(parsed, width, height)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid rider labels: {exc}") from exc

    extension_by_type = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    extension = extension_by_type.get((file.content_type or "").lower())
    if extension is None:
        suffix = Path(file.filename or "").suffix.lower()
        extension = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"

    identifier = uuid.uuid4().hex
    metadata = {
        "id": identifier,
        "original_filename": Path(file.filename or "image").name[:255],
        "sequence": sequence,
        "width": width,
        "height": height,
        "samples": clean_samples,
        "fully_reviewed": fully_reviewed,
        "timestamp_seconds": timestamp_seconds,
        "known_sample_count": sum(
            sample["label"] in {"helmet", "no_helmet"} for sample in clean_samples
        ),
        "extension": extension,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    image_path = HELMET_REVIEW_DIR / "images" / "all" / f"{identifier}{extension}"
    label_path = HELMET_REVIEW_DIR / "labels" / "all" / f"{identifier}.txt"
    metadata_path = HELMET_REVIEW_DIR / "metadata" / "all" / f"{identifier}.json"
    with _annotation_lock:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(content)
        label_path.write_text(
            helmet_yolo_label_text(clean_samples, width, height), encoding="utf-8"
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        stats = _helmet_review_stats(_saved_helmet_review_records())
    return {"saved": True, "id": identifier, "known_samples": metadata["known_sample_count"], **stats}


@app.get("/api/helmet-review/export")
def export_helmet_status_reviews():
    with _annotation_lock:
        records, audit = load_reviews(HELMET_REVIEW_DIR)
        if audit["conflicts"]:
            raise HTTPException(status_code=409, detail="Conflicting duplicate reviews must be resolved before export. Run scripts/evaluate_riders.py --audit-only for record IDs.")
        stats = _helmet_review_stats(records)
        if not stats["export_ready"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Review helmet and no-helmet riders in at least three independent "
                    "capture groups before exporting train/val/test data."
                ),
            )
        try:
            splits = split_records_by_sequence(records)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            split_manifest = {split_name: [] for split_name in splits}
            class_counts = {split_name: {"helmet": 0, "no_helmet": 0} for split_name in splits}
            runtime_rows = []
            for split_name, split_items in splits.items():
                for record in split_items:
                    identifier = str(record["id"])
                    extension = str(record.get("extension", ".jpg"))
                    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
                        continue
                    image_path = HELMET_REVIEW_DIR / "images" / "all" / f"{identifier}{extension}"
                    label_path = HELMET_REVIEW_DIR / "labels" / "all" / f"{identifier}.txt"
                    if not image_path.is_file() or not label_path.is_file():
                        continue
                    archive.write(image_path, f"images/{split_name}/{identifier}{extension}")
                    archive.write(label_path, f"labels/{split_name}/{identifier}.txt")
                    archive.writestr(
                        f"annotations/{split_name}/{identifier}.json",
                        json.dumps(record, indent=2),
                    )
                    split_manifest[split_name].append(identifier)
                    for sample_index, sample in enumerate(record.get("samples", [])):
                        if sample.get("label") in class_counts[split_name]:
                            class_counts[split_name][sample["label"]] += 1
                            runtime_rows.append({
                                "path": f"images/{split_name}/{identifier}{extension}",
                                "label": sample["label"], "sequence": record["sequence"], "split": split_name,
                                "bbox": sample.get("runtime_bbox", sample["bbox"]),
                                "reviewed_bbox": sample["bbox"], "runtime_geometry": "runtime_bbox" in sample,
                                "image_sha256": record["image_sha256"], "sample_index": sample_index,
                            })
            if any(class_counts[split].get(label, 0) == 0 for split in splits for label in ("helmet", "no_helmet")):
                raise HTTPException(status_code=409, detail="Every capture-group split needs both known classes. Review additional groups before exporting.")
            archive.writestr("runtime_manifest.json", json.dumps({
                "version": 1, "class_order": ["helmet", "no_helmet"], "rows": runtime_rows,
                "counts": class_counts, "audit": audit,
            }, indent=2))
            archive.writestr(
                "dataset.yaml",
                "path: .\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "names:\n  0: helmet\n  1: no_helmet\n",
            )
            archive.writestr("split_manifest.json", json.dumps({
                "frames": split_manifest,
                "class_counts": class_counts,
                "split_unit": "capture group",
                "unknown_statuses_excluded_from_yolo_labels": True,
            }, indent=2))
            archive.writestr(
                "README.txt",
                "Human-reviewed rider head-status annotations from local target footage.\n\n"
                "Class 0 is helmet; class 1 is no_helmet. Unknown/uncertain status is "
                "preserved in per-frame JSON but excluded from YOLO labels. Splits are "
                "grouped by capture sequence; do not move frames between splits. Runtime crop geometry is preserved in runtime_manifest.json. This is classifier training data; incomplete/unknown head labels must not be treated as detector background.\n\n"
                "Fine-tune the head classifier from the project using:\n"
                "python training/finetune_head_classifier.py --dataset <extracted-dir> "
                "--initial models/helmet_head_edgevision.pt --output <candidate-dir>\n\n"
                "These user-provided images are local-only. Confirm rights before sharing.\n",
            )
        archive_buffer.seek(0)
    return StreamingResponse(
        archive_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="helmet-status-review-dataset.zip"'
        },
    )


@app.post("/api/annotations")
async def save_motorcycle_annotation(
    file: UploadFile = REQUIRED_IMAGE_UPLOAD,
    annotations: str = Form(...),
    sequence: str = Form(...),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    sequence = sequence.strip()
    if not sequence or len(sequence) > 120:
        raise HTTPException(
            status_code=400, detail="Capture group must contain 1 to 120 characters."
        )
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 20 MB.")
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")
    height, width = frame.shape[:2]
    try:
        parsed = json.loads(annotations)
        boxes = validate_boxes(parsed, width, height)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid annotations: {exc}") from exc

    extension_by_type = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    extension = extension_by_type.get((file.content_type or "").lower())
    if extension is None:
        suffix = Path(file.filename or "").suffix.lower()
        extension = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"

    identifier = uuid.uuid4().hex
    metadata = {
        "id": identifier,
        "original_filename": Path(file.filename or "image").name[:255],
        "sequence": sequence,
        "width": width,
        "height": height,
        "box_count": len(boxes),
        "boxes": boxes,
        "extension": extension,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    image_path = ANNOTATION_DIR / "images" / "all" / f"{identifier}{extension}"
    label_path = ANNOTATION_DIR / "labels" / "all" / f"{identifier}.txt"
    metadata_path = ANNOTATION_DIR / "metadata" / "all" / f"{identifier}.json"
    with _annotation_lock:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(content)
        label_path.write_text(yolo_label_text(boxes, width, height), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        stats = _annotation_stats(_saved_annotation_records())
    return {"saved": True, "id": identifier, "box_count": len(boxes), **stats}


@app.get("/api/annotations/export")
def export_motorcycle_annotations():
    with _annotation_lock:
        records = _saved_annotation_records()
        try:
            splits = split_records_by_sequence(records)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            split_manifest = {}
            for split_name, split_items in splits.items():
                split_manifest[split_name] = []
                for record in split_items:
                    identifier = str(record["id"])
                    extension = str(record.get("extension", ".jpg"))
                    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
                        continue
                    image_path = ANNOTATION_DIR / "images" / "all" / f"{identifier}{extension}"
                    label_path = ANNOTATION_DIR / "labels" / "all" / f"{identifier}.txt"
                    if not image_path.is_file() or not label_path.is_file():
                        continue
                    archive.write(image_path, f"images/{split_name}/{identifier}{extension}")
                    archive.write(label_path, f"labels/{split_name}/{identifier}.txt")
                    archive.writestr(
                        f"annotations/{split_name}/{identifier}.json",
                        json.dumps(record, indent=2),
                    )
                    split_manifest[split_name].append(identifier)
            dataset_yaml = (
                "path: .\n"
                "train: images/train\n"
                "val: images/val\n"
                + ("test: images/test\n" if split_manifest["test"] else "")
                + "names:\n  0: motorcycle\n"
            )
            archive.writestr("dataset.yaml", dataset_yaml)
            archive.writestr(
                "split_manifest.json", json.dumps(split_manifest, indent=2)
            )
            archive.writestr(
                "README.txt",
                "Motorcycle detection review dataset\n\n"
                "Images and YOLO labels are split by capture group, not by frame. "
                "Use different group names for independent videos/scenes. Empty label "
                "files represent reviewed images with no visible motorcycles.\n\n"
                "Fine-tune from the local road-scene checkpoint with Ultralytics, for example:\n"
                "yolo detect train model=/path/to/trafficai_motorbike.pt "
                "data=dataset.yaml imgsz=1280\n\n"
                "Do not tune thresholds on the test split.\n",
            )
        archive_buffer.seek(0)
    return StreamingResponse(
        archive_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="motorcycle-review-dataset.zip"'
        },
    )


@app.post("/api/detect")
async def detect_image(
    file: UploadFile = REQUIRED_IMAGE_UPLOAD,
    session_id: str | None = Form(default=None),
    timestamp_seconds: float | None = Form(default=None),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    if timestamp_seconds is not None and (not np.isfinite(timestamp_seconds) or timestamp_seconds < 0):
        raise HTTPException(status_code=400, detail="Timestamp must be finite and non-negative.")
    capture_timestamp = timestamp_seconds if timestamp_seconds is not None else time.monotonic()
    capture_started = perf_counter()
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 20 MB.")
    encoded = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    capture_ms = (perf_counter() - capture_started) * 1000.0
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")
    try:
        pipeline = get_pipeline()
        result = await run_in_threadpool(process_for_web, pipeline, frame, session_id, capture_timestamp)
        result = add_capture_time(result, capture_ms)
        config = pipeline.config
        annotated, result = draw_timed(frame, result, config)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    success, image_bytes = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode result image.")
    payload = result_to_dict(result)
    payload["summary"] = result_summary(result)
    payload["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(
        image_bytes.tobytes()
    ).decode("ascii")
    return JSONResponse(payload)


def process_for_web(
    pipeline: HelmetRiderPipeline,
    frame: np.ndarray,
    session_id: str | None = None,
    timestamp_seconds: float | None = None,
):
    global _active_analysis_session_id
    with _inference_lock:
        if session_id is None or session_id != _active_analysis_session_id:
            pipeline.reset()
            _active_analysis_session_id = session_id
        # Legacy test adapters may omit timestamp support when no timestamp was supplied.
        kwargs = {"temporal_consistency": session_id is not None}
        if timestamp_seconds is not None:
            kwargs["timestamp_seconds"] = timestamp_seconds
        return pipeline.process(frame, **kwargs)


def camera_frames(
    camera,
    pipeline: HelmetRiderPipeline | None,
    session_id: str | None = None,
    stop_event: threading.Event | None = None,
):
    global _camera_stats
    config = pipeline.config if pipeline is not None else None
    try:
        while stop_event is None or not stop_event.is_set():
            capture_started = perf_counter()
            ok, frame = camera.read()
            capture_ms = (perf_counter() - capture_started) * 1000.0
            if (stop_event is not None and stop_event.is_set()) or not ok or frame is None:
                break
            if pipeline is not None:
                result = add_capture_time(
                    process_for_web(pipeline, frame, session_id, time.monotonic()), capture_ms
                )
                annotated, result = draw_timed(frame, result, config)
                frame_summary = result_summary(result)
                timings = result_to_dict(result)["timings"]
            else:
                annotated = frame
                frame_summary = {
                    "motorcycles": 0, "bicycles": 0, "riders": 0,
                    "helmet": 0, "no_helmet": 0, "unknown": 0,
                }
                timings = {"capture_ms": round(capture_ms, 2)}
            with _summary_lock:
                _camera_stats = {
                    "active": True,
                    "preview_only": pipeline is None,
                    "summary": frame_summary,
                    "timings": timings,
                }
            ok, data = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + data.tobytes()
                    + b"\r\n"
                )
    finally:
        camera.release()
        if session_id is not None:
            _finish_camera_stream(session_id)


@app.get("/api/camera/stream")
def camera_stream(source: str = "webcam", session_id: str | None = None, quality: str = "original"):
    if quality not in {"original", "1280", "640"}:
        raise HTTPException(status_code=400, detail="Choose original, 1280, or 640 camera detail.")
    camera_width, camera_height = {"original": (1920, 1080), "1280": (1280, 720), "640": (640, 480)}[quality]
    if source not in {"webcam", "picamera"}:
        raise HTTPException(status_code=400, detail="Source must be webcam or picamera.")
    try:
        try:
            pipeline = get_pipeline()
        except (FileNotFoundError, RuntimeError) as exc:
            pipeline = None
            LOGGER.info("Starting camera preview without inference: %s", exc)
        camera = (
            PiCameraSource(
                camera_width,
                camera_height,
                pipeline.config.camera.fps if pipeline else 25,
            )
            if source == "picamera"
            else OpenCVCamera(
                0,
                camera_width,
                camera_height,
                pipeline.config.camera.fps if pipeline else 25,
            )
        )
        analysis_session_id = session_id or secrets.token_urlsafe(18)
        stop_event = _register_camera_stream(analysis_session_id, pipeline is None)
        return StreamingResponse(
            camera_frames(camera, pipeline, analysis_session_id, stop_event),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/camera/rtsp/stream/{session_id}")
def camera_rtsp_stream(session_id: str):
    with _rtsp_lock:
        entry = _rtsp_sessions.pop(session_id, None)
    if entry is None:
        raise HTTPException(status_code=404, detail="RTSP session expired. Connect to the camera again.")
    camera, expiry = entry
    expiry.cancel()
    try:
        try:
            pipeline = get_pipeline()
        except (FileNotFoundError, RuntimeError) as exc:
            pipeline = None
            LOGGER.info("Starting RTSP preview without inference: %s", exc)
        stop_event = _register_camera_stream(session_id, pipeline is None)
        return StreamingResponse(
            camera_frames(camera, pipeline, session_id, stop_event),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception:
        camera.release()
        raise

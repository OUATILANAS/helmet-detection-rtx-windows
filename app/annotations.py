"""Helpers for validated motorcycle boxes and YOLO dataset packaging."""

from __future__ import annotations

import math

from helmet_rider.evaluation.reviews import split_records_by_sequence as split_records_by_sequence


def validate_boxes(boxes, width: int, height: int, maximum: int = 500) -> list[list[float]]:
    """Clamp submitted xyxy boxes to the image and reject malformed entries."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive.")
    if not isinstance(boxes, list) or len(boxes) > maximum:
        raise ValueError(f"Annotations must be a list of at most {maximum} boxes.")
    clean: list[list[float]] = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("Each annotation must contain x1, y1, x2, and y2.")
        if any(isinstance(value, bool) for value in box):
            raise ValueError("Box coordinates must be numbers.")
        try:
            coordinates = [float(value) for value in box]
        except (TypeError, ValueError) as exc:
            raise ValueError("Box coordinates must be numbers.") from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("Box coordinates must be finite numbers.")
        x1, y1, x2, y2 = coordinates
        x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
        y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            raise ValueError("Each box must be at least 2 pixels wide and high.")
        clean.append([x1, y1, x2, y2])
    return clean


def yolo_label_text(boxes: list[list[float]], width: int, height: int) -> str:
    """Convert pixel xyxy boxes to the one-class YOLO normalized xywh format."""
    lines = []
    for x1, y1, x2, y2 in boxes:
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def validate_helmet_samples(
    samples, width: int, height: int, maximum: int = 500
) -> list[dict]:
    """Validate human-reviewed rider head boxes and their helmet labels."""
    if not isinstance(samples, list) or len(samples) > maximum:
        raise ValueError(f"Samples must be a list of at most {maximum} riders.")
    clean = []
    allowed = {"helmet", "no_helmet", "unknown"}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Each rider sample must be an object.")
        label = sample.get("label")
        if not isinstance(label, str) or label not in allowed:
            raise ValueError("Helmet labels must be helmet, no_helmet, or unknown.")
        box = validate_boxes([sample.get("bbox")], width, height, maximum=1)[0]
        track_id = sample.get("track_id")
        if track_id is not None and (isinstance(track_id, bool) or not isinstance(track_id, int)):
            raise ValueError("Rider track IDs must be integers.")
        confidence = sample.get("model_confidence")
        if confidence is not None:
            if isinstance(confidence, bool):
                raise ValueError("Model confidence must be a number.")
            try:
                confidence = float(confidence)
            except (TypeError, ValueError) as exc:
                raise ValueError("Model confidence must be a number.") from exc
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("Model confidence must be between 0 and 1.")
        extra = {}
        if sample.get("entity_id"):
            entity = sample["entity_id"]
            if not isinstance(entity, str) or len(entity) > 120:
                raise ValueError("Human rider reference must be a string of at most 120 characters.")
            extra["entity_id"] = entity.strip()
        for key in ("runtime_bbox", "rider_bbox", "vehicle_bbox"):
            if sample.get(key) is not None:
                extra[key] = validate_boxes([sample[key]], width, height, maximum=1)[0]
        clean.append({
            **extra,
            "bbox": box,
            "label": label,
            "track_id": track_id,
            "model_confidence": confidence,
        })
    return clean


def helmet_yolo_label_text(samples: list[dict], width: int, height: int) -> str:
    """Write reviewed head boxes as helmet=0 and no_helmet=1 YOLO labels."""
    lines = []
    class_ids = {"helmet": 0, "no_helmet": 1}
    for sample in samples:
        class_id = class_ids.get(sample["label"])
        if class_id is None:
            continue
        x1, y1, x2, y2 = sample["bbox"]
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")

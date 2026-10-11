"""Readable result drawing with optional association diagnostics."""

from __future__ import annotations

import cv2
import numpy as np

from helmet_rider.association.geometry import bottom_center, expand
from helmet_rider.config import AppConfig
from helmet_rider.detection.types import FrameResult, HelmetStatus

VEHICLE_COLOR = (210, 180, 70)
HELMET_COLOR = (70, 190, 80)
NO_HELMET_COLOR = (55, 65, 230)
UNKNOWN_COLOR = (130, 145, 155)
POSSIBLE_HELMET_PROBABILITY = 0.50


def _draw_status_tag(
    image: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]
) -> None:
    font_scale = max(0.34, min(0.48, min(image.shape[:2]) / 1000.0))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    width, height = image.shape[1], image.shape[0]
    tag_width = text_width + 8
    tag_height = text_height + baseline + 6
    left = min(max(0, x), max(0, width - tag_width))
    top = y - tag_height if y >= tag_height else y
    top = min(max(0, top), max(0, height - tag_height))
    cv2.rectangle(
        image,
        (left, top),
        (min(width - 1, left + tag_width), min(height - 1, top + tag_height)),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        text,
        (left + 4, min(height - 1, top + text_height + 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_result(frame: np.ndarray, result: FrameResult, config: AppConfig) -> np.ndarray:
    output = frame.copy()
    color_by_status = {
        HelmetStatus.HELMET: HELMET_COLOR,
        HelmetStatus.NO_HELMET: NO_HELMET_COLOR,
        HelmetStatus.UNKNOWN: UNKNOWN_COLOR,
    }
    line_width = max(1, min(output.shape[:2]) // 360)
    for person in result.persons:
        if person.motorcycle_status == "no_motorcycle_match":
            x1, y1, x2, y2 = map(int, person.bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), UNKNOWN_COLOR, line_width)
            _draw_status_tag(output, x1, y1, "NO MOTO MATCH", UNKNOWN_COLOR)
    label_area = output.shape[0] * output.shape[1] * 0.025
    for vehicle in result.vehicles:
        x1, y1, x2, y2 = (int(value) for value in vehicle.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), VEHICLE_COLOR, line_width)
        vehicle_area = (x2 - x1) * (y2 - y1)
        small_motorcycle_label = (
            vehicle.vehicle_type == "motorcycle"
            and vehicle.confidence < 0.10
            and vehicle_area >= label_area * 0.40
        )
        if vehicle_area >= label_area or small_motorcycle_label:
            label = (
                "Moto?"
                if small_motorcycle_label
                else ("Moto" if vehicle.vehicle_type == "motorcycle" else "Bike")
            )
            cv2.putText(output, label, (x1, max(14, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, VEHICLE_COLOR, 1, cv2.LINE_AA)
    subjects = result.heads if result.heads is not None else result.riders
    for rider in subjects:
        if (
            rider.helmet.status == HelmetStatus.UNKNOWN
            and getattr(rider, "helmet_bbox", None) is None
            and rider.head_bbox is None
        ):
            continue
        color = color_by_status[rider.helmet.status]
        if (
            getattr(rider, "helmet_bbox", None) is not None
            and rider.helmet.status == HelmetStatus.HELMET
        ):
            status_box = rider.helmet_bbox
        else:
            # The fallback classifies the head crop, so mark that crop instead
            # of surrounding the full rider box with a large, noisy label.
            status_box = rider.head_bbox or rider.bbox
        x1, y1, x2, y2 = (int(value) for value in status_box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, line_width)
        label = {
            HelmetStatus.HELMET: "H",
            HelmetStatus.NO_HELMET: "NO",
            HelmetStatus.UNKNOWN: "?",
        }[rider.helmet.status]
        if (
            rider.helmet.status == HelmetStatus.UNKNOWN
            and rider.helmet.helmet_probability > POSSIBLE_HELMET_PROBABILITY
        ):
            label = "H?"
        _draw_status_tag(output, x1, max(0, y1), label, color)
        if config.visualization.show_association_lines:
            vehicle = next((v for v in result.vehicles if v.track_id == rider.vehicle_id), None)
            if vehicle:
                cv2.line(output, (int((x1 + x2) / 2), int(y2)),
                         (int((vehicle.bbox[0] + vehicle.bbox[2]) / 2),
                          int((vehicle.bbox[1] + vehicle.bbox[3]) / 2)), color, 1, cv2.LINE_AA)
        if config.visualization.debug and rider.head_bbox is not None:
            hx1, hy1, hx2, hy2 = (int(value) for value in rider.head_bbox)
            cv2.rectangle(output, (hx1, hy1), (hx2, hy2), (0, 210, 220), 1)
    if config.visualization.debug:
        for detection in result.detections:
            if detection.class_name == "person":
                x1, y1, x2, y2 = map(int, detection.bbox)
                cv2.rectangle(output, (x1, y1), (x2, y2), (120, 120, 120), 1)
                px, py = bottom_center(detection.bbox)
                cv2.circle(output, (int(px), int(py)), 3, (255, 200, 0), -1)
        for association in result.associations:
            if association.vehicle_index >= len(result.vehicles):
                continue
            vehicle = result.vehicles[association.vehicle_index]
            expanded = expand(
                vehicle.bbox,
                config.association.vehicle_expand_x,
                config.association.vehicle_expand_y,
            )
            cv2.rectangle(output, (int(expanded[0]), int(expanded[1])),
                          (int(expanded[2]), int(expanded[3])), (90, 90, 90), 1)
            text = f"score={association.score:.2f} lower={association.lower_body_overlap:.2f}"
            cv2.putText(output, text, (int(vehicle.bbox[0]), int(vehicle.bbox[3] + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        tile_size = min(88, max(32, output.shape[0] // 5), output.shape[1] // 4)
        for index, rider in enumerate(subjects):
            if rider.head_bbox is None:
                continue
            hx1, hy1, hx2, hy2 = (int(value) for value in rider.head_bbox)
            crop = frame[max(0, hy1):min(frame.shape[0], hy2),
                         max(0, hx1):min(frame.shape[1], hx2)]
            if crop.size == 0:
                continue
            thumb = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            left = max(0, output.shape[1] - tile_size - 8)
            top = 8 + index * (tile_size + 22)
            if top + tile_size + 18 >= output.shape[0]:
                break
            output[top:top + tile_size, left:left + tile_size] = thumb
            cv2.rectangle(output, (left, top), (left + tile_size, top + tile_size),
                          (0, 210, 220), 1)
            cv2.putText(output, f"HEAD #{rider.track_id}", (left, top + tile_size + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 210, 220), 1, cv2.LINE_AA)
        cv2.putText(output, f"{result.timings.total_ms:.1f} ms | {result.fps:.1f} FPS",
                    (12, output.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (245, 245, 245), 1, cv2.LINE_AA)
    return output


def result_summary(result: FrameResult) -> dict[str, int]:
    subjects = result.heads if result.heads is not None else result.riders
    statuses = [rider.helmet.status for rider in subjects]
    return {
        "motorcycles": sum(v.vehicle_type == "motorcycle" for v in result.vehicles),
        "bicycles": sum(v.vehicle_type == "bicycle" for v in result.vehicles),
        "riders": len(result.riders),
        "helmet": sum(status == HelmetStatus.HELMET for status in statuses),
        "no_helmet": sum(status == HelmetStatus.NO_HELMET for status in statuses),
        "unknown": sum(status == HelmetStatus.UNKNOWN for status in statuses),
        "motorcycle_riders": sum(p.motorcycle_status == "motorcycle" for p in result.persons),
        "no_motorcycle_match": sum(p.motorcycle_status == "no_motorcycle_match" for p in result.persons),
        "motorcycle_uncertain": sum(p.motorcycle_status == "uncertain" for p in result.persons),
        "detector_ran": int(result.detector_ran),
    }

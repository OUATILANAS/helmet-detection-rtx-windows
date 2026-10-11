#!/usr/bin/env python3
"""Run rider-only helmet inference on a local video."""

from __future__ import annotations

import argparse
from time import perf_counter

import cv2
from common import add_runtime_arguments, configured, make_pipeline, output_path

from helmet_rider.utils.timing import add_capture_time, draw_timed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video file path")
    parser.add_argument("--save", nargs="?", const="", help="Save annotated MP4")
    parser.add_argument("--show", action="store_true")
    add_runtime_arguments(parser)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        parser.error(f"Could not open video: {args.source}")
    config = configured(args)
    pipeline = make_pipeline(config)
    target = output_path(args.save, "annotated.mp4")
    writer = None
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        while True:
            capture_started = perf_counter()
            ok, frame = capture.read()
            capture_ms = (perf_counter() - capture_started) * 1000.0
            if not ok:
                break
            result = add_capture_time(pipeline.process(frame, temporal_consistency=True, timestamp_seconds=(capture.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps), capture_ms)
            annotated, result = draw_timed(frame, result, config)
            if target and writer is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (annotated.shape[1], annotated.shape[0]),
                )
            if writer:
                writer.write(annotated)
            if args.show:
                cv2.imshow("Helmet Rider Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    if target:
        print(f"Saved annotated video: {target}")


if __name__ == "__main__":
    main()

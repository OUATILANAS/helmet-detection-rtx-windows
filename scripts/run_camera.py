#!/usr/bin/env python3
"""Run live USB webcam or Raspberry Pi Camera inference."""

from __future__ import annotations

import argparse
import logging
from time import monotonic, perf_counter

import cv2
from common import add_runtime_arguments, configured, make_pipeline, output_path

from helmet_rider.camera.picamera import PiCameraSource
from helmet_rider.camera.webcam import OpenCVCamera
from helmet_rider.utils.timing import add_capture_time, draw_timed

LOGGER = logging.getLogger("helmet_rider.camera")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0", help="USB camera index or device path")
    parser.add_argument("--source", choices=("webcam", "picamera"))
    parser.add_argument("--save", nargs="?", const="", help="Save an annotated video file")
    parser.add_argument("--show", action="store_true", help="Show a local preview; press q to quit")
    add_runtime_arguments(parser)
    args = parser.parse_args()

    config = configured(args)
    camera_source = args.source or config.camera.source
    if camera_source == "picamera":
        camera = PiCameraSource(config.camera.width, config.camera.height, config.camera.fps)
    else:
        source = int(args.camera) if args.camera.isdigit() else args.camera
        camera = OpenCVCamera(
            source, config.camera.width, config.camera.height, config.camera.fps
        )
    pipeline = make_pipeline(config)
    target = output_path(args.save, "camera-output.mp4")
    writer = None
    LOGGER.info("camera opened; source=%s", camera_source)
    frame_index = 0
    try:
        while True:
            capture_started = perf_counter()
            ok, frame = camera.read()
            capture_ms = (perf_counter() - capture_started) * 1000.0
            if not ok or frame is None:
                LOGGER.warning("camera frame unavailable")
                break
            frame_index += 1
            if frame_index % (config.camera.frame_skip + 1):
                continue
            result = add_capture_time(pipeline.process(frame, temporal_consistency=True, timestamp_seconds=monotonic()), capture_ms)
            annotated, result = draw_timed(frame, result, config)
            if target and writer is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(target), cv2.VideoWriter_fourcc(*"mp4v"), config.camera.fps,
                    (annotated.shape[1], annotated.shape[0]),
                )
            if writer:
                writer.write(annotated)
            if args.show:
                cv2.imshow("Helmet Rider Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        LOGGER.info("camera stopped")
    finally:
        camera.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    if target:
        print(f"Saved annotated camera video: {target}")


if __name__ == "__main__":
    main()

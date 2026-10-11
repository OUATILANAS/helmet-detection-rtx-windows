#!/usr/bin/env python3
"""Run rider-only helmet inference on one local image."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import cv2
from common import add_runtime_arguments, configured, make_pipeline, output_path

from helmet_rider.utils.results import result_to_dict
from helmet_rider.utils.timing import add_capture_time, draw_timed
from helmet_rider.visualization.draw import result_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image file path")
    parser.add_argument("--save", nargs="?", const="", help="Save annotated image")
    parser.add_argument("--json", nargs="?", const="", help="Save structured JSON result")
    parser.add_argument("--show", action="store_true", help="Display the annotated image")
    add_runtime_arguments(parser)
    args = parser.parse_args()

    capture_started = perf_counter()
    frame = cv2.imread(args.source)
    capture_ms = (perf_counter() - capture_started) * 1000.0
    if frame is None:
        parser.error(f"Could not read image: {args.source}")
    config = configured(args)
    pipeline = make_pipeline(config)
    result = add_capture_time(pipeline.process(frame), capture_ms)
    annotated, result = draw_timed(frame, result, config)

    print(json.dumps({"summary": result_summary(result),
                      "timings": result_to_dict(result)["timings"]}, indent=2))
    image_path = output_path(args.save, "annotated.png")
    if image_path:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), annotated):
            raise OSError(f"Could not save annotated image: {image_path}")
        print(f"Saved: {image_path}")
    json_path = output_path(args.json, "result.json")
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result_to_dict(result), indent=2), encoding="utf-8")
        print(f"Saved JSON: {json_path}")
    if args.show:
        cv2.imshow("Helmet Rider Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

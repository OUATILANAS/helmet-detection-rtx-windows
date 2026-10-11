import numpy as np

from helmet_rider.config import HeadROIConfig
from helmet_rider.helmet.head_roi import HeadROIExtractor


def test_extracts_expected_top_region():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop, bbox = HeadROIExtractor().extract(frame, (100, 100, 200, 400))
    assert crop.shape[0] == 75
    assert bbox == (114.0, 100.0, 186.0, 175.0)


def test_clips_roi_near_image_edges():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    crop, bbox = HeadROIExtractor().extract(frame, (-20, -10, 50, 120))
    assert bbox[0] == 0 and bbox[1] == 0
    assert bbox[2] <= frame.shape[1] and bbox[3] <= frame.shape[0]
    assert crop.size > 0


def test_small_and_invalid_person_boxes_return_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    extractor = HeadROIExtractor(HeadROIConfig(minimum_size=8))
    assert extractor.extract(frame, (10, 10, 13, 20)) is None
    assert extractor.extract(frame, (30, 30, 20, 40)) is None


def test_empty_frame_returns_none():
    assert HeadROIExtractor().extract(np.zeros((0, 0, 3), dtype=np.uint8), (1, 1, 5, 10)) is None

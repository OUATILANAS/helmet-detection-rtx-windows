from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pilot_crowdhuman_heads import detect


class FakeSession:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [SimpleNamespace(name='input', shape=[1, 3, 640, 640])]

    def run(self, _, feeds):
        assert feeds['input'].shape == (1, 3, 640, 640)
        return self.output


def test_separate_output_contract_undoes_letterbox_and_filters_body_class():
    session = FakeSession([
        np.array([[[320., 320., 64., 128.], [160., 320., 64., 128.]]]),
        np.array([[[.9], [.9]]]), np.array([[[1.], [0.]]])])
    detections = detect(session, np.zeros((100, 200, 3), np.uint8), 1, 'bgr')
    assert len(detections) == 1
    assert detections[0]['head_bbox'] == pytest.approx([90, 30, 110, 70])


def test_standard_single_class_output_preserves_box_geometry():
    session = FakeSession([np.array([[[320.], [320.], [64.], [128.], [.9]]])])
    detections = detect(session, np.zeros((100, 200, 3), np.uint8), 1, 'rgb')
    assert len(detections) == 1
    assert detections[0]['head_bbox'] == pytest.approx([90, 30, 110, 70])


def test_unknown_output_contract_is_rejected():
    session = FakeSession([np.zeros((1, 6, 8400))])
    with pytest.raises(ValueError, match='output contract'):
        detect(session, np.zeros((100, 200, 3), np.uint8), 1, 'rgb')

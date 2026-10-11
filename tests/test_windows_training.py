"""Check leakage rejection, class semantics and content-bound preflight inputs."""

import pytest
import yaml
from PIL import Image

from training.rtx_train import audit


def fixture(tmp_path):
    for split, color in [('train', 'red'), ('val', 'blue')]:
        (tmp_path/'images'/split).mkdir(parents=True)
        (tmp_path/'labels'/split).mkdir(parents=True)
        Image.new('RGB', (32, 32), color).save(tmp_path/'images'/split/'one.png')
        (tmp_path/'labels'/split/'one.txt').write_text('0 0.5 0.5 0.3 0.3\n')
    data = tmp_path/'dataset.yaml'
    data.write_text(yaml.safe_dump({'train': 'images/train', 'val': 'images/val',
                                   'names': {0: 'helmet', 1: 'no_helmet'}}))
    return data


def test_audit_does_not_require_or_open_test(tmp_path):
    data = fixture(tmp_path)
    rows, before = audit(data)
    assert len(rows['train']) == len(rows['val']) == 1
    (tmp_path/'labels/train/one.txt').write_text('1 0.5 0.5 0.3 0.3\n')
    assert audit(data)[1] != before


def test_audit_rejects_cross_split_duplicate(tmp_path):
    data = fixture(tmp_path)
    (tmp_path/'images/val/one.png').write_bytes((tmp_path/'images/train/one.png').read_bytes())
    with pytest.raises(ValueError, match='overlap'):
        audit(data)


def test_audit_rejects_physical_head_only_data(tmp_path):
    data = fixture(tmp_path)
    cfg = yaml.safe_load(data.read_text())
    cfg['names'] = {0: 'physical_head'}
    data.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='physical-head-only'):
        audit(data)

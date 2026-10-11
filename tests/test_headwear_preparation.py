import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "training/prepare_headwear_finetuning.py"
spec = importlib.util.spec_from_file_location("headwear_preparation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_standalone_hat_and_hat_below_head_are_rejected():
    assert module.worn_head_box([30, 10, 60, 25], [], 100, 200) is None
    assert module.worn_head_box([30, 120, 60, 140], [[10, 0, 90, 190, .9]], 100, 200) is None


def test_worn_hat_crop_includes_face_and_stays_in_image():
    box = module.worn_head_box([30, 10, 60, 25], [[10, 5, 90, 190, .9]], 100, 200)
    assert box is not None
    assert box[3] > 25 and box[0] < 30 and box[2] > 60
    assert 0 <= box[0] < box[2] <= 100
    assert 0 <= box[1] < box[3] <= 200


def test_canonical_hash_groups_rotated_and_flipped_copies():
    frame = np.random.default_rng(17).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    assert module.phash(frame) == module.phash(np.rot90(frame))
    assert module.phash(frame) == module.phash(np.fliplr(frame))


def test_augmented_export_filenames_share_original_identity():
    assert module.original_key("cap123_jpg.rf.abcdef.jpg") == "cap123_jpg"
    assert module.original_key("cap123_jpg.rf.123456.jpg") == "cap123_jpg"

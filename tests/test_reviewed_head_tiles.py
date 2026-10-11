from training.export_reviewed_head_tiles import reviewed_tiles, tile_labels


def test_unknown_helmet_status_still_trains_physical_head():
    row = {"samples": [{"label": "unknown", "physical_head": True,
                        "bbox": [10, 20, 30, 40]}]}
    assert tile_labels(row, [0, 0, 100, 100]) == [[0, 0.2, 0.3, 0.2, 0.2]]


def test_unresolved_region_prevents_false_negative_training():
    row = {"samples": [], "ignore_regions": [[10, 10, 20, 20]]}
    assert tile_labels(row, [0, 0, 100, 100]) is None
    assert tile_labels(row, [30, 30, 100, 100]) == []


def test_partial_head_is_not_silently_dropped_into_background():
    row = {"samples": [{"bbox": [90, 20, 110, 40]}]}
    assert tile_labels(row, [0, 0, 100, 100]) is None


def test_head_near_ignored_region_gets_safe_smaller_crop():
    row = {"image_size": [1000, 1000],
           "samples": [{"bbox": [200, 200, 220, 220]}],
           "ignore_regions": [[330, 0, 600, 1000]]}
    positives = [box for box in reviewed_tiles(row, 384) if tile_labels(row, box)]
    assert any(box[2] - box[0] < 384 for box in positives)
    assert all(box[2] <= 330 for box in positives)
    assert any(box[0] <= 200 and box[1] <= 200 and box[2] >= 220
               and box[3] >= 220 for box in positives)

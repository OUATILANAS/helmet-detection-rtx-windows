from scripts.report_strict_improvement import matched_classification, membership


def test_legacy_vehicle_type_normalization_retains_missed_subjects():
    row = {'samples': [
        {'bbox': [0, 0, 20, 20], 'motorcycle_status': 'motorcycle'},
        {'bbox': [30, 0, 50, 20], 'motorcycle_status': 'no_motorcycle'},
        {'bbox': [60, 0, 80, 20], 'motorcycle_status': 'motorcycle'}],
        'result': {'heads': None, 'riders': [
            {'head_bbox': [0, 0, 20, 20], 'vehicle_type': 'motorcycle'},
            {'head_bbox': [30, 0, 50, 20], 'vehicle_type': 'bicycle'}]}}
    assert membership([row]) == {
        'motorcycle': {'motorcycle': 1, 'missed': 1},
        'no_motorcycle': {'no_motorcycle': 1}}


def test_conditional_classification_diagnostic_does_not_hide_misses():
    row = {'samples': [
        {'bbox': [0, 0, 20, 20], 'label': 'helmet'},
        {'bbox': [30, 0, 50, 20], 'label': 'no_helmet'}],
        'result': {'heads': [
            {'head_bbox': [0, 0, 20, 20], 'helmet': {'status': 'helmet'}}]}}
    assert matched_classification([row]) == {
        'matched': 1, 'correct': 1, 'missed': 1,
        'accuracy_conditional_on_matched_head': 1.0}

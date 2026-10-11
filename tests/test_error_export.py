from scripts.export_detection_errors import export_errors


def test_error_export_excludes_unknown_truth_and_preserves_human_labels():
    records = [{'id': 'one', 'image': 'scene.jpg', 'sequence': 'camera-a',
                'samples': [{'label': 'helmet', 'bbox': [1, 2, 9, 10]},
                            {'label': 'unknown', 'bbox': [11, 2, 19, 10]},
                            {'label': 'no_helmet', 'bbox': [21, 2, 29, 10]}]}]
    metrics = {'frames': [{'id': 'one', 'traces': [
        {'truth': 'helmet', 'predicted': 'unknown', 'sample_index': 0,
         'prediction_index': 0, 'iou': .6}],
        'predictions': [{'head_bbox': [1, 3, 9, 11], 'evidence': {'confidence': .6}}],
        'missed_sample_indices': [1, 2]}]}
    result = export_errors(metrics, records)
    assert result['count'] == 2
    assert result['errors'][0]['reviewed_label'] == 'helmet'
    assert result['errors'][0]['kind'] == 'unknown'
    assert result['errors'][1]['reviewed_label'] == 'no_helmet'
    assert result['errors'][1]['kind'] == 'missed_match'
    assert records[0]['samples'][0]['label'] == 'helmet'

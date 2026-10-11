import pytest

from helmet_rider.evaluation.strict_heads import motorcycle_membership_score, strict_score_frame


def prediction(box, label='helmet'):
    return {'head_bbox':box, 'helmet':{'status':label}}


def test_strict_requires_complete_annotation():
    with pytest.raises(ValueError, match='complete'):
        strict_score_frame([],[])


def test_background_and_duplicate_detections_count_as_false_positives():
    p = [prediction([0,0,20,20]), prediction([0,0,20,20]), prediction([40,40,60,60],'no_helmet')]
    result = strict_score_frame(p,[{'bbox':[0,0,20,20],'label':'helmet'}],fully_reviewed=True)
    assert result['confusion']['helmet'] == {'helmet':1}
    assert result['unmatched_predictions'] == {'helmet':1,'no_helmet':1}


def test_uncertain_head_cannot_steal_known_match_and_only_one_is_ignored():
    p = [prediction([0,0,20,20]),prediction([30,0,50,20]),prediction([30,0,50,20])]
    truth = [{'bbox':[0,0,20,20],'label':'helmet'}, {'bbox':[0,0,20,20],'label':'unknown'},
             {'bbox':[30,0,50,20],'label':'unknown'}]
    result = strict_score_frame(p,truth,fully_reviewed=True)
    assert result['confusion']['helmet'] == {'helmet':1}
    assert result['unmatched_predictions'] == {'helmet':1}
    assert result['strict_scope']['ignored_predictions_on_unresolved_heads'] == 1


def test_small_heads_have_explicit_scope_and_abstentions_remain_misses():
    p = [prediction([0,0,7,7]),prediction([30,0,50,20],'unknown')]
    truth = [{'bbox':[0,0,7,7],'label':'helmet'},{'bbox':[30,0,50,20],'label':'no_helmet'}]
    result = strict_score_frame(p,truth,fully_reviewed=True)
    assert result['confusion']['no_helmet'] == {'unknown':1}
    assert result['strict_scope']['subminimum_predictions'] == 1


def test_motorcycle_membership_retains_misses_and_maps_runtime_status():
    p = [{**prediction([0,0,20,20]),'motorcycle_status':'no_motorcycle_match'}]
    truth = [{'bbox':[0,0,20,20],'motorcycle_status':'no_motorcycle'},
             {'bbox':[30,0,50,20],'motorcycle_status':'motorcycle'}]
    result = motorcycle_membership_score(p,truth)
    assert result == {'motorcycle':{'missed':1},'no_motorcycle':{'no_motorcycle':1}}

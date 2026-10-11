import numpy as np
import pytest

from training.calibrate_classifier import calibrate, probabilities


def test_calibration_changes_confidence_without_reversing_classes():
    logits=np.array([[4.,0.],[0.,4.],[3.,0.],[0.,3.],[2.,0.],[0.,2.]])
    labels=np.array([0,1,0,1,1,0])
    result=calibrate(logits,labels,minimum_negative_precision=.5)
    assert result['nll_after'] <= result['nll_before']
    assert np.array_equal(probabilities(logits,1.).argmax(1),probabilities(logits,result['temperature']).argmax(1))


def test_calibration_rejects_one_class_review_data():
    with pytest.raises(ValueError,match='both'):
        calibrate([[1.,0.],[2.,0.]],[0,0])

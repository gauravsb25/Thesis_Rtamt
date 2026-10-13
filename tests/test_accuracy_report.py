import numpy as np
from show_accuracy import summarize

def test_balanced_accuracy_does_not_follow_majority_class():
    truth=np.array(['LEFT','RIGHT','DOWN']+['STRAIGHT']*9)
    pred=np.array(['STRAIGHT']*12)
    result=summarize(truth,pred)
    assert result['accuracy']==.75 and result['balanced_accuracy']==.25
    assert result['correct']==9 and result['errors']==3

def test_absent_class_does_not_get_a_four_class_score():
    result=summarize(np.array(['LEFT']),np.array(['LEFT']))
    assert result['accuracy']==1 and result['balanced_accuracy'] is None

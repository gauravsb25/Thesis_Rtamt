from itertools import product
import numpy as np
from monitoring.rtamt_persistence import RtamtDecisionPersistence
from monitoring.live_session import LiveStreamSession


def test_actual_rtamt_matches_independent_finite_history_oracle():
    sequence = [label for seq in product(('LEFT','RIGHT','DOWN','STRAIGHT','INVALID'), repeat=3) for label in seq]
    for horizon in (1, 3, 5):
        monitor = RtamtDecisionPersistence(horizon)
        for i, label in enumerate(sequence):
            result, rho = monitor.update(label)
            tail = sequence[max(0, i-horizon+1):i+1]
            expected = label if i+1 >= horizon and len(set(tail)) == 1 and label != 'INVALID' else None
            assert result == expected
            assert rho == {c:min(1.0 if v == c else -1.0 for v in tail) for c in monitor.classes}


def test_live_rtamt_gap_restarts_temporal_and_eeg_history():
    class Constant:
        def classify_window(self, x):
            return 'DOWN'
    live = LiveStreamSession(monitor=Constant(), temporal_backend='rtamt', decide_every=16, confirmations=3)
    for i in range(800):
        out = live.push(np.ones(8), i/256)
    assert out['confirmed_label'] == 'DOWN'
    assert out['persistence_robustness']['DOWN'] == 1
    assert live.push(np.ones(8), 5)['status'] == 'GAP_RESET'
    assert live.temporal.tick == 0

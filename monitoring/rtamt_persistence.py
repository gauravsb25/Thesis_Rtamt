"""Actual RTAMT online STL checks over emitted window decisions.

This verifies decision persistence, not EEG class accuracy. Robustness is +/-1
label agreement, not a probability or calibrated physiological margin.
"""
from pathlib import Path
import sys


class RtamtDecisionPersistence:
    classes = ('LEFT', 'RIGHT', 'DOWN', 'STRAIGHT')

    def __init__(self, confirmations=3):
        if not isinstance(confirmations, int) or confirmations < 1:
            raise ValueError('confirmations must be a positive integer')
        try:
            import rtamt
        except ModuleNotFoundError as exc:
            if exc.name != 'rtamt':
                raise
            bundled = Path(__file__).resolve().parents[1]/'.runtime/rtamt'
            if not (bundled/'rtamt').is_dir():
                raise ImportError('RTAMT is required for temporal_backend="rtamt"') from exc
            sys.path.insert(0, str(bundled))
            import rtamt
        self.confirmations = confirmations
        self.specs = {}
        for label in self.classes:
            spec = rtamt.StlDiscreteTimeSpecification()
            spec.declare_var('evidence', 'float')
            spec.spec = f'historically[0:{confirmations-1}](evidence > 0)'
            spec.parse()
            self.specs[label] = spec
        self.tick = 0

    def update(self, label):
        if label not in (*self.classes, 'INVALID'):
            raise ValueError('unexpected window label')
        rho = {name: float(spec.update(self.tick, [['evidence', 1.0 if label == name else -1.0]]))
               for name, spec in self.specs.items()}
        self.tick += 1
        positive = [name for name, value in rho.items() if value > 0]
        confirmed = positive[0] if self.tick >= self.confirmations and len(positive) == 1 else None
        return confirmed, rho

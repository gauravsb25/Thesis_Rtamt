"""Four RTAMT monitors, one per head movement.

Each class is a Signal Temporal Logic specification evaluated by RTAMT
(https://github.com/nickovic/rtamt, v0.4.10, bundled under ``.runtime/rtamt``):

    from monitoring import HeadMovementMonitor
    monitor = HeadMovementMonitor()
    monitor.classify(window)       # (8, 768) microvolts -> LEFT / RIGHT / DOWN / STRAIGHT / INVALID
    monitor.classify(window, history)   # history: the samples just before the window (stream-background gate)
    monitor.robustness(window)     # RTAMT robustness of each of the four specifications
    monitor.formulas               # the four STL specifications

    from monitoring import StraightMonitor, DownMonitor, LeftMonitor, RightMonitor
    from monitoring import StreamMonitor, LiveStreamSession     # live streaming
    from monitoring import RowStreamMonitor                      # a verdict at every incoming row

To see the four monitors and run them on real windows:  python -m monitoring
"""

from .rtamt_signals import (
    CHANNELS,
    FS,
    SIGNALS,
    WINDOW_SAMPLES,
    InvalidWindow,
    RtamtConfig,
    compute_signals,
    stream_background,
)
from .rtamt_monitors import (
    CLASSES,
    ClassMonitor,
    DownMonitor,
    HeadMovementMonitor,
    LeftMonitor,
    RightMonitor,
    RtamtDownMonitor,
    RtamtHeadMovementMonitor,
    RtamtLeftMonitor,
    RtamtRightMonitor,
    RtamtStraightMonitor,
    StraightMonitor,
    StreamMonitor,
    class_formulas,
    satisfied_class,
)
from .rtamt_persistence import RtamtDecisionPersistence
from .row_stream import RowStreamMonitor, RowVerdict, row_formulas
from .live_session import LiveStreamSession
from .stream_history import windows_with_history

__all__ = [
    "CLASSES", "CHANNELS", "FS", "SIGNALS", "WINDOW_SAMPLES",
    "RtamtConfig", "InvalidWindow", "compute_signals", "stream_background", "windows_with_history",
    "HeadMovementMonitor", "StraightMonitor", "DownMonitor", "LeftMonitor", "RightMonitor",
    "ClassMonitor", "StreamMonitor", "LiveStreamSession", "RtamtDecisionPersistence",
    "RowStreamMonitor", "RowVerdict", "row_formulas",
    "class_formulas", "satisfied_class",
    "RtamtHeadMovementMonitor", "RtamtStraightMonitor", "RtamtDownMonitor",
    "RtamtLeftMonitor", "RtamtRightMonitor",
]

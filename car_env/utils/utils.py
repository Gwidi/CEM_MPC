from typing import NamedTuple
import numpy as np


Info = NamedTuple(
    "Info",
    [
        ("avg_speed", float),
        ("off_track", float),
        ("reward", float),
    ],
)

Track = NamedTuple(
    "Track",
    [
        ("s", np.ndarray),
        ("x", np.ndarray),
        ("y", np.ndarray),
        ("width", np.ndarray),
        ("curvature", np.ndarray),
        ("heading", np.ndarray),
    ],
)

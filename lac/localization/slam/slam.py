import numpy as np

from dataclasses import dataclass


@dataclass
class MapPoint:
    xyz: np.ndarray
    descriptor: np.ndarray
    label: str

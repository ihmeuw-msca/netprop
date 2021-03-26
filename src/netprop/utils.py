"""
Utility functions
"""
from typing import List, Iterable
import numpy as np


def sizes_to_slices(sizes: Iterable) -> List[slice]:
    ends = np.cumsum(sizes)
    starts = np.insert(ends, 0, 0)[:-1]
    return [slice(*pair) for pair in zip(starts, ends)]

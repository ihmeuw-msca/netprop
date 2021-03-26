from typing import Iterable
from numpy import ndarray
from netprop.data import Data


class DormModel:
    """
    Definition or method model
    """

    def __init__(self,
                 name: str,
                 covs: Iterable[str]):
        self.name = name
        self.covs = list(covs)

    @property
    def size(self) -> int:
        return len(self.covs)

    def get_mat(self, data: Data) -> ndarray:
        return data[self.covs]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}, covs={self.covs})"

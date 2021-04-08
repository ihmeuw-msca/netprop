from typing import Dict, Iterable, Union
from operator import attrgetter
import numpy as np
from numpy import ndarray
from netprop.data import Data


class DormModel:
    """
    Definition or method model
    """

    def __init__(self,
                 name: str,
                 covs: Iterable[str],
                 uprior: Dict[str, Iterable[float]] = None,
                 gprior: Dict[str, Iterable[float]] = None):
        self.name = name
        self.covs = list(covs)
        self.uprior = uprior
        self.gprior = gprior

    @property
    def size(self) -> int:
        return len(self.covs)

    def get_prior(self,
                  prior_info: Union[Dict[str, Iterable[float]], None],
                  default_prior: Iterable[float]) -> ndarray:
        prior = np.repeat(np.asarray(default_prior)[:, None], self.size, axis=1)
        if prior_info is not None:
            for k, v in prior_info.items():
                if k in self.covs:
                    prior[:, self.covs.index(k)] = v
        return prior

    uprior = property(attrgetter("_uprior"))

    @uprior.setter
    def uprior(self, uprior_info: Union[Dict[str, Iterable[float]], None]):
        default_uprior = [-np.inf, np.inf]
        if uprior_info is not None:
            for p in uprior_info.values():
                assert p[0] <= p[1], "Uniform prior lower bound <= upper bound."
        self._uprior = self.get_prior(uprior_info, default_uprior)

    gprior = property(attrgetter("_gprior"))

    @gprior.setter
    def gprior(self, gprior_info: Union[Dict[str, Iterable[float]], None]):
        default_gprior = [0.0, np.inf]
        if gprior_info is not None:
            for p in gprior_info.values():
                assert p[1] > 0, "Gaussian prior sd must be positive."
        self._gprior = self.get_prior(gprior_info, default_gprior)

    def get_mat(self, data: Data) -> ndarray:
        return data[self.covs]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}, covs={self.covs})"

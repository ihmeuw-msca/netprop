from typing import Iterable
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from netprop.data import Data, DormColumn
from netprop.dorm_model import DormModel
from netprop.utils import sizes_to_slices


class Model:
    """
    Model include data and dorm model
    """

    def __init__(self,
                 data: Data,
                 dorm_models: Iterable[DormModel],
                 gold_dorm: str):

        self.data = data
        self.dorm_models = {
            dorm_model.name: dorm_model
            for dorm_model in dorm_models
        }
        self.gold_dorm = gold_dorm

        self.dorms = list(self.data.unique_dorms)
        self.dorm_model_sizes = [self.dorm_models[name].size
                                 for name in self.dorms]
        slices = sizes_to_slices(self.dorm_model_sizes)
        self.dorm_model_index = {
            name: slices[i]
            for i, name in enumerate(self.dorms)
        }
        self.size = sum(self.dorm_model_sizes)

        self.ref_dorm_weights = self.get_dorm_weights(self.data.ref_dorm)
        self.alt_dorm_weights = self.get_dorm_weights(self.data.alt_dorm)

        self.dorm_model_mats = {
            name: self.dorm_models[name].get_mat(self.data)
            for name in self.dorms
        }

        self.bounds = np.repeat(np.array([[-np.inf, np.inf]]), self.size, axis=0)
        self.bounds[self.dorm_model_index[self.gold_dorm]] = 0.0
        self.soln = None
        self.beta = np.zeros(self.size)

    def get_dorm_weights(self, col: DormColumn) -> ndarray:
        values = col.values
        weights = np.zeros((len(values), len(self.dorms)))
        for i, dorms in enumerate(values):
            for dorm in dorms:
                weights[i, self.dorms.index(dorm)] = 1.0
        return weights

    def get_dorm_values(self, beta: ndarray) -> ndarray:
        return np.exp(np.hstack([
            self.dorm_model_mats[name].dot(beta[self.dorm_model_index[name]])[:, None]
            for name in self.dorms
        ]))

    def objective(self, beta: ndarray) -> float:
        dorm_values = self.get_dorm_values(beta)
        ref_pred = np.log(np.sum(self.ref_dorm_weights*dorm_values, axis=1))
        alt_pred = np.log(np.sum(self.alt_dorm_weights*dorm_values, axis=1))
        pred = alt_pred - ref_pred

        residual = self.data.obs.values - pred
        se = self.data.obs_se.values
        return 0.5*np.sum(residual**2/se**2)

    def gradient(self, beta: ndarray) -> float:
        dorm_values = self.get_dorm_values(beta)
        ref_values = self.ref_dorm_weights*dorm_values
        alt_values = self.alt_dorm_weights*dorm_values
        ref_pred = np.log(ref_values.sum(axis=1))
        alt_pred = np.log(alt_values.sum(axis=1))
        pred = alt_pred - ref_pred

        residual = self.data.obs.values - pred
        se = self.data.obs_se.values
        alpha = -residual/se**2

        ref_values /= ref_values.sum(axis=1)[:, None]
        alt_values /= alt_values.sum(axis=1)[:, None]

        weights_mat = alpha*((alt_values - ref_values).T)
        gradient_mat = np.hstack([
            weights_mat[i][:, None]*self.dorm_model_mats[name]
            for i, name in enumerate(self.dorms)
        ])

        return gradient_mat.sum(axis=0)

    def fit_model(self, beta: ndarray = None, **fit_options):
        if beta is None:
            beta = self.beta.copy()

        self.soln = minimize(self.objective,
                             beta,
                             jac=self.gradient,
                             bounds=self.bounds,
                             **fit_options)
        self.beta = self.soln.x

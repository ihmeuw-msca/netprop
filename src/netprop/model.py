from typing import Iterable
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import minimize
from netprop.data import Data, DormColumn
from netprop.dorm_model import DormModel
from netprop.utils import sizes_to_index


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
        index = sizes_to_index(self.dorm_model_sizes)
        self.dorm_model_index = {
            name: index[i]
            for i, name in enumerate(self.dorms)
        }
        self.var_index = np.hstack([
            list(self.dorm_model_index[name])
            for name in self.dorms if name != self.gold_dorm
        ])
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
        self.beta_vcov = None

    def get_dorm_weights(self, col: DormColumn) -> ndarray:
        values = col.values
        weights = np.zeros((len(values), len(self.dorms)))
        for i, dorms in enumerate(values):
            for dorm in dorms:
                weights[i, self.dorms.index(dorm)] = 1.0
        return weights

    def get_dorm_values(self, beta: ndarray, dorm_model_mats=None) -> ndarray:
        if dorm_model_mats is None:
            dorm_model_mats = self.dorm_model_mats
        return np.exp(np.hstack([
            dorm_model_mats[name].dot(beta[self.dorm_model_index[name]])[:, None]
            for name in self.dorms
        ]))

    def objective(self, beta: ndarray) -> float:
        beta = np.asarray(beta)
        dorm_values = self.get_dorm_values(beta)
        ref_pred = np.log(np.sum(self.ref_dorm_weights*dorm_values, axis=1))
        alt_pred = np.log(np.sum(self.alt_dorm_weights*dorm_values, axis=1))
        pred = alt_pred - ref_pred

        residual = self.data.obs.values - pred
        se = self.data.obs_se.values
        return 0.5*np.sum(residual**2/se**2)

    def gradient(self, beta: ndarray) -> ndarray:
        beta = np.asarray(beta)
        dorm_values = self.get_dorm_values(beta)
        ref_values = self.ref_dorm_weights*dorm_values
        alt_values = self.alt_dorm_weights*dorm_values
        pred = np.log(alt_values.sum(axis=1)) - np.log(ref_values.sum(axis=1))

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

    def jacobian2(self, beta: ndarray) -> ndarray:
        beta = np.asarray(beta)
        dorm_values = self.get_dorm_values(beta)
        ref_values = self.ref_dorm_weights*dorm_values
        alt_values = self.alt_dorm_weights*dorm_values
        ref_values /= ref_values.sum(axis=1)[:, None]
        alt_values /= alt_values.sum(axis=1)[:, None]
        alpha = 1.0/self.data.obs_se.values**2

        weights_mat = (alt_values - ref_values).T
        jacobian_mat = np.hstack([
            weights_mat[i][:, None]*self.dorm_model_mats[name]
            for i, name in enumerate(self.dorms)
        ])
        return (jacobian_mat.T*alpha).dot(jacobian_mat)

    def fit_model(self, beta: ndarray = None, **fit_options):
        if beta is None:
            beta = self.beta.copy()

        self.soln = minimize(self.objective,
                             beta,
                             method="trust-constr",
                             jac=self.gradient,
                             hess=self.jacobian2,
                             bounds=self.bounds,
                             **fit_options)
        self.beta = self.soln.x
        jacobian2 = self.jacobian2(self.beta)[self.var_index, self.var_index]
        jacobian2 = jacobian2.reshape(len(self.var_index), len(self.var_index))
        self.beta_vcov = np.zeros((self.size, self.size))
        self.beta_vcov[self.var_index, self.var_index] = np.linalg.inv(jacobian2)

    def predict(self,
                df: DataFrame,
                beta: ndarray = None) -> ndarray:
        if beta is None:
            beta = self.beta

        dorm_model_mats = {
            name: self.dorm_models[name].get_mat(df).values
            for name in self.dorms
        }

        dorm_values = self.get_dorm_values(beta, dorm_model_mats)

        return dorm_values/np.sum(dorm_values, axis=1)[:, None]

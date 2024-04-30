import numpy as np
from sklearn.metrics import mutual_info_score
from abc import ABC, abstractmethod


class FeatureSelector(ABC):
    def __init__(
        self,
        data_frame,
        unique_th=50,
        stopping_criterium: float = None,
        stopping_n_features: int = None,
    ) -> None:
        self.data_frame = data_frame.copy()
        self._data_frame_np = data_frame.copy().to_numpy()
        for col_id in range(data_frame.shape[1]):
            if len(np.unique(self._data_frame_np[:, col_id])) > unique_th:
                self._data_frame_np[:, col_id] = discretize(
                    self._data_frame_np[:, col_id]
                )
        self._features: list[int] = []
        self._features_path: list = []
        self.numeric_criterium_path: list[list] = []
        self.stopping_criterium = stopping_criterium
        self.stopping_n_features = (
            stopping_n_features
            if stopping_n_features is not None
            else data_frame.shape[1] - 1
        )
        self.time: float = np.nan
        self.time_path: list[float] = []

    @property
    def features(self):
        return self.data_frame.columns[self._features]

    @property
    def features_path(self):
        return [
            list(self.data_frame.columns[features]) for features in self._features_path
        ]

    @abstractmethod
    def run_fs(self):
        pass


def discretize(vec: np.array, n_bins: int = 10):
    vec_bins = np.linspace(vec.min(), vec.max(), n_bins)
    vec_discrete = np.digitize(vec, bins=vec_bins)
    return vec_discrete


def conditional_mutual_information(X, Y, Z):
    z_values = np.unique(Z)
    n_z_values = len(z_values)
    n = len(Z)

    cmi = 0

    for i in range(n_z_values):
        z_value_tmp = z_values[i]
        z_condition = Z == z_value_tmp

        X_z = X[z_condition]
        Y_z = Y[z_condition]

        mi_XY_z = mutual_info_score(X_z, Y_z)
        p_z = np.sum(z_condition) / n

        cmi += p_z * mi_XY_z

    return cmi

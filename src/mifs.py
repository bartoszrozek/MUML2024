from src.feature_selector_step import FeatureSelectorStep
from sklearn.metrics import mutual_info_score
import numpy as np


class MIFS(FeatureSelectorStep):
    def __init__(
        self,
        data_frame,
        unique_th=50,
        stopping_criterium: float = None,
        stopping_n_features: int = None,
        beta: int = 1,
    ) -> None:
        self.beta = beta
        super().__init__(data_frame, unique_th, stopping_criterium, stopping_n_features)

    def select_next_feature(self):
        y = self._data_frame_np[:, 0]
        if len(self._features) == 0:
            values = {
                col_idx: mutual_info_score(y, self._data_frame_np[:, col_idx])
                for col_idx in range(1, self._data_frame_np.shape[1])
            }
            return {
                "idx": max(values, key=values.get),
                "criterium": max(values.values()),
            }
        else:
            values = {
                col_idx: mutual_info_score(y, self._data_frame_np[:, col_idx])
                - self.second_term(col_idx)
                for col_idx in range(1, self._data_frame_np.shape[1])
                if col_idx not in self._features
            }
            return {
                "idx": max(values, key=values.get),
                "criterium": max(values.values()),
            }

    def second_term(self, new_feature_idx):
        features_idxs = self._features
        return self.beta * np.sum(
            [
                mutual_info_score(
                    self._data_frame_np[:, new_feature_idx],
                    self._data_frame_np[:, feature_id],
                )
                for feature_id in features_idxs
            ]
        )

from src.feature_selector import conditional_mutual_information
from src.feature_selector_step import FeatureSelectorStep
from sklearn.metrics import mutual_info_score
import numpy as np


class MiniMax(FeatureSelectorStep):
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
                new_col_idx: self.calculate_min(y, new_col_idx)
                for new_col_idx in range(1, self._data_frame_np.shape[1])
                if new_col_idx not in self._features
            }

            return {
                "idx": max(values, key=values.get),
                "criterium": max(values.values()),
            }

    def calculate_min(self, y, new_col_idx):
        return np.min(
            [
                conditional_mutual_information(
                    y,
                    self._data_frame_np[:, new_col_idx],
                    self._data_frame_np[:, col_idx],
                )
                for col_idx in self._features
            ]
        )

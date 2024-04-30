from feature_selector import conditional_mutual_information
from feature_selector_step import FeatureSelectorStep
from sklearn.metrics import mutual_info_score
import numpy as np


class JMI(FeatureSelectorStep):
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
        return np.sum(
            [
                mutual_info_score(
                    self._data_frame_np[:, new_feature_idx],
                    self._data_frame_np[:, feature_id],
                )
                - conditional_mutual_information(
                    self._data_frame_np[:, new_feature_idx],
                    self._data_frame_np[:, feature_id],
                    self._data_frame_np[:0],
                )
                for feature_id in features_idxs
            ]
        ) / len(features_idxs)

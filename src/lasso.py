from src.feature_selector_arbitrary import FeatureSelectorArbitrary
import numpy as np
import time
from sklearn.linear_model import LassoCV
from itertools import compress


class LassoFS(FeatureSelectorArbitrary):
    def run_fs(self):
        start = time.time()
        row_sums = self._data_frame_np.sum(axis=1)
        self._data_frame_np = self._data_frame_np / row_sums[:, np.newaxis]
        X = self._data_frame_np[:, 1:]
        y = self._data_frame_np[:, 0]
        reg = LassoCV(cv=5, random_state=0).fit(X, y)
        features = list(
            compress(list(range(1, self._data_frame_np.shape[1])), reg.coef_ != 0)
        )
        self.time = time.time() - start
        self._features = features

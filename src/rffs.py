from src.feature_selector_arbitrary import FeatureSelectorArbitrary
import numpy as np
import time
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel


class RandomForestFS(FeatureSelectorArbitrary):
    def run_fs(self, verbose=False):
        start = time.time()
        row_sums = self._data_frame_np.sum(axis=1)
        self._data_frame_np = self._data_frame_np / row_sums[:, np.newaxis]
        X = self._data_frame_np[:, 1:]
        y = self._data_frame_np[:, 0]
        clf = ExtraTreesRegressor(n_estimators=50)
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        features = [i + 1 for i in range(X.shape[1]) if model.get_support()[i - 1]]
        self.time = time.time() - start
        self._features = features

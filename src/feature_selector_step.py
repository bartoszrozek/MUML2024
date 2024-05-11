from src.feature_selector import FeatureSelector
from abc import abstractmethod
import time
from kneed import KneeLocator


class FeatureSelectorStep(FeatureSelector):
    @abstractmethod
    def select_next_feature(self):
        pass

    def run_fs(self):
        start = time.time()
        while len(self._features) < self.stopping_n_features:
            start_one = time.time()
            values = self.select_next_feature()
            self.time_path.append(time.time() - start_one)
            self._features.append(values["idx"])
            self._features_path.append(self._features.copy())
            self.numeric_statistic_path.append(values["criterium"])
            print("Feature selected: ", self.data_frame.columns[values["idx"]])
            print("Criterium value: ", values["criterium"])
            if len(self._features) > 1:
                kn = KneeLocator(
                    range(1, len(self.numeric_statistic_path) + 1),
                    self.numeric_statistic_path,
                    curve="convex",
                    direction="decreasing",
                )
                if self.stopping_criterium and kn.knee is not None:
                    break
        self.time = time.time() - start

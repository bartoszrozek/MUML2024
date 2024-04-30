from feature_selector import FeatureSelector
from abc import abstractmethod
import time


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
            self.numeric_criterium_path.append(values["criterium"])
            if (
                self.stopping_criterium is not None
                and self.numeric_criterium_path[-1] < self.stopping_criterium
            ):
                break
        self.time = time.time() - start

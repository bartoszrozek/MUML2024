# Just to keep structure

from feature_selector import FeatureSelector
from abc import abstractmethod


class FeatureSelectorArbitrary(FeatureSelector):
    @abstractmethod
    def run_fs(self):
        pass

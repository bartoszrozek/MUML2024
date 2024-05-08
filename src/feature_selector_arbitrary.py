# Just to keep structure

from src.feature_selector import FeatureSelector
from abc import abstractmethod


class FeatureSelectorArbitrary(FeatureSelector):
    @abstractmethod
    def run_fs(self):
        pass

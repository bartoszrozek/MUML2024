from statsmodels.tools import add_constant
import numpy as np
import pandas as pd


class Experiment:
    def __init__(
        self,
        fs_list,
        model,
        df,
        target_variable: str,
        sign_columns_names: list | None = None,
    ) -> None:
        self.fs_list = fs_list
        self.model = model
        self.df = df
        self.target_variable = target_variable
        self.results = {}
        self._results_artifical = {}
        self.sign_columns_names = sign_columns_names

    def fit_fs(self, verbose=False):
        for fs in self.fs_list:
            if verbose:
                print(f"----{type(fs).__name__}----")
            fs.run_fs(verbose=verbose)

        for features_set in self.fs_list:
            X = self.df[features_set.features]
            y = self.df[self.target_variable]
            model = self.model(y, add_constant(X)).fit()
            fs_result = {}
            fs_result["rsquared_adj"] = model.rsquared_adj
            fs_result["bic"] = model.bic
            fs_result["len"] = len(features_set.features)
            self.results[type(features_set).__name__] = fs_result

    def test_artifical(self):
        if self.sign_columns_names is not None:
            sign_names = self.sign_columns_names
        else:
            raise ValueError("Sigificance columns not passed!")
        for features_set in self.fs_list:
            sign_score = -np.sum(
                [x in sign_names["significant_names"] for x in features_set.features]
            )
            semisign_score = 0.5 * np.sum(
                [
                    x in sign_names["semi_significant_names"]
                    for x in features_set.features
                ]
            )
            nonsign_score = 2 * np.sum(
                [
                    x in sign_names["non_significant_names"]
                    for x in features_set.features
                ]
            )
            start_score = len(sign_names["significant_names"])
            all_ = np.sum(
                np.array([1, 0.5, 2])
                * np.array([len(v) for _, v in sign_names.items()])
            )
            score = (
                1 - (start_score + (sign_score + semisign_score + nonsign_score)) / all_
            )
            self._results_artifical[type(features_set).__name__] = score

    @property
    def results_artifical(self):
        if len(self.results.keys()) == 0:
            self.fit_fs()
        if len(self._results_artifical.keys()) == 0:
            self.test_artifical()
        return pd.DataFrame(self._results_artifical, index=[0])

    @property
    def n_variables(self):
        if len(self.results.keys()) == 0:
            self.fit_fs()
        return pd.DataFrame({k: v["len"] for k, v in self.results.items()}, index=[0])

    def print_results(self):
        for key in self.results.keys():
            fs_result = self.results[key]
            print(f"R-squared adjusted for {key} = {fs_result['rsquared_adj']} \
                  and bic = {fs_result['bic']} with {fs_result['len']} features.")


from statsmodels.tools import add_constant

class Experiment():
    def __init__(
        self,
        fs_list,
        model,
        df,
        target_variable: str,
    ) -> None:
        self.fs_list = fs_list
        self.model = model
        self.df = df
        self.target_variable = target_variable
        self.results = {}

    def fit_fs(self):
        for fs in self.fs_list:
            print(f"----{type(fs).__name__}----")
            fs.run_fs()

        for features_set in self.fs_list:
            X = self.df[features_set.features]
            y = self.df[self.target_variable]
            model = self.model(y, add_constant(X)).fit()
            fs_result = {}
            fs_result["rsquared_adj"] = model.rsquared_adj
            fs_result["bic"] = model.bic
            fs_result["len"] = len(features_set.features)
            self.results[type(features_set).__name__] = fs_result

    def print_results(self):
        for key in self.results.keys():
            fs_result = self.results[key]
            print(f"R-squared adjusted for {key} = {fs_result['rsquared_adj']} \
                  and bic = {fs_result['bic']} with {fs_result['len']} features.")

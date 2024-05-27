import numpy as np
import random
import pandas as pd


def new_column(y, corr, func):
    noise = np.random.normal(0, corr, len(y))
    return func(y + noise)


def draw_function():
    function_set = {
        "lin": lambda x: x,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "sqrt": lambda x: np.sqrt(np.abs(x)),
        "log": lambda x: np.log(np.abs(x)),
        "abs": lambda x: np.abs(x),
    }
    key = random.choice(list(function_set.keys()))
    return {"key": key, "func": function_set[key]}


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


class DataSetGenerator:
    def __init__(
        self, n: int, n_features: int, n_significant: int, n_semi_significant: int
    ):
        if n_features < n_significant:
            raise ValueError("n_features should be greater than n_significant")

        self.n = n
        self.n_features = n_features
        self.n_significant = n_significant
        self.n_semi_significant = n_semi_significant

    def set_1(
        self,
        n_features: int | None = None,
        n_significant: int | None = None,
        n_semi_significant: int | None = None,
        n: int | None = None,
    ):
        """Function that generates data with set number of significant features.
        Each new significant column is a transformed correlated y

        Args:
            n_features (_type_): _description_
            n_significant (_type_): _description_
            n (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        n_features = n_features if n_features is not None else self.n_features
        n_significant = (
            n_significant if n_significant is not None else self.n_significant
        )
        n_semi_significant = (
            n_semi_significant
            if n_semi_significant is not None
            else self.n_semi_significant
        )
        n = n if n is not None else self.n

        sign_names = {
            "significant_names": [],
            "semi_significant_names": [],
            "non_significant_names": [],
        }

        generator = np.random.default_rng()
        y = generator.standard_normal(n)
        columns = []
        names = ["y"]
        for _ in range(n_significant):
            corr = generator.uniform(0.3, 0.9)
            func = draw_function()
            columns.append(new_column(y, corr, func["func"]).reshape(n, 1))
            col_name = f"significant_{func['key']}_{np.round(corr,5)}"
            names.append(col_name)
            sign_names["significant_names"].append(col_name)
        for _ in range(n_significant, n_significant + n_semi_significant):
            corr = generator.uniform(0.3, 0.9)
            func = draw_function()
            idx = random.choice(list(range(n_significant)))
            columns.append(
                new_column(columns[idx].flatten(), corr, func["func"]).reshape(n, 1)
            )
            col_name = f"{names[idx+1]}_{np.round(corr,5)}"
            names.append(col_name)
            sign_names["semi_significant_names"].append(col_name)
        for i in range(n_significant + n_semi_significant, n_features):
            func = draw_function()
            columns.append(func["func"](generator.standard_normal(n)).reshape(n, 1))
            col_name = f"nsignificant_{func['key']}_{i+1}"
            names.append(col_name)
            sign_names["non_significant_names"].append(col_name)

        return (
            pd.DataFrame(
                np.concatenate((y.reshape(n, 1), *columns), axis=1), columns=names
            ),
            sign_names,
        )

    def set_2(
        self,
        n_features: int | None = None,
        n_significant: int | None = None,
        n_semi_significant: int | None = None,
        n: int | None = None,
        mu: int = 3,
        sigma2: int = 1,
        p: float = 0.5,
    ):
        """Function that generates data with set number of significant features.
        Significant columns represent dimensions of a hypercube.
        Half of each significant feature's observations are from distribution N(mu, sigma2)
        and the other are from N(-mu, sigma2).

        Points in clusters at each corner of the hypercube are assigned binary classes randomly
        with probability p.

        Semi-significant columns are obtained by transforming significant features.

        Args:
            n_features (int): number of
            n_significant (int): _description_
            n (int, optional): _description_. Defaults to 1000.
            mu (int): parameter of the distribution of significant functions
            sigma2 (int): parameter of the distribution of significant functions

        Returns:
            _type_: _description_
        """
        n_features = n_features if n_features is not None else self.n_features
        n_significant = (
            n_significant if n_significant is not None else self.n_significant
        )
        n_semi_significant = (
            n_semi_significant
            if n_semi_significant is not None
            else self.n_semi_significant
        )
        n = n if n is not None else self.n

        sign_names = {
            "significant_names": [],
            "semi_significant_names": [],
            "non_significant_names": [],
        }
        generator = np.random.default_rng()
        columns = []
        names = ["y"]

        significant_features = generator.normal(mu, sigma2, (n, n_significant))
        y = np.ones(n)

        n_vertices = 2**n_significant
        for i in range(n_vertices):
            lower = (n // n_vertices) * i
            upper = min((n // n_vertices) * (i + 1), n)
            significant_features[lower:upper] = significant_features[lower:upper] * (
                bin_array(i, n_significant) * 2 - 1
            )
            y[lower:upper] = y[lower:upper] * generator.binomial(1, p)
        for i in range(n_significant):
            columns.append(significant_features[:, i].reshape(n, 1))
            col_name = f"significant_{i}"
            names.append(col_name)
            sign_names["significant_names"].append(col_name)
        for _ in range(n_significant, n_significant + n_semi_significant):
            corr = generator.uniform(0.3, 0.9)
            func = draw_function()
            idx = random.choice(list(range(n_significant)))
            columns.append(
                new_column(columns[idx].flatten(), corr, func["func"]).reshape(n, 1)
            )
            col_name = f"{names[idx+1]}_{np.round(corr,5)}"
            names.append(col_name)
            sign_names["semi_significant_names"].append(col_name)
        for i in range(n_significant + n_semi_significant, n_features):
            func = draw_function()
            columns.append(func["func"](generator.standard_normal(n)).reshape(n, 1))
            col_name = f"nsignificant_{func['key']}_{i+1}"
            names.append(col_name)
            sign_names["non_significant_names"].append(col_name)

        return (
            pd.DataFrame(
                np.concatenate((y.reshape(n, 1), *columns), axis=1), columns=names
            ),
            sign_names,
        )

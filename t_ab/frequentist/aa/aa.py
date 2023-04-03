from typing import Iterable, Generator

import numpy as np
import pandas as pd
from scipy.stats import kstest
from statsmodels.stats.multitest import multipletests

from ..ab import ABTest


class AATest:
    def __init__(
        self,
        ab_test: ABTest,
        uniform_test_method: str = "ks",
        mcp_correction_method: str = "hs",
    ) -> None:
        assert uniform_test_method in {"ks"}
        self.ab_test = ab_test
        self.alpha = ab_test.alpha
        self.uniform_test_method = uniform_test_method
        self.mcp_correction_method = mcp_correction_method

    def __call__(
        self, dfs_loader: Generator[list[pd.DataFrame], None, None]
    ) -> pd.DataFrame:
        df = pd.concat([self.ab_test(dfs) for dfs in dfs_loader], ignore_index=True)
        pvalues = df.groupby(["metrics_base", "metrics", "pair"])["pvalue"].apply(list)
        uniform_pvalues = (
            pvalues.apply(self.uniform_test)
            .groupby(["metrics_base", "metrics"])
            .apply(list)
        )
        multipletests_result = uniform_pvalues.apply(self.multiple_test)
        df = pd.concat(
            [
                pvalues.groupby(["metrics_base", "metrics"])
                .apply(list)
                .rename("pvalues"),
                uniform_pvalues.rename("uniform_pvalues"),
                multipletests_result.rename("multipletests_result"),
            ],
            axis=1,
        )
        df["is_rejected"] = df["multipletests_result"].apply(lambda x: x[0].any())
        return df

    def uniform_test(self, pvalues: Iterable[float]) -> float:
        if self.uniform_test_method == "ks":
            return kstest(pvalues, "uniform").pvalue
        raise ValueError

    def multiple_test(
        self, pvalues: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        return multipletests(
            pvalues, alpha=self.alpha, method=self.mcp_correction_method
        )

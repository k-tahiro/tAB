from itertools import combinations
from typing import Callable, Generator, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import kstest
from statsmodels.stats.multitest import multipletests

from ..ctr import CTRTestResult


class AATestResult(NamedTuple):
    pvalues_for_aa_test: list[list[float]]
    multipletests_result: tuple[np.ndarray, np.ndarray, float, float]

    @property
    def is_rejected(self) -> bool:
        return bool(self.multipletests_result[0].any())


class AATest:
    def __init__(
        self,
        *test_funcs: Callable[[pd.DataFrame, pd.DataFrame], CTRTestResult],
        alpha: float = 0.05,
        uniform_test_method: str = "ks",
        mcp_correction_method: str = "hs",
    ) -> None:
        assert uniform_test_method in {"ks"}
        self.test_funcs = test_funcs
        self.alpha = alpha
        self.uniform_test_method = uniform_test_method
        self.mcp_correction_method = mcp_correction_method

    def __call__(
        self, dfs_loader: Generator[list[pd.DataFrame], None, None]
    ) -> list[AATestResult]:
        return [
            self.test_pvalues(pvalues_for_aa_test)
            for pvalues_for_aa_test in self.run_tests(dfs_loader)
        ]

    def run_tests(
        self, dfs_loader: Generator[list[pd.DataFrame], None, None]
    ) -> Generator[list[list[float]], None, None]:
        pvalues_arr = np.array(
            [
                [
                    [
                        test_func(df1, df2).ttest_result.pvalue
                        for test_func in self.test_funcs
                    ]
                    for df1, df2 in combinations(dfs, 2)
                ]
                for dfs in dfs_loader
            ]
        )
        pvalues_arr = pvalues_arr.transpose(2, 1, 0)

        for pvalues in pvalues_arr:
            yield pvalues.tolist()

    def test_pvalues(self, pvalues_for_aa_test: list[list[float]]) -> AATestResult:
        pvalues = [self.uniform_test(pvalues) for pvalues in pvalues_for_aa_test]
        return AATestResult(
            pvalues_for_aa_test,
            multipletests(pvalues, alpha=self.alpha, method=self.mcp_correction_method),
        )

    def uniform_test(self, pvalues: list[float]) -> float:
        if self.uniform_test_method == "ks":
            return kstest(pvalues, "uniform").pvalue
        raise ValueError

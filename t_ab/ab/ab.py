from itertools import combinations
from typing import Callable

import pandas as pd

from ..ctr.base import CTRTestResult, CTRTtestBase


class ABTest:
    def __init__(self, *test_funcs: CTRTtestBase, alpha: float = 0.05) -> None:
        self.test_funcs = test_funcs
        self.alpha = alpha

    def __call__(self, dfs: list[pd.DataFrame]) -> dict[str, dict[str, CTRTestResult]]:
        return {
            test_func.metrics_name: self.run_test(test_func, dfs)
            for test_func in self.test_funcs
        }

    def run_test(
        self,
        test_func: Callable[[pd.DataFrame, pd.DataFrame], CTRTestResult],
        dfs: list[pd.DataFrame],
    ) -> dict[str, CTRTestResult]:
        return {
            f"({i}, {j})": test_func(df_i, df_j)
            for (i, df_i), (j, df_j) in combinations(zip(range(len(dfs)), dfs), 2)
        }

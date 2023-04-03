from itertools import combinations

import pandas as pd

from ..ctr.base import CTRTtestBase


class ABTest:
    def __init__(self, *test_funcs: CTRTtestBase, alpha: float = 0.05) -> None:
        self.test_funcs = test_funcs
        self.alpha = alpha

    def __call__(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(
            [self.run_test(test_func, dfs) for test_func in self.test_funcs],
            ignore_index=True,
        )

    def run_test(
        self, test_func: CTRTtestBase, dfs: list[pd.DataFrame]
    ) -> pd.DataFrame:
        results = {
            f"({i}, {j})": test_func(df_i, df_j)
            for (i, df_i), (j, df_j) in combinations(zip(range(len(dfs)), dfs), 2)
        }
        return pd.DataFrame(
            [
                {
                    "metrics_base": test_func.METRICS_BASE,
                    "metrics": test_func.metrics_name,
                    "pair": pair,
                    "mean_l": result.statistics[0].mean,
                    "std_l": result.statistics[0].std,
                    "mean_r": result.statistics[1].mean,
                    "std_r": result.statistics[1].std,
                    "pvalue": result.test_result.pvalue,
                    "is_rejected": result.test_result.is_rejected,
                }
                for pair, result in results.items()
            ]
        )

from typing import Callable, NamedTuple

import numpy as np
from scipy.stats import kstest
from statsmodels.stats.multitest import multipletests


class AATestResult(NamedTuple):
    pvalues_arr: np.ndarray
    pvalues: list[float]
    is_rejected: bool


class AATest:
    def __init__(
        self,
        data_loader: Callable[[int], list],
        n_tests: int = 1000,
        alpha: float = 0.05,
        uniform_test_method: str = "ks",
        mcp_correction_method: str = "hs",
    ) -> None:
        assert uniform_test_method in {"ks"}
        self.data_loader = data_loader
        self.n_tests = n_tests
        self.alpha = alpha
        self.uniform_test_method = uniform_test_method
        self.mcp_correction_method = mcp_correction_method

    @property
    def uniform_test(self) -> Callable:
        if self.uniform_test_method == "ks":
            return kstest
        raise ValueError

    def __call__(
        self, *test_funcs: Callable[[list], list[float]]
    ) -> list[AATestResult]:
        pvalues_list = self.run_tests(*test_funcs)
        results = [self.test_pvalues(pvalues) for pvalues in pvalues_list]
        return results

    def run_tests(self, *test_funcs: Callable[[list], list[float]]) -> np.ndarray:
        pvalues_list: list[list[list[float]]] = [[] for _ in test_funcs]
        for i in range(self.n_tests):
            data = self.data_loader(i)
            for j, test_func in enumerate(test_funcs):
                pvalues_list[j].append(test_func(data))
        return np.array(pvalues_list).transpose(0, 2, 1)

    def test_pvalues(self, pvalues_arr: np.ndarray) -> AATestResult:
        pvalues = [kstest(pvalues, "uniform").pvalue for pvalues in pvalues_arr]
        return AATestResult(
            pvalues_arr,
            pvalues,
            multipletests(pvalues, alpha=self.alpha, method=self.mcp_correction_method),
        )

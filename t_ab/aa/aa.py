from typing import Callable, Generator, NamedTuple

import numpy as np
from scipy.stats import kstest
from statsmodels.stats.multitest import multipletests


class AATestResult(NamedTuple):
    pvalues_for_aa_test: list[list[float]]
    multipletests_result: tuple[np.ndarray, np.ndarray, float, float]

    @property
    def is_rejected(self) -> bool:
        return bool(self.multipletests_result[0].any())


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

    def __call__(
        self, *test_funcs: Callable[[list], list[float]]
    ) -> list[AATestResult]:
        return [
            self.test_pvalues(pvalues_for_aa_test)
            for pvalues_for_aa_test in self.run_tests(*test_funcs)
        ]

    def run_tests(
        self, *test_funcs: Callable[[list], list[float]]
    ) -> Generator[list[list[float]], None, None]:
        pvalues_list: list[list[list[float]]] = [[] for _ in test_funcs]
        for i in range(self.n_tests):
            data = self.data_loader(i)
            for j, test_func in enumerate(test_funcs):
                pvalues_list[j].append(test_func(data))

        for pvalues in pvalues_list:
            yield np.array(pvalues).T.tolist()

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

from typing import Callable, NamedTuple

from scipy.stats import kstest


class AATestResult(NamedTuple):
    pvalues: list[float]
    pvalue: float
    is_rejected: bool


class AATest:
    def __init__(
        self,
        data_loader: Callable[[int], list],
        n_tests: int = 1000,
        method: str = "ks",
        alpha: float = 0.05,
    ) -> None:
        assert method in {"ks"}
        self.data_loader = data_loader
        self.n_tests = n_tests
        self.method = method
        self.alpha = alpha

    def __call__(self, *test_funcs: Callable[[list], float]) -> list[AATestResult]:
        pvalues_list = self.run_tests(*test_funcs)
        results = [self.test_pvalues(pvalues) for pvalues in pvalues_list]
        return results

    def run_tests(self, *test_funcs: Callable[[list], float]) -> list[list[float]]:
        pvalues_list: list[list[float]] = [[] for _ in test_funcs]
        for i in range(self.n_tests):
            data = self.data_loader(i)
            for j, test_func in enumerate(test_funcs):
                pvalues_list[j].append(test_func(data))
        return pvalues_list

    def test_pvalues(self, pvalues: list[float]) -> AATestResult:
        if self.method == "ks":
            pvalue = kstest(pvalues, "uniform").pvalue
        return AATestResult(pvalues, pvalue, pvalue < self.alpha)

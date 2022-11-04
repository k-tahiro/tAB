from typing import Callable, NamedTuple

from scipy.stats import kstest


class AATestResult(NamedTuple):
    pvalues: list[float]
    pvalue: float
    is_rejected: bool


class AATest:
    def __init__(
        self,
        n_tests: int = 1000,
        method: str = "ks",
        alpha: float = 0.05,
    ) -> None:
        self.n_tests = n_tests
        self.method = method
        self.alpha = alpha

    def __call__(
        self,
        data_loader: Callable[[int], list],
        *test_funcs: Callable[[list], float],
    ) -> list[AATestResult]:
        pvalues: list[list[float]] = [[] for _ in test_funcs]
        for i in range(self.n_tests):
            data = data_loader(i)
            for j, test_func in enumerate(test_funcs):
                pvalues[j].append(test_func(data))

        results = []
        for pvs in pvalues:
            pvalue = kstest(pvs, "uniform").pvalue
            results.append(AATestResult(pvs, pvalue, pvalue < self.alpha))
        return results

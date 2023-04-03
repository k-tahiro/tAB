from typing import NamedTuple

from .statistics import Statistics


class TestResult(NamedTuple):
    statistic: float
    pvalue: float
    is_rejected: bool


class TwoSamplesTestResult(NamedTuple):
    statistics: tuple[Statistics, Statistics]
    test_result: TestResult

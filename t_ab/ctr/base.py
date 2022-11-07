from abc import ABC, abstractmethod
from typing import NamedTuple

import pandas as pd
from scipy.stats import ttest_ind_from_stats


class Statistics(NamedTuple):
    mean: float
    std: float
    nobs: int


class TtestResult(NamedTuple):
    statistic: float
    pvalue: float
    is_rejected: bool


class CTRTtestBase(ABC):
    def __init__(
        self,
        cluster_col: str,
        denominator_col: str,
        alpha: float = 0.05,
        equal_var: bool = True,
    ) -> None:
        self.cluster_col = cluster_col
        self.denominator_col = denominator_col
        self.alpha = alpha
        self.equal_var = equal_var

    def __call__(
        self, df_c: pd.DataFrame, df_t: pd.DataFrame, numerator_col: str
    ) -> tuple[tuple[Statistics, Statistics], TtestResult]:
        stats_c = self.calc_stats(self.agg_cluster(df_c, numerator_col))
        stats_t = self.calc_stats(self.agg_cluster(df_t, numerator_col))
        statistics, pvalue = ttest_ind_from_stats(
            *stats_c, *stats_t, equal_var=self.equal_var
        )
        return (stats_c, stats_t), TtestResult(statistics, pvalue, pvalue < self.alpha)

    def agg_cluster(self, df: pd.DataFrame, numerator_col: str) -> pd.DataFrame:
        return (
            df[[self.cluster_col, self.denominator_col, numerator_col]]
            .groupby(self.cluster_col)
            .sum()
        )

    @abstractmethod
    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        raise NotImplementedError

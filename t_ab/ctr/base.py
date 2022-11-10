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
        equal_var: bool = True,
        alpha: float = 0.05,
    ) -> None:
        """CTR testing base class.

        Args:
            cluster_col (str): The cluster column name in dataframes. Cluster should be randomization unit which is typically user.
            denominator_col (str): The denominator column name in dataframes. Denominator is typically the number of impressions.
            equal_var (bool, optional): Whether to assume equal population variances in T-test or not. Defaults to True.
            alpha (float, optional): Confidence level for T-test. Defaults to 0.05.
        """
        self.cluster_col = cluster_col
        self.denominator_col = denominator_col
        self.equal_var = equal_var
        self.alpha = alpha

    def __call__(
        self, df_c: pd.DataFrame, df_t: pd.DataFrame, numerator_col: str
    ) -> tuple[tuple[Statistics, Statistics], TtestResult]:
        """Run T-test for two independent groups.

        Args:
            df_c (pd.DataFrame): The dataframe for control group.
            df_t (pd.DataFrame): The dataframe for treatment group.
            numerator_col (str): The numerator column name in dataframes. Numerator is typically the number of actions like `click`.

        Returns:
            tuple[tuple[Statistics, Statistics], TtestResult]: Test statistics and T-test result.
        """
        stats_c = self.calc_stats(self.agg_cluster(df_c, numerator_col))
        stats_t = self.calc_stats(self.agg_cluster(df_t, numerator_col))
        statistic, pvalue = ttest_ind_from_stats(
            *stats_c, *stats_t, equal_var=self.equal_var
        )
        return (stats_c, stats_t), TtestResult(statistic, pvalue, pvalue < self.alpha)

    def agg_cluster(self, df: pd.DataFrame, numerator_col: str) -> pd.DataFrame:
        """Aggregate raw metrics for each cluster.

        Args:
            df (pd.DataFrame): Input dataframe.
            numerator_col (str): The numerator column name in dataframes. Numerator is typically the number of actions like `click`.

        Returns:
            pd.DataFrame: Aggregated dataframe.
        """
        return (
            df[[self.cluster_col, self.denominator_col, numerator_col]]
            .groupby(self.cluster_col)
            .sum()
        )

    @abstractmethod
    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        raise NotImplementedError

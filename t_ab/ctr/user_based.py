from typing import Any, Optional

import pandas as pd

from .base import CTRTtestBase, Statistics


class UserBasedCTRTtest(CTRTtestBase):
    @property
    def default_metrics_name(self) -> str:
        return f"{self.numerator_col} / {self.denominator_col} (User-Based)"

    def ignore_outliers(
        self, df: pd.DataFrame, outlier_percentile: float
    ) -> pd.DataFrame:
        df_cluster = self.agg_cluster(df)
        metrics = self.calc_metrics(df_cluster)
        threshold = metrics.quantile(outlier_percentile)
        return df[
            df[self.cluster_col].isin(df_cluster[metrics < threshold][self.cluster_col])
        ]

    def calc_metrics(self, df: pd.DataFrame) -> pd.Series:
        return df[self.numerator_col] / df[self.denominator_col]

    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        metrics = self.calc_metrics(df)
        return Statistics(metrics.mean(), metrics.std(), len(metrics))

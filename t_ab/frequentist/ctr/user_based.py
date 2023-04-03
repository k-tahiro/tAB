from typing import Optional

import pandas as pd

from ..common import Statistics
from .base import CTRTtestBase


class UserBasedCTRTtest(CTRTtestBase):
    METRICS_BASE = "User-Based"

    def ignore_outliers(
        self, df: pd.DataFrame, outlier_percentile: Optional[float] = None
    ) -> pd.DataFrame:
        df_cluster = self.agg_cluster(df)
        metrics = self.calc_metrics(df_cluster)
        if outlier_percentile is None:
            q1 = metrics.quantile(0.25)
            q3 = metrics.quantile(0.75)
            iqr = q3 - q1
            lower_threshold = q1 - 1.5 * iqr
            upper_threshold = q3 + 1.5 * iqr
            return df[
                df[self.cluster_col].isin(
                    df_cluster[
                        (metrics >= lower_threshold) & (metrics <= upper_threshold)
                    ].index
                )
            ]
        else:
            threshold = metrics.quantile(outlier_percentile)
            return df[df.index.isin(df_cluster[metrics <= threshold].index)]

    def calc_metrics(self, df: pd.DataFrame) -> pd.Series:
        return df[self.numerator_col] / df[self.denominator_col]

    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        metrics = self.calc_metrics(df)
        return Statistics(metrics.mean(), metrics.std(), len(metrics))

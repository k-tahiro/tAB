import pandas as pd
from scipy.stats import beta

from .base import CTRTtestBase, Statistics


class UserBasedCTRTtest(CTRTtestBase):
    METRICS_BASE = "User-Based"

    def ignore_outliers(
        self,
        df: pd.DataFrame,
        minimum_sample_size: int = 100,
        outlier_percentile: float = 0.99,
        estimate_dist: bool = True,
    ) -> pd.DataFrame:
        df_cluster = self.agg_cluster(df)
        df_cluster = df_cluster[df_cluster[self.denominator_col] >= minimum_sample_size]

        metrics = self.calc_metrics(df_cluster)
        if estimate_dist:
            a, b, _, _ = beta.fit(metrics, floc=0, fscale=1, method="MM")
            threshold = beta.ppf(outlier_percentile, a, b)
        else:
            threshold = metrics.quantile(outlier_percentile)

        return df[df[self.cluster_col].isin(df_cluster[metrics <= threshold].index)]

    def calc_metrics(self, df: pd.DataFrame) -> pd.Series:
        return df[self.numerator_col] / df[self.denominator_col]

    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        metrics = self.calc_metrics(df)
        return Statistics(metrics.mean(), metrics.std(), len(metrics))

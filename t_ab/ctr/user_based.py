from typing import Any, Optional

import pandas as pd

from .base import CTRTtestBase, Statistics


class UserBasedCTRTtest(CTRTtestBase):
    def __init__(
        self, outlier_percentile: Optional[float] = None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.outlier_percentile = outlier_percentile

    @property
    def default_metrics_name(self) -> str:
        return f"{self.numerator_col} / {self.denominator_col} (User-Based)"

    def ignore_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_percentile is None:
            return df
        metrics = self.calc_metrics(df)
        threshold = metrics.quantile(self.outlier_percentile)
        return df[metrics < threshold]

    def calc_metrics(self, df: pd.DataFrame) -> pd.Series:
        return df[self.numerator_col] / df[self.denominator_col]

    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        metrics = self.calc_metrics(df)
        return Statistics(metrics.mean(), metrics.std(), len(metrics))

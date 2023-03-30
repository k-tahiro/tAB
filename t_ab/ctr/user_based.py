import pandas as pd

from .base import CTRTtestBase, Statistics


class UserBasedCTRTtest(CTRTtestBase):
    @property
    def default_metrics_name(self) -> str:
        return f"{self.numerator_col} / {self.denominator_col} (User-Based)"

    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        metrics = df[self.numerator_col] / df[self.denominator_col]
        return Statistics(metrics.mean(), metrics.std(), len(metrics))

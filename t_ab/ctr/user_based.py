import pandas as pd

from .base import CTRTtestBase, Statistics


class UserBasedCTRTtest(CTRTtestBase):
    @property
    def default_metrics_name(self) -> str:
        return f"{self.numerator_col} / {self.denominator_col} (User-Based)"

    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        metrics = df.iloc[:, 1] / df.iloc[:, 0]
        return Statistics(metrics.mean(), metrics.std(), len(metrics))

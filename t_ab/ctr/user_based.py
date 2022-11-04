import pandas as pd

from .base import CTRTtestBase, Statistics


class UserBasedCTRTtest(CTRTtestBase):
    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        metrics = df.apply(lambda x: x[1].sum() / x[0].sum(), axis=1)
        return Statistics(metrics.mean(), metrics.std(), len(metrics))

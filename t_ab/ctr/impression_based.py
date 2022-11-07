import numpy as np
import pandas as pd

from .base import CTRTtestBase, Statistics


class ImpressionBasedCTRTtest(CTRTtestBase):
    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        nobs = len(df)
        cov_mat = df.cov()
        stats_s = Statistics(df.iloc[:, 1].mean(), np.sqrt(cov_mat.iloc[1, 1]), nobs)
        stats_n = Statistics(df.iloc[:, 0].mean(), np.sqrt(cov_mat.iloc[0, 0]), nobs)
        cov_sn = cov_mat.iloc[0, 1]
        return Statistics(
            stats_s.mean / stats_n.mean,
            np.sqrt(self.var_delta(stats_s, stats_n, cov_sn)),
            nobs,
        )

    @staticmethod
    def var_delta(stats_s: Statistics, stats_n: Statistics, cov_sn: float) -> float:
        mean_ratio = stats_s.mean / stats_n.mean
        return (
            (
                stats_s.std**2
                - 2 * mean_ratio * cov_sn
                + (mean_ratio * stats_n.std) ** 2
            )
            / (stats_n.mean**2)
            # / stats_s.nobs
        )

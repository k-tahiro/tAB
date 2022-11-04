import numpy as np
import pandas as pd

from .base import CTRTtestBase, Statistics


class ImpressionBasedCTRTtest(CTRTtestBase):
    def calc_stats(self, df: pd.DataFrame) -> Statistics:
        df = df.applymap(lambda x: x.sum())
        nobs = len(df)
        cov_mat = df.cov()
        stats_x = Statistics(df.iloc[:, 0].mean(), np.sqrt(cov_mat.iloc[0, 0]), nobs)
        stats_y = Statistics(df.iloc[:, 1].mean(), np.sqrt(cov_mat.iloc[1, 1]), nobs)
        cov_xy = cov_mat.iloc[0, 1]
        return Statistics(
            self.mean_delta(stats_x, stats_y, cov_xy),
            np.sqrt(self.var_delta(stats_x, stats_y, cov_xy)),
            nobs,
        )

    @staticmethod
    def mean_delta(stats_x: Statistics, stats_y: Statistics, cov_xy: float) -> float:
        return stats_y.mean / stats_x.mean

    @staticmethod
    def var_delta(stats_x: Statistics, stats_y: Statistics, cov_xy: float) -> float:
        return (
            (stats_y.std**2)
            - 2 * stats_y.mean / stats_x.mean * cov_xy
            + (stats_y.mean**2) / (stats_x.mean**2) * (stats_x.std**2)
        ) / (stats_x.mean**2)

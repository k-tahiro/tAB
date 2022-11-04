from dataclasses import dataclass
from typing import Self

import pandas as pd
from scipy.stats import beta


@dataclass
class BinomialData:
    name: str
    tot: int = 0
    pos: int = 0
    prior_pos: int = 1
    prior_neg: int = 1

    @property
    def posterior(self):
        neg = self.tot - self.pos
        return beta(self.prior_pos + self.pos, self.prior_neg + neg)

    def update_obs(self, tot: int, pos: int) -> Self:  # type: ignore
        self.tot += tot
        self.pos += pos
        return self


class BinomialTest:
    def __init__(self, *data: BinomialData) -> None:
        self.data = {bd.name: bd for bd in data}
        self.df: pd.DataFrame = None

    def update_obs(self, name: str, tot: int, pos: int) -> Self:  # type: ignore
        self.data[name].update_obs(tot, pos)
        return self

    def sample(self, n_samples: int = 100000) -> Self:  # type: ignore
        self.df = pd.DataFrame(
            {name: bd.posterior.rvs(n_samples) for name, bd in self.data.items()}
        )
        return self

    @property
    def probs(self) -> dict[str, float]:
        assert self.df is not None, "`sample` method must be called before."
        return (
            (self.df.idxmax(axis=1).value_counts() / len(self.df))
            .reindex(self.data, fill_value=0)
            .to_dict()
        )

    def hist(self, bins: int = 100) -> None:
        assert self.df is not None, "`sample` method must be called before."
        self.df.plot.hist(bins=bins)

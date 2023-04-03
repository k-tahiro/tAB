from dataclasses import dataclass

from scipy.stats import beta


@dataclass
class BinomialData:
    name: str
    tot: int = 0
    pos: int = 0
    prior_pos: int = 1
    prior_neg: int = 1

    def __post_init__(self) -> None:
        assert self.tot >= 0
        assert self.pos >= 0
        assert self.tot >= self.pos, "`pos` must be lower than or equal to `tot`."

    @property
    def posterior(self):
        neg = self.tot - self.pos
        return beta(self.prior_pos + self.pos, self.prior_neg + neg)

    def update_obs(self, tot: int, pos: int) -> "BinomialData":
        self.tot += tot
        self.pos += pos
        return self

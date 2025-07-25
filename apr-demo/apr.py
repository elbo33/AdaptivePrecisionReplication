# Adaptive‑Precision Replication (inner loop)
import numpy as np
from scipy.stats import t, norm

class APR:
    """
    One call of this class evaluates a single design point `x`
    (here `x` is unused because the black‑box has fixed μ,σ).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n0: int = 5,
        eps_tight: float = 0.05,
        fidelity: float = 0.5,
        n_max: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.alpha     = alpha
        self.n0        = n0
        self.eps_tight = eps_tight
        self.fidelity  = fidelity
        self.n_max     = n_max
        self.rng       = rng or np.random.default_rng()

    # -------------------------------------------------------------
    def _eps_target(self, eps_loose: float) -> float:
        f  = self.fidelity
        return (eps_loose ** (1 - f)) * (self.eps_tight ** f)

    # -------------------------------------------------------------
    def run(self, sim_fn):
        """
        Parameters
        ----------
        sim_fn : callable(int) -> np.ndarray
            Function that returns an array of `k` i.i.d. draws.

        Returns
        -------
        mean_hat, se_hat, n_used
        """
        y = sim_fn(self.n0)
        n = self.n0
        mean = y.mean()
        s    = y.std(ddof=1)
        eps_loose = s / np.sqrt(n)

        while True:
            c = t.ppf(1 - self.alpha / 2, df=n - 1) if n < 30 else norm.ppf(
                1 - self.alpha / 2
            )
            se = s / np.sqrt(n)
            if c * se <= self._eps_target(eps_loose):
                return mean, se, n

            n_star   = int(np.ceil((s / self._eps_target(eps_loose)) ** 2))
            n_next   = max(n + 1, n_star)
            if self.n_max is not None:
                n_next = min(n_next, self.n_max)
            add      = n_next - n
            if add == 0:  # capped
                return mean, se, n

            y_new = sim_fn(add)
            y     = np.concatenate([y, y_new])
            n     = y.size
            mean  = y.mean()
            s     = y.std(ddof=1)

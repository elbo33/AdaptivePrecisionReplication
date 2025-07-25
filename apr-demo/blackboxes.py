# Synthetic distributions with known μ, σ
import numpy as np
from scipy.stats import t, poisson

class Dist:
    """Factory for sampling and for retrieving true μ, σ."""

    def __init__(self, name, mu, sigma, df=None):
        self.name  = name
        self.mu    = mu
        self.sigma = sigma
        self.df    = df

    # ---------------------------------------------------------
    def sampler(self, rng):
        mu, sig = self.mu, self.sigma
        name    = self.name

        def _sample(k: int):
            if name == "N":
                return rng.normal(mu, sig, size=k)
            if name == "LN":
                # solve for log‑normal parameters
                var      = sig**2
                phi      = np.sqrt(var + mu**2)
                sigma_ln = np.sqrt(np.log((phi**2) / (mu**2)))
                mu_ln    = np.log(mu) - 0.5 * sigma_ln**2
                return rng.lognormal(mu_ln, sigma_ln, size=k)
            if name.startswith("T"):
                return mu + sig * t.rvs(self.df, size=k, random_state=rng)
            if name == "Pois":
                return rng.poisson(mu, size=k)  # variance=μ
            if name == "Mix":
                comp = rng.choice(2, p=[0.9, 0.1], size=k)
                vals = rng.normal(mu, sig, size=k)
                heavy = rng.normal(mu, 5 * sig, size=k)
                vals[comp == 1] = heavy[comp == 1]
                return vals
            raise ValueError("unknown distribution")

        return _sample

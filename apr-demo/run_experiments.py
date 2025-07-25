import itertools, pandas as pd, numpy as np
from apr import APR
from blackboxes import Dist
import baselines

RNG = np.random.default_rng
SEEDS = [11, 13, 17, 19, 23]

# --------------------------------------------------------------
dists = []
for mu, sig in itertools.product([0, 1, 2, 5], [0.5, 1, 2, 5]):
    for code in ["N", "LN", "T3", "T5", "T10", "Pois", "Mix"]:
        df = int(code[1:]) if code.startswith("T") else None
        name = "T" if code.startswith("T") else code
        dists.append(Dist(name, mu, sig, df))

rows = []
for d in dists:
    for seed in SEEDS:
        rng = RNG(seed)
        sim = d.sampler(rng)

        # APR with three fidelity levels
        for f in [0.0, 0.5, 1.0]:
            apr   = APR(fidelity=f, rng=rng)
            m, se, n = apr.run(sim)
            rows.append(
                dict(
                    solver=f"APR_f{f}",
                    dist=d.name,
                    mu=d.mu,
                    sigma=d.sigma,
                    seed=seed,
                    n=n,
                    abs_err=abs(m - d.mu),
                )
            )

        # Baselines
        for label, kk in [("FIX10", 10), ("FIX50", 50)]:
            m, se, n = baselines.fixed_budget(sim, kk)
            rows.append(
                dict(
                    solver=label,
                    dist=d.name,
                    mu=d.mu,
                    sigma=d.sigma,
                    seed=seed,
                    n=n,
                    abs_err=abs(m - d.mu),
                )
            )

        m, se, n = baselines.worst_case(sim, eps_tight=0.05, sigma_max=5)
        rows.append(
            dict(
                solver="WORST",
                dist=d.name,
                mu=d.mu,
                sigma=d.sigma,
                seed=seed,
                n=n,
                abs_err=abs(m - d.mu),
            )
        )

df = pd.DataFrame(rows)
df.to_csv("results.csv", index=False)
print("Saved results.csv with", len(df), "rows")

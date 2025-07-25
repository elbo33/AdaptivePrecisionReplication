import numpy as np

def fixed_budget(sim_fn, n):
    y = sim_fn(n)
    return y.mean(), y.std(ddof=1) / np.sqrt(n), n

def worst_case(sim_fn, eps_tight, sigma_max, alpha=0.05):
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    n = int(np.ceil((sigma_max / eps_tight) ** 2))
    return fixed_budget(sim_fn, n)

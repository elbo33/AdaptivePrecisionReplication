# Adaptive Precision Replication (APR)

This tiny repo benchmarks **Adaptive Precision Replication (APR)** against
simple fixed‑budget policies on synthetic noisy functions whose true mean and
variance are known.

## Quick‑start

```bash
# 1 · create an isolated environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2 · install minimal dependencies
pip install -r requirements.txt

# 3 · run the benchmark (≈ 60 s on a laptop)
python run_experiments.py      # writes results.csv
```

## What’s in `results.csv`

| column   | description                                                                |
|----------|----------------------------------------------------------------------------|
| solver   | `APR_f0`, `APR_f0.5`, `APR_f1`, `FIX10`, `FIX50`, `WORST`                  |
| dist     | noise family: `N`, `LN`, `T` (df = 3/5/10), `Pois`, `Mix`                  |
| mu       | true mean used to generate samples                                         |
| sigma    | true standard deviation                                                    |
| seed     | RNG seed (11 / 13 / 17 / 19 / 23)                                          |
| n        | replications consumed by the policy                                        |
| abs_err  | absolute error = `| estimated μ − true μ |`                                |

Each line corresponds to one **instance × seed × solver**.  
There are 5 noise families × 20 (μ,σ) pairs × 5 seeds × 6 solvers = 3 000 rows.

## Next steps

* Load the CSV in a notebook to build the data‑profile curves from the paper.
* Tweak `apr.py` parameters (e.g. `eps_tight`, fidelity ladder) and re‑run.
* Replace the synthetic samplers in `blackboxes.py` with your own simulator.

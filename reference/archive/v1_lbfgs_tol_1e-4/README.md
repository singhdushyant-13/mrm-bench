# Archived reference results, version 1 (lbfgs at its default tolerance, 1e-4)

**Superseded on 25 October 2026. Kept so the original experiments stay reproducible. Do not edit these files.**

The four files here are byte-identical copies of `reference/` as it stood at commit `2b8d648`:

- `metrics.json`
- `label_counts.csv`
- `coefficients.csv`
- `calibration.csv`

They were the official baseline results from 22 to 25 October 2026. The current official results are in `reference/`
(see `reference/README.md`).

## Provenance and configuration

- **Produced by** the original script `src/baseline_logreg.py` at commit `9253e97` (22 October 2026, recoverable with
  `git show 9253e97:src/baseline_logreg.py`). The notebook `notebooks/baseline_logistic_regression.ipynb` at commit
  `2b8d648` reproduced every value within 1e-9.
- **Data:** Freddie Mac SFLLD 2015 sample, Release 47.
- **Outcome:** 12-month 90+ DPD serious delinquency, December snapshots.
- **Split:** train Dec 2015–2018, test Dec 2021–2024, stress Dec 2019–2020; loan-disjoint, SHA-256(loan ID) mod 100 < 70.
- **Features:** 23.
- **Model:** L2 logistic regression, C = 1, `max_iter = 5000`, seed 20261022.
- **Solver:** lbfgs with scikit-learn's **default tolerance `tol = 1e-4`**. It was not set explicitly, and
  `metrics.json` does not record it.
- **Libraries:** Python 3.12.12, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1.

## Headline results (version 1)

| set | rows | events | mean prediction | ROC-AUC [95% CI] | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| train 2015–2018 | 101,843 | 462 | 0.46% | 0.890 [0.872, 0.908] | 0.256 | 0.00383 |
| test 2021–2024 | 15,716 | 119 | 0.43% | 0.894 [0.849, 0.930] | 0.307 | 0.00620 |
| stress 2019–2020 | 12,445 | 422 | 0.61% | 0.732 [0.706, 0.757] | 0.154 | 0.03205 |

## Why it was superseded

lbfgs stopped after 40 iterations by its own rule: scipy status 0, "NORM OF PROJECTED GRADIENT <= PGTOL", no
convergence warning. But it stopped before the optimum of the model's stated objective. Its penalized objective is
2030.98, against 2023.11 at the optimum.

The difference sits mainly in the one-hot coefficients. Each categorical feature's level coefficients summed to
−1.277 instead of 0, and the contrasts of rare levels were barely moved from their starting value. That cost almost
nothing in overall discrimination (test ROC-AUC 0.894 vs 0.893 converged). It did change the individual categorical
coefficients, and the predictions for loans with rare category levels by up to 0.12 on test. The full study is in
`reports/solver_tolerance/`.

Version 2, in `reference/`, changes only the tolerance, to `tol = 1e-8`. Data, labels, features, splits, solver and C
are unchanged. `label_counts.csv` is identical in both versions.

## How to reproduce version 1

1. **Check out the old commit and run the notebook.** This uses the exact code that produced these files:

   ```bash
   git worktree add ../mrm-bench-v1 2b8d648      # a separate folder at the old commit
   ```

   Copy or link `data/raw/sample_2015.zip` into its `data/raw/`, run `uv sync` there, then run the notebook. Section
   12 prints `MATCHES THE REFERENCE MODEL`.

2. **Or, in the current repository:** `uv run python reports/solver_tolerance/run_sensitivity.py` refits the current
   notebook pipeline with `tol = 1e-4`. It checks every metric, CI, coefficient and calibration value against these
   files, and prints the largest difference, which must be at most 1e-9.

3. **Or set `TOL = 1e-4`** in the notebook's SETTINGS cell. The results then equal these files. Section 12 compares
   with `reference/`, not with this folder, so it lists the differences from version 2, and `config.tol` appears as
   an extra item.

# Reference results of the baseline logistic regression

These four files are the **official results** of the MRM-Bench clean baseline, **version 2** (since 25 October 2026).
The baseline predicts 12-month 90+ DPD serious delinquency with a logistic regression on the Freddie Mac SFLLD 2015
sample, Release 47.

| File | Contents |
|---|---|
| `metrics.json` | train / test 2021–2024 / stress 2019–2020: rows, events, event rate, mean prediction, ROC-AUC, PR-AUC, Brier, 95% loan-bootstrap CIs; the run configuration (including `tol`); library versions |
| `label_counts.csv` | number of loans per December snapshot and label (event, non_event, censored_prepay, censored_other_termination), before the train/test split |
| `coefficients.csv` | the 39 fitted coefficients (numeric features standardized) and the intercept |
| `calibration.csv` | test 2021–2024 calibration by decile of predicted risk: rows, mean prediction, observed rate, events |

Headline numbers: train 101,843 rows / 462 events; test 15,716 rows / 119 events; test ROC-AUC 0.893 [0.848, 0.928],
PR-AUC 0.301, Brier 0.00622 (base-rate model 0.00751).

## Versions

| | version 2 (current, this folder) | version 1 (archived: `archive/v1_lbfgs_tol_1e-4/`) |
|---|---|---|
| Solver | lbfgs, `tol = 1e-8`, `max_iter = 5000`: 810 iterations, at the optimum (matches newton-cholesky) | lbfgs, scikit-learn's default `tol = 1e-4`: 40 iterations, stopped before the optimum (no warning) |
| Everything else | identical: data, labels, features, splits, loan partition, preprocessing, C = 1, seed | identical |
| Test ROC-AUC [95% CI] / PR-AUC / Brier | 0.893 [0.848, 0.928] / 0.301 / 0.00622 | 0.894 [0.849, 0.930] / 0.307 / 0.00620 |
| Produced by | the notebook, 25 October 2026 | `src/baseline_logreg.py` at commit `9253e97`, 22 October 2026 |

`label_counts.csv` is identical in both versions.

The correction changes overall discrimination and calibration only in the third decimal. It does change:

- the one-hot coefficients;
- the predictions for loans with rare category levels, by up to 0.12 on test and 0.24 in stress.

Rare-level coefficients rest on 0–2 training events and should not be interpreted. The study is in
`reports/solver_tolerance/`.

## Provenance (version 2)

- **How it was made.** `notebooks/baseline_logistic_regression.ipynb` was run headless (nbconvert) with its SETTINGS
  unchanged (`TOL = 1e-8`) except `SAVE_OUTPUTS = True`. The four files were copied from `outputs/`.
- **Reproduced.** A second independent run produced byte-identical files. The notebook then printed
  `MATCHES THE REFERENCE MODEL` against them.
- **Environment.** Python 3.12.12, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1 (also recorded in `metrics.json`).
- **Independent check.** `uv run python reports/solver_tolerance/run_sensitivity.py` refits the pipeline. It checks
  these files within 1e-9, checks the version-1 archive within 1e-9 with `tol = 1e-4`, and confirms the optimum with
  scikit-learn's newton-cholesky solver.

## How they are used

The baseline is `notebooks/baseline_logistic_regression.ipynb`. Its section 12 compares every number above with these
files and prints `MATCHES THE REFERENCE MODEL` when all of them agree (tolerance 1e-9).

**Aggregates only.** No file here or in `archive/` contains a loan ID or any loan-level record, so they can be
committed. Freddie Mac loan-level data must never be added here.

## Changing them

Do not edit these files by hand. If a model change is approved:

1. Archive the current files under `archive/` with a README.
2. Run the notebook with `SAVE_OUTPUTS = True`.
3. Review the differences that section 12 reports.
4. Copy the new files from `outputs/` into this folder.

Do this in a separate, reviewed commit that also updates `CLAUDE.md`.

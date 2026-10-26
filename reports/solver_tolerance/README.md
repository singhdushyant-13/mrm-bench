# Solver-tolerance sensitivity of the baseline logistic regression

**Status: adopted.** On 25 October 2026 the official baseline was changed from scikit-learn's default lbfgs tolerance
(`tol = 1e-4`) to `tol = 1e-8`. Nothing else changed: data, labels, features, splits, solver and C are the same.

| version | where | solver tolerance | test ROC-AUC [95% CI] | test PR-AUC |
|---|---|---|---|---|
| **current official results (v2)** | `reference/` | 1e-8 (converged) | 0.893 [0.848, 0.928] | 0.301 |
| previous results (v1, archived) | `reference/archive/v1_lbfgs_tol_1e-4/` | 1e-4 (stopped early) | 0.894 [0.849, 0.930] | 0.307 |

This folder holds the study behind the change. The missing-month labelling rule and the censoring method were **not**
changed; they wait for Dr. Ku's decision (see `CLAUDE.md`, "Outcome specification").

## How to reproduce

```bash
uv run python reports/solver_tolerance/run_sensitivity.py      # about 1 minute; writes only into this folder
```

The script runs the notebook's code cells for sections 1–8 in a fresh namespace. The data, labels, features,
chronological split, loan partition, preprocessing and C = 1 are therefore exactly the notebook's. It then refits the
same pipeline and changes **only** the solver setting:

- lbfgs with `tol` = 1e-4 (v1), 1e-6 and 1e-8 (official);
- an independent check: `solver="newton-cholesky"`, which minimizes the same objective with Newton steps.

Metrics, the 95% loan-bootstrap CIs (1,000 resamples) and the test calibration table come from the notebook's own
`evaluate()` and calibration code. Two reproduction checks run every time, each over every metric, CI, coefficient
and calibration value:

- the tol = 1e-8 fit against `reference/`: largest difference 0.0;
- the tol = 1e-4 fit against the v1 archive: largest difference 0.0.

**Outputs** (aggregates only; no loan IDs or loan-level rows):

- `report.md`: all tables;
- `results.json`;
- `metrics_by_fit.csv`, `coefficients_by_fit.csv`, `categorical_coefficient_changes.csv`;
- `probability_differences.csv`, `probability_differences_by_level.csv`;
- `rare_level_subgroups.csv`, `largest_changes_composition.csv`, `level_contrasts.csv`;
- `calibration_test_by_fit.csv`.

Run with scikit-learn 1.9.1 and scipy 1.18.1.

**The objective.** scikit-learn's lbfgs minimizes f(w, b) = mean log-loss + ‖w‖² / (2·C·n), with the intercept b
unpenalized. This is the documented objective J = C·Σ log-loss + ‖w‖²/2, divided by C·n. With n = 101,843 the penalty
weight in f is 1/(C·n) = 9.8e-6. L-BFGS-B stops when the largest gradient component of f is ≤ `tol`, or when the
relative decrease of f is ≤ 64·machine epsilon. scikit-learn raises a `ConvergenceWarning` only if scipy reports a
failure (status ≠ 0), for example by reaching `max_iter`.

## 1. Did the v1 fit converge?

**It stopped successfully by its own rule, with no convergence warning, but short of the optimum.**

- lbfgs stopped after 40 of 5,000 allowed iterations with scipy status 0 and the message
  `CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`. The largest gradient component was 7.8e-5, below `tol` = 1e-4.
  No warning of any kind was raised.
- Its objective is J = 2030.981. At the optimum (newton-cholesky, largest gradient 6e-11) J = 2023.114.
- The intercept's optimality condition is not met: at the optimum the mean training prediction equals the training
  event rate (0.4536%), but v1 gives 0.4556%.

## 2–3. Results by tolerance (same data and model; only the solver setting differs)

| | tol 1e-4 (v1) | tol 1e-6 | tol 1e-8 (official) | newton-cholesky (check) |
|---|---|---|---|---|
| Iterations | 40 | 390 | 810 | 9 |
| Stop reason (scipy status 0 for all lbfgs fits) | gradient ≤ tol | gradient ≤ tol | relative decrease of f ≤ 64·eps | Newton converged |
| Convergence warnings | 0 | 0 | 0 | 0 |
| Largest gradient component of f | 7.8e-5 | 9.3e-7 | 4.3e-8 | 6.1e-11 |
| Penalized objective J | 2030.9809 | 2023.1154 | 2023.1144 | 2023.1144 |
| J − J(optimum) | 7.87 | 9.5e-4 | 2.8e-7 | 0 |
| Test ROC-AUC [95% CI] | 0.8943 [0.849, 0.930] | 0.8928 [0.848, 0.928] | 0.8929 [0.848, 0.928] | 0.8929 [0.848, 0.928] |
| Test PR-AUC [95% CI] | 0.3073 [0.221, 0.406] | 0.2995 [0.212, 0.398] | 0.3005 [0.213, 0.400] | 0.3005 [0.213, 0.400] |
| Test Brier (base-rate model 0.00751) | 0.006196 | 0.006215 | 0.006215 | 0.006215 |
| Test mean prediction (event rate 0.757%) | 0.435% | 0.448% | 0.448% | 0.448% |
| Test calibration intercept / slope | 0.388 / 0.911 | 0.315 / 0.900 | 0.316 / 0.900 | 0.316 / 0.900 |
| Test top decile: predicted / observed | 3.56% / 5.79% | 3.67% / 5.85% | 3.67% / 5.85% | 3.67% / 5.85% |
| Stress ROC-AUC / PR-AUC / Brier | 0.7324 / 0.1540 / 0.03205 | 0.7329 / 0.1529 / 0.03207 | 0.7329 / 0.1530 / 0.03207 | 0.7329 / 0.1530 / 0.03207 |
| Train ROC-AUC / Brier | 0.8902 / 0.003835 | 0.8912 / 0.003814 | 0.8912 / 0.003814 | 0.8912 / 0.003814 |

**Coefficient differences** (39 coefficients plus the intercept, numeric features standardized):

| comparison | largest change (where) | numeric features | one-hot levels | intercept |
|---|---|---|---|---|
| v1 → official | 4.14 (intercept) | 0.020 (`dti_missing`; median 0.005) | 1.088 (`assist_plan_t_none`) | 4.14 |
| tol 1e-6 → official | 0.037 (intercept) | 0.0003 | 0.012 | 0.037 |
| official → newton-cholesky | 0.0005 (`first_time_homebuyer_infrequent`, 10 rows) | 2e-6 | 0.0005 | 0.0002 |

**Probability differences, v1 → official:** test mean |Δp| = 0.00033, 99th percentile 0.0065, largest 0.121.
Spearman rank correlation 0.993. 2,172 of 15,716 test rows change calibration decile, mostly among near-tied low
predictions. Between official and newton-cholesky the largest |Δp| is 6e-5 on test (8e-5 in stress).

## 4. Do the tighter solutions stabilize, and do the conclusions change?

**They stabilize at tol = 1e-8.** That fit agrees with the independent Newton solution to 2.8e-7 in the objective,
5e-4 in any coefficient (a level with 10 training rows and no events), 8e-5 in any probability, and 4 decimals in
every metric. tol = 1e-6 is close but not finished: the intercept is 0.037 away and test PR-AUC 0.2995 vs 0.3005.

**Overall discrimination and calibration do not change materially.** Test ROC-AUC moves 0.894 → 0.893, PR-AUC
0.307 → 0.301 and Brier 0.00620 → 0.00622. All three stay far inside the v1 CIs, and the model still beats the
base-rate Brier. The model still under-predicts the test level (mean 0.45% vs observed 0.76%; calibration slope
0.90). Stress discrimination is unchanged (0.732 → 0.733). Numeric coefficients move by at most 0.020.

## 5. Rare category levels: are the changed predictions material?

Unchanged overall metrics do not imply unchanged individual predictions, so the changes were examined level by
level. A level is "rare" here if fewer than 1,000 training rows have it:

- `status_t` 01 (596 rows) and 02 (138);
- `assist_plan_t` F (39) and infrequent codes (14);
- `property_type` CP (192) and MH (376);
- `first_time_homebuyer` infrequent codes (10).

Full tables: `report.md` §6, `rare_level_subgroups.csv`, `largest_changes_composition.csv`, `level_contrasts.csv`.

**Where the changes are.** On test, the 346 rows with a rare level have mean |Δp| = 0.0087 (largest 0.121). The
other 15,370 rows have mean 0.00014 (largest 0.030). The largest test changes are all in rare-level rows:

| row type (test, v1 → official) | rows among the 25 largest changes | mean predicted v1 → official |
|---|---|---|
| `status_t` 01 + infrequent assistance code | 2 | 14.5% → 3.2% |
| `status_t` 02 | 15 | 43.7% → 48.5% |
| `status_t` 01 + assistance plan F | 5 | 38.2% → 32.8% |
| property type MH; first-time-homebuyer infrequent; infrequent assistance code | 1 each | 13.2% → 5.1%; 17.1% → 9.3%; 6.7% → 1.6% |

In stress the largest change is 0.242: one row with `status_t` 01 and property type MH, 48.8% → 24.6%.

**Subgroup calibration (test, observed rate with Wilson 95% interval vs mean prediction v1 → official):**

| subgroup | rows | events | observed [95% CI] | predicted v1 → official | effect of the change |
|---|---|---|---|---|---|
| all rows | 15,716 | 119 | 0.76% [0.63, 0.91] | 0.43% → 0.45% | none: under-predicted in both |
| rows with a rare level | 346 | 60 | 17.3% [13.7, 21.7] | 10.0% → 10.1% | none: under-predicted in both |
| `status_t` 01 | 143 | 34 | 23.8% [17.5, 31.4] | 10.6% → 10.5% | none: under-predicted in both |
| `status_t` 02 | 43 | 24 | 55.8% [41.1, 69.6] | 42.7% → 45.1% | both inside the interval |
| assistance plan F | 16 | 5 | 31.3% [14.2, 55.6] | 28.4% → 25.6% | both inside the interval |
| property type MH | 117 | 0 | 0% [0, 3.2] | 0.46% → 0.18% | both inside; official closer |
| infrequent assistance code | 3 | 2 | 67% [21, 94] | 11.9% → 2.7% | both outside; 3 rows say little |

- **Training rows.** The official fit matches the infrequent-assistance subgroup better: 14 rows, observed 14.3%;
  v1 predicted 42.4%, outside the interval; official predicts 20.0%. Every training subgroup lies inside its
  interval under the official fit.
- **Top risk decile.** Test top-decile membership changes for 58 of 1,572 rows. 31 rows with a rare level leave it,
  26 of them manufactured-housing (MH) rows, a subgroup with 0 test events. 58 rows without a rare level enter it.

**Interpretation of the categorical coefficients.** Odds ratios against each feature's most common level:

- **Well-populated levels barely move:**
  - `status_t` 01 vs 00: 6.33 → 6.11;
  - `status_t` 02 vs 00: 20.8 → 21.9;
  - occupancy I vs P: 0.61 → 0.67;
  - PUD vs single-family: 1.43 → 1.39.
- **Four rare levels with 0–2 training events reverse direction:**
  - property type CP: 1.40 → 0.45 (192 rows, 0 events);
  - property type MH: 1.26 → 0.47 (376 rows, 1 event);
  - infrequent assistance code: 2.15 → 0.41 (14 rows, 2 events);
  - infrequent first-time-homebuyer code: 1.65 → 0.82 (10 rows, 0 events).
- **Assistance plan F stays above 1:** 2.69 → 1.91 (39 rows, 17 events).

The official odds ratios match newton-cholesky to three decimals. The v1 values near 1 for rare levels were not a
property of the model. lbfgs starts every coefficient at 0 and stopped before these weakly informed coordinates had
moved, which amounts to extra, unplanned shrinkage toward "no effect". With 0–2 events, the official values are set
mostly by the L2 penalty: without it, a zero-event level's coefficient would go to −∞.

**Conclusion.** The correction is not material for the baseline's conclusions:

- overall discrimination;
- overall calibration;
- calibration of the subgroups that have enough data;
- the numeric coefficients.

It is material for three narrower things:

- individual predictions in rare-category combinations, by up to 0.12 on test and 0.24 in stress;
- membership of the top risk decile for about 30 rare-level test rows;
- the direction of four rare-level odds ratios.

Those rare-level coefficients rest on 0–2 events, so they should not be interpreted in either version. If
rare-category predictions ever matter, grouping rare levels is a modelling question for Dr. Ku. It is not part of
this numerical correction.

## 6. Why some coefficients changed by up to 1.09

Every categorical feature is one-hot encoded with **all** of its levels, next to an unpenalized intercept. Every
training row has exactly one active level per feature (checked: 101,843 of 101,843 for all 7 features, and no
test or stress row has an unseen level). So each feature's columns add up exactly to the intercept's column of ones.

**(a) A direction the data cannot see.** Add d to every level of one feature and subtract d from the intercept:
every prediction stays the same. Checked: adding 0.5 to `assist_plan_t` changes the training log-loss by 4.5e-13.
Only the L2 penalty depends on d. It is smallest when the feature's coefficients **sum to zero**.

**(b) Why the default stopping rule left it unfinished.** In scikit-learn's objective the penalty weight is
1/(C·n) ≈ 1e-5. The gradient along this direction is therefore about 1e-5 × the coefficient sum, below `tol` = 1e-4.
v1 stopped with every feature's coefficients summing to −1.277. They sum to 0.009 at tol 1e-6, −7e-5 at the official
1e-8, and about 1e-14 for newton-cholesky.

Centring each feature of v1, which keeps its predictions fixed, lowers J from 2030.98 to 2029.19. That is 1.79 of the
7.87 gap, with no prediction changing. The remaining 6.07 is a real improvement in fit, mostly from the rare-level
contrasts of §5. The dominant levels ("no plan" in 99.95% of rows, status 00 in 99.3%) are nearly collinear with the
intercept, so their contrasts trade off against the effective intercept (−4.152 → −5.484), and for typical loans the
trade-off nets out.

**Decomposition of the largest change.** `assist_plan_t_none` moves −1.009 → +0.079 (Δ = +1.088). Of that, +0.426 is
the common shift from (a), which affects no prediction. The other +0.663 is the "no plan versus F or infrequent"
contrast, which only 53 training rows inform. 12 of the 23 one-hot coefficients changed sign between the versions.
On their own, one-hot coefficients cannot be interpreted in either version.

## 7. Decision and remaining recommendations

- **Adopted:** lbfgs with `tol = 1e-8`, `max_iter = 5000` (810 iterations used). It is the smallest change that
  reaches the optimum, confirmed by newton-cholesky.
- **Unaffected:** the leakage tests and the four MRM-Bench pairs. They do not use the fitted model.
- **Not changed, pending Dr. Ku:** the missing-month non-event rule (one training row) and the censoring method.
- **Not recommended now:** removing a level per feature (`drop="first"`) or grouping rare levels. These are
  modelling decisions, not numerical fixes.

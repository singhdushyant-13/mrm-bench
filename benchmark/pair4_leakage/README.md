# MRM-Bench Pair 4 — Leakage: a lookback window without an upper bound

**Category:** leakage  **Pair ID:** P4  **Cases:** `P4_F` (faulty/, defect) and `P4_V` (valid/, no defect)

## What this pair tests

At a prediction month t, the model may only use what was known by the end of month t. The features here
summarise a loan's recent delinquency history, for example the worst status over the last 12 months.

Both implementations differ in one line, the one that selects the "last 12 months":

- **Valid:** `months > t − 12` **and** `months ≤ t`, which means months t−11 … t.
- **Faulty:** only `months > t − 12`. This also selects every month *after* t.

The faulty code runs without errors, but its features "see" delinquency that happens after the prediction
date. That is part of the outcome the model is meant to predict, so the model would look better in testing
than it can ever be in use.

**False-alarm trap:** the valid code includes month t itself (`≤ t`). That is correct, because month t's
status is known at the prediction date (the end of month t). A validator that flags `≤ t` as leakage
produces a false positive.

## Files

| file | what it is |
|------|------------|
| `faulty/spec.md`, `valid/spec.md` | The specification: feature definitions with exact month ranges (R1–R3), the information-set rule (R4), why leakage is wrong, what is *not* a defect, the executable future-invariance criterion (R5), interface (R6), applicability conditions, definition differences from the baseline, and provenance. **The two files are identical.** |
| `faulty/impl.py`, `valid/impl.py` | `build_features(panel, prediction_month)`. **They differ only in the window line** (see below). |
| `faulty/label.json`, `valid/label.json` | Ground truth: `is_defect`, the violated rule (or `null`), evidence and source. |
| `faulty/test_case.py`, `valid/test_case.py` | Tests T1–T11 on one synthetic panel (F01–F10, t = 12). **Identical except for the import line.** |
| `__init__.py` (here and in each sub-folder) | Empty package markers, as in Pairs 1–3. |

Related files one level up: `benchmark/run_pair4.py` has the same design as `run_pair3.py`, without the horizon
columns. `benchmark/test_no_real_data.py` already scans this folder.

## The single difference between the implementations

```diff
--- valid/impl.py
+++ faulty/impl.py
@@ -60,7 +60,7 @@
     panel = panel[panel["loan_id"].isin(ids)]
 
     # 12-month lookback window
-    window = panel[(panel["month"] > t - 12) & (panel["month"] <= t)]
+    window = panel[panel["month"] > t - 12]
     history = panel[panel["month"] <= t]
 
     out = pd.DataFrame(index=ids)
```

`status_t` and `ever_90_before_t` do not use `window`, so they are the same in both versions. Only
`max_dq_12m` and `n_dq_months_12m` can see the future.

## Expected test outcomes (answer key)

| test | rule | `P4_V` | `P4_F` | why |
|---|---|---|---|---|
| T1 exact output, all rows and columns | R1–R3 | pass | **fail** | F03 gets 8/8 instead of 0/0; F09 gets 1/1 instead of 0/0 |
| T2 future invariance: remove records after t | R4, R5 | pass | **fail** | the features change, which proves dependence on the future |
| T3 future invariance: set statuses after t to "05" | R4, R5 | pass | **fail** | almost every loan's features change |
| T4 key case F03 (delinquent only from t+1) | R4 | pass | **fail** | max 8, count 8 |
| T5 boundary t+1 (F09) | R3, R4 | pass | **fail** | max 1 |
| T6 month t included (F05) | R3, R4 | pass | pass | the faulty window also contains t |
| T7 month t−12 excluded (F04) | R3 | pass | pass | the lower bound is the same in both |
| T8 past 90+ then cured (F06) | R3 | pass | pass | its months after t are all "00" |
| T9 row set and sorted index | R1 | pass | pass | the row set does not use `window` |
| T10 input not modified | R6 | pass | pass | |
| T11 validation errors | R2, R6 | pass | pass | identical validation |

**T2 and T3 are the strongest evidence.** They do not depend on a hand-picked loan: *if changing only the
records after t changes any feature, that feature uses the future* (spec.md, R5). T4 and T5 illustrate the same
failure on specific loans.

## How to run (from the repository root)

```bash
uv run python benchmark/run_pair4.py               # summary table; exit code 0 = both cases agree with labels
uv run pytest benchmark/pair4_leakage/valid -v      # expected: all pass
uv run pytest benchmark/pair4_leakage/faulty -v     # expected: T1-T5 fail
```

## Limitations and decisions

- **Definition differences from the baseline `build_snapshot()`** (spec.md, A4). The baseline clips
  `max_dq_12m` at 3, ignores "RA" in `max_dq_12m` and `n_dq_months_12m` (it treats "RA" as NaN), keeps
  `status_t` as a string, and builds features only for loans at risk at t. None of these affects the
  expected values here.
- **The baseline's own leakage assertion** (`assert (feat.max_feature_period <= t).all()`) checks the latest
  month in `hist`, which is already restricted to months ≤ t. It therefore cannot fail, and would not catch
  this pair's defect if it were introduced into the 12-month window. A future-invariance test (T2/T3)
  would catch it. This is an observation; the baseline was not changed.
- **Status "XX" (Not Available)** has no severity in this specification and is rejected with `ValueError`.
  The 2015 sample has no "XX" records.
- **Month t is treated as known at the prediction date.** This follows the project convention (spec.md, A2).
- **Label blindness:** folder names, case IDs, labels and tests reveal the answer. A future validator should
  see only `impl.py` and `spec.md`, under neutral names.

## Data

One synthetic panel with the fictional loan IDs F01–F10. No Freddie Mac data are used.

# TASK: Implement MRM-Bench Pair 4 — Future-Information Leakage Defect

> **How to use:** start a NEW Claude Code conversation and type:
> `Read CLAUDE.md and docs/claude_code_task_pair4.md, then carry out the task phase by phase.`

## ROLE

Act as a senior Python machine learning engineer, quantitative model validation researcher, and academic research software developer, contributing to **MRM-Bench: Assumption-Grounded Executable Verification of Credit Risk Models Using LLM Agents** (Master's research, UNT, supervised by Dr. Chih-Hao Ku).

Pairs 1–3 are finished, verified and committed. **Follow the Pair 1/Pair 2 conventions:** identical specs, implementations differing by exactly one line, test files identical except the import line, `label.json`, a runner that only counts assertion failures as defect evidence, and the data-hygiene check. Do not modify Pairs 1–3.

---

# PHASE 0 — READ FIRST

1. Read `CLAUDE.md` completely (especially "Outcome specification").
2. Read `benchmark/pair2_formula/` and `benchmark/pair3_horizon/` (README, spec, impl, test file), `benchmark/run_pair3.py` and `benchmark/test_no_real_data.py`.
3. Read `build_snapshot()` in `src/baseline_logreg.py`, especially how `hist`, `h12`, `max_dq_12m`, `n_dq_months_12m` and `ever_90_before_t` are built, and the leakage assertion. Pair 4 mirrors these features. Do not modify the file.
4. Run `git status`.

Do not write implementation code until this is done.

**Protected (do not modify):** `src/baseline_logreg.py`; all of `benchmark/pair1_preprocessing/`, `pair2_formula/`, `pair3_horizon/`, `run_pair1.py`, `run_pair2.py`, `run_pair3.py`, `test_no_real_data.py`; `data/`, `outputs/`. If something seems to need changing, report it instead.

---

# PHASE 1 — THE RULE

**No future information:** every feature for prediction month t may use only records with month ≤ t (CLAUDE.md: "Features: origination fields + performance records with period <= t ONLY").

Leakage is dangerous because it makes a model look much better in testing than it can ever be in use. For example, a feature that "sees" a loan's delinquency in month t+1 almost reveals the outcome being predicted.

Features in this pair (mirroring the baseline):
- `status_t`: delinquency severity in month t;
- `max_dq_12m`: maximum severity over the 12 months **t−11 … t** (t included, t−12 excluded);
- `n_dq_months_12m`: number of months in t−11 … t with severity ≥ 1;
- `ever_90_before_t`: 1 if any month ≤ t has severity ≥ 3, else 0.

Severity: `int(delinq_status)` for numeric codes; `"RA"` maps to 99. No RA appears in the test data, but handle it identically in both versions.

**Important nuance (the false-alarm trap):** using month t itself is **not** leakage. Its status is known at the end of month t, which is the prediction date (CLAUDE.md: prediction date = end of month t). A validator that flags `month <= t` as "leakage because it includes t" is producing a false positive. Leakage starts at t+1.

---

# PHASE 2 — DESIGN

```text
benchmark/
├── pair4_leakage/
│   ├── README.md   __init__.py
│   ├── faulty/  __init__.py  spec.md  impl.py  label.json  test_case.py
│   └── valid/   __init__.py  spec.md  impl.py  label.json  test_case.py
└── run_pair4.py
```
Synthetic data only.

---

# PHASE 3 — IMPLEMENTATIONS

## Common interface
```python
build_features(panel, prediction_month) -> pandas.DataFrame
```
- `panel`: DataFrame with `loan_id` (str), `month` (int, a consecutive month index), `delinq_status` (str), `zero_balance_code` (str or None; only on the last record).
- Output rows: loans with a record at month t **and** no zero-balance code at or before t, indexed by sorted `loan_id`.
- Output columns, in this order: `status_t`, `max_dq_12m`, `n_dq_months_12m`, `ever_90_before_t`, all int.
- Must not modify its input.
- Identical validation in both files (`ValueError`): missing columns; duplicate (`loan_id`, `month`); `prediction_month` not in the panel; unrecognized status codes.

## Valid (`valid/impl.py`)
Build the 12-month window as
```python
window = panel[(panel["month"] > t - 12) & (panel["month"] <= t)]
```
and compute `max_dq_12m` and `n_dq_months_12m` from `window`. Compute `ever_90_before_t` from `panel[panel["month"] <= t]`, and `status_t` from the month-t record.

## Faulty (`faulty/impl.py`)
Identical except for the window line, which lacks the upper bound:
```python
window = panel[panel["month"] > t - 12]
```
This is a realistic mistake ("last 12 months" written as a lower bound only). The window then includes every month after t, so `max_dq_12m` and `n_dq_months_12m` see the future. `status_t` and `ever_90_before_t` are unaffected.

Constraints:
- No comments or names that reveal the defect (no "leak", "bug", "future", "wrong"). Give the window line the same neutral comment in both files, e.g. `# 12-month lookback window`.
- The faulty version must run normally and return wrong numbers, not raise.
- `diff faulty/impl.py valid/impl.py` must show only the window line.

---

# PHASE 4 — SPECIFICATION (`faulty/spec.md` = `valid/spec.md`, byte-identical)

Include:
1. Purpose: prediction-time features for the 12-month 90+ DPD logistic regression.
2. Feature definitions (Phase 1) with exact month ranges.
3. **Information-set rule:** features at t may use only records with month ≤ t. Month t is allowed (end-of-month prediction date); month t+1 or later is forbidden.
4. **Why leakage is wrong:** the outcome is 90+ in months t+1…t+12, so a feature using those months contains part of the outcome. Illustrate with loan F03 (Phase 6): current through t, delinquent from t+1. Its features must show no delinquency at t. A leaky version reports max 8, effectively encoding that the loan became seriously delinquent.
5. **What is NOT a defect:**
   - using month t;
   - a loan whose history is shorter than 12 months (window uses available months);
   - loans that terminate after t (they are included with features from ≤ t);
   - a non-monotone status history (cure from 03 to 00).
6. **Executable leakage criterion (metamorphic property):** deleting or changing any record with month > t must not change any feature at t.
7. Applicability conditions and provenance: CLAUDE.md outcome specification; `src/baseline_logreg.py` (`build_snapshot`: `hist = perf[perf.t <= t]`, leakage assertion); Freddie Mac General User Guide, Release 47 (monthly reporting period, delinquency status, pp.10 and 16 of `data/raw/general_user_guide_july_2026.pdf`). Confirm the page references in the PDF; do not invent quotes.

---

# PHASE 5 — LABELS

- `faulty/label.json`: `"case_id": "P4_F"`, `"pair_id": "P4"`, `"category": "leakage"`, `"is_defect": true`, `"violated_rule": "Features at prediction month t must use only records with month <= t; the 12-month window has no upper bound and includes months after t."`, and evidence naming F03 (max_dq_12m 8 instead of 0), F09 (1 instead of 0), and the failed future-invariance tests.
- `valid/label.json`: `"case_id": "P4_V"`, `"is_defect": false`, `"violated_rule": null`, and evidence that all features match and are invariant to future records.
- `"source"` in both: `"Project outcome specification (CLAUDE.md); src/baseline_logreg.py build_snapshot; Freddie Mac SFLLD General User Guide, Release 47 (July 2026)"`.

---

# PHASE 6 — SYNTHETIC DATA AND EXACT EXPECTED VALUES (pre-computed; do not change)

Prediction month **t = 12**. Months 0–20 unless stated. IDs `F01`…`F10` are synthetic and must not match the Freddie Mac loan-ID pattern.

| loan | statuses by month | zero-balance | tests |
|---|---|---|---|
| F01 | "00" m0–20 | – | baseline |
| F02 | "00" m0–4, "01" m5, "00" m6–20 | – | delinquency inside the lookback |
| F03 | "00" m0–12, then "01","02",…,"08" m13–20 | – | **future delinquency** (key leak case) |
| F04 | "02" m0, "00" m1–20 | – | month t−12 = 0 is outside the lookback |
| F05 | "00" m0–11, "01" m12, "00" m13–20 | – | month t itself is included (trap) |
| F06 | "00" m0–1, "01" m2, "02" m3, "03" m4, "00" m5–20 | – | 90+ in the past, then cured |
| F07 | "00" m0–15 | "01" on m15 | terminates after t, so still included |
| F08 | "00" m0–10 | "01" on m10 | terminated before t, so omitted |
| F09 | "00" m0–12, "01" m13, "00" m14–20 | – | delinquency at exactly t+1 (boundary) |
| F10 | "00" m8–20 (first record at month 8) | – | short history |

### Expected output (correct = valid)

| loan | status_t | max_dq_12m | n_dq_months_12m | ever_90_before_t |
|---|---|---|---|---|
| F01 | 0 | 0 | 0 | 0 |
| F02 | 0 | 1 | 1 | 0 |
| F03 | 0 | 0 | 0 | 0 |
| F04 | 0 | 0 | 0 | 0 |
| F05 | 1 | 1 | 1 | 0 |
| F06 | 0 | 3 | 3 | 1 |
| F07 | 0 | 0 | 0 | 0 |
| F09 | 0 | 0 | 0 | 0 |
| F10 | 0 | 0 | 0 | 0 |

F08 is omitted. For reference only, **not** to be asserted: the faulty version differs only on F03 (max 8, count 8) and F09 (max 1, count 1).

---

# PHASE 7 — TESTS (identical files except the import line)

- T1 Exact expected output (all rows and columns, dtype int) on the panel above.
- T2 **Future invariance (truncation):** the output on the full panel equals the output on the panel with all rows month > t removed.
- T3 **Future invariance (perturbation):** the output is unchanged when every status with month > t is replaced by "05".
- T4 Key case F03: `max_dq_12m == 0` and `n_dq_months_12m == 0`.
- T5 Boundary t+1 (F09): `max_dq_12m == 0`.
- T6 Month t included (F05): `status_t == 1`, `max_dq_12m == 1`, `n_dq_months_12m == 1`.
- T7 Lookback lower boundary (F04): month t−12 is excluded, `max_dq_12m == 0`.
- T8 Past 90+ (F06): `ever_90_before_t == 1`, `max_dq_12m == 3`, `n_dq_months_12m == 3`.
- T9 Row set: F08 omitted; F07 and F10 included; index sorted.
- T10 Input not modified.
- T11 Validation: identical `ValueError`s for missing columns, duplicate (loan_id, month), t not in panel, unknown status code (e.g. "ZZ").

**Expected outcome:** valid passes all. Faulty **fails T1–T5** and **passes T6–T11**. Any other pattern means the pair is wrong; fix the implementation or tests, never the expected values.

T2 and T3 are the **general** leakage tests (metamorphic). Mark them in the docstrings as the strongest evidence, because they do not depend on any single hand-picked loan.

---

# PHASE 8 — RUNNER `benchmark/run_pair4.py`

Same design as `run_pair3.py` (minus the horizon columns): each suite in its own process; only assertion failures count as defect evidence; summary table; data-hygiene check; exit 0 only if both cases agree with their labels.

---

# PHASE 9 — RUN

```bash
uv run pytest benchmark/pair4_leakage/valid -v
uv run pytest benchmark/pair4_leakage/faulty -v      # expected assertion failures
uv run pytest benchmark/test_no_real_data.py -v
uv run python benchmark/run_pair4.py
uv run python benchmark/run_pair1.py && uv run python benchmark/run_pair2.py && uv run python benchmark/run_pair3.py
```
No new dependencies.

---

# PHASE 10 — INDEPENDENT VALIDATION

1. Write a separate, loop-based reference feature builder in your scratchpad (not copied from `impl.py`), and reproduce the expected table.
2. Check that `impl.py` (valid) agrees with how `build_snapshot()` in `src/baseline_logreg.py` computes `max_dq_12m`, `n_dq_months_12m` and `ever_90_before_t` on this panel. Note: the baseline clips `max_dq_12m` at 3 and computes `n_dq_months_12m` from numeric statuses. Report any differences in definition instead of hiding them. (The expected table has no value above 3 for the valid case, so clipping does not change it.)
3. Confirm `diff` of the impl files = the window line only; test files = the import line only; specs byte-identical.
4. Confirm the faulty version fails exactly T1–T5.
5. Data hygiene; protected files unchanged (`git diff --stat`); all four runners exit 0.

---

# PHASE 11 — REPORT BACK

A. File tree with a simple-English explanation of each file.
B. Explain, using F03, F05 and F09, why the faulty window leaks, why including month t is not leakage, and why T2/T3 detect any leak, not just these loans.
C. Actual outputs of all Phase 9 commands.
D. Research interpretation: why P4_F is defective, why P4_V is valid (and is a false-positive trap), what the pair proves and does NOT prove.
E. Repository checks, limitations, and any definition differences from `build_snapshot()`.
F. Suggested commit message. Do not commit unless I ask.

---

# ACCEPTANCE CRITERIA
- Same interface and validation; the impl files differ only in the window line; specs byte-identical; test files differ only in the import line.
- Valid passes all; faulty fails exactly T1–T5 as assertion failures.
- Expected values match the table and were independently recomputed.
- Synthetic data only; hygiene passes; protected files unchanged; all four runners exit 0.
- Report includes real outputs and a commit message.

**Do not begin VALID-Agent, RAG, XGBoost, or more pairs. This task is limited to Pair 4.**

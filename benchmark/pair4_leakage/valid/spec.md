# Specification — MRM-Bench Pair 4: prediction-time features (no future information)

This specification is the same for both implementations in Pair 4. It states the required behaviour of
`build_features(panel, prediction_month)`. The ground-truth labels and the tests in `test_case.py` are
derived from it, and the tests cite the rules R1–R6 below.

---

## 1. Purpose

The project's baseline logistic regression predicts the **12-month 90+ DPD outcome**: whether a loan is 90+
days delinquent, or is acquired through REO ("RA"), in some month t+1 … t+12 after the prediction month t
(CLAUDE.md, "Outcome specification"). `build_features` computes four delinquency-history features at t
that are inputs to that model. They mirror the features of the same names in the baseline's
`build_snapshot()`. The remaining definition differences are listed in §7.

---

## 2. Feature definitions

Notation: t is the prediction month. Months are consecutive integers, so t−1 is the previous calendar month.
s_i(m) is the severity of loan i in month m, defined only for months in which the loan has a record.

- **R1 — Row set.** There is one output row per loan that has a record at month t and no zero-balance code
  at or before t. The index is `loan_id`, sorted. There is no filter on the status at t.
- **R2 — Severity.** s = the integer value of a two-digit numeric status code ("00" … "99"); "RA" (REO
  acquisition) counts as 99. Any other code is invalid and raises `ValueError`. That includes "XX" (Not
  Available): the guide documents it, but this specification defines no severity for it. (The project's
  2015 sample contains no "XX" records.)
- **R3 — Features** (integers, in this column order):

| feature | definition | months used |
|---|---|---|
| `status_t` | s_i(t) | t |
| `max_dq_12m` | max { s_i(m) : t−11 ≤ m ≤ t } | t−11 … t (t included, t−12 excluded) |
| `n_dq_months_12m` | number of months m with t−11 ≤ m ≤ t and s_i(m) ≥ 1 | t−11 … t |
| `ever_90_before_t` | 1 if some month m ≤ t has s_i(m) ≥ 3, else 0 | every month ≤ t |

Months without a record are skipped, so a short history uses only the months that are available. Every
output loan has a record at t, so the lookback is never empty.

---

## 3. Information-set rule

**R4 — No future information.** Features at t may use only records with month ≤ t. Every feature in R3, and
the row set in R1, must be a function of the loan's records in months ≤ t alone.

- **Month t is allowed.** The project's prediction date is the end of month t (CLAUDE.md), and the month-t
  record ("the as-of month for loan information contained in the record", guide p. 10) is part of the
  information available at that date. Treating month t as "future" is a false alarm.
- **Months t+1, t+2, … are forbidden.**

---

## 4. Why leakage is wrong

The outcome is "90+ in some month t+1 … t+12". A feature that reads those months contains part of the
outcome it is supposed to predict.

Take loan **F03** (synthetic): current ("00") through t = 12, then "01", "02", …, "08" in months 13–20. At
the prediction date nothing is wrong with it. A lookback written with only a lower bound (months > t−12,
no upper bound) also reads months 13–20:

| loan | history around t = 12 | `max_dq_12m` / `n_dq_months_12m` required | with no upper bound |
|---|---|---|---|
| F03 | "00" to month 12, "01" … "08" in months 13–20 | 0 / 0 | **8 / 8** |
| F09 | "00" to month 12, "01" in month 13 only | 0 / 0 | **1 / 1** (one month after t is enough to leak) |
| F05 | "01" in month 12 = t | 1 / 1 | 1 / 1 (month t is legitimately included) |

For F03 the leaky `max_dq_12m` is 8 ≥ 3. That means the loan reaches 90+ after t, which is the event the
model is meant to predict. In general, for a loan with no 90+ month in t−11 … t, the leaky max is ≥ 3
exactly when some month after t is 90+. For months t+1 … t+12, that is the outcome itself.

The damage is that the model looks better in testing than it can be in use. During training and back-testing
the panel contains months after t, so the leaky feature "knows" the outcome. When the model is used, the
records after t do not exist yet, and the same code silently returns the non-leaky value. The model is
therefore fitted and validated on information it will never have.

---

## 5. What is NOT a defect

- **Using month t** (F05: "01" at t gives `status_t` = 1, `max_dq_12m` = 1, `n_dq_months_12m` = 1).
- **A history shorter than 12 months.** F10's first record is at month 8, so its lookback uses months 8–12.
- **A loan that terminates after t.** F07 has zero-balance code 01 at month 15 > t. It is included, with
  features from months ≤ t only.
- **Omitting a loan that terminated before t.** F08 has a zero-balance code at month 10 and no record at t.
- **A non-monotone history (cure).** F06 is "03" at month 4 and then "00" again. Its features are
  `max_dq_12m` = 3, `n_dq_months_12m` = 3 and `ever_90_before_t` = 1. The project also lists status jumps
  as a known false-alarm trap (CLAUDE.md).
- **Leaving out month t−12.** F04's "02" at month 0 = t−12 lies outside t−11 … t, so `max_dq_12m` = 0.
  This is the definition, not a lost record.

---

## 6. Executable leakage criterion (metamorphic property)

**R5 — Future invariance.** Let P be a valid panel. Let P′ be any valid panel that agrees with P on every
record with month ≤ t; records after t may be deleted, added or changed. Then
`build_features(P, t)` = `build_features(P′, t)`, i.e. every row and every value is the same.

*Why this is the right test.* R4 implies R5: if the output depends only on records ≤ t, changing later
records cannot change it. Conversely, if some P′ changes the output, then the output depends on a record
after t, which is a violation of R4. So R5 tests the information-set rule for **every** loan and feature
at once, not just for hand-picked loans. The tests check it for two choices of P′:

- every record after t removed (truncation);
- every status after t set to "05" (perturbation).

A pass means no dependence on the future was found for those perturbations. A failure is a proof of leakage.

**R6 — Interface and validation.** The input panel is not modified. The function raises `ValueError` for
missing required columns (`loan_id`, `month`, `delinq_status`, `zero_balance_code`), duplicate (`loan_id`,
`month`) rows, a prediction month with no records, or a status code that is invalid under R2. Validation
applies to the whole panel.

---

## 7. Applicability conditions and provenance

**Applicability conditions**

- **A1.** The panel follows the Freddie Mac SFLLD monthly performance data (Release 47). `month` is a
  consecutive month index, such as the baseline's `ym_index`, not a raw YYYYMM value.
- **A2.** *Prediction-date convention.* As in the project, the month-t record is treated as available at the
  prediction date (end of month t). The guide notes that a monthly record can combine data from several
  reporting cycles, and that for accounting cycles in April 2019 and earlier the cycle for scheduled payments
  ran from the 16th of a month to the 15th of the next (p. 16). A deployment that had to score loans before
  the month-t record is published would need a stricter rule (months ≤ t−1). This specification follows the
  project convention.
- **A3.** This specification covers the features only. The outcome label is defined in CLAUDE.md and in
  Pair 3.
- **A4.** *Definition differences from `build_snapshot()`.* These are deliberate in this pair, and none of
  them affects the expected values in `test_case.py`:
  - The baseline clips `max_dq_12m` at 3. This specification does not clip it.
  - The baseline converts statuses with `pd.to_numeric(..., errors="coerce")`. "RA" therefore becomes NaN
    and is ignored in `max_dq_12m` and `n_dq_months_12m`, but it does count in `ever_90_before_t`. Here
    "RA" = 99 in all features.
  - The baseline keeps `status_t` as the code string ("00", "01", "02"). Here it is the integer severity.
  - The baseline builds features only for loans at risk at t, i.e. it also excludes loans that are 90+ or
    "RA" at t. Here the row set is R1.

**Provenance**

- **CLAUDE.md, "Outcome specification (draft, pending professor approval)":** "Prediction date t: end of each
  December 2015..2024 (loan-snapshot design)." "Features: origination fields + performance records with
  period <= t ONLY." "Event (y=1): any month t+1..t+12 with numeric status >= 03 or status 'RA'."
- **`src/baseline_logreg.py`, `build_snapshot()` (commit 9c743e2):**
  - `hist = perf[perf.t <= t]` (commented "ONLY information at or before t");
  - `h12 = hist[(hist.t > t - 12) & hist.loan_id.isin(ids)]`;
  - `max_dq_12m`, `n_dq_months_12m` and `ever_90_before_t` are built from `h12` and `hist`;
  - the leakage assertion `assert (feat.max_feature_period <= t).all()`.
- **Freddie Mac, *Single-Family Loan-Level Dataset — General User Guide*, Release 47, July 2026.** Local copy:
  `data/raw/general_user_guide_july_2026.pdf` (gitignored). The text was extracted with `pypdf` on
  2026-10-22, and the quotations below were copied from it.
  - **p. 10, "Monthly Performance Data File":** "Period": "The as-of month for loan information contained in the
    record." "Current Loan Delinquency Status": codes "00 = Current, or less than 30 days delinquent", "01 =
    30-59 days delinquent", "02 = 60 – 89 days delinquent", "03 = 90 – 119 days delinquent etc.", "RA = REO
    Acquisition", "XX = Not Available".
  - **p. 16, "Interpreting the Data" → "Monthly Reporting Period":** "For a given loan, each period in the
    monthly performance data file may combine data elements from multiple reporting cycles and systems at
    Freddie Mac." "For accounting cycles April 2019 and prior, Freddie Mac’s accounting cycle begins on the
    16th of each month and ends on the 15th of the following month for scheduled Principal and Interest
    (P&I) payments."
- **Executable evidence:** `test_case.py` in this folder. It uses one synthetic panel (loan IDs F01–F10, t =
  12), and its expected values were computed in advance and rechecked with a separate reference builder.

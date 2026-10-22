# Specification — MRM-Bench Pair 1: Original DTI Ratio preprocessing

This specification is the same for both implementations in Pair 1. It states the required behaviour
of `prepare_dti(df, training_median=None)`. The ground-truth labels and the tests in `test_case.py`
are derived from it; the tests cite the rules below as R1–R7.

---

## 1. Official documented rule

**Source:** Freddie Mac, *Single-Family Loan-Level Dataset — General User Guide*, Release 47, July 2026.
Section "Glossary" → "Origination Data File", attribute **Original Debt-to-Income (DTI) Ratio**, PDF page 5.

Quoted text (copied from the PDF's extracted text; see §5 for how this was checked):

- Description: "The ratio obtained by dividing the total monthly debt expense by the total monthly
  income of the borrower at the time the loan was originated."
- Description: "Ratios > 65% will be disclosed as Not Available. All HARP loans in the dataset will be
  disclosed as Not Available."
- Enumerations / Calculations: "0% - 65%", "999 = Not Available", and for HARP loans
  "HARP: 999 = Not Available".

**In plain English.** DTI is a percentage between 0 and 65 (36 means debts are 36% of income).
The number 999 is not a percentage. It is a code that means "no value is disclosed for this loan". That happens
because the ratio was unavailable, because it was above 65% (such ratios are never disclosed), or because the
loan is a HARP refinance (all HARP loans get 999).

Data context (aggregate counts only, from the project's earlier verified data check; no records copied):
in the 2015 sample, 4,020 of 50,000 loans (8.04%) have DTI = 999, and 4,017 of those are HARP loans.

---

## 2. Required behaviour (correct mathematical preprocessing)

Notation: rows i = 1, …, n of the input; x_i is the raw `orig_dti` value of row i.

**R1 — Required column.** `df` must contain an `orig_dti` column; otherwise raise `ValueError`.

**R2 — Legal input domain.** Each x_i must be one of:
- a number in the closed interval [0, 65] (the boundaries 0 and 65 are legal and are kept unchanged);
- the documented code 999;
- a genuine null (`None`, `NaN`).

Numeric text such as `"32"` or `"999"` (as read from the raw pipe-delimited file) counts as the
corresponding number. Any other value (for example 66, 70, −5, 500, 9999, `"abc"`, or an empty string) is
an invalid input and must raise `ValueError`. Invalid inputs are never silently accepted or repaired.

**R3 — Missing indicator.**

    m_i = 1   if x_i is null or x_i = 999
    m_i = 0   otherwise (x_i is an observed ratio in [0, 65])

**R4 — Training median** (training mode, `training_median=None`). Let O = { x_i : m_i = 0 } be the
observed ratios of the training data. Sort them as o_(1) ≤ … ≤ o_(k). Then

    M = o_((k+1)/2)                         if k is odd
    M = ( o_(k/2) + o_(k/2+1) ) / 2         if k is even

If O is empty (empty DataFrame, or every row missing), no median can be fitted and the function must
raise `ValueError`. It must not fall back to an arbitrary value.

**R5 — Inference** (validation/test mode, `training_median` supplied). M is the supplied value, which
was fitted on training data. It must be a finite number in [0, 65], otherwise `ValueError`. **No median
is computed from the validation/test rows.**

**R6 — Output values.**

    x̃_i = x_i   if m_i = 0
    x̃_i = M     if m_i = 1

The returned DataFrame holds x̃ in `orig_dti` (float) and m in `dti_missing` (integer 0/1), and stores M in
`result.attrs["dti_training_median"]` (float).

**R7 — Structure.** The function returns a new DataFrame with the same number of rows, the same row order
and the same index as the input. Existing columns keep their order, all columns other than `orig_dti` are
unchanged, and `dti_missing` is appended. The input DataFrame, including its `attrs`, is not modified.

**Consequence: range invariant.** Every processed value lies in [0, 65].
*Proof.* If m_i = 0, then x̃_i = x_i, which lies in [0, 65] by R2, because 999 always has m_i = 1. If m_i = 1, then x̃_i = M.
In training mode, M is a median of numbers in [0, 65]. A median is either one of those numbers or the
midpoint of two of them, so it also lies in [0, 65]. In inference mode, M ∈ [0, 65] by R5. ∎

**Why a median plus an indicator.** This is the design chosen by this specification and by the project's
clean baseline. The indicator keeps the information "DTI was not disclosed" (mostly HARP loans) available to
the model. The median fills the numeric column with a typical value so that every row has a number.

**Interface note.** `prepare_dti` returns a DataFrame, so the fitted median is passed back through
`DataFrame.attrs`. `attrs` is an experimental pandas feature that some operations (such as `concat` and
`merge`) may drop. Read the median from the returned DataFrame immediately and pass it explicitly as
`training_median` when processing validation or test data.

---

## 3. Why treating 999 as a DTI value violates this specification

Worked example with the fictional training values `[10, 20, 30, 40, 999, 999]` (synthetic, not Freddie
Mac records):

| quantity                         | correct: 999 is missing     | incorrect: 999 used as a number |
|----------------------------------|-----------------------------|---------------------------------|
| values used for statistics       | 10, 20, 30, 40              | 10, 20, 30, 40, 999, 999        |
| median M                         | (20 + 30) / 2 = **25**      | (30 + 40) / 2 = **35**          |
| mean                             | 25.00                       | 349.67                          |
| sample standard deviation        | 12.91                       | 503.07                          |
| `dti_missing`                    | 0, 0, 0, 0, 1, 1            | 0, 0, 0, 0, 0, 0                |
| processed `orig_dti`             | 10, 20, 30, 40, 25, 25      | 10, 20, 30, 40, 999, 999        |
| z-score of a 40% DTI             | +1.16 (above average)       | −0.62 (looks below average)     |

These statistics were computed with Python's `statistics` module, independently of both
implementations.

1. **Domain violation (R2, R6).** 999 is outside [0, 65]. A DTI of 999% would mean monthly debt payments of
   about ten times monthly income. The guide says ratios above 65% are never disclosed, so 999 cannot be a
   real measurement. An output containing 999 breaks the range invariant.
2. **Wrong missing indicator (R3).** The HARP and not-disclosed loans get `dti_missing = 0`. The model can no
   longer tell "DTI not disclosed" apart from "DTI disclosed".
3. **Code-invariance argument.** 999 is a label, and the number chosen for it is arbitrary. A statistic that
   describes borrowers must not change if the publisher had picked a different label. The correct median
   is 25 whatever the code is. If the code is used as a number, the median is 35 with code 999 and 15 with
   code −1. That statistic depends on the labelling, not on the borrowers, so it is not a valid DTI statistic.
4. **Downstream distortion.** When the column is standardised, the mean and standard deviation are dominated
   by the code. A 40% DTI, which is above the typical value, gets a negative z-score (−0.62). The
   spread of the real ratios shrinks by a factor of about 12.91 / 503.07 ≈ 0.026. In logistic regression,
   z = b0 + b·x + …, the coded loans enter with x = 999, about 40 times the typical value of 25. They are extreme
   leverage points that pull the fitted DTI coefficient. **The direction and size of the change in the
   coefficients depend on the data.** They depend on the outcome rate of the coded loans, their correlation
   with other features, scaling and regularisation. This specification does **not** claim that any
   coefficient moves in a particular direction. The defect is established by R2–R6, not by a coefficient
   change.

---

## 4. Applicability conditions

This specification, and therefore the labels of this pair, applies when all of the following hold:

- **A1.** The column is the Freddie Mac SFLLD field "Original Debt-to-Income (DTI) Ratio" (`orig_dti`), as
  documented in Release 47 (July 2026).
- **A2.** DTI is stored as a percentage (36 = 36%), not as a proportion (0.36). With a proportion
  encoding, the range [0, 65] and the code 999 would not describe the data and this specification would
  not apply.
- **A3.** 999 is designated "Not Available" (including all HARP loans), and ratios above 65% are disclosed
  as 999.
- **A4.** The median is fitted on training data only, before it is applied to validation or test data. In
  inference mode, the supplied `training_median` comes from a training run.
- **A5.** The chosen missing-data design is median imputation plus a missing indicator.

**Outside these conditions** the rule may not apply. For example:

- Other datasets or releases may use different codes or ranges. Check that release's guide instead of
  assuming 999.
- A pipeline that keeps missing DTI as NaN for a model that handles missing values natively, or that uses a
  different imputation statistic, can also be valid if its specification says so. Median imputation itself
  is not what this pair tests. The only violation tested here is "the code 999 is treated as a measurement."
- This pair does not judge whether DTI is a good predictor, coefficient sizes, model accuracy, or
  distribution shift.

**False-alarm traps** (valid behaviour that can look suspicious):

- A DTI of exactly 65 or exactly 0 is a legal value and must be kept.
- A large block of missing DTI values concentrated in HARP loans is expected and documented.
- Filling missing validation or test values with the *training* median is required by R5. It is not
  leakage.

---

## 5. Evidence and provenance

- **Document:** Freddie Mac, *Single-Family Loan-Level Dataset — General User Guide*, Release 47, July 2026
  (title page wording).
- **Section:** "Glossary" → "Origination Data File" (the table of contents lists both on page 4).
- **Attribute:** "Original Debt-to-Income (DTI) Ratio". The "Description" column and the "Enumerations /
  Calculations" column are quoted in §1.
- **Page:** PDF page 5 (printed page number 5).
- **Local copy:** `data/raw/general_user_guide_july_2026.pdf`. It is gitignored and not redistributed with the
  benchmark.
- **How the quotations were checked:** the PDF text was extracted with `pypdf` on 2026-10-22, and the
  quotations in §1 were copied from that text.
- **Aggregate data context:** 4,020 / 50,000 loans with DTI 999, of which 4,017 are HARP. These counts come
  from the project's earlier verified data check (`CLAUDE.md`), and no loan-level records are reproduced.
- **Executable evidence:** `test_case.py` in this folder. It uses small synthetic data with exact expected
  values that were derived by hand from R1–R7.

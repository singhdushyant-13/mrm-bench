# MRM-Bench Pair 1 — Preprocessing: Original DTI Ratio code 999

**Category:** preprocessing  **Pair ID:** P1  **Cases:** `P1_F` (faulty/, defect) and `P1_V` (valid/, no defect)

## What this pair tests

The Freddie Mac loan-level dataset reports a borrower's debt-to-income ratio (DTI) as a percentage
from 0 to 65. When no ratio is disclosed, the field holds the code **999** instead. This covers HARP loans
and ratios above 65%. The code is a label meaning "Not Available"; it is not a number of percent.

Both implementations below look almost the same and run without errors. One of them handles 999 correctly
by turning it into a missing value and flagging it. The other uses 999 as if it were a real ratio of 999%.
A model validator (human or AI) should flag the second and **not** flag the first.

## Files

| file | what it is |
|------|------------|
| `faulty/spec.md`, `valid/spec.md` | The specification both implementations must satisfy: the documented rule, the required behaviour as rules R1–R7, the maths, the applicability conditions and the provenance. **The two files are identical.** |
| `faulty/impl.py`, `valid/impl.py` | The implementation of `prepare_dti(df, training_median=None)`. The two files differ only in the step that turns 999 into missing (see below). |
| `faulty/label.json`, `valid/label.json` | Ground truth set by the benchmark designers: `is_defect`, the violated rule (or `null`), evidence and source. |
| `faulty/test_case.py`, `valid/test_case.py` | Executable tests of spec.md on small synthetic data with exact expected values. **The two files are identical except for the import line.** |
| `__init__.py` (here and in each sub-folder) | Empty package markers. Both cases use the same file names (`impl.py`, `test_case.py`), so without these pytest cannot collect both suites in one run and the two `impl` modules could be confused with each other. |

Related files one level up: `benchmark/run_pair1.py` runs both suites and compares the outcomes with the labels.
`benchmark/test_no_real_data.py` checks that no Freddie Mac loan IDs or large files are in `benchmark/`.

## The single difference between the implementations

```diff
--- valid/impl.py
+++ faulty/impl.py
@@ -59,9 +59,6 @@
         raise ValueError(f"prepare_dti: orig_dti values outside [0, 65] and not 999: "
                          f"{raw[~legal].unique()[:5].tolist()}")
 
-    # 999 is the documented "Not Available" code (all HARP loans carry it), not a ratio -> missing
-    dti = dti.mask(dti == DTI_NOT_AVAILABLE)
-
     missing = dti.isna()
     if training_median is None:                   # training: fit the median on observed values only
         if missing.all():
```

Everything else is the same in both files: input validation (both accept [0, 65], 999 or null and raise the
same `ValueError` for anything else), the train/inference interface, imputation and output columns. The
faulty version still treats 999 as a *legal input* because it is documented, but then uses it as a number.

## Expected test outcomes (answer key)

On the synthetic training data `orig_dti = [10, 20, 30, 40, 999, 999]`:

| | correct (spec) | faulty implementation |
|---|---|---|
| processed `orig_dti` | 10, 20, 30, 40, 25, 25 | 10, 20, 30, 40, 999, 999 |
| `dti_missing` | 0, 0, 0, 0, 1, 1 | 0, 0, 0, 0, 0, 0 |
| `attrs["dti_training_median"]` | 25 | 35 |

`P1_F` is expected to fail exactly the tests marked **[999 rule]** in `test_case.py`, all as assertion
failures. These are the six-row outputs (values, median, indicator, range), the raw-text input containing
`"999"`, inference on `[999, 12, 65]`, and the all-999 training case. The key evidence is
`test_training_processed_dti`. All tests that do not involve 999 pass on both implementations. Examples are
the boundaries 0 and 65, genuine nulls, inference that must not refit the median, input immutability,
index preservation, and invalid-input errors. This supports the claim that the faulty version has a
single defect.

## How to run (from the repository root)

```bash
uv run python benchmark/run_pair1.py                 # summary table; exit code 0 = both cases agree with labels
uv run pytest benchmark/pair1_preprocessing/valid -v  # expected: all pass
uv run pytest benchmark/pair1_preprocessing/faulty -v # expected: the [999 rule] tests fail
uv run pytest benchmark/pair1_preprocessing -v        # nonzero exit is EXPECTED (faulty case fails)
```

## Interface and limitations

- `prepare_dti` both fits the median (training mode) and applies it (inference mode) through one function. The
  fitted median is returned in `result.attrs["dti_training_median"]`. `attrs` is an experimental pandas
  feature that some operations (such as `concat` and `merge`) may drop, so read it right after the call and pass
  it explicitly as `training_median` for validation and test data.
- The pair isolates one rule on one field. It does not test the project's full baseline pipeline, which
  handles the same rule with its own code.
- **Label blindness:** `impl.py` and `spec.md` contain no hint of which case is faulty. The folder names
  (`faulty/`, `valid/`), case IDs, `label.json` and `test_case.py` do reveal it. A future validator must be
  shown only `impl.py` and `spec.md`, under neutral names.
- A side effect of the defect: in the faulty version, a training set that is mostly 999 can fit a "median"
  above 65, and the same file would reject that value if it were passed back as `training_median`. This is
  a consequence of the 999 defect, not a second defect.

## Data

All values in this folder are small, fictional, synthetic numbers. No Freddie Mac loan-level records are
included. The only real-data facts mentioned anywhere are aggregate counts (4,020 of 50,000 loans with
DTI 999; 4,017 of them HARP), which were recorded earlier by the project.

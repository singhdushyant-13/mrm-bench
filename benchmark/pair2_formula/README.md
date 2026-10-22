# MRM-Bench Pair 2 — Formula: sign of the logistic (sigmoid) exponent

**Category:** formula  **Pair ID:** P2  **Cases:** `P2_F` (faulty/, defect) and `P2_V` (valid/, no defect)

## What this pair tests

A logistic regression turns a loan's score z = b0 + x′b into a probability with p = 1 / (1 + exp(−z)).
A higher score must mean a higher probability of becoming 90+ days delinquent.

Both implementations below run without errors and differ in one line.

- **Faulty.** It writes the textbook-looking `1 / (1 + exp(z))`, with the minus sign missing. That formula
  gives 1 − p, the probability of *not* becoming delinquent. The riskiest loans therefore get the lowest
  probabilities.
- **Valid.** It uses the less familiar but mathematically identical `0.5 * (1 + tanh(z / 2))`. This form
  stays accurate even for extreme scores.

A model validator (human or AI) should flag the first and **not** flag the second. The valid version is
a deliberate false-alarm trap: it does not look like the textbook formula, but it is correct.

## Files

| file | what it is |
|------|------------|
| `faulty/spec.md`, `valid/spec.md` | The specification both implementations must satisfy. It gives the required formula, the properties P1–P6, the numerical requirement N1, the interface rules I1–I4, why a sign error is wrong, what is *not* a defect, the applicability conditions and the provenance. **The two files are identical.** |
| `faulty/impl.py`, `valid/impl.py` | The implementation of `predict_proba(X, coef, intercept)`. The two files differ only in the probability line and its comment (see below). |
| `faulty/label.json`, `valid/label.json` | Ground truth set by the benchmark designers: `is_defect`, the violated rule (or `null`), evidence and source. |
| `faulty/test_case.py`, `valid/test_case.py` | Executable tests of spec.md on small synthetic data with exact expected values, numbered [T1]–[T11]. **The two files are identical except for the import line.** |
| `__init__.py` (here and in each sub-folder) | Empty package markers, as in Pair 1. They let pytest import the two same-named `test_case.py` and `impl.py` files without name clashes. |

Related files one level up: `benchmark/run_pair2.py` runs both suites and compares the outcomes with the labels. It
has the same design as `run_pair1.py`. `benchmark/test_no_real_data.py` already scans all of `benchmark/`,
including this folder.

## The single difference between the implementations

```diff
--- valid/impl.py
+++ faulty/impl.py
@@ -49,6 +49,6 @@
 
     z = X @ coef + intercept                      # linear score b0 + x'b, one value per row
 
-    # logistic function via tanh identity; stable for large |z|
-    p = 0.5 * (1.0 + np.tanh(0.5 * z))
+    # logistic function
+    p = 1.0 / (1.0 + np.exp(z))
     return p
```

Everything else is the same in both files: input validation, the score line `z = X @ coef + intercept`, the
return type and the comment style.

## Expected test outcomes (answer key)

| test | property | `P2_V` (valid) | `P2_F` (faulty) | why |
|---|---|---|---|---|
| T1 test set A exact values | P6-style | pass | **fail** | e.g. z = 0.5 gives 0.3775 instead of 0.6225 |
| T2 σ(ln 3) = 0.75, σ(−ln 3) = 0.25 | P4 | pass | **fail** | gives 0.25 and 0.75; **key evidence** |
| T3 strictly increasing on [−20, 20] | P3 | pass | **fail** | decreasing at every step |
| T4 range [0, 1] | P1 | pass | pass | 1 − σ also lies in [0, 1] |
| T5 σ(0) = 0.5 | P2 | pass | pass | 1 − 0.5 = 0.5 |
| T6 p(z) + p(−z) = 1 | P5 | pass | pass | 1 − σ is also symmetric |
| T7 z = ±1000 finite, ≈ [1, 0] | N1 | pass | **fail** | finite but reversed: [0, 1] |
| T8 agrees with scikit-learn | P6 | pass | **fail** | max difference ≈ 0.998 |
| T9 inputs unchanged | I3 | pass | pass | |
| T10 output shape, empty input | I2 | pass | pass | |
| T11 invalid inputs raise ValueError | I1 | pass | pass | identical validation |

T4–T6 show why this pair is subtle: range, centre and symmetry tests alone cannot detect a sign error.
The faulty version prints a numpy overflow warning (`exp(1000)`) in T4 and T7. That warning is not what
makes T7 fail. T7 fails because the finite result [0, 1] is reversed.

## How to run (from the repository root)

```bash
uv run python benchmark/run_pair2.py             # summary table; exit code 0 = both cases agree with labels
uv run pytest benchmark/pair2_formula/valid -v    # expected: all pass
uv run pytest benchmark/pair2_formula/faulty -v   # expected: T1, T2, T3, T7, T8 fail
uv run pytest benchmark/pair2_formula -v          # nonzero exit is EXPECTED (faulty case fails)
```

## Limitations

- The pair isolates one formula. It does not check that `X` was preprocessed as during fitting (A4 in
  spec.md), and it does not test the project's baseline pipeline, which uses scikit-learn's own
  `predict_proba`.
- **Label blindness:** `impl.py` and `spec.md` contain no hint of which case is faulty. The folder names
  (`faulty/`, `valid/`), case IDs, `label.json` and `test_case.py` do reveal it. A future validator must be
  shown only `impl.py` and `spec.md`, under neutral names.
- The textbook form `exp(z) / (1 + exp(z))` was deliberately **not** used as the valid twin. It returns NaN
  at z = 1000 and would therefore carry a second, numerical defect (spec.md §5).

## Data

All values in this folder are synthetic: hand-chosen numbers (test sets A–C) and random draws from
`numpy.random.default_rng(0)` (test set D). No Freddie Mac data are used.

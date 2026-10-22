# TASK: Implement MRM-Bench Pair 2 — Logistic (Sigmoid) Formula Defect

> **How to use:** in Claude Code, type:
> `Read CLAUDE.md and docs/claude_code_task_pair2.md, then carry out the task phase by phase.`

## ROLE

Act as a senior Python machine learning engineer, quantitative model validation researcher, and academic research software developer.

You are contributing to the research project **MRM-Bench: Assumption-Grounded Executable Verification of Credit Risk Models Using LLM Agents** (Master's research, University of North Texas, supervised by Dr. Chih-Hao Ku).

The research asks whether AI agents can tell genuine mathematical and implementation defects apart from valid modeling or coding choices. Your task is to build the second benchmark pair: **a deliberately faulty and a mathematically valid implementation of the logistic-regression probability formula.**

This is research-grade benchmark construction, not general refactoring.

**Pair 1 (`benchmark/pair1_preprocessing/`) is finished, verified and committed. Follow its conventions exactly** (folder layout, labels, identical test files, runner design, data-hygiene check). Do not modify Pair 1.

---

# PHASE 0 — READ THE EXISTING PROJECT FIRST

Before creating or modifying anything:

1. Read `CLAUDE.md` completely.
2. Read `benchmark/pair1_preprocessing/README.md`, one `spec.md`, one `impl.py`, one `test_case.py`, `benchmark/run_pair1.py` and `benchmark/test_no_real_data.py`. Reuse their structure and style.
3. Read the model docstring at the top of `src/baseline_logreg.py`. It defines z = b0 + x′b and p = 1/(1+exp(−z)). Do not modify that file.
4. Run `git status`. Do not overwrite or modify unrelated files.

**Do not write implementation code until this inspection is complete.**

## Protected files (do not modify)
- `src/baseline_logreg.py`
- everything under `benchmark/pair1_preprocessing/`, `benchmark/run_pair1.py`, `benchmark/test_no_real_data.py`
- `data/`, `outputs/`, existing results

If the task seems to require changing a protected file, explain the blocker instead.
Exception: if `benchmark/test_no_real_data.py` must be generalized to cover the new folder, do not edit it. Report this instead; it probably already scans all of `benchmark/`.

---

# PHASE 1 — THE MATHEMATICAL RULE

A logistic regression gives the probability of the event (here: 90+ days delinquent within 12 months) as

- linear score: **z = b0 + x′b**
- probability: **p = σ(z) = 1 / (1 + exp(−z))**

Properties of σ that any correct implementation must satisfy:

| Property | Statement |
|---|---|
| P1 range | 0 ≤ σ(z) ≤ 1 for all real z |
| P2 centre | σ(0) = 0.5 |
| P3 direction | σ is **strictly increasing** in z: a higher score means a higher probability |
| P4 known value | σ(ln 3) = 0.75 exactly, σ(−ln 3) = 0.25 |
| P5 symmetry | σ(−z) = 1 − σ(z) |
| P6 consistency | for a fitted scikit-learn `LogisticRegression`, σ(b0 + x′b) equals `predict_proba(X)[:, 1]` |

Sources for the formula: the project's model docstring (`src/baseline_logreg.py`) and the scikit-learn documentation for `LogisticRegression`. Do not invent citations. If you cite a textbook, give only one you are certain of, or none.

**Important (why this pair is subtle):**
- The faulty formula **1/(1 + exp(+z)) = σ(−z) = 1 − σ(z)** still satisfies P1, P2 and P5. Tests that check only range, σ(0) = 0.5 or symmetry **will not detect it**. Your tests must include P3, P4 and P6.
- The textbook "alternative" **exp(z)/(1 + exp(z))** is algebraically equal to σ(z), but it returns **NaN for large z** (e.g. z = 1000, because inf/inf). It is therefore **not** used as the valid twin, since it would contain a second, numerical defect.

---

# PHASE 2 — DESIGN

Create:

```text
benchmark/
├── pair2_formula/
│   ├── README.md
│   ├── __init__.py
│   ├── faulty/   __init__.py  spec.md  impl.py  label.json  test_case.py
│   └── valid/    __init__.py  spec.md  impl.py  label.json  test_case.py
└── run_pair2.py
```

Synthetic data only. No Freddie Mac rows anywhere.

---

# PHASE 3 — IMPLEMENT THE TWO VERSIONS

## A. Common interface (identical in both files)

```python
predict_proba(X, coef, intercept) -> numpy.ndarray
```

- `X`: 2-D array-like of shape (n, k) of feature values (already preprocessed).
- `coef`: 1-D array-like of length k.
- `intercept`: a float.
- Returns a 1-D float array of length n with P(event = 1) for each row.
- Must not modify its inputs.

**Input validation must be identical in both files.** Raise the same `ValueError` for:
- `X` not 2-D;
- `len(coef) != X.shape[1]`;
- any non-finite value (NaN or inf) in `X`, `coef` or `intercept`;
- non-numeric input.

An empty `X` with shape (0, k) returns an empty array.

Compute the score in both files with exactly the same line: `z = X @ coef + intercept`.

## B. Valid implementation: `valid/impl.py`

Use the algebraically equivalent, numerically stable form:

```python
p = 0.5 * (1.0 + np.tanh(0.5 * z))
```

Identity: σ(z) = 1/(1+e^(−z)) = (1 + tanh(z/2)) / 2.
This form deliberately looks unfamiliar. It is the **false-alarm trap**: a validator that expects the textbook formula and flags this one is producing a false positive. Give it an ordinary, neutral code comment (for example, "logistic function via tanh identity; stable for large |z|"). Do not mention the benchmark or the answer.

## C. Faulty implementation: `faulty/impl.py`

Identical to the valid file in every line except the probability line:

```python
p = 1.0 / (1.0 + np.exp(z))
```

This is the single intended defect: a sign error in the exponent. Every probability becomes 1 − σ(z), so risky loans receive low probabilities and safe loans high ones.

Constraints:
- Same function name, signature, validation, return type, and the same comment style as valid.
- No comments or names that reveal the defect (no "bug", "wrong", "faulty", "sign error").
- It must import and run normally on the synthetic data, and produce wrong numbers, not an exception.
- Do not introduce any other difference. Show me `diff faulty/impl.py valid/impl.py` at the end. It must show only the probability line, plus at most its comment line.

---

# PHASE 4 — SPECIFICATIONS (`faulty/spec.md` = `valid/spec.md`, byte-identical)

Both describe the **same correct behaviour**:

1. **Purpose:** turn the logistic-regression score into the 12-month 90+ DPD probability used by the baseline model.
2. **Required formula:** p = 1/(1 + exp(−z)) with z = b0 + x′b. Any algebraically equivalent implementation is acceptable **provided it satisfies P1–P6 and returns finite values for |z| ≤ 1000**.
3. **Properties P1–P6** (the table above), each with a one-line justification.
4. **Why a sign error is wrong:** show that 1/(1+e^z) = 1 − σ(z), that it is decreasing in z, and that it reverses the ranking of loans. On any data, ROC-AUC becomes 1 − (correct AUC). Give the small numerical illustration: σ(ln 3) = 0.75 versus the faulty 0.25.
5. **What is NOT a defect:** different but equivalent algebraic forms (e.g. the tanh identity, `scipy.special.expit`); tiny floating-point differences (≤ 1e-12).
6. **Applicability conditions:** binary logistic regression; p is the probability of the event coded 1; coefficients come from a model fitted with the standard log-odds link.
7. **Provenance:** the project model docstring (`src/baseline_logreg.py`) and the scikit-learn `LogisticRegression` documentation.

---

# PHASE 5 — LABELS

`faulty/label.json`:
```json
{
  "case_id": "P2_F",
  "pair_id": "P2",
  "category": "formula",
  "is_defect": true,
  "violated_rule": "The logistic probability must be 1/(1+exp(-z)); the implementation uses 1/(1+exp(+z)), which equals 1 - sigma(z) and reverses the direction of risk.",
  "evidence": "Synthetic tests: sigma(ln 3) returns 0.25 instead of 0.75; probabilities decrease as z increases; disagrees with scikit-learn predict_proba.",
  "source": "Project model specification (src/baseline_logreg.py docstring); scikit-learn LogisticRegression documentation"
}
```

`valid/label.json`: `"case_id": "P2_V"`, `"is_defect": false`, `"violated_rule": null`, and evidence that all properties hold, including agreement with scikit-learn to within 1e-12 and finite output at |z| = 1000. Use the same `pair_id`, `category` and `source`.

---

# PHASE 6 — SYNTHETIC TEST DATA AND EXACT EXPECTED VALUES

These values were computed in advance. **Do not change them to match an implementation.**

### Test set A (score from features)
```python
X = [[ 0.0, 0.00],
     [ 1.0, 0.00],
     [ 0.0, 1.00],
     [-0.5, 0.00],
     [-1.0, 0.25]]
coef = [1.0, -2.0]
intercept = 0.5
```
z = `[0.5, 1.5, -1.5, 0.0, -1.0]`

| | expected p (correct) |
|---|---|
| row 1 (z = 0.5) | 0.622459331202 |
| row 2 (z = 1.5) | 0.817574476194 |
| row 3 (z = −1.5) | 0.182425523806 |
| row 4 (z = 0.0) | 0.5 |
| row 5 (z = −1.0) | 0.268941421370 |

For reference only, **not** to be asserted, the faulty version gives `[0.377540668798, 0.182425523806, 0.817574476194, 0.5, 0.731058578630]`. Note that row 4 is 0.5 for **both**, which is why a z = 0 test alone proves nothing.

### Test set B (exact known values)
`X = [[0.0]]`, `coef = [0.0]`:
- `intercept = ln(3)` → p = **0.75**
- `intercept = −ln(3)` → p = **0.25**

### Test set C (numerical stability)
`X = [[1.0], [-1.0]]`, `coef = [1000.0]`, `intercept = 0.0` (z = ±1000) → output must be **finite**, approximately `[1.0, 0.0]`, with no NaN. Suppress or ignore overflow `RuntimeWarning`s in the test only if needed. Do not hide failures.

### Test set D (scikit-learn reference)
Generate data with `numpy.random.default_rng(0)`: `X` = 200×2 standard normal; `y = (X @ [1.5, -1.0] + noise > 0)`, where noise is another 200 standard-normal draws from the same generator. Fit `sklearn.linear_model.LogisticRegression()` with default settings. Then `predict_proba(X, model.coef_[0], model.intercept_[0])` must match `model.predict_proba(X)[:, 1]` within `atol=1e-12`.
(Pre-computed check: the tanh form matches to about 1.1e-16. ROC-AUC is about 0.916 with the correct formula and about 0.084 with the faulty one. You may report these as information only; do not assert the exact AUC.)

Use `np.testing.assert_allclose(..., rtol=0, atol=1e-12)` for numerical comparisons.

---

# PHASE 7 — PYTEST TESTS

The two `test_case.py` files must be **identical except for the single import line** (as in Pair 1). Show the `diff` at the end.

Required tests (each docstring cites the property it checks):
1. Test set A: exact probabilities (P6-style exact values).
2. Test set B: σ(ln 3) = 0.75 and σ(−ln 3) = 0.25 (P4).
3. Monotonicity (P3): over `z = np.linspace(-20, 20, 401)`, probabilities strictly increase.
4. Range (P1): all outputs in [0, 1].
5. Centre (P2): σ(0) = 0.5.
6. Symmetry (P5): p(z) + p(−z) = 1.
7. Stability: test set C gives finite values ≈ [1, 0].
8. scikit-learn agreement (P6): test set D.
9. Inputs are not modified.
10. Output shape is (n,); an empty X returns an empty array.
11. Validation: identical `ValueError`s for wrong dimensions, length mismatch, NaN/inf and non-numeric input.

**Expected outcome:**
- Valid passes all tests.
- Faulty **fails** tests 1, 2, 3, 7 and 8 and **passes** 4, 5, 6, 9, 10 and 11. If the faulty version fails or passes anything else, the pair is wrong; investigate and fix the implementation, not the expected values.
- (On test 7: the faulty version returns [0.0, 1.0] at z = ±1000. That is finite but reversed, so it fails because of the sign, not because of overflow.)

Mark the assertion that shows the sign error most directly (test 2, σ(ln 3)).

---

# PHASE 8 — RUNNER: `benchmark/run_pair2.py`

Copy the design of `run_pair1.py`:
- run each suite in its own pytest process;
- load the labels;
- classify each failure as an assertion failure, a collection/import error or a runtime error. **Only assertion failures count as evidence of a defect**;
- print `case_id | expected label | test result | agrees with label`;
- run the data-hygiene check;
- exit 0 only if both cases agree with their labels and there were no errors.

Results must be generated from real runs, not hard-coded.

---

# PHASE 9 — DEPENDENCIES

Use the existing uv environment. numpy, pandas, scikit-learn and pytest are already installed. Add nothing unless it is genuinely missing (`uv add --dev ...`). Do not upgrade packages.

---

# PHASE 10 — RUN

From the repository root:
```bash
uv run pytest benchmark/pair2_formula/valid -v
uv run pytest benchmark/pair2_formula/faulty -v      # expected: assertion failures
uv run pytest benchmark/test_no_real_data.py -v
uv run python benchmark/run_pair2.py
uv run python benchmark/run_pair1.py                 # Pair 1 must still pass (regression check)
```
Do not change tests to make commands look successful. Do not hide failures.

---

# PHASE 11 — INDEPENDENT VALIDATION

Before finishing:
1. Recompute test sets A and B with plain Python `math`, independently of both implementations, and confirm the expected values.
2. Confirm the faulty version fails exactly tests 1, 2, 3, 7 and 8.
3. Confirm `diff` of the impl files shows only the probability line (plus at most its comment).
4. Confirm `diff` of the test files shows only the import line.
5. Confirm the two spec files are byte-identical.
6. Confirm no real Freddie Mac data is in `benchmark/`.
7. Confirm `src/baseline_logreg.py` and all Pair 1 files are unchanged (`git diff --stat`).
8. Confirm Pair 1's runner still exits 0.

---

# PHASE 12 — REPORT BACK

## A. Files created
Show the tree, and explain each file in simple English.

## B. Mathematical explanation
Using σ(ln 3), explain why the correct answer is 0.75 and the faulty one 0.25. Explain why the faulty formula still passes the range, centre and symmetry tests, and why the tanh form is correct even though it looks different.

## C. Actual results
Paste the real output of the pytest runs and of both runners.

## D. Research interpretation
- Why P2_F is objectively defective.
- Why P2_V is valid, and why it is a false-positive trap.
- What this pair proves, and what it does NOT prove (no claims about VALID-Agent performance or novelty).

## E. Repository checks
List files created or changed; confirm the protected files are unchanged; note any limitations or assumptions.

## F. Suggested commit message
Do not commit unless I ask.

---

# ACCEPTANCE CRITERIA
- `CLAUDE.md` and the Pair 1 files were read first; Pair 1 conventions are followed.
- Same interface and identical validation in both versions; one-line difference in the probability formula.
- The valid version uses the stable tanh identity and passes all tests, including z = ±1000 and scikit-learn agreement.
- The faulty version fails exactly tests 1, 2, 3, 7 and 8, all as assertion failures.
- Identical specs; test files identical except for the import line; labels valid JSON.
- Synthetic data only; the hygiene check passes; protected files unchanged; Pair 1 runner still exits 0.
- The final report includes real outputs and a suggested commit message.

**Do not begin Pair 3, Pair 4, VALID-Agent, RAG or XGBoost. This task is limited to Pair 2.**

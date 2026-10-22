# Specification — MRM-Bench Pair 2: logistic-regression probability formula

This specification is the same for both implementations in Pair 2. It states the required behaviour
of `predict_proba(X, coef, intercept)`. The ground-truth labels and the tests in `test_case.py` are
derived from it. The tests cite the properties **P1–P6**, the numerical requirement **N1** and the interface
rules **I1–I4** defined below. (Here P1–P6 name properties of the logistic function, not benchmark pairs.)

**Numerical tolerance.** Every equality below is checked to an absolute tolerance of 1e-12.

---

## 1. Purpose

The project's baseline model estimates, for each loan at a prediction date t, the probability of
"12-month 90+ DPD serious delinquency". The event is any month t+1..t+12 with delinquency status ≥ 03 or status 'RA'.
This is not a regulatory default definition. `predict_proba` turns fitted logistic-regression
coefficients and preprocessed features into that probability, one value per row.

---

## 2. Required formula

For row i with feature vector x_i, slope coefficients b and intercept b0:

    linear score   z_i = b0 + x_i′b
    probability    p_i = σ(z_i) = 1 / (1 + exp(−z_i))   =  P(y_i = 1 | x_i)

Here y_i = 1 is the event and y_i = 0 the non-event. Any algebraically equivalent implementation is
acceptable provided it satisfies P1–P6 (§3) and N1:

**N1 — Numerical stability.** For every score with |z| ≤ 1000 the returned value is finite (no NaN, no
±inf) and within 1e-12 of σ(z). For example σ(1000) ≈ 1 and σ(−1000) ≈ 0. Overflow *warnings* during the
computation are not a violation; non-finite or inaccurate *results* are.

**Interface rules.**

- **I1 — Inputs.** `X` is 2-D with shape (n, k), `coef` is 1-D with length k, and `intercept` is a single number.
  All are numeric (bool, integer or float) and finite. Otherwise the function raises `ValueError`. That
  covers: `X` not 2-D; `coef` not 1-D; `len(coef) != X.shape[1]`; `intercept` not a single number; any
  non-numeric input (text, `None`, objects); any NaN or ±inf.
- **I2 — Output.** A 1-D float array of shape (n,). For n = 0 (shape (0, k)) it is an empty array.
- **I3 — No side effects.** The inputs are not modified.
- **I4 — Score.** z = `X @ coef + intercept`, i.e. b0 + x′b for every row.

---

## 3. Properties of σ that any correct implementation satisfies

| Property | Statement | One-line justification |
|---|---|---|
| **P1** range | 0 ≤ σ(z) ≤ 1 for all real z | exp(−z) > 0, so the denominator exceeds 1. In floating point σ can round to exactly 0.0 or 1.0, hence "≤". |
| **P2** centre | σ(0) = 0.5 | 1 / (1 + e⁰) = 1/2. |
| **P3** direction | σ is strictly increasing: a higher score means a higher probability | σ′(z) = σ(z)(1 − σ(z)) > 0. In float64, σ can saturate for very large \|z\|, so strict increase is tested on z ∈ [−20, 20]. |
| **P4** known value | σ(ln 3) = 0.75 and σ(−ln 3) = 0.25 | e^(−ln 3) = 1/3, so σ(ln 3) = 1/(1 + 1/3) = 3/4. Likewise σ(−ln 3) = 1/(1 + 3) = 1/4. |
| **P5** symmetry | σ(−z) = 1 − σ(z) | σ(−z) = 1/(1 + eᶻ) = e⁻ᶻ/(e⁻ᶻ + 1) = 1 − σ(z). |
| **P6** consistency | For a fitted scikit-learn `LogisticRegression`, σ(b0 + x′b) with b = `coef_[0]`, b0 = `intercept_[0]` equals `predict_proba(X)[:, 1]` | For binary problems scikit-learn computes exactly 1/(1 + exp(−(Xw + w0))) (see §7), so the same function on the same score must give the same number. |

---

## 4. Why a sign error is wrong

Let q(z) = 1 / (1 + exp(+z)) be the formula with the sign of the exponent flipped.

1. **It is the complement of σ.** By P5, q(z) = σ(−z) = 1 − σ(z). It is the probability of the class coded
   **0** (no 90+ DPD), not of the event.
2. **It is decreasing.** q′(z) = −σ(z)(1 − σ(z)) < 0, so it violates P3.
3. **It reverses the ranking of loans.** If loan A has a higher score than loan B (z_A > z_B), then
   σ(z_A) > σ(z_B) but q(z_A) < q(z_B). The riskiest loans receive the lowest probabilities.
4. **ROC-AUC becomes 1 − (correct AUC) on any data.** AUC = P(s⁺ > s⁻) + ½ P(s⁺ = s⁻) over
   (event, non-event) pairs. Replacing σ by 1 − σ turns every strict ordering around and keeps ties as ties,
   so AUC_q = 1 − AUC_σ in exact arithmetic. (In floating point, saturation at very large |z| can create or
   remove ties, so the identity can then hold only approximately.)
5. **It breaks calibration.** With an unpenalised intercept, a fitted logistic regression's mean predicted
   probability on its training data equals the training event rate, because ∂loss/∂b0 = Σ(p_i − y_i) = 0
   at the optimum. Under the sign error the mean becomes 1 − (event rate). For example, with the project's
   training event rate of 462 / 101,843 ≈ 0.45%, the average reported probability would be about 99.5%.
   (This figure is derived from the identity, not from a model run.)
6. **Small numerical illustration (P4).** A loan whose log-odds of the event are ln 3 has event odds of 3 : 1:

       correct   σ(ln 3) = 1 / (1 + 1/3) = 0.75
       sign error q(ln 3) = 1 / (1 + 3)   = 0.25

**Why range, centre and symmetry tests cannot detect it.** q(z) lies in [0, 1] (P1 holds), q(0) = 0.5 (P2
holds), and q(−z) = σ(z) = 1 − q(z) (P5 holds). A test suite that checks only these properties passes the
defective formula. Only direction (P3), known values (P4), exact values from the score, and agreement
with a reference (P6) can detect it.

Exact values on the synthetic test set A (`coef` = [1, −2], `intercept` = 0.5):

| row | z | σ(z), required | q(z), sign error |
|---|---|---|---|
| 1 | 0.5 | 0.622459331202 | 0.377540668798 |
| 2 | 1.5 | 0.817574476194 | 0.182425523806 |
| 3 | −1.5 | 0.182425523806 | 0.817574476194 |
| 4 | 0.0 | 0.5 | 0.5 |
| 5 | −1.0 | 0.268941421370 | 0.731058578630 |

Row 4 is 0.5 in both columns, so a test at z = 0 alone proves nothing.

On the synthetic test set D (defined in `test_case.py`: 200 rows, scikit-learn default fit), the observed ROC-AUC
is 0.9163 with σ and 0.0837 with q. Their sum is 1, as point 4 predicts. These values are reported for
information only and are not asserted by the tests.

---

## 5. What is NOT a defect

- **Different but equivalent algebraic forms.** These are acceptable whenever they satisfy P1–P6 and N1:
  - The tanh identity `0.5 * (1 + tanh(z / 2))`. Proof: tanh(z/2) = (1 − e⁻ᶻ)/(1 + e⁻ᶻ), so
    1 + tanh(z/2) = 2/(1 + e⁻ᶻ), and half of that is σ(z).
  - `scipy.special.expit`, which scikit-learn itself uses.
  - Piecewise "stable sigmoid" code.
  - The textbook `1 / (1 + exp(-z))`. It emits an overflow warning at z = −1000 but returns the finite value 0.0.
- **Tiny floating-point differences** of at most 1e-12.
- **Saturation** to exactly 0.0 or 1.0 at extreme scores.
- **Model quality.** Coefficient sizes, accuracy, AUC level and distribution shift are not judged here.

Note: `exp(z) / (1 + exp(z))` is algebraically equal to σ(z), but in float64 it returns NaN at z = 1000
(inf / inf). It violates N1, which is a *numerical* defect and a different defect from the sign error.

---

## 6. Applicability conditions

This specification, and therefore the labels of this pair, applies when all of the following hold:

- **A1.** The model is a **binary** logistic regression. Multinomial (softmax) models with more than two
  classes are not covered.
- **A2.** p is the probability of the event **coded 1** (90+ DPD within 12 months), which is the quantity the
  baseline reports. If an interface promised the probability of the class coded 0, then 1/(1 + exp(z))
  would be the correct formula for it. The sign error is a defect *here* because this interface promises
  P(event = 1).
- **A3.** The coefficients come from a model fitted with the standard **log-odds (logit) link**,
  log(p / (1 − p)) = b0 + x′b, with b and b0 oriented toward class 1. In scikit-learn these are `coef_[0]` and
  `intercept_[0]` with `classes_ = [0, 1]`. Probit or complementary log-log links use different formulas.
- **A4.** `X` has been preprocessed exactly as during fitting (same columns, order and scaling).
  `predict_proba` cannot check this.

**False-alarm traps** (valid code that can look suspicious):

- A probability formula that does not look like the textbook expression, such as the tanh identity.
- Outputs of exactly 0.0 or 1.0 for extreme scores.
- Overflow warnings from a correct formula, for example the textbook form at z = −1000.

---

## 7. Evidence and provenance

- **Project model specification:** `src/baseline_logreg.py`, module docstring, section "Model (written out so
  the benchmark can test against it)" (lines 11–12 at commit 9c743e2): `z_i = b0 + x_i' b` and
  `p_i = sigmoid(z_i) = 1 / (1 + exp(-z_i))`.
- **scikit-learn documentation:** User Guide, section 1.1.11 "Logistic regression" → 1.1.11.1 "Binary Case"
  (scikit-learn 1.9.1 documentation, https://scikit-learn.org/stable/modules/linear_model.html, retrieved
  2026-10-22). Once fitted, `predict_proba` predicts P(y_i = 1 | X_i) as
  p̂(X_i) = expit(X_i w + w_0) = 1 / (1 + exp(−X_i w − w_0)).
- **scikit-learn source (installed version 1.9.1):** for binary problems `LogisticRegression.predict_proba`
  (`sklearn/linear_model/_logistic.py`) returns `_predict_proba_lr(X)`. The docstring of that method
  (`sklearn/linear_model/_base.py`) reads: "Positive class probabilities are computed as
  1. / (1. + np.exp(-self.decision_function(X)))".
- **Mathematics:** P1–P6 and §4–§5 are derived above with elementary algebra and calculus. No textbook is
  cited.
- **Executable evidence:** `test_case.py` in this folder. It uses small synthetic data only, and its exact
  expected values were computed in advance and rechecked with Python's `math` module.

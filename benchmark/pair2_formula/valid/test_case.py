"""
MRM-Bench Pair 2 - executable tests of spec.md (logistic-regression probability formula)
========================================================================================
Both implementations in this pair face these SAME tests: the two test_case.py files are identical
except for the import line. P1-P6, N1 and I1-I4 refer to spec.md; [T1]-[T11] number the tests.

All data are small SYNTHETIC values - not Freddie Mac records. Expected values were computed in advance
from the formula and rechecked with Python's math module; they were not copied from an implementation.

The most direct evidence of a sign error is test_known_values_ln3: sigma(ln 3) must be 0.75.
"""
import math

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from benchmark.pair2_formula.valid.impl import predict_proba  # the only line that differs between the pair's test files

ATOL = 1e-12                                     # spec.md numerical tolerance

# Test set A: z = X @ coef + intercept = [0.5, 1.5, -1.5, 0.0, -1.0]
X_A = [[0.0, 0.00],
       [1.0, 0.00],
       [0.0, 1.00],
       [-0.5, 0.00],
       [-1.0, 0.25]]
COEF_A, INTERCEPT_A = [1.0, -2.0], 0.5
P_A = [0.622459331202, 0.817574476194, 0.182425523806, 0.5, 0.268941421370]  # sigma(z), 12 decimals

Z_GRID = np.linspace(-20, 20, 401)               # scores for the P1, P3 and P5 tests


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)


def _p_of_z(z):
    """Probabilities for given scores: one feature equal to z, coefficient 1, intercept 0 (score = z)."""
    z = np.asarray(z, dtype=np.float64)
    return predict_proba(z.reshape(-1, 1), [1.0], 0.0)


# ---------------------------------------------------------------- formula values and direction
def test_set_a_exact_probabilities():
    """[T1] (P6-style exact values, I4) test set A gives sigma(z) for z = 0.5, 1.5, -1.5, 0.0, -1.0."""
    _close(predict_proba(X_A, COEF_A, INTERCEPT_A), P_A)


def test_known_values_ln3():
    """[T2] KEY EVIDENCE (P4): sigma(ln 3) = 0.75 and sigma(-ln 3) = 0.25 exactly."""
    _close(predict_proba([[0.0]], [0.0], math.log(3.0)), [0.75])
    _close(predict_proba([[0.0]], [0.0], -math.log(3.0)), [0.25])


def test_strictly_increasing():
    """[T3] (P3) a higher score gives a higher probability on z = -20, -19.9, ..., 20."""
    steps = np.diff(_p_of_z(Z_GRID))
    not_increasing = int(np.sum(steps <= 0))
    assert not_increasing == 0, f"probability does not increase with the score at {not_increasing} of {steps.size} steps"


# ---------------------------------------------------------------- properties a sign error does NOT break
def test_range():
    """[T4] (P1) every probability lies in [0, 1], including at z = -1000 and z = 1000."""
    p = _p_of_z(np.concatenate([Z_GRID, [-1000.0, 1000.0]]))
    assert np.all((p >= 0.0) & (p <= 1.0)), f"probabilities outside [0, 1]: {p[~((p >= 0.0) & (p <= 1.0))]}"


def test_centre():
    """[T5] (P2) sigma(0) = 0.5."""
    _close(_p_of_z([0.0]), [0.5])


def test_symmetry():
    """[T6] (P5) p(z) + p(-z) = 1 on z = -20, ..., 20."""
    _close(_p_of_z(Z_GRID) + _p_of_z(-Z_GRID), np.ones_like(Z_GRID))


# ---------------------------------------------------------------- numerical stability and reference agreement
def test_stability_large_scores():
    """[T7] (N1) z = +1000 and -1000 give finite probabilities, approximately 1 and 0."""
    p = predict_proba([[1.0], [-1.0]], [1000.0], 0.0)
    assert np.all(np.isfinite(p)), f"non-finite probabilities at |z| = 1000: {p}"
    _close(p, [1.0, 0.0])


def test_agrees_with_sklearn():
    """[T8] (P6) equals scikit-learn LogisticRegression.predict_proba(X)[:, 1] on synthetic test set D."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 2))
    noise = rng.standard_normal(200)
    y = (X @ np.array([1.5, -1.0]) + noise > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    assert list(model.classes_) == [0, 1]        # coef_[0], intercept_[0] are oriented toward class 1 (A3)
    _close(predict_proba(X, model.coef_[0], model.intercept_[0]), model.predict_proba(X)[:, 1])


# ---------------------------------------------------------------- interface (I1-I3)
def test_inputs_not_modified():
    """[T9] (I3) array and list inputs are unchanged after the call."""
    X, coef = np.array(X_A), np.array(COEF_A)
    X_before, coef_before = X.copy(), coef.copy()
    X_list, coef_list = [row[:] for row in X_A], list(COEF_A)
    predict_proba(X, coef, INTERCEPT_A)
    predict_proba(X_list, coef_list, INTERCEPT_A)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(coef, coef_before)
    assert X_list == X_A and coef_list == COEF_A


def test_output_shape_and_empty_input():
    """[T10] (I2) output is a 1-D float array of shape (n,); X of shape (0, k) gives an empty array."""
    p = predict_proba(X_A, COEF_A, INTERCEPT_A)
    assert isinstance(p, np.ndarray) and p.shape == (5,) and p.dtype == np.float64
    empty = predict_proba(np.empty((0, 2)), COEF_A, INTERCEPT_A)
    assert isinstance(empty, np.ndarray) and empty.shape == (0,)


INVALID_INPUTS = {
    "X_1d": ([0.0, 1.0], COEF_A, INTERCEPT_A),
    "X_3d": (np.zeros((2, 2, 1)), COEF_A, INTERCEPT_A),
    "coef_2d": (X_A, [COEF_A], INTERCEPT_A),
    "coef_length_mismatch": (X_A, [1.0, -2.0, 3.0], INTERCEPT_A),
    "X_nan": ([[np.nan, 0.0]], COEF_A, INTERCEPT_A),
    "X_inf": ([[np.inf, 0.0]], COEF_A, INTERCEPT_A),
    "coef_nan": (X_A, [np.nan, -2.0], INTERCEPT_A),
    "coef_inf": (X_A, [1.0, -np.inf], INTERCEPT_A),
    "intercept_nan": (X_A, COEF_A, np.nan),
    "intercept_inf": (X_A, COEF_A, np.inf),
    "intercept_not_single": (X_A, COEF_A, [0.5, 0.5]),
    "X_text": ([["a", "b"]], COEF_A, INTERCEPT_A),
    "X_none": ([[None, 0.0]], COEF_A, INTERCEPT_A),
    "coef_text": (X_A, ["1.0", "-2.0"], INTERCEPT_A),
    "intercept_text": (X_A, COEF_A, "0.5"),
}


@pytest.mark.parametrize("X, coef, intercept", list(INVALID_INPUTS.values()), ids=list(INVALID_INPUTS))
def test_invalid_inputs_raise(X, coef, intercept):
    """[T11] (I1) ValueError for wrong dimensions, length mismatch, NaN/inf and non-numeric input."""
    with pytest.raises(ValueError):
        predict_proba(X, coef, intercept)

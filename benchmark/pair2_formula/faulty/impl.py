"""
MRM-Bench Pair 2 (formula) - logistic-regression probability
=============================================================
Specification: spec.md in this folder.

Interface
---------
  predict_proba(X, coef, intercept) -> 1-D numpy float array of length n

  X          : 2-D array-like of shape (n, k): numeric feature values, already preprocessed.
  coef       : 1-D array-like of length k: fitted slope coefficients b.
  intercept  : a single real number: fitted intercept b0.

  Returns P(event = 1) for each row, computed from the linear score z = b0 + x'b.
  The inputs are not modified. An X of shape (0, k) returns an empty array.

  Raises ValueError if any input is non-numeric or contains NaN/inf, X is not 2-D, coef is not 1-D,
  len(coef) != X.shape[1], or intercept is not a single number.
"""
import numpy as np


def _as_real_array(name, value):
    """Numeric (bool/int/float) input as a NEW float64 array of finite values; anything else -> ValueError."""
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        raise ValueError(f"predict_proba: {name} is not a numeric array") from None
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"predict_proba: {name} must be numeric, got dtype {arr.dtype}")
    arr = arr.astype(np.float64)                  # always a copy, so the caller's data cannot change
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"predict_proba: {name} contains NaN or infinite values")
    return arr


def predict_proba(X, coef, intercept):
    X = _as_real_array("X", X)
    coef = _as_real_array("coef", coef)
    intercept = _as_real_array("intercept", intercept)
    if X.ndim != 2:
        raise ValueError(f"predict_proba: X must be 2-D with shape (n, k), got shape {X.shape}")
    if coef.ndim != 1:
        raise ValueError(f"predict_proba: coef must be 1-D, got shape {coef.shape}")
    if len(coef) != X.shape[1]:
        raise ValueError(f"predict_proba: len(coef) = {len(coef)} does not match X.shape[1] = {X.shape[1]}")
    if intercept.ndim != 0:
        raise ValueError(f"predict_proba: intercept must be a single number, got shape {intercept.shape}")

    z = X @ coef + intercept                      # linear score b0 + x'b, one value per row

    # logistic function
    p = 1.0 / (1.0 + np.exp(z))
    return p

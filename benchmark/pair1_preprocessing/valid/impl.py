"""
MRM-Bench Pair 1 (preprocessing) - Original DTI Ratio preprocessing
====================================================================
Specification: spec.md in this folder.

Interface
---------
  prepare_dti(df, training_median=None) -> new DataFrame

  df               : DataFrame with an `orig_dti` column (Freddie Mac "Original Debt-to-Income (DTI) Ratio",
                     a percentage). Values may be numbers or numeric strings (as read from the raw file).
  training_median  : None  -> training mode: the median is fitted on this DataFrame.
                     float -> validation/inference mode: this median (fitted on training data) is used;
                              no median is computed from `df`.

  Returns a copy of `df` (same index, row order and other columns) with
    orig_dti                         : float, missing entries imputed with the median
    dti_missing                      : int 0/1 missing-value indicator
    attrs["dti_training_median"]     : the median that was used (float)

  Raises ValueError if the column is absent, a value is not a legal input, the training data has no
  observed DTI value to fit a median, or a supplied training_median is not a ratio in [0, 65].

Limitation: DataFrame.attrs is an experimental pandas feature and some operations (e.g. concat, merge)
may drop it, so read attrs["dti_training_median"] directly from the returned DataFrame.
"""
import pandas as pd

DTI_MIN, DTI_MAX = 0.0, 65.0      # documented range of reported DTI ratios (percent)
DTI_NOT_AVAILABLE = 999           # documented "Not Available" code (General User Guide, July 2026)


def _checked_median(value):
    """A supplied training median must itself be a DTI ratio in [0, 65]."""
    try:
        median = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"prepare_dti: training_median must be a number, got {value!r}") from None
    if not DTI_MIN <= median <= DTI_MAX:          # also rejects NaN and inf
        raise ValueError(f"prepare_dti: training_median must be in [0, 65], got {value!r}")
    return median


def prepare_dti(df, training_median=None):
    if "orig_dti" not in df.columns:
        raise ValueError("prepare_dti: input DataFrame has no 'orig_dti' column")
    out = df.copy()                               # never modify the caller's DataFrame

    # numbers or numeric strings -> float; a non-null value that is not a number is rejected
    raw = out["orig_dti"]
    dti = pd.to_numeric(raw, errors="coerce").astype("float64")
    unparseable = raw.notna() & dti.isna()
    if unparseable.any():
        raise ValueError(f"prepare_dti: non-numeric orig_dti values: {raw[unparseable].unique()[:5].tolist()}")

    # legal inputs: a ratio in [0, 65], the documented code 999, or a genuine null
    legal = dti.isna() | dti.between(DTI_MIN, DTI_MAX) | (dti == DTI_NOT_AVAILABLE)
    if not legal.all():
        raise ValueError(f"prepare_dti: orig_dti values outside [0, 65] and not 999: "
                         f"{raw[~legal].unique()[:5].tolist()}")

    # 999 is the documented "Not Available" code (all HARP loans carry it), not a ratio -> missing
    dti = dti.mask(dti == DTI_NOT_AVAILABLE)

    missing = dti.isna()
    if training_median is None:                   # training: fit the median on observed values only
        if missing.all():
            raise ValueError("prepare_dti: cannot fit the training median - no observed orig_dti values "
                             "(empty or entirely missing training data)")
        median = float(dti.median())
    else:                                         # validation / inference: reuse the training median
        median = _checked_median(training_median)

    out["orig_dti"] = dti.fillna(median)
    out["dti_missing"] = missing.astype("int64")
    out.attrs["dti_training_median"] = median
    return out

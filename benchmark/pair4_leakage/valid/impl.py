"""
Delinquency-history features at a prediction month
==================================================
Specification: spec.md in this folder.

Interface
---------
  build_features(panel, prediction_month) -> DataFrame

  panel             : one row per loan-month with columns
                        loan_id            str
                        month              int (consecutive month index)
                        delinq_status      str: two-digit code "00".."99", or "RA" (REO acquisition)
                        zero_balance_code  str or missing; present only on a loan's last record
  prediction_month  : the month t at which features are built

  Returns one row per loan that has a record at t and no zero-balance code at or before t, indexed by
  loan_id (sorted), with int columns
    status_t           severity at month t
    max_dq_12m         maximum severity over the 12-month lookback
    n_dq_months_12m    number of lookback months with severity >= 1
    ever_90_before_t   1 if any month up to t has severity >= 3, else 0
  where severity is the numeric status code and "RA" counts as 99. The input panel is not modified.

  Raises ValueError for missing columns, duplicate (loan_id, month) rows, a prediction_month with no
  records, or an unrecognized status code.
"""
import pandas as pd

REQUIRED_COLUMNS = ["loan_id", "month", "delinq_status", "zero_balance_code"]
FEATURES = ["status_t", "max_dq_12m", "n_dq_months_12m", "ever_90_before_t"]
REO_SEVERITY = 99                                   # "RA" ranks with the most severe numeric codes


def _severity(code):
    """Two-digit numeric status -> int; "RA" -> 99; anything else -> ValueError."""
    if code == "RA":
        return REO_SEVERITY
    if isinstance(code, str) and len(code) == 2 and code.isascii() and code.isdigit():
        return int(code)
    raise ValueError(f"build_features: unrecognized delinq_status {code!r}")


def build_features(panel, prediction_month):
    missing = [c for c in REQUIRED_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"build_features: panel is missing columns {missing}")
    if panel.duplicated(["loan_id", "month"]).any():
        raise ValueError("build_features: panel has duplicate (loan_id, month) rows")
    t = prediction_month
    if not (panel["month"] == t).any():
        raise ValueError(f"build_features: prediction_month {t} has no records in the panel")
    # work on a new frame with a numeric severity column; the caller's panel is never modified
    panel = panel[REQUIRED_COLUMNS].assign(severity=[_severity(c) for c in panel["delinq_status"]])

    # loans with a record at t and no zero-balance code at or before t
    now = panel[panel["month"] == t].set_index("loan_id")
    closed = panel.loc[(panel["month"] <= t) & panel["zero_balance_code"].notna(), "loan_id"]
    ids = pd.Index(sorted(now.index.difference(closed)), name="loan_id")
    panel = panel[panel["loan_id"].isin(ids)]

    # 12-month lookback window
    window = panel[(panel["month"] > t - 12) & (panel["month"] <= t)]
    history = panel[panel["month"] <= t]

    out = pd.DataFrame(index=ids)
    out["status_t"] = now["severity"].reindex(ids)
    out["max_dq_12m"] = window.groupby("loan_id")["severity"].max().reindex(ids)
    out["n_dq_months_12m"] = (window["severity"] >= 1).groupby(window["loan_id"]).sum().reindex(ids)
    out["ever_90_before_t"] = (history["severity"] >= 3).groupby(history["loan_id"]).any().reindex(ids)
    return out[FEATURES].astype("int64")

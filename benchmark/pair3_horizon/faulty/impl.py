"""
Outcome labelling for a monthly loan panel
==========================================
Specification: spec.md and spec.json in this folder.

Interface
---------
  label_outcomes(panel, prediction_month, data_cutoff=None) -> DataFrame

  panel             : one row per loan-month with columns
                        loan_id            str
                        month              int (consecutive months, e.g. months since a fixed origin)
                        delinq_status      str ("00", "01", "02", "03", ..., "RA")
                        zero_balance_code  str or missing; present only on a loan's last record
  prediction_month  : the month t at which loans are scored
  data_cutoff       : last month of available data; None means panel["month"].max().
                      Records after data_cutoff are ignored.

  Returns one row per loan at risk at t, indexed by loan_id (sorted), with a single column `label`:
    event, non_event, censored_prepay, censored_other_termination or censored_end_of_data.
  The input panel is not modified.

  Raises ValueError for missing columns, duplicate (loan_id, month) rows, or a prediction_month with
  no records in the panel.
"""
import pandas as pd

HORIZON_MONTHS = 6                                  # outcome window: months t+1 .. t+6 after the prediction month

REQUIRED_COLUMNS = ["loan_id", "month", "delinq_status", "zero_balance_code"]
AT_RISK_STATUSES = {"00", "01", "02"}               # current, 30-59 and 60-89 days delinquent
PREPAY_CODE = "01"                                  # zero-balance code: prepaid or matured


def _is_serious(status):
    """90+ days delinquent (numeric status >= 03) or REO acquisition ("RA")."""
    return (pd.to_numeric(status, errors="coerce") >= 3) | (status == "RA")


def label_outcomes(panel, prediction_month, data_cutoff=None):
    missing = [c for c in REQUIRED_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"label_outcomes: panel is missing columns {missing}")
    if panel.duplicated(["loan_id", "month"]).any():
        raise ValueError("label_outcomes: panel has duplicate (loan_id, month) rows")
    cutoff = panel["month"].max() if data_cutoff is None else data_cutoff
    df = panel.loc[panel["month"] <= cutoff, REQUIRED_COLUMNS]     # new frame; the input is never modified
    t = prediction_month
    if not (df["month"] == t).any():
        raise ValueError(f"label_outcomes: prediction_month {t} has no records in the panel")

    # at risk at t: a record at t with status 00/01/02 and no zero-balance code at or before t
    now = df[df["month"] == t]
    closed = df.loc[(df["month"] <= t) & df["zero_balance_code"].notna(), "loan_id"]
    at_risk = now.loc[now["delinq_status"].isin(AT_RISK_STATUSES) & ~now["loan_id"].isin(closed), "loan_id"]
    ids = pd.Index(sorted(at_risk), name="loan_id")
    loans = df[df["loan_id"].isin(ids)]

    # outcome window t+1 .. t+H
    end = t + HORIZON_MONTHS
    window = loans[(loans["month"] > t) & (loans["month"] <= end)]
    event = _is_serious(window["delinq_status"]).groupby(window["loan_id"]).any().reindex(ids, fill_value=False)
    observed_to_end = loans.groupby("loan_id")["month"].max().reindex(ids) >= end
    code = loans.dropna(subset=["zero_balance_code"]).groupby("loan_id")["zero_balance_code"].last().reindex(ids)

    # labels are assigned from lowest to highest precedence: event > non_event > censoring reasons
    label = pd.Series("censored_end_of_data", index=ids)
    label[code.notna()] = "censored_other_termination"
    label[code == PREPAY_CODE] = "censored_prepay"
    label[observed_to_end] = "non_event"
    label[event] = "event"
    return label.to_frame("label")

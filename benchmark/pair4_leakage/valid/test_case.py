"""
MRM-Bench Pair 4 - executable tests of spec.md (prediction-time features, no future information)
================================================================================================
Both implementations in this pair face these SAME tests: the two test_case.py files are identical
except for the import line. R1-R6 refer to spec.md; [T1]-[T11] number the tests.

All data are one small SYNTHETIC panel (loan IDs F01-F10, prediction month t = 12) - not Freddie Mac
records. Expected values were computed in advance from the definitions and rechecked with a separate
reference builder; they were not copied from an implementation.

T2 and T3 (future invariance) are the general leakage tests: they do not depend on any hand-picked loan.
"""
import pandas as pd
import pandas.testing as pdt
import pytest

from benchmark.pair4_leakage.valid.impl import build_features  # the only line that differs between the pair's test files

T = 12                                           # prediction month
FEATURES = ["status_t", "max_dq_12m", "n_dq_months_12m", "ever_90_before_t"]


def _codes(first, last):
    """Two-digit status codes, e.g. _codes(1, 3) -> ["01", "02", "03"]."""
    return [f"{d:02d}" for d in range(first, last + 1)]


# loan -> (first month, statuses for consecutive months, zero-balance code on the last record or None)
PANEL = {
    "F01": (0, ["00"] * 21, None),                                   # current, months 0-20
    "F02": (0, ["00"] * 5 + ["01"] + ["00"] * 15, None),             # "01" at month 5 (inside the lookback)
    "F03": (0, ["00"] * 13 + _codes(1, 8), None),                    # current to t, "01".."08" in months 13-20
    "F04": (0, ["02"] + ["00"] * 20, None),                          # "02" at month 0 = t-12 (outside)
    "F05": (0, ["00"] * 12 + ["01"] + ["00"] * 8, None),             # "01" at month 12 = t
    "F06": (0, ["00", "00", "01", "02", "03"] + ["00"] * 16, None),  # 90+ at month 4, then cured
    "F07": (0, ["00"] * 16, "01"),                                   # prepaid at month 15 (after t)
    "F08": (0, ["00"] * 11, "01"),                                   # prepaid at month 10 (before t)
    "F09": (0, ["00"] * 13 + ["01"] + ["00"] * 7, None),             # "01" at month 13 = t+1
    "F10": (8, ["00"] * 13, None),                                   # first record at month 8
}

# answer key: status_t, max_dq_12m, n_dq_months_12m, ever_90_before_t (F08 is omitted)
EXPECTED = pd.DataFrame.from_dict({
    "F01": [0, 0, 0, 0],
    "F02": [0, 1, 1, 0],
    "F03": [0, 0, 0, 0],
    "F04": [0, 0, 0, 0],
    "F05": [1, 1, 1, 0],
    "F06": [0, 3, 3, 1],
    "F07": [0, 0, 0, 0],
    "F09": [0, 0, 0, 0],
    "F10": [0, 0, 0, 0],
}, orient="index", columns=FEATURES).astype("int64").rename_axis("loan_id")


def _panel(loans=PANEL):
    """Long loan-month panel. Rows are ordered by month, with loans in reverse order within each month,
    so a sorted output cannot be inherited from the input order."""
    rows = [{"loan_id": loan, "month": first + i, "delinq_status": s,
             "zero_balance_code": code if i == len(statuses) - 1 else None}
            for loan, (first, statuses, code) in loans.items() for i, s in enumerate(statuses)]
    return pd.DataFrame(rows).sort_values(["month", "loan_id"], ascending=[True, False], ignore_index=True)


def _differences(a, b):
    """Rows or cells that differ between two feature tables, as 'loan.feature: a vs b'."""
    rows = sorted(set(a.index) | set(b.index))
    out = [f"{r}: {'present' if r in a.index else 'missing'} vs {'present' if r in b.index else 'missing'}"
           for r in rows if (r in a.index) != (r in b.index)]
    out += [f"{r}.{c}: {a.at[r, c]} vs {b.at[r, c]}" for r in rows if r in a.index and r in b.index
            for c in FEATURES if a.at[r, c] != b.at[r, c]]
    return "; ".join(out)


def _features(panel=None):
    return build_features(_panel() if panel is None else panel, T)


# ---------------------------------------------------------------- exact values and future invariance
def test_exact_expected_output():
    """[T1] (R1-R3) every row, column and value equals the answer key, with int dtype."""
    out = _features()
    diff = _differences(out, EXPECTED)
    assert not diff, f"got vs expected: {diff}"
    pdt.assert_frame_equal(out, EXPECTED)


def test_future_invariance_truncation():
    """[T2] STRONGEST EVIDENCE (R4, R5): removing every record after t changes no feature of any loan."""
    panel = _panel()
    full, truncated = build_features(panel, T), build_features(panel[panel["month"] <= T], T)
    diff = _differences(full, truncated)
    assert not diff, f"features change when records after t are removed (full vs truncated): {diff}"


def test_future_invariance_perturbation():
    """[T3] STRONGEST EVIDENCE (R4, R5): setting every status after t to "05" changes no feature of any loan."""
    panel = _panel()
    perturbed = panel.copy()
    perturbed.loc[perturbed["month"] > T, "delinq_status"] = "05"
    diff = _differences(build_features(panel, T), build_features(perturbed, T))
    assert not diff, f"features change when statuses after t are changed (original vs perturbed): {diff}"


def test_key_case_f03():
    """[T4] (R4) F03 is current through t and delinquent only from t+1: no delinquency in its features."""
    f03 = _features().loc["F03"]
    assert (f03["max_dq_12m"], f03["n_dq_months_12m"]) == (0, 0), \
        f"F03 max_dq_12m, n_dq_months_12m = {f03['max_dq_12m']}, {f03['n_dq_months_12m']}; expected 0, 0"


def test_boundary_t_plus_1_f09():
    """[T5] (R3, R4) F09's only delinquent month is t+1, which is outside the lookback."""
    value = _features().at["F09", "max_dq_12m"]
    assert value == 0, f"F09 max_dq_12m = {value}; expected 0"


# ---------------------------------------------------------------- rules a missing upper bound does not break
def test_month_t_included_f05():
    """[T6] (R3, R4) month t itself is part of the information set: F05 has "01" at t."""
    f05 = _features().loc["F05"]
    assert (f05["status_t"], f05["max_dq_12m"], f05["n_dq_months_12m"]) == (1, 1, 1)


def test_lookback_lower_boundary_f04():
    """[T7] (R3) month t-12 is outside the lookback t-11 .. t: F04's "02" at month 0 is not counted."""
    assert _features().at["F04", "max_dq_12m"] == 0


def test_past_90_plus_then_cured_f06():
    """[T8] (R3) F06 was "03" at month 4 and then cured."""
    f06 = _features().loc["F06"]
    assert (f06["ever_90_before_t"], f06["max_dq_12m"], f06["n_dq_months_12m"]) == (1, 3, 3)


def test_row_set():
    """[T9] (R1) F08 (terminated before t) is omitted; F07 (terminates after t) and F10 (short history) are
    included; the index is sorted."""
    out = _features()
    assert out.index.name == "loan_id"
    assert list(out.index) == ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F09", "F10"]
    assert list(out.columns) == FEATURES


def test_input_not_modified():
    """[T10] (R6) the panel passed in is unchanged."""
    panel = _panel()
    before = panel.copy(deep=True)
    build_features(panel, T)
    pdt.assert_frame_equal(panel, before)


def _with_status(code):
    panel = _panel()
    panel.loc[(panel["loan_id"] == "F01") & (panel["month"] == 3), "delinq_status"] = code
    return panel


INVALID_INPUTS = {
    "missing_zero_balance_code": (_panel().drop(columns="zero_balance_code"), T),
    "missing_delinq_status": (_panel().drop(columns="delinq_status"), T),
    "duplicate_loan_month": (pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True), T),
    "prediction_month_after_panel": (_panel(), 99),
    "prediction_month_before_panel": (_panel(), -1),
    "unknown_status_ZZ": (_with_status("ZZ"), T),
    "status_XX_not_available": (_with_status("XX"), T),
    "missing_status": (_with_status(None), T),
}


@pytest.mark.parametrize("panel, prediction_month", list(INVALID_INPUTS.values()), ids=list(INVALID_INPUTS))
def test_invalid_input_raises(panel, prediction_month):
    """[T11] (R2, R6) ValueError for missing columns, duplicate (loan_id, month), t not in the panel, or an
    invalid status code."""
    with pytest.raises(ValueError):
        build_features(panel, prediction_month)

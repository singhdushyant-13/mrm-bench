"""
MRM-Bench Pair 3 - executable tests of spec.md / spec.json (outcome window)
===========================================================================
Both cases in this pair run these SAME tests: the two test_case.py files are identical except for the
import line. The horizon H is read from THIS folder's spec.json, and the expected labels for that H are
selected from a table holding both columns (H = 12 and H = 6). O1-O7 refer to spec.md; [T1]-[T11]
number the tests.

All data are small SYNTHETIC panels (loan IDs S01-S10 and E01) - not Freddie Mac records. Expected labels
were computed in advance from the rule and rechecked with a separate reference labeller; they were not
copied from an implementation.
"""
import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from benchmark.pair3_horizon.faulty.impl import label_outcomes  # the only line that differs between the pair's test files

H = json.loads((Path(__file__).parent / "spec.json").read_text())["horizon_months"]  # this case's specified horizon
T = 0                                            # prediction month
ALLOWED = {"event", "non_event", "censored_prepay", "censored_other_termination", "censored_end_of_data"}
OMITTED = None                                   # not at risk at t -> no output row


def _codes(first, last):
    """Two-digit status codes, e.g. _codes(3, 5) -> ["03", "04", "05"]."""
    return [f"{d:02d}" for d in range(first, last + 1)]


# loan -> (statuses for months 0, 1, 2, ..., zero-balance code on the loan's last record or None)
PANEL_A = {
    "S01": (["00"] * 15, None),                                              # current, months 0-14
    "S02": (["00"] * 5 + ["01", "02"] + _codes(3, 10), None),                # first 90+ at month 7
    "S03": (["00", "00", "01", "02", "03", "04", "05"] + ["RA"] * 4, "09"),  # 90+ at month 4, REO, code 09 at 10
    "S04": (["00"] * 9, "01"),                                               # prepaid at month 8
    "S05": (_codes(3, 17), None),                                            # already 90+ at month 0
    "S06": (["00"] * 11, "15"),                                              # whole loan sale at month 10
    "S07": (["00", "00", "01", "02"] + ["00"] * 11, None),                   # 60-89 days at month 3, then current
    "S08": (["00"] * 12 + _codes(3, 5), None),                               # first 90+ at month 12
    "S09": (["00"] * 6 + _codes(3, 11), None),                               # jump 00 -> 03 at month 6
    "S10": (["00"] * 13 + _codes(3, 4), None),                               # first 90+ at month 13
}
PANEL_B = {"E01": (["00"] * 9, None)}                                         # panel ends at month 8

# answer key: expected label for each horizon (both columns; the test uses the column for H)
EXPECTED_A = {
    "S01": {12: "non_event", 6: "non_event"},
    "S02": {12: "event", 6: "non_event"},
    "S03": {12: "event", 6: "event"},
    "S04": {12: "censored_prepay", 6: "non_event"},
    "S05": {12: OMITTED, 6: OMITTED},
    "S06": {12: "censored_other_termination", 6: "non_event"},
    "S07": {12: "non_event", 6: "non_event"},
    "S08": {12: "event", 6: "non_event"},
    "S09": {12: "event", 6: "event"},
    "S10": {12: "non_event", 6: "non_event"},
}
EXPECTED_B = {"E01": {12: "censored_end_of_data", 6: "non_event"}}
EVENT_COUNT_A = {12: 4, 6: 2}
BOUNDARY = {12: {"S08": "event", "S10": "non_event"},                        # month t+12 included, t+13 excluded
            6: {"S09": "event", "S02": "non_event"}}                         # month t+6 included, t+7 excluded
CENSORING = {12: {"S04": "censored_prepay", "S06": "censored_other_termination"},
             6: {"S04": "non_event", "S06": "non_event"}}


def _panel(loans):
    """Long loan-month panel. Rows are ordered by month, with loans in reverse order within each month,
    so a sorted output cannot be inherited from the input order."""
    rows = [{"loan_id": loan, "month": m, "delinq_status": s,
             "zero_balance_code": code if m == len(statuses) - 1 else None}
            for loan, (statuses, code) in loans.items() for m, s in enumerate(statuses)]
    return pd.DataFrame(rows).sort_values(["month", "loan_id"], ascending=[True, False], ignore_index=True)


def _labels(loans):
    return label_outcomes(_panel(loans), T)["label"].to_dict()


def _expected(table):
    return {loan: by_h[H] for loan, by_h in table.items() if by_h[H] is not OMITTED}


def _mismatches(got, expected):
    """Loans whose label differs, as 'loan: got -> expected'."""
    loans = sorted(set(got) | set(expected))
    return "; ".join(f"{k}: {got.get(k, '(omitted)')} -> expected {expected.get(k, '(omitted)')}"
                     for k in loans if got.get(k) != expected.get(k))


# ---------------------------------------------------------------- labels that depend on the horizon
def test_exact_labels_panel_a():
    """[T1] (O2-O6) every label on panel A equals the expected label for this spec's horizon."""
    got, expected = _labels(PANEL_A), _expected(EXPECTED_A)
    assert got == expected, f"H = {H}: {_mismatches(got, expected)}"


def test_window_boundaries():
    """[T2] (O3) month t+H is inside the window and month t+H+1 is outside."""
    got, expected = _labels(PANEL_A), BOUNDARY[H]
    got = {k: got.get(k) for k in expected}
    assert got == expected, f"H = {H}: {_mismatches(got, expected)}"


def test_event_count():
    """[T3] (O4) the number of events on panel A for this spec's horizon."""
    n_events = sum(label == "event" for label in _labels(PANEL_A).values())
    assert n_events == EVENT_COUNT_A[H], f"H = {H}: {n_events} events, expected {EVENT_COUNT_A[H]}"


def test_censoring_before_end_of_window():
    """[T4] (O5, O6) prepayment / other termination before t+H is censored; at or after t+H it is non_event."""
    got, expected = _labels(PANEL_A), CENSORING[H]
    got = {k: got.get(k) for k in expected}
    assert got == expected, f"H = {H}: {_mismatches(got, expected)}"


def test_end_of_data_panel_b():
    """[T5] (O1, O6) E01 is current until the data end at month 8."""
    got, expected = _labels(PANEL_B), _expected(EXPECTED_B)
    assert got == expected, f"H = {H}: {_mismatches(got, expected)}"


# ---------------------------------------------------------------- rules that do not depend on the horizon
def test_at_risk_rule():
    """[T6] (O2) S05 is already 90+ at t, so it is not at risk and has no output row."""
    assert "S05" not in _labels(PANEL_A)


def test_60_89_days_is_not_an_event():
    """[T7] (O4) S07 reaches "02" (60-89 days) but never 90+, so it is non_event."""
    assert _labels(PANEL_A)["S07"] == "non_event"


def test_status_jump_is_valid():
    """[T8] (O4, spec section 4) S09 jumps from "00" straight to "03" at month 6: a valid event, not an error."""
    assert _labels(PANEL_A)["S09"] == "event"


def test_output_format():
    """[T9] (O7) index = sorted, unique loan IDs named loan_id; one `label` column; values from the allowed set."""
    out = label_outcomes(_panel(PANEL_A), T)
    assert out.index.name == "loan_id"
    assert list(out.index) == sorted(out.index) and out.index.is_unique
    assert list(out.columns) == ["label"]
    assert set(out["label"]) <= ALLOWED


def test_input_not_modified():
    """[T10] (O7) the panel passed in is unchanged."""
    panel = _panel(PANEL_A)
    before = panel.copy(deep=True)
    label_outcomes(panel, T)
    pdt.assert_frame_equal(panel, before)


INVALID_INPUTS = {
    "missing_zero_balance_code": (_panel(PANEL_A).drop(columns="zero_balance_code"), T),
    "missing_delinq_status": (_panel(PANEL_A).drop(columns="delinq_status"), T),
    "duplicate_loan_month": (pd.concat([_panel(PANEL_A), _panel(PANEL_A).iloc[[0]]], ignore_index=True), T),
    "prediction_month_after_panel": (_panel(PANEL_A), 99),
    "prediction_month_before_panel": (_panel(PANEL_A), -1),
}


@pytest.mark.parametrize("panel, prediction_month", list(INVALID_INPUTS.values()), ids=list(INVALID_INPUTS))
def test_invalid_input_raises(panel, prediction_month):
    """[T11] (O7) ValueError for missing columns, duplicate (loan_id, month) rows, or a prediction month not in the panel."""
    with pytest.raises(ValueError):
        label_outcomes(panel, prediction_month)

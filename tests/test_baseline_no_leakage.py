"""
Future-invariance (no-leakage) tests for build_snapshot() in the baseline notebook
==================================================================================
The function under test now lives in the notebook notebooks/baseline_logistic_regression.ipynb, the only
implementation of the baseline (the original script, src/baseline_logreg.py, is in git history at commit 9253e97).
The code cells tagged "leakage-test-source" (the cells that define ym_index() and build_snapshot()) are read with
nbformat and executed in a fresh namespace that holds only numpy, pandas and HORIZON = 12.

Rule (CLAUDE.md, "Outcome specification"): features at prediction month t may use only performance
records with period <= t. If that holds, deleting or changing any record after t cannot change any
feature at t. The label is excluded from the comparison: it is defined on months t+1..t+12.

  T1 truncation and T2 perturbation are the leakage tests (they cover every feature column and loan).
  T3 is a power check: one textual mutation of the notebook's build_snapshot() cell builds the 12-month window from
     ALL records instead of only those up to t, and the T1 comparison must FAIL for it. A test that can never fail
     proves nothing.
  T4 spot-checks a few feature values worked out by hand.

All data are SYNTHETIC (loan IDs T01-T08) - no Freddie Mac records.
"""
import functools
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "baseline_logistic_regression.ipynb"
TAG = "leakage-test-source"                      # cell tag on the notebook cells that define ym_index, build_snapshot
HORIZON = 12                                     # outcome window of the specification (in the notebook: SETTINGS cell)

T = 201712                                       # prediction month (YYYYMM), one of the baseline's snapshots
PERIODS = [f"{y}{m:02d}" for y in (2016, 2017, 2018) for m in range(1, 13)]   # 201601 .. 201812
NOT_FEATURES = {"label", "snapshot", "max_feature_period"}                   # the label uses the future by design

# loan -> overrides by period; statuses default to "00", the other fields to missing.
# Every loan runs 201601..201812 unless "last" ends it (with a zero-balance code on that record).
LOANS = {
    "T01": {"status": dict(zip(PERIODS[24:], [f"{d:02d}" for d in range(1, 13)]))},  # current to t, 01..12 after t
    "T02": {"status": {"201801": "01"}},                                            # delinquent exactly at t+1
    "T03": {"status": {"201706": "01"}},                                            # delinquent inside the lookback
    "T04": {"status": {"201603": "01", "201604": "02", "201605": "03"}},            # 90+ before t, then cured
    "T05": {"last": "201803", "zero_balance": "01"},                                # prepaid after t
    "T06": {"modification": {p: "Y" if p == "201802" else "P" for p in PERIODS if p >= "201802"}},  # modified after t only
    "T07": {"disaster": {"201803": "Y"}, "assistance": {p: "F" for p in ("201803", "201804", "201805")}},  # after t only
    "T08": {"modification": {p: "Y" if p == "201703" else "P" for p in PERIODS if p >= "201703"}},  # modified before t
}


# ---------------------------------------------------------------- the code under test, read from the notebook
@functools.cache
def _tagged_source():
    """Source of the notebook's code cells tagged TAG, in notebook order."""
    cells = [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells
             if c.cell_type == "code" and TAG in c.metadata.get("tags", [])]
    if not cells:
        pytest.fail(f"no code cell in {NOTEBOOK.name} is tagged {TAG!r}: the cells that define ym_index() and "
                    f"build_snapshot() must carry this tag in their metadata")
    return "\n\n".join(cells)


def _execute(source):
    """Run notebook code in a fresh namespace (numpy, pandas and HORIZON only) and return that namespace."""
    namespace = {"np": np, "pd": pd, "HORIZON": HORIZON}
    exec(compile(source, f"{NOTEBOOK.name} (cells tagged {TAG})", "exec"), namespace)
    missing = [name for name in ("ym_index", "build_snapshot") if not callable(namespace.get(name))]
    if missing:
        pytest.fail(f"the cells tagged {TAG!r} in {NOTEBOOK.name} do not define {', '.join(missing)}")
    return namespace


def _baseline():
    """ym_index() and build_snapshot() exactly as the notebook's tagged cells define them."""
    return _execute(_tagged_source())


def _t_idx():
    """The prediction month T as build_snapshot's internal month index t."""
    return int(_baseline()["ym_index"]([T])[0])


# ---------------------------------------------------------------- synthetic panel and comparison helpers
def _derive(perf):
    """Derived columns exactly as load() builds them."""
    perf = perf.copy()
    perf["t"] = _baseline()["ym_index"](perf.period).values
    perf["dq"] = pd.to_numeric(perf.delinq_status, errors="coerce")     # 'RA' -> NaN, as in load()
    perf["sev"] = (perf.dq >= 3) | (perf.delinq_status == "RA")
    return perf


def _perf(loans=LOANS):
    """Synthetic monthly panel in the format build_snapshot() receives from load()."""
    rows = []
    for loan, spec in loans.items():
        last = spec.get("last", PERIODS[-1])
        for age, period in enumerate(p for p in PERIODS if p <= last):
            rows.append({"loan_id": loan, "period": period, "current_upb": 200000.0 - 500.0 * age,
                         "delinq_status": spec.get("status", {}).get(period, "00"), "loan_age": age + 1,
                         "modification_flag": spec.get("modification", {}).get(period, np.nan),
                         "zero_balance_code": spec["zero_balance"] if period == last and "zero_balance" in spec else np.nan,
                         "delinq_due_to_disaster": spec.get("disaster", {}).get(period, np.nan),
                         "borrower_assistance_plan": spec.get("assistance", {}).get(period, np.nan)})
    return _derive(pd.DataFrame(rows))


def _features(snapshot_fn, perf):
    """Every column build_snapshot() returns except the label-related ones, one row per loan."""
    out = snapshot_fn(perf, T)
    return out[[c for c in out.columns if c not in NOT_FEATURES]].set_index("loan_id").sort_index()


def _differences(a, b):
    """Rows, columns or cells that differ between two feature tables, as 'loan.feature: a vs b'."""
    if list(a.index) != list(b.index) or list(a.columns) != list(b.columns):
        return f"rows/columns differ: {list(a.index)} vs {list(b.index)}; {list(a.columns)} vs {list(b.columns)}"
    return "; ".join(f"{r}.{c}: {a.at[r, c]!r} vs {b.at[r, c]!r}" for r in a.index for c in a.columns
                     if not (a.at[r, c] == b.at[r, c] or (pd.isna(a.at[r, c]) and pd.isna(b.at[r, c]))))


LEAKY_MUTATION = ("h12 = hist[(hist.t > t - 12) & hist.loan_id.isin(ids)]",   # window t-11..t (from hist)
                  "h12 = perf[(perf.t > t - 12) & perf.loan_id.isin(ids)]")   # same window, no upper bound


def _mutated_build_snapshot():
    """The notebook's build_snapshot() with the 12-month window built from all records (a deliberate leak)."""
    source = _tagged_source()
    old, new = LEAKY_MUTATION
    if source.count(old) != 1:
        pytest.fail(f"power check cannot run: the cells tagged {TAG!r} no longer contain exactly one line {old!r}")
    return _execute(source.replace(old, new))["build_snapshot"]


# ---------------------------------------------------------------- leakage tests
def test_future_invariance_truncation():
    """[T1] removing every record after t changes no feature of any loan (notebook's build_snapshot)."""
    build_snapshot = _baseline()["build_snapshot"]
    perf = _perf()
    diff = _differences(_features(build_snapshot, perf), _features(build_snapshot, perf[perf.t <= _t_idx()]))
    assert not diff, f"features change when records after t are removed (full vs truncated): {diff}"


def test_future_invariance_perturbation():
    """[T2] changing status, modification, disaster and assistance fields after t changes no feature (notebook's
    build_snapshot)."""
    build_snapshot = _baseline()["build_snapshot"]
    perf = _perf()
    after = perf.t > _t_idx()
    perturbed = perf.copy()
    perturbed.loc[after, "delinq_status"] = "05"
    perturbed.loc[after, "modification_flag"] = "Y"
    perturbed.loc[after, "delinq_due_to_disaster"] = "Y"
    perturbed.loc[after, "borrower_assistance_plan"] = "F"
    perturbed = _derive(perturbed)                   # recompute dq and sev, as load() would
    diff = _differences(_features(build_snapshot, perf), _features(build_snapshot, perturbed))
    assert not diff, f"features change when records after t are changed (original vs perturbed): {diff}"


def test_power_check_mutation_is_detected():
    """[T3] the T1 comparison FAILS for a mutated copy of the notebook's build_snapshot() whose 12-month window reads
    records after t."""
    leaky = _mutated_build_snapshot()
    perf = _perf()
    diff = _differences(_features(leaky, perf), _features(leaky, perf[perf.t <= _t_idx()]))
    assert diff, "the truncation comparison did not detect a 12-month window that reads records after t"
    assert "T01.max_dq_12m" in diff and "T02.n_dq_months_12m" in diff, f"unexpected differences: {diff}"


# ---------------------------------------------------------------- hand-checked values
def test_spot_check_feature_values():
    """[T4] a few features of the notebook's build_snapshot(), worked out by hand at t = 201712 (lookback
    201701..201712)."""
    out = _baseline()["build_snapshot"](_perf(), T).set_index("loan_id")
    assert out.loc["T01", "label"] == "event"                  # the key leak case really is an event after t ...
    assert (out.at["T01", "max_dq_12m"], out.at["T01", "n_dq_months_12m"]) == (0, 0)   # ... but clean at t
    assert out.at["T02", "max_dq_12m"] == 0                    # month t+1 is outside the lookback
    assert (out.at["T03", "max_dq_12m"], out.at["T03", "n_dq_months_12m"]) == (1, 1)   # 201706 is inside
    assert (out.at["T04", "ever_90_before_t"], out.at["T04", "max_dq_12m"]) == (1, 0)  # 90+ in 201605, before t-11
    assert (out.at["T06", "ever_modified"], out.at["T08", "ever_modified"]) == (0, 1)
    assert (out.at["T07", "disaster_flag_t"], out.at["T07", "assist_plan_t"]) == (0, "none")
    assert sorted(out.index) == [f"T0{i}" for i in range(1, 9)]  # T05 terminates after t, so it is included

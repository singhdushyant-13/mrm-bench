"""
MRM-Bench Pair 1 - executable tests of spec.md (Original DTI Ratio preprocessing)
=================================================================================
Both implementations in this pair face these SAME tests: the two test_case.py files are identical
except for the import line. Rules R1-R7 refer to spec.md.

All data are small SYNTHETIC values chosen by hand - not Freddie Mac records. Every expected value
was derived by hand from spec.md (see comments), not copied from an implementation's output.

Tests marked [999 rule] exercise the documented code "999 = Not Available". The key evidence for the
999 rule is test_training_processed_dti: processed values must be 10, 20, 30, 40, 25, 25.
"""
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from benchmark.pair1_preprocessing.valid.impl import prepare_dti  # the only line that differs between the pair's test files

# Synthetic training data: observed ratios 10, 20, 30, 40 and two "Not Available" codes.
TRAIN_DTI = [10, 20, 30, 40, 999, 999]
TRAIN_MEDIAN = 25.0                                      # R4: observed {10, 20, 30, 40} -> (20 + 30) / 2
TRAIN_MISSING = [0, 0, 0, 0, 1, 1]                       # R3: 1 exactly for the 999 rows
TRAIN_PROCESSED = [10.0, 20.0, 30.0, 40.0, 25.0, 25.0]   # R6: 999 rows get the median


def _dti(values, index=None):
    return pd.DataFrame({"orig_dti": values}, index=index)


def _assert_dti(out, processed, missing, median):
    """Exact check of the three outputs of R6 (values, indicator, stored median)."""
    pdt.assert_series_equal(out["orig_dti"], pd.Series(processed, index=out.index, name="orig_dti",
                                                       dtype="float64"), obj="processed orig_dti")
    pdt.assert_series_equal(out["dti_missing"], pd.Series(missing, index=out.index, name="dti_missing",
                                                          dtype="int64"), obj="dti_missing")
    assert out.attrs["dti_training_median"] == median


# ---------------------------------------------------------------- training mode (training_median=None)
def test_training_processed_dti():
    """[999 rule] KEY EVIDENCE (R3, R4, R6): the 999 rows are imputed with 25; observed ratios unchanged."""
    out = prepare_dti(_dti(TRAIN_DTI))
    pdt.assert_series_equal(out["orig_dti"], pd.Series(TRAIN_PROCESSED, name="orig_dti", dtype="float64"),
                            obj="processed orig_dti")


def test_training_median_is_25():
    """[999 rule] (R4) median of the observed ratios {10, 20, 30, 40} only - not of all six inputs."""
    out = prepare_dti(_dti(TRAIN_DTI))
    assert out.attrs["dti_training_median"] == TRAIN_MEDIAN


def test_training_missing_indicator():
    """[999 rule] (R3) dti_missing is 1 exactly for the two 999 rows."""
    out = prepare_dti(_dti(TRAIN_DTI))
    pdt.assert_series_equal(out["dti_missing"], pd.Series(TRAIN_MISSING, name="dti_missing", dtype="int64"),
                            obj="dti_missing")


def test_processed_dti_within_documented_range():
    """[999 rule] (range invariant of R2 + R6) no processed DTI is missing or outside [0, 65]."""
    processed = prepare_dti(_dti(TRAIN_DTI))["orig_dti"]
    assert processed.notna().all()
    assert processed.max() <= 65, f"processed DTI above 65: {processed[processed > 65].tolist()}"
    assert processed.min() >= 0


def test_boundary_values_kept():
    """(R2, R6) 0 and 65 are legal boundary ratios and are kept unchanged (false-alarm trap)."""
    # observed {0, 30, 65} -> median 30; nothing is missing
    _assert_dti(prepare_dti(_dti([0, 65, 30])), [0.0, 65.0, 30.0], [0, 0, 0], 30.0)


def test_genuine_nulls_are_missing():
    """(R3, R4, R6) a genuine null is missing too: observed {10, 30} -> median 20."""
    _assert_dti(prepare_dti(_dti([10, np.nan, 30])), [10.0, 20.0, 30.0], [0, 1, 0], 20.0)


def test_numeric_strings_from_raw_file():
    """[999 rule] (R2, R3, R6) raw-file text including the code "999": observed {40, 10} -> median 25."""
    _assert_dti(prepare_dti(_dti(["40", "999", "10", None])), [40.0, 25.0, 10.0, 25.0], [0, 1, 0, 1], 25.0)


# ---------------------------------------------------------------- inference mode (training_median supplied)
def test_inference_uses_supplied_training_median():
    """[999 rule] (R5, R6) validation rows 999, 12, 65 with training median 25 -> 25, 12, 65."""
    _assert_dti(prepare_dti(_dti([999, 12, 65]), training_median=25), [25.0, 12.0, 65.0], [1, 0, 0], 25.0)


def test_inference_does_not_refit_median():
    """(R5) no median is fitted on validation data: refitting on the observed {12, 65} would give 38.5,
    but the missing row must receive the supplied training median 25."""
    _assert_dti(prepare_dti(_dti([np.nan, 12, 65]), training_median=25), [25.0, 12.0, 65.0], [1, 0, 0], 25.0)


def test_empty_inference_data():
    """(R5, R7) empty validation data is processed normally when the training median is supplied."""
    out = prepare_dti(_dti([]), training_median=25)
    assert len(out) == 0
    assert list(out.columns) == ["orig_dti", "dti_missing"]
    assert out.attrs["dti_training_median"] == 25.0


# ---------------------------------------------------------------- structure (R7)
def test_input_dataframe_unchanged():
    """(R7) the caller's DataFrame, including its attrs, is not modified; a new DataFrame is returned."""
    df = _dti(TRAIN_DTI)
    df["loan_purpose"] = ["P", "C", "N", "P", "C", "N"]
    before = df.copy(deep=True)
    out = prepare_dti(df)
    assert out is not df
    pdt.assert_frame_equal(df, before)
    assert df.attrs == {}


def test_row_count_index_and_other_columns_preserved():
    """(R7) same rows, same order, same (non-default) index; other columns untouched; dti_missing appended."""
    index = pd.Index([42, 7, 3, 100, 8, 9], name="row")
    df = pd.DataFrame({"loan_purpose": ["P", "C", "N", "P", "C", "N"], "orig_dti": TRAIN_DTI,
                       "fico": [700, 720, 680, 750, 640, 810]}, index=index)
    out = prepare_dti(df)
    assert len(out) == len(df)
    pdt.assert_index_equal(out.index, index)
    assert list(out.columns) == ["loan_purpose", "orig_dti", "fico", "dti_missing"]
    pdt.assert_frame_equal(out[["loan_purpose", "fico"]], df[["loan_purpose", "fico"]])


# ---------------------------------------------------------------- errors (R1, R2, R4, R5)
def test_missing_orig_dti_column_raises():
    """(R1) the required column is absent."""
    with pytest.raises(ValueError, match="orig_dti"):
        prepare_dti(pd.DataFrame({"fico": [700, 720]}))


@pytest.mark.parametrize("bad", [66, 70, -5, 500, 9999, "abc"])
def test_invalid_values_raise(bad):
    """(R2) values outside [0, 65] that are not the code 999 or a null are rejected, in both modes."""
    with pytest.raises(ValueError):
        prepare_dti(_dti([10, 20, bad]))
    with pytest.raises(ValueError):
        prepare_dti(_dti([10, 20, bad]), training_median=25)


@pytest.mark.parametrize("bad_median", [np.nan, -1, 66, 999, "abc"])
def test_invalid_training_median_raises(bad_median):
    """(R5) a supplied training median must be a finite number in [0, 65]."""
    with pytest.raises(ValueError):
        prepare_dti(_dti([10, 999]), training_median=bad_median)


def test_empty_training_data_raises():
    """(R4) an empty training DataFrame has no observed ratio to fit a median."""
    with pytest.raises(ValueError):
        prepare_dti(_dti([]))


@pytest.mark.parametrize("values", [[np.nan, np.nan], [999, np.nan, 999]], ids=["all_null", "all_999_or_null"])
def test_all_missing_training_data_raises(values):
    """(R4) every training row is missing, so no median can be fitted ([999 rule] for all_999_or_null)."""
    with pytest.raises(ValueError):
        prepare_dti(_dti(values))

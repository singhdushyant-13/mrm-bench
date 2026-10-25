# TASK: Replace the baseline's ineffective leakage check with a real future-invariance test

> **How to use:** start a NEW Claude Code conversation and type:
> `Read CLAUDE.md and docs/claude_code_task_baseline_leakage_fix.md, then carry out the task phase by phase.`

## Background (simple English)

`src/baseline_logreg.py` contains this check inside `build_snapshot()`:

```python
feat["max_feature_period"] = ever.t.max()                  # used by the leakage assertion
assert (feat.max_feature_period <= t).all(), "LEAKAGE: feature uses a record after t"
```

`ever` is built from `hist = perf[perf.t <= t]`, so its maximum month is always ≤ t. **The assertion can never fail.** In the Pair 4 work, a copy of `build_snapshot()` with a leaky 12-month window (no upper bound) leaked on two loans while this assertion stayed silent.

The baseline's **results are correct**, because `h12` is built from `hist`. Only the safety check is ineffective. Your job is to replace it with a check that really works, **without changing any model result**.

The working check is the one Pair 4 uses: **future invariance**. If features at month t use only months ≤ t, then deleting or changing any record after t must not change them.

---

## PHASE 0 — Read first
1. Read `CLAUDE.md` (see "KNOWN ISSUE").
2. Read `src/baseline_logreg.py` completely, especially `load()` (the derived columns `t`, `dq`, `sev`) and `build_snapshot()`.
3. Read `benchmark/pair4_leakage/valid/test_case.py` for the T2/T3 invariance-test style.
4. Run `git status`. Do not modify anything under `benchmark/`, `data/` or `outputs/`.

---

## PHASE 1 — Add a real leakage test: `tests/test_baseline_no_leakage.py`

1. **Import `build_snapshot`** from `src/baseline_logreg.py` without running the script (use `importlib.util.spec_from_file_location`; the script is guarded by `if __name__ == "__main__"`).

2. **Build a small synthetic `perf` panel** in exactly the format `build_snapshot()` expects after `load()`:
   - columns `loan_id`, `period` (YYYYMM string), `t` (month index, computed with the module's own `ym_index`), `current_upb`, `delinq_status`, `loan_age`, `modification_flag`, `zero_balance_code`, `delinq_due_to_disaster`, `borrower_assistance_plan`, `dq`, `sev`;
   - derive `dq` and `sev` the same way `load()` does.

   Use synthetic loan IDs (e.g. `T01`…), never the Freddie Mac pattern. Use a prediction month T = 201712 and months 201601–201812, so there are records both before and after t. Include at least:
   - a loan that is current through t and delinquent after t (the key leak case);
   - a loan delinquent exactly at month t+1;
   - a loan delinquent inside the lookback;
   - a loan that was 90+ before t and then cured;
   - a loan that terminates (zero-balance code) after t;
   - a loan with a modification flag after t only;
   - a loan with a disaster flag or assistance plan after t only.

3. **Feature columns to compare:** every column `build_snapshot()` returns **except** `label`, `snapshot` and `max_feature_period`. The label legitimately uses the future.

4. **Tests:**
   - **T1 truncation:** the features from `build_snapshot(perf, T)` equal the features from `build_snapshot(perf_truncated, T)`, where `perf_truncated` keeps only rows with `t <= t_T`. Compare the feature columns only.
   - **T2 perturbation:** the features are unchanged when every row with month > t gets `delinq_status = "05"` (recompute `dq` and `sev`), `modification_flag = "Y"`, `delinq_due_to_disaster = "Y"` and `borrower_assistance_plan = "F"`.
   - **T3 power check (mutation test):** prove the test can actually catch a leak. Read the source of `build_snapshot` with `inspect.getsource`, and make one textual mutation that removes the upper bound from the 12-month window (`h12 = hist[...]` built from `perf` instead of `hist`). `exec` the mutated function in a copy of the module namespace, and assert that T1's comparison **fails** for it. If the exact text to replace is not found, the test must fail loudly, not pass silently.
   - **T4 sanity:** spot-check 2–3 feature values by hand (e.g. the key leak loan has `max_dq_12m == 0` at T).

---

## PHASE 2 — Remove the misleading assertion from the baseline

In `src/baseline_logreg.py`:
- delete the two lines that create `max_feature_period` and assert on it;
- replace them with a one-line comment: `# Leakage is tested by tests/test_baseline_no_leakage.py (future-invariance + mutation check).`

Make **no other change** to the file. Show the full `git diff src/baseline_logreg.py`.

---

## PHASE 3 — Prove the model results did not change

1. Run `uv run pytest tests -v`. All tests must pass, including T3 (the mutated function is detected).
2. Re-run the baseline: `uv run python src/baseline_logreg.py --data data/raw/sample_2015.zip`. It takes about 3 minutes.
3. Confirm the output is **identical** to the committed run:
   - label counts per snapshot unchanged;
   - `Train rows 101,843 (events 462) | Test rows 15,716 (events 119) | loan overlap = 0`;
   - test ROC-AUC 0.894 [0.849 0.93], PR-AUC 0.307, Brier 0.00620;
   - same coefficients.

   If anything differs, stop and report. Do not "fix" numbers.
4. Run all four benchmark runners (`run_pair1.py` … `run_pair4.py`). All must exit 0.
5. Check that no file under `benchmark/` changed (`git diff --stat`).

---

## PHASE 4 — Update CLAUDE.md
Replace the "KNOWN ISSUE" paragraph with a short note that the issue was fixed: the tautological assertion was removed and the leakage check now lives in `tests/test_baseline_no_leakage.py` (future invariance plus mutation power check), with results unchanged.

---

## PHASE 5 — Report back
A. What was wrong with the old assertion, in two sentences.
B. The new test file explained in simple English, especially why T3 matters (a test that can never fail proves nothing).
C. `git diff src/baseline_logreg.py`.
D. Actual output of `uv run pytest tests -v`, the key baseline lines (counts, train/test, the three metric lines), and the four runner verdicts.
E. Confirmation that results are identical and that no benchmark file changed.
F. Suggested commit message. Do not commit unless I ask.

**Scope:** only this fix. No new features, no model changes, no XGBoost or hazard model, no new benchmark pairs.

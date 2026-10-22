# MRM-Bench Pair 3 — Horizon: 6-month outcome window

**Category:** horizon  **Pair ID:** P3  **Cases:** `P3_F` (faulty/, defect) and `P3_V` (valid/, no defect)

## What this pair tests

The model's outcome asks whether a loan becomes 90+ days delinquent within the next H months. Both cases
in this pair use **exactly the same code**, which looks 6 months ahead (`HORIZON_MONTHS = 6`). What differs
is the specification the code is supposed to meet.

- **`P3_F`, defect.** Its specification is the project's main task, a **12-month** outcome. Code that looks only 6
  months ahead misses events in months 7–12 and labels some unfinished observations as "no event". A realistic way
  this happens is reusing code from an older 6-month model.
- **`P3_V`, valid.** Its specification is a separately documented **6-month** early-warning task. The same
  code is correct here.

Pairs 1 and 2 had identical specifications and code that differed by one line. This pair is the reverse:
**identical code, specifications that differ in one parameter.** A validator therefore cannot decide it by
reading the code alone. It has to check whether the code fits the documented specification:

- flagging "6 months" without reading the spec gives a false positive on `P3_V`;
- never comparing the code with the spec misses `P3_F`.

## Files

| file | what it is |
|------|------------|
| `faulty/spec.md`, `valid/spec.md` | The specification: task, outcome rule O1–O7 for horizon H, why the horizon matters (with the synthetic example), what is *not* a defect, applicability conditions and provenance. **Same wording; they differ only in the task name, H (12 vs 6) and directly dependent sentences.** |
| `faulty/spec.json`, `valid/spec.json` | The machine-readable parameter of each spec: `{"task": ..., "horizon_months": 12}` for the faulty case and `6` for the valid one. The tests read H from here. |
| `faulty/impl.py`, `valid/impl.py` | `label_outcomes(panel, prediction_month, data_cutoff=None)`. **Byte-identical** (checked with `cmp`), with `HORIZON_MONTHS = 6`. |
| `faulty/label.json`, `valid/label.json` | Ground truth set by the benchmark designers: `is_defect`, violated rule (or `null`), evidence and source. |
| `faulty/test_case.py`, `valid/test_case.py` | Tests T1–T11 on synthetic panels. **Identical except for the import line.** Each reads H from its own folder's `spec.json` and picks the matching column of an answer key that holds both columns (H = 12 and H = 6). |
| `__init__.py` (here and in each sub-folder) | Empty package markers, as in Pairs 1–2. |

Related files one level up: `benchmark/run_pair3.py` has the same design as `run_pair2.py`. Its table also
shows each case's spec horizon and the code's `HORIZON_MONTHS`. `benchmark/test_no_real_data.py` already
scans this folder.

## The single difference between the cases

```diff
--- faulty/spec.json
+++ valid/spec.json
@@ -1,4 +1,4 @@
 {
-  "task": "12-month 90+ DPD serious delinquency",
-  "horizon_months": 12
+  "task": "6-month 90+ DPD early-warning indicator",
+  "horizon_months": 6
 }
```

`spec.md` differs in the same way: in the task name, in H, and in sentences that depend directly on them. Run
`diff faulty/spec.md valid/spec.md` to see it. `impl.py` is identical.

## Expected test outcomes (answer key)

| test | rule | `P3_V` (6-month spec) | `P3_F` (12-month spec) | why |
|---|---|---|---|---|
| T1 exact labels, panel A | O2–O6 | pass | **fail** | S02, S04, S06 and S08 are mislabelled |
| T2 window boundaries | O3 | pass | **fail** | S08's 90+ month at t+12 is missed |
| T3 event count, panel A | O4 | pass (2) | **fail** (2 instead of 4) | events in months 7–12 are missed |
| T4 censoring | O5, O6 | pass | **fail** | S04 (prepaid at 8) and S06 (sold at 10) become non_event |
| T5 end of data, panel B | O1, O6 | pass | **fail** | E01 (data end at 8) becomes non_event |
| T6 at-risk rule (S05 omitted) | O2 | pass | pass | does not depend on H |
| T7 60–89 days is not an event (S07) | O4 | pass | pass | does not depend on H |
| T8 status jump 00→03 is valid (S09) | O4 | pass | pass | event at month 6 is inside both windows |
| T9 output format | O7 | pass | pass | |
| T10 input not modified | O7 | pass | pass | |
| T11 validation errors | O7 | pass | pass | |

## How to run (from the repository root)

```bash
uv run python benchmark/run_pair3.py               # summary table; exit code 0 = both cases agree with labels
uv run pytest benchmark/pair3_horizon/valid -v      # expected: all pass
uv run pytest benchmark/pair3_horizon/faulty -v     # expected: T1-T5 fail
```

## Limitations and decisions

- **Label blindness matters more here than in Pairs 1–2.** A future validator must receive `impl.py` *together
  with that case's* `spec.md`/`spec.json` under neutral names. Without the spec, the two cases are
  indistinguishable by design. Folder names, case IDs, labels and tests reveal the answer.
- **Records that stop with no zero-balance code** before t+H, while the data continue, are labelled
  `censored_end_of_data`. This matches the project baseline (`build_snapshot()`); the brief's five
  categories did not name this case. It does not occur in the test panels.
- **Status "XX" (Not Available) at t** makes a loan not at risk under this spec (only "00", "01" and "02"
  qualify). The baseline would keep it. The 2015 sample has no "XX" records, so the two agree on the
  project data (spec.md, A4).
- **`data_cutoff`:** records after it are ignored. `month` must be a consecutive month index, not raw
  YYYYMM values (spec.md, A2).

## Data

All panels are synthetic and use the fictional loan IDs S01–S10 and E01. No Freddie Mac data are used.

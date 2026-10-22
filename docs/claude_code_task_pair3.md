# TASK: Implement MRM-Bench Pair 3 — Prediction-Horizon Defect

> **How to use:** start a NEW Claude Code conversation and type:
> `Read CLAUDE.md and docs/claude_code_task_pair3.md, then carry out the task phase by phase.`

## ROLE

Act as a senior Python machine learning engineer, quantitative model validation researcher, and academic research software developer, contributing to **MRM-Bench: Assumption-Grounded Executable Verification of Credit Risk Models Using LLM Agents** (Master's research, UNT, supervised by Dr. Chih-Hao Ku).

Pairs 1 and 2 are finished, verified and committed. **Follow their conventions** (folder layout, labels, runner design, data-hygiene check, identical test files) except for the one deliberate change explained in Phase 1. Do not modify Pairs 1–2.

---

# PHASE 0 — READ FIRST

1. Read `CLAUDE.md` completely (especially "Outcome specification").
2. Read `benchmark/pair1_preprocessing/` and `benchmark/pair2_formula/` (one README, one spec, one impl, one test file), plus `benchmark/run_pair2.py` and `benchmark/test_no_real_data.py`.
3. Read `build_snapshot()` in `src/baseline_logreg.py`. It is the project's reference labelling logic (at-risk rule, event, non-event, censoring). Do not modify it.
4. Run `git status`.

Do not write implementation code until this is done.

**Protected (do not modify):** `src/baseline_logreg.py`; everything in `benchmark/pair1_preprocessing/`, `benchmark/pair2_formula/`, `run_pair1.py`, `run_pair2.py`, `test_no_real_data.py`; `data/`, `outputs/`. If something seems to need changing, report it instead.

---

# PHASE 1 — WHAT IS DIFFERENT ABOUT THIS PAIR (read carefully)

In Pairs 1–2 the two specs were identical and the code differed by one line.

**In Pair 3 it is the reverse: the two `impl.py` files are byte-identical, and the two specs differ in one parameter.**

- Both implementations label outcomes using a **6-month** window.
- **Faulty case (P3_F):** its spec is the project's main task, a **12-month** 90+ DPD outcome. A 6-month window does not meet that spec, so it is a defect (realistic scenario: code reused from an older 6-month model).
- **Valid case (P3_V):** its spec is a **separately documented 6-month early-warning task**. The same code is correct here.

Why this matters for the research: a validator cannot decide this pair by reading the code alone. It must check whether the code is **applicable to the documented specification**. This tests the central research question ("assumption-applicability checks") directly. A validator that flags "6 months" as wrong without reading the spec raises a false positive on P3_V; one that never compares against the spec misses P3_F.

---

# PHASE 2 — THE OUTCOME RULE (from CLAUDE.md, generalized to horizon H)

For a monthly loan panel and a prediction month t:

- **At risk at t:** the loan has a record at month t, no zero-balance code at or before t, and status at t in {"00","01","02"}. Loans not at risk are **omitted** from the output.
- **Window:** months t+1 … t+H (t excluded, t+H included).
- **event:** any month in the window with numeric status ≥ 03, or status "RA". A 90+ month before a termination still counts as an event.
- **non_event:** no event, and the loan's last record is at month ≥ t+H (a zero-balance record exactly at t+H counts as observed).
- **censored_prepay:** no event, the loan ends before t+H with zero-balance code "01".
- **censored_other_termination:** no event, the loan ends before t+H with any other zero-balance code.
- **censored_end_of_data:** no event, no zero-balance code, and t+H > data_cutoff.
- Status can jump (e.g. "00" straight to "03"); this is valid data, not an error (Freddie Mac reporting cycles; see `docs` and CLAUDE.md).
- This is a research outcome ("H-month 90+ DPD serious delinquency"), not a regulatory default definition.

---

# PHASE 3 — DESIGN

```text
benchmark/
├── pair3_horizon/
│   ├── README.md   __init__.py
│   ├── faulty/  __init__.py  spec.md  spec.json  impl.py  label.json  test_case.py
│   └── valid/   __init__.py  spec.md  spec.json  impl.py  label.json  test_case.py
└── run_pair3.py
```

New file type: **`spec.json`**, the machine-readable parameter of each spec:
- faulty: `{"task": "12-month 90+ DPD serious delinquency", "horizon_months": 12}`
- valid: `{"task": "6-month 90+ DPD early-warning indicator", "horizon_months": 6}`

---

# PHASE 4 — IMPLEMENTATION (one file, copied byte-for-byte into both folders)

```python
label_outcomes(panel, prediction_month, data_cutoff=None) -> pandas.DataFrame
```
- `panel`: DataFrame with columns `loan_id` (str), `month` (int), `delinq_status` (str, e.g. "00", "03", "RA"), `zero_balance_code` (str or None; only on a loan's last record).
- `prediction_month`: int t.
- `data_cutoff`: int or None; None means `panel["month"].max()`.
- Returns a DataFrame indexed by `loan_id` (sorted), one row per at-risk loan, with a column `label` taking one of the five values in Phase 2.
- A module-level constant **`HORIZON_MONTHS = 6`** with an ordinary docstring/comment describing a 6-month outcome window. No comments mentioning the benchmark, the 12-month task, defects or bugs.
- Must not modify its input.
- Validation (raise `ValueError`): missing required columns; duplicate (`loan_id`, `month`) rows; `prediction_month` not present in the panel.

**The two `impl.py` files must be byte-identical.** Verify with `cmp` and report it.

---

# PHASE 5 — SPECIFICATIONS (the ONLY content difference between the cases)

Both `spec.md` files share the same structure and wording, differing **only** in the task name and horizon (12 vs 6) and directly dependent sentences. Include:
1. Purpose: the task named in `spec.json`.
2. The outcome rule of Phase 2 with H replaced by the case's horizon.
3. Why the horizon matters: a shorter window misses events in months H_impl+1 … H_spec and turns some censored loans into non-events. Give the synthetic example (see Phase 7).
4. What is NOT a defect: status jumps 00→03; prepayment while delinquent; a 90+ month before termination counting as an event; a 6-month window **when the spec documents a 6-month task**.
5. Applicability conditions; provenance (Freddie Mac SFLLD General User Guide, Release 47, delinquency status and zero-balance code definitions, `data/raw/general_user_guide_july_2026.pdf` pp.10–11; project outcome specification in CLAUDE.md). Confirm the page references by opening the PDF; do not invent quotes.

Show `diff faulty/spec.md valid/spec.md` in the report. It should touch only horizon and task-name lines.

---

# PHASE 6 — LABELS

- `faulty/label.json`: `"case_id": "P3_F"`, `"pair_id": "P3"`, `"category": "horizon"`, `"is_defect": true`, `"violated_rule": "Specification requires a 12-month outcome window (t+1..t+12); the implementation uses 6 months (t+1..t+6)."`, evidence naming the synthetic counterexamples (S02, S08 missed events; S04, S06 censoring turned into non-events; E01).
- `valid/label.json`: `"case_id": "P3_V"`, `"is_defect": false`, `"violated_rule": null`, evidence that the 6-month implementation matches the documented 6-month specification.
- `"source"` in both: `"Freddie Mac SFLLD General User Guide, Release 47 (July 2026); project outcome specification (CLAUDE.md)"`.

---

# PHASE 7 — SYNTHETIC DATA AND EXACT EXPECTED LABELS (pre-computed; do not change them)

Use synthetic loan IDs `S01`… and `E01`. They must never match the Freddie Mac ID pattern. Prediction month **t = 0**. `data_cutoff=None`.

### Test panel A (months 0–14 unless stated)

| loan | statuses by month | zero-balance |
|---|---|---|
| S01 | "00" for months 0–14 | – |
| S02 | "00" m0–4, "01" m5, "02" m6, then "03","04",…,"10" for m7–14 | – |
| S03 | "00" m0–1, "01" m2, "02" m3, "03" m4, "04" m5, "05" m6, "RA" m7–10 | "09" on m10 (last record) |
| S04 | "00" m0–8 | "01" on m8 (last record) |
| S05 | "03" m0, then "04",…,"17" m1–14 | – |
| S06 | "00" m0–10 | "15" on m10 (last record) |
| S07 | "00","00","01","02" m0–3, then "00" m4–14 | – |
| S08 | "00" m0–11, "03","04","05" m12–14 | – |
| S09 | "00" m0–5, then "03","04",…,"11" m6–14 | – |
| S10 | "00" m0–12, "03","04" m13–14 | – |

### Test panel B
| loan | statuses | zero-balance |
|---|---|---|
| E01 | "00" for months 0–8 (panel ends at 8) | – |

### Expected labels

| loan | H = 12 | H = 6 | what it tests |
|---|---|---|---|
| S01 | non_event | non_event | always current |
| S02 | event | non_event | event in month 7 (outside 6, inside 12) |
| S03 | event | event | 90+ at month 4 before REO/termination still an event |
| S04 | censored_prepay | non_event | prepay at month 8 |
| S05 | (omitted) | (omitted) | already 90+ at t, so not at risk |
| S06 | censored_other_termination | non_event | loan sale at month 10 |
| S07 | non_event | non_event | 60–89 days is not the event |
| S08 | event | non_event | jump 00→03 exactly at month 12 (boundary included) |
| S09 | event | event | jump 00→03 at month 6 (valid data, not an error) |
| S10 | non_event | non_event | event at month 13 is outside the 12-month window |
| E01 | censored_end_of_data | non_event | panel ends at month 8 |

Event counts on panel A: **4** for H = 12 (S02, S03, S08, S09); **2** for H = 6 (S03, S09).

These were computed with an independent reference implementation of the rule. Recompute them yourself (Phase 11) with code written separately from `impl.py`.

---

# PHASE 8 — TESTS (identical files except the import line)

Each `test_case.py` reads the horizon from **its own folder's** `spec.json` (`Path(__file__).parent / "spec.json"`) and selects the matching expected column from a dictionary containing BOTH columns of the table above. The two test files must be identical except for the import line. Show the `diff`.

Required tests (docstrings cite the rule):
- T1 Exact labels, panel A, for the spec's horizon.
- T2 Window boundaries: H=12 → S08 event (month t+12 included), S10 non_event (t+13 excluded); H=6 → S09 event (t+6 included), S02 non_event (t+7 excluded).
- T3 Event count on panel A equals the expected count for the spec's horizon.
- T4 Censoring: H=12 → S04 censored_prepay and S06 censored_other_termination; H=6 → both non_event.
- T5 End of data, panel B: E01 label for the spec's horizon.
- T6 At-risk rule: S05 is omitted.
- T7 60–89 days is not an event: S07 non_event.
- T8 Status jump is valid: S09 event.
- T9 Output format: index is sorted loan IDs, a single `label` column, values from the allowed set.
- T10 Input not modified.
- T11 Validation: `ValueError` for missing columns, duplicate (loan_id, month), and a prediction month not in the panel.

**Expected outcome:** valid passes everything. Faulty (6-month code against a 12-month spec) **fails T1–T5** and **passes T6–T11**. If the faulty case fails or passes anything else, the pair is wrong; fix the implementation or tests, never the expected values.

---

# PHASE 9 — RUNNER `benchmark/run_pair3.py`

Same design as `run_pair2.py` (each suite in its own process; only assertion failures count as defect evidence; collection, import and runtime errors never do; summary table; data-hygiene check; exit 0 only if both cases agree with labels). In the summary, also print each case's `spec.json` horizon and the implementation's `HORIZON_MONTHS`, so the reason for the verdict is visible.

---

# PHASE 10 — RUN (from the repo root)

```bash
uv run pytest benchmark/pair3_horizon/valid -v
uv run pytest benchmark/pair3_horizon/faulty -v     # expected assertion failures
uv run pytest benchmark/test_no_real_data.py -v
uv run python benchmark/run_pair3.py
uv run python benchmark/run_pair1.py && uv run python benchmark/run_pair2.py   # regression check
```
No new dependencies should be needed. Do not hide failures or change tests to look green.

---

# PHASE 11 — INDEPENDENT VALIDATION

1. Write a separate, throwaway reference labeller (in your scratchpad, not in `benchmark/`), loop-based and not copied from `impl.py`, and reproduce both columns of the expected table.
2. `cmp` the two `impl.py` files (must be identical); `diff` the specs (only horizon/task lines) and the test files (only the import line).
3. Confirm the faulty case fails exactly T1–T5.
4. Check that the labelling logic in `impl.py` (with H = 12) agrees with `build_snapshot()` in `src/baseline_logreg.py` on panel A, by reasoning or a small scratch script. Report any disagreement instead of silently resolving it.
5. Data hygiene; protected files unchanged (`git diff --stat`); Pair 1 and Pair 2 runners still exit 0.

---

# PHASE 12 — REPORT BACK

A. File tree with a simple-English explanation of each file (explain `spec.json`).
B. Explain, using S02, S04, S08 and E01, exactly how the 6-month code changes the labels under the 12-month spec, and why the identical code is correct under the 6-month spec.
C. Actual outputs of all commands in Phase 10.
D. Research interpretation: why P3_F is defective, why P3_V is valid, why this pair specifically tests assumption-applicability, and what it does NOT prove.
E. Repository checks and limitations.
F. Suggested commit message. Do not commit unless I ask.

---

# ACCEPTANCE CRITERIA
- Byte-identical `impl.py` files with `HORIZON_MONTHS = 6`; specs differ only in horizon and task name; `spec.json` present in both.
- Test files identical except the import line; they read the horizon from their own `spec.json`.
- Valid passes all; faulty fails exactly T1–T5 as assertion failures.
- Expected labels match the table exactly and were independently recomputed.
- Synthetic data only; hygiene passes; protected files unchanged; Pair 1 and Pair 2 runners still exit 0.
- Report includes real outputs and a commit message.

**Do not begin Pair 4, VALID-Agent, RAG or XGBoost. This task is limited to Pair 3.**

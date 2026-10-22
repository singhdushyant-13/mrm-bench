# CLAUDE.md — MRM-Bench pilot (Finance AI System)

## What this project is
Master's research project (Dushyant Pratap Singh, UNT; advisor Dr. Chih-Hao Ku).
Working title: "MRM-Bench: Assumption-Grounded Executable Verification of Credit Risk Models Using LLM Agents."
Central question (do not change): "At matched defect-detection recall, can assumption-applicability checks and independent verification reduce false-positive findings?"

We are NOT building a lending-decision system. We are building a benchmark of credit-risk model code:
clean reference code, deliberately faulty variants, and similar-looking VALID variants, each with an
independently defensible answer key. An AI validator (VALID-Agent) comes later — do not build it yet.

## Professor's rules (highest priority)
- Logistic regression only. Do NOT use XGBoost or hazard/survival models.
- Pilot first: 3–4 defect types, each with a similar-looking valid twin.
- The answer key must be independently defensible (spec + math + executable test).

## Environment
- macOS, VS Code, uv. Python 3.12 (.python-version). Run code with `uv run python ...`.
- Add packages with `uv add <pkg>` (never pip). Commit pyproject.toml and uv.lock.
- Data: `data/raw/sample_2015.zip` (Freddie Mac SFLLD 2015 sample, Release 47, July 2026 layout),
  plus file_layout_july_2026.xlsx and general_user_guide_july_2026.pdf in data/raw/.
- NEVER commit or copy Freddie Mac loan-level records into the repo, benchmark, tests, or docs.
  data/, sflld_extracted/, outputs/ are gitignored. Benchmark tests must use small SYNTHETIC data.

## Done so far (verified)
- Data checked: 50,000 loans (31 origination fields), 3,467,166 monthly records (35 fields),
  periods 2015-01..2026-03, IDs match, no duplicate loan-months.
- Clean baseline: `src/baseline_logreg.py` (git commit 9c743e2). Do not modify it casually — it is the
  reference. Run: `uv run python src/baseline_logreg.py --data data/raw/sample_2015.zip` (~3 min).
  Expected: train 101,843 rows / 462 events; test 15,716 rows / 119 events; test ROC-AUC 0.894.

## Outcome specification (draft, pending professor approval)
- Prediction date t: end of each December 2015..2024 (loan-snapshot design).
- At risk at t: record at t, no zero-balance code at or before t, delinquency status at t in {00,01,02}.
- Features: origination fields + performance records with period <= t ONLY.
- Event (y=1): any month t+1..t+12 with numeric status >= 03 or status 'RA'.
  This is "12-month 90+ DPD serious delinquency", NOT a regulatory default definition.
- Non-event (y=0): no event and loan still observed at t+12.
- Censored (excluded, counted): prepay (zero-balance code 01), other termination (02/03/09/15/16/96)
  before t+12 without a 90+ month, or window past 2026-03.
- Split: train Dec 2015–2018, test Dec 2021–2024, 2019–2020 = stress test only (COVID forbearance).
  Loan-disjoint: SHA-256(loan_id) % 100 < 70 -> train pool. No loan in both train and test.
- Documented sentinels -> NaN: FICO 9999; LTV/CLTV/DTI/MI 999; units/borrowers 99.
  DTI 999 = all HARP loans (4,017 of 4,020).

## Benchmark status and plan
Folder: `benchmark/`. One subfolder per pair. The detailed brief for each pair is in
`docs/claude_code_task_pair<N>.md` and takes precedence over this summary.
1. Preprocessing (DONE, commit 7fa098c): `pair1_preprocessing/`. Faulty uses DTI 999 as a number; valid
   converts 999 to NaN, adds a missing flag, and imputes with the training median.
2. Formula (BUILT, verified): `pair2_formula/`. Faulty p = 1/(1+exp(+z)); valid p = 0.5*(1+tanh(z/2)).
   NOTE: exp(z)/(1+exp(z)) is NOT used as the valid twin, because it returns NaN at large z (second defect).
3. Horizon (BUILT, verified): `pair3_horizon/`. The impl.py files are BYTE-IDENTICAL (HORIZON_MONTHS = 6); the SPECS
   differ (faulty spec = 12-month task, valid spec = 6-month early-warning task; also `spec.json`). The code
   alone cannot decide this pair; it tests spec applicability.
4. Leakage (BUILT, verified): `pair4_leakage/`. Faulty 12-month window has no upper bound (sees months > t);
   valid uses months t-11..t (month t is allowed: false-positive trap). Future-invariance tests (truncate or
   perturb months > t) are the general leakage check.
FIXED (baseline leakage check): the tautological assertion `max_feature_period <= t` (computed from hist, which is
already filtered to months <= t, so it could never fail) was removed from build_snapshot(). The leakage check now lives
in `tests/test_baseline_no_leakage.py`: future invariance (truncate or perturb all records after t) plus a mutation
power check (a copy of build_snapshot() with a leaky 12-month window must be detected). Run: `uv run pytest tests`.
Baseline results unchanged: console output and all files in outputs/ byte-identical to the committed run.

Actual structure (follow it for new pairs). Each pair folder has `README.md` plus `faulty/` and `valid/`,
each containing:
- `spec.md`: normally byte-identical in both cases (EXCEPTION: Pair 3, where only the spec differs), with math and provenance;
- `impl.py`: same interface and identical validation; exactly one intended difference; no comments revealing the label;
- `label.json`: {"case_id", "pair_id", "category", "is_defect", "violated_rule", "evidence", "source"};
- `test_case.py`: exact assertions on synthetic data, IDENTICAL in both except the import line.
`benchmark/run_pair<N>.py` runs each suite in its own process and compares outcomes with the labels. Only
assertion failures count as defect evidence; import, collection, and runtime errors never do.
`benchmark/test_no_real_data.py` scans all of benchmark/ for Freddie Mac loan IDs and large files.
Running pytest over the whole repo exits non-zero BY DESIGN (the faulty suites fail). The runners are the verdict.
Also keep false-alarm traps in mind (valid behaviour that looks suspicious): delinquency can jump
from 00 to 03 in one month; loans can prepay while 90+; zero-balance date can be one month after period.

## How to work
- Explain in simple English first, then the math/code. Comment important code sections.
- Separate clearly: completed, proposed, unverified.
- Never invent data, results, or citations. Do not report metrics you did not actually run.
- Do not treat poor accuracy, odd coefficients, or distribution shift as implementation defects.
- Small steps; run tests (`uv run pytest`) before saying something works; suggest a git commit after each step.

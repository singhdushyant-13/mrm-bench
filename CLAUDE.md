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

## Benchmark plan (next work)
Folder: `benchmark/`. One subfolder per case, e.g. `benchmark/pair1_preprocessing/`.
Defect categories and pairs:
1. Preprocessing — faulty: DTI 999 used as a real number; valid: 999 -> NaN + missing flag (same-looking code).
2. Formula — faulty: wrong sign in sigmoid, p = 1/(1+exp(+z)); valid: algebraically equivalent p = exp(z)/(1+exp(z)).
3. Horizon — faulty: 6-month window under a 12-month spec; valid: 6-month window under a separately documented 6-month spec.
4. Leakage — faulty: feature uses records after t (e.g., max status over t+1..t+12 or final zero-balance code);
   valid: max status over t-11..t.
Each pair folder contains two case folders (faulty, valid) plus an independent checker and pytest file:
- per case: `spec.md` (the rule + math justification), `impl.py` (no comments revealing the label),
  `label.json` {"case_id", "pair_id", "category", "is_defect", "violated_rules", "evidence", "source"}.
- per pair: `check_*.py` (independent checker written from the spec, returns violated rule IDs on fixed
  synthetic data with exact expected values) and `test_pair*.py` (valid -> no violations; faulty -> the
  exact expected violations; labels agree with checker; no real data in benchmark/).
- The full suite must be green: "faulty is detected" is a passing test.
Detailed task briefs live in `docs/claude_code_task_pair*.md`.
Also keep false-alarm traps in mind (valid behaviour that looks suspicious): delinquency can jump
from 00 to 03 in one month; loans can prepay while 90+; zero-balance date can be one month after period.

## How to work
- Explain in simple English first, then the math/code. Comment important code sections.
- Separate clearly: completed, proposed, unverified.
- Never invent data, results, or citations. Do not report metrics you did not actually run.
- Do not treat poor accuracy, odd coefficients, or distribution shift as implementation defects.
- Small steps; run tests (`uv run pytest`) before saying something works; suggest a git commit after each step.

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
- The project is not a package (`[tool.uv] package = false`); `uv sync` installs only the dependencies.
- nbstripout strips notebook outputs on commit; run `uv run nbstripout --install` once per clone. It does not clean
  the file on disk: the notebook shows real loan rows, so Clear All Outputs before sharing the .ipynb file itself.

## Folder layout
- `notebooks/baseline_logistic_regression.ipynb`: the baseline (the only implementation).
- `reference/`: official baseline results (aggregates only, committed; see `reference/README.md`).
  `reference/archive/v1_lbfgs_tol_1e-4/` keeps the superseded first results (never edit either).
- `benchmark/`: the defect/valid pairs and their runners (frozen answer keys).
- `tests/test_baseline_no_leakage.py`: leakage test for the notebook's `build_snapshot()`.
- `reports/solver_tolerance/`: the reproducible solver-tolerance study behind the v2 reference (script + report).
- `docs/`: the two-page pilot plan for Dr. Ku (.docx and .pdf).
- `data/raw/` (Freddie Mac files), `sflld_extracted/` (unzipped data), `outputs/` (optional notebook output): gitignored.

## Done so far (verified)
- Data checked: 50,000 loans (31 origination fields), 3,467,166 monthly records (35 fields),
  periods 2015-01..2026-03, IDs match, no duplicate loan-months.
- Clean baseline: `notebooks/baseline_logistic_regression.ipynb` is the ONLY implementation of the baseline,
  explained step by step. Run it in VS Code with the `.venv` kernel: Run All, about 1 minute. Headless:
  `uv run jupyter nbconvert --to notebook --execute notebooks/baseline_logistic_regression.ipynb --output-dir <a folder outside the repo>`.
  Section 12 must print `MATCHES THE REFERENCE MODEL` (every result compared with `reference/`, tolerance 1e-9).
  Expected: train 101,843 rows / 462 events; test 15,716 rows / 119 events; lbfgs 810 iterations;
  test ROC-AUC 0.893 [0.848, 0.928], PR-AUC 0.301, Brier 0.00622.
  Do not change the model casually: an official change is a separate, reviewed step (notebook section 13 reminder).
- Official results (v2, since 2026-10-25): `reference/` (metrics.json, label_counts.csv, coefficients.csv,
  calibration.csv), produced by the notebook with `TOL = 1e-8` and `SAVE_OUTPUTS = True` (a second run was
  byte-identical). Never edit `reference/` by hand.
- Solver tolerance (the only change from v1): v1 used scikit-learn's default lbfgs `tol = 1e-4`, which stopped after
  40 iterations with no warning but before the optimum (penalized objective 2030.98 vs 2023.11). v1 was made by the
  original script `src/baseline_logreg.py` at commit 9253e97 (recoverable with `git show 9253e97:src/baseline_logreg.py`;
  test ROC-AUC 0.894, PR-AUC 0.307) and is archived in `reference/archive/v1_lbfgs_tol_1e-4/`. `TOL = 1e-8` matches
  scikit-learn's newton-cholesky solver (objective within 3e-7, probabilities within 1e-4). The correction does not
  materially change overall discrimination or calibration, but rare category levels are sensitive: predictions moved
  by up to 0.12 (test) / 0.24 (stress), and rare-level one-hot coefficients (0–2 training events) are penalty-driven,
  so do not interpret them. Only differences between levels of one feature mean anything (levels sum to 0 at the
  optimum). Study and checks: `uv run python reports/solver_tolerance/run_sensitivity.py` (reproduces v2 and v1
  within 1e-9).
- `SAVE_OUTPUTS = True` in the notebook's SETTINGS cell writes the script's five result files to `outputs/`
  (default False); with default settings the four CSV/JSON files are byte-identical to `reference/`.

## Outcome specification (draft, pending professor approval)
- Prediction date t: end of each December 2015..2024 (loan-snapshot design).
- At risk at t: record at t, no zero-balance code at or before t, delinquency status at t in {00,01,02}.
- Features: origination fields + performance records with period <= t ONLY.
- Event (y=1): any month t+1..t+12 with numeric status >= 03 or status 'RA'.
  This is "12-month 90+ DPD serious delinquency", NOT a regulatory default definition.
- Non-event (y=0): no event and loan still observed at t+12 (implemented as: a record at t+12 or later).
  Known limitation, kept until Dr. Ku decides (do not change it without approval): the rule does not require every
  window month to be observed. Exactly 1 of 237,720 non-event rows has a gap (Dec 2015 snapshot, training: its t+12
  record is missing, status 00 before and after). Requiring all 12 months would censor it (train 101,842 rows,
  events unchanged; test/stress unchanged). Pair 3's spec (rule O5) uses the same rule in both cases.
- Censored (excluded, counted): prepay (zero-balance code 01), other termination (02/03/09/15/16/96)
  before t+12 without a 90+ month, or window past 2026-03.
- Split: train Dec 2015–2018, test Dec 2021–2024, 2019–2020 = stress test only (COVID forbearance).
  Loan-disjoint: SHA-256(loan_id) % 100 < 70 -> train pool. No loan in both train and test.
- Documented sentinels -> NaN: FICO 9999; LTV/CLTV/DTI/MI 999; units/borrowers 99.
  DTI 999 = all HARP loans (4,017 of 4,020).

## Benchmark status and plan
Folder: `benchmark/`. One subfolder per pair. Each pair's README.md and spec.md are the detailed record
(the original task briefs were removed from the tree; they are in git history before this commit).
The benchmark's references to `src/baseline_logreg.py` (specs, labels) mean the original script at commit 9253e97
(pair 4's spec cites the earlier 9c743e2, which differs only by the removed tautological assertion); they are
historical provenance, so do not edit them.
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
FIXED (baseline leakage check, commit 9253e97): the tautological assertion `max_feature_period <= t` (computed from
hist, which is already filtered to months <= t, so it could never fail) was removed from build_snapshot(). The leakage
check lives in `tests/test_baseline_no_leakage.py`. It loads `ym_index()` and `build_snapshot()` from the notebook's
code cells tagged `leakage-test-source` (read with nbformat, executed in a fresh namespace holding only np, pd and
HORIZON = 12), then runs future invariance (truncate or perturb all records after t) plus a mutation power check (the
`h12 = hist[...]` line rewritten to read `perf`; must be detected). Keep those two cells tagged and self-contained.
Run: `uv run pytest tests -v` (4 tests).

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
- Small steps; run tests (`uv run pytest tests -v`, plus the benchmark runners) before saying something works;
  suggest a git commit after each step.

# TASK: Build an explained, editable Jupyter notebook version of the baseline model

> **How to use:** save this file as `docs/claude_code_task_notebook.md`, start a NEW Claude Code conversation, and type:
> `Read CLAUDE.md and docs/claude_code_task_notebook.md, then carry out the task phase by phase.`

## ROLE

Act as a patient Python and machine-learning teacher, and a careful research software engineer, working on **MRM-Bench** (Master's research, UNT, supervised by Dr. Chih-Hao Ku). The reader is **Dushyant**, a Master's student who knows Python, pandas and scikit-learn and wants to understand **every step** of the baseline well enough to change it himself.

## GOAL

Create **one notebook**, `notebooks/01_baseline_logreg_walkthrough.ipynb`. It reproduces `src/baseline_logreg.py` step by step, explains every step in simple English, shows what the data looks like at each stage, and ends with a check proving its results are **identical** to the script's.

The script remains the official reference. The notebook is for understanding and experimenting. If Dushyant edits the notebook, the final check must tell him clearly what changed compared with the reference.

---

## PHASE 0 — READ FIRST (no code yet)

1. Read `CLAUDE.md` completely, especially "Outcome specification".
2. Read `src/baseline_logreg.py` completely. Note every function, constant and output.
3. Read `tests/test_baseline_no_leakage.py`.
4. Look at `outputs/metrics.json`, `outputs/label_counts.csv` and `outputs/coefficients.csv`. These are the reference results.
5. Run `git status`.

**Protected (do not modify):** `src/baseline_logreg.py`, everything in `benchmark/`, `tests/`, `data/`, and the existing files in `outputs/`.

---

## PHASE 1 — ENVIRONMENT

- Add notebook support with uv only: `uv add --dev ipykernel nbstripout`. Never use pip.
- Configure nbstripout for this repository (`uv run nbstripout --install`), so that **cell outputs are never committed to git**. Verify by checking `.git/config` or `.gitattributes`. This matters because the notebook will display real Freddie Mac rows, and the license does not allow redistributing them.
- The notebook must run in VS Code using the project's `.venv` kernel ("finance-ai-system"). Write the kernel-selection steps in the notebook's first cell.

---

## PHASE 2 — NOTEBOOK CONTENT

### Writing rules (apply to every section)

- **Before every code cell**, a short markdown cell with three parts:
  - **What this cell does**, in one or two plain sentences.
  - **Why it matters**: the reason in research or credit-risk terms.
  - **What to look for**: what the output should show, and which numbers to check.
- Code cells are short (roughly 5–25 lines) and do **one** thing. Give each important line a plain-English comment.
- **Copy the logic from `src/baseline_logreg.py`; do not reinvent it.** Keep the same function names, constants, column names, seed and feature lists, so the two can be compared. You may split a long function across cells, as long as the computation is identical.
- After each transformation, show its effect: shapes, counts, `value_counts()`, `describe()`, or a small sample. Real loan rows may be displayed (outputs are stripped before commit), but keep samples small (5 rows).
- Mathematical formulas go in markdown with LaTeX (`$$ ... $$`).
- No new features, models or outcome rules. **No XGBoost or hazard model.**

### Required sections, in this order

**0. Title and how to use this notebook**
- What the notebook does; the run order (top to bottom); how to select the `.venv` kernel in VS Code; expected run time (about 3–4 minutes, mostly the bootstrap); a warning that outputs are stripped on commit.

**1. Setup and configuration**
- Imports and printed package versions (Python, numpy, pandas, scikit-learn).
- **One clearly marked "SETTINGS — change things here" cell** holding every setting from the script: data path (`data/raw/sample_2015.zip`), `SEED`, `TRAIN_SNAPSHOTS`, `TEST_SNAPSHOTS`, `STRESS_SNAPSHOTS`, `HORIZON`, `TRAIN_LOAN_SHARE`, `C`, and a new `N_BOOT = 1000`. `N_BOOT` controls the bootstrap so a quick run can use 100, but the reference comparison needs 1000. Next to each setting, explain in a comment what changing it would do.

**2. Loading the raw files**
- The 31 and 35 column names from the July 2026 file layout, and why the files have no header row.
- Reading the files as text first, so the missing-value codes are not lost.
- The structural checks (unique loan IDs, matching IDs, no duplicate loan-months), with the counts printed: 50,000 loans; 3,467,166 records; months 201501–202603.

**3. Preprocessing the origination data**
- A **table of the documented "Not Available" codes** (field, code, meaning, source = General User Guide July 2026), with the **count of each code before and after** conversion to missing. DTI 999 should show 4,020, and you should show that 4,017 of those are HARP loans.
- Each derived feature (`dti_missing`, `log_orig_upb`, `term_360`, `multi_borrower`, `mi_flag`) explained in one sentence, with a small before/after view.
- A short note on the categorical "9"/"99" codes: they are kept as their own category, and only 8 loans have one. See the optional exercises in section 13.

**4. Preprocessing the monthly performance data**
- The month index `t` (why YYYYMM numbers can't be subtracted directly: 201601 − 201512 = 89, not 1).
- The delinquency status codes (00, 01, 02, 03+, RA) and the derived `dq` and `sev` columns, with the distribution of `delinq_status`.

**5. A toy example to understand labels and features (synthetic data)**
- Build a tiny synthetic panel of 4–5 made-up loans. Use IDs like `X01`, never the Freddie Mac pattern. Include:
  - a loan that stays current;
  - one that becomes 90+ days late within 12 months;
  - one that prepays early;
  - one already 90+ at t;
  - one that is late only after t+12.
- Run the script's `build_snapshot()` on it and show the result.
- Then walk through each loan in markdown: why it is at risk or excluded, what each feature value means, and why its label is event, non_event or censored.
- This section is the most important for understanding. Take care over it.

**6. Building the real snapshots**
- Run `build_snapshot()` for every December snapshot, merge the origination features, and show the label-count table.
- Compare it with `outputs/label_counts.csv`; it must match exactly.
- Explain the December 2019 spike (1,199 events, COVID forbearance) and why 2019–2020 are used only as a stress test.

**7. The train/test split**
- The hash-based loan split, in simple terms: why it is reproducible without a random seed.
- Why train is 2015–2018 and test 2021–2024.
- Print the rows and events per set, and assert zero loan overlap. Expected: train 101,843 rows and 462 events; test 15,716 rows and 119 events.
- Why censored rows are dropped before fitting, with a link to the outcome rules in CLAUDE.md.

**8. The logistic regression model**
- The math in LaTeX: z = b0 + x′b, p = 1/(1+e^(−z)), and the L2-penalized log-loss. Explain each symbol in plain words.
- The preprocessing pipeline (median imputation, scaling, one-hot encoding) and **why it is fit on training data only**.
- Fit the model.

**9. Evaluation**
- ROC-AUC, PR-AUC and Brier score, each explained in one or two sentences with an everyday interpretation.
- The loan-clustered bootstrap: why it resamples loans, not rows.
- Print train, test and stress results. Expected test results: ROC-AUC 0.894 [0.849, 0.930], PR-AUC 0.307, Brier 0.00620.
- The calibration table and plot. Explain why the model under-predicts on test (0.43% vs 0.76%) and why that is a population shift, not a code defect.

**10. Coefficients**
- Show the sorted coefficient table.
- Explain how to read standardized coefficients.
- Explain the one-hot caveat: compare category levels with each other, not with zero.

**11. Leakage check**
- Explain future invariance in plain words: if you delete or change the future, the features must not change.
- Run the check on the section 5 toy panel inside the notebook: truncate the months after t, rerun `build_snapshot()`, and show the feature columns are identical.
- Mention that `tests/test_baseline_no_leakage.py` does this formally, including the mutation power check, and show the command to run it.

**12. Does the notebook match the reference?**
- Compare the notebook's results with `outputs/metrics.json`, `outputs/label_counts.csv` and `outputs/coefficients.csv`:
  - label counts and set sizes must be equal;
  - metrics must match to within 1e-9;
  - coefficients must match to within 1e-9.
- Print a clear verdict:
  - `MATCHES THE REFERENCE MODEL`, or
  - `DIFFERS FROM THE REFERENCE` followed by a table of what differs (old vs new).
- Report differences rather than crashing, because Dushyant will deliberately change things. Mention that with `N_BOOT` other than 1000, only the confidence intervals will differ.

**13. Where to make changes, and optional exercises**
- A table of safe edit points: what to change → which cell → which sections to rerun → which reference numbers will change. Cover:
  - snapshots;
  - horizon;
  - the C value;
  - adding or removing a feature;
  - the censoring policy (prepayment as y = 0, as a sensitivity run).
- Optional exercises, **not applied**, each with a hint:
  1. Map categorical "9"/"99" codes to an explicit "unknown" category.
  2. Turn current balance into a feature (current UPB ÷ original UPB), using only month-t data.
  3. Use one-hot encoding with one reference level dropped, for readable coefficients.
  4. Count prepayment as y = 0 and compare event rates and AUC (one of the decisions waiting for Dr. Ku).
- A reminder: if a change should become official, update `src/baseline_logreg.py` and its tests in a separate, reviewed step. The notebook is not the reference.

---

## PHASE 3 — RUN AND VERIFY

1. Execute the whole notebook top to bottom with the project environment:
   `uv run jupyter nbconvert --to notebook --execute notebooks/01_baseline_logreg_walkthrough.ipynb --output /tmp/nb_check.ipynb`
   (or the equivalent). Add `nbconvert` or `jupyter` as a dev dependency only if needed. Write the executed copy **outside** the repository.
2. Section 12 must print `MATCHES THE REFERENCE MODEL`. If it doesn't, fix the notebook, never the reference numbers.
3. Run `uv run pytest tests -v`: still 4 passed.
4. Confirm that no protected file changed (`git diff --stat`), that `outputs/` timestamps are unchanged, and that the notebook in the repo has **no stored outputs** (nbstripout works).
5. Confirm that no Freddie Mac loan ID appears in the committed notebook file: search for the regex `F1[0-9]Q[1-4][0-9]{7}`.

---

## PHASE 4 — REPORT BACK

A. The notebook's section list, with one line per section.
B. Section 12's verdict, copied from the actual run.
C. The key printed numbers (label counts, train/test sizes, test metrics), confirming they match the reference.
D. Confirmation that outputs are stripped, protected files are unchanged, tests pass, and no loan IDs are committed.
E. Anything you could not reproduce exactly, and why.
F. A suggested commit message. Do not commit unless I ask.

---

## ACCEPTANCE CRITERIA

- One notebook, sections 0–13 in order. Every code cell has a "what / why / what to look for" markdown cell before it.
- The logic is copied from the script with the same names and settings, and section 12 prints `MATCHES THE REFERENCE MODEL` on the default settings.
- The toy example (section 5) explains labels and features loan by loan.
- A single settings cell; safe edit points and optional exercises are documented but not applied.
- nbstripout is active; no outputs or loan IDs are committed; protected files are unchanged; the tests still pass.

**Scope:** only this notebook. Do not change the model, the outcome rules, the benchmark or the tests. Do not start VALID-Agent, XGBoost or a hazard model.

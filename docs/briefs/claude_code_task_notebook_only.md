# TASK: Make the notebook the only baseline implementation and clean up the repository

> **How to use:** save this file as `docs/claude_code_task_notebook_only.md`, start a NEW Claude Code conversation, and type:
> `Read CLAUDE.md and docs/claude_code_task_notebook_only.md, then carry out the task phase by phase.`

## ROLE

Act as a careful research software engineer on **MRM-Bench** (Master's research, UNT, supervised by Dr. Chih-Hao Ku).

## GOAL (the user's decision)

From now on, **`notebooks/01_baseline_logreg_walkthrough.ipynb` is the only implementation of the baseline model.** `src/baseline_logreg.py` and the files no longer needed are to be deleted.

Nothing that works today may break:
- the leakage test;
- the notebook's reference check;
- the four benchmark runners;
- `uv sync`.

Everything stays recoverable through git history (the last script version is commit `9253e97`).

---

## PHASE 0 — SAFETY FIRST

1. Read `CLAUDE.md`, the notebook, `src/baseline_logreg.py`, `tests/test_baseline_no_leakage.py`, `pyproject.toml` and `.gitignore`.
2. Run `git status`. If the notebook from the previous task is not committed yet, **stop and ask me** whether to commit it first. Suggested message: "Add explained walkthrough notebook for the baseline logistic regression". Do not delete anything while uncommitted work exists.
3. List every file in the repository (excluding `.venv/` and `data/`), and **show me the deletion plan from Phase 3 before deleting**. Mark anything you are unsure about. Continue only when the plan matches this brief; ask if anything does not.

**Never delete or modify:**
- `data/` (the Freddie Mac files);
- `benchmark/` (frozen answer keys; do not edit even path references inside it);
- `.venv/`, `.git/`, `.python-version`, `uv.lock` (except for changes uv makes itself), `.gitignore` (except additions).

---

## PHASE 1 — SAVE THE REFERENCE RESULTS PERMANENTLY

`outputs/` is gitignored, so today the reference numbers exist only on this laptop.

1. Create `reference/` and copy into it, from the current `outputs/`:
   - `metrics.json`
   - `label_counts.csv`
   - `coefficients.csv`
   - `calibration.csv`

   These contain **aggregate numbers only**. Confirm that no file has a loan ID (regex `F1[0-9]Q[1-4][0-9]{7}`) or a loan-level row.
2. Add `reference/README.md` explaining:
   - these are the official baseline results;
   - they were produced by `src/baseline_logreg.py` at commit `9253e97`;
   - they were reproduced on two machines;
   - the notebook's section 12 checks against them.
3. Make sure `reference/` is **not** gitignored.

---

## PHASE 2 — MAKE THE NOTEBOOK SELF-SUFFICIENT

1. **Section 12:** compare against `reference/` instead of `outputs/`. It must still print `MATCHES THE REFERENCE MODEL` on default settings.
2. **Optional saving:** add `SAVE_OUTPUTS = False` to the SETTINGS cell. When it is True, the notebook writes the same five output files the script used to write into `outputs/` (gitignored). Default False.
3. **Remove the dependency on the script:**
   - search the notebook for any import or mention of `src/baseline_logreg.py`;
   - code must not import it;
   - text that mentions it should say it was the original script, now in git history at commit `9253e97`.
4. **Help the leakage test find the functions:**
   - the code cells that define `ym_index` and `build_snapshot` must be self-contained: they define everything they use, apart from imports listed in the cell or in the setup cell;
   - add the cell metadata tag `leakage-test-source` to those cells, so the test can find them reliably.
5. **Section 11:** also add the perturbation check (change every status after t to "05"; the features must not change), so the notebook alone demonstrates both invariance checks.

### Rewrite `tests/test_baseline_no_leakage.py` to test the notebook

- Load the notebook with `nbformat`. Add it as a dev dependency with `uv add --dev nbformat` if it is missing.
- Collect the cells tagged `leakage-test-source` and `exec` them in a fresh namespace with the needed imports, to obtain `build_snapshot` and `ym_index`. **If the tagged cells are missing, fail with a clear message.**
- Keep all four tests, with the same synthetic panel and logic:
  - T1 truncation;
  - T2 perturbation;
  - T3 the mutation power check (rewrite the `h12` line to use `perf` instead of `hist`; the test must detect it). **If the line to mutate is not found, the test fails loudly**;
  - T4 the hand-checked spot values.
- Docstrings say the function under test now lives in the notebook.

---

## PHASE 3 — DELETE (only after Phases 1–2 pass)

Use `git rm` for tracked files, so the history keeps them.

| Delete | Why |
|---|---|
| `src/baseline_logreg.py` | Replaced by the notebook (kept in git history) |
| `src/finance_ai_system/` and then the empty `src/` | uv's unused package stub |
| `docs/chatgpt_review_pair2.md` | A one-off review request, not project documentation |
| `docs/pilot_prompt.md` | Superseded by `CLAUDE.md` and refers to the deleted script |
| `outputs/` | Old script outputs; the official copies are now in `reference/` |
| `sflld_extracted/` | Temporary unzip of the data; recreated automatically from the zip |
| all `__pycache__/` folders, `.pytest_cache/` | Caches, recreated automatically |

**Keep:**
- `notebooks/`, `reference/`, `benchmark/`, `tests/`, `data/`;
- `CLAUDE.md`, `README.md` (rewritten below), `pyproject.toml`, `uv.lock`, `.gitignore`, `.python-version`;
- the task briefs `docs/claude_code_task_*.md` (the record of how each benchmark pair was built). Move them into `docs/briefs/` with `git mv`.

**Package fix (required when `src/` is deleted):** in `pyproject.toml`, stop building the project as a package. Add `[tool.uv]` with `package = false`, and remove the `[build-system]` table and any `[project.scripts]` entry that points to `finance_ai_system`. Then run `uv sync` and confirm it succeeds, and that `uv run python -c "import numpy, pandas, sklearn"` works. Show the `pyproject.toml` diff.

Also add `outputs/` to `.gitignore` if it isn't there (the notebook can recreate it). Keep `sflld_extracted/` and `data/` ignored.

---

## PHASE 4 — UPDATE THE DOCUMENTATION

1. **`CLAUDE.md`:**
   - the baseline is now the notebook;
   - how to run it: VS Code with the `.venv` kernel, Run All, about 1 minute;
   - official results are in `reference/`;
   - the original script is in git history at `9253e97`;
   - the leakage test now loads its functions from the notebook's tagged cells;
   - update "Done so far" with the notebook;
   - update the folder layout;
   - remove instructions that run `src/baseline_logreg.py`.
2. **`README.md`** (replace uv's empty one): 10–20 lines covering what MRM-Bench is, the folder layout, how to set up (`uv sync`, `uv run nbstripout --install`), how to run the notebook, tests (`uv run pytest tests -v`) and runners (`uv run python benchmark/run_pairN.py`), and a note that the Freddie Mac data is not included and must not be committed.
3. **Do not edit `benchmark/`**, even where its specs mention `src/baseline_logreg.py`. Those references are historical provenance. Add one line to `CLAUDE.md` saying that the benchmark's references to `src/baseline_logreg.py` mean commit `9253e97`.

---

## PHASE 5 — VERIFY EVERYTHING

1. Run `uv sync`; it succeeds.
2. Run the notebook top to bottom headlessly, writing the executed copy outside the repo. Section 12 prints `MATCHES THE REFERENCE MODEL`.
3. Run `uv run pytest tests -v`: 4 passed, including T3 (the mutation is detected).
4. Run `uv run python benchmark/run_pair1.py` through `run_pair4.py`; all exit 0.
5. In `git status` and `git diff --stat`: only the intended deletions, moves and edits. Nothing under `benchmark/` or `data/` changed.
6. The committed notebook has no stored outputs, and no file to be committed contains a loan ID.
7. Run `git show 9253e97:src/baseline_logreg.py | head`, to show the old script is still recoverable.

---

## PHASE 6 — REPORT BACK

A. The final folder tree (excluding `.venv`, `data`, `.git`).
B. A table of every file deleted or moved, with the reason.
C. The `pyproject.toml` diff.
D. Actual outputs: the section 12 verdict, `pytest`, the four runner verdicts, and `uv sync`.
E. Anything that could not be done as briefed, and why.
F. A suggested commit message. **Do not commit unless I ask.**

---

## ACCEPTANCE CRITERIA
- The notebook is the only baseline implementation; it imports no script; section 12 matches `reference/`.
- The reference results are committed in `reference/`, aggregates only.
- The leakage test loads its functions from the notebook's tagged cells; 4 tests pass, including the mutation check.
- The listed files are deleted with `git rm`; the briefs are moved to `docs/briefs/`; `benchmark/` and `data/` are untouched.
- `uv sync` works with `package = false`; all four runners exit 0.
- `CLAUDE.md` and `README.md` describe the new layout; the old script is recoverable from `9253e97`.

**Scope:** only this restructuring. Do not change the model, the outcome rules, the benchmark or the reference numbers. Do not start VALID-Agent, XGBoost or a hazard model.

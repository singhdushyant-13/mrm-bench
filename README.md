# MRM-Bench (pilot)

A benchmark of credit-risk model code for testing AI model validators: a clean logistic-regression baseline, plus
deliberately faulty and similar-looking valid variants, each with an independently defensible answer key
(specification, math and an executable test). Master's research project at the University of North Texas.

**Layout.** `notebooks/baseline_logistic_regression.ipynb` is the baseline (12-month 90+ DPD logistic regression,
explained step by step). `reference/` holds its official results (aggregates only). `benchmark/pairN_*/` holds the
faulty/valid pairs, and `benchmark/run_pairN.py` scores them. `tests/` holds the leakage test for the notebook's
`build_snapshot()`. `CLAUDE.md` has the project rules and specification.

**Setup.** `uv sync`, then `uv run nbstripout --install` once per clone (strips notebook outputs on commit).

**Run.** Open the notebook in VS Code, select the `.venv` kernel and Run All (about 1 minute); section 12 must print
`MATCHES THE REFERENCE MODEL`. Tests: `uv run pytest tests -v`. Benchmark: `uv run python benchmark/run_pair1.py`
(likewise `run_pair2.py` to `run_pair4.py`); exit code 0 means every case agrees with its label.

**Data.** The Freddie Mac Single-Family Loan-Level Dataset (2015 sample, `data/raw/sample_2015.zip`) is not included.
Its terms do not allow redistribution, so loan-level records must never be committed. Download it yourself from the
official source: <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset> (free registration required).
Select the 2015 origination vintage sample file, plus the current file layout and General User Guide from the same
page, and place them in `data/raw/` as named above.

**Benchmark cases.** Each of the 4 defect pairs (`benchmark/pairN_*/`) has a `faulty/` and `valid/` variant, and each
variant has: `spec.md` (the documented rule, a worked numeric counterexample, and provenance back to the source PDF
and page), `impl.py` (the implementation), `label.json` (the ground-truth answer: `is_defect`, `violated_rule`,
`evidence`, `source`), and `test_case.py` (the executable check). `benchmark/run_pairN.py` runs both variants and
reports whether each one's test outcome matches its label.

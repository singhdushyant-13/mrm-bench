# Reference results of the baseline logistic regression

These four files are the **official results** of the MRM-Bench clean baseline (12-month 90+ DPD serious
delinquency, logistic regression, Freddie Mac SFLLD 2015 sample, Release 47).

| File | Contents |
|---|---|
| `metrics.json` | train / test 2021–2024 / stress 2019–2020: rows, events, event rate, mean prediction, ROC-AUC, PR-AUC, Brier, 95% loan-bootstrap CIs; the run configuration; library versions |
| `label_counts.csv` | number of loans per December snapshot and label (event, non_event, censored_prepay, censored_other_termination), before the train/test split |
| `coefficients.csv` | the 39 fitted coefficients (numeric features standardized) and the intercept |
| `calibration.csv` | test 2021–2024 calibration by decile of predicted risk: rows, mean prediction, observed rate, events |

Headline numbers: train 101,843 rows / 462 events; test 15,716 rows / 119 events; test ROC-AUC 0.894.

**Provenance.** The files were produced by the original script `src/baseline_logreg.py` at commit `9253e97`
(`git show 9253e97:src/baseline_logreg.py`), run as
`uv run python src/baseline_logreg.py --data data/raw/sample_2015.zip`, with Python 3.12.12, numpy 2.5.4,
pandas 3.1.0 and scikit-learn 1.9.1 (also recorded in `metrics.json`). The results were reproduced on two machines.
They are byte-identical copies of the files the script wrote to `outputs/`.

**How they are used.** The baseline is now `notebooks/01_baseline_logreg_walkthrough.ipynb`. Its section 12 compares
every number above with these files and prints `MATCHES THE REFERENCE MODEL` when all of them agree (tolerance
1e-9).

**Aggregates only.** No file contains a loan ID or any loan-level record, so they can be committed. Freddie Mac
loan-level data must never be added here.

**Changing them.** Do not edit these files by hand. If a model change is approved, run the notebook with
`SAVE_OUTPUTS = True`, review the differences that section 12 reports, and copy the new files from `outputs/` into
this folder in a separate, reviewed commit that also updates `CLAUDE.md`.

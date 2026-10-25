# TASK: Implement MRM-Bench Pair 1 — DTI Preprocessing Defect

## ROLE

Act as a senior Python machine learning engineer, quantitative model validation researcher, and academic research software developer.

You are contributing to an ongoing research project:

**MRM-Bench: Assumption-Grounded Executable Verification of Credit Risk Models Using LLM Agents**

The project is supervised by Professor Chih-Hao Ku at the University of North Texas.

Our research investigates whether AI agents can distinguish genuine mathematical and implementation defects from valid modeling choices.

Your task is to create the first benchmark pair: **a deliberately faulty and a mathematically valid implementation of mortgage DTI preprocessing**.

This is research-grade benchmark construction, not a general code-refactoring task.

---

# PHASE 0 — READ THE EXISTING PROJECT FIRST

Before creating or modifying files:

1. Read `CLAUDE.md` completely.
2. Examine the repository structure.
3. Identify the existing Python environment and package manager.
4. Inspect relevant project conventions and existing benchmark files.
5. Locate the Freddie Mac July 2026 General User Guide and relevant schema documentation if present.
6. Check whether `benchmark/` already contains files or conventions that should be preserved.
7. Check `git status` before modifying anything.

Follow existing project instructions unless they conflict with the explicit requirements below.

**Do not begin writing implementation code until this inspection is complete.**

Do not overwrite unrelated user changes.

## Protected files

Do not modify:

- `src/baseline_logreg.py`
- Existing trained models
- Dataset files
- Existing research results
- Unrelated source code

If the requested task requires changing a protected file, explain the blocker instead of modifying it.

---

# PHASE 1 — UNDERSTAND THE MATHEMATICAL RULE

Use the official Freddie Mac Single-Family Loan-Level Dataset General User Guide, Release 47, July 2026.

Relevant origination attribute:

**Original Debt-to-Income (DTI) Ratio**

According to the documentation:

- DTI represents monthly debt expenses divided by monthly income, expressed as a percentage.
- Ordinary reported values are in the range 0–65.
- The value `999` means **Not Available**.
- HARP loans have DTI disclosed as `999`.

Primary source (local copy, already in the repository):

`data/raw/general_user_guide_july_2026.pdf`, section "Glossary → Origination Data File", attribute "Original Debt-to-Income (DTI) Ratio" (page 5 of the PDF).

The guide states: "0% - 65%, 999 = Not Available"; "HARP: 999 = Not Available"; "Ratios > 65% will be disclosed as Not Available. All HARP loans in the dataset will be disclosed as Not Available."

Open the PDF and confirm this text yourself before citing it. If you cannot read it, say so and rely on the rule as stated here. Do not invent quotations, URLs, or page numbers.

Real-data context (already verified by the project, aggregate only): 4,020 of 50,000 loans in the 2015 sample have DTI = 999; 4,017 of them are HARP loans.

## Mathematical interpretation

For this benchmark, define:

- `0` through `65`: valid reported DTI percentages.
- `999`: missing information, not a valid debt-to-income measurement.
- Other nonmissing values outside the documented valid range: invalid input that should raise a clear error.

Treat `999` as missing **before computing descriptive statistics, fitting transformations, or imputing values**.

Incorrectly using `999` as a numeric observation can distort:

- The mean DTI.
- Standard deviations and scaling parameters.
- Median calculations.
- The fitted Logistic Regression coefficients.

Do not assert that every fitted coefficient must change in the same direction. Explain that the numerical distortion depends on the data and preprocessing pipeline.

---

# PHASE 2 — DESIGN THE BENCHMARK PAIR

Create the following directory structure:

```text
benchmark/
├── pair1_preprocessing/
│   ├── README.md
│   ├── faulty/
│   │   ├── spec.md
│   │   ├── impl.py
│   │   ├── label.json
│   │   └── test_case.py
│   └── valid/
│       ├── spec.md
│       ├── impl.py
│       ├── label.json
│       └── test_case.py
└── run_pair1.py
```

You may add a small shared testing helper only if necessary.

Do not add unnecessary modules, frameworks, generated datasets, or abstractions.

All benchmark examples must use **synthetic data only**.

Never copy real Freddie Mac loan-level records into `benchmark/`.

---

# PHASE 3 — IMPLEMENT THE TWO VERSIONS

## A. Required common interface

Both implementations must expose the same public function:

```python
prepare_dti(df, training_median=None)
```

The function must:

- Accept a Pandas DataFrame containing an `orig_dti` column.
- Return a new DataFrame without modifying the input.
- Produce a numeric `orig_dti` column after preprocessing.
- Produce a `dti_missing` indicator column.
- Preserve row ordering and the original index.
- Avoid changing unrelated columns.
- Support training and inference using the same public interface.

### Training and inference behavior

When `training_median=None`:

- Treat the call as a training-data preprocessing operation.
- Calculate the median according to the implementation's behavior.
- Record the median used in a reproducible, testable way.

When `training_median` is provided:

- Treat the call as validation or inference preprocessing.
- Use the supplied training median.
- Do not recalculate the median from validation or test data.

Because the public function returns a DataFrame, store the fitted median in:

```python
result.attrs["dti_training_median"]
```

Use this design consistently in both implementations.

Document this interface and its limitations.

## B. Valid implementation

File:

`benchmark/pair1_preprocessing/valid/impl.py`

Implement the correct behavior:

1. Confirm that `orig_dti` exists.
2. Work on a copy of the DataFrame.
3. Convert the `orig_dti` values to numeric safely.
4. Reject unrecognized or invalid nonmissing values rather than silently accepting them.
5. Identify `999` as the documented missing-value sentinel.
6. Create `dti_missing`:
   - `1` when original DTI is `999` or genuinely missing.
   - `0` for valid DTI measurements.
7. Replace `999` with `NaN`.
8. Calculate the median using valid, nonmissing **training DTI values only**.
9. If a training median is supplied, use it rather than fitting a new median.
10. Impute missing DTI entries using the training median.
11. Return a new DataFrame containing the processed DTI and missingness indicator.
12. Store the median used in DataFrame attributes.

Handle edge cases:

- Missing `orig_dti` column.
- All DTI values missing.
- Empty DataFrame.
- Existing null values.
- Invalid values outside the documented range.
- A supplied training median that is missing, invalid, or outside the permitted DTI range.

Choose a clear documented policy for empty or entirely missing training data. Prefer raising a meaningful `ValueError` over silently using an arbitrary median.

## C. Faulty implementation

File:

`benchmark/pair1_preprocessing/faulty/impl.py`

Create a deliberately defective implementation that looks very similar to the valid implementation.

The **single intended defect** is:

**The implementation fails to recognize `999` as the documented missing-value sentinel and treats it as a real DTI observation.**

Important constraints:

- Preserve the same function name and parameters.
- Preserve the same return type and output column names.
- Preserve the same training-versus-inference interface.
- Keep the surrounding code structure as similar as possible.
- Do not deliberately introduce unrelated defects.
- Do not introduce future-information leakage or a separate train/test median leak.
- Do not make the faulty implementation fail simply because of an artificial exception.
- The faulty implementation should execute normally for the synthetic test data and produce incorrect values.
- In its faulty behavior, `999` should remain numeric and contaminate the preprocessing result.

The defect should be realistic and detectable through mathematical testing, not an obvious syntax error.

**Input validation must be identical in both implementations.** Both must accept exactly the same inputs: values in [0, 65], the documented code 999, or genuine NaN. Both must raise the same `ValueError` for anything else (e.g., 70, -5, 500). The ONLY difference between the two files is that the faulty version never converts 999 to missing. It still treats 999 as a legal input (it is documented), but then uses it as a number. This keeps the pair a single-defect pair; if the validation differed, the pair would contain two differences, and the label would be harder to defend.

---

# PHASE 4 — WRITE THE MATHEMATICAL SPECIFICATIONS

Create:

- `faulty/spec.md`
- `valid/spec.md`

Both files should describe the **same intended correct behavior**.

Do not write one specification that permits `999` as a real DTI observation.

Include:

### 1. Official documented rule

Explain that Freddie Mac reports DTI as a percentage and uses `999` for unavailable information, including HARP loans.

### 2. Correct mathematical preprocessing

For an input DTI value x:

- Missing indicator = 1 when x is missing or 999.
- Otherwise, missing indicator = 0.

Define the training median as the median of valid observed DTI measurements, excluding missing values and sentinel codes.

Define the imputed value as:

- The original DTI when it is valid.
- The training median when DTI is missing.

### 3. Why the faulty implementation violates the rule

Explain mathematically why treating 999 as an actual DTI value is incorrect.

Use a small numerical illustration.

### 4. Applicability conditions

Identify the assumptions under which this specification is valid:

- Freddie Mac Release 47 `orig_dti` field.
- DTI values stored as percentages, not proportions.
- `999` designated as unavailable.
- The training-data median fitted before application to validation/test data.

### 5. Evidence and provenance

Reference the July 2026 official General User Guide.

Specify the source document title, release, field name, and relevant section or page.

Do not invent quotation text or page references.

The specifications should be understandable to an independent reviewer.

---

# PHASE 5 — CREATE THE GROUND-TRUTH LABELS

Create exactly one `label.json` for each implementation.

Use this structure:

```json
{
  "case_id": "P1_F",
  "category": "preprocessing",
  "is_defect": true,
  "violated_rule": "Freddie Mac DTI 999 must be treated as missing, not numeric.",
  "evidence": "Synthetic counterexample demonstrates that 999 contaminates the processed DTI values."
}
```

For the valid case:

```json
{
  "case_id": "P1_V",
  "category": "preprocessing",
  "is_defect": false,
  "violated_rule": null,
  "evidence": "Synthetic tests demonstrate correct 999 handling and use of the training-only median."
}
```

Also add `"pair_id": "P1"` and `"source": "Freddie Mac SFLLD General User Guide, Release 47 (July 2026), Original DTI Ratio"` to both files.

Use these exact case IDs:

- `P1_F`: Faulty case.
- `P1_V`: Valid case.

The label is ground truth assigned by the benchmark designers.

The tests must independently justify the label.

Do not infer the label from whether pytest passes or fails.

Validate that both JSON files are syntactically valid.

---

# PHASE 6 — CREATE SYNTHETIC TEST DATA

Use a small DataFrame of six fictional DTI observations:

```python
orig_dti = [10, 20, 30, 40, 999, 999]
```

The correct interpretation is:

Valid observed values:

```text
10, 20, 30, 40
```

Training median:

```text
25
```

Expected `dti_missing` flags:

```text
0, 0, 0, 0, 1, 1
```

Expected processed DTI values:

```text
10, 20, 30, 40, 25, 25
```

These numbers are synthetic and must not be described as real Freddie Mac observations.

## Additional inference test

Create a small synthetic validation DataFrame:

```python
orig_dti = [999, 12, 65]
```

When supplied with `training_median=25`, the expected valid outputs are:

```text
25, 12, 65
```

The value 65 is the legal upper boundary and must be kept unchanged. It is a deliberate false-alarm trap: a validator (or implementation) that treats 65 as suspicious is wrong.

The expected missing indicators are:

```text
1, 0, 0
```

The function must not calculate a new median from validation data.

This test isolates train-versus-inference behavior.

---

# PHASE 7 — WRITE PYTEST TEST CASES

Create:

- `faulty/test_case.py`
- `valid/test_case.py`

Each file tests its own implementation, but **the two test files must be identical except for the single line that imports the implementation**. Both cases are judged against the same specification, so they must face the same tests. Show me a `diff` of the two test files at the end; it must differ only in the import line. Do not write weaker or different tests for the faulty case.

Required tests:

1. Correct synthetic six-row DTI outputs.
2. Correct training median of 25.
3. Correct missing indicators.
4. No processed DTI values above 65.
5. Correct use of a supplied training median during inference.
6. Input DataFrame remains unchanged.
7. Row count and index are preserved.
8. Appropriate behavior when the required input column is missing.
9. Invalid inputs (e.g., 70, -5, 500) raise `ValueError` in both versions.
10. (In the runner or a separate `benchmark/test_no_real_data.py`): no file under `benchmark/` contains a Freddie Mac loan-ID pattern (regex `F1[0-9]Q[1-4][0-9]{7}`), and no file is larger than 200 KB.

Add edge-case tests where they meaningfully improve the benchmark without introducing additional defect categories.

## Expected outcome

The valid implementation must pass its tests.

The faulty implementation must fail at least one meaningful test that demonstrates the intended preprocessing defect.

The faulty implementation must remain importable and executable.

Avoid fragile tests that fail because of import paths, test discovery, or unrelated implementation differences.

Use exact output assertions wherever possible.

Use NumPy/Pandas testing utilities where appropriate.

Clearly identify the assertion that demonstrates the actual `999` error.

For reference, the faulty implementation on the six-row training data should produce: processed DTI `[10, 20, 30, 40, 999, 999]`, `dti_missing` `[0, 0, 0, 0, 0, 0]`, and a stored median of 35 (the median of all six values, including the two 999s), instead of 25. If the faulty version fails for any other reason, the pair is not valid. Fix it rather than accepting the failure.

---

# PHASE 8 — BUILD THE BENCHMARK RUNNER

Create:

`benchmark/run_pair1.py`

The runner must:

1. Locate the faulty and valid test files.
2. Load the corresponding JSON ground-truth labels.
3. Execute each test suite independently using pytest.
4. Capture exit codes and concise output.
5. Avoid terminating the entire runner when the faulty suite fails.
6. Produce a readable summary table.

Required columns:

```text
case_id | expected label | test result | agrees with label
```

Example expected interpretation:

```text
P1_F | Defect | FAIL | YES
P1_V | Valid  | PASS | YES
```

The runner must clearly explain that:

**FAIL means the implementation violates at least one specified test. It does not mean the test script failed to execute.**

Distinguish:

- Assertion failure.
- Import/collection error.
- Runtime error.
- Test pass.

Only a genuine expected assertion failure should count as evidence supporting the faulty case's label.

A missing package, broken import, or test collection error must not count as correct defect detection.

## Runner exit behavior

If both cases behave as expected, return exit code 0.

If either case disagrees with its label, or any test suite experiences a collection/runtime failure, return a nonzero exit code.

Do not hardcode the displayed test results.

Generate them from actual executed tests.

---

# PHASE 9 — CHECK DEPENDENCIES

Use the existing repository's Python environment.

If `uv` is configured, use `uv`.

Check whether these dependencies are already present:

- pandas
- numpy
- pytest

Only add missing dependencies.

For example:

```bash
uv add --dev pytest
```

If the repository does not use dependency groups, follow its existing dependency conventions.

Avoid unnecessary package upgrades.

---

# PHASE 10 — EXECUTE ALL REQUIRED TESTS

Run the following commands from the repository root:

```bash
uv run pytest benchmark/pair1_preprocessing -v
```

Important: This combined command is expected to return a nonzero exit code because the faulty implementation intentionally fails specification tests.

Then run separately:

```bash
uv run pytest benchmark/pair1_preprocessing/valid -v
```

```bash
uv run pytest benchmark/pair1_preprocessing/faulty -v
```

```bash
uv run python benchmark/run_pair1.py
```

If this repository cannot run `uv`, use the established equivalent and explain why.

## Required test outcomes

- Valid case: PASS.
- Faulty case: At least one genuine assertion FAIL.
- Combined pytest: Expected nonzero exit code because of the faulty case.
- Benchmark runner: PASS overall when the faulty and valid cases agree with the ground-truth labels.

Do not change the tests simply to make the commands appear successful.

If a command fails unexpectedly, debug the implementation or test environment and rerun it.

Do not hide failures.

---

# PHASE 11 — INDEPENDENT VALIDATION

Before finishing:

1. Independently calculate the expected training median.
2. Independently calculate the correct missing indicators.
3. Confirm that `999` is excluded from legitimate numerical DTI calculations.
4. Confirm that the valid implementation uses the supplied training median during inference.
5. Confirm that the faulty implementation really retains the intended defect.
6. Verify no real Freddie Mac data was copied into the benchmark.
7. Verify `src/baseline_logreg.py` has not changed.
8. Verify all generated files are located in the requested directories.
9. Check the final Git diff.
10. Make sure no unrelated files were modified.

Do not modify the mathematical ground truth merely to match implementation behavior.

---

# PHASE 12 — DELIVER THE RESULTS

At the end of your work, provide the following information.

## A. Files created

Show the final directory structure.

Explain each file in simple English, including:

- What `spec.md` does.
- What `impl.py` does.
- What `label.json` means.
- What `test_case.py` checks.
- What `run_pair1.py` reports.

## B. Mathematical explanation

Explain, using the six synthetic DTI values, why the correct median is 25 and why 999 cannot be treated as a real observation.

## C. Actual test results

Include the actual terminal output from:

```bash
uv run pytest benchmark/pair1_preprocessing -v
```

Also include the output from:

```bash
uv run python benchmark/run_pair1.py
```

Do not fabricate terminal results.

## D. Research interpretation

Explain:

- Why `P1_F` is objectively defective.
- Why `P1_V` is valid.
- How the pair will help evaluate false positives in a future LLM validation agent.
- Why the benchmark must contain both faulty and valid cases.
- What this test proves.
- What this test does NOT prove.

In particular, passing this benchmark does not yet establish research novelty or VALID-Agent performance.

## E. Repository checks

Report:

- Files created or changed.
- Whether the protected baseline file remained unchanged.
- Whether the worktree contains unrelated changes.
- Any incomplete work or technical limitations.

## F. Git commit message

Suggest one concise, professional Git commit message.

Do not make a Git commit unless I explicitly request it.

---

# ACCEPTANCE CRITERIA

Consider the task complete only when:

- The existing `CLAUDE.md` instructions have been read.
- Both implementations follow the same public interface.
- The faulty implementation contains only the intended `999` preprocessing defect.
- The valid implementation treats `999` as missing.
- The valid implementation calculates and reuses a training-only median.
- Both cases have documented specifications and JSON labels.
- Tests use synthetic data only.
- The faulty case fails for the intended mathematical reason.
- The valid case passes.
- The runner reports the correct interpretation of both outcomes.
- The actual test commands have been executed.
- No unrelated source code has been changed.
- The final response includes test results and a suggested commit message.

**Do not begin building VALID-Agent, RAG, XGBoost, or any other benchmark pair. This task is limited to Pair 1: DTI preprocessing.**
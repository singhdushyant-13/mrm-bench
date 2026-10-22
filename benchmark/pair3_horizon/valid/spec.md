# Specification — MRM-Bench Pair 3: outcome window (horizon)

**Task: 6-month 90+ DPD early-warning indicator. Horizon H = 6 months.**
The machine-readable form of this parameter is `spec.json` in this folder (`"horizon_months": 6`).

This specification states the required behaviour of `label_outcomes(panel, prediction_month,
data_cutoff=None)`. The ground-truth label of this case and the tests in `test_case.py` are derived from
it. The tests read H from `spec.json` and cite the rules O1–O7 below.

---

## 1. Purpose

For each loan that is at risk at a prediction month t, assign the outcome label of the task
**6-month 90+ DPD early-warning indicator**. The question is whether the loan became seriously delinquent (90+ days past due, or REO
acquisition) in the 6 months after t, and, if it did not, whether it was observed long enough to say so.
This is a separately documented early-warning task: a shorter-horizon indicator of the same kind of event. It is a different target from the project's main 12-month task, so labels built for one task must not be used for the other.
This is a research outcome, not a regulatory default definition.

---

## 2. Outcome rule (H = 6)

Notation: t is the prediction month. Months are consecutive integers, so t+1 is the next calendar month.
A loan's *last record* is its latest month on or before the data cutoff.

- **O1 — Data cutoff.** `data_cutoff` is the last month of available data (default: the latest month in
  the panel). Records after it are ignored.
- **O2 — At risk at t.** A loan is at risk if it has a record at t, its status at t is "00", "01" or "02",
  and it has no zero-balance code at or before t. Loans that are not at risk are omitted from the output.
- **O3 — Window.** The window is months t+1 … t+6. Month t is excluded and month t+6 is included.
- **O4 — event.** Some month in the window has numeric status ≥ 03 or status "RA". This label takes
  precedence over all others: a 90+ month followed by a termination is still an event.
- **O5 — non_event.** There is no event, and the loan's last record is at month ≥ t+6. A zero-balance
  record exactly at t+6 counts as observed.
- **O6 — censored** (no event, and the last record is before t+6):
  - `censored_prepay`: the zero-balance code is "01" (prepaid or matured);
  - `censored_other_termination`: any other zero-balance code (02, 03, 09, 15, 16, 96);
  - `censored_end_of_data`: there is no zero-balance code. Normally this is because the data end before
    the window does (t+6 > data_cutoff). A loan whose records stop earlier without a zero-balance code
    gets the same label, because its outcome over the window was not observed. This matches the
    project baseline (`build_snapshot()`).
- **O7 — Output and validation.** The output has one row per at-risk loan, indexed by `loan_id` in sorted
  order, with a single column `label`. The input panel is not modified. The function raises `ValueError`
  for missing required columns (`loan_id`, `month`, `delinq_status`, `zero_balance_code`), duplicate
  (`loan_id`, `month`) rows, or a prediction month with no records.

Formally, for an at-risk loan i with record months M_i, last record L_i = max M_i, statuses s_i(m) and
zero-balance code c_i (if any):

    event_i = there is an m in M_i with t < m ≤ t+6 and (s_i(m) ≥ 03 or s_i(m) = "RA")

    label_i = event                        if event_i
            = non_event                    else if L_i ≥ t+6
            = censored_prepay              else if c_i = "01"
            = censored_other_termination   else if c_i exists
            = censored_end_of_data         otherwise

---

## 3. Why the horizon matters

The window length H is part of the definition of the outcome. A window of any other length H′ ≠ 6
labels a different outcome:

- **H′ < H:** events in months t+H′+1 … t+H are missed, so those loans become non_event instead of event.
  Loans that terminate or run out of data between t+H′ and t+H become non_event instead of censored.
- **H′ > H:** events in months t+H+1 … t+H′ are wrongly counted. Loans observed through t+H but not
  through t+H′ are wrongly censored.

Either change alters the event rate and which loans enter model fitting. A model trained on such labels
therefore estimates a different probability from the one specified here.

**Synthetic example** (fictional loans; t = 0; panel A covers months 0–14; panel B ends at month 8):

| loan | situation | label required here (H = 6) | label with a 12-month window |
|---|---|---|---|
| S01 | current ("00") throughout | non_event | non_event |
| S02 | "00" to month 4, "01", "02", first 90+ ("03") at month 7 | non_event | event |
| S03 | "03" at month 4, "RA" from month 7, zero-balance code 09 at month 10 | event | event |
| S04 | current, zero-balance code 01 (prepaid) at month 8 | non_event | censored_prepay |
| S05 | already "03" at month 0 | omitted (not at risk) | omitted (not at risk) |
| S06 | current, zero-balance code 15 (whole loan sale) at month 10 | non_event | censored_other_termination |
| S07 | "00", "00", "01", "02", then "00" | non_event | non_event |
| S08 | "00" to month 11, "03" at month 12 | non_event | event |
| S09 | "00" to month 5, "03" at month 6 | event | event |
| S10 | "00" to month 12, "03" at month 13 | non_event | non_event |
| E01 | current, panel B ends at month 8 | non_event | censored_end_of_data |

Events on panel A: **2** with H = 6, and 4 with a 12-month window.

---

## 4. What is NOT a defect

- **Status jumps.** A status can move from "00" to "03" between consecutive monthly records (S09: "00" to
  month 5, "03" at month 6). The rule does not require delinquency to pass through "01" and "02". The
  status is derived from the due date of last paid installment reported by servicers (guide p. 10), and
  the project lists such jumps as a known false-alarm trap (CLAUDE.md).
- **Termination after a 90+ month.** A loan that prepays or terminates after a 90+ month is still an event
  (O4). For example, S03 is 90+ at month 4, becomes REO acquisition ("RA") from month 7, and gets
  zero-balance code 09 (REO disposition) at month 10. It is an event.
- **Omitting a loan already 90+ at t.** Such a loan (S05) is not at risk, so leaving it out is correct.
- **60–89 days ("02").** This is not the event (S07 → non_event).
- **The window length on its own.** A 6-month window is correct for a documented 6-month task, and a
  12-month window is correct for a documented 12-month task. What matters is agreement with *this*
  specification (H = 6, `spec.json`). Code whose window differs from some other task's window is not
  defective for that reason alone.
- **Censored loans.** These are not errors. They are excluded from model fitting and counted separately
  (CLAUDE.md, "Outcome specification").

---

## 5. Applicability conditions

- **A1.** The panel follows the Freddie Mac SFLLD monthly performance data (Release 47). Delinquency status
  and zero-balance codes are as listed in §6, and a zero-balance code appears only on a loan's final
  record.
- **A2.** `month` is a consecutive monthly index, so t+1 is the next calendar month. An example is the
  baseline's `ym_index`, which counts months since year 0. Raw YYYYMM integers do not qualify, because
  201512 + 1 is not a valid month.
- **A3.** The task and H are those in `spec.json`. `spec.md` and `spec.json` must agree.
- **A4.** Status "XX" (Not Available) is not "00", "01" or "02", so under O2 a loan with "XX" at t is not at
  risk. Inside the window, "XX" is not an event. The project baseline would treat a loan with "XX" at t as
  at risk, because it excludes only 90+ and "RA". The 2015 sample contains no "XX" records (aggregate check,
  2026-10-22), so the two rules agree on the project data.
- **A5.** Features are outside this function. The separate rule that features may use only records at or
  before t is not tested here.

---

## 6. Evidence and provenance

- **Freddie Mac**, *Single-Family Loan-Level Dataset — General User Guide*, Release 47, July 2026, section
  "Monthly Performance Data File". Local copy: `data/raw/general_user_guide_july_2026.pdf` (gitignored). The
  text was extracted with `pypdf` on 2026-10-22, and the quotations below were copied from it.
  - **p. 10, "Current Loan Delinquency Status":** "A value corresponding to the number of days the borrower
    is delinquent, based on the Due Date of Last Paid Installment (“DDLPI”) reported by servicers to
    Freddie Mac." "If a loan is acquired through REO, then the Current Loan Delinquency Status will reflect
    “RA” for REO Acquisition (instead of the value corresponding to the number of days the borrower is
    delinquent)." Enumerations: "00 = Current, or less than 30 days delinquent", "01 = 30-59 days
    delinquent", "02 = 60 – 89 days delinquent", "03 = 90 – 119 days delinquent etc.", "RA = REO
    Acquisition", "XX = Not Available".
  - **p. 11, "Zero Balance Code":** "A code indicating the reason the loan's balance was reduced to zero."
    Enumerations: "01 = Prepaid or Matured (Voluntary Payoff)", "02 = Third Party Sale", "03 = Short Sale
    or Charge Off", "09 = REO Disposition", "15 = Whole Loan sales", "16 = Reperforming loan
    securitizations", "96 = Confirmed Underwriting Defect or Major Servicing Defect prior to credit event".
- **Project outcome specification:** `CLAUDE.md`, section "Outcome specification (draft, pending professor
  approval)", and its reference implementation `build_snapshot()` in `src/baseline_logreg.py`. Together
  they supply the at-risk, event, non-event and censoring rules. That specification uses H = 12 for the project's main task. This task keeps all of its rules and changes only the horizon, to H = 6, as documented here and in `spec.json`.
- **Executable evidence:** `test_case.py` in this folder. It uses synthetic panels only (loan IDs S01–S10
  and E01), and its expected labels were computed in advance and rechecked with a separate reference
  labeller.

"""
MRM-Bench Pair 1 runner - DTI preprocessing
===========================================
Runs each case's test suite in its own Python process (pytest), compares the outcome with the case's
ground-truth label (label.json), runs the data-hygiene check, and prints a summary table.

HOW TO RUN (from the repository root)
  uv run python benchmark/run_pair1.py              # summary
  uv run python benchmark/run_pair1.py --verbose    # also print each suite's own pytest output

How a suite's "test result" is classified (always from the executed tests, never hardcoded)
  PASS              every test passed
  FAIL              at least one test failed on an assertion (assert, pandas assert_*, or pytest.raises
                    "DID NOT RAISE") and no test crashed -> the implementation violates the specification
  RUNTIME ERROR     a test raised an unexpected exception (e.g. a TypeError inside the implementation)
  COLLECTION ERROR  the test file could not be imported or collected (broken import, missing package, ...)
  other             NO TESTS / SKIPPED TESTS / PYTEST ERROR / RUNNER ERROR
Only FAIL counts as evidence for a defect label; an error never counts as correct defect detection.

Exit code: 0 if every case agrees with its label and the hygiene check passes; 1 otherwise.
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
PAIR_DIR = BENCHMARK_DIR / "pair1_preprocessing"
CASES = ["faulty", "valid"]                      # sub-folders, each with spec.md, impl.py, label.json, test_case.py
HYGIENE_TEST = BENCHMARK_DIR / "test_no_real_data.py"
LABEL_KEYS = {"case_id", "pair_id", "category", "is_defect", "violated_rule", "evidence", "source"}


def _short(text):
    """First line of an error message plus pandas' [left]/[right] lines, on one line."""
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    keep = lines[:1] + [s for s in lines[1:] if s.startswith(("[left]", "[right]"))]
    return " | ".join(keep)[:300]


# ---------------------------------------------------------------- 1. child process: run one suite under pytest
def child_run_suite(test_file, out_json):
    """Runs pytest on one test file and writes one record per test, including the exception type."""
    import pytest

    class Recorder:
        def __init__(self):
            self.tests, self.collection_errors = [], []

        def pytest_collectreport(self, report):
            if report.failed:                    # import / collection problem, not a test verdict
                last = [s for s in str(report.longrepr).splitlines() if s.strip()][-1:]
                self.collection_errors.append(" ".join(last).strip()[:300])

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_makereport(self, item, call):
            report = yield
            doc = (getattr(item, "function", None).__doc__ or "").strip().splitlines()
            rec = {"name": item.name, "doc": doc[0] if doc else "", "message": ""}
            if report.failed:
                exc = call.excinfo
                # an assertion failure is a specification verdict; any other exception is a crash
                is_assertion = (call.when == "call" and exc is not None
                                and issubclass(exc.type, (AssertionError, pytest.fail.Exception)))
                rec["outcome"] = "assertion_failure" if is_assertion else "runtime_error"
                rec["message"] = _short(exc.exconly()) if exc is not None else ""
                self.tests.append(rec)
            elif report.skipped:
                self.tests.append(rec | {"outcome": "skipped"})
            elif report.when == "call":
                self.tests.append(rec | {"outcome": "passed"})
            return report

    recorder = Recorder()
    exit_code = pytest.main([str(test_file), "-q", "-p", "no:cacheprovider"], plugins=[recorder])
    Path(out_json).write_text(json.dumps({"exit_code": int(exit_code), "tests": recorder.tests,
                                          "collection_errors": recorder.collection_errors}))


# ---------------------------------------------------------------- 2. parent: run, classify, compare with labels
def classify(rec):
    outcomes = [t["outcome"] for t in rec["tests"]]
    if rec["collection_errors"]:
        return "COLLECTION ERROR"
    if rec["exit_code"] not in (0, 1, 5):        # pytest: 2 interrupted, 3 internal error, 4 usage error
        return f"PYTEST ERROR (exit {rec['exit_code']})"
    if "runtime_error" in outcomes:
        return "RUNTIME ERROR"
    if not outcomes:
        return "NO TESTS"
    if "skipped" in outcomes:
        return "SKIPPED TESTS"
    if "assertion_failure" in outcomes:
        return "FAIL"
    return "PASS"


def run_suite(test_file):
    """Runs one suite in a fresh Python process, so the suites cannot affect each other."""
    if not test_file.exists():
        return {"result": "RUNNER ERROR", "tests": [], "exit_code": None, "output": f"missing {test_file}"}
    with tempfile.TemporaryDirectory() as tmp:
        out_json = Path(tmp) / "result.json"
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(test_file),
                               str(out_json)], cwd=REPO_ROOT, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        if not out_json.exists():                # the child itself crashed (e.g. pytest not installed)
            return {"result": "RUNNER ERROR", "tests": [], "exit_code": proc.returncode, "output": output}
        rec = json.loads(out_json.read_text())
    rec["output"] = output
    rec["result"] = classify(rec)
    return rec


def load_label(case_dir):
    label = json.loads((case_dir / "label.json").read_text())
    missing = LABEL_KEYS - label.keys()
    if missing or not isinstance(label["is_defect"], bool):
        raise SystemExit(f"invalid label {case_dir / 'label.json'}: missing keys {sorted(missing)} "
                         f"or non-boolean is_defect")
    return label


def main(verbose):
    rows, details = [], []
    for case in CASES:
        case_dir = PAIR_DIR / case
        label = load_label(case_dir)
        rec = run_suite(case_dir / "test_case.py")
        expected = "FAIL" if label["is_defect"] else "PASS"   # a defect must fail the spec tests; valid must pass
        agrees = rec["result"] == expected
        rows.append((label["case_id"], "Defect" if label["is_defect"] else "Valid", rec["result"],
                     "YES" if agrees else "NO"))
        details.append((label["case_id"], rec))

    hygiene = run_suite(HYGIENE_TEST)

    print("MRM-Bench Pair 1 - DTI preprocessing (each suite run in its own pytest process)\n")
    header = ("case_id", "expected label", "test result", "agrees with label")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(4)]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(r, widths)))

    print("\nDetails")
    for case_id, rec in details:
        n = {k: sum(t["outcome"] == k for t in rec["tests"])
             for k in ("passed", "assertion_failure", "runtime_error", "skipped")}
        print(f"  {case_id}: {len(rec['tests'])} test records - {n['passed']} passed, "
              f"{n['assertion_failure']} assertion failures, {n['runtime_error']} runtime errors, "
              f"{n['skipped']} skipped (pytest exit code {rec['exit_code']})")
        for t in rec["tests"]:
            if t["outcome"] in ("assertion_failure", "runtime_error"):
                kind = "assertion failure" if t["outcome"] == "assertion_failure" else "RUNTIME ERROR"
                print(f"    - {kind}: {t['name']}\n        spec: {t['doc']}\n        got : {t['message']}")
        for err in rec.get("collection_errors", []):
            print(f"    - COLLECTION ERROR: {err}")
        if verbose or rec["result"] not in ("PASS", "FAIL"):
            print("    pytest output:\n" + "\n".join("      " + s for s in rec["output"].splitlines()))

    hygiene_ok = hygiene["result"] == "PASS"
    print(f"\nData hygiene ({HYGIENE_TEST.relative_to(REPO_ROOT)}): {hygiene['result']}"
          + ("  (no Freddie Mac loan-ID pattern, no file > 200 KB under benchmark/)" if hygiene_ok else ""))
    if not hygiene_ok:
        print("\n".join("      " + s for s in hygiene["output"].splitlines()))

    print("\nHow to read 'test result':\n"
          "  FAIL means the implementation violates at least one specified test. It does not mean the test\n"
          "  script failed to execute. Only assertion failures count as FAIL; RUNTIME ERROR (a crash inside a\n"
          "  test) and COLLECTION ERROR (import/collection problem) never count as detecting a defect.\n"
          "  Labels are ground truth set by the benchmark designers; the tests independently check them.")

    ok = all(r[3] == "YES" for r in rows) and hygiene_ok
    print("\nOverall: " + ("OK - every case agrees with its ground-truth label." if ok else
                           "PROBLEM - a case disagrees with its label, or a suite/hygiene check did not run cleanly."))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run MRM-Bench Pair 1 and compare outcomes with ground-truth labels.")
    ap.add_argument("--verbose", action="store_true", help="print each suite's full pytest output")
    ap.add_argument("--child", nargs=2, metavar=("TEST_FILE", "OUT_JSON"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        child_run_suite(*args.child)
        sys.exit(0)
    sys.exit(main(args.verbose))

"""
Data-hygiene check for benchmark/ (CLAUDE.md: never copy Freddie Mac loan-level records into the benchmark).

Fails if any file under benchmark/ contains text shaped like a Freddie Mac SFLLD loan ID, or is larger than
200 KB (a sign that a data extract was copied in). The regex fully matches all 50,000 loan IDs of the 2015
sample (checked by count only, 2026-10-22).
"""
import re
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
LOAN_ID = re.compile(rb"F1[0-9]Q[1-4][0-9]{7}")
MAX_BYTES = 200 * 1024


def _files():
    return sorted(p for p in BENCHMARK_DIR.rglob("*") if p.is_file())


def test_no_loan_id_pattern():
    hits = [str(p.relative_to(BENCHMARK_DIR)) for p in _files() if LOAN_ID.search(p.read_bytes())]
    assert hits == [], f"files containing a Freddie Mac loan-ID pattern: {hits}"


def test_no_large_files():
    big = [f"{p.relative_to(BENCHMARK_DIR)} ({p.stat().st_size:,} bytes)" for p in _files()
           if p.stat().st_size > MAX_BYTES]
    assert big == [], f"files larger than 200 KB: {big}"

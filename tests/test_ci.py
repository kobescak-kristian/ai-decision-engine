"""CI assertion wrapper: keyless simulation-mode pipeline run.

Runs `python main.py` then `python data/seed_outcomes.py` with an empty
OPENAI_API_KEY (simulation mode). Assertions are message substrings only
(logger lines carry wall-clock [HH:MM:SS] prefixes and ANSI colours) and
cover counts and exit codes exclusively — never run IDs, timestamps,
per-record durations, or OS-dependent path separators. All asserted
lines are pure functions of the frozen 50-record sample input, the
hardcoded simulated-output dict, and fixed thresholds; none reads the
clock. Insert-counts on seeding are re-run-variant ("32 inserted" vs
"32 already recorded"), so only the invariant coverage line is frozen.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 300

# Frozen 2026-07-27 from piped fresh-clone runs, Python 3.12.0 and
# 3.14.4: two identical fresh-state runs + one PYTHONHASHSEED-varied
# run per interpreter, identical after stripping ANSI colours,
# wall-clock prefixes, run IDs, and millisecond timings.
FROZEN_PIPELINE = [
    "MODE: SIMULATION",
    "Total records : 50",
    "FinalDecision.SEND_TO_SALES: 25",
    "FinalDecision.ARCHIVE: 14",
    "FinalDecision.MANUAL_REVIEW: 11",
    "Fallbacks     : 0",
]
FROZEN_SEED = [
    "Coverage: 32/50 = 64%",
]


def _run(args):
    env = dict(os.environ, OPENAI_API_KEY="")
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=TIMEOUT_S,
    )


def test_simulation_pipeline():
    r = _run(["main.py"])
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}"
    for frozen in FROZEN_PIPELINE:
        assert frozen in r.stdout, f"missing frozen line: {frozen!r}"


def test_seed_outcomes_coverage():
    r = _run(["data/seed_outcomes.py"])
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}"
    for frozen in FROZEN_SEED:
        assert frozen in r.stdout, f"missing frozen line: {frozen!r}"

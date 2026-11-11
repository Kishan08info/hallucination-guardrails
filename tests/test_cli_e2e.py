"""End-to-end CLI tests: check one answer, exit codes, JSON output."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = "/home/hatch/workspace/ml-projects/venv/bin/python"


def run_cli(*args):
    return subprocess.run(
        [VENV_PY, "-m", "guardrails.cli", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_pass_clean_answer():
    r = run_cli(
        "check",
        "--answer",
        "The Pro plan costs $199 per month.",
        "--sources",
        "data/corpus.json",
        "--freshness-cutoff",
        "2025-01-01",
    )
    assert r.returncode == 0, r.stderr
    assert "Decision: PASS" in r.stdout


def test_cli_block_wrong_price():
    r = run_cli(
        "check",
        "--answer",
        "The Growth plan costs $79 per month.",
        "--sources",
        "data/corpus.json",
        "--freshness-cutoff",
        "2025-01-01",
    )
    assert r.returncode == 2, r.stdout
    assert "Decision: BLOCK" in r.stdout
    assert "numbers not in source: ['79']" in r.stdout


def test_cli_json_output_and_cutoff_effect():
    # Stale price: passes with the stale doc in the corpus...
    r = run_cli(
        "check",
        "--answer",
        "The Starter plan costs $29 per month.",
        "--sources",
        "data/corpus.json",
        "--json",
    )
    assert r.returncode == 0, r.stdout
    assert json.loads(r.stdout)["decision"] == "PASS"
    # ...and is blocked once the freshness cutoff excludes the 2024 doc.
    r = run_cli(
        "check",
        "--answer",
        "The Starter plan costs $29 per month.",
        "--sources",
        "data/corpus.json",
        "--freshness-cutoff",
        "2025-01-01",
        "--json",
    )
    assert r.returncode == 2, r.stdout
    assert json.loads(r.stdout)["decision"] == "BLOCK"

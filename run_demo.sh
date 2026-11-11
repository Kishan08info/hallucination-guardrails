#!/usr/bin/env bash
# Re-run the demo and the test suite from a clean state.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf outputs
python3 demo.py
python3 -m pytest tests/ -q

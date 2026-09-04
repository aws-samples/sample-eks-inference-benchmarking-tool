#!/usr/bin/env bash
set -euo pipefail

# Create the Python virtual environment.
echo "==> Creating Python virtual environment"
python3 -m venv .venv

# Install the benchmark, tokenizer, optional AIPerf, and CLI dependencies.
echo "==> Installing benchmark dependencies"
.venv/bin/python -m pip install --no-cache-dir \
  -r app/bench/requirements.txt \
  -r app/bench/requirements-tokenizer.txt \
  -r app/bench/requirements-aiperf.txt \
  -r app/requirements-cli.txt

#!/bin/bash
# smoke_test.sh - Expanded smoke test for i2t CLI

set -e

SCRIPT_DIR=$(dirname "$0")
VENV_DIR="$SCRIPT_DIR/.venv"

# Activate the venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
  if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
  else
    echo "Virtual environment not found! Please run setup_env.sh first."
    exit 1
  fi
fi

echo "[1] Test: Show help"
python3 -m src.i2t.cli --help

echo "[2] Test: BLIP model, text output (default)"
python3 -m src.i2t.cli test.jpg --model blip

echo "[3] Test: BLIP model, JSON output"
python3 -m src.i2t.cli test.jpg --model blip --format json

echo "[4] Test: Joy model, text output"
python3 -m src.i2t.cli test.jpg --model joy

echo "[5] Test: Joy model, JSON output"
python3 -m src.i2t.cli test.jpg --model joy --format json

echo "[6] Test: Show image before captioning (BLIP)"
python3 -m src.i2t.cli test.jpg --model blip --show

echo "[7] Test: Pre-cache BLIP model"
python3 -m src.i2t.cli --model blip --precache

echo "[8] Test: Pre-cache Joy model"
python3 -m src.i2t.cli --model joy --precache

echo "[9] Test: Batch mode, BLIP, text output"
python3 -m src.i2t.cli --batch-dir batch_test --model blip

echo "[10] Test: Batch mode, BLIP, JSON output"
python3 -m src.i2t.cli --batch-dir batch_test --model blip --format json

echo "[11] Test: Batch mode, Joy, text output"
python3 -m src.i2t.cli --batch-dir batch_test --model joy

echo "[12] Test: Batch mode, Joy, JSON output"
python3 -m src.i2t.cli --batch-dir batch_test --model joy --format json

echo "All smoke tests passed!"
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

echo "[2b] Test: BLIP-Large model, text output (default)"
python3 -m src.i2t.cli test.jpg --model blip-large

echo "[3b] Test: BLIP-Large model, JSON output"
python3 -m src.i2t.cli test.jpg --model blip-large --format json

echo "[6] Test: Show image before captioning (BLIP)"
python3 -m src.i2t.cli test.jpg --model blip --show

echo "[7] Test: Pre-cache BLIP model"
python3 -m src.i2t.cli --model blip --precache

echo "[7b] Test: Pre-cache BLIP-Large model"
python3 -m src.i2t.cli --model blip-large --precache

echo "[9] Test: Batch mode, BLIP, text output"
python3 -m src.i2t.cli --batch-dir batch_test --model blip

echo "[9b] Test: Batch mode, BLIP-Large, text output"
python3 -m src.i2t.cli --batch-dir batch_test --model blip-large

echo "[10] Test: Batch mode, BLIP, JSON output"
python3 -m src.i2t.cli --batch-dir batch_test --model blip --format json

echo "[10b] Test: Batch mode, BLIP-Large, JSON output"
python3 -m src.i2t.cli --batch-dir batch_test --model blip-large --format json

echo "All smoke tests passed!"
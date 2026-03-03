#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NB_PATH="${1:-$ROOT_DIR/report/final/report.ipynb}"
MODE="${2:-webpdf}" # webpdf | pdf

if [[ ! -f "$NB_PATH" ]]; then
  echo "Notebook not found: $NB_PATH"
  exit 1
fi

OUT_DIR="$(dirname "$NB_PATH")"
OUT_NAME="$(basename "${NB_PATH%.ipynb}")"

echo "[1/1] Execute and export $MODE PDF"
if [[ "$MODE" == "webpdf" ]]; then
  "$ROOT_DIR/scripts/py.sh" -m jupyter nbconvert "$NB_PATH" \
    --to webpdf \
    --execute \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "$OUT_NAME" \
    --output-dir "$OUT_DIR" \
    --allow-chromium-download \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_input_tags='["hide-input","remove-input"]'
elif [[ "$MODE" == "pdf" ]]; then
  "$ROOT_DIR/scripts/py.sh" -m jupyter nbconvert "$NB_PATH" \
    --to pdf \
    --execute \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "$OUT_NAME" \
    --output-dir "$OUT_DIR" \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_input_tags='["hide-input","remove-input"]'
else
  echo "Unsupported mode: $MODE (use: webpdf | pdf)"
  exit 1
fi

echo "Done: $OUT_DIR/$OUT_NAME.pdf"

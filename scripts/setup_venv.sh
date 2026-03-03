#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
DEFAULT_PY311="/opt/miniconda3/envs/rl-risk/bin/python"

choose_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${PYTHON_BIN}" ]]; then
      echo "${PYTHON_BIN}"
      return 0
    fi
    echo "PYTHON_BIN is set but not executable: ${PYTHON_BIN}" >&2
    return 1
  fi

  if [[ -x "${DEFAULT_PY311}" ]]; then
    echo "${DEFAULT_PY311}"
    return 0
  fi

  if command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  echo "No suitable python interpreter found (tried: PYTHON_BIN, ${DEFAULT_PY311}, python3.11, python3)." >&2
  return 1
}

if [[ ! -d "$VENV_DIR" ]]; then
  PYTHON_BIN_SELECTED="$(choose_python_bin)"
  echo "Creating venv at $VENV_DIR"
  echo "Using interpreter: $PYTHON_BIN_SELECTED"
  "$PYTHON_BIN_SELECTED" -m venv "$VENV_DIR"
else
  echo "Using existing venv at $VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Venv ready: $VENV_DIR"
echo "Use commands via: $VENV_DIR/bin/python -m <module>"

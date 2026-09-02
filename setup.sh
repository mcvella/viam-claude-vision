#!/usr/bin/env bash
# setup.sh -- bootstrap a Python virtualenv for this module via uv

set -euo pipefail

cd "$(dirname "$0")"

if [ -n "${VIAM_MODULE_DATA:-}" ]; then
  VIRTUAL_ENV="${VIAM_MODULE_DATA}/.venv"
else
  VIRTUAL_ENV="$(pwd)/.venv"
fi

export PATH="${PATH}:${HOME}/.local/bin"

if [ -f "${VIRTUAL_ENV}/.install_complete" ]; then
  echo "virtualenv already prepared at ${VIRTUAL_ENV}"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install uv" >&2
    exit 1
  fi
  echo "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${PATH}:${HOME}/.local/bin"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH after install" >&2
  exit 1
fi

echo "creating virtualenv at ${VIRTUAL_ENV}"
uv venv --allow-existing "${VIRTUAL_ENV}"

echo "installing dependencies from requirements.txt"
uv pip install --python "${VIRTUAL_ENV}/bin/python" -r requirements.txt

touch "${VIRTUAL_ENV}/.install_complete"
echo "setup complete"

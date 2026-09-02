#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -n "${VIAM_MODULE_DATA:-}" ]; then
  VIRTUAL_ENV="${VIAM_MODULE_DATA}/.venv"
else
  VIRTUAL_ENV="$(pwd)/.venv"
fi

export PATH="${PATH}:${HOME}/.local/bin"

./setup.sh

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
exec "${VIRTUAL_ENV}/bin/python" -m src "$@"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Prefer the writable module data dir so installs survive package swaps.
if [ -n "${VIAM_MODULE_DATA:-}" ]; then
  VIRTUAL_ENV="${VIAM_MODULE_DATA}/.venv"
else
  VIRTUAL_ENV="$(pwd)/.venv"
fi

./setup.sh

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
exec "${VIRTUAL_ENV}/bin/python" -m src "$@"

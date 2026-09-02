#!/usr/bin/env bash
# setup.sh -- bootstrap a Python virtualenv for this module

set -euo pipefail

cd "$(dirname "$0")"

# Prefer the writable module data dir so installs survive package swaps.
if [ -n "${VIAM_MODULE_DATA:-}" ]; then
  VIRTUAL_ENV="${VIAM_MODULE_DATA}/.venv"
else
  VIRTUAL_ENV="$(pwd)/.venv"
fi

SUDO=sudo
if ! command -v "$SUDO" >/dev/null 2>&1; then
  echo "no sudo on this system, proceeding as current user"
  SUDO=""
fi

if command -v apt-get >/dev/null 2>&1; then
  if ! dpkg --status python3-venv >/dev/null 2>&1; then
    if ! apt-cache show python3-venv >/dev/null 2>&1; then
      echo "package info not found, trying apt update"
      $SUDO apt-get -qq update
    fi
    echo "installing python3-venv"
    $SUDO apt-get install -qqy python3-venv
  fi
else
  echo "Skipping apt install (no apt-get). If setup fails, install python3-venv (or equivalent) for your system."
fi

if [ -f "${VIRTUAL_ENV}/.install_complete" ]; then
  echo "virtualenv already prepared at ${VIRTUAL_ENV}"
  exit 0
fi

echo "creating virtualenv at ${VIRTUAL_ENV}"
python3 -m venv "${VIRTUAL_ENV}"

# Some hosts ship venv without pip; bootstrap if needed.
if ! "${VIRTUAL_ENV}/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "pip missing in venv; running ensurepip"
  "${VIRTUAL_ENV}/bin/python" -m ensurepip --upgrade
fi

echo "installing dependencies from requirements.txt"
"${VIRTUAL_ENV}/bin/python" -m pip install --upgrade pip
"${VIRTUAL_ENV}/bin/python" -m pip install -r requirements.txt
touch "${VIRTUAL_ENV}/.install_complete"
echo "setup complete"

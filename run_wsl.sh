#!/usr/bin/env bash
# Run a command inside the WSL venv from the repo root (Windows Git Bash helper).
# Usage: ./run_wsl.sh python scripts/foo.py [args]
set -e
here="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
export MSYS_NO_PATHCONV=1
exec wsl.exe -d Ubuntu-22.04 --cd "$here" -- env XLA_PYTHON_CLIENT_PREALLOCATE=false PATH=/home/tomas/.venvs/lvad-rm/bin:/usr/bin:/bin "$@"

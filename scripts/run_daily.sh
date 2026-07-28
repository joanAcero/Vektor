#!/usr/bin/env bash
# run_daily.sh
# ------------
# cron/systemd entrypoint for the daily VEKTOR report.
#
# This is the entire "deploy updates the process" mechanism: it always runs
# whatever is currently on origin/main, never whatever happens to be sitting
# on disk. Push to main -> tomorrow's run uses the new code. No CI/CD needed.
#
# Repo root is derived from this script's own location, not hardcoded, so
# moving the repo (e.g. this machine -> a VPS later) needs zero edits here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${VEKTOR_VENV_DIR:-$REPO_DIR/vektor-venv}"
LOG_DIR="$REPO_DIR/results/logs"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

# Always run committed code, never a dirty local edit -- discard anything
# uncommitted before pulling, rather than have `git pull` fail on conflict.
git fetch origin main
git reset --hard origin/main

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Sanity gate: confirms the pulled tree actually contains the code you think
# it does (catches an incomplete push / bad merge), independent of its
# original PyCharm-console-caching motivation -- a fresh `python` subprocess
# here doesn't have that problem, but a content check after `git reset --hard`
# is still a real, cheap correctness check.
python check_versions.py

python daily_report.py >> "$LOG_DIR/daily_$(date +%F).log" 2>&1

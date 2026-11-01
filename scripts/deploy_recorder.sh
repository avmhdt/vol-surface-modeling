#!/usr/bin/env bash
# Deploy a git ref to the recorder's run-only clone on the WSL ext4 filesystem and install
# its cron lines. Run between scheduled runs, never across hh:15 UTC. Usage:
#   bash scripts/deploy_recorder.sh [ref]     (default origin/main)
set -euo pipefail

ref="${1:-origin/main}"
repo="https://github.com/avmhdt/vol-surface-modeling.git"
deploy="$HOME/deploy/vol-surface-modeling"
export PATH="$HOME/.local/bin:$PATH"

minute=$(date -u +%-M)
if (( minute >= 12 && minute <= 20 )); then
  echo "refusing to deploy at minute $minute: too close to the hh:15 run" >&2
  exit 1
fi

mkdir -p "$HOME/deploy" "$HOME/data/vol-surface-modeling/logs"
if [[ ! -d "$deploy/.git" ]]; then
  git clone --quiet "$repo" "$deploy"
fi
git -C "$deploy" fetch --quiet origin
git -C "$deploy" checkout --quiet --detach "$ref"
(cd "$deploy" && uv sync --quiet --frozen --no-dev --python 3.12)
"$deploy/.venv/bin/vsm" --help > /dev/null

{ crontab -l 2>/dev/null | grep -v 'vsm-recorder' || true; cat "$deploy/scripts/recorder.cron"; } | crontab -
echo "deployed $(git -C "$deploy" rev-parse --short HEAD)"
crontab -l | grep 'vsm-recorder'

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ARIN_ROOT"
arin_view_run_dir="${1:-}"
if [[ -z "$arin_view_run_dir" ]]; then
    arin_view_run_dir="$(.venv/bin/python -c 'import json; from pathlib import Path; print(json.loads(Path("runs/latest-main.json").read_text())["run_dir"])')"
fi
if [[ ! -d "$arin_view_run_dir/wandb/latest-run" ]]; then
    echo "No local W&B run found at $arin_view_run_dir/wandb/latest-run" >&2
    exit 2
fi
exec .venv/bin/wandb leet "$arin_view_run_dir/wandb/latest-run"

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ARIN_ROOT"
exec .venv/bin/python -u -m local_grpo.run "$@"

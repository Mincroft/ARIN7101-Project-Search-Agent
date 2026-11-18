#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ARIN_ROOT"
uv venv --python 3.11 .venv
arin_training_requirements="$(mktemp)"
trap 'rm -f "$arin_training_requirements"' EXIT
# CUDA extensions must build against the already installed, pinned PyTorch.
rg -v '^causal-conv1d==' requirements-train.txt > "$arin_training_requirements"
uv pip install --python .venv/bin/python -r "$arin_training_requirements"
if rg -q '^causal-conv1d==' requirements-train.txt; then
    MAX_JOBS="${MAX_JOBS:-4}" CAUSAL_CONV1D_FORCE_BUILD=TRUE \
        uv pip install --python .venv/bin/python --no-build-isolation --no-deps 'causal-conv1d==1.6.0'
fi
uv venv --python 3.11 .venv-vllm
uv pip install --python .venv-vllm/bin/python -r requirements-vllm.txt
.venv/bin/python scripts/download_assets.py --dataset --model

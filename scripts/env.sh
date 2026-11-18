#!/usr/bin/env bash
# Source this file before local GPU commands. The system compat directory
# contains a CUDA 575 library, while the loaded kernel driver is 580.
ARIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ARIN_ROOT:$ARIN_ROOT/upstream/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/lib:/home/opt/nvidia_lib:/usr/local/cuda/lib64"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8
export WANDB_PROJECT=ARIN7101-Project
export PYTORCH_ALLOC_CONF=expandable_segments:True

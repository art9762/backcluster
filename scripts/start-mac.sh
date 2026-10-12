#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:/opt/homebrew/bin:$PATH"
MODEL=${MODEL:-$HOME/Models/Qwen3.8-27B-IQ4_XS.gguf}
[[ -s "$MODEL" ]] || { echo "Model missing: $MODEL. Model download is deferred."; exit 1; }
[[ -x llama.cpp/build-metal/bin/llama-server ]] || { echo "Run scripts/prepare-apple.sh first"; exit 1; }
export MODEL PHONE=0 PHONE_ANE=0 DRAFT= SPEC_TYPE=none SPEC_REPLAY=0
export SME=0 MM_SME=0 PROXY=0 LOAD_MODE=mmap CTX="${CTX:-16384}" CTX_CHECKPOINTS=0
exec scripts/serve.sh "$@"

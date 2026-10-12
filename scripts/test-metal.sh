#!/usr/bin/env bash
# Run against the actual Mac build; registration alone can hide shader errors.
set -euo pipefail
cd "$(dirname "$0")/.."
BC_METAL_ROOT="$PWD"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "Metal smoke requires an Apple Silicon Mac" >&2
  exit 1
fi
BC_ENGINE_SOURCE=$(python3 -m tools.backcluster.engine_overlay --source-path)
BC_METAL_LIBS="${BC_METAL_ROOT}/llama.cpp/build-metal/bin"
mkdir -p .backcluster
xcrun clang++ -std=c++17 -O1 -Wall -Wextra \
  -I"${BC_ENGINE_SOURCE}/ggml/include" tests/backcluster/metal-smoke.cpp \
  -L"${BC_METAL_LIBS}" -lggml-metal -lggml -lggml-base \
  -Wl,-rpath,"${BC_METAL_LIBS}" -o .backcluster/metal-smoke
echo "Metal smoke: default device capabilities"
.backcluster/metal-smoke
# Exercise the older-device shader path even on a runner with BF16 support.
echo "Metal smoke: BF16 disabled"
GGML_METAL_BF16_DISABLE=1 .backcluster/metal-smoke

#!/usr/bin/env bash
# Tools and builds only. Never downloads model weights.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
[[ $(uname -m) == arm64 ]] || { echo "Requires an Apple Silicon Mac"; exit 1; }
xcodebuild -version
command -v cmake >/dev/null || HOMEBREW_NO_AUTO_UPDATE=1 brew install cmake
if ! xcrun -sdk iphoneos metal -v >/dev/null 2>&1; then
  xcodebuild -downloadComponent MetalToolchain
fi
git submodule update --init --recursive --depth 1
[[ $(git -C llama.cpp rev-parse HEAD) == 9f17fc31cf60753eb6bd6ec075894edab773cb91 ]] || { echo "Unexpected engine revision"; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -e llama.cpp/gguf-py
# macOS 27 CLI SDK may be newer than the installed Xcode linker.
# Select Xcode's own SDK consistently instead of the CLI-tools default.
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
cmake -S llama.cpp -B llama.cpp/build-metal -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT="$SDKROOT"
cmake --build llama.cpp/build-metal --target llama-server llama-quantize -j "${JOBS:-4}"
unset SDKROOT
IPA=1 JOBS="${JOBS:-4}" scripts/build-iphone.sh
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)" tests/security/run.sh
echo "Mac binaries: llama.cpp/build-metal/bin; mobile IPA: ios/build/Backburner.ipa"

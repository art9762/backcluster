#!/usr/bin/env bash
# Build the pinned patched engine and verify real three-stage CPU inference.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/check-engine.py "$@"

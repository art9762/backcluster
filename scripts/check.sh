#!/usr/bin/env bash
# Portable checks. Apple SDK/security tests are a separate explicit gate.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m unittest discover -s tests/backcluster -v
while IFS= read -r -d '' script; do bash -n "$script"; done < <(git ls-files -z '*.sh')
scripts/check-listeners.sh
tests/security/check-sensitive-test.sh
python3 scripts/hooks/check-sensitive.py --all
python3 scripts/hooks/check-sensitive.py --staged
if command -v c++ >/dev/null; then
  BC_CHECK_TMP=$(mktemp -d)
  trap 'rm -rf "$BC_CHECK_TMP"' EXIT
  c++ -std=c++17 -O1 -Wall -Wextra -pthread tests/backcluster/link-bench-test.cpp -o "$BC_CHECK_TMP/link-test"
  "$BC_CHECK_TMP/link-test"
else
  echo "Native link test skipped: no c++ compiler. CI runs it on Linux and macOS."
fi
echo "Portable checks passed. Apple security, app builds and device inference are separate gates."

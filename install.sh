#!/usr/bin/env bash
# Install the checkout's CLI only. Models, Xcode signing and engine builds are explicit steps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${BACKCLUSTER_BIN_DIR:-$HOME/.local/bin}"
case "${1:-}" in
  --help|-h)
    echo "Usage: ./install.sh [--build]"
    echo "Installs backcluster into BACKCLUSTER_BIN_DIR (default: ~/.local/bin)."
    echo "--build also builds the pinned Mac engine. No weights are downloaded."
    exit 0 ;;
  --build|'') ;;
  *) echo "install: unknown option; use --help" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "install: use --help" >&2; exit 2; }
[ -f "$ROOT/tools/backcluster/cli.py" ] || { echo "install: run this from a BackCluster checkout" >&2; exit 1; }
command -v python3 >/dev/null || { echo "install: Python 3.9+ is required" >&2; exit 1; }
python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' || { echo "install: Python 3.9+ is required" >&2; exit 1; }
mkdir -p "$PREFIX"
if [ -e "$PREFIX/backcluster" ] || [ -L "$PREFIX/backcluster" ]; then
  [ -L "$PREFIX/backcluster" ] && [ "$(readlink "$PREFIX/backcluster")" = "$ROOT/backcluster" ] || {
    echo "install: another backcluster command exists; choose BACKCLUSTER_BIN_DIR or remove it explicitly" >&2; exit 1;
  }
else
  ln -s "$ROOT/backcluster" "$PREFIX/backcluster"
fi
"$ROOT/backcluster" --version
if [ "${1:-}" = --build ]; then "$ROOT/backcluster" build; fi
echo "Installed. Keep this checkout in place; the command points to it."
case ":$PATH:" in
  *":$PREFIX:"*) ;;
  *) echo "Add the installation directory to PATH in your shell configuration." ;;
esac
echo "Next: backcluster doctor --human"

#!/usr/bin/env bash
# IPA=1 for an ad-hoc IPA; DEVELOPMENT_TEAM + UDID for signed device installation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export APP_BUNDLE_ID="${APP_BUNDLE_ID:-app.backcluster.sidecar}"
export APP_DISPLAY_NAME=BackCluster
export IPA_NAME=BackCluster
exec "${ROOT}/scripts/build-iphone.sh"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
if [[ -z ${DEVELOPMENT_TEAM:-} || -z ${UDID:-} ]]; then
  echo 'Set DEVELOPMENT_TEAM (Xcode > Settings > Accounts) and UDID.'
  echo 'Install separately for each device; connect and unlock it, enable Developer Mode.'
  xcrun devicectl list devices
  exit 1
fi
JOBS="${JOBS:-4}" scripts/build-iphone.sh
APP="$PWD/ios/build/Sidecar.xcarchive/Products/Applications/Sidecar.app"
# Explicitly require installation success (the upstream helper may exit 0 on a failure).
xcrun devicectl device install app --device "$UDID" "$APP"
BID=$(/usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$APP/Info.plist")
xcrun devicectl device process launch --device "$UDID" "$BID"

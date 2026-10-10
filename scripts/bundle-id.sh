#!/bin/bash
# bundle-id.sh [UDID] - print the installed BackCluster/Backburner bundle id on a wired device. Used by phone-up/phone-relaunch/phone-tail.
#
# Why: the id depends on how the app was installed. scripts/build-iphone.sh signs it as app.backburner.<your team id>;
# AltStore (docs/INSTALL-IPHONE.md) installs the release IPA under its own id (app.backburner.sidecar.<team id>).
#   1. BUNDLE_ID, if set
#   2. the installed BackCluster app (devicectl), including AltStore suffixes
#   3. an installed Backburner app for upstream compatibility
# Ambiguous installations require BUNDLE_ID. A team id alone does not identify the app.
set -u
[ -n "${BUNDLE_ID:-}" ] && { echo "$BUNDLE_ID"; exit 0; }

UDID=${1:-${UDID:-}}
if [ -z "$UDID" ]; then
  raw=$(ioreg -p IOUSB -w 0 -l 2>/dev/null | sed -n 's/.*"kUSBSerialNumberString" = "\(00008[0-9A-F]*\)".*/\1/p' | head -1)
  if [ ${#raw} -eq 24 ]; then UDID=${raw:0:8}-${raw:8}; fi
fi
[ -n "$UDID" ] || { echo "bundle-id: no iPhone or iPad on the USB cable" >&2; exit 1; }

J=$(mktemp "${TMPDIR:-/tmp}/backcluster-apps.XXXXXX")
trap 'rm -f "$J"' EXIT
ID=
for FLAGS in "" --include-default-apps; do   # developer-signed apps first (Xcode, AltStore), then every app
  perl -e 'alarm 40; exec @ARGV' xcrun devicectl device info apps --device "$UDID" $FLAGS --json-output "$J" -q >/dev/null 2>&1
  ID=$(python3 - "$J" <<'EOF' 2>/dev/null
import json, sys
apps = json.load(open(sys.argv[1]))['result']['apps']
cluster = {a['bundleIdentifier'] for a in apps if a.get('name') == 'BackCluster' or a.get('bundleIdentifier', '').startswith('app.backcluster.')}
upstream = {a['bundleIdentifier'] for a in apps if a.get('name') == 'Backburner' or a.get('bundleIdentifier', '').startswith('app.backburner.')}
ids = cluster or upstream
print(next(iter(ids)) if len(ids) == 1 else '')
EOF
)
  [ -n "$ID" ] && break
done
[ -n "$ID" ] || { echo "bundle-id: no unique BackCluster/Backburner app found on $UDID (unlock the phone; install: docs/INSTALL-IPHONE.md), or set BUNDLE_ID" >&2; exit 1; }
echo "$ID"

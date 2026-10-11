#!/bin/bash
# check-listeners.sh - static rules for the phone app's network surface (GitHub issue #2). Run by tests/security/run.sh and the
# pre-commit hook. Fails (exit 1) when:
#   - an accept() in the app's servers is not followed by a connection filter (CableOnly.h cable_accept / the servers'
#     accept_ok_ / accept_loopback_only) within a few lines
#   - ggml's RPC server is started on anything but 127.0.0.1 (it has no authentication; the cable gate fronts it)
#   - the app advertises itself (Bonjour: NetService, NSBonjourServices) or a Swift file binds 0.0.0.0
#   - the control port's fetch / path checks (fetch_url_ok, safe_doc_path) are gone
#   - the Wi-Fi tunnel accepts a connection without its handshake
#   - the app's and the Mac's copies of phone-attn.h differ
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
bad() { echo "check-listeners: $*" >&2; fail=1; }

SERVERS=(ios/Backburner/Sidecar/RPCBridge.mm phone-attn/phone-attn.h llama.cpp/ggml/src/ggml-metal/phone-attn.h
         llama.cpp/tools/split-prefill/tail-server.h)
if [ -n "${BACKCLUSTER_ENGINE_SOURCE:-}" ]; then
  SERVERS+=("${BACKCLUSTER_ENGINE_SOURCE}/tools/split-prefill/tail-server.h")
  cmp -s phone-attn/phone-attn.h "${BACKCLUSTER_ENGINE_SOURCE}/ggml/src/ggml-metal/phone-attn.h" || \
    bad "patched engine phone-attn.h differs from the mobile worker"
fi
for f in "${SERVERS[@]}"; do
  [ -f "$f" ] || { bad "$f is missing"; continue; }
  # every accept( call (not common_sampler_accept etc.) needs a filter within the next 12 lines
  while IFS=: read -r ln _; do
    win=$(sed -n "${ln},$((ln + 12))p" "$f")
    grep -qE 'cable_accept\(|accept_ok_|accept_loopback_only\(' <<< "$win" || bad "$f:$ln: accept() without a connection filter (CableOnly.h)"
  done < <(grep -nE '(^|[^_[:alnum:]])(::)?accept\(' "$f" | grep -vE '^\s*[0-9]+:\s*//')
done

grep -q 'ggml_backend_rpc_start_server' ios/Backburner/Sidecar/RPCBridge.mm && \
  ! grep -qE 'endpoint = "127\.0\.0\.1:"' ios/Backburner/Sidecar/RPCBridge.mm && bad "RPCBridge.mm: ggml RPC must listen on 127.0.0.1 only"

grep -rnE 'NetService\(|NWListener\(.*service|DNSServiceRegister' ios/Backburner/Sidecar --include='*.swift' --include='*.mm' --include='*.m' >/dev/null && \
  bad "the app advertises itself (Bonjour); the Mac finds the phone over the cable"
grep -q 'NSBonjourServices' ios/Backburner/Sidecar/Info.plist && bad "Info.plist: NSBonjourServices is back"
grep -rn '"0\.0\.0\.0"' ios/Backburner/Sidecar --include='*.swift' >/dev/null && bad "a Swift file binds 0.0.0.0"

cmp -s phone-attn/phone-attn.h llama.cpp/ggml/src/ggml-metal/phone-attn.h || \
  bad "phone-attn/phone-attn.h and llama.cpp/ggml/src/ggml-metal/phone-attn.h differ: the app and the Mac must speak the same protocol"
grep -q 'fetch_url_ok' ios/Backburner/Sidecar/RPCBridge.mm || bad "RPCBridge.mm: fetch no longer checks its URL (bb::fetch_url_ok)"
grep -q 'safe_doc_path' ios/Backburner/Sidecar/RPCBridge.mm || bad "RPCBridge.mm: file arguments are no longer checked (bb::safe_doc_path)"

if [ -f ios/Backburner/Sidecar/Tunnel.swift ]; then
  grep -q 'func serverHandshake' ios/Backburner/Sidecar/Tunnel.swift || bad "Tunnel.swift: the server handshake is gone"
  grep -qE 'serverHandshake\(' ios/Backburner/Sidecar/WifiTunnel.swift 2>/dev/null || bad "WifiTunnel.swift: Wi-Fi connections must pass serverHandshake before any byte is forwarded"
fi

[ $fail = 0 ] && echo "check-listeners: ok"
exit $fail

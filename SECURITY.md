# Security

## Reporting a problem

Please report security problems privately: on GitHub, **Security → Report a vulnerability** on this repository. Don't open a
public issue for them. Include the affected revision and a minimal reproduction without credentials or device identifiers.

## What the phone app exposes

The inherited worker app runs four servers on the phone: ggml RPC (50052), the prefill tail (50060), a control port (50061) and the
phone-held context (50062). Their protocols have no authentication of their own: whoever can connect can load models, write
files into the app's Documents, and read and write GPU memory. So the app decides who can connect:

- **The USB cable: always.** A connection is served only if it arrived on the wired interface to the Mac with both ends
  link-local (169.254.x.y or fe80::), checked right after it is accepted and before a single byte is read
  (`ios/Backburner/Sidecar/CableOnly.h`). Wi-Fi, cellular, AWDL, VPNs and routable addresses are dropped.
- **Wi-Fi: only once paired, and only encrypted.** `scripts/phone-wifi.sh pair` (phone on the cable) has the phone create a
  key, keep it in its Keychain and give it to the Mac. Over Wi-Fi the only open port is then the tunnel (50070): the Noise
  `NNpsk0` handshake with that key, then ChaCha20-Poly1305 for every byte (`Tunnel.swift`, checked against the published Noise
  test vectors). Without the key nothing gets past the first message; pairing itself works over the cable only.
  `scripts/phone-wifi.sh unpair` deletes the key and closes the port. Details: [docs/WIFI.md](docs/WIFI.md).
- **Nothing is advertised.** The app doesn't announce itself over Bonjour; the Mac finds it on the cable.
- **Files and downloads stay inside the app.** Control-port paths must be plain relative paths inside Documents, and `fetch`
  only downloads from the Mac's cable address.

On the Mac, `llama-server` listens on 127.0.0.1 only, `scripts/phone-push.py` serves exactly the files being pushed to the
phone's cable address, and the Wi-Fi tunnel's local ports are on 127.0.0.1.

Before 0.0.3 (the 0.0.2 pre-release and earlier) the servers accepted connections from every network the phone was on, and
the app advertised itself over Bonjour. Update the app; until then, turn Wi-Fi off on the phone while Backburner is open.

## Checking it

- `tests/security/run.sh` (no phone needed): the connection rules, the path and URL checks, the tunnel's crypto against the
  Noise test vectors plus tampering, replay and wrong-key cases, a lint that every listener is gated, and the commit checks
  below.
- `scripts/check-phone-exposure.py` (Backburner open on a phone on the cable, the Mac on the same Wi-Fi): from the Mac, every
  server answers over the cable, nothing is advertised, every raw port refuses Wi-Fi without answering, and the tunnel lets
  in only the paired key.

## Keeping private data out of the repository

`scripts/install-hooks.sh` installs pre-commit, commit-msg and pre-push checks (`scripts/hooks/check-sensitive.py`). They
refuse secrets and tokens, home-folder paths, device UDIDs, Apple team ids, device names, personal emails, LAN addresses,
model and binary files, signing material, files over 5 MB, and commits from a non-noreply email. Register additional private strings with `python3 scripts/hooks/check-sensitive.py --private-add`; only keyed fingerprints are stored in `.git/info/`, with the key in the macOS Keychain. `python3 scripts/hooks/check-sensitive.py --all` audits every tracked file.

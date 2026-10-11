# Contributing to BackCluster

Submit changes and device results to [art9762/backcluster](https://github.com/art9762/backcluster). See [UPSTREAM.md](UPSTREAM.md) for source provenance. Original licenses and attribution remain intact.

## Local workflow

1. Clone with `--recurse-submodules`. Keep the engine at the committed pin.
2. Install checks with `scripts/install-hooks.sh`.
3. Make a focused branch and add regression coverage for changed behavior.
4. Run `scripts/check.sh`. For engine changes, run `scripts/test-native-pipeline.sh`; for client integration, run `python3 scripts/check-opencode.py`. On an Apple Silicon Mac, also run `tests/security/run.sh`, build affected app/runtime targets, and run `scripts/test-metal.sh` against the Mac build.
5. Open a PR against `main`, explaining behavior, validation and remaining hardware gates.

The CLI uses Python 3.9+ and the standard library. Keep it usable offline. Engine compilation, conversion and Apple builds have separate prerequisites; see [PRE_HARDWARE.md](docs/backcluster/PRE_HARDWARE.md).

## Required evidence

- Model math changes need greedy-output and numerical comparisons with the same weights, quantization and settings. A successful build is not an inference test.
- Speed claims need hardware, command, model lock, launch receipt, raw repeats and before/after samples. Keep Mac-only working. Benchmark chunks are not token counts.
- Plans are estimates. Unknown memory/compatibility must fail visibly; losing unique state must not silently fall back to an incomplete session.
- Retain cable gates, loopback host endpoints and authenticated Wi-Fi tunnels. Networking changes also need Apple security and real-device exposure checks.
- Update working commands and limitations. Keep future architecture separate from implemented behavior.

## Private local files

Store reports and registries in ignored `.backcluster/`. Never commit credentials, signing data, device IDs, private addresses, weights or generated builds. Launch receipts contain local paths: review them before sharing. Do not bypass hooks.

Additional private strings can be registered with `python3 scripts/hooks/check-sensitive.py --private-add`. They are stored as fingerprints in `.git/info/`, with the key in the macOS Keychain; `scripts/install-hooks.sh` migrates the older plain-text format.

## Engine and licensing

`llama.cpp/` is a pinned submodule. Project-owned engine changes may be stored as verified patches in `engine-patches/`; update its manifest and validate a clean materialized source with the native gate. Before changing the submodule URL/gitlink, publish the engine commit in an accessible downstream fork. A clean recursive clone must retrieve the exact base and reproduce the patched source. Do not incidentally run `submodule update --remote`.

Contributions use MIT. Sign commits with `git commit -s` using a GitHub noreply address; sign-off follows the [Developer Certificate of Origin](https://developercertificate.org/). Preserve upstream copyright. See [SECURITY.md](SECURITY.md) for private reporting.

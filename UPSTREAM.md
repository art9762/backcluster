# Provenance and sources

Audit date: **2026-10-09**. The audit below records the planning baseline; subsequent implementation is documented in CHANGELOG.md. Target-device inference remains unmeasured.

## Reproducible source baseline

| Component | Repository | Pinned commit |
|---|---|---|
| Backburner | https://github.com/StayLameBro/backburner | `0d2bf5d7df342eb4da253b440dd6959eedbce7ff` |
| Engine submodule | https://github.com/StayLameBro/backburner-llama.cpp | `9f17fc31cf60753eb6bd6ec075894edab773cb91` |

The engine was materialized and inspected at the exact gitlink above. The published gitlink remains unchanged. Version 0.4 adds reviewed project-owned patches in `engine-patches/`, materialized on that exact base by `tools/backcluster/engine_overlay.py`; their hashes define the downstream build identity. `README.upstream.md` remains an exact copy of the parent README at the pinned commit. Original licenses and attribution are retained. Existing rows under `bench/results/` belong to upstream.

Immutable source bases for the paths in the audit:

- [Parent](https://github.com/StayLameBro/backburner/tree/0d2bf5d7df342eb4da253b440dd6959eedbce7ff)
- [Engine](https://github.com/StayLameBro/backburner-llama.cpp/tree/9f17fc31cf60753eb6bd6ec075894edab773cb91)

## External primary sources consulted

Read or searched on 2026-10-09. Mutable pages are discovery sources, not artifact locks.

1. [Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-27B)
2. [Official model configuration](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)
3. [GGUF publisher's file table](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF)
4. [DFlash2 publisher](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
5. [Apple: iPad Pro 13-inch M5](https://support.apple.com/en-us/125407)
6. [Apple: iPhone 17 Pro Max](https://support.apple.com/en-us/125091)
7. [Available app memory](https://developer.apple.com/documentation/os/os_proc_available_memory)
8. [Increased-memory entitlement](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit)
9. [Metal background restrictions](https://developer.apple.com/documentation/metal/preparing-your-metal-app-to-run-in-the-background)
10. [Continued-processing GPU resource](https://developer.apple.com/documentation/backgroundtasks/bgcontinuedprocessingtaskrequest/resources/gpu)
11. [OpenCode providers](https://opencode.ai/docs/providers)
12. [OpenCode v2 providers](https://opencode.ai/v2/docs/providers)

OpenCode's two documented configuration generations differ. Select the schema after identifying the installed client version.

## Model metadata snapshot

Repository revisions returned by the Hugging Face model API during the audit:

| Repository | Revision |
|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `bartowski/Qwen3.8-27B-GGUF` | `0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a` |
| `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |

Weights have not been downloaded or hash-verified. Before hardware comparisons, record revision, filename, exact byte count, SHA-256, GGUF architecture, chat-template hash and quantization layout. Derived shards also need the parent hash, conversion command and engine revision.

The GGUF publisher describes a changed tensor layout under familiar filenames. Names alone do not establish identity with the artifacts behind upstream's earlier benchmarks.

## Publication boundaries

Project files contain hardware classes and technical design, without device identifiers, signing material, private project contents or credentials. GitHub denied creation of a downstream engine fork with the available integration. Reproducible parent-owned patches preserve a publicly retrievable clean base instead of creating an unpublished gitlink. No issue, pull request or message has been sent to upstream.

## Downstream tooling

The control CLI and BackCluster app preparation are implemented on top of the engine pin above. `scripts/install-upstream.sh` retains the original upstream installer; `install.sh` installs the local BackCluster CLI. Original benchmark rows and README remain historical upstream artifacts. See CHANGELOG.md for downstream behavior and hardware limits.

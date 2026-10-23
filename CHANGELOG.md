# Changelog

## Unreleased

- Add explicit tool-result reconciliation after a crash, with immutable audit evidence and atomic SQLite v1 → v2 migration.
- Add `journal-call` and `journal-reconcile` commands; block reconciliation while a callback is active in another process and preserve at-most-once replay.
- Cover process death/reopen, request mismatch, audit corruption/write failure, schema migration and CLI input protection. Native server/agent integration remains separate work.

## 0.4.0 — offline cluster runtime

- Add reproducible, hashed downstream engine patches without changing the published gitlink.
- Implement native Qwen35 middle stage and three-stage compute, bounded framing, source/range/epoch checks and fail-closed recovery.
- Fix residual M-RoPE positions and honor explicit Flash Attention disable; compare full and split CPU logits/greedy with f32/f16 residuals.
- Guard BF16 narrow-tile Metal shaders on unsupported devices; require actual Metal backend initialization and graph execution in Apple CI.
- Add streaming host/middle/tail GGUF splitter, source-derived receipts, exact partition and transformed tensor verification.
- Launch explicit pairs, pipeline and dual remote KV with fresh cable routes, loaded artifact hashes, build identity and independent peak memory budgets.
- Compare private raw benchmark evidence; validate OpenAI SSE/JSON/reasoning and a fixed read/edit/test cycle.
- Generate local-only OpenCode config; verify it with the real OpenCode client.
- Add transactional BCTX socket coordinator, atomic complete checkpoints and durable tool journal with crash/unknown-state recovery tests.

Target-device inference, signing, RAM/page-ins, throughput and thermal acceptance remain unmeasured. BCTX checkpoint/journal integration with the native OpenAI server, weighted KV and distributed speculation remain separate implementation work.

## 0.3.0 — 2026-10-10

- Merge device diagnostics and pre-hardware preparation into main.
- Add readable doctor checks, suggested fixes and predictable build/cancellation errors.
- Validate model-lock identities consistently and reject duplicate JSON fields.
- Clamp host estimates to measured physical memory and model context limits.
- Count tool-only SSE output; normalize hashes across stream chunk boundaries.
- Isolate conversation caches by tools/template/settings; reject changed tool calls and incomplete responses.
- Install the BackCluster CLI from the current checkout; retain the original installer separately.
- Clarify mobile missing-model and foreground states, restore brightness on app switching and support iPad landscape.
- Refresh documentation, issue templates, CI and local checks.

Hardware acceptance remains pending. Three-stage inference and measured speedups are not claimed.

## 0.2.0 — pre-hardware preparation

- GGUF inspection, artifact locks and conservative planning.
- Explicit worker registry and bounded payload benchmarks.
- Pinned Mac build, loopback baseline and SSE measurements.
- BackCluster app identity, protocol metadata and Apple build/security CI.

## 0.1.0 — planning baseline

- Preserve upstream sources, history, licenses and engine pin.
- Define Apple trio architecture, memory analysis, roadmap and device runbook.

# Changelog

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

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 546. Serve reports ready while its model cannot fit, and refuses every request with a 500

**Lesson:** `brain serve` touches its ready file once its surfaces are bound, before the model it serves is known to
fit. With `--max-batch 128 --adapter-slots 24 --serve-adapters DIR` on one GH200, Qwen3-4B was sized at 97,164 MiB.
The log has one line, `[residency] UNSERVABLE Qwen/Qwen3-4B[default] ... exceeds every device's budget even with
nothing resident`, after `ready`, and every request to the base or to any of its adapters is then answered
`500 failed_to_load`. Measured 2026-11-21 (brain-bcd67e5ad): two such servers were ready for 40 minutes and answered
nothing else. A caller that trusts the ready file runs a whole exam against them (2,358 agent trials, every one a
0-turn failure). The same flags with `--adapter-slots 8` serve normally.

**Fix (serving-runtime G14, lane FIX, 2026-11-21):** before any surface binds, `brain serve` plans every served model's
every action at its default instance (`crates/cli/src/servable.rs`, reading only the residency plan's new `never_fits`
and `instance_model`; model-agnostic). A model whose every instance could never fit any device this process may use is
named with its size (`brain serve: UNSERVABLE <model> (instance ...) needs N MiB: ...`) and withdrawn, so it is never
listed or accepted. When such an instance carries more than one served model (adapters riding on their base), or when
nothing servable is left, the start is refused, exit 1, before the ready file. The message names the adapter bank and
`--max-batch` as part of the size; it does not compute which smaller slot count would fit (the size is the model's own
estimate). Spec test: `crates/cli/tests/it/serve_cli.rs`
`a_configuration_whose_adapter_cannot_fit_refuses_to_start_and_never_reports_ready` (a host-RAM ceiling of 1 MiB).

**How to apply:** a server that comes up ready now serves every model it lists; a refused start names the instance to
shrink (`--adapter-slots`, `--adapter-rank-cap`, `--max-batch`) or to move to another server.

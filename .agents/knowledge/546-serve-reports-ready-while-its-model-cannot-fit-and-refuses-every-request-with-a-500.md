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

**How to apply:** ready must mean servable. Size every configured model (its batch, KV and adapter bank) before the
ready file is written, and refuse at startup, by name, with the term that does not fit (serving-runtime gap G14).
Until then, a caller sends one real request per served model before it starts work.

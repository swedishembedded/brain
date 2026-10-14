<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 217. One resident copy serves the base and its adapters; the cost per adapter is a bank slot, not a weight copy

What was measured. Qwen3-0.6B, fp32 weights, context 2048, 8 decode slots, one Tesla
P40, whole-device memory read from `nvidia-smi` while `brain serve` answered
greedy chat requests for the base and two rank-8 adapters that correct all seven
linears (`Qwen/Qwen3-0.6B:lab:canary:v1`, `:v2`), three concurrent requests per
id per round. Each reading is the peak of the device's used memory, the device
being otherwise idle.

| server | peak device memory |
|---|---|
| base only, no adapter | 3447 MiB |
| base + 2 adapters, one resident (adapter bank: 4 slots, rank cap 16, all linears) | 3525 MiB |
| base + 2 adapters, before: each adapter folded into its own copy of the weights | 10320 MiB |

The bank adds 78 MiB over the base, which is `AdapterBankSettings::device_bytes`
plus the row buffers; the per-adapter-copy design added about 3.4 GiB per adapter
(a second KV pool, scratch and weight copy each). The saving grows with the base:
the bank scales with `slots x rank_cap x (k + n)`, a copy with the whole model.

What the replies showed. A request for an adapter id returned, byte for byte, the
message the folded-adapter server returned for the same adapter and request
(content and reasoning text), in a batch that also held base requests. A base
request on an otherwise idle server returned the same message on the base-only
server, the folded server and the mixed server.

A caveat that is NOT about adapters. Greedy decoding of one prompt is not
bit-stable across batch compositions even on a server with no adapter: two
concurrent identical base requests on the base-only server returned two different
messages (one per slot), and the mixed server's base replies fell in that same
set. The numerics of a batched GEMM differ from a single row's, and over 160
tokens an argmax near a tie flips. Compare replies at equal batch composition, or
by logits, never by "the base reply must equal the earlier base reply" under load.

Where it lives. `crates/catalog/src/adapter_registry.rs` (the registry and the
views), `crates/qwen3/src/serve/adapters.rs` (the bank),
`crates/catalog/tests/multi_adapter.rs` (the exactness-versus-folded gate, which
runs on the CPU backend in the test suite).

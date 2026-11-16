<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 493. Freed device memory becomes KV room only through the plan's charge, and the live tally by kind is the breakdown

**Lesson:** the planner sizes a serving engine's sequences and KV pool from what it charges, not from what the engine
allocates. Removing an allocation the plan never charged lowers the device's use and nothing else; removing one it
does charge (and lowering the charge with it) is what admits more sequences. Audit both columns: the plan's figure and
the device's, per batch, and treat the gap between them as either waste the plan does not see or an under-charge that
can fail an allocation on a fuller card. The engine's own breakdown by kind is `Gpu::live_tally` (device, pinned,
system, pool, managed, sparse bytes); the CUDA backend tallies no labels or call sites.

## What happened

A device-memory audit of the dense 27B (HF FP8, `Qwen35GgufResident`) and the 35B-A3B MoE (`qwen35moe`) on a GH200
(`perf run longctx --context 32768 --ladder 1,512`, base/new interleaved):

| item | plan | device used | admitted at 32K |
|---|---|---|---|
| 35B-A3B embedding table fp32 -> int8 (`embed_i8`, charged at 9/8 bytes) | 42.1 -> 40.7 GiB | 41.0 -> 39.6 GiB | 39 -> 40 |
| 27B model instance's own decode state made on first use (never charged) | 29.2 GiB | 33.5 -> 33.3 GiB | 13 |
| 27B host prefix-cache snapshot rows backed as the cache fills (host, never charged) | - | host 4788 -> 1197 MiB | 13 |

The 27B's device use at batch 1 stays 4.1 GiB over its plan (CUDA context and library workspaces included); the MoE's
is 1.1 GiB under.

## What to do

- Before calling an allocation waste, find its readers; before counting its removal as capacity, find its charge
  (`serving_cost` / `layer_state_bytes` for the 27B, `qwen35moe::serve::weight_bytes` and `activation_bytes` for the
  MoE) and move both together.
- A table the checkpoint stores int8 (a GGUF Q8_0 embedding) is held int8 and gathered exactly (`Ops::embed` over a
  `Weight::I8`, `model::int8::quantize_from`'s byte repack): fp32 expansions of stored tensors are the largest single
  waste found.
- Host memory a cache fills over a server's life is backed in chunks off the serving thread
  (`qwen35::int8_gguf_resident::HostSnapshots`), never all at load.
- Left (ledgers `qwen35.md`, `qwen35moe.md`): the 27B's verify scratch held while speculation is off, its three tape
  scratch sets, the MoE's whole-context KV reservation per sequence.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 498. A Qwen3 pool sized without the card and a tied head charged once left a 32K-row round nowhere to go

**Lesson:** a KV pool is whatever the card holds beside everything else, so every builder of an engine sizes it through
the planner, and the planner's charges are the engine's own: one function next to the allocation for the activations
(held to the engine's peak by a census on the real checkpoint), and a weight charge that counts what the engine
uploads, not what the checkpoint stores. A tied head is the embedding table twice.

## What happened

`brain perf run serve --target qwen:<Qwen3-4B> --concurrency 16 --input 32768 --output 128` on an empty GH200 (gpu3)
exited 101: `device out of memory allocating 99614720 bytes for storage`. The allocation was a weight upload
(`Engine::build` -> `upload_fused_from_source` -> `DeviceWeightSink::begin`): the engine had already allocated its KV
pool and its scratch, and the weights did not fit after them. The same shape at 8K prompts served; the 27B and the MoE,
which build through their planners, served 16 x 32K.

## Why

Five charges were missing or wrong, each found by the census `engine_activation_charge` (the engine handle's peak live
bytes less its KV pool and fp32 tables, after a prefill of two widest rounds and a decode step of every row) or by the
planner's arithmetic against the card:

- **The `qwen:` perf target sized its pool from the workload alone:** `--max-batch` (32) whole contexts of 32904
  tokens, 65856 blocks, about 80 GB of int8 KV, with no card in the sum. Only the weight tier came from the serving
  plan (`be31bb33e`), not the pool.
- **The fused decode's partial records were sized for the widest round,** `max(max_batch, max_prefill)` = 32768 rows,
  where a decode step carries at most `max_batch` rows (the engine now asserts it); the build also planned all 32768
  row counts to find the largest.
- **The planner charged a width formula for the step** (`DecoderWidths::serve_peak_bytes`, 43K words a row for the
  4B) where the engine's fixed scratch is 72K words a row and a running round adds a layer's temporaries (the bf16
  operands, about one fp32 word a row per column of the query and feed-forward widths) and the recorded decode steps
  keep every layer's (knowledge 487).
- **Every weight formula counted a tied head once, at the head's tier.** The engine gathers the embedding from an fp32
  table and uploads the head from it again at the head's tier: Qwen3-4B bf16 holds 9.60 GB, the plan charged 8.05 GB.
  With the pool fitted to the card by that plan, every 32K round's 608 MiB SwiGLU operand was refused and the
  scheduler, correctly, refused all 16 requests after three failures each alone.
- **`brain serve`'s auto context charged the never-fused attention slab on a GPU's budget.** At 32 heads and 512-row
  rounds the slab of a 32768-token context is past one binding, so a 96 GB card gave Qwen3-4B 24576 tokens and
  refused every 32K prompt (`needs 33498 tokens, engine capacity is 24576`). The budget is always a GPU's
  (`serving::executor` fills it from the cards), and every GPU backend fuses that attention.

| Qwen3-4B, bf16, int8 KV | held beyond pool and tables | charged |
|---|---|---|
| 16 sequences, 512-row rounds | 0.279 GiB | 0.284 GiB |
| 16 sequences, 4096-row rounds | 1.524 GiB | 1.543 GiB |
| 32 sequences, 16384-row rounds | 5.581 GiB | 5.711 GiB |

## What to do

- `qwen3::serve::activation_bytes` (attention scratch plus `step_activation_bytes`) is the one activation charge; the
  catalog plan and `catalog::resident_llm::kv_blocks_within` (any engine sized from a workload, the perf target first)
  take it. Re-run `engine_activation_charge` on the real checkpoint when the round's kernels or the scratch change.
- `qwen3::footprint::held_weight_bytes_by` is the one weight charge (the catalog's `weights_bytes_at` and the tier
  default call it); it counts a tied head beside the fp32 embedding.
- A step's device failure is the scheduler's (knowledge 435): `Gpu::set_device_cap` refuses an allocation past a
  handle's cap exactly as a full card does, so `engine_device_oom` runs a real Qwen3 engine out of memory at a chosen
  step and checks the sequences come back with their tokens, and that a request the card cannot hold alone is refused
  with the retryable reason while the engine serves on. `DeviceOutOfMemory::raise` prints the allocation's backtrace
  under `RUST_BACKTRACE` (the unwind skips the panic hook, which is how this one hid its site).
- Left: `brain serve` reads a Qwen3 prompt in rounds of at most 512 rows (`QwenServeConfig::max_prefill_cap`, kept
  for the unfused slab), where the perf target reads it in rounds of the whole prompt; the qwen3 ledger has the gap.

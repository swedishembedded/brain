<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 476. A prefill built as one tape held every layer's temporaries at once, so the pool grew by the whole stack and gave it back before every prompt

**Lesson:** holding freed blocks (knowledge 474) only helps when the working set that is live at once fits what the
allocator may keep. A forward recorded as one step list keeps every temporary it names alive until the list is
submitted and dropped, so its live set is the sum over all layers, not one layer's. Submit per layer and let a replay
arena hand a layer's blocks to the next; and read back only the rows the caller uses.

## What happened

Serving Qwen3-4B (bf16 weights, int8 KV) at `perf run serve --concurrency 16 --input 8192 --output 128 --requests 32`
on a GH200 left the card 79 % busy inside the serving window, with 1.97 s of gaps over 1 ms. Two kinds of gap made up
nearly all of it (nsys with CPU sampling):

- **Before every prompt's first kernel** (after the previous decode step's `argmax_final`, 39 gaps, 1.25 s): the host
  was inside `cuMemAllocFromPoolAsync`, mostly in the kernel mapping and clearing pages. `Engine::batched_tape` built
  a whole prefill chunk (one 8K-row chunk, 36 layers) before submitting any of it, and every bf16 library product
  rounds its activation to a fresh scratch (`cast_bf16`), four per layer, 0.04 to 0.16 GB each at 8K rows: several GB
  live at once. The pool's release threshold (the retention cap) is far below that, so every synchronise between
  prompts returned the growth to the driver and the next prompt mapped it again (11.6K pool allocations, mean 107 us,
  max 65 ms in the window). This is why raising the retention to `serve_peak_bytes` did not help: that bound is one
  forward's widths, not 36 layers of them.
- **After every prompt's last layer** (34 gaps, 17-220 ms each): `prefill_mixed` read the chunk's whole final-norm
  block (`[8192, 2560]` fp32, 80 MB) into a fresh host `Vec` to keep one row. The copy engine took under 1 ms; the rest
  was the kernel faulting in the pageable destination inside `cuStreamSynchronize`.

## What to do

- An unrecorded forward of the qwen3 engine submits each layer as it is built, inside
  `Gpu::scratch_scope_in(PREFILL_LAYER_ARENA)` and a `pass_scope` (`Engine::batched_steps`), as qwen35moe's prefill
  round does; a backend whose submissions are not ordered drains between layers. The card starts on layer 0 while the
  host builds layer 1, and a steady-state prefill takes no block from the pool.
- `prefill_mixed` reads only the prompt's last row, with `Gpu::read_into` at its offset, and does not read between
  chunks at all (except under latent sparse attention, whose chunk set-up reads device state).
- Gate: `crates/qwen3/tests/it/engine_prefill_census.rs` asserts that a cold multi-chunk prefill repeated on one engine
  calls neither `cuMemAlloc` nor the pool (`CallTotals::pool_alloc_calls`); it fails before the change (32 pool
  allocations a prefill of the test model) and passes after.
- When a temporary's size depends on the row count, ask what is live at once over the whole submission, not per
  call: per-layer allocation counts look fine while the live set is the stack.

## Evidence

Same command, gpu3 of a GH200 node, `--warmup 2 --seed 1`, no profiler for the client numbers, nsys
`-t cuda,nvtx --cuda-graph-trace=node` for the busy share (union of kernels, copies and memsets inside the `serve.*`
NVTX window):

| | before | after |
|---|---|---|
| c=16: req/s, output tok/s | 2.42, 310 | 2.75, 352 |
| c=16: TTFT p50 / p95 | 604 / 5070 ms | 524 / 4262 ms |
| c=16: card busy, gaps over 1 ms | 79.2 %, 1.97 s | 90.5 %, 0.21 s |
| c=64 (128 requests): req/s, TTFT p50 / p95 | 2.66, 12.1 / 20.1 s | 3.08, 10.5 / 17.2 s |

The perf harness reports both runs valid. `serve_conformance`, `engine_exact_decode`, `engine_half_weights`,
`engine_prefill_mixed`, `engine_prefill_hd128`, `lsa_engine`, `engine_adapter_bank` and `engine_variants` pass on
CUDA before and after; `batched_prefill::batched_prefill_matches_per_token_decode` fails identically before and after
(the `Qwen` model path, int8 weights, 1.16e-3 against a 1e-3 bound), which this change does not touch.

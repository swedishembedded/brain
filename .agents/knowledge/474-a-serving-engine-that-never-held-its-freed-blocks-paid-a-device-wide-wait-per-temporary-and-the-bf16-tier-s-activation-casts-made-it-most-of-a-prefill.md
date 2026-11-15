<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 474. A serving engine that never held its freed blocks paid a device-wide wait per temporary, and the bf16 tier's activation casts made it most of a prefill

**Lesson:** a CUDA context frees exactly as the driver would until a handle asks it to hold freed blocks, and a
`cuMemFree` waits for the whole device. Every serving engine that allocates per step must hold its freed blocks on
every handle it allocates from; the qwen35 and qwen35moe engines did (the latter only since its own prefill lane),
the qwen3 paged engine never did. A new weight tier that adds a temporary per product turns that gap into the
dominant cost, while the per-kernel table shows nothing wrong.

## What happened

Serving Qwen3-4B at its stored bf16 tier (knowledge 468) made decode faster but prefill slower than the fp32 tier:
`perf run longctx --target qwen:<Qwen3-4B> --context 32768 --prefill 4096,16384` on a GH200 gave 12.8K and 11.4K
tok/s at bf16 against 19.5K and 14.9K at fp32. An nsys timeline of the 16K-token prefill showed 0.52 s of kernels in a
1.16 s prefill; CPU samples inside the idle gaps were in `cuMemAlloc_v2` and `cuMemFree_v2` under
`LibraryGemmProvider::product`: each library GEMM rounds its fp32 activations to a fresh bf16 scratch
(`cast_bf16`), so every linear of every chunk allocated from and freed to the driver. The fp32 tier has no such
temporary, which is why it looked fine.

## What to do

- The qwen3 engine's handle holds its freed blocks from construction (`Engine::build`), under the backend's
  retention bound; a prefill chunk's temporaries are reissued.
- A model's serving engine gets a call census of a steady-state prefill (no `device_alloc_calls` growth across
  repeated prefills of one prompt), as `crates/qwen3/tests/it/engine_prefill_census.rs` and qwen35moe's
  `prefill_round_census.rs` do. Check it after any change that adds a temporary to a product.
- When a tier change moves end-to-end numbers the wrong way, compare the kernel sum of a pass with its wall time
  before reading kernel tables.

## Evidence

`crates/qwen3/tests/it/engine_prefill_census.rs` fails before the change (a repeated bf16 prefill calls `cuMemAlloc`)
and passes after. Same command, gpu3 of a GH200 node, int8 KV, context 32768: bf16 prefill 4096 -> 26.8K tok/s (from
12.8K), 16384 -> 19.4K (from 11.4K); fp32 19.1K / 16.8K; decode unchanged (batch 1 5.62 ms, batch 16 916 tok/s).

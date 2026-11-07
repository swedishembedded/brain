<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 354. The streamed trainer pinned its LoRA products to the fp32 reference and left the fast kernels unregistered

**Lesson:** A second trainer over the same blocks must take its kernel ids and matrix-product
provider from the same seams as the first. Otherwise its adapter products stay on the slow
reference path, and step time on a contended card is misattributed to the device.

## What happened
Question: why did the 27B streamed LoRA step spend 5-7 s in compute on an idle GH200 when the
base-weight GEMMs take 0.4 s?

## Why
`stream_train` passed `lora.rt.reference_gemm()` (the fp32 reference provider, never tensor cores)
for every adapter product, used the materialised `[H,T,T]` attention chain, and
`stream::kernel_ids` / `gdn_mixer_ids` left `rmsnorm_rows`, `rmsnorm_dx_rows` and the GDN fast
prefill kernels unregistered. The resident trainer registers all of them.

On an idle card the per-kernel table (runs/qstream/profile.md) showed:
- fp32 `matmul` 0.76 s;
- `bmm` 0.59 s;
- `rmsnorm` 0.34-0.59 s;
- `gqa_bwd_dscores` 0.29 s;
- against 0.36 s of library GEMMs for the whole frozen base.

The earlier "compute 26-30 s" was contention.

## What to do
- LoRA products go through `ops.tensor_gemm()`.
- Attention goes through `gqa_mixer_fwd_tc` (flash trio / library path from
  `FlashGqaIds::MIN_TRAIN_T`).
- The shared id sets register the coalesced norms and GDN fast kernels when the pipeline set has
  them.

Compute fell to 2.1-2.8 s with the loss unchanged to three digits.

Rule: a second trainer over the same blocks takes its kernel ids and GEMM provider from the same
seams as the resident one; profile an idle card before attributing step time to the device.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 354. The streamed trainer pinned its LoRA products to the fp32 reference and left the fast kernels unregistered

Question: why did the 27B streamed LoRA step spend 5-7 s in compute on an idle GH200 when the base-weight GEMMs take 0.4 s?

Cause: `stream_train` passed `lora.rt.reference_gemm()` (the fp32 reference provider, never tensor cores) for every adapter
product, used the materialised `[H,T,T]` attention chain, and `stream::kernel_ids` / `gdn_mixer_ids` left `rmsnorm_rows`,
`rmsnorm_dx_rows` and the GDN fast prefill kernels unregistered, while the resident trainer registers all of them. On an idle
card the per-kernel table (runs/qstream/profile.md) showed fp32 `matmul` 0.76 s, `bmm` 0.59 s, `rmsnorm` 0.34-0.59 s and
`gqa_bwd_dscores` 0.29 s against 0.36 s of library GEMMs for the whole frozen base; the earlier "compute 26-30 s" was contention.

Fix: LoRA products go through `ops.tensor_gemm()`, attention through `gqa_mixer_fwd_tc` (flash trio / library path from
`FlashGqaIds::MIN_TRAIN_T`), and the shared id sets register the coalesced norms and GDN fast kernels when the pipeline set has
them. Compute fell to 2.1-2.8 s with the loss unchanged to three digits. Rule: a second trainer over the same blocks takes its
kernel ids and GEMM provider from the same seams as the resident one; profile an idle card before attributing step time to the
device.

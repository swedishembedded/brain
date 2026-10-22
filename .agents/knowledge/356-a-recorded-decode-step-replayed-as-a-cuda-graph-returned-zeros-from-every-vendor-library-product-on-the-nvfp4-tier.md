<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 356. A recorded decode step replayed as a CUDA graph returned zeros from every vendor-library product on the NVFP4 tier

Question: why did the serving engine give garbage (and, in the tiny fixture, token 0 after the first token) for
nvidia/Qwen3.6-35B-A3B-NVFP4 while the same engine with decode tapes off, and the single-sequence path, were correct?

Cause: a tape replayed as a graph returned all-zero results for the cuBLAS products inside it (the TF32 router product
`router_ext`, the bf16 tile product the LM head takes at more than two rows). The int8 tier never showed it: its mixers and
head are DP4A kernels, and its router product happens to survive. Found by dumping one layer's `logits`, `ids`, `weight`, `y`
and the head's logits after a replay: the router logits and the head logits were exactly 0.0 while the activations feeding
them were not. The same recording with the library declined (`Gpu::set_declines_library`) was bit-correct, and the tiny
fixture test only failed at `top_k >= 4`, which is why the first bisection (fusion, native kernels, mixers, head) all
exonerated their suspects.

Fix: `Qwen35::record_decode` declines the library on the NVFP4 tier while it records, and `Ops::small_m_gemv` then serves
the stored-format GEMVs up to `GEMV_MAX_ROWS` (the tile product that took more than two rows is a library product); the
engine does not record a batch taller than that. Rule: when a recorded step and the same step built eagerly disagree, dump the
device buffers of one layer after the replay before suspecting the kernels; and a tier that is recorded holds in-house
kernels only until the library-in-graph defect is understood.

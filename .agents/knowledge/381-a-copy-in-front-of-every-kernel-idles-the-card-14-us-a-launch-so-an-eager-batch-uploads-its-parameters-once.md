<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 381. A copy in front of every kernel idles the card 14 us a launch, so an eager batch uploads its parameters once

Question: knowledge 380 left the mixed iteration at 39-66 % card busy with ~5000 launches and ~4500 `cuMemcpyHtoDAsync`. Is the
cost the host's API calls or what the device does between kernels?

Measured (gpu1, FP8 Qwen3.8-27B, `--max-batch 64 --kv-cache int8 --speculative off`, chat 1K/128 c=16, nsys cuda+nvtx, the
median `serve.mixed` range, memcpy/kernel/memset activity ordered on the one stream):

1. **Every WGSL-catalogue kernel was preceded by its own parameter copy** (4545 copies, 4710 kernels). The gap kernel -> copy was 6.6 us,
   the copy 1.8 us, copy -> kernel 5.7 us: 14 us of idle card per kernel, 61 of 234 ms; kernel -> kernel with nothing between
   was 0.9 us. The copy is a stream operation, so the kernel before it must finish and the next may not start until it has - the cost is on
   the device, not only in the 2.4 us host call. Per-shape uniforms are needed for graph capture's stable addresses, not for eager issue.
2. **Eager issue now packs the parameters of a batch** (`CudaBackend::issue_eager`): one image per 16K words, one copy into one
   device block, each kernel's uniform argument pointing at its slice. The stream orders the next batch's copy after the last kernel
   of the previous one, so one block serves every batch. The model batches by wrapping each layer of a prefill/mixed round in
   `pass_scope` (`advance_layer_stack`): the layer's ~100 dispatches are held and issued together (a decode step already is one pass).
3. Result (nsys, median mixed iteration): wall 234 -> 162 ms, card busy 66 -> 79 % (82 % of the iterations that carry a slice),
   copies 4545 -> 480, kernel -> kernel gap 0.4 us; total mixed time 27.9 -> 20.8 s for the same run. Client output tok/s at c = 1 / 4 / 16 / 64:
   47 / 192 / 291 / 384 -> 49 / 200 / 340 / 460 (vLLM 79 / 299 / 759 / 1241); ITL p99 at 16 292 -> 238 ms; TTFT p50 at 16 1.29 -> 1.05 s,
   at 64 2.99 -> 1.00 s; 31K-prompt TTFT 7.28 -> 7.10 s.

Left in that iteration (nsys, per median mixed iteration): ~5400 launches (15 us host each under the profiler), ~5900 pool alloc +
5100 free, ~2750 event create/record/destroy (cross-handle fences between the model's and the ops handle), ~690 memsets, 1008 `concat_split`
row copies, 26800 `cuCtxSetCurrent`. These are the step plan's job (serving-runtime.md "step plan and replay").

Quality: `qwen35_fp8_parity` (2 prompts, 851 + 845 tokens, 32 steps) is identical before and after: agreement 98.44 % (one near-tie flip at margin 0.074),
mean KL 1.526e-3, worst rel-L2 7.45e-2. Card tests: `gguf_mixed_step` 3/3, `gguf_conformance` 12/13 (`prompts_sharing_a_document_resume_from_where_they_part`
fails on main too), `hf_fp8_conformance`, qwen35 lib 156 on CPU. `brain-backend-cuda` tests: `catalogue_coverage` (rdft ReverseBits), `mem_pool`
and `staged_writes` fail on the commit before this change too.

Increment B1 (same note): the zero-fresh clears and per-submission events. nsys of the median mixed iteration, memsets 930 -> 20 (the serving
GDN fold cleared eighteen scratch buffers of which its kernels use five, plus the chunk-major query/key, end state and output; attention
context, the fused decode output and the uploaded index words were cleared before being written in full), `cuEventCreate`/`Destroy` 3100/3000 -> 0
(ordering events are re-recorded once no buffer fence names them), allocs 6950 -> 5450. Card busy 79 -> 81 %, median iteration 166 -> 158 ms; client
output tok/s at c = 1 / 4 / 16 / 64 49 / 200 / 340 / 460 -> 49 / 200 / 337 / 465, ITL p99 at 16 238 -> 231 ms, 31K TTFT 7.10 -> 7.03 s (within noise except
the API counts). What bounds the iteration now (nsys, CPU sampling): 15 ms of host work before the first kernel (tables, index uploads, 15 % of the samples in
`madvise` from large host frees) and a 7 ms host gap after the last layer, together 22 of 158 ms, plus 5400 launches at 15 us host cost each.
Card tests in a debug build (0xFF-filled uninit): `gdn_chunk_fwd_train_tc`, `gguf_mixed_step` pass; `gguf_conformance` 12/13 as before.

Increment B2 (same note): the host work in front of and behind the forward, found by symbolizing an nsys CPU profile of a frame-pointer
release build (`CARGO_PROFILE_RELEASE_STRIP=none`, `-C force-frame-pointers=yes`; the IP base recovered from the return addresses, which
must follow a `bl`). Two costs, both host-side and both in brain:
1. **15 ms before the first kernel was the embedding lookup.** `ResidentCheckpoint::embed_rows` read each row of the FP8 table from the
   mapped file one after the other and `madvise(DONTNEED)`-ed the pages after every row (a 64 KiB page and a fault per row, 117 of 193
   `forward_mixed` samples). Point reads now keep their pages (`PackedSource::read_range`; the streaming tile reads still release) and the
   rows decode in parallel (`PackedSource::read_rows`).
2. **The 7 ms gap behind the last layer was `release_dead_uniforms`.** Every recorded step took a per-shape device allocation for its
   parameters (keyed on kernel and buffers, so fresh buffers made a fresh key each round) and the next drain freed the dead ones with a
   synchronising `cuMemFree`. Since increment A no path reads that allocation: an eager batch copies parameters into the parameter block, a
   graph into its own. A step carrying parameters now allocates nothing when recorded (`Resolved::owns_params`).
Median mixed iteration (nsys): 158 -> 144.5 ms, card busy 81 -> 90 % (all iterations 89 %), pool allocs 5450 -> 2140, memsets 20 -> 138
(kept by other paths), API calls 38000 -> 27000. Client output tok/s at c = 1 / 4 / 16 / 64: 49 / 200 / 337 / 465 -> 50 / 203 / 360 / 500; TTFT p50 at 16
1.05 -> 0.95 s; 31K-prompt TTFT 7.03 -> 6.64 s. Parity identical (98.44 %, KL 1.526e-3).

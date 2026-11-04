<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# lsa

Scope: Latent sparse attention - the semantic key index, selection-driven sparse attention over the paged KV cache, and their training (`model::lsa`, `model::paged_sparse`).

The method is Kubicek et al., arXiv:2610.42147; the authors' reference
implementation (Apache-2.0) is the oracle, attributed in `/NOTICE.md`. LSA is a
shared subsystem of `crates/model`: a model supplies its attention heads and
head layout; clustering, selection and the sparse read are written once.

## Done

- [x] Host reference (`model::lsa`): validated config (paper defaults), index
      space from GQA heads with a model-declared rotary layout, Ward merging,
      the append-only bucket index (closed blocks immutable, open tail
      truncatable), the flat read (top-p, expansion, window, sinks, early
      budget) and the attention oracle over a selection. Gated bit-exact on
      bucket membership and selected keys against the reference's own output
      (`tools/goldens/lsa_dump_reference.py` -> `testdata/lsa/`); a budget
      past the context reproduces dense causal attention.
- [x] Sparse attention over the paged KV cache (`paged_sparse_flash_decode`,
      `model::paged_sparse`): each query row reads an explicit list of logical
      positions through its block table, f32 or bf16 pools. Selecting every
      key in order reproduces `paged_flash_decode` bit for bit; an LSA
      selection matches the host oracle; a position past the sequence's live
      keys is refused when packing and masked by the kernel (wgpu and Vulkan).
- [x] Prefill-shaped sparse kernel `paged_sparse_flash_gqa`: one workgroup per
      (row, KV head) serves the whole query-head group from each staged row,
      16-key tiles, no redundant P.V work; matches the per-head kernel at
      groups 1, 2, 4, 8. Measured one Qwen3-0.6B-shaped layer on a P40 (wgpu,
      f32 KV, `crates/model/tests/lsa_prefill_bench.rs`, k=2048, w=512):
      dense `paged_flash_prefill` 45 / 143 / 569 / 2249 / 9011 ms at
      4k / 8k / 16k / 32k / 64k tokens against sparse 556 / 1371 / 2957 /
      6312 / 12869 ms - linear against quadratic, but the per-query gather
      runs at ~155 GB/s and only crosses dense past 64k. The decode-shaped
      per-head kernel was 1.6-1.8x slower still.

## Not yet done

- [ ] Sparse prefill bandwidth: bf16 KV tier for the sparse read, the
      shared window as a dense sliding tile (it is a fifth of every
      selection), and more bytes in flight per workgroup.
- [ ] Device indexing: index-key projection, pooling, Ward and centroid
      scoring, top-p and expansion as WGSL kernels gated against the host
      reference; selection stays device-resident so decode needs no readback.
- [ ] Serving: per-sequence, per-layer `BucketIndex` in the paged decoders'
      lifecycle (fork shares closed blocks, release, prefix cache, speculative
      rollback within the open block), memory accounted in `memplan`, a flag
      to turn it on per model, metrics for selected-key counts.
- [ ] A model adapter: the first decoder (Qwen3 GQA) supplying pre-RoPE Q/K
      and its rotary layout; dense attention stays the default.
- [ ] D-LSA: PCA initialisation, loading the released distilled maps, the
      distillation objective (forward KL against head-averaged teacher
      attention, top-2048 targets) on the generic trainer.
- [ ] N-LSA: sparse attention backward (selection inverted per key group, no
      atomics) and its gradient check with the selection held fixed.
- [ ] Tree search (node-bucket levels and beam descent) after flat accuracy
      is proven at long context.
- [ ] Quantized index storage (bf16 centroids, fp8 tree nodes) gated on
      retrieval agreement separately from KV-tier numerical error.

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

## Not yet done

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

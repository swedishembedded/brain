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
- [x] Device indexing (`model::lsa::device`): `lsa_index_vectors` turns
      post-RoPE heads (a chunk's queries or a cached key's pool slot) into
      unit index vectors (un-rotate with the forward kernels' own angle, sum
      each KV group, project, normalise); `lsa_read` scores a query's visible
      paged centroids, radix-selects the top-p, sorts and expands them, and
      writes the sparse attention rows, reading positions from the live
      lengths so a device-advanced decode window needs no host write. A
      lagging page table widens the window instead of dropping keys. Gated
      against the host index space and `read_scores` (exact, scrambled pages,
      lagging rows, early budget).
- [x] Engine machinery (`model::lsa::engine`, `model::lsa::indexer`): an
      `Indexer` artifact (per-layer query/key projections, configuration and
      the model geometry it is valid for; random and PCA initialisations;
      safetensors with a strict header) and `LsaAttention`, which records a
      layer's sparse attention as three shape-only dispatches and keeps each
      sequence's index pages up to date from the cached keys
      (`BlockTable::index_pages`). Pages are keyed by cluster block and KV
      block chain, so a shared prefix shares its pages; truncation and
      release free them. Gated: device context equals the oracle over the
      device's selection, page sharing and freeing, artifact round trip.
- [ ] Device Ward: clustering of a closed block still runs on the host
      (`cluster_block`), one readback per closed block per layer.
- [x] Qwen3 serving engine (`Engine::set_lsa`): the covered layers record
      the sparse path; prefill chunks are cut at cluster boundaries and every
      forward (prefill, scoring, decode windows, speculative verify) brings
      its sequences' pages up to date first; release frees them. Gates on the
      real Qwen3-0.6B: a budget past the context generates exactly the dense
      continuation (pages built mid-decode), last-hidden relative error
      1.2e-6 at that budget against 0.29 at 8 keys; on the tiny model the
      scheduler conformance suite passes with LSA on (batched = single,
      prefix hits prefill only the suffix and report their cached tokens,
      speculation = greedy). Refused on int8/fp8 KV and under YaRN.
- [ ] Serving surface: a policy flag to attach an indexer, memory accounted
      in `memplan`, selected-key metrics.
- [x] Packed (int8/fp8) KV, the served default: `paged_sparse_flash_gqa_i8`
      (per KV head, decode) and `paged_sparse_flash_rows_i8` (one workgroup
      per query row and every head: each selected key resolved once and its
      whole packed K/V row staged in one coalesced sweep - the prefill
      read), `lsa_index_keys_i8` building pages from the packed pool; fp8 by
      the shared decode rewrite. Selection rows carry their header, freeing
      the bindings. Gates: packed kernels bit-identical to f32 over the
      dequantised pool, whole-row within float rounding of the per-KV-head
      kernel at 16/8/128, 32/8/128, 8/2/64; real Qwen3-0.6B over int8 KV at
      a full budget is as close to f32 as dense int8 (4.6e-2 vs 4.5e-2);
      conformance with sparse attention over int8 KV. One layer at the
      0.6B shape, int8 KV, k=1024, w=512, P40 shared with another GPU job
      (indicative): dense `paged_flash_prefill_i8` 2238 / 10284 ms at
      32k / 64k against whole-row sparse 787 / 1610 ms (2.8x / 6.4x);
      crossover near 12k.
- [ ] YaRN: the index-vector kernel un-rotates plain RoPE only.
- [ ] D-LSA: PCA initialisation, loading the released distilled maps, the
      distillation objective (forward KL against head-averaged teacher
      attention, top-2048 targets) on the generic trainer.
- [ ] N-LSA: sparse attention backward (selection inverted per key group, no
      atomics) and its gradient check with the selection held fixed.
- [ ] Tree search (node-bucket levels and beam descent) after flat accuracy
      is proven at long context.
- [ ] Quantized index storage (bf16 centroids, fp8 tree nodes) gated on
      retrieval agreement separately from KV-tier numerical error.

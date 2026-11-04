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

- [x] End-to-end prefill, Qwen3-0.6B, every layer sparse, int8 KV (the
      served default), one sequence of another repository's documentation, 512-row chunks,
      P40 (wgpu, no other job on the card), the distilled indexer at top_k
      1024, top_p 256, window 512 (`qwen_lsa prefill --kv-cache int8`):
      | T | dense | sparse | speedup |
      |---|---|---|---|
      | 2048 | 0.85 s | 0.85 s | 0.99x |
      | 4096 | 2.35 s | 2.35 s | 1.00x |
      | 8192 | 7.34 s | 6.96 s | 1.05x |
      | 16384 | 25.07 s | 16.59 s | 1.51x |
      | 32768 | 92.61 s | 37.21 s | 2.49x |
      | 65536 | 357.41 s | 83.59 s | 4.28x |
      Dense throughput falls 2414 -> 183 tok/s (quadratic); sparse stays
      near 800-1200 tok/s past the crossover (linear). The engine measured
      its crossover at 5120 (chunks ending there or below read densely and
      build no index; the first sparse chunk builds the backlog), so short
      prompts run at dense speed. With every chunk sparse the same prompts
      ran 0.60x at 4k and 0.84x at 8k, the per-query gather costing more
      than dense tiles while the selection holds most causal keys anyway.
      Index pages are built by the device Ward, one dispatch per closed
      block for every layer.
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
- [x] Qwen3 serving engine (`Engine::set_lsa`): the covered layers record
      the sparse path; prefill chunks are cut at cluster boundaries and every
      forward (prefill, scoring, decode windows, speculative verify) brings
      its sequences' pages up to date first; release frees them. Gates on the
      real Qwen3-0.6B: a budget past the context generates exactly the dense
      continuation (pages built mid-decode), last-hidden relative error
      1.2e-6 at that budget against 0.29 at 8 keys; on the tiny model the
      scheduler conformance suite passes with LSA on (batched = single,
      prefix hits prefill only the suffix and report their cached tokens,
      speculation = greedy). Under YaRN (past the native context) the index
      un-rotates by the engine's own frequency table: a full budget is within
      1.2e-6 of dense on Qwen3-0.6B at factor 2.
- [x] Serving surface: `--sparse-attention FILE` (`model::policy`), honoured
      by the Qwen3 resident (`Engine::set_lsa`) and refused by name by every
      model that does not declare it.
- [x] Memory planning: the Qwen3 resident charges `--sparse-attention`'s
      projections, index pages and scratch as a `memplan` demand before
      loading (`qwen3::serve::lsa_device_bytes`, the formula `set_lsa`
      allocates by; gated equal on the tiny engine).
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
- [x] Device Ward (`lsa_ward`): every sparse layer's index pages in one
      pool, a slot per layer; a closed block's index keys for every layer and
      one Ward dispatch (a workgroup per layer) write the page with no
      readback. The pools' Gram matrix is kept current by linearity and read
      from one triangle, so a merge cost is the same both ways (costs that
      differed by rounding stalled merging on real Qwen3 keys and dropped
      keys from the index; the real-model full-budget gates caught it).
      Gates: twin-direction keys bucket exactly as the host does; spread,
      correlated and repeated keys each land in exactly one bucket.
- [x] Measured dense crossover (`Engine::set_lsa`): one layer's dense chunk
      attention timed against the sparse read and attention at growing
      positions on the device in use; prefill chunks ending at or below the
      crossover read densely and skip index building
      (`set_lsa_dense_until` overrides it, `qwen_lsa prefill --dense-until`).
- [x] Serving metrics: `lsa_read` adds each row's selected keys and live
      keys to 64-bit device counters (one atomic per row, read only when a
      snapshot asks); `PagedDecoder::sparse_attention_stats` carries them
      with index bytes, pages and dense-read rows into the scheduler's
      `lsa_*` metrics. Gates: the counters equal the selection rows' counts
      over two reads; a budget past the context selects exactly the causal
      keys over a real generation; chunks below the crossover read and index
      nothing.
- [x] D-LSA training (`model::lsa::distill`): teacher targets (head-averaged
      attention over the keys outside sink and window, top 2048,
      renormalised), the forward-KL objective with analytic gradients of both
      projections and the temperature through the row normalisation, Adam,
      layers in parallel; `Engine::capture_attention` keeps each layer's
      rotated queries and keys of a dense prefill as teacher material;
      `qwen_lsa distill | eval | prefill` drives it on a real checkpoint (PCA
      initialisation, retention of dense attention, held-out loss, prefill
      timing). Gates: gradients against central differences; held-out KL
      2.14 -> 0.0006 on a learnable teacher, recovering its temperature.
      Run on Qwen3-0.6B (`qwen_lsa distill`, brain's `crates` sources, 48
      documents of 8192 tokens, 64 sampled queries each, 2 epochs, lr 2e-3,
      PCA initialisation, about 40 s a document on a P40 and 48 cores):
      mean KL over the 28 layers 0.770 (epoch 1) -> 0.686 (epoch 2), the
      same documents scoring lower on their second pass; learned
      temperatures 0.090-0.094.
      Held-out evaluation (`qwen_lsa eval`, another repository's
      documentation, 6 documents of 8192 tokens, top_k 1024, top_p 256,
      PCA fitted on the training corpus):
      | selection | dense attention kept | next-token loss, last 256 positions |
      |---|---|---|
      | dense | 1 | 3.0759 (ppl 21.67) |
      | window 512 + sinks | 0.8506 | - |
      | random projection | 0.9247 | 3.1870 (+0.111) |
      | PCA | 0.9483 | 3.1044 (+0.029) |
      | distilled (D-LSA) | 0.9528 | 3.0867 (+0.011) |
      Distillation closes 90% of the random indexer's loss gap and 62% of
      PCA's: +1.1% perplexity while each query reads about a third of its
      keys.
- [x] N-LSA training (`Qwen::enable_native_lsa`, `model::lsa::native`):
      every layer's queries read only their selection, made each forward on
      the device from the layer's own rotated queries and keys (identity
      index space, no learned projection: the index kernels take index
      widths up to 1024 and an unprojected mode, Ward clusters every block of
      every sequence in one dispatch) and held fixed while gradients flow
      exactly through the selected entries. Attention runs over the
      selection only - `lsa_native_fwd` keeps each row's log-sum-exp,
      `lsa_native_bwd_dq` rebuilds the probabilities per query,
      `lsa_native_bwd_dkv` gathers per key over the queries that selected it
      - a workgroup per (row, KV head) sharing each K/V row across the group.
      The per-key inverse is a stable radix sort of (key, query) pairs
      (`gpu_core::sort`), so it is deterministic and needs no atomics; the
      whole selection is recorded on the training tape (no readback) and the
      backward's recompute reuses it. Memory is linear in the context: the
      index is `b * t * (2 * row width + 4)` words a layer, no `[t, t]`
      buffer. `long_range: false` is the window-only baseline. Gates: the
      kernels match an f64 oracle of attention over random selections and
      its gradients to 7e-7 (grouped heads 1 to 8, head_dim 8 to 256); the
      device inverse is the host packing word for word; at a full budget a
      training step's loss equals dense and its gradients agree to 1e-8
      (largest 3.3e-2); window-only changes them. Training step, 4 layers,
      d_model 512, 8 heads over 2 KV heads, top_k 512, window 512, P40,
      dense the flash kernels:
      | t | dense | native sparse | speedup |
      |---|---|---|---|
      | 2048 | 234 ms | 443 ms | 0.53x |
      | 4096 | 731 ms | 1075 ms | 0.68x |
      | 8192 | 2584 ms | 2699 ms | 0.96x |
      | 16384 | 9775 ms | 6285 ms | 1.56x |
      | 32768 | 38437 ms | 14376 ms | 2.67x |
      Dense grows 3.9x a doubling, sparse 2.3x (the flat read's bucket
      scoring is the superlinear part). Below about 8k the gathers cost more
      than dense tiles; training semantics need the sparse read there.
      Experiment (`lsa_native`, run by hand): a 2-layer, 128-wide model
      trained from scratch for 3000 steps on recall whose key/value pairs sit
      over 200 tokens before their queries, past a 16-key window (top_k 16,
      top_p 8): held-out answer loss dense 1.047, native sparse 1.001,
      window only 3.103 (chance 3.466; with the earlier masked-dense
      training 0.930 / 1.050 / 3.083) - sparse attention learns the
      retrieval a window cannot. The sparse model, saved and served by the
      engine reading sparsely through the identity indexer, gives its answers
      the training model's loss (0.8361 both, largest single answer 2.7e-6
      apart).

## Not yet done

- [ ] Sparse prefill bandwidth: bf16 KV tier for the sparse read, the
      shared window as a dense sliding tile (it is a fifth of every
      selection), and more bytes in flight per workgroup.
- [ ] D-LSA on the generic trainer (`ModelFit`) and device kernels for the
      distillation arithmetic, which is host math today.
- [ ] N-LSA at short contexts: the window as a dense banded tile in the
      training kernels (it is half of every selection), so training below
      8k stops paying gathers for it.
- [ ] Tree search (node-bucket levels and beam descent) after flat accuracy
      is proven at long context.
- [ ] Quantized index storage (bf16 centroids, fp8 tree nodes) gated on
      retrieval agreement separately from KV-tier numerical error.

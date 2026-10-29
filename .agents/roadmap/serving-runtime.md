<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# serving-runtime - one shared inference runtime for long context and high concurrency

Scope: The shared LLM serving runtime (`model::serve`, `model::paged`, `model::kv_offload`, `llmserve`) and the engines that plug into it, for 128K-1M contexts at high concurrency.

The goal: every served decoder approaches the bandwidth, compute and capacity
limits of its device while maximising completed requests per second under
time-to-first-token and inter-token latency objectives. The unit of
optimisation is the whole request workload (cache reuse, scheduling, memory
movement, execution), not one forward. Evidence for the principle: fixing
cross-turn prefix reuse took a recorded 240-turn agent workload from 549 s to
312 s on the same card (#347); native decode attention cut attention time per
token from 2.06 to 0.58 ms (#350).

Model crates describe the computation and the state they need; the shared
runtime decides scheduling, allocation, placement and execution. A one-off
optimisation inside one engine is debt unless it lands as a shared mechanism.
Kernel selection and rooflines belong to `kernel-performance.md` and `probe.md`;
machine facts to `gh200.md`; this ledger references them rather than restating.

## Verified gaps (source audit, rev 6b9eae851, 2026-10-28)

An external review listed gaps; each was checked against the source. Status
column: **open** unless stated.

| # | gap | evidence | note |
|---|---|---|---|
| G1 | Admission charged a request for its whole prompt, blocks it adopts from the prefix cache included | `model::serve::Scheduler::admit_one`, `prefill_debt` | **fixed** (below). Worse than over-reservation: the next turn of a conversation whose cached history exceeded `pool - prompt` was never admitted, even on an idle server |
| G2 | The 35B MoE serves only through a one-block-per-sequence engine: `max_concurrent x max_seq_len` fp32 KV, no paging, no prefix sharing | `crates/qwen35moe/src/serve.rs` (`pool_rows`), `crates/catalog/src/resident_qwen35moe.rs` | the paged GGUF 27B engine is the pattern to generalise |
| G3 | The native 27B engine (`crates/qwen35/src/serve.rs`, `block_size == max_seq_len`) is still served for brain-format checkpoints | `crates/serving/src/executor.rs`, `model_dir.rs` | released GGUFs use the paged `int8_gguf_resident` engine (256-row blocks) |
| G4 | Sequence identity and KV pages are not separate resources in the one-block engines: the first block is the sequence | same files as G2/G3 | the paged GGUF engine already separates them (state row on `BlockTable`) |
| G5 | Sampling sees at most 32 device candidates on the 27B path; top-p tails past rank 32 are truncated | `int8_gguf_resident/engine.rs` `TOPK_CAPACITY`, `model::serve` sampler docs | exact full-vocabulary device sampling is open |
| G6 | Recorded decode replayed as a CUDA graph returned zeros from library GEMMs on the NVFP4 tier; recording declines the library | knowledge #356 | root cause not found; replay loses the library path |
| G7 | Block tables, positions and lengths are rebuilt on the host and uploaded every decode step | `crates/qwen35/src/model.rs` decode-tape upload | no persistent device batch state, no CPU/GPU step overlap |
| G8 | Any adapter row turns multi-sequence prefill sequential and disables speculation | `int8_gguf_resident/engine.rs` `prefill_slices`, `serve/speculate.rs` | per-row adapter routing exists in decode |
| G9 | Host KV offload swaps whole sequences through host vectors, synchronously; only `qwen3` adopts it; the 27B resident refuses `--kv-offload-gb` | `model::kv_offload`, `catalog/resident_qwen35.rs` | no page-granular async tier, no use of coherent Grace memory |
| G10 | Layer-sharded serving stages the boundary residual through host memory per step; collectives stage through host and block on TCP | `int8_gguf_resident.rs` shard path, `model::collective`, `netcollective` | device peer copies are the first step |
| G11 | Multi-request prefill is off by default (`batch_tokens = 0`, slice 512) because batching hurt early requests' latency in the measured workload | `PrefillPolicy::default` | wants a cost model, not a new constant |
| G12 | The GGUF engine's decode window is 1 | `decode_window_capacity` | |

Not gaps (claims checked and found already in place): arrival-time continuous
batching through `llmserve::run_open_scheduled`; prefix caches keyed by adapter
namespace; an adaptive, measured speculation cost gate (`serve/speculate.rs`
`CostGate`); a pinned upload ring (`backend-cuda/src/upload_ring.rs`, used for
weight writes, not KV).

## Plan

Ordered by dependency. Each item ends with a measured improvement on the
serving qualification suite (item 1) or a correctness gate; no speed-up is
claimed before its baseline exists.

| order | item | owner | done when | status |
|---|---|---|---|---|
| 1 | Serving qualification suite: contexts 4K-128K (256K/1M where supported) x concurrency 1-64 x traffic (steady, burst, recorded agent trajectories, shared prefixes); TTFT, TPOT, p50/p95/p99, goodput under SLO, cache-hit tokens, KV occupancy, bytes per phase | `perf`, `cli`, `llmserve` | repeatable GH200 numbers with per-phase attribution | in progress: `qwen35:` request-level target, honest `longctx` and `shared_prefix`, one-card baseline below; recorded agent trajectories and 256K/1M open |
| 2a | Admission and prefill debt charge only blocks not adopted from the prefix cache (`PagedDecoder::cached_prefix_blocks`) | `model::serve` | `model/tests/serve_prefix_admission.rs` | done |
| 2b | Incremental reservation: charge the next scheduled slice plus bounded decode headroom, not the whole prompt, with a guaranteed route to completion | `model::serve` | mixed long/short prompts admit without full-prompt reservation; no deadlock or leak under the KV-pressure tests | open |
| 2c | Admission/eviction as one transaction; victim choice by restore cost, not recency | `model::serve` | a failed admission rolls back without losing cache entries | open |
| 3 | One paged layout for every served decoder: stable sequence id, KV pages and recurrent slots as independent pools (G2-G4) | `model`, `qwen35`, `qwen35moe` | no served path allocates `max_seq_len` per sequence outside bounded-state modes | open |
| 4 | Persistent device batch state: row diffs scattered into device buffers, double-buffered metadata, schedule N+1 prepared while N runs (G7) | `gpu-core`, `backend-cuda`, `model::serve` | no host sync inside the token loop; per-step upload bytes proportional to changes | open |
| 5 | Graph-safe library execution: find #356's root cause, verify eager vs replay per tier and shape | `backend-cuda`, `gpu-core` | replay keeps library GEMMs with bit-level parity | open |
| 6 | Shape-driven attention and GEMM selection over phase, batch, KV length, KV tier, page size | `kernel-performance.md` owns it | selection table covers the context x batch matrix | open |
| 7 | Exact device sampling over the full vocabulary (G5) | `model::serve`, `kernels-cuda` | distribution tests match the host reference; no logits readback | open |
| 8 | Cost- and cache-aware scheduling: order by uncached work, shared in-flight prefixes, bounded-fairness aging, calibrated step-cost model (G11) | `model::serve` | goodput and tail latency improve on mixed traffic with bounded starvation | in progress: shared in-flight prefixes done; uncached-work ordering, aging and the step-cost model open |
| 9 | One logical prefix index (radix over content, namespace, revision) over block-sharing and snapshot physical forms; retention by saved work per byte | `model::paged`, `model::snapshot_prefix` | longest-prefix match without scanning entries | in progress: hybrid branch points (a shared document is resumable) done; the radix index and retention by saved work open |
| 10 | Grace memory as a managed warm KV tier: page-granular async promotion/demotion through the existing upload ring; direct coherent reads measured separately (G9) | `model::kv_offload`, `backend-cuda` | warm-prefix hits served from host without whole-sequence swaps | open |
| 11 | Batched adapters in prefill and speculation (G8) | `qwen35`, `llmserve` | adapter traffic keeps multi-sequence prefill | open |
| 12 | Device-resident inter-GPU transfers and collectives (G10) | `model::collective`, residency | no host-staged per-token transfer on supported devices | in progress: peer copy and link measurement (A) and the 27B stage boundary (B) and the NCCL communicator (C) done, tensor-parallel use and the head hand-off open |

## Baseline (2026-10-29, one GH200, Qwen3.8-27B Q8_0, int8 KV)

Measured on an otherwise idle card (`brain perf run longctx --target qwen35:<gguf>:kvint8`,
`qwen35_decode_profile`, `brain perf run sweep --target qwen35:<gguf>`); artifacts are kept outside the repository.

| context | single stream | best total decode | largest batch that fits | prefill (1K / 8K / 32K prompt) |
|---|---|---|---|---|
| 8K | 72.5 tok/s | 1307 tok/s at batch 128 | >= 128 | 2852 / 2740 / 2392 tok/s, after the prefill GEMM work 3347 / 3607 / 2892 |
| 32K | 69.9 tok/s | 692 tok/s at batch 48 | 53 | same |
| 128K | 62.5 tok/s | 234 tok/s at batch 8 | 15 (KV capacity: ~4 GiB per 128K sequence) | same |

Gap list, by what the profile attributes (in the order it would move goodput):

- Decode linears stream weights at 1.3-1.9 TB/s of the measured ~3.7 TB/s roof within 32 rows, and above 32 rows
  the m128 tensor-core GEMM reads them at ~0.8 TB/s (batch 48 at 8K: 35 of 53 ms). Extending the weight-streaming
  regime past 32 rows is the high-concurrency lever (`cuda.md`). First step landed 2026-10-29: streaming candidates for
  33-128 rows chosen by measurement (batch 48 at 8K 51.9 -> 47.6 ms, batch 64 58.3 -> 53.9 ms; 27 % of the roof at 48
  rows), the 7-8 ms target is still open (knowledge 373).
- Prefill is bound by the group-32 scale fold (knowledge 372): 15-23 % of the int8 tensor-core roof.
- 128K concurrency is KV-capacity bound: item 10 (Grace tier) and a smaller KV tier, or the second card (item 12).
- Prefill slices block decode: inter-token p99 400-830 ms under load (item 8's step-cost model).

Page size (256 rows today) is part of item 3 and is chosen per cache group by
measurement (16-256), not fixed globally.

## Done

- 2026-10-28 (2a): `PagedDecoder::cached_prefix_blocks` peeks how many blocks a
  fresh prefill adopts from the prefix cache; `admit_one` and `prefill_debt`
  subtract them. Implemented for `qwen3` (block chains) and the paged GGUF 27B
  engine (snapshot entries, less the tail block it copies); engines without
  block sharing keep the default 0. A prompt with embedding rows is charged in
  full. Regression test: the next turn of a conversation whose history fills
  10 of 16 blocks livelocked the scheduler before; it now resumes from all 40
  cached tokens and yields the cold run's tokens.

- 2026-10-29 (12, A): `Gpu::copy_to` copies a byte range of a buffer on one
  card into a buffer of another over the card link when the driver offers one
  (`cuMemcpyPeerAsync` on the source's stream, peer access enabled per copy,
  ordered against both cards with the existing event fences, no device-wide
  sync) and through the host otherwise, stating the reason once per pair.
  `Gpu::peer_link` is the driver's topology answer
  (`cuDeviceCanAccessPeer` / `cuDeviceGetP2PAttribute`), `gpu_core::link`
  measures both paths with a short copy probe (8 MiB x 32, best of 3) and
  caches the result on disk like the roofs, and `brain devices` prints each
  pair's path and GB/s. Gates: `gpu-core/tests/peer_copy.rs` (exact on the CPU
  backend through the host; two-card test is skip-if-absent).
- 2026-10-29 (12, B): the layer-sharded GGUF resident hands the boundary
  residual from stage to stage as the previous stage's own device buffer
  (`StageRows::Device`, one implementation: the host-row entry points are thin
  wrappers) through decode, batched decode, verify, per-token and multi-sequence
  prefill and the background prefill pipeline, all via
  `Qwen35GgufInstance::chain_stages`. The host now carries only the embedding
  rows into stage 0 and the last stage's block to the head, and no longer waits
  for a stage to finish before the next is issued. Gates:
  `two_shard_int8_decode_with_a_device_boundary_matches_the_whole_shard_model`
  and `two_shard_int8_prefill_round_with_a_device_boundary_matches_the_host_boundary`
  (bit-equal; run on the CPU backend, where the hand-off falls back to the
  host). Open: the head still receives the last block through the host.
- 2026-10-29 (12, C): `gpu_core::devgroup::DeviceGroup` runs all-reduce (sum)
  and all-gather over `DeviceBuffer`s of several cards. On CUDA it is an NCCL
  communicator (`backend_cuda::nccl`, `libnccl.so.2` by `dlopen` like cuBLAS,
  one `ncclCommInitAll` for the process, one kernel-less handle per rank whose
  stream carries the collective, ordered against the producers and consumers by
  the same event fences as `copy_to`); an absent library, another backend or a
  refused card set reduces through the host and says why once. A failure of an
  opened device collective is an error, not a retry through the host. Not done
  and recorded here: tensor-parallel use (the collective is not yet called from
  any model), and `model::collective`'s host `Collective` seam still stages
  through host fp32.
- 2026-10-29 (12, measured on the two-GH200 node, 8 MiB copies, GPU tests of
  `peer_copy` and `device_group`): the pair reports direct peer access (driver
  performance rank 1); `Gpu::copy_to` sustains 127.1 GB/s gpu0->gpu1 and
  129.1 GB/s gpu1->gpu0 against 31.2 and 30.0 GB/s host-staged (about 4.1x);
  copies are bit-exact in both directions and wait for the kernel that wrote
  their source; NCCL 2.32.3 all-reduce and all-gather over the two cards are
  bit-exact. The device-boundary two-shard gates pass on CUDA too (both stages
  on one card there, so the hand-off is the same-card copy). Not measured: the
  27B resident's end-to-end step with a real model (no model may load on this
  node for this work), so the per-token saving is arithmetic only: a 20 KiB
  decode boundary is about 0.2 us on the link, and the host wait it replaces is
  what is saved. `brain devices` lists the pair: direct peer access,
  128.7 GB/s gpu0->gpu1 and 129.2 GB/s gpu1->gpu0.
- 2026-10-29 (1): measurement fixes found by the baseline: `perf run <scenario>
  --target qwen35:<gguf>` measures the served engine through the shared
  scheduler; `longctx` plans only on the `--device` cards (it counted both cards
  of the node) and times the served greedy tape (a full-logits readback made a
  false 32-row cliff); the `shared_prefix` workload now shares its document
  (7168 of 8192 tokens; it shared nothing). The int8 tuner no longer persists
  classes its budget did not reach as tuned (17-32-row decode steps ran on the
  slowest candidate: 32 rows 76.9 -> 47.4 ms at 32K).
- 2026-10-29 (9, 8): a prompt that leaves a cached one files the recurrent
  state where they part (a branch point), and requests arriving together that
  share a document wait for the first prefill to pass it
  (`PagedDecoder::hint_shared_prefix`); paged entries are cached the moment they
  are snapshotted. shared_prefix at concurrency 1/4/16: output 25.6/29.3/28.9 ->
  57.2/110.4/143.4 tok/s, TTFT p99 3.6/16.1/66 -> 0.49/2.1/11.5 s, prefix hit
  rate 0 -> 86 %. Gates: `model/tests/serve_inflight_prefix.rs`,
  `qwen35/tests/gguf_conformance.rs` (resumed prefills bit-equal to cold ones).

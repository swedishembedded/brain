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
| G2 | The 35B MoE serves only through a one-block-per-sequence engine: `max_concurrent x max_seq_len` fp32 KV, no paging, no prefix sharing | `crates/qwen35moe/src/serve.rs` (`pool_rows`), `crates/catalog/src/resident_qwen35moe.rs` | the paged GGUF 27B engine is the pattern to generalise; design, baseline and plan: `qwen35moe.md` "Serving on the shared paged engine" (the engine already batches decode and chunks prefill, 183 tok/s single stream, 986 tok/s at 16; what it lacks is paging, the prefix cache and the shared scheduler features) |
| G3 | The native 27B engine (`crates/qwen35/src/serve.rs`, `block_size == max_seq_len`) is still served for brain-format checkpoints | `crates/serving/src/executor.rs`, `model_dir.rs` | released GGUFs and qwen35 HuggingFace directories (the FP8 release, bf16 releases) use the paged `int8_gguf_resident` engine (256-row blocks) through `ResidentCheckpoint`; brain-format checkpoints still use the native engine |
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
| 8 | Cost- and cache-aware scheduling: order by uncached work, shared in-flight prefixes, bounded-fairness aging, calibrated step-cost model (G11) | `model::serve` | goodput and tail latency improve on mixed traffic with bounded starvation | in progress: shared in-flight prefixes and the mixed iteration with its step-cost model done (knowledge 379); uncached-work ordering and aging open |
| 9 | One logical prefix index (radix over content, namespace, revision) over block-sharing and snapshot physical forms; retention by saved work per byte | `model::paged`, `model::snapshot_prefix` | longest-prefix match without scanning entries | in progress: hybrid branch points (a shared document is resumable) done; the radix index and retention by saved work open |
| 10 | Grace memory as a managed warm KV tier: page-granular async promotion/demotion through the existing upload ring; direct coherent reads measured separately (G9) | `model::kv_offload`, `backend-cuda` | warm-prefix hits served from host without whole-sequence swaps | open |
| 11 | Batched adapters in prefill and speculation (G8) | `qwen35`, `llmserve` | adapter traffic keeps multi-sequence prefill | open |
| 12 | Device-resident inter-GPU transfers and collectives (G10) | `model::collective`, residency | no host-staged per-token transfer on supported devices | in progress: peer copy and link measurement (A) and the 27B stage boundary (B) and the NCCL communicator (C) done, tensor-parallel use and the head hand-off open |
| 13 | Precision admission by input resolution: a candidate is admitted when its local error stays within alpha (8) x the error of rounding its inputs to their storage resolution, with model-level quality evaluation as a separate gate; the fixed 2e-5 bar against brain's own int8 semantics admits nothing but that semantics | `gpu-core` (`native_tune`), `model` | candidates of a different arithmetic (bf16 dequant-in-register, fp8) are admitted or refused by the rule, and the quality gate runs on the real checkpoint | done 2026-10-29 (knowledge 376): candidates declare their input forms (`native_upgrade::Input`), the tuner grades against the f64 reference of the stored representation; every int8 candidate and every bf16 one is admitted; the gate is `qwen35_gguf_logit_check` (`--kernel slot=candidate`) on the released 27B; fp8 rows join by declaring a form |
| 14 | Dequantize-in-register int8 linears: weight codes x group scale to bf16 in registers, bf16 activations, one fp32 accumulation over all of K (no per-group fold, no activation quantization), for decode GEMV/streaming and prefill GEMM (the fold bounds both, #372 #373) | `kernels-cuda`, `gpu-core` | decode linears and prefill GEMM measured against the read and tensor-core roofs with nsys/ncu | in progress (knowledge 377): streaming candidates for 33-128 rows (`matmul_i8_dq_stream_{r,q}*`, 14-55 % less time than the best integer candidate) and the Hopper prefill GEMM (`matmul_i8_dq_wgmma`, 0.8x the time) landed and measured; 5-32 rows lost to the integer stream (2.0-2.2 against 2.4-2.8 TB/s) and rows 1-4 are held by the exact-decode contract |
| 15 | Workload-weighted tuning: classes timed in order of the served model's shape histogram x expected saving per second of tuning, not grid order (the budget cut left 17-32 rows untuned) | `gpu-core` (`native_tune`) | the served model's dominant classes are always measured first | open |
| 16 | Compressed KV read natively: K8/V4 (4-bit values) with attention computed on the codes in decode, prefill, append and verify; capacity +33 % over int8 KV where KV binds (128K) | `model::kv_tier`, `kernels-cuda` | quality gate on long prompts; 128K concurrency and decode tok/s measured | open |
| 17 | Native FP8 tier for block-128 FP8 releases: fp8 weights resident, fp8 tensor-core prefill with one fold per 128 of K | `model`, `kernels-cuda`, `qwen35` | the FP8 27B served natively, prefill and decode measured against the int8 tier | in progress (knowledge 378): served natively through the shared candidate seam (weights 1 byte each, e4m3 wgmma prefill, dequantize-in-register streaming decode), quality 100 % greedy agreement on the real checkpoint; faster than int8 from 4 rows up (prefill 1K 4989 against 3669 tok/s, batch 128 decode 1863 against 1475) and slower single stream (59.3 against 72.5 tok/s); behind vLLM/SGLang on prefill (31K: 4439 against about 12800 tok/s) and on the m <= 8 decode GEMV (1.9 of 3.7 TB/s) |
| 18 | Speculative decoding with the checkpoint's own MTP head for the FP8 release (exact; mixed-iteration verify rows; recorded verify rounds) | `qwen35`, `model::serve` | c = 1 / 4 / 16 / 64 above the plain scoreboard and vLLM + MTP | in progress (knowledge 395): the head loads from `mtp.safetensors` and is exact; measured it loses to plain at every concurrency (75 / 235 / 475 / 684 against 81 / 251 / 561 / 838), so the HF default is off (`--speculative mtp` opts in). Open: device-resident drafter, plain-step recording with a resident hidden row, device prompt feed |

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
  rows), the 7-8 ms target is still open (knowledge 373). Second step landed 2026-10-29 (item 14): bf16 dequantize-in-register streaming
  candidates for 33-128 rows take 45-86 % of the integer candidates' time (batch 48 at 8K: 48.0 -> 40.3 ms, batch 64 53.3 -> 49.8 ms,
  batch 128 92.5 -> 88.0 ms); below 33 rows the integer stream stays ahead.
- Prefill is bound by the group-32 scale fold (knowledge 372): 15-23 % of the int8 tensor-core roof. Landed 2026-10-29 (item 14, knowledge
  377): the Hopper dequantize GEMM runs the 27B's linears in 0.8x the time (1024 rows: 207.7 -> 171.6 ms of 300.5 -> 274.5 ms, 3408 -> 3730
  tok/s) and reads 330-370 TFLOPS of a bf16 roof near 990; the bound is now the staging, not the arithmetic.
- 128K concurrency is KV-capacity bound: item 10 (Grace tier) and a smaller KV tier, or the second card (item 12).
- Prefill blocking decode: found and mostly fixed 2026-10-29 (knowledge 374): inter-token p99 at concurrency 4 / 16 went
  202 / 357 ms to 120 / 179 ms (max 393 / 410 to 201 / 198 ms) on `chat` 1024-token prompts. What is left of it: the MTP
  head's prompt feed (`finish_slice`, 42-56 ms of synchronous normal-lane work per prompt), decode steps running 1.5-2x
  slower while a prefill shares the card (SM sharing; item 8's step-cost model should price it), and a lone prompt reading
  in at half the sliced speed in background mode (1.56 s against 0.76 s for 1731 tokens, `gguf_resident_real`): all
  but the SM sharing fixed 2026-10-29 (knowledge 375), which is avoided by time-slicing, not reduced.

## Step plan and replay (design, 2026-10-29; items 4, 5, 17 and the host cost of knowledge 379-381)

The engine today records a layer round as thousands of independent dispatches, each allocating its intermediates, clearing some,
and (on CUDA) uploading its own parameters; only an exactly repeated decode step is captured. Under concurrency that makes brain
host- and idle-bound (card busy 39-66 % of a mixed iteration, 15-30 % of vLLM's throughput). The architecture that replaces it,
behind the existing `Gpu::step`/`submit` API so the WGSL reference path and old hardware keep working:

1. **Step plan.** The engine emits a layer round or decode step as an op list once per shape bucket (decode rows, prompt rows, batch,
   context bucket). A planner resolves the kernel choice of each op through the tuner, assigns every intermediate to an offset in
   ONE arena by lifetime (no per-op alloc/free; no clear of a buffer its writer overwrites), and packs all launch parameters into ONE
   device-resident parameter block.
2. **Per-step metadata.** A step uploads only what changed (positions, block-table rows, slot ids) into the device-resident batch
   state (item 4); the plan itself is not rebuilt.
3. **Lowering per backend.** CUDA: one graph per bucket, replayed with one launch, segmented around ops a capture cannot hold
   (item 5, #356). Vulkan: command-buffer replay. wgpu/CPU: a plain loop over the plan. Same ops, same results.

End state (user direction, 2026-10-29): the card executes a whole step standalone and the host only collects results.

- Per shape bucket the kernel choices, arena offsets, parameter block and graph are precomputed; a step is one metadata upload plus one
  graph launch.
- Decode is device-resident for K steps (gap G12, decode window): sampling, stop checks and the next inputs stay on the device, one
  launch runs K steps, and the host harvests tokens and admits work between launches.
- The plan is built per stage and device from the start. Cross-card edges are plan nodes on the same stream: peer copies
  (`Gpu::copy_to`, 128 GB/s measured over NVLink) or NCCL collectives (`gpu_core::devgroup`), so layer-split and tensor-parallel come
  from the plan and not from a separate path.
- Older hardware keeps the same plan lowered to a plain loop or a Vulkan command buffer.

Increments, each measured (nsys card-busy %, API calls per iteration, client tok/s) and pushed on its own:

- **A (done, knowledge 381): the parameter block.** A held batch of eagerly issued dispatches uploads its parameters with one copy
  (`CudaBackend::issue_eager`), and the prefill/mixed layer loop issues per layer (`pass_scope`). Card busy 66 -> 79 %, copies per
  mixed iteration 4545 -> 480, c=16 / c=64 output 291 / 384 -> 340 / 460 tok/s.
- **B1 (done, knowledge 381): the clears and fences.** Buffers a kernel overwrites are `storage_uninit` (the serving GDN fold with its five
  scratch buffers instead of eighteen, attention context, the fused decode output, the uploaded index words), and the cross-handle ordering
  events are reused: memsets per mixed iteration 930 -> 20, event create/destroy 6000 -> 0, card busy 79 -> 81 %.
- **B2 (done, knowledge 381): the host in front of and behind the forward.** Embedding rows are read without dropping their pages after each
  row and decoded in parallel (15 ms), and recording a step allocates no per-shape device uniform (the 7 ms drain-time free). Card busy 81 -> 90 %,
  c=16 / c=64 output 337 / 465 -> 360 / 500 tok/s, 31K TTFT 7.03 -> 6.64 s.
- **B: the arena (rest).** Intermediates of a bucket at fixed offsets of one allocation (no pool alloc/free: ~11000 calls per iteration; no
  clears of overwritten buffers; row views instead of the 1008 `concat_split` copies), which is also what makes C replayable.
- **C: the full round as a plan, then graph lowering** with the row count bucketed (bounded graph memory), the per-step metadata
  upload of item 4, and the other backends' lowerings.
- **Training on the same seams (2026-10-30, knowledge 389).** Shared pieces the 27B LoRA step now uses: `Gpu::retain_freed_blocks` (the pool's
  release threshold follows the working set instead of the serving-sized 4 GiB; a loop that cycles more than that must raise it or it maps its
  memory again after every synchronisation), `pass_scope` per layer, native `mul`/`splice_add`/`splice`. The training step is 96 % card busy
  without a graph; what B/C would still give it is the 0.2 s of launch gaps and a host-free step, and it needs the same two things serving
  waits for (serving re-checked after these changes: c=1/4/16/64 79.0/300.9/573.1/850.0 tok/s, 31K TTFT 3.99 s): lifetime-planned arena offsets (a W8K step requests 1.5 TB of buffers over a 40 GB peak) and graph-safe library GEMMs (item 5).

## Mixed iterations (design, 2026-10-29; items 8 and 11)

Measured gap (FP8 27B, one GH200, `vllm bench serve` chat 1K in / 128 out, output tok/s at concurrency 1 / 4 / 16 / 64):
vLLM 79 / 299 / 759 / 1241, SGLang 86 / 214 / 781 / 1044, brain 46 / 117 / 166 / 173, with TTFT p50 6.6 s at 16 and 41.8 s at 64.
Brain runs prefill and decode as separate device passes (a paced background prefill, one prompt at a time, beside the decode
step), so prompts queue and each pass reads the weights alone. The reference servers build one batch per iteration: the next
token of every running sequence plus prompt chunks of several requests under a token budget, one forward, each weight read once.

Design (the scheduler decides, the model executes):

- `model::serve` owns the policy. `PagedDecoder::mixed_step() -> Option<MixedLimits>` declares the capability (`max_rows` per
  forward, `min_prompt_rows`); `forward_mixed(tables, inputs, cands, jobs) -> MixedOutcome` runs the decode rows (`cands == 0`
  greedy, else top-k) and the prompt jobs as one forward. The default runs `prefill_chunks` then the decode step, so an engine
  that does not override it keeps today's separate passes, and every consumer of the old methods (prefix cache, branch points,
  in-flight sharing, preemption, adapters, speculation outside prompt reading) is untouched.
- The budget is `StepCost::prompt_rows`: a decayed least-squares fit of iteration seconds against rows (`fixed + per_row * rows`,
  O(1) per iteration), turned into the rows that fit `step_latency_target()` after the decode rows; never below
  `min_prompt_rows` (a prompt always advances), never above the room in the forward; with nothing decoding the forward is
  filled (no token interval to protect). Prompt rows are shared over the prefilling sequences oldest first with the existing
  per-sequence share (`allot_prefill`, O(prefilling)).
- Speculation is not run in an iteration that reads prompts in (verify rows are another shape of forward); plain iterations
  (nothing prefilling) keep speculation and the decode window.
- The qwen35 paged engine implements `forward_mixed` on the multi-sequence prefill round (`run_prefill_multi_stage`): decode
  rows go first, their recurrent layers through the pooled decode kernel and their attention through paged decode attention,
  the prompt segments through the chunked recurrence and chunk attention, every projection and MLP once over all rows. It
  supersedes `PrefillPace` time-slicing and the background lane where the mixed step is available.
- Open choice recorded by knowledge 375 (latency-paced against throughput-first) becomes one knob: `step_latency_target`
  (`None` fills every forward).

Page size (256 rows today) is part of item 3 and is chosen per cache group by
measurement (16-256), not fixed globally.

## Done

- 2026-10-29 (step plan, increment A; knowledge 381): eager batches upload their kernel parameters with one copy instead of one in front of
  every kernel (nsys: 14 us of card idle per kernel), and the prefill/mixed layer loop issues one batch per layer. FP8 27B output tok/s at
  1 / 4 / 16 / 64: 47 / 192 / 291 / 384 -> 49 / 200 / 340 / 460, card busy 66 -> 79 % of a mixed iteration; parity unchanged.
- 2026-10-29 (8, 11; knowledge 379): the mixed iteration. `PagedDecoder::mixed_step` / `forward_mixed` run the decode rows of every
  running sequence and a prompt slice of several sequences as one forward; `StepCost::prompt_rows` sizes the slice from a decayed
  fit of iteration seconds against rows and `step_latency_target` (floor 512 rows, 25 % stretch when the target is out of reach, the
  whole forward when nothing decodes); the qwen35 paged engine implements it on `Qwen35::run_mixed_stage` (pooled recurrent step and
  paged decode attention for the leading rows, chunked kernels for the segments). FP8 27B through `brain serve` + `vllm bench serve`,
  output tok/s at 1 / 4 / 16 / 64: 46 / 117 / 166 / 173 -> 50 / 187 / 208 / 237 (vLLM 79 / 299 / 759 / 1241), TTFT p50 at 16 / 64 6.6 /
  41.8 -> 2.5 / 2.0 s. Left: the mixed forward is host-bound at small slices (2970 launches, card busy 48 of 97 ms) and its fp8 GEMMs
  underfill the card at 142-600 rows (ncu 23-28 % SM); MTP builds, adapter banks and exact decode keep the background lane and
  `PrefillPace`, which therefore stay until the mixed step carries them (verify rows are more rows, adapter routing per row); the
  decode copies (`concat_split`) and a graph replay of the chunk round; speculation is off while prompts go in.

- 2026-10-29 (host-path complexity audit; every serving-path scan checked against the source, host CPU timings): streaming
  detokenisation feeds each token once (`Utf8Decoder`; G=16000 4.61 s -> 1.0 ms per generation, shared by `SeqState`,
  deepseekvl and nemotronasr); sampler penalties and DRY fold in each accepted token (`ProcessorState`; 32k context 582 -> 2 us
  per token); special tokens cut in one pass (`SpecialMatcher`; 2000 specials on 162 KB 21 ms -> 33 us); the XML tool-call scanner
  resumes its marker search (100 KB value 163 -> 3 ms); `Scheduler::streams` replaces a find per open request (B=1024 116 -> 18
  us), admission validates a prompt once, `ensure_decode_room` keeps a running total, the decode batch uses a mask; the n-gram
  drafter keeps an incremental index; the step cache looks up by borrowed key; native candidate choice is indexed. Left for the
  radix prefix index: `snapshot_prefix` peek/branch_point/take_row, `PrefixCache` per-block hashing and `evict_keeping`, shared-prefill LCP checks.
  Still linear by design: `completion_of`/`cancel` (once per request), `ChatScanner` header parse (bounded by a name), `ngram_followers` (bounded by its window).

- 2026-10-29 (13 precision admission, 14 dequantize-in-register linears; knowledge 376, 377): candidates declare the activation and
  weight forms they consume (`native_upgrade::Input`), `native_tune` admits a candidate within 8x the error of rounding those
  forms against the f64 reference of the stored representation (all int8 candidates and all bf16 ones pass), a dispatch hands each
  candidate its own activation form (bf16 rows from one `cast_f32_bf16` shared by the products of an activation; the packed int8
  pair is skipped when every class of the row bucket reads bf16). Candidates: `matmul_i8_dq_stream_{r1,r2,r4}` (33-64 rows),
  `matmul_i8_dq_stream_{q2,q4}` (65-128), `matmul_i8_dq_wgmma` (Hopper, K % 64). GH200, 27B Q8_0, kv int8, base -> new:
  `qwen35_decode_profile` (8K, 6 steps) ms/step 48 rows 48.0 -> 40.3, 64 rows 53.3 -> 49.8, 128 rows 92.5 -> 88.0, 1-32 rows within
  noise; `qwen35_gguf_prefill_profile 256 1024` round 300.5 -> 274.5 ms (3408 -> 3730 tok/s); `brain perf run longctx` prefill
  1K/8K/32K 3348/3682/3175 -> 3669/3860/3308 tok/s, decode 48/64/128 rows 1010/1201/1378 -> 1200/1310/1475 tok/s; `brain perf run
  sweep chat` output tok/s 58.5/100.0/116.2 -> 59.8/102.9/128.2 at concurrency 1/4/16, inter-token p99 43.2/67.7/75.1 -> 41.7/66.4/69.1
  ms, TTFT p99 470/3051/15088 -> 451/4012/13537 ms (a second run at concurrency 4 gave 3191 -> 2834 ms: queueing behind prefill,
  spread over 1 s between identical runs), goodput unchanged (44.9 / 2.1 / 0). Quality gate (`qwen35_gguf_logit_check`, `qwen35_gguf_score`, real
  prompts, released 27B): batched decode of 48 / 72 streams against the streams alone 1.0x the numerics floor (rel-L2 9.7e-3 / 9.2e-3, old
  9.9e-3 / 9.5e-3); perplexity of 9283 source-code tokens 7.7415 -> 7.7486 (+0.09 %), argmax identical to the old path in 98.8 %; every
  argmax flip is a near-tie (margin <= 0.12).
- 2026-10-29 (8, decode cadence under prefill; knowledge 375): background rounds are at most 512 rows (a lone 1731-token
  prompt 1.56 -> 0.75 s); a prompt gets one paced slice of device time per iteration (`PrefillPace`, `PrefillPolicy::step_target`
  45 ms), waited for before the decode step so the step runs on an idle card; the MTP prompt feed is parallel on the host
  (42 -> 22 ms) and run 256 rows with each slice; the gate does not speculate where a speculative step misses the latency
  target and a plain one meets it (`PagedDecoder::step_latency_target`). Chat 1 / 4 / 16: inter-token p99 46 / 121 / 180 ->
  39 / 63 / 68 ms, max 76 / 170 / 198 -> 77 / 97 / 125, output 64.5 / 138.5 / 181.6 -> 58.2 / 102.2 / 120.8 tok/s, TTFT p99
  473 / 1969 / 9477 -> 478 / 3023 / 14842 ms, goodput 36.3 / 2.9 / 0 -> 47.3 / 2.1 / 0. `gguf_resident_real` passes
  (20 of 20). Open: `step_target` as a serve flag; the 40-60 ms decode iterations at b=5-6; TTFT under the pacing at 4+ clients;
  an operator choice between this latency-paced mode and a throughput-first one (the overlapped pump it replaced gave
  26-33 % more output at 4-16 clients), decided by measurement once item 14 shortens prefill.

- 2026-10-29 (8, decode stalls behind prefill; knowledge 374): the scheduler thread, not the device, was where a decode
  step waited. A graph retired after a free called `cuMemFreeHost`, which waits for every stream: the urgent head of a
  new prompt's first token took 145 ms behind the next prompt's queued round (`GraphCache::retire` now keeps the
  parameter blocks for the next capture). The background pump recorded a whole 960-row round (1425 launches) per call
  and blocked in `cuLaunchKernel` for the 140 ms the device needed to drain its launch queue; a chunk round is now
  `begin_prefill_chunk` / `advance_prefill_chunk` over a resumable layer loop (and a short round's groups), and a
  non-waiting pump records for at most 16 ms (`PUMP_RECORD_BUDGET`). Chat 1024-token prompts, `perf run sweep` 1 / 4 / 16
  (inter-token p99 / max ms, output tok/s, TTFT p99 ms): 44 / 57, 202 / 393, 357 / 410; 65.5, 130.9, 164.6; 447, 1965,
  10023 became 44 / 59, 120 / 201, 179 / 198; 64.0, 136.4, 182.8; 531, 2032, 8973. Gates:
  `gpu-core/tests/urgent_lane.rs` (an invalidated graph retires without waiting for the normal lane),
  `qwen35` `model::tests::a_chunk_round_recorded_layer_by_layer_equals_the_whole_round` and
  `a_short_round_issued_a_group_at_a_time_equals_the_whole_round` (bit-equal to the whole round).

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
- 2026-10-29 (17 native FP8 tier, G3 for HuggingFace directories; knowledge 378): `Qwen/Qwen3.8-27B-FP8` is served by the paged engine
  with its weights resident as E4M3 bytes and 128 x 128 scales. Candidates of the tuned slot `matmul_fp8_native`:
  `matmul_fp8_dq_stream_{r1,r2,r4,q2,q4}` (1-128 rows), `matmul_fp8_dq_wgmma`, and `matmul_fp8_wgmma` (e4m3 x e4m3 wgmma with one fold
  per 128 of K, activations from `quant_e4m3_128`), all admitted by the input-resolution rule. GH200, kv int8: longctx 8K single stream
  59.3 tok/s, batch 64 / 128 1337 / 1863 tok/s, prefill 1K / 8K / 31K 4989 / 5482 / 4439 tok/s; `perf sweep chat` 52.9 / 128.5 / 181.7
  out tok/s at concurrency 1 / 4 / 16; `brain serve` + `vllm bench serve` 46.3 / 117.3 / 165.6 / 172.9 at 1 / 4 / 16 / 64 (vLLM 79 / 299 /
  759 / 1241, SGLang 86 / 214 / 781 / 1044). Quality against the same weights as bf16: greedy agreement 100 %, rel-L2 2.3e-2, KL 1.1e-3.
  The BPE merge loop was quadratic in the piece length and held the lane for seconds on whitespace-free prompts (fixed). Open: the m <= 8
  FP8 GEMV, a persistent e4m3 GEMM, an rms-norm -> e4m3 fusion, MTP from `mtp.safetensors`, and the prefill/decode interleave (items 8, 11).
- 2026-10-29 (launch cut, knowledge 380): the card sat inside `brain_gdn_chunk_loop_tc` for 52 % of a mixed iteration because a round was chunked
  by its largest divisor (a 600-row slice in 75 chunks of 8); sequences now run whole chunks of 64 plus one short chunk (`gdn_stream_pieces`), and
  the zero-fresh buffers a kernel overwrites became `storage_uninit` (memsets 1230 -> 858 an iteration). `brain serve` + `vllm bench serve`
  output tok/s at 1 / 4 / 16 / 64: 50 / 187 / 208 / 237 -> 47 / 190 / 294 / 366, ITL p99 at 16 589 -> 293 ms, 31K TTFT 7.3 s. Open, by weight:
  the host's CUDA API calls per mixed forward (launch, parameter upload, pool alloc; a bucketed graph replay of the layer round, row-view
  binding instead of the `concat_split` copies), the e4m3 front end, stream-K GEMM fill (items 4, 5, 8, 14, 17).
- 2026-10-30 (warpgroup prefill attention, knowledge 383): the head_dim 256 prefill over an int8 or FP8 KV cache runs on Hopper warpgroup tensor
  cores (`flash_prefill_f16_hd256_wg_{kv8,kvfp8}` + `_combine`: producer warpgroup widens the raw bytes, two consumer warpgroups run `wgmma`
  QK^T and PV, the key range of a block is split over up to 4 blocks), chosen by `paged_flash_prefill_wg_steps` for the one-sequence paged
  chunk (`KvKernels::native_flash_prefill`) where the part has `wgmma`; `set_flash_prefill_wg(false)` keeps version 2 for an A/B. 512-row chunk
  at 1K / 4K / 8K / 16K / 31K context 0.162 / 0.615 / 1.231 / 2.464 / 4.82 -> 0.088 / 0.219 / 0.387 / 0.606 / 1.088 ms (60-83 -> 110-364
  TFLOPS causal); served 31K TTFT 6.2 -> 4.9 s, the 1K sweep unchanged (52.1 / 214.9 / 365 / 525 tok/s at 1 / 4 / 16 / 64). Open: intra-warpgroup
  overlap of S and the softmax and a 128-key S tile (364 -> ~500 TFLOPS), cluster multicast of the K/V fetch, the bf16 cache (still version 2),
  the training forward/backward (dense forward and `flash_attn_bwd_f16` still on `mma.sync`, unchanged), selection through the measured
  candidate tuner instead of the capability floor (the kernel is chosen where the part has `wgmma`; the tuner has no attention slot yet).
- 2026-10-30 (GDN prefill on bf16 tensor cores, knowledge 384): the Gated DeltaNet serving prefill has a second form, `gdn_chunk_prep_tok_bf16` +
  `gdn_chunk_loop_bf16` (bf16 operands rounded once in the prep, fp32 accumulation, fp32 state in registers, prefetched tiles), chosen per device by
  measurement (`gpu-core/src/gdn_prefill_tune.rs`, slot `gdn_prefill`, `--kernel gdn_prefill=exact|bf16` pins it) and admitted within 4 bf16 steps
  of the exact TF32 form; the triangular solve of both preps goes by panels of eight rows. 48 heads x 8192 tokens a layer 3.83 -> 1.20 ms (prep
  1.66 -> 0.85, recurrence 2.97 -> 0.77 in ncu, before the prep went to three blocks an SM); served output tok/s at 1 / 4 / 16 / 64 52.6 / 213.9 / 390.1 / 535.8 -> 53.1 / 210.0 / 400.5 /
  573.8, 31K TTFT 4.87 -> 4.49 s; `qwen35_fp8_parity` (8 x 1800 tokens) 98.83 % / KL 1.55e-3 -> 99.22 % / 1.19e-3. W8K NF4 train probe 937 -> 981
  tok/s (the checkpointed forward takes it; losses equal to the fourth digit). Open, by weight: a 31K prefill is now 2.13 s FP8 GEMM (0.78 PFLOPS),
  0.51 s attention, 0.25 s GDN, 0.7 s of elementwise kernels around the GDN layers (`mul`, `quant_epilogue`, `quant_e4m3_128`, `add_rms_quant`,
  `rmsnorm_rows`, `gdn_conv_silu_tok`) and 0.8 s of wall clock over the kernel sum; the GDN train-form forward and the backward (436 ms of a 5.4 s
  W8K device step) need a restructuring, not a bf16 flag (9 % / 5 % faster with bf16 operands: bound by staging and barriers); the single-split
  attention at 1K / 4K needs the round's context in `PrefillShape` (about 0.3 ms of a 130 ms iteration).
- 2026-10-30 (mixed-iteration glue, knowledge 385): a mixed round's prompt segments run the Gated DeltaNet mixer as three launches a layer
  (`gdn_conv_silu_seg`, `gdn_chunk_prep_seg_bf16`, `gdn_chunk_loop_seg_bf16`, `model::gdn_segments`), addressed by a per-round table, on the pools' state and
  history rows and the round's token-major tensors in place (no row copies, no gather/scatter, the short remainder is one more chunk of the walk); the gates
  are formed in the prep's loads and `gdn_gated_norm` is the gated norm (+ E4M3 rows for the FP8 out projection) in one pass; the decode rows read the
  projections in place. `MixedLimits::row_quantum` fills the GEMMs' last 128-row tile; a mixed forward reads back only the rows it uses; `topk_rows_f32`
  is the whole 32-candidate head in one launch (2200 -> 209 us). Median mixed iteration at c = 16 (nsys): 119.4 ms / 690 rows -> 100.7 ms / 768 rows
  (-24 % per row), launches of the GDN kinds 184 -> 48, `concat_split` 1209 -> 293; served c = 1 / 4 / 16 / 64 output tok/s 53 / 208 / 395 / 568 ->
  57.9 / 239 / 425-451 / 655. Open, by weight: the FP8 GEMM (61 ms; the 816-block gate/up shape runs at 616 TFLOPS, a 128x256 tile or stream-K would
  address its 7-wave quantisation), the E4M3 quantisation passes (8 ms; fusing into the 1024-thread epilogue measured slower, knowledge 385), the GQA
  glue of a mixed round (~6 ms: five splits, two norms, two rotations, two appends, sigmoid/mul/splice a segment; per-round cos/sin tables and one
  gate kernel), host time before the first kernel (4-6 ms: the embedding decode of the round), the decode step (42 % of a c = 16 run's wall clock:
  `matmul_fp8_dq_stream_r4` at 2.1 TB/s of 3.5).
- 2026-10-30 (decode step, knowledge 386): the FP8 27B decode graph (int8 KV, speculative off) at one sequence is 15.8 -> ~10.6 ms of kernels and the served
  inter-token latency 15.7 -> 11.6 ms. (a) `matmul_fp8_gemv_f16_*`: per-lane `cp.async` A fragments into a warp-private ring, hardware E4M3 -> f16 decode,
  the block scale folded in f32 once per 128 k, first copies before `griddepcontrol.wait` (1-8 rows; r4 2.0-2.4 -> 2.7-3.1 TB/s isolated, 11.8 -> 7.5 ms of GEMV in
  the graph); `matmul_fp8_gemv_f16x_*` takes the rows of x as the A operand for 9-16 rows (16 sequences: GEMV 12.2 -> 9.9 ms). (b) the fused front ends write the
  bf16 rows (256 -> 48 casts a token). (c) the pooled GDN decode is `gdn_conv_pool` + `gdn_recur_pool` (a block per sequence and value head, the state staged by
  `cp.async`; bit-identical): 17.3 / 18.7 / 72.5 / 270 -> 14.9 / 18.5 / 56.8 / 155 us a layer at 1 / 4 / 16 / 64 sequences. (d) the gate projections are one
  multi-GEMV, `gqa_decode_prep_i8` fuses the int8-KV attention front half, `topk_rows` 187 -> 103 us, the head norm 40 -> 2.9 us. Decode step (nsys, c = 1 / 4 /
  16 / 64) 16.0 / 16.2 / 20.2 / 40.0 -> 12.1 / 13.0 / 16.2 / 34.0 ms; served output tok/s at 1 / 4 / 16 / 64 (same client) 58.7 / 242 / 465 / 675 -> 76.5 /
  295 / 529 / 717; 31K TTFT 4.46 -> 4.50 s (unchanged); `qwen35_fp8_parity` identical to base on the same text (98.44 %, KL 2.74e-3), greedy chat outputs 4 of 6
  prompts identical to base token for token (the rest diverge after 300-400 characters, as batch 1 vs 6 does in the base itself); W8K NF4 train probe 1003-1022
  tok/s against 1003-1012 for the base on the same node. Open, by weight: the host (443 us idle before a token's first kernel, 291 us between its two graphs, 88 us
  at the head round trip: a 110 us and a 377 us `cuGraphLaunch`, the last layer read back and uploaded again), 17-64 rows (`dq_stream_r2` 54 us and `wgmma` 54 us
  a big GEMM: 12 + 7.8 ms of a 31 ms step at 64 rows, 1.2-1.7 TB/s; x as the A operand re-reads x once per 64 weight rows), the 48 GDN gated-output casts, the
  103 us top-k (a two-phase form), `add_rms_quant` as a 4-block cluster.

- 2026-10-30 (host time out of the step, knowledge 388): (a) the first stage holds a plain bf16/f16 embedding table (2.5 GB, charged in `serving_cost`, refused above 1/16 of the
  smallest card; GGUF and scaled tables keep the host path) and a decode step / mixed forward uploads token ids only: `mixed.embed` 15.6 ms -> 1 us, `decode.embed` 1.1 ms -> 1 us, c=64 804 -> 837 tok/s, card
  busy 88 -> 92 % (steady window). (b) a mixed round runs the GQA front half and the K/V append once over all its rows and the gate rides on the `o_proj` quantiser: launches per iteration 2625 -> 1972,
  c=64 837 -> 857. (c) the decode step's head reads the layers' output on the card in the same submission: the 105 us layer-to-head gap is gone, step 16.14 -> 15.85 ms at c=16 (served +-1 %).
  Output tok/s c=1/4/16/64: 78.5 / 297.5 / 553.4 / 804.4 -> 78.5 / 299.4 / 568.8 / 850.9; 31K TTFT 4.46 -> 4.08 s; greedy chats byte-identical; W8K NF4 probe 1051 / 1037 -> 1042 / 1048 (not a
  training path). Step plan status: B (arena with stable offsets) and C (graph per row-count bucket, K-step device-resident decode, device sampler) are NOT done; what the 7.8 % idle is and the
  levers in order (decode-step recording per (rows, context bucket) as `decode_tape` does for rings, device greedy sampler, `fresh` GDN segments instead of 144 memsets/scatters a new prompt,
  in-place row offsets for the prefill attention) are listed at the end of knowledge 388. Cross-card edges: `StageRows::Device` / `Gather` already make the first-stage input a plan input, not a host
  array, and `chain_stages_with` hands the stage output on as a device buffer.
- 2026-10-30 (FP8 GEMM family, knowledge 387): (a) 17-64 rows: `matmul_fp8_wgmma_swap_n{64,32}b{64,128}[s{2,4}]` put the WEIGHT rows in the MMA's M and the rows of x in its N (no
  128-row padding, 64 weight rows a block, K split over a cluster of 2-4 blocks for the K = 17408 / N = 5120 shapes). 64-row GEMMs (us): gate/up 89.1 -> 73.9, gate 57.5 ->
  36.7, down 85.7 -> 35.6, qkv 30.6 -> 25.6, out 34.3 -> 18.5 (2.0-2.5 TB/s of weight bytes; the x tile re-read from the L2 is as many bytes as the weights, so ~2.4 TB/s is
  the 5.5 TB/s the SM ingests). Pure decode (tiny prompts, 384 outputs) median ITL at 32 / 64 sequences 24.05 / 33.23 -> 18.15 / 27.33 ms, output tok/s 1255 / 1786 -> 1636 /
  2130; 16 sequences unchanged (the f16x GEMV keeps 9-16 rows). (b) prefill: `matmul_fp8_wgmma_tma` (persistent, TMA tile copies through tensor maps the kernel builds on the
  device from a driver-encoded template, producer thread + two consumer warpgroups on mbarriers, grouped tile order), bit-identical to `matmul_fp8_wgmma`: 616-970 -> 880-970
  TFLOPS at 768-2048 rows (gate/up M = 768 393 -> 308 us), 31K TTFT 4.46 -> 4.11 s. A `cp.async` warp-specialised pipeline, a 256 x 128 tile and double-buffered accumulators
  did not help (knowledge 387). Served 1 / 4 / 16 / 64 sequences (1K in / 128 out, same client, same node, base run first): 76.2 / 295 / 525 / 712 -> 78.2 / 296 / 545 / 800 output
  tok/s. `qwen35_fp8_parity` (8 x 1500 tokens, 32 steps) identical to base: 98.05 % / KL 1.392e-3. Training: the NF4 W8K probe does not dispatch the FP8 GEMM (`--layers 64 --tokens 8192 --rank 16 --base nf4 --residuals bf16 --mlp-rows 512`: 801 tok/s base, 836 now, noise). Open, by weight:
  1.2 PFLOPS needs an N of 192-256 per wgmma on one accumulator set (the loop is bound by the ~96 of 128 B/cycle of shared-memory operand reads, 1.08 PFLOPS without the fold); the
  17-64 row GEMMs' SM ingest ceiling (x multicast over a 2-block cluster); wave quantisation of the 272-tile gate/up shape.
- 2026-10-30 (FP8 prefill GEMM at 128 x 256, knowledge 393): `matmul_fp8_wgmma_tma_n256` (two consumer warpgroups of `wgmma m64n256k32`, one accumulator set with the block scale applied as a
  ratio on the running sum, output through a TMA store, 4 stages) is a measured candidate of the FP8 linear; the tuner takes it for 10 of 96 classes (the large-N and 513+-row ones). TFLOPS at
  M = 768 / 2048 / 8192 / 31000 (`matmul_fp8_wgmma_tma` -> `_n256`): gate/up 894 / 969 / 948 / 818 -> 1030 / 1128 / 1119 / 981, down 906 / 992 / 952 / 803 -> 1144 / 1060 / 1086 / 1053,
  qkv 10240 838 / 912 / 951 / 928 -> 1034 / 1117 / 1091 / 1108, out 822 / 906 / 945 / 938 -> 987 / 968 / 1142 / 1127. Not 1.2: the loop alone on real operands is 1.54-1.68 PFLOPS at 1.6 GHz,
  the full tile flow holds 1.35 GHz under the card's power limit (660 W module), rescale and the f32 output cost 8-10 % each, and latency-hiding changes (scale prefetch, chunked scales, warpgroup
  lag/alternation, a 2-block multicast cluster) moved nothing. Served (base gpu0 / new gpu1 at the same time, 1K in / 128 out, same client) c = 1 / 4 / 16 / 64 output tok/s 80.6 / 249 / 540 /
  757 -> 79.6 / 253 / 545 / 772, 31K TTFT 3.88 -> 3.73 s. `qwen35_fp8_parity` (8 x 1500 tokens, 32 steps, two texts) 98.05 % / KL 2.51e-3 and 98.05 % / 1.74e-3 -> 98.05 % / 3.16e-3 and 99.22 % /
  1.50e-3. Open: K split of the tail tiles (down 17408 x 5120: 119 us at 768 rows but 224 us at 896-1280, two waves of 140-200 tiles; 5 row tiles leave 24 % of the card idle), a bf16 output for the FP8 linears, the 17-64 row swap kernels.
- 2026-10-30 (recorded decode step, knowledge 390): a plain paged decode step (single stage, embedding on the card, base rows) is recorded with its head per (rows, 128-key capacity bucket, greedy/top-k) after 3 steps
  of the same key and replayed (`Qwen35GgufInstance::decode_planned`, `set_decode_plans`, 8 kept); frozen programs are cut into a 64-step graph then 256-step graphs, and fences skip events a stream already waited for and
  programs whose buffers no other stream touched (`Fence::seq`, `CudaBackend::foreign_fences`); `Gpu::reclaim_due` drops the `cuMemGetInfo` of every prefill round. Output tok/s c = 1 / 4 / 16 / 64
  74.0 / 287.8 / 536.3 / 777.6 -> 80.4 / 305.9 / 580.8 / 836.6 (same session), gap between tokens at c = 1 550 -> ~150 us, greedy chats byte-identical. Open: device sampler + K-step decode, `fresh` GDN segments, mixed tail on the card,
  prefill attention row offsets, lifetime-planned arena + graph per row-count bucket for mixed iterations and training, per-stage plans for cross-card chains.
- 2026-10-30 (mixed-iteration audit, knowledge 391): the c=64 figures of shared-seed sweeps (851, 857, 882 tok/s, ITL p50 25 ms) carried prefix-cache hits of the c=16 prompts; with a seed per
  concurrency the same code serves c=64 at 735-745 tok/s (ITL p50 90 ms), c=16 520. New `--itl-target-ms N` (model-agnostic serving limit, Qwen3.8 GGUF engine; others report it ignored): 45 / 90 ms
  at c=64 trade TTFT (3.8 s / 1.2 s) for gaps, throughput 798 / 845 against 851 default (shared-seed numbers), it does not buy throughput. Landed `silu_mul_quant_e4m3_128` (MLP product + E4M3 rows
  in one pass, bit-identical to the pair, 104 -> 48 us a layer, +0.5 % served). Tried and not kept (no un-profiled gain): retaining the pool's working set, an M-RoPE row cache. Open items and the
  per-kernel table in knowledge 391.

- 2026-10-31 (recurrent-state precision): `--gdn-state f32|bf16` (`model::policy::GdnState`, declared in `PolicySupport`; only the qwen35 GGUF/FP8 resident
  declares it; default f32 until a quality gate decides) stores the resident's per-sequence Gated DeltaNet state and conv-history pools as bf16 (two elements a
  word, half the words; `ShardCaches::pool_bf16`). The work copies, the prefix-cache snapshots and the zero row stay f32; pool<->work copies use the bf16
  gather/scatter builds, the decode step the bf16 streaming kernel, the segment prefill kernels take the bf16 pools. The placement charges a running sequence's
  row at half (`ServingShape::gdn_bf16`). Refused up front with `--exact-decode` and any `--speculative` mode (the verify/commit rounds read f32 rows). Tests:
  a bf16-pool multi-sequence prefill round agrees with the sequence alone to bf16 resolution (`model::tests`), policy, placement and refusal unit tests.
  Open: the quality gate that decides the default; a bf16 verify round if speculation over bf16 pools is wanted.

- 2026-10-30 (activation precision, knowledge 394): `--act-precision auto|f32|bf16` (`gpu_core::ActPrecision`, bf16 from compute capability 8.0; a model
  declares support in `PolicySupport`) declares the element type of the activations between the FP8 products. The qwen35 serving forward stores the MLP's gate/up
  and the Gated DeltaNet mixer's qkv/z as bf16 (`matmul_fp8_wgmma_tma_n256_bf16o`, bit-identical to the f32 kernel rounded to bf16), reads them as bf16 in the
  SwiGLU pass (product no longer stored), the segment conv/prep/loop/gated norm (leading decode rows widened by `cast_bf16_f32`), and a front end of its own
  (`add_rms_quant_e4m3`) writes the E4M3 rows of the next FP8 product and leaves `xn` unwritten when nothing reads it. c=64 794.5 -> 835.9 tok/s (+5.2 %), c=16
  553.2 -> 562.5, c=1/4 unchanged, 31K TTFT 3.71 -> 3.53 s; a mixed iteration 84.8 -> 80.2 ms, a decode-only step 25.3 -> 23.5 ms; kernel time -3.9 %.
  `qwen35_fp8_parity` inside the base's spread, six greedy chats identical, W8K probe 1608 tok/s with the same losses. Not reached: the 950 / 650 targets. Tried
  and not kept: wave-aligned mixed-iteration rows (`WaveCost`: 813 -> 790 tok/s at c=64, TTFT -13 %; longer iterations cost every sequence's token interval),
  direct 4-byte bf16 stores from the accumulator (+16-29 % on the GEMM) and bf16 twins of the 128x128 kernel. Open, by weight: the FP8 GEMMs are 52 % of kernel
  time and nothing here moved them more than 4 %; `gdn_recur_pool` (7.7 %, f32 state traffic), the GDN chunk kernels (11 %, bound by the chain), `add_rms_quant_e4m3`
  at 20.8 us against a 15 us byte roof, the bf16 residual (a, sum, b: another ~5 us a front end, needs the stage hand-off, taps and head to read bf16), a bf16 `b`
  from the down/out products, the GQA layers' q/ctx (16 layers, 3 % of the stream), and the 128x128 tile's twin through a TMA store.

Mixed-iteration idle (knowledge 392, 2026-10-30): a prompt start no longer waits on the device (zeroed pool row) and a finished prompt's first token rides the forward's head pass; scheduling floor curve is flat (713-775 tok/s at c=64 over floors 256-2048); remaining idle <= 1.4 ms, so asynchronous scheduling (row 4) is worth ~1.5 %.

- 2026-10-31 (GDN prefill, knowledge 397): prep + recurrence of a layer at 8K tokens 0.146 -> 0.057 us/token (1202 -> 467 us), 700-row round 0.19 -> 0.09. The recurrence runs on
  warpgroup tensor cores where the tuner measures it faster (`gdn_prefill_loop` slot, `mma|wg`): `gdn_chunk_loop_seg_wg_half` (a block a segment, head and 64-column half; the
  single-stage `_lean` when blocks outnumber multiprocessors), folded (the convolution's normalised rows, row scalings applied in the recurrence, u as fp16 in thread order);
  the prep solves by blocked forward substitution on tensor cores; `gdn_conv_silu_seg_vec` normalises query/key once. Open: 700-row round target 0.08 (block latency: 12.9 k cycles
  a prep block alone), the gated norm (150 us at 8K, 250 MB roof 76) and the convolution (195, roof ~100), a concurrent prep/recurrence kernel to keep the scratch in L2, the train-form
  forward (needs T for the backward).

- 2026-10-31 (long-prompt TTFT, knowledge 398): uncached 32.5K prompt first token 3.50-3.60 -> 2.93-3.00 s (scoreboard long row median TTFT 3405 -> 2929 ms, c=1 131 -> 115 ms, c=4/16/64 unchanged:
  c=64 861/862 base vs 863/868 new on the same card). A lone prompt's slice takes the multi-sequence forward (embedding gathered on the card, one row read, fused kernels: idle card 276 -> 29 ms,
  kernel time 3224 -> 2993 ms), reads in at 4096 rows (`MixedLimits::lone_prompt_rows`; shared forwards stay 2048; default context 774144 -> 729088), tokenises on up to 16 threads (87 -> 14 ms)
  and runs the prefill attention unsplit from 2048 rows (2-12 % per call). Request timeline: JSON 1 ms, tokenise 14, 8 rounds 2927, first token out 25. Bound: GEMMs 54 % at 1.0 PFLOPS and
  attention 20 % at 385 TFLOPS under the SW power cap (SM clock 1.62 GHz mean, cap active in 73 % of samples); 2.2 s needs a GEMM near 1.3 PFLOPS and attention past 500 TFLOPS at the same power. Open,
  by weight: GEMM energy per FLOP, attention K/V widening shared by the 6 heads of a kv head (30 % of the kernel), the GQA q/k/v split + norm + rope + append fusion (~0.6 ms a layer),
  the 25 ms between the last round and the first token. A card-to-card gap of 2.5 % (gpu0 faster) confounds A/B runs that put base and new on different cards.

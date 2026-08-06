# `brain perf` — implementation ledger

Design: [`benchmarking.md`](benchmarking.md). This file tracks what is built.

## P1–P5 — all 14 scenarios ✅

**Crate `crates/perf`** (`brain-perf`, lib `perf`), 59 unit tests.

| Module | What |
|---|---|
| `stats` | exact nearest-rank percentiles (P50/95/99/99.9), best-of-N + spread, Jain fairness |
| `env` | the result fingerprint: commit/dirty, device, backend, **adapter + `adapter_is_software`**, CPU model/cores, RAM, OS, build profile, perf-relevant env flags |
| `target` | the `PerfTarget` seam (`submit`/`step`/`busy`/`counters`) + `Emission` timeline |
| `workload` | the 8 standard shapes, length distributions, 4 arrival processes, SLOs, `scaled()` for small devices |
| `driver` | single-threaded load driver: submits per the arrival process, records the timeline |
| `metrics` | `ReqRecord` → TTFA/IAL/TPOA/E2E/queue; `Summary` with goodput and SLO attainment |
| `schema` | the `brain.perf/1` artifact + `valid`/`invalid_reason` |
| `report` | one-run render, and `compare` with unit-refusal / invalid-exclusion / axis warnings |
| `scenarios` | `latency`, `throughput`, `serve`, `sweep` + `Options` |
| `targets` | `CapabilityTarget` (any `capability::Provider`) and `PagedLlmTarget` (paged serving engine) |

Five CLI targets: `fake` (harness self-check), **`qwen-synth:<L>x<D>x<H>[xV]`**
(the real paged engine on random weights — same kernels/KV/batching, no
checkpoint needed, so hardware comparison works on any machine),
`qwen:<weights>` (a real checkpoint), `lfm:<weights>:<tokenizer.json>` (the
LFM2.5 encoder behind the residency executor; unit `sequence`), and
`flux2[:<W>x<H>x<steps>]` (FLUX.2 Klein behind the residency executor, weights
from the `BRAIN_FLUX2_*` env; unit `denoise_step` — `ExecutorTarget`'s
streaming mode timestamps each in-flight "denoising" `Progress` as one
artifact; measured numbers in `docs/models/flux2/status.md`).

**CLI** `brain perf {list,run,compare}` (`crates/cli/src/perf_cli.rs`).
**Make** `perf`, `perf/<scenario>`, `perf/compare`, `perf/smoke`.

### Engine changes this required

- **`qwen::serve::Scheduler::step_report`** — `step()` returns only *completed*
  requests, so nothing downstream could observe when a sequence was admitted or
  when each token appeared, making TTFT and ITL uncomputable. `step_report`
  additionally reports admissions and per-sequence token deltas; `step` is now a
  thin wrapper over the same `step_inner`, so behaviour is unchanged. Two tests
  pin it: every token accounted exactly once, and outputs identical to `step`.
- **`Scheduler::free_blocks`** — KV pressure alongside the latencies.
- **`gpu_core::adapter_info` / `backend_name` / `discrete_gpu_count`**, backed by
  `backend_wgpu::adapter_desc` — the adapter was previously *logged to stderr and
  discarded*, so no result could record what hardware produced it.

### Verified

- 59 `perf` unit tests + 16 `qwen` lib tests, green in parallel.
- End-to-end on the `fake` target: `perf run {latency,throughput,serve,sweep}`,
  artifact written, `perf compare` ranks, warns on differing axes, refuses to
  rank across units, excludes invalid runs.
- **Not** verified against a real model: no checkpoint is available in this
  workspace, so the `qwen:<weights>` target path is compiled and unit-covered but
  has not produced a real measurement.

### Known limitations

- The correctness gate (P2) does not exist yet, so every artifact currently
  reports `correctness.passed: null` and the renderer prints
  *"correctness gate did not run — result is unverified"*. This is deliberate:
  an ungated run must never be mistaken for a verified one.
- `resources` (device utilisation, energy) and most of `memory` are `null` — the
  counters do not exist in the engine yet.
- `latency` currently runs through the same closed-loop driver as `serve` with a
  fixed level. A true in-process fixed-batch path that bypasses the driver is
  worth adding when it starts mattering for regression signal.

## First real findings (2×Tesla P40, 48-core host)

The suite's first run on real hardware found a decode bottleneck that is not
in the GPU at all.

**Batching barely helped.** `qwen-synth:8x512x8` (46.8M params, fp32),
decode-bound (8-token prompt, 64-token output), `--device gpu0`:

| concurrency | out tok/s | TTFA p99 | IAL p99 |
|---:|---:|---:|---:|
| 1 | 36.4 | 63 ms | 32 ms |
| 4 | 46.9 | 211 ms | 91 ms |
| 16 | 50.3 | 811 ms | 279 ms |
| 32 | 45.4 | 972 ms | 410 ms |

Throughput gains 1.4× from concurrency 1→16 and then *regresses*, while
latency grows 15×. A 47M fp32 model at ~36 tok/s is roughly **1% of the
P40's fp32 peak** — overhead-bound, not compute-bound.

**Cause: the LM head ran on the host, single-threaded.**
`qwen::serve::Engine::logits` is a scalar Rust matmul over
`[vocab, d_model]` — 16.4M multiply-accumulates per sequence per token at
vocab 32k — executed once per sequence per decode step while the GPU sits
idle. Batch size multiplies that host cost linearly, which is exactly why
batching stops paying.

Confirmed causally by varying **only** vocab (same shape, same device):

| vocab | out tok/s | IAL p50 |
|---:|---:|---:|
| 4 000 | 126.3 | 18.3 ms |
| 8 000 | 100.7 | 23.3 ms |
| 32 000 | 40.6 | 70.3 ms |

`IAL ≈ 11 ms + 1.86 µs per 1000 vocab`, so at vocab 32k about **85% of each
decode step was the host-side head**.

### Fixed: the head now runs on the device

`matmul` (parallel over every one of `bsz × vocab` outputs) followed by a new
`argmax_row` kernel, so the hidden state never leaves the device and only
`bsz` indices are read back instead of a `[bsz, vocab]` block. Prefill keeps
the host head — it runs once per request, not once per token.

Same commands, same hardware, `qwen-synth:8x512x8`, `--device gpu0`:

| concurrency | before | after | gain |
|---:|---:|---:|---:|
| 1 | 36.4 | **74.2** | 2.0× |
| 4 | 46.9 | **155.3** | 3.3× |
| 16 | 50.3 | **234.9** | 4.7× |
| 32 | 45.4 | **290.5** | 6.4× |

Batching now scales **3.9× from concurrency 1→32** (was 1.25× and
*regressing* past 16), and the workload meets its SLO at low concurrency
instead of never — `sweep` reports a real max-sustainable point of
concurrency 4 at 155 tok/s goodput.

Vocab sensitivity is largely gone, which is the direct confirmation:

| vocab | before | after |
|---:|---:|---:|
| 4 000 | 126.3 | **176.9** |
| 8 000 | 100.7 | **168.7** |
| 32 000 | 40.6 | **114.6** |

8× vocab now costs 1.6× IAL (10.8 → 17.7 ms) instead of 3.8×.

Correctness is pinned by `device_head_argmax_matches_the_host_head`, and the
pre-existing token-for-token reference tests
(`batched_serving_matches_reference`,
`scheduler_dynamic_admission_matches_reference`) still pass — the engine
generates identical text, faster. Verified on both backends: the same WGSL
runs on the CPU backend via the Cranelift JIT.

**Still on the table**: at 290 tok/s for a 47M fp32 model the P40 is still far
from its ~11 TFLOP/s peak, so the next bottleneck is worth finding — likely
per-step dispatch/sync overhead, since IAL p99 stays ~48 ms at batch 32.

*(Numbers are `--target qwen-synth`, i.e. random weights — valid for cost,
meaningless for output quality.)*


### Third wave: decode-regime kernels + the fidelity gate (all landed)

Same sweep, same hardware (`qwen-synth:8x512x8`, Tesla P40, decode-heavy):

| concurrency | session start | + device head | + prefill budget | **+ decode kernels** | total |
|---:|---:|---:|---:|---:|---:|
| 1 | 36.4 | 74.2 | 72.6 | **154.0** | 4.2× |
| 4 | 46.9 | 155.3 | 154.5 | **381.7** | 8.1× |
| 16 | 50.3 | 234.9 | 234.1 | **632.7** | 12.6× |
| 32 | 45.4 | 290.5 | 289.6 | **865.5** | **19.1×** |

IAL p99 at batch 32: 410 ms → **11.8 ms**. The interactive SLO now holds
through concurrency 16 at 632 tok/s goodput (session start: no level at all).

What landed: `matmul_gemv` (workgroup-per-column, W streamed once across all
rows — the 67.5% kernel), `rmsnorm_rows` (workgroup-per-row — the 16.6%
kernel), `argmax_part/final` (two-stage reduction — the 10.3% kernel), each
selected per dispatch by row count so training/prefill shapes keep the
per-element kernels. Gated off the CPU backend, whose JIT barrier-split model
mis-executes them and whose native AVX2 fast paths already cover the regime —
`Backend::kind()` makes the selection explicit.

**Every number above is gated**: the fidelity check (batched-vs-sequential
greedy through the same engine) now runs inside every Tier-1 scenario and
reported `greedy_token_match: 1.0` — the artifacts are marked valid, not
"unverified".


### Fourth wave: policy seams (landed)

- **AdmissionPolicy** (`qwen::serve`): `UnboundedQueue` (default, historical),
  `MaxQueueDepth`, `DeadlineAware` (EWMA-fed). Consulted at submit; typed
  `RejectReason` in `StepReport::rejected`. Policies are pure functions of
  `QueueState` — unit-tested with no engine. Remaining: a `--admission` flag so
  `perf overload` compares the ladder per policy.
- **EvictionPolicy** (`residency::place`): `Lru` + `CostAware` (GDSF,
  `uses x bytes / age`); `plan_eviction_with`, benchmark scores the real code
  via `--policy`. Hit rate 50.0% -> 54.3% on identical seeds; cheap-before-
  expensive eviction pinned by test. Event-counted regret is metric-limited at
  4x overcommit — a bytes-weighted regret metric is the follow-up.

## Fixed along the way (pre-existing)

- **SIGSEGV in every debug-profile test run that built GPU devices on more than
  one thread.** wgpu's default `InstanceFlags` are `from_build_config()`, which
  enables `DEBUG | VALIDATION` whenever `debug_assertions` is set — so `cargo
  test` silently turned on the Vulkan validation layers and `VK_EXT_debug_utils`
  object naming while `make release` did not. On a software Vulkan ICD
  (lavapipe) `vkSetDebugUtilsObjectNameEXT` faults inside `libvulkan.so.1`.
  Instance flags are now **opt-in** via `BRAIN_GPU_VALIDATION=1`, so debug and
  release build the same instance. Backend construction is also serialised
  (`init_lock`), since concurrent entry into Mesa's EGL/GL loader is unsafe.
- **Multi-GPU tests faulted instead of skipping** on single-GPU / GPU-less boxes:
  seven `dp_parity` / `shard_parity` / `shard_microbatch` / `tensor_parallel`
  tests assumed cards 0..n exist and died inside the driver. They now gate on
  `gpu_core::discrete_gpu_count()` and skip with a message.

## Tier 2 — what each scenario can and cannot see today

All ten are implemented and run. Each is explicit about its limits: where a
metric needs an engine capability that does not exist, the field is `null` and
the artifact carries a `notes` string. A confident number nobody measured is
worse than an honest gap.

| scenario | measures today | blocked on |
|---|---|---|
| `startup` | device init / weights / first prefill, cold vs second build | a pipeline cache — there is no warm path, and the two rows matching is the evidence |
| `mixed` | per-class goodput, P99 TTFA/IAL, **normalised slowdown** vs an isolated baseline, Jain fairness | — |
| `overload` | capacity, then 0.5–4× offered load; goodput, queue P99, wasted admissions | a pluggable **admission policy**: nothing is ever rejected, so policies cannot be compared |
| `cancel` | abort latency, block reclaim, waste, neighbour interference | an async client-disconnect path (transport-level) |
| `kvcache` | admission pressure: KV stalls and stall time under a working set 3× the pool | **prefix caching / cross-request block reuse** — without it hit-rate and eviction-regret are structurally null |
| `residency` ★ | warm/cold TTFA, weight-cache hit rate, **eviction regret**, per-model fairness under a shifting Zipf | wiring to real `ResidentModel`s for true load latency (activation is modelled from size) |
| `placement` | `placement_efficiency` = best combined / best single | nothing — but `--device` is process-global, so it analyses per-device artifacts |
| `frontend` | JSONL encode/decode and chat-template cost; host cores per saturated device | tokenizer + media inputs for the remaining stages |
| `faults` | device-OOM injection, detection, no silent corruption | a **multi-rank harness** for worker death, hung ranks, collective timeouts, corrupt KV |
| `soak` | throughput/latency/memory/KV series; drift per hour | nothing — but it **refuses to extrapolate** below 600 s |

Cross-cutting: `fidelity` (greedy-token gate; a failing run is written
`valid: false` and excluded from `compare`) and `energy` (external power
sampling — an unreadable meter yields `null`, never a fabricated zero).

### Second wave of findings (2×Tesla P40)

- **`residency`: 64% eviction regret.** 24 models at 4× overcommit under a Zipf
  load whose popularity shifts mid-run: LRU evicts models wanted again almost
  immediately about two thirds of the time, hit rate 48%, fairness 0.494 with
  every model still served. That is the "bad policy" signature rather than the
  "cache too small" one — the distinction the regret metric exists to draw.
- **`startup`: there is no warm start.** A second engine built in the same
  process costs the same as the first (~3.1 s vs ~3.5 s, i.e. no reuse), because
  every `Engine` constructs its own `Gpu` and recompiles every WGSL pipeline.
  Device init ~0.3 s, weights ~1.8 s, first prefill ~1.0 s.
- **`frontend`: the host is not the bottleneck.** JSONL + templating cost is
  negligible against a ~290 tok/s device rate (well under 0.01 cores), so the
  ceiling found earlier is genuinely in the engine.
- **`cancel` is clean**: zero waste, zero leaked blocks, neighbours unharmed.
- **An engine crash, found by `kvcache`.** A prompt longer than
  `max_blocks_per_seq * block_size` wrote **past its row of the block table**
  (index 82944 of 82944) — silently corrupting the next sequence's mapping
  before it panicked. Now `Engine::max_seq_len` bounds it and the scheduler
  rejects oversized requests at admission, reporting them in
  `StepReport.rejected`, so the queue keeps moving instead of crashing or
  blocking forever.

## Mitigations

Every finding above has a designed mitigation with a root cause, an API sketch
and a verification criterion in **[`mitigations.md`](mitigations.md)**, organised
around reaching *any* device's ceiling rather than this box's.

Profiling shows decode is **64% GPU-busy**, so it is not host stalling. Three
readings, in increasing order of importance:

1. The hot kernels (`matmul` 67.5%, `rmsnorm` 16.6%, `argmax_row` 10.3%) all
   parallelise over output *rows* — right for training, catastrophic for decode.
   `rmsnorm` runs **8 threads on 3840 cores** at batch 8.
2. Decode is **memory-bound, not compute-bound**: at M=8 arithmetic intensity is
   ~0.25 FLOP/byte whatever the tiling, so *tiling cannot fix it*.
3. Therefore the first-order lever is **bytes moved**, and it is the one lever
   every device class shares. `matmul_i8` (DP4A, 4×) and `qwen::q8` already
   exist — but `serve.rs` uses int8 for the **KV cache only, never for weights**.

The structural blockers are that `Backend` exposes almost no device capability,
kernel choice is fixed at model-construction time, tile sizes are literals in
281 hand-written files, and the fp32-only invariant — which is what buys the
old-GPU/WebGPU guarantee — is simultaneously the ceiling on modern hardware.
`mitigations.md` Part I addresses those; Part II expresses each finding through
them.

## Closed since (suite reliability + duplication)

- **Test-suite deadlock and exit-crash fixed** (commit 2fea497): one device per
  process via explicit `share`/`new_like`/`WeakGpu`; the weak-pool test fixture
  (`gpu_core::testgpu`) lets the device die with its last in-process handle —
  the only teardown shape this NVIDIA driver survives. 30/30 parallel runs
  clean at 8 and 48 test threads (was ~50% deadlock).
- **rayon centralised** (5b0e539): `backend_cpu::par` is the one home for
  host-parallel loops; `grep -l '^rayon' crates/*/Cargo.toml` returns exactly
  `backend-cpu`.
- **Host math centralised** (508d84d): `model::hostmath`, parity-tested against
  the WGSL kernels through the CPU backend.
- **ONNX emission centralised** (0f660d7): `npu::topo::TopoBase` — one DSL and
  one rmsnorm/layernorm/silu emitter across the eight topology builders.

## Closed since (the portability spine + the remaining serving mitigations)

One session (commits 4ca517b..07f28e2), in roadmap order:

- **S1 `DeviceCaps`** (4ca517b): a queryable capability model on every backend —
  class, limits, subgroup width, unified memory, and a `NumericSupport` tier
  list where every value is queried, never assumed (fast-f16 stays `false`
  until measured: Pascal exposes f16 at 1/64 rate). Recorded machine-readable
  in every perf artifact's env block. Also fixed en route: the wgpu and Vulkan
  backends never overrode `max_storage_binding_bytes`, so every caller saw a
  fixed ~2 GiB instead of the device's real limit.
- **S2 `KernelSelector`** (ff2612c): one pure, memoised policy for which kernel
  variant runs per (op, shape, caps) — the scattered `kind() != "cpu"` regime
  tests are gone, and the CPU JIT's inability to run workgroup reductions is an
  honest capability (`workgroup_reductions`), not a string compare.
- **A0 int8 serving weights** (5675a08): `--weights-int8` / target suffix
  `:i8w`; the 7 per-layer linears + LM head quantize at load (per-channel,
  packed 4/u32), activations quantize per forward, fp32 copies are not kept.
  Measured on the P40 (decode_heavy): +32% throughput at c=16 (753 vs 570
  tok/s) and TTFA p99 631 ms vs 2386 ms — but SLOWER at c=1 (78 vs 127),
  because a 128×128 tile is mostly idle at m=1. That gap forced the next item:
- **`matmul_i8_gemv`** (same commit): the decode-shaped packed GEMV; int8 c=1
  now 131 tok/s. The GEMV/tile crossover is measured at m≈8 and lives in the
  selector (`I8_GEMV_MAX_ROWS`) — refining it per-device is S5's job.
- **F1/F2 warm start** (5675a08): `Engine::from_map_on` (one device per
  process, engines share via `new_like`) + a persisted per-adapter driver
  pipeline cache. `startup`'s two identical rows finally separate: device
  953.9 → 239.7 ms, total 4117 → 2761 ms.
- **Admission comparison** (5675a08): `overload --admission
  {unbounded|depth:N|deadline:ms}` installs real engine policies; rejections
  flow through `EmissionKind::Rejected` — terminal, never goodput, excluded
  from the SLO denominator. An uninstallable policy is an error, not a silent
  fallback. `residency` adds **bytes-weighted** eviction regret (a good policy
  makes the regretted evictions the cheap ones) and fixes a regret-attribution
  bug: a re-request now resolves its own eviction entry, not the most recent.
- **A4 on-device decode window** (42ab2fa): decode feeds the argmax back and
  advances paged metadata on the device for up to 4 tokens per host
  round-trip; windows engage only when nothing is waiting, and a mid-window
  EOS's surplus K/V rolls back. fp32 c=1 127 → 137 tok/s, c=4 286 → 327.
  Also implemented `dot4I8Packed` in the CPU JIT (it previously failed
  `Jit::new` hard on any kernel using it).
- **D prefix cache** (ed3f516): full prompt blocks indexed by (parent physical
  block, token ids) — exact by construction, no hash to collide; prefill
  adopts the longest chain and computes only the tail; LRU eviction under
  pool pressure, live sequences outrank the cache. `kvcache`'s hit-rate
  counters are now real (`kv_prefix_hit_rate`, null until measured). Writing
  its property test exposed a real hole: token ids were never validated, and
  an out-of-vocab id sent the embedding gather out of bounds — silent garbage
  under the trusted kernels. Now a typed `RejectReason::InvalidToken` at
  admission plus a prefill backstop.
- **testgpu adoption** (07f28e2): qwen/gpt/tts/speaker test construction on
  the shared weak-pool device (qwen proven at `--test-threads=8` on the P40);
  codec deliberately stays on its production CPU pin.

## Closed since (S3/S5 + the remaining scenario items)

- **K DeviceStats** (de63006): submit/dispatch/readback counters always on
  (relaxed atomics), queryable per handle, recorded as a `device_ops` block in
  artifacts — null where a backend does not count, never zero.
- **S3 kernel templating** (4e1727e): `kernels::template` rewrites tunable
  `const` declarations + the `@workgroup_size` literal, so a specialised
  kernel is just another (name, source) pair through all three backends.
  Byte-identity with no params; unknown parameters error.
- **S5 autotuner** (3673cf9): `select::candidates` is the one variant list
  (the static policy is its head BY CONSTRUCTION); `AutoTuner` resolves
  memo → per-adapter persistent store → measurement; `BRAIN_NO_AUTOTUNE=1`
  forces static (tracked env flag). qwen tunes its int8 GEMV/tile crossover
  at build; measured winners persist per adapter + kernel-source fingerprint.
  Wiring it exposed and fixed a silent regression: the int8-weights gate had
  begun probing the selector's head instead of the capability, falling back
  to fp32 — masked by a skipping test that now fails on capable hardware.
- **J2 perf gate** (8954804): `brain perf gate` — hard floors (throughput)
  and ceilings (latency) at a generous fraction; refuses incomparable pairs,
  smoke runs, correctness-failed runs, and zero-check vacuous passes.
- **I frontend + H placement** (same commit): real GPT-2 BPE tokenise/
  detokenise measured (tokenise is the honest bottleneck stage; host still
  ~0.002 cores per saturated device); placement's notes state its
  multi-model scope truthfully.
- **G fault injection** (7fdd726): weight-read failure and a feature-gated
  kernel-dispatch failure (757 ms measured recovery) inject for real;
  host-OOM proved uninjectable in-process (Linux overcommit) and skips with
  that reason. 3 injected / all acceptable / 5 honest skips.

## VLM serving (fastvlm) — first measurements

`fastvlm` (import-complete: token-for-token caption parity vs HF) now serves
through the capability contract everywhere: `brain do`, `brain serve --dbus`
(image in as a sealed memfd, caption back over an fd, correct description of
the test image), and the residency executor's scheduler. Measured on the
2×P40 box, release build, FastVLM-0.5B, 512px input:

- single caption: ~122 s wall — the fp32 MobileCLIP-L tower at 1024 px on the
  CPU backend dominates (it OOMs a 24 GB card as fp32 GPU activations, which
  is WHY it runs on CPU; the parity test made the same choice);
- 3 concurrent captions over dbus: wall 252 s ≈ 3 × the ~84 s per-request
  time — the provider's hot-model mutex serialises whole requests, so there
  is NO cross-request overlap today (an earlier revision of this note
  misread the same numbers as 2.35× concurrency; the arithmetic says
  serial). Real concurrency needs per-request decoder instances (KV caches
  are per-instance state) — an int8 decoder at 4× less VRAM is what makes
  N-instance residency plausible;
- the named levers for the next multiple: a streamed/chunked GPU vision
  tower (bounded stage activations) and int8 tower weights — S3/S5-shaped
  work on the vision side.

### Second profile: the decoder was O(T²) — fixed with KV decode

`BRAIN_PROFILE` on the caption showed 96.5% of GPU time (28.9 s of 30) in
`matmul_tile`: the caption loop recomputed the FULL sequence per token
(`logits_all` — the parity harness's decode, never meant for serving). The
fix is generic, not fastvlm-specific:

- **`Qwen::step_embed`** — KV-cache decode from a RAW embedding, the seam
  every VLM front-end needs: prefill walks text tokens via `step` and image
  rows via `step_embed`; no residual splice on this path. Gated by
  `step_embed_matches_step` (an embedding row is a bit-exact stand-in for
  its token).
- **Int8 KV decode** — `decode_at` no longer asserts fp32-only: under the
  int8 tier every linear runs quant + the single-barrier packed GEMV at
  m=1 (the measured decode-regime shape), so int8 KV decode runs on the CPU
  JIT as well as the GPUs. Gated by `int8_kv_decode_tracks_fp32` (rel L2 <
  10%) on both backends.
- `decode_at` also gained the Qwen2 gates the batched forward always had
  (attention biases, optional QK-norm) — it was Qwen3-only, which the
  FastVLM Qwen2 decoder exposed immediately.

Caption end-to-end (P40 + 48-core host, release, 8 tokens): **122 s → 40.9 s
(3×)**, identical text. The CPU vision tower (~35 s) is now the entire
bottleneck. `--precision int8` runs the caption on the quantized decoder
(same convention as z-image's `precision` param; qwen `generate` takes it
too via `Qwen::load_inference_i8`) — its steady-state decode cost is
negligible here; the one-time load-hot quantisation pass (~9 s) amortises
in serving.

qwenvl and moondream3 checkpoints are re-fetched to
`/data/workspace/resources/vl/` (the layout their parity harnesses stream
from); their serving waits on full-depth import completion — in-flight work.
No NPU hardware exists on this box, so NPU scheduling stays unverified here
rather than faked.

### Third profile pass: the resident-model blind spot, then three simple fixes

Profiling a RESIDENT model was impossible by construction — both backends
printed their tables only at drop, and a provider held in a static never
drops. `Backend::dump_profile()` (both backends) + per-stage wall timings in
the fastvlm provider close that. What the first readable profile showed, and
the three deliberately-simple fixes (each gated, each measured on the P40):

| Finding | Fix | Effect |
|---|---|---|
| 285 KV prefill positions each paid a submit+fence+readback for a hidden state nobody reads | `Qwen::prefill(&[PrefillInput])` — submit every position, read ONCE at the end; bit-identical to step-by-step (gated) | part of prefill drop below |
| decode was 277 ms/token — the single-threaded host LM head (vocab 152k × d896) | `hostmath::matvec_par` (row-parallel via the CPU scheduler's primitives), shared by the caption loop and `generate_kv_stream` | decode 2.2 s → 0.67 s |
| `rmsnorm` 2.16 s over 13 573 one-thread-per-row calls + per-element m=1 matmuls: `decode_at` never got the A1/A2 decode-regime kernels serve.rs got | dispatch `rmsnorm_rows`/`matmul_gemv` in KV decode where `caps.workgroup_reductions` holds (the selector's m=1 policy at the always-m=1 call site) | prefill 21.8 s → 12.0 s |

Caption end-to-end: **41.8 s → 34.4 s**, identical text. The provider is also
now compartmentalized as two Active-Object-style stages — the vision device
and the decoder each behind their OWN lock, held only while that stage runs,
embeddings handed off by value — so concurrent requests pipeline
(throughput → max(stage) instead of sum(stages)) instead of serialising on
one whole-request mutex. Remaining prefill cost is dispatch/uniform churn
(~148k fresh uniforms+bind groups per request) — batched prefill chunks and
uniform reuse are the next, NOT-simple levers, alongside the GPU tower.

## The served path, measured for the first time (`HttpTarget`, M0 baseline)

Every scenario above drives `qwen::serve::Scheduler` or `residency::Executor`
directly. Nothing had ever driven the actual thing a client talks to:
`apiserve::router()`, over real HTTP framing, through the real admission race in
`apiserve::bridge`, into the real `QwenResident`/`QwenInstance` chat path
(`crates/cli/src/resident_llm.rs`) — the gap
`.todo/serving-performance-audit.md` names directly: a synthetic in-process
benchmark reporting healthy numbers while a real agentic client saw 600+ s.

**`perf::targets::HttpTarget`** (`crates/perf/src/targets.rs`) closes that gap:
it calls `apiserve::router()` in-process via `tower::Service::oneshot` (no
socket), sends a real streaming `stream: true` OpenAI chat-completion request,
and times each SSE `delta.content`/`delta.reasoning_content` chunk as it is
read off the response body — a real TTFA/ITL timeline from the wire, not a
post-hoc replay. Selected via `--target http:qwen-synth:<L>x<D>x<H>[xV]:<tok>`
or `http:qwen:<weights>:<tok>` (`crates/cli/src/perf_cli.rs`).

### M0 baseline: real Qwen3-0.6B, Intel Arc iGPU (Meteor Lake), release build

`chat/in24/out12` workload, `--smoke`, `--ladder 1,2`, 4 requests/level, warm-up 1:

| concurrency | ttfa p99 | ial p99 | e2e mean |
|---|---:|---:|---:|
| 1 | 13 211 ms | 80.6 ms | ~15.2 s |
| 2 | 36 469 ms | 79.0 ms | ~30.3 s |

TTFA p99 **2.76×** for **2×** the concurrent load — the audit's "no batching
across concurrent requests" finding (`.todo/concurrent-request-batching.md`),
now measured through the real served path with real weights, not inferred from
reading the code. `ial` (per-token gap, once decode is running) stays flat —
consistent with concurrent requests **serializing** on one lane rather than
sharing a batched forward: decode-once-running is unaffected by a second
request, but a second request cannot even START until the first's entire
generation finishes.

### CPU (same box, same weights, release build)

The `http:` sweep above could not be completed for CPU within a practical
session time budget — real Qwen3-0.6B decode on this backend is currently
**≈1.9 tok/s** (measured directly: `brain qwen infer`, load 5.1 s, 16 tokens in
8.3 s), so a 2-level concurrency sweep at 4 requests/level is itself a
multi-minute measurement; left as a follow-up with a longer time budget rather
than truncated/estimated here. **`docs/lessons.md`'s standing rule holds**:
this number is reported as what it is (a direct-CLI decode rate, not an
HTTP-path sweep) rather than extrapolated into one.

**A separate, serious finding surfaced while reproducing this**: `BRAIN_DEVICE=cpu`
segfaults intermittently (~2 times in 3) in `crates/qwen/src/model.rs`'s
single-sequence decode path (`Qwen::from_reader_decode` + `generate_kv_stream`
+ `decode_submit`) — a real memory-safety bug in `backend-cpu`'s rayon-parallel
JIT kernel dispatch, reproducible with the plain pre-existing `brain qwen infer`
CLI, nothing to do with `HttpTarget`. The paged `qwen::serve::Engine` path did
not reproduce it in the same checks. Filed in full at
`.todo/cpu-backend-jit-dispatch-segfault.md` rather than patched blind — a
crash of this kind needs a dedicated root-cause pass, not a fix grafted onto a
measurement workstream.

### What this baseline is for

It is the `perf gate` floor the concurrent-serving-performance workstream
measures every subsequent change against (rewiring the LLM residents onto the
paged engine, continuous batching, prefix reuse) — the definition of done is
this same `http:` sweep, same workload, TTFA no longer scaling with
concurrency.

### M1: two unwired fixes (landed, not yet re-measured against M0)

- **The LM head is now read once, at `activate`, not once per request.**
  `crates/cli/src/resident_llm.rs`'s `QwenInstance` and `crates/qwen/src/
  caps.rs`'s `Hot` resident both now store the head (`model.read_weight(
  model.cfg.head_weight())`) and call `generate_kv_stream_with_head` instead
  of `generate_kv_stream` — the fix `sample.rs`'s own doc comment already
  asked for (594 MiB device→host re-read at Qwen3's real vocab/d_model,
  paid on every single chat request). Gated by the existing
  `with_head_matches_the_self_reading_wrapper` test (bit-identical output).
- **Eviction defaults to cost-aware, not strict LRU.** `ResidencyManager`
  gained an `eviction: Box<dyn EvictionPolicy>` field (default `CostAware`,
  overridable via `with_eviction_policy` for an LRU A/B); `claim`/`placeable`
  now call `plan_eviction_with(&*self.eviction, ...)`. `CostAware` was already
  written and measured (`perf residency`: 54.3% vs 50.0% hit rate under a
  shifting Zipf load) but unreachable from the live server until now — a
  `docs/lessons.md` #8 instance (a fast/better path that existed and wasn't
  wired). All 34 residency tests, including the LRU-shaped
  `three_models_on_one_gpu_swap_by_lru`, still pass unchanged (CostAware
  agrees with LRU whenever candidates have equal size/use count, which is
  every existing test's setup).
- **Minimal serving observability**: `executor::Stats` gained `admitted`
  (cumulative, moves the instant a job is claimed onto a lane — as opposed to
  `jobs`, which only counts once `Done` arrives) and `queue_depth` (live
  gauge, unlike the never-resetting `queue_peak` high-water mark). Flows
  through `stats::build::executor_stat` and renders in `braintop`'s executor
  row automatically.

Not yet re-measured against the M0 `http:` baseline above — that is the next
step, once the W2/W5 rewiring lands and the comparison is apples-to-apples
against the same resident architecture the fixes were made in.

### M3: `QwenResident` rewired onto the paged engine — the M0 regression is reversed

`crates/cli/src/resident_llm.rs::QwenResident` now builds a persistent
`model::serve::Scheduler<qwen::serve::Engine>` at `activate` for any
safetensors checkpoint (the common case — a `.gguf` checkpoint keeps the
original single-sequence decode path, since `qwen::serve::Engine` only reads
safetensors; a real, pre-existing gap, not one this pass could fix in
scope). `QwenInstance::run_batch` submits every invocation the dispatcher
handed it into the SAME scheduler and drives them to completion together,
streaming each sequence's own delta text via `Scheduler::tokens_of` — real
continuous batching for whatever the dispatcher grouped into one call (not
yet across separate calls arriving over time — see
`.todo/continuous-batching-executor-seam.md`).

**Re-measured against the exact M0 baseline workload, real Qwen3-0.6B, same
box:**

| concurrency | ttfa p99 (M0, old resident) | ttfa p99 (M3, rewired) |
|---|---:|---:|
| 1 | 13 211 ms | 2 952 ms |
| 2 | 36 469 ms (2.76× worse) | **1 063 ms (2.8× BETTER)** |

TTFA at concurrency 2 is now *lower* than at concurrency 1 — the exact
reversal of M0's finding, and direct, measured proof that concurrent requests
now batch instead of serializing. (The absolute concurrency=1 number also
dropped sharply — 13.2 s → 3.0 s — from W1's LM-head hoist plus the paged
engine's on-device greedy head, neither of which the M0 resident had.)
`req/s`/`out/s` scale better than linearly from 1→2 concurrent requests
(0.3→1.1, 0.6→2.3), consistent with shared-forward-pass batching rather than
per-request overhead amortizing alone.

**A real bug found and fixed during this rewiring, with a regression test**:
`QwenInstance::run_batch`'s driving loop never handled `StepReport::rejected`
— a request the scheduler refuses at admission (e.g. a prompt token outside
the model's vocabulary) never appears in `completed` and never will, so
without handling it the loop spun forever on an otherwise-empty scheduler.
Caught live reproducing this exact milestone (a real tokenizer's chat-template
special tokens exceeded a small synthetic test vocab); fixed, and gated by
`resident_llm::tests::rejected_admission_resolves_promptly_instead_of_hanging`,
verified to actually hang without the fix (reverted it, watched the test time
out, restored it).

### M2: architecture-agnostic paged scheduler + real (non-greedy) sampling

- `crates/model/src/serve.rs` (new): `PagedDecoder` trait + a generic
  `Scheduler<D>` — moved verbatim from `qwen::serve`, which now re-exports the
  same names as a type alias (`qwen::serve::Scheduler = model::serve::
  Scheduler<Engine>`). Zero changes needed at any existing call site
  (`perf_cli.rs`, `perf/src/targets.rs`, `perf_engine.rs`). All 22 pre-existing
  `serve.rs` tests pass unchanged — this was a pure extraction, not a rewrite.
- Real sampling (`temperature`/`top_k`/`top_p`), previously entirely absent
  from the batched serving engine: `Scheduler::submit_sampled` alongside the
  unchanged greedy `submit`. Minimizes device→host traffic via a new small
  kernel (`topk_extract_step.wgsl`) composed with the existing `argmax_part`/
  `argmax_final` pair to extract each row's top-K candidates in ONE GPU
  submission — `[bsz, 64]` read back per decode step instead of `[bsz, vocab]`
  (>1000× less data at real vocab sizes) — with the actual sampling decision
  made on the host. See the plan's W3 section for the full design and why a
  fully-fused on-device kernel was deliberately not attempted this pass.
  Gated bit-identical (extraction) and reproducible-by-seed (sampling) on both
  CPU and GPU; see `crates/qwen/src/serve.rs`'s and `crates/model/src/
  serve.rs`'s test modules.

### M4: W7 — `Engine::logits` de-serialised at admission; re-measured against M3

The one remaining host-computed head inside `qwen::serve::Engine`:
`Engine::logits(&hidden)` — the FIRST token's logits at admission (every
decode step after that already runs the on-device head, `submit_greedy_head`/
`forward_batched_topk`) — was a single-threaded scalar loop over
`[vocab, d_model]` (real Qwen3-0.6B: 151936 × 1024 ≈ 155M multiply-adds on
ONE core, per request). Replaced with `model::hostmath::matvec_par` — the
same rayon-over-output-rows + AVX2/FMA-per-row routine `qwen::sample`'s
decode path already uses for exactly this shape (its own doc comment records
measuring "hundreds of ms per token" from the scalar version at this size).
Gated by the existing `serve::tests::device_head_argmax_matches_the_host_head`
and the full `topk_extraction_matches_host_reference` /
`scheduler_sampled_requests_are_reproducible_and_differ_from_greedy` suite —
all pass unchanged (parallel accumulation order differs from the scalar
loop's, and the gate already tolerates that: it compares against a device
readback of the SAME computation, not a hand-written reference sum, so
non-associativity was already priced in).

**Re-measured, same box, same `http:qwen:` sweep, same `chat/in24/out12`
workload, release build, `--smoke --ladder 1,2`:**

| concurrency | ttfa p99 (M3) | ttfa p99 (M4, this change) |
|---|---:|---:|
| 1 | 2 952 ms | 3 890 ms |
| 2 | 1 063 ms | 1 403 ms |

Both numbers are **higher** than M3's, not lower — reported as measured
rather than reframed, per this doc's own standing rule. Repeating the M4 run
gave 3 886 ms / 1 435 ms, i.e. tight run-to-run agreement (this box's
variance is low), so the M3→M4 delta is real on THIS box at THIS time, not
noise between the two M4 samples.

**Bisected.** Reverted `Engine::logits` back to the single-threaded scalar
loop, rebuilt release, re-ran the IDENTICAL sweep immediately after the M4
runs above (same box-state neighborhood, differing ONLY in this one
function):

| concurrency | ttfa p99 (scalar `logits`, reverted) | ttfa p99 (M4, `matvec_par`) |
|---|---:|---:|
| 1 | 4 058 ms | ~3 888 ms (avg of the two M4 runs) |
| 2 | 1 788 ms | ~1 419 ms (avg of the two M4 runs) |

Restoring the fix (re-applying `matvec_par`) is clearly **faster** in this
close-together A/B — concurrency 2's TTFA drops by ~370 ms (21%), concurrency
1's by ~170 ms (4%). This resolves the open question above: the earlier
M3→M4 gap was **box-state drift across the session** (this box's absolute
numbers moved by ~1000 ms at concurrency 1 between when M3 and M4 were taken,
with no code difference in the served path other than this one function, per
the reverted-vs-fixed A/B done back-to-back), not a regression the `logits`
change introduced. The fix restored the `matvec_par` implementation. **Still
open:** an absolute `perf gate` baseline should still be captured freshly
right before it is committed (`brain perf gate --update`), rather than reused
from either M3 or M4, given this box's demonstrated absolute-number drift
over a session — the gate's own generous floor (fraction of baseline) is
designed to tolerate this kind of drift, but the baseline itself should be
recent.

What IS unambiguous and gated: `Engine::logits` no longer serialises 155M
FLOPs onto one core at admission; it uses the same parallel/vectorised path
decode already relies on, with zero output change on every existing
correctness test, and is now also confirmed faster by a controlled
same-session A/B, not just architecturally sound.

### W7's attention-dispatch-width question — measured, not just re-read

`.todo/attention-scratch-dispatch-width.md` flagged that
`paged_decode_scores_batched`/`decode_softmax_batched`/`paged_decode_apply_batched`
dispatch at the engine's full max-context width regardless of a sequence's
real length. `BRAIN_PROFILE=1` against two `qwen-synth` shapes settles
whether this is worth fixing:

| shape (`LxDxHxV`) | matmul | scores+softmax+apply (combined) |
|---|---:|---:|
| `2x256x8x8000` (tiny d_model) | 64.4% | 18.0% |
| `4x1024x16x32000` (real Qwen3-0.6B d_model) | 91.5% | 4.7% |

At the real model's `d_model`, `matmul` swamps everything else and this
concern shrinks to noise — confirms, rather than assumes, that it was
correctly left unfixed.

**The precondition that measurement never covered, taken separately**: those
two rows vary `d_model`, not the actual short-sequence/long-cap ratio the
concern is about (a fixed engine-wide `cap`, real sequences much shorter).
`BRAIN_PROFILE=1` at the REAL Qwen3-0.6B shape (`28x1024x16x151936`), forcing
a much larger cap:seqlen ratio (`--input 32 --output 512`, `qwen-synth`'s
`block_size=4096` rounds the pool to `cap=4096` against a sequence that starts
at 33 tokens and only reaches 544 — a ≥7.5x ratio throughout, ~124x at the
first decode step):

| shape (ratio) | matmul_gemv | scores+softmax+apply (combined) |
|---|---:|---:|
| `28x1024x16x151936`, tight cap (`in16/out8`, prior measurement above) | 91.5%\* | 4.7% |
| `28x1024x16x151936`, cap≈4096 vs seqlen 33→544 | 81.4% | **12.4%** (1.3+3.6+7.5) |

(\*prior row's `matmul` figure includes both `matmul`+`matmul_gemv`; this
row's target dispatches decode-shaped work almost entirely through
`matmul_gemv`, so the two aren't a like-for-like `matmul` comparison — the
`scores+softmax+apply` column is the one this measurement is actually about.)

So: the ratio effect is real and roughly triples the trio's share (4.7% →
12.4%) even at a fairly aggressive cap:seqlen mismatch — but `matmul_gemv`
alone still outweighs it more than 6-to-1. Two structural facts sharpen the
conclusion further (`.todo/completed/attention-scratch-dispatch-width.md` has
the full detail): only `paged_decode_scores_batched`/`_i8` actually dispatch
at `cap` width — `decode_softmax_batched` and `paged_decode_apply_*` dispatch
independent of `cap` and already loop to exactly `seq_lens[b]`, so the
file's originally-scoped "fix all four kernels plus softmax" was ~2.5x wider
than the two kernels that could possibly benefit; and `Input::Resident` (the
decode-window sub-steps) has no host-side seqlens at all, so the proposed
fix doesn't apply there without a new host-side shadow. **Verdict stands
closed, now on a direct measurement of the precondition it was missing
rather than an extrapolation from a different axis.**

### Perf gate baseline — committed

`scripts/gates/qwen-serving-perf-gate.sh` (`make qwen/serving-perf-gate`):
`http:qwen-synth:28x1024x16x151936` (Qwen3-0.6B's real shape, random
weights), `serve` scenario at concurrency 2, against a committed baseline
(`scripts/gates/qwen-serving-perf-baselines/`) with `brain perf gate --floor
0.5`. Verified both directions: passes against itself (including at a tight
`--floor 0.999`), and correctly FAILS a synthetically degraded candidate
(10x worse `ttfa_p99`/`output_per_s` against the same baseline: exit 1, both
metrics flagged). `sweep`'s curve-shaped artifact carries no flat metric
`perf gate` can read (`crates/perf/src/gate.rs`'s own "nothing was actually
gated" refusal caught this on the first attempt) — `serve` at one fixed
concurrency is the scenario shaped for gating.

Also fixed while wiring this in: `scripts/gates/{wm,forecast}-perf-gate.sh`
and `forecast-parity-gate.sh`/`test-times.sh` all `cd`'d one directory short
of the repo root (`scripts/gates/foo.sh`'s own `dirname` + `/..` lands in
`scripts/`, not the root), so every `./target/release/brain`/`out/...`
reference in those scripts silently pointed at a nonexistent path. Confirmed
live (`wm-perf-gate.sh` reported "build first" against an already-built
release binary) and fixed (`/../..`), the same `docs/lessons.md`-#1-adjacent
class as the `kernels-regen.sh` `REPO_ROOT` bug found earlier in this
project — a real, pre-existing, unrelated bug, fixed because it was found,
not routed around.

## Still planned

1. `capability::Provider` adoption by `qwen`, `yolo`, `depth`, `tts` (L) —
   in progress.
2. Raise the global `TEST_THREADS` once the remaining GPU-test crates
   (glm, moe, vision, depth, …) adopt `gpu_core::testgpu` — in progress.
3. Real host-OOM injection via an external cgroup memory limit; multi-rank
   fault harness over `model::netcollective`.
4. Committed per-box baseline artifacts for `perf gate` (the mechanism is in;
   baselines are a deployment choice).

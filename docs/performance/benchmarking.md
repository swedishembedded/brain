# brain performance benchmarking (`brain perf`)

How brain measures **how fast it is, at what cost, and whether the answer is
still correct** — across models, devices and hardware.

> **`brain bench` vs `brain perf`.** They answer different questions and must not
> be conflated.
>
> | | question | unit of comparison |
> |---|---|---|
> | `brain bench` (`crates/bench`) | *Can this architecture learn task X?* | architectures |
> | `brain perf` (`crates/perf`) | *How much correct work does this deliver per unit of hardware, memory, energy and time?* | models × hardware × configuration |
>
> Both use the same idiom — a registry, one runner, a JSON artifact per run, and a
> `compare` that diffs artifacts — so there is one way to do things in this repo.

---

## 1. Principles

These are the rules that make the numbers mean something. They are not
negotiable; a benchmark that violates one produces a number nobody can use.

1. **There is no single "brain score."** Minimum latency, saturated throughput
   and sustained behaviour under realistic load are three different properties
   that trade against each other. Always report all three.

2. **Headline the *output* side.** A workload with enormous prompts can post an
   impressive *total* tokens/s while delivering poor decode rate and terrible
   interactive latency. The headline number is **output artifacts/s** plus the
   **full latency distribution** — never total tokens/s alone.

3. **Goodput beats throughput.** The primary comparison metric is *maximum output
   rate while still inside the declared latency SLO*. An engine is rewarded for
   refusing impossible work early, not for producing tokens nobody can use.

4. **A number without its fingerprint is not a number.** Every artifact embeds
   the device, the real adapter string, backend, core count, RAM, quantisation,
   kernel flags and the build commit. Cross-hardware comparison is the entire
   point of this suite, and it is worthless if the hardware is implicit.
   *(Concretely: a box can report `--device gpu` and be running llvmpipe, a
   software rasteriser. The adapter string is what distinguishes "GPU result"
   from "CPU result wearing a GPU label".)*

5. **Performance results are invalid without a correctness gate.** Every perf run
   carries a fidelity check against a reference run. Exceeding the declared
   tolerance marks the artifact `valid: false`, and an invalid artifact is
   excluded from comparison. Otherwise the suite actively rewards optimisations
   that quietly break the model. brain already has the machinery for this — the
   cross-backend parity gate (`make parity`) and `gradcheck` — and `perf` reuses
   it rather than inventing a second notion of correct.

6. **Percentiles, not means.** P50/P95/P99/P99.9 for every latency. A mean hides
   exactly the behaviour that makes a server unusable.

7. **Measure the arrival process, not just the batch.** Requests that all exist
   at t=0 exercise a different engine than requests arriving independently. Both
   are legitimate; they are different benchmarks and are never mixed.

8. **Noise is a first-class concern.** brain's own performance docs record that
   shared, thermally-throttled boxes produce noisy end-to-end numbers and that
   the stable signals are min-of-N microbenchmarks and per-stage timings. `perf`
   therefore reports best-of-N *and* the spread, and its regression gates are
   hard floors rather than tight deltas — the same discipline as
   `scripts/gates/wm-perf-gate.sh`.

---

## 2. The metric model — why this generalises past LLMs

Most inference benchmarking is written for text decoders and hard-codes "token".
brain serves detection, depth, TTS, image generation, forecasting, 3D
reconstruction and world models from one engine, and the suite has to compare
them on the same hardware and compare hardware across them. So `perf` does not
measure tokens. It measures **artifacts arriving over the progress seam.**

brain already has exactly the right abstraction: `capability::Action::run` takes
a `progress: &mut dyn FnMut(Progress)` callback and returns an `Outcome`. Every
streaming model already calls it. That gives a model-agnostic timeline:

```
t_submit ──► t_admit ──► t_first ──► t_prog[1..n] ──► t_done
   │           │            │                            │
   └ enqueued  └ scheduler  └ first Progress             └ Outcome returned
     by driver   admitted     (or first output unit)
```

From that timeline, for **any** model:

| Metric | Definition | For a decoder LM this is |
|---|---|---|
| **queue** | `t_admit − t_submit` | scheduler queue time |
| **TTFA** — time to first artifact | `t_first − t_submit` | **TTFT** (includes queue + prefill) |
| **IAL** — inter-artifact latency | successive `t_prog` gaps | **ITL** |
| **TPOA** — time per output artifact | `(t_done − t_first) / (n − 1)` | **TPOT** |
| **E2E** | `t_done − t_submit` | end-to-end latency |

A one-shot model (detect, depth, a single forecast) simply has `n = 1`: TTFA
degenerates to E2E and IAL is empty. Nothing special-cases it.

**The artifact unit is named per model family**, and the schema records it, so
comparisons stay honest:

| Family | Artifact | Natural rate | Also report |
|---|---|---|---|
| Decoder LM (`gpt`/`qwen`/`glm`/`moe`) | output token | output tok/s | prefill tok/s separately |
| TTS (`tts`+`codec`) | audio chunk | audio-seconds/s | **RTF** = audio_s / wall_s |
| Detection (`yolo`) / depth (`depth`) | frame | FPS | per-stage pre/forward/post |
| Diffusion (`zimage`) | denoise step | steps/s | images/min |
| World model (`wm-*`) | frame | FPS | already `ms_per_frame_mean` |
| Forecast (`chronos2`/`kronos`/`fincast`) | series-horizon | series/s | |
| 3D (`mirror`/`splat`) | view / rendered frame | views/s, FPS | |

You may compare **one model across hardware** and **one hardware across models**.
Comparing *different models' absolute rates* is meaningless and the report
refuses to rank across artifact units.

### The target seam

```rust
pub trait PerfTarget {
    fn describe(&self) -> TargetInfo;               // model, unit, config fingerprint
    fn submit(&mut self, req: PerfRequest) -> ReqId;
    fn poll(&mut self) -> Vec<Emission>;            // (ReqId, EmissionKind, Instant)
}
```

### Measuring without a checkpoint

Weight *values* do not affect execution cost: the same kernels run, the same KV
traffic moves, the same blocks are allocated and the same batches form whatever
the numbers are. So the suite can build the **real** serving engine on randomly
initialised weights of a chosen shape (`--target qwen-synth:<L>x<D>x<H>[xV]`),
and measure it on any machine with nothing downloaded.

This is the right tool for **hardware and configuration comparison** — which is
most of what this suite is for — and the wrong tool for anything about output
quality, since the generated tokens are meaningless. Artifacts record
`weights: "random"`, and no correctness gate can pass on such a run.

### The target adapters

- **`CapabilityTarget`** — wraps any `capability::Provider`. Every model that
  implements the seam becomes benchmarkable with **zero** new benchmark code.
  This is the strategic path: today `demo`, `imageops` and `zimage` implement
  `Provider`; as `qwen`, `yolo`, `depth` and `tts` adopt it they are covered for
  free.
- **`PagedLlmTarget`** — wraps `qwen::serve::{Engine, Scheduler}` directly, since
  the paged continuous-batching engine is the thing most worth measuring and does
  not yet sit behind a `Provider`.
- **`ExecutorTarget`** — wraps a `residency::Executor`, so a resident model's real
  batching/placement (not a synchronous provider mutex) is what gets measured.
- **`HttpTarget`** — drives the REAL served path: `apiserve::router()`, called
  in-process via `tower::Service::oneshot` (no socket) — auth, the edge
  concurrency limiter, the admission race, chat-template rendering,
  tokenization, generation, all the way down to `residency::Executor` and the
  resident model. This is the ONLY target that would have shown
  `.todo/serving-performance-audit.md`'s 600s regression: every target above
  measures a scheduler/engine/executor directly, which stayed fast the whole
  time the actual served path (`crates/cli/src/resident_llm.rs`) never reached
  any of it (`docs/lessons.md` #22). Selected as
  `--target http:qwen-synth:<L>x<D>x<H>[xV]:<tokenizer.json>` (random weights,
  no checkpoint needed — same rationale as `qwen-synth:` above) or
  `http:qwen:<weights.brain>:<tokenizer.json>` (a real checkpoint).
  `crates/cli/src/perf_cli.rs` builds the real `QwenResident` + `Executor` +
  router underneath. Requests are OpenAI-dialect streaming chat completions, so
  the artifact timeline comes from real SSE `delta.content` chunks as they
  arrive over the wire.

New model families need an adapter only until they adopt `capability::Provider`.

---

## 3. Scenario catalogue

### Tier 1 — core (every model, every device)

| Scenario | Measures | Use it for |
|---|---|---|
| `latency` | fixed batch, in-process, no transport | kernel/engine work, regression gating |
| `throughput` | offline, fully saturated, all requests available at t=0 | engine + device efficiency ceiling |
| `serve` | arrival process at a fixed concurrency, over the real transport | realistic behaviour, cross-config comparison |
| `sweep` | a concurrency/rate ladder → the throughput-vs-latency curve | maximum sustainable concurrency under SLO |
| `startup` | cold and warm load: weights, pipeline compile, first artifact | deployment, autoscaling, `precompile` value |

`latency` deliberately excludes transport, arrival timing and queueing. It is the
right tool for "did this kernel change help" and the wrong tool for "how does
brain behave in production". `throughput` is an optimistic upper bound — always
full queue, no arrival realism, no per-user latency concern. Neither is the
headline; `sweep` is.

**Arrival processes** supported by `serve`/`sweep`: closed-loop (fixed
concurrency, the most deterministic engine pressure — the default), infinite rate
(saturation), fixed rate, Poisson, Gamma-distributed burstiness, and linear or
exponential ramp.

**Standard workload matrix.** Every workload is defined by input/output artifact
counts and a concurrency range, so the same grid runs on every device:

| Workload | in/out (tokens) | concurrency | stresses |
|---|---:|---:|---|
| `interactive` | 128 / 256 | 1–64 | TTFA floor |
| `chat` | 1 024 / 256 | 1–128 | the balanced case |
| `rag` | 4 096 / 128 | 1–128 | prefill-weighted |
| `rag_long` | 16 384 / 256 | 1–64 | long-context attention |
| `agent` | 8 192 / 1 024 | 1–64 | stable ITL under long decode |
| `decode_heavy` | 128 / 2 048 | 1–128 | pure decode rate |
| `prefill_heavy` | 32 768 / 64 | 1–32 | prefill and chunking |
| `shared_prefix` | 8 192 shared + 256 unique / 256 | 1–128 | prefix reuse |

Small-device profiles (`--profile edge`) scale these down by a fixed factor so
the same eight shapes run on an integrated GPU or an NPU without OOM, and the
artifact records the scale factor.

### Tier 2 — the scenarios that make this brain's suite

These are where brain has something to measure that a conventional LLM-serving
benchmark structurally cannot, because brain is a *multi-model, multi-backend,
edge-to-server* engine rather than one model on one GPU behind HTTP.

#### `residency` — multi-model residency, loading and eviction ★ headline

brain's `crates/residency` tiers weights across GPU/RAM/disk by LRU inside a
memory budget and schedules jobs across per-device lanes. Serving many models
from **one** engine is a genuine architectural differentiator, so it gets a
first-class benchmark.

```
device budget:     <B> GB               (--reserve-gb headroom respected)
catalogue:         N models, ΣN ≫ B     (deliberately 3–5× over budget)
popularity:        Zipf(α)
traffic shift:     popularity re-rolled every 5–15 min
```

Measure: warm-model TTFA vs cold-model TTFA · model load and eviction latency ·
requests blocked behind a load · aggregate goodput across all models · per-model
fairness (Jain) · weight-cache hit rate by tier (GPU/RAM/disk) · bytes read per
tier · **eviction regret** (instances evicted shortly before reuse) · useful work
lost to switching · residual memory after unload · cost of idle resident models.

#### `kvcache` — session lifecycle under memory pressure

Not "does prefix reuse help" but "what happens when the working set does not
fit". Sessions grow, idle, resume and branch, with the active set deliberately
2–5× KV capacity:

```
session A:  4k → 8k → 16k → 48k tokens
session B:  2k → idle 10 min → resume
session C:  80k → branch into 4 agent sub-sessions
session D:  repeatedly reuses a 32k system/repo prefix
```

Measure: useful KV hit rate (overall and per tier) · TTFA after resumption ·
**eviction regret** · recomputed tokens · KV bytes moved per generated token ·
write amplification · effective vs theoretical capacity · internal fragmentation ·
preemption count and duration · promotion latency · cross-tenant pollution.

Run the matrix cold-cache / warm-cache / caching-disabled, and at high and low
reuse ratios, with enough unique prefixes to force eviction. Prefix reuse
improves prefill and TTFA only — it must never be reported as a decode speedup.

#### `placement` — heterogeneous CPU / GPU / Vulkan / NPU

brain runs the *same WGSL* on four backends and has a separate whole-graph NPU
path. So it can ask a question most engines cannot: **given a machine with mixed
devices, does the engine place work well?**

Run the same model across every available device, then mixed placements
(embeddings, attention, dense FFN, MoE experts, vision/audio encoders, draft
model, sampler, tokeniser and KV tiers are each placeable). Report:

```
placement_efficiency = observed_goodput / oracle_goodput
```

plus cross-device bytes per artifact · device idle time · slowest-stage
utilisation · pipeline imbalance · migration cost · placement decision overhead ·
energy per request · behaviour when a preferred device disappears.

This distinguishes *a genuinely intelligent placer* from *an engine that merely
can run operators on several backends* — and it is the number that matters for
edge deployment.

#### `mixed` — traffic-class isolation

Run classes concurrently, each with its own arrival distribution, priority,
tenant, SLO, concurrency cap and token budget:

| Class | in / out | requirement |
|---|---:|---|
| interactive chat | 128 / 128 | low TTFA |
| RAG | 8k / 256 | moderate TTFA |
| coding agent | 16k / 2k | stable IAL |
| summarisation | 128k / 256 | background throughput |
| batch generation | 1k / 8k | no strict latency |
| embeddings | var / — | high throughput |

Measure per class: SLO goodput · P99 TTFA and IAL · **normalised slowdown**
versus running alone · starvation time · Jain fairness · queue time · decode
stall caused by prefills · throughput lost to priority handling. This is what
exposes head-of-line blocking that aggregate tokens/s hides.

#### `overload` — admission control and collapse

Offered load at 0.5×, 0.8×, 1.0×, 1.2×, 2.0×, 4.0× measured capacity, across
admission policies (unlimited queue · max queue depth · deadline-aware · token
budget · per-tenant quota · priority reservation · early rejection · load
shedding). Measure SLO goodput (not completions) · queue memory growth ·
rejection accuracy · requests admitted that provably could not meet their
deadline · recovery time after load returns to normal · healthy traffic harmed ·
OOM/process failure · work wasted on requests that time out.

#### `cancel` — cancellation and deadline waste

Cancel during queueing, tokenisation, prefill, decode, KV transfer, structured
output and final streaming. Measure abort propagation latency · artifacts
produced after cancellation · device-ms wasted after cancellation · time to
reclaim KV blocks · leaked queue entries/tasks · effect on unrelated requests ·
correctness under cancellation races · recovery after a cancellation storm.
Headline:

```
cancelled_compute_waste = compute_after_client_disconnect / total_compute
```

#### `soak` — long-duration drift and fragmentation

6 / 24 / 72 hours of mixed lengths, session resumption, cancellation, model and
adapter churn, bursts, idle windows and injected failures. Hourly: throughput
drift · P99 drift · device and host memory baseline · largest allocatable block ·
effective cache capacity · open descriptors and threads · error rate · recovery
from idle · restarts · output correctness drift. *An engine 5% faster for ten
minutes that degrades after twelve hours is not the better engine.*

#### `frontend` — host-side saturation

The accelerator is not always the bottleneck. Separately measure tokenisation,
chat-template rendering, JSONL framing, transport (stdio vs TCP vs unix vs
D-Bus), image decode/resize, audio resample, detokenisation, serialisation and
stream flush. Report **host cores required per saturated device**, per-stage
queue depth and latency, allocations per request, event-loop lag, slow-client
backpressure, and behaviour at 1 / 10 / 100 / 1 000 concurrent streams.

#### `faults` — injection and recovery

Kill a device worker; reset a device; fail a kernel; time out a collective; hang
a rank; drop a connection; partition the network; corrupt a KV transfer; fail a
weight read; CPU OOM; device OOM; restart the router. Measure detection time ·
recovery time · requests lost or duplicated · partial/corrupt output returned ·
capacity while degraded · whether one failed rank freezes all ranks · KV loss ·
session recovery rate · time to restore full goodput · whether retries stay
inside the original deadline.

#### `energy` — joules and cost at SLO

Where RAPL / device counters are available: joules per input artifact, per output
artifact and **per SLO-satisfying request** · idle power · power while merely
resident · energy of a cold load · artifacts/s per watt · requests per kWh ·
hardware cost per million output artifacts, and per million *while meeting P99
SLO*. For edge deployment this is frequently the deciding number, and it is the
one raw tokens/s most badly misleads on.

#### `fidelity` — the validity gate (runs with every scenario)

A perf result is accepted only when the output still matches a trusted reference
within tolerance:

```
greedy_token_match      ≥ 99.99%
structured_validity     = 100%
mean_logprob_error      ≤ 1e-3
invalid_numeric_outputs = 0
cross_backend_parity    = pass        (reuses `make parity`)
```

Checked: greedy agreement · per-token logprob error · KL divergence · top-k
ranking agreement · structured-output validity · tool-call syntax · stop-token
and max-token behaviour · determinism across repeats · numerical stability at
long context · quantised degradation · speculative-decoding equivalence. On
failure the artifact is written with `valid: false` and a reason, and `compare`
excludes it.

---

## 4. Result schema

One artifact per run, `results/perf-<scenario>-<model>-<device>-<seed>.json`.
Every scenario emits at least this; scenario-specific blocks are additive.

```jsonc
{
  "schema": "brain.perf/1",
  "scenario": "sweep",
  "valid": true,                  // false => excluded from compare
  "invalid_reason": null,

  "env": {                        // §1 principle 4 — the fingerprint
    "commit": "d1f2bbe", "dirty": false,
    "device": "gpu",              // requested
    "backend": "wgpu",            // resolved
    "adapter": "llvmpipe (LLVM 20.1.2, 256 bits) (Cpu, Vulkan)",
    "adapter_is_software": true,  // guards against "GPU" numbers that are not
    "cpu": { "model": "...", "cores": 48, "threads": 48 },
    "ram_gb": 184, "device_mem_mb": null,
    "os": "linux 6.17.0", "rustc": "...", "build": "release",
    "flags": { "BRAIN_NO_FASTCONV": null, "BRAIN_PROFILE": null }
  },

  "target": {
    "model": "qwen", "params": 630000000, "quant": "fp32",
    "kv_dtype": "int8", "block_size": 16, "max_batch": 32,
    "artifact_unit": "token",     // never compare across differing units
    "weights_sha256": "..."
  },

  "workload": {
    "name": "chat", "arrival": "closed_loop", "concurrency": 32,
    "request_rate": null, "burstiness": 1.0,
    "input_artifacts": { "dist": "fixed", "value": 1024 },
    "output_artifacts": { "dist": "fixed", "value": 256 },
    "num_requests": 2000, "warmup_requests": 32,
    "ignore_stop": true,          // force the requested output length
    "seed": 1234
  },

  "performance": {
    "wall_s": 187.4,
    "requests_per_s": 10.7,
    "input_artifacts_per_s": 10940.0,
    "output_artifacts_per_s": 2735.0,     // ← the headline
    "goodput_per_s": 2610.0,              // ← the comparison metric
    "slo": { "ttfa_ms_p99": 2000, "ial_ms_p99": 50 },
    "ttfa_ms": { "p50": 412, "p95": 980, "p99": 1740, "p999": 2210, "mean": 498 },
    "ial_ms":  { "p50": 11.2, "p95": 19.0, "p99": 31.5, "p999": 88.0 },
    "tpoa_ms": { "p50": 11.6, "p95": 20.1, "p99": 33.0, "p999": 91.0 },
    "e2e_ms":  { "p50": 3380, "p95": 5120, "p99": 6900, "p999": 8400 },
    "best_of_n": 3, "spread_pct": 4.1
  },

  "scheduling": {
    "queue_ms": { "p50": 3, "p99": 210 },
    "normalised_slowdown": 1.34, "starvation_ms_max": 940,
    "jain_fairness": 0.97, "preemptions": 12,
    "decode_stall_ms_from_prefill": 4100
  },

  "memory": {
    "kv_effective_capacity_tokens": 262144, "kv_theoretical_tokens": 294912,
    "kv_hit_rate": 0.61, "eviction_regret": 0.08,
    "recomputed_artifacts": 18320, "fragmentation": 0.11,
    "bytes_moved_per_artifact": 3072, "peak_device_mb": null, "peak_host_mb": 9210
  },

  "reliability": {
    "cancelled_compute_waste": 0.0, "failure_detect_ms": null,
    "recovery_ms": null, "lost_requests": 0, "corrupted_responses": 0,
    "errors": 0, "rejections": 0, "timeouts": 0, "ooms": 0
  },

  "resources": {
    "device_util": null, "host_cpu_util": 0.83,
    "host_mem_mb": 9210, "storage_read_mb": 0,
    "energy_j": null, "j_per_output_artifact": null
  },

  "correctness": {
    "gate": "greedy_match", "reference": "cpu-fp32-seq",
    "greedy_token_match": 1.0, "mean_logprob_error": 2.1e-6,
    "structured_validity": 1.0, "protocol_errors": 0, "passed": true
  },

  "per_class": [ /* `mixed` only: the same blocks, per traffic class */ ],
  "curve":     [ /* `sweep` only: one point per concurrency level */ ]
}
```

Fields that do not apply are `null`, never omitted and never zero — "not
measured" and "measured as zero" must stay distinguishable.

---

## 5. Comparing runs

`brain perf compare a.json b.json …` prints a leaderboard and refuses to rank
artifacts whose `artifact_unit` differs or whose `valid` is false.

**For a comparison to be meaningful, these must be identical across runs:** model
weights and revision (`weights_sha256`) · numeric format / quantisation ·
tokeniser and chat template · input and requested output lengths · stop-token
handling · sampling configuration · hardware, clocks and power limits ·
concurrency and arrival process · cache state (cold vs warm) · warm-up
procedure. `compare` diffs the `env`/`target`/`workload` blocks and **prints a
warning line for every axis that differs**, so an accidental
apples-to-oranges comparison is visible rather than silent.

**Two supported comparison modes:**

- **One model, many hardware** — the primary use. Everything but `env` is pinned.
  Report output artifacts/s, goodput, TTFA P99, and artifacts/s/watt.
- **One hardware, many models/configs** — pins `env`; compares configurations
  (int8 KV vs fp32, speculative on/off, block size, batch caps). Only compare
  absolute rates within one `artifact_unit`.

**Noise discipline.** Report best-of-N with the observed spread. Regression gates
are *hard floors* on best-of-N, not tight deltas — a laptop- or shared-class box
throttles, and tight deltas flap. This matches `scripts/gates/wm-perf-gate.sh`.

---

## 6. CLI

```bash
brain perf list                             # registered scenarios + workloads
brain perf run <scenario> --target <spec> [--workload W] [--device D]
                          [--seed S] [--out F] [--smoke] [--best-of N]
    # --target fake                     harness self-check, no model
    #          qwen-synth:12x768x12     the real engine on random weights
    #          qwen:out/qwen.safetensors    the real engine on a real checkpoint
brain perf sweep --target <spec> --workload chat --concurrency 1,2,4,8,16,32,64
brain perf compare results/perf-*.json      # leaderboard + differing-axis warnings
brain perf gate --baselines scripts/perf-baselines.json   # hard-floor regression gate
```

```bash
make perf                    # core scenarios, current device
make perf/<scenario>
make perf/sweep TARGET=... WORKLOAD=chat
make perf/compare
make perf/gate               # regression gate (hard floors)
```

`--smoke` shrinks every workload to a few seconds so the suite is CI-runnable;
smoke artifacts are marked `smoke: true` and never compared against full runs.

---

## 7. Status and phasing

| Phase | Contents | State |
|---|---|---|
| **P1** | harness core (clock, streaming percentiles, env fingerprint, schema, report), `PerfTarget` seam + `CapabilityTarget` + `PagedLlmTarget`, scenarios `latency` / `throughput` / `serve` / `sweep`, `brain perf` CLI + `compare`, Makefile targets | see `docs/performance/status.md` |
| **P2** | `startup`, `fidelity` gate wired into every scenario, `perf gate` + committed baselines, workload matrix complete | planned |
| **P3** | `residency` ★, `kvcache` | planned |
| **P4** | `placement`, `mixed`, `overload`, `cancel` | planned |
| **P5** | `frontend`, `soak`, `faults`, `energy` | planned |

Engine work this suite depends on, tracked here because the benchmark is what
makes it observable:

- **Per-step emission reporting in `qwen::serve::Scheduler`.** `step()` currently
  returns only *completed* requests, so no caller can observe when each token was
  produced — TTFA and IAL are unmeasurable without it. P1 adds a per-step
  emission report.
- **`capability::Provider` adoption** by `qwen`, `yolo`, `depth`, `tts`. Each one
  makes its model benchmarkable with no new benchmark code.
- **Cancellation** as an engine concept (`cancel` needs a request to be
  abortable mid-decode).
- **Admission policy** as a pluggable seam (`overload`).
- **Device/energy counters** in `gpu-core` (`resources`, `energy`).

### The question the suite exists to answer

> Under a realistic, changing and partially failing workload, how much **correct
> work meeting its SLO** does brain deliver per unit of hardware, memory, energy
> and time?

Maximum output tokens per second is not that answer, and this suite is
deliberately built so that number cannot be reported alone.

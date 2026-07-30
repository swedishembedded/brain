# Time-series forecasting over D-Bus + the scheduler — status ledger

Chronological, measured-only. NPU box: Intel "AI Boost" (Core Ultra), OpenVINO
2026.2, kernel 6.17 (the same box as the LFM NPU numbers). CPU host-compute is
brain's `wgsl-cpu` JIT backend.

## What "forecasting is served" means

brain's three forecasting foundation models — **chronos2**, **fincast**, **kronos**
— are reachable over the Brain1 D-Bus surface (`com.swedishembedded.Brain1`) and
scheduled through the residency `Executor`, exactly like every other model
(`docs/serving-contract.md`). There is **no forecasting-specific transport**: the
generic `Run(model, action, params, in_fds, in_meta)` carries the context series
in as an f32-LE blob fd and returns the forecast as an f32-LE blob fd. Many
clients share one `Executor`; identical-shape requests batch, different
models/devices run on parallel lanes.

- Residents: `crates/cli/src/resident_forecast.rs` (one `forecast` action each,
  env-gated `BRAIN_CHRONOS2` / `BRAIN_FINCAST` / `BRAIN_KRONOS_TOKENIZER`+`_DECODER`).
- Client + example: `examples/forecast/` (`forecast_client.py` + README).
- Wire format: input `context` blob = raw f32-LE + meta `{shape}`; output
  `forecast` blob = raw f32-LE + meta `{shape, kind, levels}`. chronos2 →
  `[levels, horizon]` quantiles (21 levels); fincast → `[horizon, 1+levels]`
  (col 0 mean, 9 quantiles); kronos → `[horizon, feat]` OHLCV sample bars.

## NPU placement

chronos2 and fincast advertise `MemCost::with_npu(..)`, so `place::pick_device`
auto-schedules them on the NPU when one is budgeted (NPU-first, then GPU, then
CPU). `activate(Device::Npu)` wraps the bespoke `Chronos2Session` /
`FincastSession` through the model's pluggable-core seam
(`forecast_quantiles_with_core` / `forecast_full_with_core`) — the host does
scaler/patch/embed + head/denorm on `gpu_core`, the transformer core runs on
OpenVINO; a compiled session is cached per context-length bucket. Every other
device runs the identical math on `gpu_core`, so the NPU and CPU/GPU paths are
bit-comparable. **kronos** runs on the NPU too, via `KronosModel::forecast_with_cores`
(the post-embedding s1/s2 core-injection seam): the host does
normalize/tokenize/embed/sample/denormalize on `gpu_core`, the two decoder graphs
(s1 + dep-s2, `KronosS1Session`/`KronosS2Session`, cached per context length)
run the autoregressive rollout on the NPU. Because the NPU graph is fixed-shape,
that rollout uses a fixed sliding window of `T` (the context length) rather than
the growing window of the CPU `forecast`, so kronos-NPU tracks but does not
bit-match kronos-CPU once the horizon slides the window.

An NPU compile/infer failure for one model is caught (`guard_npu`) and returned
as a per-request error, so it never unwinds and kills the shared NPU lane thread
(which would take every other NPU-scheduled model down with it).

## Measured (2026-07-29)

Live over `brain serve --dbus`, one `Run` per forecast, synthetic
trend+seasonality context.

| model | context | horizon | device | result |
|---|---|---|---|---|
| chronos2 | 96 | 24 | **CPU** (`--device cpu`) | `[21, 24]` quantiles; median tracks the trend |
| chronos2 | 96 | 24 | **NPU** (`--device cpu,npu`, auto-placed) | `device=NPU`, `[21, 24]` quantiles — scheduler chose the NPU by budget |
| fincast | 128 | 24 | CPU (`gpu_core`) | `device=gpu_core`, `[24, 10]` mean+9-quantiles (real `Vincent05R/FinCast` weights) |
| fincast | 64 | 12 | **NPU** (`--device cpu,npu`, auto-placed) | `device=NPU`, `[12, 10]` — ~1 B-param core via the external-data export + `load_path` |
| kronos | 96 | 24 | CPU (`gpu_core`) | `[24, 6]` OHLCV sample bars; univariate close expanded to bars server-side |
| kronos | 64 | 8 | **NPU** (`--device cpu,npu`, auto-placed) | `device=NPU`, `[8, 6]` bars — two decoder graphs (s1 + dep-s2) compiled + rolled out on the NPU |

All three foundation models were confirmed together on `--device cpu,npu`
(`device=NPU` for each) in one server, exercising concurrent NPU-lane scheduling.

- The scheduler picks the NPU with **no client change** — the client asks for
  `Run("chronos2","forecast",…)`; `place::pick_device` does the rest. The returned
  `device` field reports where it actually ran (`NPU` vs `gpu_core`).
- kronos accepts either full `[T, feat]` OHLCV bars or a univariate `[T]` close
  series (expanded to bars in the resident) — the demo uses the latter.

## Remaining / caveats

- **fincast weights**: fetched from `Vincent05R/FinCast` (`v1.pth`, ~4 GB) to
  `resources/time-series/checkpoints/fincast/`, converted with
  `tools/fincast_convert.py` → `fincast.safetensors`, imported via
  `brain forecast import --fincast <safetensors> --out out/fincast.weights`
  (991 M params). Live-validated over D-Bus on CPU (table above).
- **fincast on NPU**: DONE. FinCast's ~1 B-param ONNX core exceeds protobuf's
  2 GB `read_model_from_buffer` limit, so it is exported with an external-data
  sidecar (`fincast_export::export_external` → `finish_external`) and compiled via
  the new `FincastSession::load_path` (the LFM pattern). fincast advertises
  `MemCost::with_npu` and is auto-placed on the NPU (row above).
- **Batching**: forecast `run_batch` is the sequential default. chronos2/fincast
  share a batchable transformer core (equal-shape contexts could batch one
  forward); wiring a genuine batched forward is a follow-up.
- **Host GPU on this box**: `brain serve --dbus` with no `--device` now works —
  the canonical device registry (`gpu_core::devices::gpus()`) enumerates the Intel
  Arc iGPU (Vulkan `INTEGRATED_GPU`) and reports its shared DEVICE_LOCAL heap as
  `vram_bytes`, so `query_gpu_mem` budgets it as a schedulable `Gpu` lane (no
  discrete card required; was a panic before). `run_dbus` keeps a fallback that
  budgets a `discrete_gpu_count()`-sized lane at a modest shared-RAM fraction for
  the case the registry yields no VRAM. `--device cpu,npu` still gives CPU
  host-compute + an NPU lane.

## Inference optimization pass — kronos host KV path (2026-07-30)

Profile-driven (bench: `crates/kronos/tests/bench_cpu.rs`, min-of-N; a single
forecast is ~72–94% prefill, decode was dominated by re-projected dep-K/V). All
parity-gated (`tests/kvcache_parity.rs` cosine 1.000000 after each).

| optimization | effect | parity |
|---|---|---|
| **AVX2+FMA matvec** (`fast_ops::matmul_abt` for host matvecs, `21bbdb1`) | forecast 3128 → 2129 ms (**1.47×**) | cosine 1.0 |
| **dep-KV cache** (`dep_step_cached`, `f5ccc7f`) | decode 36.6 → 15.8 ms/step (**≥2.3×**) | cosine 1.0 |
| **shared-prefill sampling** (`forecast_cached_samples`, `bbbd8e8`) | nsamp=4: 3189 vs 10916 ms (**3.42×**) | **bit-identical** per seed |
| **serving uses the cached path** | the D-Bus kronos resident now calls `forecast_cached` (was the O(T²)/step `forecast`) + a `samples` param | cosine >0.999 |

Net: one kronos forecast ≈ **3.1 s → ~1.3 s**; a samples=N request pays one
prefill. These compound with the fine-tuned-`.weights` NPU path (`load_decoder`
takes a file or dir).

## NPU KV-cache pass — kronos cached rollout (2026-07-30)

The NPU rollout used to re-run the full-window s1/s2 graphs every decode step
(`forecast_with_cores`, O(T²)/step) — so once the host path got the KV-cache +
AVX work, the optimized CPU path overtook the NPU (~4.4×). Ported the same
KV-cache to the NPU: a **prefill** graph seeds a fixed-`cap` K/V cache, then a
**single-token decode** graph appends one token attending the cache (O(cap)/step),
mirroring qwen's `build_talker_{prefill,decode}_graph`.

- Graphs (`crates/npu/src/kronos_topology.rs`): `build_kronos_s1_{prefill,decode}_
  graph` (causal, per-layer `past_k/past_v[1,heads,cap,hd]` + `past_mask` +
  per-position RoPE → `new_k/new_v`, `ctx`, `s1_logits`) and `build_kronos_dep_
  {prefill,decode}_graph` (the s2 cross-attn, its K/V a projection of `ctx` cached
  like host `ensure_dep_kv`). `export_cached_onnx` builds all four from one load.
- Driver (`kronos::generate`): a `CachedCores` trait + `forecast_cached_with_cores`
  / `forecast_cached_samples_with_cores` — structurally identical to
  `forecast_cached` / `_samples`, so the tokenize/embed/sample/detokenize is
  shared and only the per-step graph math changes. Shared-prefill snapshots the
  cache after prefill and forks per sample.
- Backend (`cli::resident_forecast::KronosCachedNpu`): the four `NpuGraph`s +
  host-side cache buffers, cached per `(t, cap)`; `KronosNpuInstance` drives it.

Parity (`crates/npu/tests/kronos_kvcache.rs`, OpenVINO CPU): s1 cached rollout,
dep cached rollout, and the full interleaved driver rollout each match the
full-window graphs over the growing context — **cosine 1.000000** on ctx / s1 /
s2 logits at every position. The NPU forecast is now the KV-cache path (O(cap)
per step, one prefill per samples=N request), not the O(T²) full-window re-run.

Measured live (`brain serve --dbus` in a private session on the Intel NPU,
context 96 / horizon 32, kronos-small, best of 4 steady-state requests):

| path | steady-state | device |
|---|---|---|
| **NPU (cached)** | **~640 ms** | NPU |
| CPU (host cached) | ~1373 ms | gpu_core |

The NPU is now **~2.1× faster than CPU** — before this pass the NPU (full-window,
O(T²)/step) was ~4.4× *slower*. First request per `(t, cap)` pays a one-time
OpenVINO compile of the four graphs (~10 s); the instance stays resident, so
steady-state serving amortises it. `MemCost::with_npu` can now keep kronos on the
NPU by default without penalty.

## Training optimization pass — batched decoder fine-tune (2026-07-30)

`KronosTrain` was a batch-of-one trainer (one window per forward/backward). It is
now parameterized by a batch dim `b`: activation buffers scale to `b*t` rows
(batch-major), attention keeps a per-sequence `t×t` score matrix (`b·heads·t·t`,
so sequences never attend across the batch), and every weight-grad accumulates
over all `b·t` tokens. No kernel changes were needed — the GQA and `*_bidir`
kernels already carry a leading `bsz` dim. `FinetuneOpts.batch` (CLI
`brain forecast finetune --batch B`) groups training windows into chunks of `b`
(drop_last); held-out eval falls back to a b=1 model of the fine-tuned weights.

Correctness (`crates/kronos/tests/train_gradcheck.rs`):

| gate | result |
|---|---|
| **batched gradcheck** (b=2) | full fwd/bwd finite-diff check, 50 params pass |
| **batch == mean-of-singles** (b=3) | loss to 6 digits; grads elementwise-allclose (worst margin 0.0) |
| **promotion gate + universe** (b=2) | fine-tune promotes end-to-end (base 2.79 → ft 1.32) |

Throughput (`bench_cpu.rs::finetune_step_batch_scaling`, 22-thread CPU,
d_model=128 / 4 layers / t=64, min-of-5, ms/window):

| batch | ms/step | ms/window | vs b=1 |
|---|---|---|---|
| 1 | 60.4 | 60.40 | 1.00× |
| 2 | 111.4 | 55.68 | 1.08× |
| 4 | 176.7 | 44.19 | 1.37× |
| 8 | 231.5 | **28.94** | **2.09×** |

The per-window cost more than halves at b=8 — the step amortises per-submit
overhead and fills the AVX matmul rows, with the math held identical by the
parity gate. Remaining (see repo tasks): batched cross-sectional forward already
landed for inference; the Vulkan-OOM streaming for very long training contexts is
the last training-side item.

## Perf harness + regression gate (2026-07-30)

All three forecasters are first-class `brain perf` targets, measured through the
residency executor (scheduler + budgets + device lanes — the real serving path),
so the optimizations above get a defended baseline:

- `brain perf run <latency|throughput|sweep> --target kronos:<tok-dir>:<dec-dir>`
- `… --target chronos2:<weights>` · `… --target fincast:<weights>`

`artifact_unit` is `forecast`; `input_artifacts` = context length in bars (so a
prefill/decode sweep is `--ladder`/`--input` over context), horizon/samples from
`BRAIN_FORECAST_HORIZON`/`_SAMPLES`. Reports feed the existing hard-floor gate:
`brain perf gate <cand.json> --baseline <b.json> --floor 0.85` (exit 1 on
regression). Live-validated: kronos (`--input 64`) and chronos2 (`--warmup 1
--input 96` → 1.59 forecasts/s, e2e p50 1258 ms) both emit gate-passing JSON.

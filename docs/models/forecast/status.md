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
bit-comparable. kronos (autoregressive OHLCV rollout) serves on CPU/GPU; its
two-graph NPU rollout is the remaining follow-up.

## Measured (2026-07-29)

Live over `brain serve --dbus`, one `Run` per forecast, synthetic
trend+seasonality context.

| model | context | horizon | device | result |
|---|---|---|---|---|
| chronos2 | 96 | 24 | **CPU** (`--device cpu`) | `[21, 24]` quantiles; median tracks the trend |
| chronos2 | 96 | 24 | **NPU** (`--device cpu,npu`, auto-placed) | `device=NPU`, `[21, 24]` quantiles — scheduler chose the NPU by budget |
| kronos | 96 | 24 | CPU (`gpu_core`) | `[24, 6]` OHLCV sample bars; univariate close expanded to bars server-side |

- The scheduler picks the NPU with **no client change** — the client asks for
  `Run("chronos2","forecast",…)`; `place::pick_device` does the rest. The returned
  `device` field reports where it actually ran (`NPU` vs `gpu_core`).
- kronos accepts either full `[T, feat]` OHLCV bars or a univariate `[T]` close
  series (expanded to bars in the resident) — the demo uses the latter.

## Remaining / caveats

- **fincast**: served + structurally identical to the live-validated chronos2
  path (same `base_forecast_spec`, same `*_with_core` NPU seam that
  `brain npu fincast` already exercises, plus the fincast crate's own parity
  tests). **Not** live-validated over D-Bus here for lack of a local FinCast
  checkpoint (`BRAIN_FINCAST` unset; the repo under `resources/.../FinCast` ships
  code, not weights). Import one with `brain forecast import --fincast <ckpt>
  --out fincast.weights` to validate.
- **kronos on NPU**: the two-graph autoregressive rollout (`KronosS1Session` +
  `KronosS2Session`, the `kronos_rollout` in `npu_cli.rs`) is not yet wired into
  the resident — kronos stays on CPU/GPU. Follow-up.
- **Batching**: forecast `run_batch` is the sequential default. chronos2/fincast
  share a batchable transformer core (equal-shape contexts could batch one
  forward); wiring a genuine batched forward is a follow-up.
- **Host GPU on this box**: `brain serve --dbus` with no `--device` faults in the
  wgpu backend (integrated-only GPU + an unrelated uncommitted `backend-wgpu`
  change). Use `--device cpu` or `--device cpu,npu` — the latter gives CPU
  host-compute + an NPU lane, which is how the NPU number above was measured.

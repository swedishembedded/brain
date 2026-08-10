# Serving & runtime stack

A prose map of the crates that take a trained model and make it **discoverable,
scheduled, batched, and driven over a transport** — `capability`, `residency`,
`server`, `dbus`, `runtime`, `events`, `hfsm`, and `stats`. This doc owns the
architectural tour; the per-model checklist lives in
[`docs/serving-contract.md`](serving-contract.md), the stats detail in
[`docs/observability.md`](observability.md), and the NPU-as-schedulable-target
plan in [`docs/npu-residency.md`](npu-residency.md). `AGENTS.md`'s "Serving &
runtime stack" table is the index.

The governing principle, stated in `docs/serving-contract.md` and mirrored as an
AGENTS.md invariant, is **one capability interface, one scheduler, one
transport**: a model that bolts on its own subcommand, thread pool, or socket is
a maintenance island and a benchmark blind spot — `brain perf` measures anything
behind `capability::Provider` for free.

## Layering

```
            ┌─────────────── transports ───────────────┐
 D-Bus      │ server (JSONL: stdio/TCP/unix)           │  brain serve / brain run
(dbus)      │ runtime + events + hfsm (the `brain run` │
           │   stdio HFSM controller)                  │
            └───────────────────┬──────────────────────┘
                                ▼  capability::Invocation
                      ┌─────────────────────┐
                      │ capability (contract)│  Manifest / Action / Provider
                      └─────────┬───────────┘
                                ▼
                      ┌─────────────────────┐
                      │ residency (scheduler)│  Executor / ResidentModel / Instance
                      └─────────┬───────────┘
                                ▼  Instance::run_batch
                     model::serve::{Scheduler, PagedDecoder}
                  (continuous batching, prefix reuse, cancellation —
                   the seam autoregressive decoder LMs implement; see
                   serving-contract.md §3)
                                │
                      ┌─────────────────────┐
                      │ stats (observation)  │  StatsSnapshot → braintop / D-Bus
                      └─────────────────────┘
```

A request enters through a transport (D-Bus method, JSONL line, or HFSM event),
is translated into a `capability::Invocation`, dispatched generically, scheduled
by the residency `Executor`, and run as an `Instance::run_batch` on a placed
device; results and blobs come back the same way, and the whole thing is
observable through `stats`. For an autoregressive decoder LM, `run_batch`
drives every invocation the dispatcher grouped into that call on a shared
`model::serve::Scheduler` (real continuous batching for ONE dispatcher round —
joining a batch that is already mid-flight is a separate, not-yet-built seam,
tracked as a follow-up at the time in a note that is no longer recoverable,
but the gap it named is real: joining an in-flight batch is still not built)
instead of one sequential decode loop per request.

## The contract — `crates/capability`

The self-describing interface every model exposes (`crates/capability/src/lib.rs`):

- `Provider` — `manifest() -> Manifest` + `action(name) -> Option<Arc<dyn Action>>`.
- `Action` — `spec() -> ActionSpec` + `run(&Invocation, &mut Progress cb) -> Outcome`.
- `ActionSpec` — name/summary/`ParamSpec`[]/`BlobSpec`[]/`streaming`; `ParamType`
  (Str/Int/Float/Bool/Enum), `Media` (Image/Mask/Audio/Text/Bytes).
- `Invocation` — params JSON + named `Blob`s + a `CancelToken`;
  `Outcome` — outputs + blobs; `Progress` — step/total/message/delta.
- `CancelToken` — cooperative cancel (`is_cancelled()` is one relaxed atomic
  load; the `Default` token is inert so short actions can ignore it).
- `Registry` — the shared dispatcher: `register/find/run(model, action, inv, cb)`
  validates via `ActionSpec::validate` then runs.

A `ResidentModel::manifest` returns a `capability::Manifest`; both D-Bus and the
HFSM translate their wire requests into `capability::Invocation`. **Adding a
capability ≠ adding a subcommand** — implement `Action` and list it in a
`Provider`; `brain do` and the event API pick it up.

## The scheduler — `crates/residency`

Automatic model residency + job scheduling (`crates/residency/src/`):

- **Tiers** (`lib.rs`): `Cold` (mem-mapped on disk), `Warm` (host RAM), `Hot`
  (built on device, ready). `MemCost{vram,ram,npu}`; NPU-eligibility is
  `npu > 0` (a CPU-only model is never placed on the NPU).
- **Budgets** (`budget.rs`): per-device `Budget{total,reserved,used}`;
  `--reserve-gb` keeps headroom per card.
- **LRU** (`lru.rs`): `Residents` maps `InstanceKey -> Entry{cost,device,last_use,
  uses,pinned}`. A pinned instance (a job is running on it) is never evicted.
- **Placement** (`place.rs`): `pick_device` prefers the NPU class, then GPU,
  then CPU fallback (most-free device within a class; a zero-cost stateless
  instance goes to CPU). `plan_eviction` evicts lowest-score victims
  (`EvictionPolicy`: strict `Lru` or GDSF-style `CostAware`) until the deficit
  is covered, protecting the target's `keep` set.
- **Scheduler** (`scheduler.rs`): `Policy{max_batch:8, age_weight_per_ms:1.0,
  batch_weight:200.0, max_wait_ms:2000}`; a group whose oldest job exceeds
  `max_wait_ms` is force-picked.
- **Executor** (`executor.rs`): one dispatcher thread owns the
  `ResidencyManager` + queue + running set; one lane thread per device
  (`brain-lane-{device}`). **Activation runs on the lane** — a slow
  `activate` (weight load, NPU graph compile) stalls only its own device, never
  the dispatcher. Models on different devices run in parallel; same-device jobs
  serialize on that lane.
- **`ResidentModel` adapter** (`model.rs` + `bridge.rs` +
  `crates/cli/src/resident_*.rs`): `manifest()` / `instance_key()` (a config
  fingerprint, e.g. z-image `"WxH:precision:adapter"`) / `estimate()` (the
  budgeted footprint) / `activate()` (builds once; the `Instance` owns the
  weights so `Drop` frees VRAM — RAII). Adapters are env-gated (`BRAIN_*` vars;
  `from_env -> None` when unset) and registered in `resident::build_executor`;
  `ProviderResident::stateless` wraps a no-weight `Provider`.

## The transports

**`crates/server`** — one JSONL `events::Event` protocol over three transports
(`transport.rs`, `controller_session.rs`): `serve_stdio`, `serve_unix`,
`serve_tcp`. A `Session` (`on_line` / `on_line_streaming` / `greeting`) +
`LineSink` (`send`); `ControllerSession` adapts the `runtime::Controller`,
streaming each emitted `Envelope` to the wire as produced.
`pump_connection` is transport-independent and `catch_unwind`-isolates each
connection (a panicking client never takes the server down); `max_connections`
defaults to 64.

**`crates/dbus`** — the optional D-Bus control surface over
`com.swedishembedded.Brain1` (`service.rs`, `stream.rs`, `fd.rs`):
- Methods: `manifests`, `list_models`, `Run`, `Subscribe`, `StreamTranscribe`,
  `Cancel`, `stats`, `stats_snapshot`, `stats_stream` (signal, ≥2 Hz); properties
  `version` / `active_jobs` / `models`.
- **fd blob transport** (`fd.rs`) — sealed memfd (mmap read) by default,
  dmabuf best-effort via `/dev/dma_heap/system`; `read_fd_to_vec` /
  `bytes_to_fd`.
- **Stream frames** (`stream.rs`) — `SOCK_SEQPACKET` `socketpair` (preserves
  message boundaries, no length-prefix): `progress` / `segment` / `blob` (with
  an out-of-band memfd via `SCM_RIGHTS`) / `done` / `error`. Droppable frames
  (`progress`/`segment`) are non-blocking (EAGAIN→dropped); terminal frames
  (`blob`/`done`/`error`) get a bounded writability wait so a briefly-stalled
  subscriber never silently loses its completion (EPIPE→disconnected either
  way).

Each D-Bus method only validates + translates: it builds an `Invocation` from
the params + in_fds, arms a `CancelToken`, submits a `residency::Job`, and
returns the outcome fds.

## The HFSM controller — `crates/runtime` + `events` + `hfsm` (`brain run`)

- **`events`** (`crates/events/src/lib.rs`) — the JSONL protocol: the `Event`
  enum (`UserText`, `BrainTextChunk`, `CameraFrame`, `ObjectDetected`,
  `UserSynthRequest`, `AudioChunk`, `Cancel`, forecast/action/manifest pairs,
  …), `Envelope{req_id,event}`, `encode_line`/`decode_line`, base64 + ppm
  helpers.
- **`hfsm`** (`crates/hfsm/src/lib.rs`) — a generic hierarchical state machine:
  `Machine` trait (`dispatch`/`parent`/`on_entry`/`on_exit`), `Hsm<M>` with an
  RTC queue and LCA-correct exit/entry chains (non-reentrant).
- **`runtime`** (`crates/runtime/src/lib.rs` + `pump.rs`) — the `Brain` machine:
  `St` (Operational/Idle/Chatting/Detecting/Synthesizing/Forecasting/…), typed
  seams (`InferModel`/`DetectModel`/`SynthModel`/`ForecastModel`), and a generic
  `Brain::run_action` path (`ActionRequest` → `capability::Registry::find` →
  `ActionSpec::validate` → `Action::run`, streaming `ActionProgress` inline).
  `Controller::register_provider` plugs in a `capability::Provider`.

`brain run` (`crates/cli/src/run_cli.rs`) parses `--gpt`/`--yolo`/`--dbus`/
`--models-dir`/`--anthropic`/`--openai`/`--openrouter`/…; with no surface flag
it is the stdio controller loop, with `--dbus` it serves the executor over
D-Bus. **The D-Bus/HTTP path bypasses the HFSM and drives the `Executor`
directly** — the HFSM is the stdio event-driven path. An unknown flag is a
hard error (exit 2, usage to stderr), not a warning — see `brain serve --help`
for the full flag reference, including that OpenAI/OpenRouter serve every
route both with and without the `/v1` prefix (either works as a client's
`base_url`) while Anthropic is `/v1`-only.

**Readiness.** `--ready-file PATH` (`brain_shutdown::ready::Gate`) touches an
empty marker file once *every* requested surface — every HTTP dialect plus
D-Bus — has actually bound, and never at all if one fails to come up. Because
`--api-keys-out` and the `APIKEY` stdout lines are both written before any
bind, the marker appearing implies both the keys are on disk and the socket(s)
are accepting, so a launcher script waits on one file instead of polling a
port or grepping the log. There is deliberately no `/healthz` route: readiness
here is a *process*-level property (it must also cover D-Bus, which has no
HTTP route to answer for it), and adding an unauthenticated route would be the
first one outside the auth layer (see `docs/api-security-audit.md`) for no
gain over the existing `GET /v1/models` probe. If an orchestrator ever needs
an `httpGet`-shaped readiness probe, the answer is an *authenticated* route
with an injected key, not an unauthenticated one.

## Request data flow

- **`Run`** (one-shot): client → `Manager::run` → `build_inv` (params + in_fds →
  `Invocation`, `register_job` arms a `CancelToken`) → `Executor::submit` →
  dispatcher queues → `choose_next` → `ResidencyManager::claim` (`pick_device`
  or `plan_eviction`; `Hot` or `Build`) → lane `activate`s then `run_group` →
  `Instance::run_batch` → per-job reply → `outcome_to_fds` (memfd/dmabuf) →
  D-Bus reply.
- **`Subscribe`** (streaming): `Executor::submit` with an `on_progress` (→
  `progress` frame) + `reply` (→ `blob`/`done`/`error`); returns a job id + a
  SEQPACKET event fd. `Cancel(job)` flips the token; the action aborts at its
  next poll → `Err("cancelled")` → `error` frame.
- **`StreamTranscribe`** (continuous input): a `stream_reader` thread reads mono
  f32 LE 16 kHz PCM from the pipe, windows it, and submits each window as a `Job`
  → `segment`/`done` frames. The pattern for any live-input model, not just ASR.
- **`brain run`** (stdio): stdin JSONL → `feed_line_streaming` →
  `decode_envelope` → HSM dispatch → (for `ActionRequest`) `run_action` →
  `Envelope` → stdout, flushed per line.

## Observation — `crates/stats`

Stats detail is owned by [`docs/observability.md`](observability.md) — do not
duplicate. The seam to know here: a `StatsSource` contributes into a snapshot;
an `Assembler` walks all registered sources (no central switchboard); the live
wiring is `ExecutorSource` (`crates/stats/src/build.rs`) reading
`Executor::stats` / `manifests` / `residency`. `crates/stats` is serde + assembly
only (it depends on `brain-residency` + `brain-capability`, no engine code), so
it stays light enough for any front-end to depend on. It is surfaced over D-Bus
as `stats_snapshot()` (one-shot pull) and the `StatsStream` signal at ≥2 Hz
(`STATS_INTERVAL = 500 ms`), which `braintop` subscribes to instead of polling.

## Configuration — the `BRAIN_*` environment

Which models a `brain serve` actually serves is **entirely env-driven**: each
resident adapter's `from_env` returns `None` (model simply not served) when its
weights variable is unset. There is no config file. This table is the operator
reference; `scripts/gates/check-env-docs.sh` (run by `make check/scripts`)
greps the serving crates' `env::var("BRAIN_…")` reads against this document so
a new variable cannot ship undocumented.

### Which models serve (gating — unset ⇒ not served)

| Variable | Serves | Value |
| --- | --- | --- |
| `BRAIN_QWEN_WEIGHTS` + `BRAIN_QWEN_TOKENIZER` | Qwen3 chat (`generate`) | `.brain` checkpoint + `tokenizer.json` |
| `BRAIN_GPT_WEIGHTS` | char-level GPT baseline | checkpoint (embeds its vocab) |
| `BRAIN_GLM_WEIGHTS` | GLM decoder | checkpoint (char-level) |
| `BRAIN_LFM` + `BRAIN_LFM_TOKENIZER` | LFM2.5-Encoder (`fill-mask`/`embed`) | weights + `tokenizer.json` |
| `BRAIN_FLUX2_DIT` / `_VAE` / `_TE` / `_TOKENIZER` | FLUX.2 Klein text2image/edit | the four component paths (all required) |
| `BRAIN_ZIMAGE_DIT` / `_VAE` / `_QWEN` / `_TOKENIZER` | Z-Image text2image/edit | the four component paths (all required) |
| `BRAIN_YOLO` | YOLOv8 detection | checkpoint |
| `BRAIN_DEPTH_WEIGHTS` | ZipDepth monocular depth | `.pth` checkpoint |
| `BRAIN_SAM2_WEIGHTS` | SAM 2.1 segmentation | `sam2.1_hiera_*.pt` checkpoint |
| `BRAIN_FACENET_DIR` | antelopev2 face detect/embed | dir holding `glintr100.onnx` + `scrfd_10g_bnkps.onnx` |
| `BRAIN_ESRGAN_WEIGHTS` | Real-ESRGAN upscale | `RealESRGAN_x4plus.pth` (or any RRDBNet) |
| `BRAIN_RESTORE_WEIGHTS` | CodeFormer face restore | `codeformer.pth` (or its dir) |
| `BRAIN_VQGAN_WEIGHTS` | CodeFormer VQ encode/decode | checkpoint (or its dir) |
| `BRAIN_CLIP_DIR` | CLIP text/image embeddings | SDXL-layout root (`tokenizer[_2]/`, `text_encoder[_2]/`, EVA `.pt`) |
| `BRAIN_CHRONOS2` | Chronos-2 forecasting | weights |
| `BRAIN_FINCAST` | FinCast forecasting | weights |
| `BRAIN_KRONOS_TOKENIZER` + `BRAIN_KRONOS_DECODER` | Kronos OHLCV forecasting | the two checkpoint dirs |
| `BRAIN_TTS_WEIGHTS` (+ `BRAIN_TTS_CKPT`) | Qwen3-TTS `speak` | brain-format weights dir (+ HF ckpt dir for config/tokenizer) |
| `BRAIN_NEMOTRON` | Nemotron 3.5 streaming ASR | HF checkpoint dir |
| `BRAIN_QWEN_ASR` | Qwen3-ASR offline ASR | HF checkpoint dir |
| `BRAIN_OMNI_HF_DIR` | Qwen3-Omni Thinker (validation tier) | HF checkpoint dir |
| `BRAIN_MOCK` | deterministic weight-free mock model | any non-empty value |

### Per-model tuning (optional; defaults in the reader)

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_QWEN_CTX` | Qwen built context length | 24576 (`resident_llm.rs::QwenResident::ctx`) |
| `BRAIN_QWEN_MAX_BATCH` | Qwen serving batch slots | 16 |
| `BRAIN_QWEN_KV_INT8` | int8 KV cache (`0` opts out) | on |
| `BRAIN_QWEN_KV_CALIB` | per-head KV clip ranges from `brain qwen calib` | unset |
| `BRAIN_LFM_BATCH` | LFM batched-forward slots per instance | 2 |
| `BRAIN_FLUX2_MAX_BATCH` | FLUX.2 concurrent same-size batch cap | 4 |
| `BRAIN_YOLO_BATCH` | YOLO true-batch forward width | 1 |
| `BRAIN_SAM2_VARIANT` | SAM 2.1 variant (`tiny`/`large`) | `tiny` |
| `BRAIN_ZIMAGE_RETAIN_INT8_CACHE` | `1` retains the DiT int8 cache for demote/promote (multi-GB host RAM) | off |
| `BRAIN_TTS_LANG` / `BRAIN_TTS_REF` / `BRAIN_TTS_REF_TEXT` | TTS language / reference voice `.wav` / its transcript | `english` / none / none |
| `BRAIN_CODEC_WEIGHTS` | TTS codec weights override | derived from `BRAIN_TTS_WEIGHTS` |
| `BRAIN_QWEN_ASR_WINDOW` / `BRAIN_QWEN_ASR_MAXNEW` | Qwen3-ASR window (s) / max tokens | 30 / 200 |
| `BRAIN_FORECAST_HORIZON` / `BRAIN_FORECAST_SAMPLES` | forecast horizon / sample count (perf targets + defaults) | 64 / 1 |
| `BRAIN_MOCK_DELAY_MS` | mock model artificial latency | 0 |
| `BRAIN_OV_CACHE` | OpenVINO compiled-graph cache dir (ASR/NPU) | `$TMPDIR/brain_ov_cache` |
| `BRAIN_TTS_RES` | resources base for `brain tts serve`'s default engine paths | unset (flags supply paths) |
| `BRAIN_TTS_NPU_DEVICE` | OpenVINO device for the TTS NPU talker | auto |

### Server & scheduler knobs

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_DEVICE` | schedulable compute set (`cpu`, `gpu`, `npu`, `gpu0`, `gpu,cpu`, …) | all devices |
| `BRAIN_MODELS_DIR` | model directory scanned at startup (also `--models-dir`) | `$XDG_DATA_HOME/brain/models` |
| `BRAIN_AUTO_FETCH` | `0`/`false`/`off` disables transparent auto-fetch | enabled |
| `BRAIN_ADMIT_DEADLINE_MS` | bound on waiting for a lane before a request is shed | `residency::admission` default (10s) |
| `BRAIN_COLD_BUILD_ADMIT_DEADLINE_MS` | longer bound applied instead, only while the request's own model is still cold-building (first activation) | `residency::admission` default (180s) |
| `BRAIN_CONF` | stdio-loop YOLO confidence threshold (also `--conf`) | 0.25 |
| `BRAIN_GPT` | stdio-loop GPT checkpoint (also `--gpt`) | fake echo model |

`brain serve --help` points here; deeper per-model knobs (e.g.
`BRAIN_ZIMAGE_ENCODER_GPU`, `BRAIN_FLUX2_TE_DEVICE`, `BRAIN_TTS_STREAM_CHUNK`)
are documented in their model's `docs/models/*` ledger and read inside the
model crates, not the serving layer.

## How the docs divide

- [`docs/serving-contract.md`](serving-contract.md) — the per-model
  **five-obligation checklist** (capability / residency / batching / D-Bus /
  verify) and the self-audit question.
- [`docs/observability.md`](observability.md) — the **stats subsystem** detail
  (`StatsSnapshot` shape, `braintop`, the "add a metric" rule).
- [`docs/npu-residency.md`](npu-residency.md) — the **NPU as a schedulable
  target** plan (`Device::Npu`, `place::pick_device` NPU preference).
- This doc — the **architectural map** of the seven crates and how a request
  flows through them.
- `AGENTS.md` "Serving & runtime stack" — the index table; its invariants
  ("Adding a capability ≠ adding a subcommand" and "Every new model ships the
  full serving contract") are the rules this doc explains mechanically.

## See also

- `crates/capability/src/lib.rs`
- `crates/residency/src/{lib,budget,lru,place,model,manager,scheduler,executor,bridge}.rs`
- `crates/server/src/{transport,controller_session}.rs`
- `crates/dbus/src/{service,stream,fd}.rs`
- `crates/runtime/src/{lib,pump}.rs`, `crates/events/src/lib.rs`,
  `crates/hfsm/src/lib.rs`
- `crates/stats/src/{snapshot,source,build}.rs`
- resident adapters: `crates/cli/src/resident*.rs`
- `crates/cli/src/run_cli.rs` (`brain run` / `brain serve`)

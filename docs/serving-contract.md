# The serving contract — what "adding a model" means

A model in brain is not done when it trains and passes parity. It is done when it can
be **discovered, scheduled, batched, and driven over D-Bus** like every other model.
This is the checklist every new model must satisfy *in the same change that adds it*,
and the rule is mirrored as an invariant in `AGENTS.md` (Conventions & invariants →
"Every new model ships the full serving contract"). Keep the two in sync.

The point is uniformity: one capability interface, one scheduler, one transport. A
model that bolts on its own subcommand, its own thread pool, or its own socket is a
maintenance island and a benchmark blind spot (`brain perf` measures anything behind
`capability::Provider` for free — see `docs/performance/benchmarking.md`).

## The five obligations

### 1. Capability — expose actions through the generalized interface
Implement `capability::Action` for each action and advertise them in a
`capability::Manifest` (via a `Provider`, or directly from the model's
`ResidentModel::manifest`). Never add a bespoke `brain <model>` subcommand as the
*only* entry point — `brain do <model> <action>` and the event API must work.
- Reference: `crates/capability/src/lib.rs`; ASR: `crates/nemotron/src/caps.rs`,
  `crates/qwen-asr/src/caps.rs`; shared audio-in/text-out contract in
  `crates/audio/src/asr_caps.rs` (one implementation, both models).
- Blob conventions are shared and typed (`capability::Media`): images = raw
  interleaved HWC f32 in `[0,1]`, little-endian, meta `{"w","h","c"}`; audio = raw
  mono f32 LE 16 kHz + `{sample_rate}`. Reuse them; don't invent a per-model encoding.
- The image codec has ONE implementation: **`capability::blob`**
  (`image_blob` to encode; `decode_hwc` / `decode_image` / `decode_plane` to decode,
  with media + `{w,h}` + payload-size validation). Providers and resident adapters
  call it directly — never a local copy or a local alias.
- **Training actions return their artifact as a blob.** An action that writes a
  trained artifact server-side (a `save` path param) must ALSO declare an output
  `BlobSpec` and attach the saved bytes to the `Outcome`
  (`.blob("adapter", Blob::new(Media::Bytes, bytes).with_meta(json!({"path": save})))`)
  — a remote client has no access to the server's filesystem, and the D-Bus layer
  already turns outcome blobs into result fds for free. Reference:
  `zimage::caps` `lora_train`.
- **Long-running actions must be cancellable.** Every [`Invocation`] carries a
  `capability::CancelToken` (`inv.cancel`; the `Default` token is unarmed and never
  fires, so short actions can ignore it). An action whose run is longer than a few
  seconds — multi-minute generation, multi-hour training — polls
  `inv.cancel.is_cancelled()` between steps (once per denoising step / training
  step; the check is one relaxed atomic load) and returns `Err("cancelled".into())`.
  Front-ends arm the token (the D-Bus `Cancel` method flips it); the token rides
  inside the `Invocation` through the executor, so `residency::Job` needs nothing.
  Reference: `zimage::caps` `text2image` + `lora_train`
  (`pipeline::HotPipeline::generate`, `finetune::run`).

### 2. Residency — be scheduled, budgeted, swappable
Add a `residency::ResidentModel` adapter under `crates/cli/src/resident_*.rs` and
register it in `resident::build_executor` (env-gated, `from_env` → `None` when its
weights var is unset). `activate` **builds the model once** (weights uploaded once)
and the `Instance` owns it so dropping frees the memory; `estimate` reports the Hot
footprint the manager budgets against.
- Reference: `crates/residency/src/{model,executor}.rs`; ASR:
  `crates/cli/src/resident_asr.rs`; the pattern to copy: `resident.rs` (yolo/z-image).

### 3. Batching — a real batched forward where the architecture allows
Override `Instance::run_batch`. When the model has a batchable forward (a shared
encoder, a per-frame matmul stack), do a **genuine** batch — one device forward over N
inputs — not a serial loop. If a stage is inherently sequential (autoregressive
decode), batch what you can and say why in a comment.
- ASR example: `resident_asr::NemotronInstance::run_batch` groups concurrent
  stream-windows by language prompt and runs one FastConformer forward per group
  (`nemotron::encoder::Encoder::transcribe_batch`, which row-concatenates the
  per-frame matmuls and is bit-identical to the single-utterance path). Qwen3-ASR is
  offline/autoregressive, so its `run_batch` is the sequential default on a
  build-once, fixed-window instance — documented as such.
- **Autoregressive decoder LMs: use `model::serve::{Scheduler, PagedDecoder}`,
  don't reinvent continuous batching.** `crates/model/src/serve.rs` is an
  architecture-agnostic paged-KV continuous-batching scheduler — admission,
  prefix reuse, an on-device decode window, cancellation, and the
  `StepReport` timeline `brain perf` measures TTFA/ITL from — generic over a
  `PagedDecoder` impl (`prefill`/`forward_batched_greedy(_window)`/
  `logits`/block-pool accessors). `qwen3::serve::Engine` is the first
  implementation; `qwen3::serve::Scheduler` is `model::serve::Scheduler<Engine>`
  under a type alias, so every existing caller of the qwen-specific names
  needs zero changes. A new decoder LM (glm, gpt, moe) that wants real
  concurrent serving should write its own paged/batched forward (its
  `run_batched_submit` equivalent, over `model::paged::{BlockAllocator,
  BlockTable}`) and implement `PagedDecoder` over it — admission, prefix
  cache, batching, cancellation and streaming then come from `model::serve`
  for free, rather than each decoder growing its own copy of this machinery.

### 4. D-Bus surface — reachable over the bus, with a runnable example
The actions MUST be callable over `crates/dbus` (`com.swedishembedded.Brain1`) and
demonstrated by an example under `examples/<domain>/` with a README.
- **Fit first.** If the model's shape matches the existing surface, use it as-is:
  - `Run` — one-shot request → result + output fds (memfd/dmabuf).
  - `Subscribe` — a job that streams progress/blob/done frames out over a SEQPACKET.
  - `Cancel(job)` — cooperative cancel of a `Subscribe` job: arms nothing new, it
    flips the `CancelToken` the server registered under the returned job id; the
    action aborts at its next poll and the stream ends with an `error` frame
    (`"cancelled"`). Returns `true` iff the job was found still in flight. The
    `ActiveJobs` property counts in-flight `Run`/`Subscribe` jobs.
  - `StreamTranscribe` — a *continuous input* fd (the client keeps writing) that the
    server windows into executor jobs and answers with `segment`/`done` frames. This
    is the pattern for any live-input model, not just ASR.
  - fd blob transport (`crates/dbus/src/fd.rs`) for bulk data in/out.
- **Extend or refactor if it doesn't fit.** If a new modality can't be expressed with
  the existing methods/frames, add a method or generalize a frame type in
  `crates/dbus` — and update every client/example the change touches
  (`brain-py/brain_py/dbus.py`, `examples/`). Do **not** add a side channel (a private
  socket, a temp file dance) to avoid touching the surface. The surface is meant to
  grow deliberately; a side channel is how it rots.
- Reference: `crates/dbus/src/{service,stream,fd}.rs`; clients:
  `brain-py/brain_py/dbus.py`; examples: `examples/dbus`, `examples/asr`.

### 5. Verify it end to end
- Cross-backend parity if imported (`make parity`), gradient-check if trained
  (`make gradcheck`).
- A capability/manifest unit test (cheap; no weights) asserting the action schema.
- The example must run against `brain serve --dbus` (WAV-file mode for a no-hardware
  smoke test where the input is live audio/video).

## Quick self-audit

> Can a fresh process, given only the weights path in an env var, do
> `brain serve --dbus` and have this model show up in `Manifests`, run its action over
> `Run`/`Subscribe`/`StreamTranscribe`, batch when two requests arrive at once, and be
> driven by an `examples/` script — with no model-specific code in the transport?

If yes, the contract is met. If any answer is "no", the model is not done.

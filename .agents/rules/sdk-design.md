# SDK design - public API surface and progressive disclosure

This is the rulebook for anything a caller outside this workspace can see:
`crates/sdk` (package `brain`), the CLI (`crates/cli`), the capability/D-Bus
surfaces (`crates/capability`, `crates/dbus`, `crates/apiserve`), and
`brain-py`. Read it BEFORE adding or changing a public type in any of those,
or before wiring a model into one for the first time.

**The goal in one sentence: a first-time embedder gets three lines that work,
and an expert can still reach every knob underneath - through the SAME API,
never a second one.** Complexity belongs inside brain, not in the caller's
code. `.agents/rules/serving-contract.md` is this document's sibling: that
one is the checklist for the CLI/D-Bus/scheduler side of "done"; this one is
the checklist for the embeddable-library side. A model can pass one and fail
the other - both are required.

The reference example already in this workspace, not a hypothetical:

```rust
let pipe = brain::ImagePipeline::from_pretrained("black-forest-labs/FLUX.2-klein-9B")?;
pipe.generate("a whale submarine")?.save("out.png")?;
```

(`crates/sdk/src/pipeline.rs`). Every rule below is either what makes that
example possible, or what keeps the NEXT pipeline this workspace adds from
regressing it.

## 1. Design from the task, never from the internals

A public type names what the caller wants, not the graph brain happens to
run to get it. `ImagePipeline::from_pretrained(...).generate(prompt)` is the
public shape; `WorldMirrorEncoder`/`CameraHead`/`GaussianRasterizer`/`AdamW`
wired up by hand is what `pipeline::ImagePipelineBuilder::load` does
*internally* and a caller of `crates/sdk` never sees. When a new capability
gets an SDK surface, write the three-line call first and work backward to
what has to exist to make it true - never start from "here are the structs
this model happens to have" and expose them as-is.

## 2. One pipeline type per capability; `from_pretrained` means ready

Every task-shaped capability gets exactly one obvious public type
(`ImagePipeline`, and eventually `TextGenerationPipeline`,
`EmbeddingPipeline`, `SplatPipeline`, `ForecastPipeline` - see
`.agents/roadmap/sdk.md`'s "not yet done" list for which of these are still
missing). `from_pretrained` resolves the model id, fetches if needed
(`brain_modelstore`), picks the concrete backend, sizes it, and builds a
resident, ready-to-call instance in one call - never a second
`.initialize()`/`.load_processor()`/`.prepare_kernels()` stage. `crates/sdk`
already does not force factories, registries, engines, sessions, or contexts
on a caller who just wants inference; a new pipeline must hold that line.

Two pipelines that do the same conceptual thing on different architectures
(flux2 vs. s3dit text-to-image) are ONE public type, not two - `ImagePipeline`
dispatches on the resolved `capability::Assembly::arch` internally
(`pipeline::resolve_arch`) and every downstream call is uniform past that
point. Do not add `Qwen3Pipeline`/`GlmPipeline` next to a generic
`TextGenerationPipeline` for models that perform the same task; a
model-specific type is only justified when the capability is genuinely
unique to that model (rule 11).

Local paths and hub ids go through the same call
(`ImagePipelineBuilder::load` calls `brain_modelref::ModelRef::parse` on
whatever string it's given): never a separate API for "load from disk" vs.
"load from the hub."

**Known exception, and why it's not a violation:** `Creature` (`crates/sdk/
src/creature.rs`) does not follow `from_pretrained`+`generate`. A connectome
driving a physics body has state that survives a call and is stepped in a
loop, so its shape is a builder (`Creature::fruit_fly().connectome(...).
body(...).build()?`) plus `drive`/`step`. Match the shape to what the domain
object actually is - a stepping simulation is not a stateless transform -
but keep the same bar: one obvious constructor, no mandatory second init
stage, and the "only way to tell a result from a coincidence" controls
(`set_plasticity`, `shuffled_connectome`) on the public surface rather than
buried in an experiment binary. Don't force every future capability into
`from_pretrained`/`generate` just for consistency with `ImagePipeline`; force
consistency with the SHAPE OF THE DOMAIN instead, and only ever create a new
top-level shape (not a third) when neither existing one fits.

## 3. Progressive disclosure - one API, three depths, no wrong turn

Every public operation must be usable at the depth the caller actually needs:

1. **Simple**: `pipe.generate(prompt)?` - every option left at brain's own
   default.
2. **Common customization**: a small builder over the same call -
   `ImageGenerationOptions::new().size(w, h).steps(n).seed(s)` (already the
   shape `pipeline.rs` uses; each setter overrides exactly its own field and
   nothing else, verified by `set_generation_options_override_only_their_own_field`).
3. **Component-level**: swap the underlying model/renderer/reconstructor
   directly. `ImagePipeline` does not offer this yet - that's a real,
   tracked gap (`.agents/roadmap/sdk.md`), not a precedent to copy. When a
   pipeline grows this level, it must not raise the cost of level 1 or 2 to
   get there.

A caller must never be forced through level 3 to use level 1. Concretely:
no config struct as the only way to call a common operation
(`pipe.generate(prompt)` must work with zero named types in the caller's
code); no mandatory builder ceremony for a call with no optional inputs.
Builders exist for OPTIONAL configuration, not as a tax on every call.

## 4. Domain objects in, domain objects out - never a bag of tensors

A capability that produces a meaningful thing returns that thing:
`ImagePipeline::generate` returns `Image`, not a raw tensor or a
backend-specific struct. `crates/sdk` already normalizes flux2's
`(Vec<u8> RGB8, w, h)` and s3dit's float-HWC output into the SAME `Image`
type (`Image::from_rgb8` / `Image::from_hwc_unit`) so a caller never learns
which backend actually ran. When the next pipeline lands
(`TextGenerationPipeline` → `GeneratedText`, `SplatPipeline` → `SplatScene`,
`ForecastPipeline` → `Forecast`), give it its own domain type the same way -
never "just return the tensor, the caller can figure out the layout."

Raw access stays available one level down (`result.tensor()`,
`model::hostmath` internals, `crates/kernels`) for callers who genuinely need
it - see rule 12 - but it is never the ONLY return type for a task-level
call.

**Save/load is symmetric and format is inferred from the destination.**
`Image::save("out.png")` picks the codec from the extension; anything brain
can load through `from_pretrained` should have an obvious save counterpart.
When a new domain object needs a serialized form, follow this pattern rather
than inventing a codec-specific API as the primary path (a codec-specific
escape hatch may still exist for callers who need to force a format).

**One canonical representation per domain, reused, not duplicated.** This is
not a new rule - it's the workspace's existing "one implementation" law
(`AGENTS.md` Conventions: `rmsnorm` existed seven times before it didn't)
applied to the SDK's own output types. `capability::Media`'s image/audio
blob conventions and `crates/imaging`'s pixel helpers are the ONE
representation every pipeline and every `Provider` normalizes into; a new
pipeline reuses them, it does not grow a parallel `Image`/`AudioBuffer` type
of its own.

## 5. Infer what brain already knows; validate the rest immediately

Don't ask the caller for information already recoverable from the model or
the input: architecture from `brain_arch::ARCHS`/the checkpoint's own
manifest, tokenizer from the resolved model's config, precision tier from
the checkpoint's own quant metadata (a `.gguf` source's packed-int8 path is
already picked for the caller by `flux2::pipeline::effective_dit_precision`,
never asked for). Where brain genuinely cannot infer something safely
(`ImagePipelineBuilder::size` on an s3dit-backed pipeline - the DiT/VAE
graphs are sized once, at construction, so there is no post-hoc inference
possible), fail with a named error at the API boundary the moment the
mismatch is knowable (`check_s3dit_size` runs before the denoise loop
starts, not after), never silently resize or silently ignore the request
(rule 7 below).

**No silent semantic fallback.** If a requested capability genuinely isn't
supported, return a typed error naming the gap
(`ImagePipeline::load_lora`'s `adapter_source_path` refuses an `owner/name`
store reference by name rather than silently trying - and failing - to
treat it as a filesystem path). Never quietly substitute a materially
different algorithm or a different model variant for what was asked.

## 6. Errors must say what to do next

Every public error variant answers: what failed, what brain expected, what
it got, and - where there is one - the corrective action. `crates/sdk`'s
`Error::Ambiguous`/`Error::Missing` carry the resolver's full structured
answer (every candidate, the exact flag to pass) rather than a flattened
string, specifically so a caller can act on it instead of re-parsing
`Display`'s text. Follow that shape for new variants: a hand-rolled enum
(not `thiserror`) is the workspace's convention here because an SDK's error
type IS its public API surface, and a derive macro can leak a dependency's
error type into every caller.

**Error hygiene differs by surface, and that's already documented, not an
inconsistency.** `crates/sdk` is in-process: its errors carry real file
paths verbatim, because a caller with filesystem access needs the real
reason. `crates/apiserve`/`crates/dbus` are network-facing and must collapse
every internal error to a path-free, provider-shaped body before it reaches
a client (`.agents/rules/api-security.md` §5). If you add an error type on
either side, match ITS surface's rule, not the other one's, and if a network
front-end ever re-exposes an SDK error's `Display` directly, that's the
disclosure bug `api-security.md` exists to catch.

## 7. Device, precision, and quantization are high-level choices - not leaks

A caller says `Device::Auto` or `Device::Cuda(0)`
(`gpu_core::devices::DeviceSpec`, re-exported as `brain::Device` - not
reinvented) and `DType::Int8` (`model::dispatch::Precision`, re-exported as
`brain::DType`), never a raw backend handle, a CUDA stream, or a WebGPU
buffer. `crates/sdk`'s rule for this is explicit in `lib.rs`'s own doc:
**re-export the type the CLI's own `--device`/`--precision` parse into,
never invent a parallel one** - two types for the same concept is how a
caller's `Device::Cuda(0)` and the CLI's `--device cuda:0` silently drift.
Weight residency (GPU/RAM/disk tiering, `crates/residency`) and int8/paged-KV
serving defaults stay invisible to a normal call; an advanced caller may
inspect or override the policy, but `model.generate(prompt)` must keep
working unmodified when residency moves weights between tiers underneath it.

## 8. Capability discovery reuses the capability system - it doesn't reinvent it

This workspace already has a structured way to answer "what can this model
do": `capability::Manifest`/`ActionSpec` (`crates/capability`). A pipeline
that wants to expose `.capabilities()` reflects THAT manifest - it does not
grow a second, weaker "supports_streaming: bool" ad-hoc struct. The same
applies to metadata a caller needs programmatically (timing, memory,
progress): if `crates/stats`' `StatsSnapshot` or an action's `Outcome`
already carries it, surface that, don't scrape a log line for it (this
workspace already treats "never hardcode a count, read it from the data" as
load-bearing for `braintop`; the same applies to anything an SDK caller
needs to inspect).

Progress and cancellation likewise reuse what exists rather than growing a
parallel mechanism: `capability::CancelToken` is already threaded through
`Invocation` and `pipeline.rs`'s own `generate` calls
(`|_step, _total, _msg| {}` is the callback shape - a real pipeline wires it
to a caller-supplied closure instead of discarding it). A long-running SDK
call (training, multi-step generation, a download) exposes progress and
cancellation through that SAME mechanism, not a bespoke channel per model.

## 9. Training obeys the same rules as inference

`Trainer`/`finetune` must clear the same bar `from_pretrained`/`generate`
does: a caller should not hand-assemble optimizers, schedulers, data
loaders, or mixed-precision contexts for a standard run. Fine-tuning methods
(LoRA, full fine-tune, future adapters) are strategies plugged into one
training call, not unrelated end-to-end implementations - this mirrors the
workspace's existing `model::train::fit` + `rl` reward-weighted-loss
composition, not a new pattern to invent. Checkpointing (periodic save,
resume, optimizer/scheduler state) is part of what a standard training
workflow gets for free, the same way `crates/checkpoint`'s manifest/SHA-256
container already backs every model's save path - reuse it, don't grow a
per-pipeline checkpoint format.

## 10. CLI, D-Bus, HTTP, and Python must call the SAME code as the SDK

**This is the rule most at risk of silent violation, and it is already
violated once, tracked in `.agents/roadmap/sdk.md`:** `crates/cli/src/
flux2_cli.rs` and `s3dit::caps::ZAction` each build their own
`flux2::Pipeline`/`s3dit::pipeline::HotPipeline` inline, independently of
`crates/sdk::ImagePipeline` - the resolve/build logic exists in two places
today, not one. Treat that as an open defect, not a precedent: a CLI handler
must parse arguments, map them onto the SDK/capability call, and print the
result - never re-derive the resolve → build → run sequence a moment time
the SDK crate already owns. If a useful behavior only exists inside a CLI
handler, the SDK (or `capability::Action`) is incomplete, not the CLI
merely "thin already."

The same law applies across every front-end this workspace has: `brain-py`
(today an event-driven subprocess client, not a direct binding -
`AGENTS.md`'s own crate table says so; that's an acknowledged gap against
this rule, not the target shape), `crates/dbus`, `crates/apiserve`, and any
future language binding must all resolve to the same underlying call brain's
own CLI makes, so a fix or a feature lands once. This is the SDK-facing
half of `.agents/rules/serving-contract.md`'s "one capability interface, one
scheduler, one transport" - that document covers reachability over D-Bus;
this rule covers reachability as a linked library, and a model/pipeline
needs both to be done.

## 11. Capability traits, not one god-trait, and not per-model public verbs

Prefer small, focused traits a model may implement several of
(`GenerateText`, `Embed`, `ReconstructScene`, `capability::Action` per
verb) over one universal `Model` interface carrying every possible
operation, and over exposing `QwenGenerate`/`GlmGenerate` when both models
perform the identical task through `TextGenerationPipeline`. A concrete
model type may still exist for advanced access (`flux2::Pipeline` is real
and reachable), but it is never the ONLY way to invoke a capability that
already has a generic pipeline.

## 12. Advanced access must exist, but never as the front door

Raw tensors, model modules, forward passes, optimizer state, and backend
handles must all remain reachable - brain is a research and training
framework as much as an inference SDK, and locking that away would break
the workspace's own use of itself. But: the escape hatch lives one level
below the task API, is never what a first example demonstrates, and is
never required for the common path. `result.tensor()` existing next to
`Image` is the model; a pipeline whose ONLY output is a raw tensor is not.

## 13. Public surface discipline

- **Keep `crates/sdk`'s top-level namespace small and deliberate.**
  `lib.rs`'s `pub use` list is short on purpose - a type only appears there
  because a normal caller needs to name it. Internal types stay
  `pub(crate)`; every new `pub` item on this crate is a compatibility
  promise, and the crate's own feature-vocabulary rule
  (`crates/sdk/Cargo.toml`'s `[features]`, gated by `make check/sdk-features`)
  already enforces that a feature names a SURFACE (`image`, `creature`), not
  an internal tier - keep new features to that same discipline; `device` and
  `resolve` are the only tiers, selected FOR a caller by a surface feature,
  never named directly by one.
- **Don't wrap a shared function for "readability."** `.agents/rules/
  architecture.md`'s "one implementation" rule applies to SDK-facing code
  exactly as it does to a kernel: a local alias over `gpu_core::devices::
  DeviceSpec` is how `brain::Device` and the CLI's device type start
  drifting at the next edit.
- **Delete, don't accumulate, competing ways to do the same thing.** If
  `ImagePipeline` eventually grows a cleaner construction path, retire the
  old one with a documented migration rather than keeping both promoted
  forever - multiple equally-documented ways to do the same task is a
  permanent tax on every new user, not a convenience.
- **Examples in rustdoc and `docs/` must show the canonical form.** A
  low-level path may remain available, but the FIRST example for a
  capability is always the shortest one that works - `crates/sdk/src/
  lib.rs`'s own doc comment already leads with the three-line
  `ImagePipeline` call before anything else; every new pipeline's module doc
  follows that order (smallest working example, then options, then
  internals) and `docs/` (user-facing) never leads with architecture the way
  `.agents/` internal docs do.

## 14. Every public pipeline needs an end-to-end test, and a real one

A forward-pass parity test or a gradcheck does not substitute for a test
that goes `from_pretrained` → the task call → inspect the domain result →
save. `crates/sdk/tests/image_pipeline.rs` is the existing shape to copy.
Where a real end-to-end run is blocked on something else (as it currently
is for `ImagePipeline`: neither flux2 nor s3dit exposes a tiny injectable
config through this SDK's public API, so the integration tests prove
resolve → dispatch → construction up to the real multi-billion-parameter
weight ceiling and then assert a clean error - see `.agents/roadmap/sdk.md`),
say so in the roadmap file and keep the test asserting as much of the real
path as it can, rather than skipping SDK-level coverage entirely in favor of
only the backend's own `#[ignore]`-gated real-checkpoint test.

## Review checklist for a new or changed public API

Before calling an SDK-facing feature done:

- [ ] There is one obvious entry point for the task, and it produces a
      ready-to-use object with no second init stage.
- [ ] The common call needs no named config type in caller code; a builder
      exists only for optional configuration.
- [ ] The result is a domain object with an obvious way to save it, not a
      raw tensor or backend-specific struct.
- [ ] Local paths and remote model ids go through the same call.
- [ ] Device/precision/quantization are the re-exported, high-level types -
      no backend-specific type crossed the public boundary.
- [ ] Errors name what failed, what was expected, and the fix; no internal
      path/stack leaks across a network-facing surface
      (`.agents/rules/api-security.md`).
- [ ] Progress and cancellation, if the call is long-running, use
      `capability::CancelToken`/the existing progress-callback shape.
- [ ] The CLI (and D-Bus, and Python) call THIS code - verify by reading the
      call site, not by assuming; a duplicate resolve/build/run sequence
      anywhere is a defect, not a style choice (rule 10).
- [ ] An end-to-end test exists: load → call → inspect → save.
- [ ] The rustdoc/`docs/` example is the shortest one that works, shown
      first.
- [ ] No new public type duplicates an existing domain representation
      (`capability::Media`, `crates/imaging`, an existing pipeline's output
      type).
- [ ] `.agents/rules/serving-contract.md`'s checklist is ALSO satisfied if
      this capability is meant to be served - the two are independent bars
      and a model needs both.

If a common task cannot be done in three lines and the caller has not had to
read anything below `crates/sdk` to do it, the API is not done yet.

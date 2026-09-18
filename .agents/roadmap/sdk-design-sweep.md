# sdk-design sweep - roadmap

The tracking document for bringing the whole workspace into line with
`.agents/rules/sdk-design.md`, across as many sessions as it takes. Read this
file FIRST before touching anything SDK-surface-shaped; it is meant to be
picked up cold by a session that has never seen this campaign before. Check
items off as they land, and add what you learn - this file decays fast if
only read, never written.

Seeded from three background audits run at kickoff (2026-09-16): a domain
inventory of every served model, a rule-by-rule self-audit of `crates/sdk`,
and an architecture map of the flux2/s3dit pipeline-construction duplication.
None of the three read every model crate's source; they used `crates/catalog`,
`crates/arch`, and targeted greps, on purpose, to keep the audit itself cheap.

## Domain inventory - what has an SDK pipeline, and what's next

`crates/arch::ARCHS` carries an explicit `Domain` enum (`crates/arch/src/
lib.rs:47-76`). 59 rows across it; `crates/catalog::models()` registers 36 of
them for serving. Today `crates/sdk` covers exactly two surfaces: `ImagePipeline`
(the `Image` domain's generation half) and `Creature` (not an `ARCHS` row at
all - the fly/flybody/connectome stack is unregistered).

| Domain bucket | Archs | Served | SDK pipeline | CLI shape | LoRA/finetune |
|---|---|---|---|---|---|
| Text decoders | 8 | 6 | none | fragmented: 6 model-specific `*_cli.rs` + `resident_llm.rs` | yes (qwen3) |
| Multimodal/VLM/OCR | 9 | 8 | **PARTIAL - `VisionLanguagePipeline`** (Qwen3-VL) **+ `GroundingPipeline`** (Florence-2: open-vocabulary visual grounding); 7 more served archs deferred, see Phase 6.1/6.2 | `omni_cli.rs`, `document_study_cli.rs` + 5 `resident_*.rs` | yes (qwen3vl) |
| Image generation | 6 | 6 | **YES - `ImagePipeline`** (flux2+s3dit only) | `flux2_cli.rs` + `s3dit::caps::ZAction` | yes (flux2, s3dit) |
| Restoration/upscaling/VAE | 5 | 5 | **PARTIAL - `UpscalePipeline`** (RRDBNet) + **`RestorePipeline`** (CodeFormer; SUPIR/VQGAN deferred, see Phase 2.5/2.7) | no dedicated CLI; `resident_restore/upscale/supir.rs` | no |
| Video generation | 2 | 2 | **PARTIAL - `VideoPipeline`** (Wan2.1 T2V; ltxv deferred, see Phase 5.1) | `wan_cli.rs`, `ltxv_cli.rs` | yes (wan) |
| ASR | 2 | 2 | **PARTIAL - `TranscribePipeline`** (qwen3-asr only; nemotron + streaming deferred, see Phase 2.4) | **no CLI at all** - `resident_asr.rs` only | no |
| TTS/music/speech codec | 7 | 3 | **PARTIAL - `TtsPipeline`** (Qwen3-TTS: speak/clone_voice/design; CosyVoice2/3: clone_voice) **+ `MusicPipeline`** (MiniMax Music 3: lyrics+caption-to-song); remaining speech-codec architectures deferred | `tts_cli.rs` + `tts_serve.rs` | no |
| Vision/detection/segmentation | 4 | 4 | **PARTIAL - `DetectionPipeline`** (YOLOv8) + **`SegmentPipeline`** (SAM2) + **`DepthPipeline`** (ZipDepth; label is a VLM captioning workflow, not a single-arch capability, out of scope here - see Phase 3.1/3.2/3.3) | `yolo_cli.rs`, `sam2_cli.rs`, `depth_cli.rs`, `label_cli.rs` | no |
| Embedding towers | 3 | 3 | none | **no CLI at all** - `resident_clip/arcface/t5encoder.rs` | no |
| Forecasting | 4 | 0 (CLI-local) | none | **one unified entry**: `forecast_cli.rs` + generic `resident_forecast.rs` | yes (`brain forecast finetune`) |
| 3D/scene | 2 | 0 (CLI-local) | none | `splat_cli.rs`, `mirror_cli.rs` | no |
| World models | 2 | 0 | none | `wm_cli.rs` | no |
| Toy (excluded from `brain caps`) | 4 | 0 | n/a | `pid_cli.rs` | n/a |
| Creature (unregistered) | 0 | 0 | **YES - `Creature`** | none (see rule-2 exception in sdk-design.md) | n/a |

`imgpipe` (in `catalog::models()` but not an arch - it *composes* capabilities
via `stage_registry()`) is a natural future `AutoPipeline` consumer, not its
own bucket.

**Priority order for the next new pipeline** (reasoning kept short; expand
when a milestone actually starts one):

1. **`ForecastPipeline`** - only bucket with ONE unified CLI entry already
   (`forecast_cli.rs`) and one generic resident dispatch; stateless call
   shape; smallest domain object (`Forecast`); already has a `finetune` verb.
   Lowest-risk proof of the *second* `from_pretrained` shape.
2. **`TextGenerationPipeline`** - highest raw unlock (8 decoders + ~9
   multimodal decoders behind them), but the CLI side is fragmented across
   six model-specific files - a consolidation project, not a wrapping one.
   Must not regress into `Qwen3Pipeline`/`GlmPipeline` per-model types
   (rule 2 forbids this explicitly).
3. **`EmbeddingPipeline`** (clip/arcface/t5encoder/ecapatdnn/campplus) -
   conceptually the simplest call shape, but no CLI entry point to lift from
   at all - written fresh.
4. **ASR `TranscribePipeline`** - same "no CLI to lift from" gap as
   embedding, plus Nemotron's streaming path needs the progress/cancel
   design (rule 8) done properly, not stubbed.
5. Restoration/upscaling, vision/detection (natural `ImagePipeline` siblings
   returning the same `Image` domain type), TTS/music, video generation, then
   3D/world models last - `SplatPipeline` is closer to `Creature` (stateful,
   steppable) than to `ImagePipeline`, and world models have no settled
   domain object yet. **Done**: `UpscalePipeline` (RRDBNet, Phase 2.5) and
   `RestorePipeline` (CodeFormer, Phase 2.6/2.7 - a genuine full `.restore()`
   forward pass at the model's real fixed geometry, and two real
   forward-pass infrastructure bugs found AND fixed reaching it). Vision/detection
   (YOLOv8 boxes / SAM2 masks - a different domain object than `Image`, not
   a sibling of either) is the next candidate within this bucket.

## `crates/sdk` self-audit findings

Rule-by-rule audit against `.agents/rules/sdk-design.md`, cross-checked
against `.agents/roadmap/sdk.md` so only genuinely NEW findings are listed
here (items already tracked there are not repeated - see that file's own
"Not yet done" section for those).

### Real violations / gaps

| # | Rule | Where | Finding | Status |
|---|---|---|---|---|
| 1 | 14/2 | `crates/sdk/src/creature.rs` | `Creature` has zero tests, inline or integration - the only public surface with none | fixed (M4) |
| 2 | 14 | `crates/sdk/src/view.rs:11-14` | `View::frame`'s documented headless-capture behavior is untested | fixed (M4) |
| 3 | 8 | `crates/sdk/src/pipeline.rs:321,331` | `generate`/`generate_with` hardcode `&CancelToken::default()` and a no-op progress closure; no public way to supply either | fixed (M5) |
| 4 | 8 | `crates/sdk/src/pipeline.rs:300,481,504` | download and s3dit-build progress are likewise discarded | fixed (M5b) |
| 5 | 6 | `crates/sdk/src/error.rs:70-71` | `Error::Cancelled` is a dead public variant - unreachable with no public cancel entry point | fixed (M5) |
| 6 | 8 | crate-wide | no `.capabilities()`/manifest introspection anywhere in `crates/sdk` | fixed (M8) |
| 7 | 9 | crate-wide | no training/finetune entry point at all in `crates/sdk` (the adjacent Dataset-layer gap is tracked in `sdk.md`; the missing training call itself was not) | open (Phase 2, per-pipeline) |
| 8 | 9/4 | `crates/sdk/src/creature.rs:492-500` | `set_plasticity`/`reward` mutate learned synapse weights with no `save()`/`load()` counterpart - all learning dies with the process | fixed (M11) |
| 9 | 7 | `crates/sdk/src/creature.rs:157` | `Creature::build` acquires its GPU via `gpu_core::testgpu::dev` - TEST-SUPPORT infra, weak-reference lifetime, shipping on the production SDK path | fixed (M3) |
| 10 | 7/13 | `crates/sdk/src/creature.rs` + `Cargo.toml` | `CreatureBuilder` has no `Device` knob at all, yet the `creature` feature's doc comment claims it "selects `device`" | fixed (M3) |
| 11 | 5 | `crates/sdk/src/pipeline.rs:416-424,432-435` | `ImagePipelineBuilder::size` is silently IGNORED on a flux2-backed pipeline (the s3dit half of this asymmetry is tracked in `sdk.md`; the flux2 silent no-op was not) | fixed (M10b) |
| 12 | 6 | `crates/sdk/src/error.rs:62-66` | `Error::Backend` is an untyped catch-all for ~8 semantically distinct failures (license refusal, no models dir, size mismatch, bad extension, missing builder arg, GPU/MuJoCo failure...) | partially fixed (M6 - see below) |
| 13 | 6 | `crates/sdk/src/creature.rs:129-130` | a caller-programming error (missing required builder field) is typed as `Error::Backend`, indistinguishable from a real backend crash | fixed (M6) |
| 14 | 8/4 | `crates/sdk/src/creature.rs:276-278,414-416` | `wiring()`/`wing_wiring()` return a pre-rendered human summary string; no structured/programmatic accessor | fixed (M12) |
| 15 | 3 | `crates/sdk/src/creature.rs:257-268` | `Creature::fruit_fly()` has two mandatory runtime-checked fields and no simple zero-arg path | **wontfix (M7)** - see below |
| 16 | 13/10 | `crates/fly/examples/watch.rs:18-60` | hand-builds `Fly`/`SdlWindow`/`Renderer` independently of `Creature`/`View` | **not a violation on reconsideration (M7)** - see below |
| 17 | 13 | `docs/` | no user-facing SDK page exists for either surface (rustdoc itself is compliant) | fixed (M9) |

### Minor / stylistic (backlog, not milestoned individually - fold into whichever nearby milestone touches that file)

18. ~~`ImagePipelineBuilder::load()` vs `CreatureBuilder::build()` - two terminal verbs for the same act in one crate.~~ **not a violation on reconsideration**: every OTHER pipeline builder's terminal verb resolves a `<vendor>/<repo>` checkpoint through the model store (`load` = "load what `from_pretrained` named"). `Creature` has no checkpoint at the center of it at all - `creature.rs`'s own module doc says so directly ("the shape here is a builder and a stepping loop rather than `from_pretrained` + `generate`"): it ASSEMBLES a connectome + a body + optional tuning, none of which is individually required. `build` names that act correctly; renaming it to `load` would misdescribe what it does, not fix an inconsistency.
19. ~~No `Creature::builder()` alongside `Creature::fruit_fly()`.~~ **not a violation on reconsideration**: `fruit_fly()` IS the builder constructor (it returns `CreatureBuilder`, pre-seeded with that species' defaults) - a bare `Creature::builder()` would have no defaults to seed and nothing left to configure differently, since this crate serves exactly one species today. Add one the day a second species exists to differentiate from.
20. ~~Unprefixed setters (`drive`, `turn`, `reward`) beside `set_`-prefixed ones (`set_wing_power`, `set_plasticity`, `set_proprioception`, `set_weights`) on the same type.~~ **not a violation on reconsideration**: the two groups name genuinely different acts, not one inconsistently-named act. `drive`/`turn` restate a standing descending command every call (paired with a `command()` getter, re-applied via `apply_command()`); `reward` delivers a one-shot neuromodulator pulse with no stored value to read back (`self.inner.modulate(delta)`) - both read as commands/events, not field assignment. `set_wing_power`/`set_plasticity`/`set_proprioception`/`set_weights` all persist a mode or override until explicitly changed again. `set_` marking "this persists" and a bare verb marking "this is a command" is a real distinction worth keeping, not drift to normalize away - and renaming any of them would break this crate's public API for a purely cosmetic gain.
21. ~~`creature.rs:283-285` - `drive()`'s doc documents a `turn` param it doesn't take; `turn()` next to it has no doc.~~ fixed in M2.
22. ~~`error.rs:82` - `Error::Backend` renders with no `brain: `-style prefix.~~ **wontfix**: `error.rs`'s own module doc documents this as an intentional VERBATIM passthrough for in-process callers (the underlying flux2/s3dit message routinely already self-identifies, e.g. `"flux2: assemble: no dit chosen"`), and `backend_carries_the_original_message_verbatim` pins exactly that contract. A generic prefix would contradict the documented behavior and break the test for no real gain.
23. ~~`creature.rs:63` - `pub use flybody::Arena;` promoted into the compatibility surface with no doc comment justifying it, unlike `Device`/`DType`.~~ fixed: doc comment added, same "re-exported, not reinvented" reasoning `MotorMap`/`WingWiringCounts` right next to it already state.
24. `view.rs:86` - `View::open(creature, title, width, height)` is four positional args with no options type. Still open: unlike `ImageGenerationOptions`/`GroundingOptions`, all four of `open`'s params are REQUIRED (no optional knob exists to carry in a struct yet), so there is no options type to extract without inventing an optional parameter that does not exist today - low priority until one does.

### Checked and clean (recorded so nobody re-audits these)

- Rule 2: one `ImagePipeline` type dispatching internally, no `Flux2Pipeline`/`S3ditPipeline` pair; `from_pretrained` has no second init stage; local paths and hub ids share one call.
- Rule 4: `Image` is the one normalized domain type both backends produce; raw access (`pixels()`) exists without being the only output.
- Rule 7: `Device`/`DType` are re-exports, not parallel types, with the reasoning recorded inline in `lib.rs`.
- Rule 6: `Error::Ambiguous`/`Missing` carry the resolver's structured answer boxed, proven by a test that the structure survives.
- Rule 3 level 2: `ImageGenerationOptions` setters override only their own field, verified by test.
- Rule 13 (feature vocabulary): `crates/sdk/Cargo.toml`'s `image`/`creature`/`device`/`resolve`/`full` features are fully compliant and mechanically gated by `scripts/gates/check-sdk-features.sh`. One accepted consequence, not a violation: `View` (SDL+MuJoCo) is bundled into the `creature` surface, so a headless embedder still links the display closure - splitting it needs a new `Domain` variant, which the gate would demand.

## The flux2 pipeline-construction duplication

Six independent places build a `flux2::Pipeline`, not four:

| # | Site | Role |
|---|---|---|
| 1 | `crates/sdk/src/pipeline.rs:495` (+ `:296` on `load_lora` rebuild) | SDK facade |
| 2 | `crates/cli/src/flux2_cli.rs:696` | one-shot CLI |
| 3 | `crates/flux2/src/caps.rs:432` | `Flux2Action` served path (D-Bus/HTTP) |
| 4 | `crates/cli/src/resident_flux2.rs:320` | flux2 residency adapter |

(The s3dit-backed sites - `crates/sdk/src/pipeline.rs:504`, `flux2_cli.rs`'s
sibling is N/A, `crates/s3dit/src/caps.rs:195`, `crates/cli/src/resident.rs`
- are NOT a duplication problem: `s3dit::pipeline::HotPipeline::build_adapted`
already IS the one shared "resolve config + pick precision + build" call;
every site above it just extracts different params. Only
`ZImageProvider::load()`'s `Paths::from_env` at `crates/cli/src/run_cli.rs:496`
is worth a note - the last env-only construction path while
`crates/catalog/src/lib.rs:262` already resolves from a real `Assembly` - not
a code change on its own.)

**Resolution itself is not duplicated** - all four flux2 sites bottom out in
one core, `loader::resolver::resolve_structured` (`crates/loader/src/
lib.rs:45`), through three deliberately different error-policy wrappers
(typed `Error` for the SDK, `process::exit` for the CLI, an `Option` +
per-role env override for the served/resident paths). That layering is
intentional; it is not the target.

**The real duplication, and a real bug it's hiding**: what happens
IMMEDIATELY BEFORE the build - picking the variant, checking the license,
loading the config, and deciding the precision - is four separate, never-composed
public functions (`flux2::caps::bind_variant`, `flux2::caps::check_license`,
`flux2::Flux2Config::from_name`, `flux2::pipeline::effective_dit_precision`).
`effective_dit_precision` **was** called at only two of the four sites
(`flux2_cli.rs` and the SDK) - `flux2::caps` and `resident_flux2.rs` never
called it, so a `.gguf` DiT served over D-Bus/HTTP missed the
fp32-to-packed-int8 correction the CLI and SDK both apply. **Fixed directly
in M10a**, then **fully unified in M10-full**: `flux2::build::{resolve,
build_resolved}` (`crates/flux2/src/build.rs`) is now the ONE
variant/license/config/precision decision every site composes, via a
`VariantSource` enum (`Assembly`/`Sniff`/`Bound`) covering the three
legitimate ways a caller already knows or has to determine the variant.

- `flux2::caps::Flux2Action::run` (text2image/edit) calls `resolve` (not
  `build_resolved`) with `VariantSource::Sniff` - it needs the resolved
  variant/precision as its hot-pipeline cache KEY before deciding whether a
  rebuild is even necessary, so the actual `Pipeline::build_sized` call
  stays local to the cache-hit/miss branch.
- `resident_flux2.rs::activate` calls `build_resolved` with
  `VariantSource::Bound` inside its existing `on_device` scope - the
  `.gguf` correction here is still reduced-fidelity (can coerce, never
  reject an explicit fp32 misuse), because the `InstanceKey` string already
  discarded whether the request was explicit before `activate` ever runs;
  documented inline, not silently accepted.
- `flux2_cli.rs`'s `generate` command calls `resolve` with
  `VariantSource::Assembly` - it still computes its own tiling/reference-image
  token ceilings before building, so the `Pipeline::build_sized` call stays
  local there too.
- `crates/sdk/src/pipeline.rs` calls `build_resolved` with
  `VariantSource::Assembly` - and ALSO keeps one line of its own:
  `check_license` is still called explicitly first, purely so the SDK can
  still map that one failure to its typed `Error::LicenseRequired` (from
  M6) - `build_resolved`'s own return type is a flat `Result<_, String>`
  shared by every flux2 caller, so it cannot carry that distinction on its
  own. Redundant but harmless (a pure, idempotent check); `build_resolved`
  still re-derives and re-checks the SAME variant internally, so the
  config/precision/build logic converges even though this one check is
  still named twice.

**Deliberately NOT touched**, and why: `flux2::caps::train_action`
(`lora_train`) and `flux2_cli.rs`'s own `lora_train` subcommand each run a
structurally DIFFERENT sequence (`bind_variant`/`check_license`/
`Flux2Config::from_name`, no `Pipeline::build_sized`, no precision decision
at all - training uses `crate::finetune::run` instead) - forcing them
through `build::resolve`, which requires a `Precision` argument that doesn't
apply to training, would be an awkward fit for a sequence that was never
part of the four inference-construction sites this item tracked. Also
untouched: `ZImageProvider::load()`'s `Paths::from_env` (the last env-only
construction path for s3dit) - out of scope, s3dit needed no extraction (see
above).

Verified with `cargo test -p brain-flux2` (101 passed), the full
`cargo test -p brain-cli` (360 + 3 + 5 + 7 + 3 + 2 + 2 + 4 + 4 + 12 + 13 + 1
passed across every test binary, 0 failed), `cargo test -p brain --features
full` (26 passed), `check/sdk-features`, and a full `cargo build --workspace`
including the `samples/imagegen/*` samples that link `crates/sdk` directly.

## Milestone checklist

- [x] **M1** - this document + the `sdk.md` stale-line fixes.
- [x] **M2** - `crates/sdk` cheap mechanical fixes (finding 21 fixed; finding 22 turned out to be intentional on inspection, marked wontfix).
- [x] **M3** - `Creature` gets a real device (findings 9, 10). Factored the
      probe/resolve/apply/publish core out of `pipeline.rs::apply_device`
      into a shared `crate::device::resolve` (gated on the `device` feature
      alone, since `creature` selects `device` but not `resolve` and has no
      loader dependency); `pipeline.rs`'s own `apply_device` now layers only
      the `resolve`-specific placer install on top. `CreatureBuilder::device`
      mirrors `ImagePipelineBuilder::device`; `Creature::build` now calls
      `gpu_core::Gpu::new(&neuro::KERNELS)` - the same production
      constructor every `resident_*.rs` uses - instead of the test pool.
- [x] **M4** - `Creature`/`View` end-to-end test (findings 1, 2).
      `crates/fly`'s own suite has no synthetic connectome/body fixture
      either - every one of its tests gates on real `BRAIN_CONNECTOME_DIR`/
      `BRAIN_FLYBODY_XML` data with a clean skip (`crates/fly/tests/
      loop_closes.rs`'s `rig()`), so `crates/sdk/tests/creature.rs` follows
      the same convention rather than inventing a fixture this workspace has
      never needed before. Two tests need no real data at all (the missing
      `.connectome()`/`.body()` argument errors, always-green); two need the
      real env vars and skip cleanly via `brain_testutil::skip_unavailable`
      when absent (build/drive/step/reset, and `View::open`/`show`/`frame`
      at the requested size) - honest about the gap per rule 14, same as
      `tests/image_pipeline.rs`'s own documented ceiling.
- [x] **M5** - generate-time progress/cancellation wired for real (findings
      3, 5). Added `ImagePipeline::generate_with_progress(prompt, opts,
      cancel, on_progress)` - the full-control call both `generate`/
      `generate_with` now delegate to at their old defaults, so existing
      callers see no change. `Error::Cancelled` is reachable through it now;
      the string-to-`Error` mapping that makes it so was factored into
      `backend_err_or_cancelled` and unit-tested directly, since neither
      existing fixture reaches a live pipeline to exercise it end to end
      (same documented ceiling as `tests/image_pipeline.rs`).
- [x] **M5b** - download progress (`ImagePipelineBuilder::load`'s
      `execute_plan` call, finding 4) and s3dit build progress
      (`HotPipeline::build_adapted`'s `impl FnMut(&str)`) are now plumbed
      through a new `ImagePipelineBuilder::load_with_progress(self,
      on_download: &mut dyn FnMut(&str, u64, Option<u64>), on_build: &mut
      dyn FnMut(&str))`, which `load()` delegates to with two no-op
      closures - the design question the original note raised ("where does
      a boxed closure live on a by-value builder") resolved by NOT storing
      either closure on the builder at all: both are call-scoped parameters,
      exactly mirroring `ImagePipeline::generate_with_progress`'s own
      shape, rather than new builder state with its own lifetime/`Send`
      questions. `on_build` is a real no-op on a flux2-resolved pipeline
      (documented, not hidden): neither `flux2::build_resolved` nor
      `Pipeline::build_sized` takes a build-progress hook at all today.
      Proven against the s3dit fixture `image_pipeline.rs` already carries:
      `load_with_progress_reports_s3dit_build_stages_before_the_same_clean_failure`
      shows `on_build` fires at least once (`HotPipeline::build_adapted`'s
      own "loading tokenizer" stage) before the same clean `Error::Backend`
      the no-progress test gets, and `on_download` never fires against a
      fixture `Store::local` already resolves with no network access.
- [x] **M6** - split `Error::Backend`'s catch-all, partially (findings 12,
      13). Added the two clearly load-bearing variants - `LicenseRequired`
      (a caller plausibly wants to react differently: surface the terms,
      fall back to an ungated variant) and `MissingArgument` (a
      caller-programming error, knowable before any backend/GPU/filesystem
      call, distinct from a real backend crash) - and wired
      `flux2::caps::check_license`/`CreatureBuilder::build`'s two required-field
      checks onto them, each proven reachable by a real unit test (the
      license one calls the real `flux2::caps::check_license`, gated on the
      `image` feature since `flux2` is optional). Deliberately did NOT add
      `UnsupportedFormat` (image.rs's bad-extension case) or a structured
      size-mismatch variant: `imaging::save`'s error is an untyped `String`
      with no way to distinguish "bad extension" from "I/O failure" short of
      either duplicating its own extension-dispatch match (a second copy to
      drift from) or substring-matching a message never designed as a
      stable sentinel (unlike `"cancelled"`, which IS one, by this crate's
      own convention). Left as `Error::Backend` until `imaging::save` itself
      grows a typed error - not a gap in this milestone's judgment, a gap in
      what's cleanly extractable today.
- [x] **M7** - `Creature` progressive disclosure + `watch.rs` (findings 15,
      16), both resolved without a code change:
      - **Finding 15 (wontfix)**: `resources/README.md` is explicit that
        nothing under `resources/` is vendored into the repo - the MANC
        connectome and the flybody body model are fetched at use time
        specifically so the license gate (attribution requirements, and one
        component whose Codex redistribution terms are unestablished) stays
        honest. A zero-arg `Creature::fruit_fly().build()` default pointing
        at `resources/connectome` would only work when run from this exact
        repo checkout with that script already run - actively misleading
        for an embedded library, not a convenience. There is no real default
        to offer; forcing `.connectome()`/`.body()` explicitly is the
        correct behavior for a dataset that cannot be bundled, not a
        builder-tax bug.
      - **Finding 16 (reconsidered, not a violation)**: unlike
        `flux2_cli.rs` (a PRODUCTION path serving real users, duplicating
        the SAME end-user capability `ImagePipeline` already implements -
        two implementations of one thing, which is where the real
        `effective_dit_precision` bug hid), `crates/fly/examples/watch.rs`
        is a crate-internal developer example exercising `brain-fly`'s own
        raw `Wiring`/`Timing`/descending-command API for debugging the
        connectome/body coupling itself - not a second implementation of an
        end-user capability. `samples/fly/interactive` already IS the
        SDK-facing "watch a fly in a window" experience for embedders, so
        there is no end-user-facing gap. Rewriting `watch.rs` onto
        `brain::Creature`/`View` would need `crates/fly` (layer 4) to take a
        dev-dependency on `crates/sdk` (layer 6, which already depends on
        `brain-fly`) purely for one example - Cargo permits a dev-dependency
        cycle like this, but it inverts the intended layering direction for
        no real gain, since the example's whole point is to exercise the
        lower-level API the SDK deliberately hides.
- [x] **M8** - `.capabilities()` introspection (finding 6). Turned out to
      need no design decision at all: `flux2::caps::manifest()` and
      `s3dit::caps::manifest()` are already free functions returning "the
      full, static capability manifest - safe to build with no weights
      loaded" (their own doc comment), the SAME one `brain caps`/D-Bus/HTTP
      read for these architectures. `ImagePipeline::capabilities()` just
      dispatches to whichever one matches the resolved backend - reflecting
      the real manifest rather than growing a second, weaker description.
      Not separately unit-tested: reaching a live `ImagePipeline` needs the
      same real weights `generate()`'s own success path does (tracked
      above), and the two manifest functions already have their own tests
      in their home crates.
- [x] **M9** - user-facing SDK docs page (finding 17): `docs/using/sdk.md`,
      leading with both surfaces' three-line examples before options,
      features, errors, and the resource-safety note; registered in
      `docs/readme.md`'s "Using brain" list next to `using/cli.md`.
- [x] **M10a** - the precision bug itself, fixed directly rather than via
      the full `build_resolved` extraction (see below for why that's
      still open). `flux2::caps::Flux2Action::run`'s served `text2image`/
      `edit` branch now calls `effective_dit_precision(&paths.dit,
      p.precision, precision_was_explicit)` before building - with FULL
      fidelity, since `inv.params.get("precision").is_some()` recovers
      whether the caller stated it explicitly, matching the CLI/SDK paths'
      behavior exactly (an explicit fp32 request against a `.gguf` DiT is
      still a named error, not silently coerced). `resident_flux2.rs`'s
      `activate()` gets the same correction, but at REDUCED fidelity: the
      `InstanceKey` string (`instance_key`) already collapsed "explicit
      fp32" and "defaulted to fp32" into one value before `activate` ever
      sees it, so it can only coerce (`f32_was_explicit: false`, always),
      never reject an explicit misuse the way the other three paths do -
      documented inline rather than silently accepted. The bug itself (a
      served `.gguf` DiT building at the wrong precision) is fixed on both
      remaining call sites either way, which is what mattered.
      Not separately regression-tested beyond the existing suites passing:
      `effective_dit_precision` itself already has full branch coverage in
      `crates/flux2/tests/placement.rs`, and proving these two NEW call
      sites build a real `Pipeline` at the corrected precision would need a
      real `.gguf` DiT checkpoint to sniff and build against, which doesn't
      exist in this workspace's fixtures (same class of gap as
      `tests/image_pipeline.rs`'s own undocumented `generate()` success
      path). `cargo test -p brain-flux2` (97 passed) and the relevant
      `brain-cli` resident_flux2 tests (2 passed) show no regression.
      **`build_resolved` itself - the actual "one implementation" extraction
      unifying all four flux2 sites' variant/license/config/precision
      decision into one function - landed separately as M10-full** (same
      session); fixing the live bug first, safely, was worth doing before
      committing to that larger refactor's shape.
- [x] **M10-full** - `flux2::build::{resolve, build_resolved}`
      (`crates/flux2/src/build.rs`, new module) unifies all four sites onto
      one variant/license/config/precision decision via a `VariantSource`
      enum (`Assembly`/`Sniff`/`Bound`). Full detail, including the one
      honest compromise that remains (the SDK still calls `check_license`
      once more itself, purely to keep its typed `Error::LicenseRequired`)
      and what was deliberately left untouched (`lora_train`'s structurally
      different sequence, s3dit's already-unified path), is in the
      duplication section above. 4 new unit tests in `build.rs` plus full
      regression runs across `brain-flux2`/`brain-cli`/`brain` (sdk) and a
      whole-workspace build - see that section for exact counts.
- [x] **M10b** - finding 11: `ImagePipelineBuilder::size` was silently
      IGNORED on a flux2-backed pipeline (flux2 always built at its own
      1024x1024 default forward-token ceiling regardless of what a caller
      asked for). New `forward_tokens_for(width, height)` reuses flux2's own
      `gen_tokens_per_forward` at a caller-chosen canvas instead of the
      hardcoded default; `ImagePipelineBuilder::load_with_progress` picks it
      when `.size(...)` was called. Also fixed the same bug's other half:
      `ImagePipeline::load_lora`'s flux2 rebuild recomputed
      `default_forward_tokens()` from scratch, so folding an adapter in
      would have silently SHRUNK a caller-requested ceiling back to the
      default - `Flux2Backend` now records its own built `forward_tokens`
      (mirroring `S3ditBackend::width`/`height`'s existing pattern) and
      `load_lora` reuses it. 2 new unit tests
      (`forward_tokens_for_matches_the_default_at_the_default_canvas`,
      `forward_tokens_for_grows_with_a_larger_requested_canvas`); no fixture
      test possible past this point without a real multi-GB flux2 build (the
      same documented ceiling `tests/image_pipeline.rs`'s own module doc
      explains).
- [x] **M11** - finding 8: `set_plasticity`/`reward` mutate a `Creature`'s
      learned synapse weights with no way to persist them - all learning
      died with the process, CLI included, since `fly::Fly` already exposes
      the raw state (`weights()`/`set_weights()`) but nothing in this
      workspace ever wired a save/load path onto it. Added
      `Creature::weights`/`set_weights`/`save_weights`/`load_weights`, the
      last two as a one-tensor safetensors checkpoint
      (`checkpoint::st::save_safetensors`/`load_safetensors`) - the same
      format every other numeric-vector checkpoint in this workspace uses,
      not a bespoke binary format, and free in the `creature` feature's
      build graph (`brain-checkpoint` was already a transitive dependency
      via `brain-fly`/`brain-connectome`, confirmed via `cargo tree`).
      `weights_tensor`, the `"weights"`-tensor lookup, is factored out
      testable with no real `Creature`/GPU/MuJoCo handle in hand (2 unit
      tests); `tests/creature.rs` gained
      `save_and_load_weights_round_trips_through_a_real_connectome`,
      gated on the same `BRAIN_CONNECTOME_DIR`/`BRAIN_FLYBODY_XML` env vars
      the file's other real-fixture tests already use - it ran for real (not
      skipped) in this session's environment, perturbing the weights before
      saving so a bug that silently kept the OLD weights could not pass by
      coincidence.
- [x] **M12** - finding 14: `wiring()`/`wing_wiring()` returned only a
      pre-rendered human summary string, with no programmatic accessor.
      `flybody::MotorMap` was ALREADY fully structured
      (`mapped()`/`actuators_driven()`/`neurons_for()`/`unmapped`/`adhesion`)
      but entirely unreachable from `Creature` - added
      `Creature::motor_map(&self) -> &flybody::MotorMap` alongside the
      existing `wiring()`, re-exported (not reinvented) the same way
      [`Arena`] already is. The wing half needed one small upstream change,
      since `fly::Fly::wing_summary` only ever computed its counts as local
      variables and threw them away: added `fly::WingWiring` (a small
      `power`/`amplitude`/`angle_of_attack` struct) and
      `Fly::wing_wiring() -> WingWiring`, refactored `wing_summary` to
      render its string FROM it rather than a separately-derived
      computation, and exposed it as `Creature::wing_wiring_counts()`.
      Proven with real fixtures at both layers: `crates/fly/tests/
      flight.rs`'s existing `wing_summary().starts_with("24 power")`
      assertion gained a sibling `wing_wiring().power == 24` (same fact, two
      representations, both pinned), and `tests/creature.rs`'s
      `build_drive_step_and_reset_a_real_fly` gained assertions that
      `motor_map().mapped() > 0` and that `wing_wiring()`'s string is
      rendered from EXACTLY `wing_wiring_counts()`'s own numbers, both
      running for real (not skipped) against this session's real
      connectome/body fixtures.
- [ ] **Phase 2** - new pipelines, in the priority order above: Forecast, Text, Embedding, ASR, then the rest. Each gets its own sub-roadmap section here (or its own file, linked from here) when it starts, written against the full `sdk-design.md` checklist from day one - including an end-to-end test, learning from M10 rather than repeating the `flux2_cli.rs` duplication gap a second time. Design each pipeline's progress/cancellation surface (rule 8) toward the `run.start()/subscribe()/cancel()/result()` shape `.agents/roadmap/orchestration-hsm.md` proposes, rather than reinventing M5's synchronous `generate_with_progress(cancel, on_progress)` a second time - M5's shape stays the right SIMPLE default, but a new pipeline's ADVANCED tier should point at where this is heading.
  - [x] **Phase 2.1** - `ForecastPipeline` (kronos, timesfm3).
  - [x] **Phase 2.2** - `TextGenerationPipeline` (qwen3 only, local path only at first).
  - [x] **Phase 2.2b** - `qwen3::spec::Qwen3Spec` (Phase 2.2's own tracked prerequisite) - `TextGenerationPipeline::from_pretrained` now also accepts a hub id, resolved through it. See its own section below (inserted after 2.2) for a real Store::local-ordering bug this would have reintroduced, caught before shipping.
  - [x] **Phase 2.3** - `EmbeddingPipeline` (CLIP text towers only).
  - [x] **Phase 2.4** - `TranscribePipeline` (qwen3-asr only).
  - [x] **Phase 2.5** - `UpscalePipeline` (RRDBNet only) - see its own section above for the real `RrdbnetSpec` bug this one found and fixed, and the new `Image::open`/`Image::from_rgb8` public API it needed.
  - [x] **Phase 2.6** - `codeformer::spec::CodeFormerSpec`, the prerequisite Phase 2.5 named - see its own section below.
  - [x] **Phase 2.7** - `RestorePipeline` (CodeFormer) - see its own section below for two real forward-pass infrastructure bugs this one found AND FIXED (a duplicate kernel registration that broke the CPU JIT backend; a `backend-wgpu` buffer-reclaim ceiling from two unpolled `Builder` scopes) - this pipeline's test reaches a genuine, complete `.restore()` forward pass at CodeFormer's real fixed geometry, the strongest end-to-end proof of any pipeline in this crate so far. SUPIR and VQGAN stay deferred with the reasons already on record.
  - [x] **Phase 3.1** - `DetectionPipeline` (YOLOv8 only) - the vision/detection domain bucket's first pipeline, and its first NEW domain object (`Detection`, not `Image`). See its own section below.
  - [x] **Phase 3.2** - `SegmentPipeline` (SAM 2.1) - see its own section below for a real, independent `Sam2Spec` bug found and fixed (the SAME `ArtifactKind::Opaque`-vs-`Torch` mistake `RrdbnetSpec` had), and a genuine end-to-end `.segment()` forward pass at SAM 2.1's real fixed geometry.
  - [x] **Phase 3.3** - `DepthPipeline` (ZipDepth) - completes the vision/detection bucket - see its own section below for a real `cfg_for_checkpoint` shape bug fixed, and a real `vision`/`image` feature-split bug the gate itself found.
  - [x] **Phase 4.1** - `TtsPipeline` (Qwen3-TTS: speak/clone_voice/design) - the TTS/music bucket's first pipeline - see its own section below. cosyvoice/minimaxmusic3 deferred.
  - [x] **Phase 5.1** - `VideoPipeline` (Wan2.1 T2V) - the video generation bucket's first pipeline - see its own section below for a real "invisible to the real scanner" gap found (a `.bin` sibling silently vanishing from `inventory::scan`). ltxv deferred (still `always!()`-registered, not resolver-based).
  - [x] **Phase 5.1b** - a real `Store::local`-ordering bug in `VideoPipelineBuilder::load`, found on RE-INSPECTION rather than a failing test: `load()` still used the naive "check `Store::local`, then fetch-if-missing, then resolve" order every other pipeline in this crate has since moved away from (Phase 4.2/2.2b/6.1 each found and fixed the SAME class). The bug was MASKED by `tests/video_pipeline.rs`'s own fixture: its DiT file was deliberately named `model.brain.safetensors` - `Store::local`'s own magic `BASE_WEIGHTS_FILE` fallback name - so the test always passed without ever exercising a REAL Wan release's own diffusers-convention filenames (`diffusion_pytorch_model.safetensors`, `Wan2.1_VAE.pth`, ...), which satisfy neither of `Store::local`'s two recognized shapes. Fixed with the same resolve-first reordering as the other three; the fixture's DiT filename changed to the real one (`wan::spec::tests::t2v_1_3b_fixture`'s own naming, mirrored not reinvented) so the test now actually proves the fix matters rather than passing by filename coincidence. A reminder that "the test passed" and "the real-world path works" are not the same claim when a fixture can accidentally satisfy a narrower check than the one being exercised.
  - [x] **Phase 4.2** - CosyVoice (cosyvoice2) joins `TtsPipeline` via `clone_voice` - see its own section below (inserted after 4.1) for a real bug found and fixed (`Store::local` never recognizes ANY cosyvoice checkpoint, even a real one - no `FilesRecipe` entry exists for it), how `speak`/`design` return `Error::MissingArgument` on this backend, and `samples/tts/clone`, a real Rust sample requested mid-sweep. minimaxmusic3 still deferred.
  - [x] **Phase 6.1** - `VisionLanguagePipeline` (Qwen3-VL: 1-8 images + text in, text out) - the Multimodal/VLM/OCR domain bucket's first pipeline. See its own section below for why qwen3vl was picked first, and a real `Store::local`-ordering bug caught before shipping (the same class Phase 4.2/2.2b each found the hard way). Video input, tool-calling, and 8 more served architectures in this bucket stay deferred.
  - [x] **Phase 6.2** - `GroundingPipeline` (Florence-2) - the Multimodal/VLM/OCR bucket's SECOND pipeline, a NEW type rather than a `VisionLanguagePipeline` second backend or a `DetectionPipeline` third one: `ground` takes an open-vocabulary text `target` phrase in and returns boxes keyed by that phrase, not free text out (rules it out of `VisionLanguagePipeline`'s shape) and not a fixed trained class index (rules it out of `Detection`'s). `florence2::spec::Florence2Spec`/`caps.rs` already existed, mature and served (`brain do florence2 ground`), and unlike qwen3vl's paramstore upload path, `FlorenceSession::load`'s own tensor-manifest check fails CLEANLY on incomplete content (`build_param_source`'s `.ok_or_else(...)?`, never a panic) - so this pipeline's test reaches real CONSTRUCTION over a fixture with fake tensor content, a stronger ceiling than Phase 6.1's own resolution-only coverage. `Florence2Spec::classify` reads a plain HF `config.json`'s `model_type` field (`TransformersRecipe`'s own catch-all shape), so the resolve-first ordering was applied proactively for consistency, not because a failure was reproduced here.
  - [x] **Phase 4.3** - cosyvoice3 variant validation: `crates/sdk/tests/tts_pipeline.rs` gained `from_pretrained_resolves_cosyvoice3_from_a_real_local_fixture_with_no_network_access`, a CosyVoice 3-shaped fixture (no `llm_embedding.weight`, `decoder.estimator.transformer_blocks.*`, hift conv_pre kernel width 5 - mirroring `cosyvoice::spec::tests`' own generation-telling-apart tensors) proving `TtsPipeline::from_pretrained` + `.clone_voice_with(..., TtsOptions::new().variant("cosyvoice3"))` resolves end to end. Zero production code changed - `cosyvoice::pipeline::generate`'s CosyVoice 3 branch (`CosyVoiceLm::load_cosyvoice3`) already existed and worked; only SDK-level proof was missing. Closes the gap Phase 4.2 left open.
  - [x] **Phase 2.2c** - closed a real unchecked-panic gap
        `TextGenerationPipelineBuilder::load`'s local-path branch had:
        unlike the hub-id path (`Qwen3Spec::classify`, free architecture
        checking), a literal local checkpoint path skipped architecture
        validation entirely and went straight to `Qwen::load_inference`,
        which panics on mismatched tensor names rather than erroring
        cleanly - the same unchecked-panic class this module's own
        `WeightReader::open`-before-`load_inference` ordering already
        guards against for a bad PATH, just not yet for a bad ARCHITECTURE.
        Promoted `qwen3::spec::{GGUF_ARCHITECTURE, CARD_FAMILY}` to `pub`
        (one literal, not a second copy) and added
        `check_local_weights_architecture`, which refuses a checkpoint by
        name ONLY on POSITIVE evidence of a mismatch (a declared
        `general.architecture`/`ModelCard.family` that isn't qwen3's own) -
        a checkpoint carrying neither marker is let through unchanged,
        mirroring `Qwen3Spec::classify_gguf`/`classify_safetensors`'s own
        "assert a positive match, never reject an absence of one"
        contract. New test proves a checkpoint declaring `ModelCard.family
        == "qwen35"` is refused with a clean, named `Error::Backend` before
        ever reaching `load_inference`, with every pre-existing test in
        `text_pipeline.rs` (whose fixtures carry no card at all) unaffected.
  - [x] **Phase 4.4** - `MusicPipeline` (MiniMax Music 3) - the TTS/music
        bucket's second pipeline, a NEW type rather than a `TtsPipeline`
        third backend: lyrics+caption in (not a speaker/voice to render text
        in), up to five minutes of 44.1 kHz STEREO out - a genuinely
        different call shape and a genuinely different output domain (see
        `crates/sdk/src/music.rs`'s own module doc for why this earns
        `Song`, not a `channels` field bolted onto `Audio`).
        `minimaxmusic3::spec::MinimaxMusic3Spec`/`caps.rs` already existed
        and read as production-representative (unlike ltxv - see that
        model's own module doc: "today's DiT is a tiny random-weight
        smoke-test pipeline" by default, real weights "only ever proven
        correct at REDUCED DEPTH / one block" - explicitly NOT the same
        maturity bar, and correctly still deferred). Applied the resolve-
        before-`Store::local` ordering PROACTIVELY from the start (the
        fourth time this campaign has needed it - Phase 4.2/2.2b/6.1/5.1b
        each found the naive order broken the hard way first). `Song::save`
        reuses `audio::wav::write_multi`, already-existing multi-channel
        infrastructure `crate::Audio::save`'s own mono `audio::wav::write`
        call is the one-channel special case of - no new WAV-writing code
        needed. Resolution-only test coverage, the same ceiling
        `tests/vlm_pipeline.rs` accepted for qwen3vl: MinimaxMusic3Spec
        classifies across five genuinely different on-disk shapes (one real
        HF `language_model` dir, four brain-native components with no
        `config.json` of their own, classified by tensor NAME alone),
        reverse-engineering all four native tensor-name manifests for a
        synthetic fixture is out of scope here - what IS proven is an
        unparseable id refused cleanly and a `Missing` naming all six roles,
        never a panic, never a network attempt.
  - [x] **Phase 3.4** - the SAME `Store::local`-ordering bug, found
        SYSTEMICALLY across the vision/detection bucket rather than one
        pipeline at a time: `DepthPipelineBuilder`/`EmbeddingPipelineBuilder`/
        `RestorePipelineBuilder::load` all still used the naive order.
        Checked EVERY naive-order pipeline against
        `brain_modelstore::recipe`'s `FILES_RECIPES`/dedicated-`Recipe`
        list (the actual determinant: a real checkpoint whose on-disk shape
        a recipe converts into a compound `brain.manifest.json` works fine
        under the naive order regardless; one with NO matching recipe does
        not, because `plan()` fails outright even for an already-local
        checkpoint) - `sam2`/`rrdbnet`/`yolov8` all have one (confirmed by
        reading `FILES_RECIPES`/`YoloRecipe`, not assumed) and are FINE as
        naive-order code, `zipdepth`/`clip`/`codeformer` have none and are
        NOT. Confirmed empirically for all three, not just reasoned about: a
        real fixture at EXACTLY the repo path its own `model_id` names, with
        no `mark_locally_present`-style manifest anywhere else in the store,
        reached `Error::Download("not found: <id>@main")` for
        zipdepth/codeformer and `Error::ModelNotFound("...: no config.json in
        repo")` for clip (SDXL's real `model_index.json`-keyed layout, not
        `TransformersRecipe`'s expected top-level `config.json`) - a real
        network hub query, or a real classification refusal, for a
        checkpoint that was already fully present on disk. Fixed all three
        with the same resolve-first reordering as Phase 4.2/2.2b/6.1/5.1b;
        each gained a NEW test proving the exact fixture that used to fail
        now resolves with no `Store::local` shortcut and no network access.
  - [ ] Still entirely uncovered domain buckets: 3D/world models (lowest priority - no settled domain object yet, `SplatPipeline` closer to `Creature` than `ImagePipeline`). See the domain inventory table. Also still open within buckets already started: ltxv (video, unproven past one block - see Phase 4.4's own note), Text decoders bucket (`TextGenerationPipeline` scoped to qwen3 only, 5 more decoders behind it - a real, larger gap than Phase 2.2c's fix: `qwen35`'s own load path differs enough from qwen3's - footprint-based sizing, int8 tier, GPU pipeline construction, a ~108GB fp32 real model - to need its own design pass, not a copy of the `Backend` enum pattern `TtsPipeline`'s CosyVoice addition used), Multimodal/VLM/OCR beyond qwen3vl/florence2 (7 more served archs, plus video/tool-calling on qwen3vl itself - assessed this session: none of fastvlm/llava/moondream3/deepseek2ocr share qwen3vl's `Resident`-style chat+tools call shape, so none fit `VisionLanguagePipeline` as a same-type second backend the way CosyVoice fit `TtsPipeline`; florence2 instead earned its own `GroundingPipeline` type, Phase 6.2), ASR bucket beyond qwen3-asr (nemotron, streaming - a genuinely different call shape, already documented as deliberately deferred in `crates/sdk/src/asr.rs`'s own module doc).

### Phase 2.1 - `ForecastPipeline` (done)

Covers **kronos and timesfm3** - the two forecasting architectures with a
real model-store `ArchSpec` today (`kronos::spec::KronosSpec`,
`timesfm3::spec::Timesfm3Spec`), resolved through the SAME `loader::
resolve_structured` call `ImagePipeline` uses. **chronos2 and fincast are a
real, tracked gap, not silently unsupported**: neither has a `default_ref`
or any resolver `ArchSpec` registered anywhere in `brain_arch::ARCHS` -
both load from a plain `BRAIN_CHRONOS2`/`BRAIN_FINCAST` env var today
(`resident_forecast.rs`), so there is nothing for a store-based resolver to
resolve them against; wiring them up is upstream work in those two model
crates, not something to fake at the SDK layer.

Simpler than `ImagePipeline` in one real way: all four forecasting
architectures (including the two not wired in here) already share ONE
object-safe trait, `forecast::ForecastModel` - so `ForecastPipeline` holds a
plain `Box<dyn ForecastModel>` with no per-architecture match arm anywhere
past construction, unlike `ImagePipeline`'s `Backend` enum. The domain
object (`forecast::Forecast`/`Panel`/`Variate`/`Capabilities`/...) already
existed, well-designed to rule 4's own standard (an explicit `derived: bool`
+ `method` on every value brain computed rather than the model emitting,
never fabricating a representation that isn't mathematically sound) -
re-exported directly rather than wrapped a second time (rule 58).

`Error::Forecast` is a NEW enum variant, carrying `code`/`message`/
`retryable` - but as a locally-defined `ForecastFailure` struct, not
`forecast::ForecastError` itself, specifically so `Error`'s own shape does
not depend on the `forecast` feature (`Error` is documented as one shape in
every configuration; a feature-gated variant would break that). The wire-only
structured `detail` JSON blob is deliberately not carried across - `message`
already states any numbers it would repeat.

`crate::device::apply` (the `resolve`-tier device+placer setup) was factored
out of `pipeline.rs`'s own private `apply_device` into `device.rs` proper
(alongside the `device`-tier-only `resolve` from M3), since `forecast` now
needs the exact same "resolve → apply → install the model-shard placer"
sequence `image` does - the alternative was a second private copy in
`forecast.rs`, which is exactly the class of duplication this whole
sweep exists to prevent.

Tested against real local, synthetic, fully-offline fixtures reproducing
`crates/kronos/src/spec.rs`'s and `crates/timesfm3/src/spec.rs`'s own
(private) classification schemas, mirroring `tests/image_pipeline.rs`'s
established pattern exactly: resolution + dispatch proven all the way to
each architecture's real, unavoidable ceiling (`Forecaster::load` needs
every weight tensor its config implies; neither model has a tiny injectable
config), then a clean typed `Error::Backend`, never a panic.

**Not done, tracked for later**: CLI migration (`forecast_cli.rs`'s
`predict`/`compare` commands are NOT rebuilt onto `ForecastPipeline` in this
change - they also own CSV parsing, baseline comparison, chart rendering and
several sampling-parameter env-var overrides well beyond this pipeline's
scope, and `--timesfm3 <path>` there is a raw path argument that never goes
through the resolver at all). This is the SAME kind of gap `ImagePipeline`
itself still has against `flux2_cli.rs` - tracked, not silently accepted,
and not repeated by pretending it's smaller than it is.

### Phase 2.2 - `TextGenerationPipeline` (done, scoped to qwen3 + a local path)

Covers **qwen3 only, loaded from a literal local checkpoint path, not a
`<vendor>/<repo>` hub id** - `crates/qwen3` has no model-store `ArchSpec` at
all (unlike `crates/qwen35/src/spec.rs`, which does, in the exact
`["weights", "tokenizer"]`-role shape a future `qwen3::spec::Qwen3Spec`
should copy). Writing that `ArchSpec` is upstream work in the qwen3 crate,
tracked here rather than attempted inline - the same treatment
chronos2/fincast got in Phase 2.1, applied to the highest-value remaining
architecture instead of the easiest one.

The construction sequence (checkpoint-open, config-parse, tokenizer
precedence, device-aware placement) is real and mirrors `qwen_cli.rs`'s own
`infer` command and `qwen3::caps::GenerateAction` byte-for-byte (same
`checkpoint::weightio::WeightReader::open` pre-check before the panicking
`Qwen::load_inference`, same `qwen3::footprint::place_and_build` VRAM-budget
wrapper, same tokenizer-precedence rule: an explicit path wins, else a
`.gguf`'s own embedded tokenizer, else a named error). The GENERATION call
is not a third implementation of chat templating/sampling/stop-strings: it
runs `qwen3::chat::parse_request` + `qwen3::chat::SeqState` +
`qwen3::sample::generate_kv_stream` - the exact sequence
`caps::GenerateAction::run` (the served `/v1/chat/completions` path) runs -
built from a plain in-process `capability::Invocation`, with no
capability-dispatch server, no scheduler and no paged KV cache anywhere in
the loop. This is what "CLI/SDK/server converge on one implementation"
(rule 10) looks like from day one, rather than something to migrate onto
later the way M10 had to for flux2.

The context budget (prompt + completion, in tokens) is fixed at BUILD time
(`TextGenerationPipelineBuilder::capacity`, default 4096) rather than
resized per call - the same `s3dit` build-time-size asymmetry
`ImagePipelineBuilder::size` already documents, for the same underlying
reason: `Qwen`'s KV cache is sized once, at construction.
`TextGenerationPipeline::generate_with` validates a request against it and
names the fix in the error rather than silently truncating or rebuilding.

Tested against a real (if minimal) local safetensors fixture: an empty `{}`
config header, which `QwenConfig::from_json`'s own defaults happen to
resolve to EXACTLY `QwenConfig::tiny()` - genuinely cheaper than
`ImagePipeline`'s or `ForecastPipeline`'s fixtures, which both need real
classifiable tensor shapes. What stays out of reach: `data::qwen_tokenizer::
QwenBpe` has no synthetic/in-memory constructor, only `from_file`/
`from_dir`/`from_gguf`/`from_json_bytes` reading a real HF `tokenizer.json`
schema, so these tests prove construction through checkpoint-open and
config-parse, then a clean `Error::MissingArgument` at the tokenizer step -
never reaching the heavier `Qwen::load_inference` call, and never a panic.

**Not done, tracked for later**: kronos/timesfm3-style model-store
resolution (needs `qwen3::spec::Qwen3Spec` first, see above); CLI migration
(the SIX qwen-family CLI files - `qwen_cli.rs`/`qwen35_cli.rs`/
`qwen35moe_cli.rs`/`glm_cli.rs`/`gpt_cli.rs`/`lfm_cli.rs` - are untouched,
same class of gap as `flux2_cli.rs`/`forecast_cli.rs`); a `TextGenerationPipeline`
type that ALSO dispatches across the other decoder families the way
`ImagePipeline` dispatches across flux2/s3dit, which is what rule 2 actually
asks for long-term - this milestone proves the shape on the single
most-complete backend first, deliberately not the full consolidation in one
change.

### Phase 2.2b - `qwen3::spec::Qwen3Spec`, closing Phase 2.2's own tracked gap, plus `TextGenerationPipeline::from_pretrained` accepting a hub id

Writes the `ArchSpec` Phase 2.2 named as prerequisite work and deferred -
`crates/qwen3/src/spec.rs`, mirroring `crates/qwen35/src/spec.rs::Qwen35Spec`
structurally (same `["weights", "tokenizer"]` role pair, same GGUF-vs-
brain-format-safetensors duality, same GGUF-self-satisfies-its-own-tokenizer
fallback), adjusted for qwen3's own real constants:
`general.architecture == "qwen3"` (`crate::gguf_import::GGUF_ARCHITECTURE`)
for a GGUF release, `ModelCard.family == "qwen"` (NOT `"qwen3"` -
`crate::import::convert`'s own `ModelCard::new(id, "qwen")`, the family
string qwen3's own brain-format conversion has always written) for a
brain-format `.safetensors` checkpoint. 7 new unit tests, mirroring
`Qwen35Spec`'s own test suite one-for-one, all passing on the first attempt.

**`TextGenerationPipeline::from_pretrained` now accepts EITHER a local path
or a hub id, through the SAME call** - the real, tracked gap Phase 2.2 left
open (rule 2: "never a separate API for load-from-disk vs. load-from-hub").
The two are told apart with NO guessing: a string that names a real file
already on disk (`Path::new(s).is_file()`) is always a local path, even if
it happens to be syntactically parseable as `<vendor>/<repo>` too (a
relative path with exactly one `/`, e.g. `out/qwen3-4b.safetensors`, is a
real, checked-for ambiguity - see `crates/sdk/src/text.rs`'s own module doc).
Only a string that is NOT an existing local file is tried as a hub id.

**A real bug this milestone's own new hub-id path would have reintroduced,
caught before it shipped by applying the lesson Phase 4.2 already
learned**: a naive "check `Store::local`, fetch-if-missing, then resolve"
order (the shape every OTHER pipeline in this crate uses) would have failed
the exact same way cosyvoice did - a real, already-downloaded qwen3 GGUF
release (the common case; `unsloth/Qwen3-4B-GGUF`-shaped, cited directly in
`Qwen35Spec`'s own test fixture comments for its sibling architecture) is
neither a compound `brain.manifest.json` nor a bare `model.brain.safetensors`,
so `Store::local` would never recognize it and `plan()` would fall through
to `TransformersRecipe`'s catch-all, which cannot read a bare GGUF's
(nonexistent) `config.json`. `resolve_hub_weights` tries `qwen3::spec::
Qwen3Spec` resolution FIRST, exactly mirroring `TtsPipelineBuilder::load`'s
own fix, before ever consulting `Store::local`/`plan`.

**`crate::device::apply` replaces the bare `crate::device::resolve` this
pipeline called before** - a real, small inconsistency this milestone's own
`check-sdk-features.sh` run against `--features text` alone surfaced as a
"function never used" warning (harmless in a `--features full` build, since
other surfaces call `apply` too, but a genuine dead-code path in a narrow
standalone `text`-only build). `text` now selects `resolve` (not bare
`device`) since a hub id genuinely can reach `crates/loader`'s model-store
resolution - `apply` is the call every OTHER `resolve`-tier pipeline
(`EmbeddingPipeline`/`DetectionPipeline`/`SegmentPipeline`/`DepthPipeline`)
already makes, so this also fixes an inconsistency, not just a warning.

**Test fixture**: `crates/sdk/tests/text_pipeline.rs` gained two tests - one
confirming a nonexistent-and-unparseable string now surfaces as
`Error::ModelNotFound` (naming BOTH reasons: not a local file, not a valid
hub reference - the prior test asserted `Error::Backend` here, which was
correct for the old "always a local path" contract but is now the less
precise answer), and one confirming a relative, hub-shaped-but-missing
reference reaches the resolver and comes back `Error::Missing` - using the
SAME `Store::local`-satisfying-but-role-incomplete manifest trick
`tests/depth_pipeline.rs`'s own `mark_locally_present` uses, so this stays
fully offline (no real network call to check a plausible-looking repo id
against the real hub). Reaching a full, real model CONSTRUCTION from a fake
GGUF hub fixture is deliberately NOT attempted - `Qwen35Spec`'s own test
suite stops at `resolve()` too, never a full model build, and reproducing a
valid minimal GGUF `Qwen::load_inference` would accept is unproven,
out-of-scope territory this milestone does not take on either.

**Not done, tracked for later** (unchanged from Phase 2.2's own list): CLI
migration (the six qwen-family CLI files); a `TextGenerationPipeline` that
ALSO dispatches across the other decoder families (GLM/LFM/GPT/qwen35/
qwen35moe) the way `ImagePipeline` dispatches across flux2/s3dit - each of
those needs its OWN `ArchSpec` first, the identical prerequisite-then-wire
shape this phase just executed for qwen3, repeatable per architecture.

Verified with `cargo test -p brain-qwen3 --lib spec::` (7 passed, all on the
first attempt), `cargo build -p brain --no-default-features --features
text` (clean, no warnings - confirms the `device::apply` fix), `bash
scripts/gates/check-sdk-features.sh` (OK, every surface including `text`
still compiles standalone), `cargo test -p brain --features text --test
text_pipeline` (4 passed: 2 existing + 2 new), `bash scripts/gates/
check-no-doc-citations.sh` (clean), `bash scripts/gates/check-doc-links.sh`
(169 pages resolve), `bash scripts/gates/check-scripts.sh` (PASS).
`cargo test -p brain-qwen3 --lib` (full crate suite) has 3 PRE-EXISTING
failures, unrelated to this change: `serve::tests::*`, a real hardware
constraint ("needs a single 2293760000-byte buffer but this device's
queried max_buffer_size is 2147483647 bytes") in code last touched by an
unrelated earlier commit (`ff9818e32`, "decide the KV binding limit from
the device, not a constant") - `spec.rs` is pure file-classification code
with no GPU/wgpu path at all, so there is no mechanism by which it could
cause a buffer-allocation failure elsewhere; confirmed via `git log` that
`serve.rs` was untouched by this session.

### Phase 2.3 - `EmbeddingPipeline` (done, scoped to CLIP text embedding)

Covers **CLIP's text towers only** (CLIP-L default, OpenCLIP-bigG via
`.builder(id).tower("openclip_bigg")`), resolved through the SAME
`loader::resolve_structured` call `ImagePipeline`/`ForecastPipeline` use,
against CLIP's real `ClipSpec` (`crates/clip/src/spec.rs`, the `"towers"`
role - a released SDXL-layout directory several image-generation
checkpoints already carry, since SDXL conditions on the same two towers).

**Named `vision`, not `embedding`, and this is a hard constraint, not a
style choice**: `scripts/gates/check-sdk-features.sh` requires every surface
name to be a `brain_arch::Domain` variant, and there is no `Embedding`
variant - CLIP's own `arch!` row is registered `Vision` (the same released
checkpoint also carries an image tower). The public TYPE is still named for
what it does (`EmbeddingPipeline`, not `VisionPipeline`) - only the Cargo
feature flag is named for the domain, same as how the `image` feature's
name and its types' names already happen to coincide by chance rather than
by rule. A future vision pipeline (detection, segmentation, depth) joins
this SAME `vision` feature rather than inventing another one.

**Deliberately excluded, not silently missing**: ArcFace (face embedding)
takes an image plus a detected-and-aligned face, not a string - a genuinely
different call shape (`embed_image`, composed with SCRFD detection), tracked
as a separate future extension rather than forced into `EmbeddingPipeline`
by pretending the inputs are the same. T5-XXL/umT5-XXL are conditioning
encoders another model's pipeline (FLUX.1) consumes internally, not a
caller-facing embedding endpoint of their own.

The domain object (`Embedding`) is a thin, deliberate wrapper over
`Vec<f32>` - unlike `Forecast`'s rich, pre-existing structure, an embedding
vector genuinely has nothing more to say than its own values and dimension,
so the wrapper exists only to keep raw tensors off the public return type
(rule 12) rather than to carry real domain richness the way `Forecast` does.

Tested against a real local, synthetic, fully-offline fixture reproducing
`crates/clip/src/spec.rs`'s own (private) SDXL-tower-root classification
schema (an `model_index.json` naming `StableDiffusionXLPipeline` plus four
component directories - empty, since only their EXISTENCE decides
classification): resolution proven all the way to `clip::caps::Session::load`
being reached with the right directory, then a clean `Error::Backend` on the
fixture's empty tokenizer directories - `data::clip_bpe::ClipBpe` has no
synthetic/in-memory constructor either, the same class of gap
`TextGenerationPipeline`'s own tests document for `QwenBpe`.

**Not done, tracked for later**: CLI migration (there is no dedicated
`clip`/embedding CLI file to migrate at all - only `resident_clip.rs` - so
this is a smaller, more contained version of the same gap the earlier
pipelines have); ArcFace's `embed_image` extension (needs SCRFD detection
composed in first); ClipSpec's optional `"eva"` role (the EVA-CLIP image
tower) is not exposed by this pipeline at all, since it is out of scope for
TEXT embedding.

### Phase 2.4 - `TranscribePipeline` (done, scoped to qwen3-asr) - and a real bug found while building it

Covers **qwen3-asr only** (offline, fixed audio window), resolved through
`loader::resolve_structured` against `qwen3asr::spec::Qwen3AsrSpec`'s one
`"weights"` role. nemotronasr (the *streaming* ASR model, true batched
forward across concurrent windows) needs a genuinely different call shape
(feed chunks, get segments back incrementally) and is a tracked future
extension, not folded in - same treatment ArcFace got in Phase 2.3.

`qwen3asr::caps::QwenAsrProvider::load` + `.transcribe(wav)` was already the
cleanest construction+call shape of any pipeline covered so far - a single
directory for both weights and tokenizer, and `.transcribe` already returns
`(String, Vec<u32>)` with no `Invocation`/`Outcome` ceremony needed. The one
real design decision was surfacing `qwen3asr::caps::window_truncation`
explicitly as `Transcript::truncated` - audio past the fixed decode window
is DROPPED, and audit F18 (named in that function's own doc) is exactly the
"a caller trusted a silently-partial transcript" bug class this field
exists to prevent; silently truncating in this pipeline would have
reintroduced the same bug class one layer up.

**A real, independent bug was found and fixed while testing this pipeline,
in `crates/qwen3asr` itself, not in `crates/sdk`**: `import::
map_audio_encoder` PANICKED the whole process on an incomplete checkpoint
(a missing tensor), reachable from both of `Qwen3Asr`'s public loaders
(`from_hf`, `from_hf_windowed`) - exactly the "never panic on caller-reachable
input" boundary this whole sweep exists to enforce, just one layer below the
SDK this time. Fixed by collecting missing tensor names into a side vec
instead of panicking inside the lookup closure, returning a named `Err`
listing how many tensors are missing and the first one, once construction
finishes - a minimal, surgical change (the closure's failure mode, not its
dozens of call sites) rather than a rewrite. `Qwen3Asr::from_tensors` (the
one function that was NOT already `Result`-shaped) gained one, its single
caller (`from_hf`) already returned `Result` so the change was purely
additive there. Two new regression tests pin this directly in
`crates/qwen3asr` itself (an empty tensor map errors by name; a complete
minimal one still builds) - not only proven indirectly through the SDK's own
fixture, which is what surfaced it in the first place (a synthetic fixture
with a config classifying correctly but no real tensors is exactly the
"reachable incomplete checkpoint" case this bug needed to hit).

Tested (SDK side) against a real local, synthetic, fully-offline fixture
reproducing `qwen3asr::spec::Qwen3AsrSpec`'s own classification schema (a
`config.json` declaring `architectures: ["Qwen3ASRForConditionalGeneration"]`
plus a loose safetensors shard, matching the same `HfDir`-collapse
requirement `forecast_pipeline.rs`'s kronos fixture already documents):
resolution proven through to `QwenAsrProvider::load` being reached with the
right directory, then - now correctly - a clean `Error::Backend`, never a
panic.

**Not done, tracked for later**: nemotronasr's streaming call shape (see
above); CLI migration (there is no dedicated ASR CLI file at all to migrate,
only `resident_asr.rs` - the same smaller, contained gap `EmbeddingPipeline`
already noted for CLIP).

### Phase 2.5 - `UpscalePipeline` (done, scoped to RRDBNet) - a real bug found while building it, and a new public `Image::open`/`Image::from_rgb8`

Covers **RRDBNet (Real-ESRGAN's generator) only**, resolved through
`loader::resolve_structured` against `rrdbnet::spec::RrdbnetSpec`'s one
`"weights"` role - the first pipeline in this family scoped to the
"restoration/upscaling" domain bucket, and a deliberate SEPARATE public type
from `ImagePipeline` rather than a third backend inside it: generation and
upscaling are different capabilities (rule 2), even though RRDBNet's
`brain_arch` row is registered the same `Domain::Image` flux2/s3dit are, so
`UpscalePipeline` lives under the SAME `image` Cargo feature as
`ImagePipeline` rather than getting its own (mirroring how `EmbeddingPipeline`
joined the `vision` feature in Phase 2.3 rather than inventing a new one).

**A real, independent bug was found and fixed while testing this pipeline, in
`crates/rrdbnet` itself, not in `crates/sdk`** - the third time in a row this
campaign's own fixture discipline has done this (qwen3asr's import panic in
Phase 2.4; before that, the flux2 precision bug M10 fixed). `rrdbnet::spec::
RrdbnetSpec::classify` checked `rec.kind != ArtifactKind::Opaque`, but
`brain_modelstore::inventory::scan` classifies a real `.pt`/`.pth` archive as
`ArtifactKind::Torch` (confirmed correct in `cosyvoice`/`wan`'s own specs,
which both check `Torch` for the same file kind) - so `RrdbnetSpec` had NEVER
successfully classified a real, scanner-produced checkpoint; every one of its
own pre-existing unit tests passed anyway because they all hand-build an
`ArtifactRecord` with an explicitly chosen `kind: ArtifactKind::Opaque`,
never going through the real scanner. This means `brain rrdbnet upscale`/
`brain do rrdbnet upscale` (the resolver-migrated CLI path) could never
actually resolve a real installed RealESRGAN checkpoint either - not a
theoretical gap, a live one. Fixed by changing the one `!=` comparison to
check `Torch`, updating that spec's own tests to match reality, and adding a
new regression test that classifies a fixture through the REAL
`brain_modelstore::inventory::scan` rather than a hand-built record, so this
exact "the check quietly diverged from the real scanner's own output" class
of bug cannot recur silently a second time (`crates/rrdbnet/src/spec.rs`'s
`classify_recognizes_a_real_pth_file_scanned_by_the_real_inventory_scanner`).

**`crates/sdk::Image` gained two new public methods this milestone forced
into existence**: before `UpscalePipeline`, every `Image` was a pipeline
OUTPUT (`ImagePipeline::generate` produces one; nothing ever took one as
input), so `Image::from_rgb8`/`from_hwc_unit` were `pub(crate)` and there was
no way at all - not even privately - to read one back off disk. A pipeline
whose task takes an `Image` as INPUT needs both: `Image::open(path)` (new,
public, delegates to the already-existing `imaging::load`) for a caller
reading a file, and `Image::from_rgb8` promoted from `pub(crate)` to `pub`
for a caller who already has decoded pixels in memory. `Image::to_hwc_unit`
(new, `pub(crate)`) is the float-HWC counterpart `UpscalePipeline::upscale`
itself needs to hand pixels to `rrdbnet::caps::Upscaler`. All four are
covered by round-trip unit tests in `crates/sdk/src/image.rs`.

**This is the pipeline family's FIRST real end-to-end
`from_pretrained -> task call -> inspect the domain result -> save` test that
does not have to stop at a clean construction error** (`.agents/rules/
sdk-design.md` rule 14's aspiration, unmet by every prior pipeline in this
crate - see `tests/image_pipeline.rs`'s own doc for why flux2/s3dit cannot).
RRDBNet's shape is DERIVED from the checkpoint rather than hardcoded to one
multi-billion-parameter release, and `RrdbConfig::param_list()` already names
every tensor a given config's forward pass reads with an exact-match
contract (`crates/rrdbnet/src/import.rs::validate`) - so a fixture at TINY
dimensions (`num_feat=8`, `num_grow_ch=4`, 2 blocks, `x2`), built from that
same `param_list()` rather than a hand-picked subset, is a genuinely
complete, genuinely buildable checkpoint. All zeros, so the output is not a
meaningful image, but every kernel dispatch, buffer size and layout
permutation on the real path runs for real, including the tiled code path
(`UpscaleOptions::tile`, a second real forward pass through different code
over the same checkpoint) - `crates/sdk/tests/upscale_pipeline.rs`.

**Not done, tracked for later, all within this same domain bucket**:

- **CodeFormer** (face restoration, `restore_face` action) - clean
  `image(+w fidelity dial) -> image` shape at a FIXED 512² geometry, but has
  NO `spec.rs`/`ArchSpec` at all today (confirmed: its `Cargo.toml` does not
  even depend on `brain-modelstore`), so there is nothing for
  `loader::resolve_structured` to resolve against - real upstream work,
  the same class of prerequisite Phase 2.2 named for `qwen3::spec::
  Qwen3Spec`. `RrdbnetSpec` is the template (derive real tensor shapes,
  `Confidence::Derived`), but CodeFormer's config (`CodeFormerConfig::
  codeformer()`) is a single hardcoded variant with no `from_tensors` - a
  future spec would classify by tensor NAME presence, not shape-derive a
  variant the way RRDBNet's does.
- **SUPIR** (heavier restoration) - deferred for three independent reasons:
  no `spec.rs`, and by its own `brain_arch` row's comment, deliberately no
  `default_ref`/`weights_env` at all (the SUPIR license is non-commercial
  only); its call shape is genuinely heavier than "image in, image out"
  (model-determined output size, an optional text-caption input that can
  cross-dispatch to a SECOND model over an injected `capability::Registry`,
  a 9-parameter 50-step cancellable/streaming sampler); and its backbone (a
  frozen ~14GB SDXL checkpoint) puts real construction in the exact same
  "fixture infeasible, prove resolve->dispatch->clean-error" bucket
  `ImagePipeline`'s flux2/s3dit backends are already in - covering it would
  regress this pipeline family's first genuine end-to-end success back to
  that weaker bar immediately.
- **VQGAN** - deferred because its real call shape (`encode`/`decode`, a
  discrete-code `Media::Bytes` intermediate travelling BETWEEN two separate
  actions, deliberately no single "reconstruct" action per that crate's own
  module doc) does not fit a unified `restore(image) -> Image` signature at
  all - the same "shape mismatch within one domain bucket" reasoning Phase
  2.3 already used to defer ArcFace's `embed_image`. Also has no `spec.rs`.
- CLI migration: `brain rrdbnet upscale`/`brain do rrdbnet upscale` still
  builds its own `Session` inline (`resident_upscale.rs`) rather than
  calling `UpscalePipeline` - same class of gap every other pipeline in this
  crate still has.

### Phase 2.6 - `codeformer::spec::CodeFormerSpec` (done) - the Phase 2.5 prerequisite, now migrated onto the resolver everywhere it fits

Writes exactly the prerequisite Phase 2.5's "Not done" list named: a real
`ArchSpec` for CodeFormer's single `"weights"` role, `crates/codeformer/src/
spec.rs`. The `RestorePipeline` SDK type it unlocks is Phase 2.7, below -
everything ELSE this spec unlocks was migrated in the same change as the
spec itself, not left half-wired:

- `crates/catalog/src/lib.rs`'s `ModelEntry` for `codeformer` now resolves
  `weights` through the model store instead of `from_env!("BRAIN_CODEFORMER_
  WEIGHTS", ...)` - mirroring `rrdbnet`'s own entry, but simpler:
  `codeformer::caps::RestoreProvider::new` builds no GPU and imports no
  checkpoint at construction (both happen lazily on the first `restore_face`
  call, per `caps.rs`'s own doc), so the provider closure is just a resolved
  path plus an existence check, no eager `Gpu::new`/`caps::load` the way
  `rrdbnet`'s entry needs.
- `stage_registry()`'s `imgpipe::RESTORE_MODEL` branch, which stood in an
  `empty_assembly()` placeholder specifically because "codeformer has not
  migrated yet" (that comment, now deleted), now calls
  `resolved_stage_assembly("codeformer", &codeformer::spec::CodeFormerSpec)`
  - the same real-assembly path `SEGMENT_MODEL`/`UPSCALE_MODEL` already use,
  so `imgpipe`'s `restore` stage picks up a real, explicitly-opted-into
  models directory instead of never resolving anything.
- `crates/cli/src/resolve.rs`'s `RESOLVER_MIGRATED_ARCHS` and `crates/cli/
  src/resolver_cli.rs`'s `with_arch_spec` both gained a `"codeformer"` row,
  the same two-line wiring `rrdbnet` needed - `brain codeformer <verb>` now
  resolves its own `--weights` override flag through the store rather than
  requiring `BRAIN_CODEFORMER_WEIGHTS` to already be set.
- `crates/arch/src/lib.rs`'s `codeformer` row gained the same explanatory
  comment `rrdbnet`'s already carries: `weights_env` was already empty (the
  macro default - codeformer never had a declared env-var role to begin
  with), so this is documentation, not a behavior change.

**Classification method, and why it differs from `RrdbnetSpec`**:
`CodeFormerConfig` is ONE fixed preset (`CodeFormerConfig::codeformer()`,
`inference_codeformer.py`'s own hardcoded constructor call), not a family of
variants differing in width/depth/scale the way RRDBNet's `x4plus`/
`x4plus_anime_6B`/`x2plus` do - so there is no `from_tensors` to derive a
config FROM, and the roadmap's own note above ("a future spec would classify
by tensor NAME presence, not shape-derive a variant") called this correctly
in advance. `classify` instead checks five tensors ONLY the `CodeFormer`
class itself declares (never the `VQAutoEncoder` it subclasses) against the
exact shape the fixed preset implies: `position_emb`, `feat_emb.weight`,
`idx_pred_layer.1.weight`, one full transformer layer's fused attention
projection, and one controllable-feature-transformation tap's scale tower.
`crate::import::load` is what fully validates all 515 tensors at real load
time; classification only needs enough to be confident, the same division of
labor `RrdbnetSpec`/`CosyVoiceSpec` already draw. `Confidence::Derived`, not
`Declared` - a raw `torch.save` state dict has no header/config field that
names an architecture, matching `RrdbnetSpec`'s own reasoning for the same
file kind (`CosyVoiceSpec` chose `Declared` for an analogous tensor-name
check; `RrdbnetSpec`'s reasoning was the more recent and more literally
correct reading of the `Confidence` enum's own doc, so this spec follows
that one).

**The real headline case a fixture must get right, and does**: CodeFormer's
515 checkpoint tensors are a strict superset of the `VQAutoEncoder` it
subclasses' 329 (`crates/vqgan`'s own released `vqgan_code1024.pth`), so a
bare VQGAN checkpoint - real tensor names, just none of the five
CodeFormer-only ones - must never be mistaken for `codeformer.pth`.
`classify_rejects_a_bare_vqgan_checkpoint` pins exactly this. Also tested,
mirroring `RrdbnetSpec`'s own discipline: an unreadable `.pth` classifies as
nothing; resolution end-to-end with `variant: None` (no variant dimension
exists here, unlike RRDBNet's `x{scale}-{blocks}b`); two equally-real
candidates report `Ambiguous`, never a silent pick; and a regression pin
through the REAL scanner (`brain_modelstore::inventory::scan`), not a
hand-built `ArtifactRecord` with an explicitly chosen `kind` - the exact
class of bug `RrdbnetSpec`'s own equivalent test caught after the fact, only
this time written correctly from day one rather than needing a fix.

**Not done, tracked for later**: `crates/cli/src/resident_restore.rs`'s
served/D-Bus path still reads `BRAIN_CODEFORMER_WEIGHTS` directly rather
than the resolver, the same tracked (not silent) gap `resident_upscale.rs`
already has for `rrdbnet`.

Verified with `cargo test -p brain-codeformer --lib` (26 passed, 6 new),
`cargo test -p brain-catalog --lib` (9 passed, including
`every_listed_model_is_constructible_by_name` and `imgpipe_stage_ids_match_
the_catalog`), `cargo test -p brain-cli` (full suite), and a full
`cargo build -p brain-arch -p brain-codeformer -p brain-catalog -p brain-cli`.

### Phase 2.7 - `RestorePipeline` (CodeFormer) - the pipeline family's first REAL end-to-end forward pass, and two real bugs found AND fixed reaching it

Covers **CodeFormer only**, resolved through `loader::resolve_structured`
against the just-added `codeformer::spec::CodeFormerSpec`'s one `"weights"`
role - a deliberate SEPARATE public type from `ImagePipeline`/
`UpscalePipeline` rather than a third backend inside either: generation,
upscaling and restoration are different capabilities (rule 2), so
`RestorePipeline` joins the SAME `image` Cargo feature the other two use
(CodeFormer's `brain_arch` row is registered the same `Domain::Image`).

**`codeformer::caps::Session` gained a typed core, the same
decode-then-typed-core split `rrdbnet::caps::Upscaler` already established**:
before this pipeline, `Session::restore_face`'s resize/normalize/`model.
restore`/denormalize sequence was inlined directly inside its
`Invocation`-decoding wrapper, with no way to call it over raw pixels.
Factored out as `Session::restore(hwc, w, h, fidelity) -> Result<
RestoreOutput, String>`, with `restore_face` now a thin wrapper calling it -
one implementation, two callers (the capability action, and `crate::sdk`'s
`RestorePipeline`), matching how `run_upscale` wraps `Upscaler::upscale`
rather than duplicating its body. `Session::config()` (new) gives the SDK's
`Debug` impl the same `dim_embd`/`n_layers`/`img_size` accessor `rrdbnet::
caps::Session::config` already has.

**Two real, independent, pre-existing bugs this pipeline's own fixture
discipline found IN shared forward-pass/backend infrastructure this crate
does not own - and both got fixed in this same milestone**, the fourth and
fifth time in a row this campaign's fixture discipline has found a real bug
by being the first thing to actually dispatch a code path (qwen3asr's
import panic in Phase 2.4; the flux2 precision bug M10; `RrdbnetSpec`'s
wrong `ArtifactKind` in Phase 2.5):

- **A duplicate kernel registration** (`crates/codeformer/src/model.rs`):
  `matmul_reg3` already lives in `vae::blocks::KERNELS`, exported as
  `vae::blocks::MATMUL_REG3_SLOT` specifically so a caller layering its own
  kernels on top reuses it rather than registering a second copy - the
  exact lesson that constant's own doc records from `crates/sdxlunet`'s
  history. `codeformer::model::kernel_set()` registered a SECOND
  `("matmul_reg3", ...)` at a new slot anyway. Harmless on `backend-wgpu`
  (both indices compile to a valid pipeline, so nothing looked wrong there),
  but `wgsl-cpu`'s JIT cannot compile `matmul_reg3` AT ALL - it is a
  work-group/shared-memory kernel, CPU-native-only by design -
  `backend_cpu`'s AVX2 fast path intercepts it by matching ONE cached
  index (`FastIdx::matmul_reg3`, resolved once at `CpuBackend::new`), so
  CodeFormer's own dispatch through the SECOND, uncaught index fell
  through to the JIT and panicked ("matmul_reg3 was not JIT-compiled").
  **Fixed** by resolving `K_MATMUL_REG3` to `vae::blocks::MATMUL_REG3_SLOT`
  directly instead of appending a new entry (shrinking `KERNELS` by one
  slot) - a regression test
  (`model::tests::matmul_reg3_reuses_the_shared_slot_not_a_second_registration`)
  pins both that the slot sits inside the shared prefix and that the name
  appears exactly once.
- **Two unreclaimed device-memory scopes** (`crates/codeformer/src/
  model.rs::CodeFormer::build`): the encoder+transformer half and the
  generator+CFT half each build their own `vae::blocks::Builder`, whose
  activation pool (`Builder::free`) reuses a same-length buffer WITHIN one
  builder's own recording, but a length that never recurs sits pooled
  until that `Builder` itself drops at the end of its block - with no poll
  anywhere in between, and CONSTRUCTION never allocates again after both
  builders finish, so nothing catches the accumulation until `.restore()`'s
  own first post-construction buffer (a readback staging buffer) is what
  actually tripped `backend-wgpu`'s "2.37 GiB of device buffers were
  dropped without an intervening `poll_wait()`" ceiling - the exact failure
  mode `gpu_core::transient`'s own module doc describes, one `Builder`
  scope at a time rather than one loop iteration at a time. **Fixed** by
  wrapping each of the two builder scopes in `gpu_core::reclaiming` - the
  four pinned encoder taps (`enc_feat`) the generator half still needs are
  returned OUT of the first closure, so they survive that closure's own
  poll, exactly as `reclaiming`'s own doc describes for a value the next
  iteration still needs.

Both are plausibly unexercised anywhere else in this workspace before this
milestone: `crates/codeformer/tests/parity.rs`'s own real-forward-pass
tests all gate on `BRAIN_CODEFORMER_WEIGHTS` (a license-gated real
checkpoint), silently skipped in any environment - this one included - that
has not fetched one. This milestone's all-zero-but-COMPLETE synthetic
fixture (built from `CodeFormerConfig::tensor_manifest()`, the same
discipline `write_complete_rrdb_checkpoint` used in Phase 2.5, just at
CodeFormer's one real fixed size rather than a shrinkable one) is the first
thing in this workspace to force CodeFormer's real graph to actually
dispatch with no real weights required anywhere - and once both bugs above
were fixed, IT COMPLETES, on both backends available in this environment
(wgpu in ~78s, the CPU JIT in ~260s - confirmed independently during
investigation, then left as the wgpu default in the committed test since it
is both faster and this crate's ordinary default device).

**This is the pipeline family's FIRST real end-to-end
`from_pretrained -> task call -> inspect the domain result -> save` test at
the model's actual, real, full release geometry** - stronger than Phase
2.5's own `UpscalePipeline` achievement, which needed RRDBNet's shrinkable
config to reach a genuine forward pass at all: CodeFormer has no such
lever, so this proves the real 512x512 graph, not a toy-sized stand-in.
`crates/sdk/tests/restore_pipeline.rs`'s
`from_pretrained_restores_a_real_complete_fixture_end_to_end` covers
resolve -> classify -> import all 515 real tensors -> build the whole real
graph -> a genuine forward pass -> a real PNG on disk, all with no network
access and no real weights anywhere. `RestoreOptions::fidelity`
out-of-range is tested too (validated before any GPU work, so it stays
fast). Construction+restore together take ~70-90s against the real
515-tensor/512x512-graph fixture on this dev machine's small integrated
GPU - slower than every prior pipeline's fixture in this crate, because
CodeFormer has no "tiny" config to shrink to the way RRDBNet's does (Phase
2.5's own doc); accepted as the real cost of proving genuine,
complete-checkpoint, complete-forward-pass coverage rather than a lighter,
less honest fixture, or a construction-only one.

**Not done, tracked for later**: CLI migration (`resident_restore.rs` still
builds its own `Session` inline, same as Phase 2.6 already tracked). The
two forward-pass bugs above are NOT tracked as open - both are fixed, with
regression coverage.

Verified with `cargo test -p brain-codeformer --lib` (27 passed, including
the new `matmul_reg3` regression test, no regressions from the `caps.rs`
refactor or the `model.rs` kernel/reclaim changes), `cargo build -p brain
--features image` (clean), `cargo test -p brain --features image` (all
pipelines' suites green, including this one's 4 tests - one of them now a
genuine, complete `.restore()` forward pass rather than a construction-only
stand-in).
### Phase 3.1 - `DetectionPipeline` (done, scoped to YOLOv8) - the vision/detection bucket's first pipeline, and its first non-`Image` domain object

Covers **YOLOv8 only**, resolved through the SAME `loader::resolve_structured`
call every earlier pipeline uses, against a NEW `yolov8::spec::YoloSpec` -
YOLOv8 had no model-store `ArchSpec` at all before this milestone (its
weights were a REQUIRED `--weights`/`BRAIN_YOLOV8` CLI param, never resolved
through the store), the exact prerequisite gap `codeformer::spec::
CodeFormerSpec` closed for the restoration bucket in Phase 2.6.

**Classification, and why it needed care despite the config being
DERIVED (unlike CodeFormer's fixed preset)**: `crates/yolov8`'s checkpoint
format is brain-native, not an upstream `.pt`/`.pth` - a plain
`.safetensors` file carrying its own config under the `brain.config`
metadata key. `YoloConfig::from_json` reads that config, but - unlike
`RrdbConfig::from_tensors`, which returns `Err` naming what's wrong -
`from_json` NEVER FAILS: every field defaults to `yolov8n`'s own value when
absent, so parse success alone would happily "classify" an EMPTY `{}` or a
different model's unrelated config as YOLOv8. The real signal
(`yolov8::spec::yolo_config_for`) derives a candidate config and then
verifies it against the file's REAL tensor names/shapes via
`YoloConfig::full_param_list()` (already existed, written to reproduce
`Yolo::new`'s own registration exactly, for parity-testing without a GPU -
reused here as-is, not re-derived) - `Confidence::Derived`, matching
`RrdbnetSpec`'s reasoning for a self-describing-but-unverified header.
`classify_rejects_a_safetensors_file_with_an_unrelated_config_and_no_real_tensors`
pins the headline case this care was for.

**`Detection`, the vision/detection bucket's first domain object that is
NOT `Image`**: `yolov8::Detection` is a bare `[f32; 6]`
(`[x1,y1,x2,y2,conf,class]`) at the crate-internal level - functional, but
exactly the kind of raw-tuple return rule 4 asks a public pipeline not to
expose. `crate::detect::Detection` wraps it as a named struct
(`x1`/`y1`/`x2`/`y2`/`confidence`/`class`); this is genuinely a NEW pipeline
shape in this crate, not a fourth backend of `ImagePipeline`/
`UpscalePipeline`/`RestorePipeline` - `image(+opts) -> boxes` does not fit
`image(+opts) -> Image`, confirming the domain inventory table's own note
that vision/detection needs a different domain object. `DetectionPipeline`
still joins the `vision` feature `EmbeddingPipeline` already occupies (both
resolve under `brain_arch::Domain::Vision`, this workspace's rule that a
surface name is a `Domain` variant, not a second word for the same
modality) - a THIRD public type sharing one feature, the same way `image`
already holds three.

**A genuinely fast, genuinely complete end-to-end test, unlike
`RestorePipeline`'s**: `YoloConfig::tiny(nc)` is a real, intentionally small
preset (unlike CodeFormer's one fixed size), so
`crates/sdk/tests/detection_pipeline.rs` reaches a real `.detect()` forward
pass - resolve, classify, import, build, DFL-decode, NMS - in ~3 seconds for
all 4 tests combined, the same "shrinkable config" advantage `UpscalePipeline`
(Phase 2.5) had over `RestorePipeline` (Phase 2.7). All-zero weights make
every anchor's class score `sigmoid(0) = 0.5`, so `detect()` returns
plenty of (meaningless but real) candidate boxes rather than none -
`detect_with_a_confidence_above_any_real_score_returns_nothing` uses a
confidence threshold above any possible sigmoid output to prove the option
is genuinely threaded through and the whole path can also return empty
cleanly.

**Not done, tracked for later**: CLI migration (`brain do yolov8 detect`/
`crates/cli`'s yolo path still takes a raw `--weights`/`BRAIN_YOLOV8`
path, not the resolver - the same class of gap every other pipeline in this
crate still has, now also true for the newly-added `YoloSpec` itself);
`SegmentPipeline` (SAM2 - already has a real `spec.rs`, so it is a smaller
lift than YOLOv8 was, but masks are a different domain object again, not
boxes, so it is not a trivial copy of this pipeline either); `Yolo::load`
itself is not `Result`-shaped (a malformed checkpoint that somehow still
classifies would panic inside `ParamStore::new`, not return a typed
`Error::Backend`) - not touched here, matching every other pipeline's
`Result`-shaped SDK wrapper over a pre-existing, non-`Result` model
constructor.

Verified with `cargo test -p brain-yolov8 --lib` (46 passed, 6 new, no
regressions), `cargo build -p brain --features vision` (clean), `cargo test
-p brain --features vision` (all pipelines' suites green, including this
one's 4 tests), and a full `cargo build --workspace --exclude brain-vulkan`.

### Phase 3.2 - `SegmentPipeline` (SAM 2.1) - a real, independent `Sam2Spec` bug found AND fixed, and a genuine end-to-end forward pass

Covers **SAM 2.1 only**, resolved through the SAME `loader::resolve_structured`
call every earlier pipeline uses, against the ALREADY-EXISTING `sam2::spec::
Sam2Spec` - unlike YOLOv8 (Phase 3.1), SAM2 already had a real `ArchSpec`
before this milestone, so the work here is the pipeline plus a bug this
milestone's own fixture discipline found in that pre-existing spec.

**A real, independent bug found AND fixed while testing this pipeline, in
`crates/sam2/src/spec.rs` itself - the SAME mistake `rrdbnet::spec::
RrdbnetSpec` had before Phase 2.5, found a second time in a DIFFERENT spec
this session did not otherwise touch**: `Sam2Spec::classify` checked
`rec.kind == ArtifactKind::Opaque` for a `.pt`/`.pth` candidate, but
`brain_modelstore::inventory::scan` classifies a real, readable `.pt`/`.pth`
archive as `ArtifactKind::Torch` - so `Sam2Spec` had NEVER successfully
classified a real, scanner-produced checkpoint; every one of its own
pre-existing unit tests passed anyway because they all hand-build an
`ArtifactRecord` with an explicitly chosen `kind: ArtifactKind::Opaque`,
never going through the real scanner - the identical structural gap
`RrdbnetSpec`'s own tests had. This means `brain sam2 track`/`brain do sam2
segment` (the resolver-migrated CLI paths `crates/cli/src/sam2_cli.rs`
already routes through `Sam2Spec`) could never actually resolve a real
installed SAM2 checkpoint either - not a theoretical gap, a live one, exactly
like `RrdbnetSpec`'s was. Fixed by changing the two `ArtifactKind::Opaque`
comparisons to `Torch`, updating the five existing hand-built test fixtures
to match reality, and adding a new regression test that classifies through
the REAL `brain_modelstore::inventory::scan`
(`classify_recognizes_a_real_pt_file_scanned_by_the_real_inventory_scanner`),
the same discipline that already caught `RrdbnetSpec`'s version of this bug
and is now proven to generalize to a second, independently-written spec.

**`sam2::caps::Session` gained a typed core, the same decode-then-typed-core
split `codeformer::caps::Session::restore`/`rrdbnet::caps::Upscaler` already
established - but with a real subtlety the earlier two didn't have**: the
"encode once, prompt many" cache (`Session::ensure_encoded`) skips the wire
`Blob` decode entirely on a cache hit, a real, documented cost saving. A
naive refactor that routed `segment(inv)` through a typed `segment_typed`
taking already-decoded pixels would have paid that decode unconditionally
even on a hit. Instead, only the "resize+normalize+trunk-encode" dispatch
sequence was factored into one shared `encode_pixels`, called from TWO
different cache-key strategies (`ensure_encoded`, unchanged, still hashing
the wire blob's bytes before ever decoding; the new `ensure_encoded_pixels`,
hashing the pixels a typed caller already has in hand) - and the actual
prompt/decode/mask-emit math was factored into a separate `run_prompt`,
assuming an image is already encoded, shared verbatim by both `segment` and
the new `segment_typed`. One real implementation of the segmentation math,
two independently-optimal encode paths, not a false unification that would
have quietly regressed the documented caching behavior.

**`Mask`, the vision/detection bucket's second non-`Image`, non-`Detection`
domain object**: a per-pixel sigmoid-probability grid at source-image
resolution plus SAM 2.1's own IoU confidence estimate and pixel-area count -
genuinely a third distinct pipeline shape in this crate (`image(+opts) ->
Image` for generation/upscaling/restoration, `image(+opts) -> Vec<Detection>`
for detection, and now `image + prompt(+opts) -> Mask` for segmentation).
`brain::Prompt` is a NEW typed builder (`.point(x, y, foreground)`/
`.bbox(x1, y1, x2, y2)`) over `sam2::caps::parse_prompt`'s own
box-before-points ordering convention, replacing that function's
`"x1,y1,x2,y2"`/`"x,y;x,y"` string parsing for the in-process caller -
`SegmentPipeline` is also this crate's first pipeline whose SIMPLE call
takes a prompt argument at all, since SAM 2.1 has no "segment everything"
default the way every other pipeline's options are all optional.

**A genuine end-to-end test, at SAM 2.1's real fixed geometry, like
`RestorePipeline`'s and unlike RRDBNet/YOLOv8's**: `sam2::spec::Sam2Spec`
only accepts a trunk width of exactly 96 (tiny) or 144 (large) - there is no
synthetic in-between size the real resolver would classify at all, so
`crates/sdk/tests/segment_pipeline.rs` builds a COMPLETE `hiera_tiny`
checkpoint (via `sam2::import::manifest_for`, the exact tensor list
`sam2::caps::load` itself validates against) and reaches a real
`.segment()` forward pass through the whole graph - trunk, FPN neck, prompt
encoder, mask decoder - at the model's real 1024x1024 frame. Unlike
`RestorePipeline`'s own CodeFormer fixture, this one hit NO forward-pass
infrastructure bugs on the wgpu backend (~150-170s total for the file's 4
tests) - the `backend-wgpu` buffer-reclaim ceiling Phase 2.7 found and
fixed in `codeformer::model::CodeFormer::build` does not reproduce here,
plausibly because SAM2's 12-block trunk (`stages: [1,2,7,2]`) is a much
shorter unpolled-scope walk than CodeFormer's ~59-block one.

**Not done, tracked for later**: CLI migration is NOT a new gap here -
`sam2_cli.rs`/`resident_sam2.rs` already route through `Sam2Spec` (they were
just silently broken by the bug above, now fixed); the video/tracking path
(`sam2::video`, `crate::model::Sam2::track`-style multi-frame memory bank)
is out of scope for this single-image pipeline, the same "genuinely
different call shape" reasoning that deferred nemotronasr's streaming ASR
in Phase 2.4.

Verified with `cargo test -p brain-sam2 --lib` (29 passed, 1 new, no
regressions from the `caps.rs`/`spec.rs` changes), `cargo build -p brain
--features vision` (clean), `cargo test -p brain --features vision --test
segment_pipeline` (4 passed), and a full `cargo build --workspace --exclude
brain-vulkan`.

Findings 8, 11, 14, 18-20, 23-24 are real but not yet milestoned - pick them
up opportunistically when touching the same file for another reason, or spin
them into their own milestone if they start blocking something.

### Phase 3.3 - `DepthPipeline` (ZipDepth) - completes the vision/detection bucket, a real `cfg_for_checkpoint` shape bug fixed, and a real `vision`/`image` feature-split bug found by finally exercising the compile gate this milestone was built to pass

Covers **ZipDepth monocular depth only**, over a NEW `zipdepth::spec::
ZipdepthSpec` (no `ArchSpec` existed for this arch before this milestone,
same starting point as YOLOv8/Phase 3.1, unlike SAM2/Phase 3.2's
already-existing spec).

**A real, derived-not-guessed config reader, added alongside the spec and
reused by it**: `zipdepth::config::ZipConfig::from_tensors` reads a
checkpoint's own tensor shapes (encoder width from `stem_quarter`'s output
channels, per-stage depth by counting `stage<N>.<i>.branch_3x3.0.weight`
keys, decoder width from `proj4`'s output channels, which upsampler from
`mask_pred` vs `where_conv`) the same `rrdbnet::config::RrdbConfig::
from_tensors` discipline this crate had not yet adopted. **A real bug this
replaced**: `zipdepth::import::cfg_for_checkpoint` used to ALWAYS return
`ZipConfig::base()`, toggling only `upsample_unfold` - so any checkpoint
whose encoder width was not exactly the `base` preset (a `small`/`large`/
`giant`-trained model, or any future release) would silently derive the
WRONG shape and fail downstream as a confusing tensor-shape mismatch rather
than a clean "wrong variant" error, or - worse, if shapes coincidentally
matched a smaller subset - load with silently wrong weights. Not a
theoretical gap: this is exactly the case `ZipdepthSpec::classify` needed
to get right to classify anything other than the one released preset at
all. `cfg_for_checkpoint` now calls `from_tensors` directly; every existing
caller (`depth_cli.rs`, `resident_depth.rs`) already wraps it in
`.unwrap_or_else(|_| ZipConfig::base())`, so this is a strict improvement
with no call-site changes needed.

**`ArtifactKind::Torch` checked correctly from the start** - `ZipdepthSpec`
is the third spec written after `RrdbnetSpec` (Phase 2.5) and `Sam2Spec`
(fixed this session, Phase 3.2) both hit the `Opaque`-vs-`Torch` mistake;
written correctly here first try, and pinned with the same
`classify_recognizes_a_real_pt_file_scanned_by_the_real_inventory_scanner`
regression test the other two needed added after the fact.

**`zipdepth::caps::Session`/`load`, a NEW typed core alongside the existing
`DepthProvider`/`Hot`, not a replacement of it**: `DepthProvider`'s `Hot`
residency is keyed by weights PATH and re-uploads a transient `ParamStore`
per call (so one long-lived D-Bus/CLI process can serve many different
checkpoints); `Session` is a bound, single-checkpoint handle a builder
resolves once (the shape `DepthPipeline` needs). Both call the SAME shared
core, `predict_normalized(gpu, cfg, init, hwc, w, h) -> DepthOutput`
(build `ParamStore`+`Predictor`, forward, min-max normalize) - factored out
of `InferAction::run`'s own inline body rather than duplicated, the
"CLI/D-Bus and SDK call the same code" rule (10) applied to a provider whose
residency shape does not fit the `Session`-owns-a-built-model pattern
`codeformer`/`sam2`/`rrdbnet` use.

**`DepthMap`, the vision/detection bucket's third non-`Image` domain
object**: a per-pixel min-max-normalized-to-`[0,1]` inverse-depth grid at
source-image resolution, plus the raw `min`/`max` bounds the map was scaled
from (so the relative distances stay recoverable) - the same "per-pixel
float grid plus metadata" shape `Mask` established, not a fourth `Image`
backend. `DepthOptions::input(side)` is the one knob, mirroring the
`infer` action's own `input` param.

**A real, independent gap this milestone's OWN verification step found and
fixed, in `crates/sdk/Cargo.toml` itself**: `cargo build -p brain --features
vision` (the exact command every prior phase's own verification section
used) never actually tests `vision` standalone, because `default = ["full"]`
silently backfills `image` underneath it - only `--no-default-features
--features vision`, the command `scripts/gates/check-sdk-features.sh`'s own
per-surface compile sweep runs, exercises a surface alone. Running that
command surfaced that `vision` (and this milestone's own `depth.rs`) could
not compile standalone at all: `detect.rs`/`segment.rs` (Phase 3.1/3.2) both
use `crate::Image`, gated behind the `image` feature, which `vision` never
selected - a real, live gap in two already-shipped pipelines, not something
this milestone introduced. Fixed by splitting `Image` out into its own
infrastructure feature, `imagetype` (alongside `device`/`resolve`, selected
BY a surface, never named by a consumer directly) - `image.rs` itself only
ever depended on `brain-imaging`, never on flux2/s3dit/rrdbnet/codeformer,
so this was a clean split, not a new dependency edge. Both `image` and
`vision` now select `imagetype`; `check-sdk-features.sh`'s own `TIERS`
exclusion set (the list that keeps `device`/`resolve` out of the
per-surface sweep and the Domain-vocabulary check) grew `imagetype` to
match. Full gate run clean after the fix (`bash scripts/gates/
check-sdk-features.sh` - `check/sdk-features: OK`, every surface including
`vision` and `image` compiling standalone).

**A genuine end-to-end test, FAST like `DetectionPipeline`'s and unlike
`RestorePipeline`'s/`SegmentPipeline`'s**: because `ZipConfig::from_tensors`
derives the whole net shape rather than requiring one fixed released preset,
`crates/sdk/tests/depth_pipeline.rs` builds a genuinely tiny (`dims: [8, 16,
32, 64]`, `depths: [1,1,1,1]`) complete checkpoint via `cfg.param_list()` and
reaches a real `.predict()` forward pass - encoder stages 1-4 (including
StripPoolingAttention/GlobalContextBlock, since the fixture inherits
`GlobalMode::Balanced` from `ZipConfig::base()`), SPPF, cross-scale fusion,
decoder, FastConvexUpsample - in ~1s. The model's input SIDE is a runtime
knob, not a stored weight, so it does not shrink with the tiny encoder
widths alone; `DepthOptions::input(32)` overrides it down to the smallest
valid (x32) size to keep the test fast, the one thing this fixture needed
that `DetectionPipeline`'s tiny-preset fixture did not.

**Not done, tracked for later**: catalog's `zipdepth` registration
(`crates/catalog/src/lib.rs`) stays `always!(zipdepth::caps::DepthProvider::
new())`, matching `yolov8::caps::YoloProvider`'s own `always!()` - both
providers take `weights` as a PER-CALL action param (not bound at
construction the way `codeformer`/`sam2`/`rrdbnet`'s providers are), so the
`assembly.role_path("weights")`-closure resolver-migration shape Phase 2.6
used does not fit either without a provider redesign; `brain zipdepth
--image`'s own CLI (`depth_cli.rs`) is a windowed SDL/V4L2 demo with many
specialized flags (camera, view modes, colormaps) and stays entirely
CLI-local, the same "genuinely different call shape" reasoning every prior
phase's CLI-migration gap note uses - not a new gap, an existing one this
milestone did not need to touch. `label_cli.rs` (VLM captioning workflow) is
excluded from this domain bucket entirely, not deferred: it is a workflow
over `captioner::Captioner`, not a single architecture's capability, the
same reason `forecast`'s finetune verb is not itself a "forecasting arch."

Verified with `cargo test -p brain-zipdepth --lib` (39 passed, no
regressions from the `config.rs`/`import.rs`/`caps.rs` changes),
`bash scripts/gates/check-sdk-features.sh` (`OK`, every surface compiling
standalone), `cargo test -p brain --features vision --test depth_pipeline`
(3 passed, ~1.3s total), `cargo test -p brain --features vision --tests`
(the full vision surface's test suites together, no regressions), and a full
`cargo build --workspace --exclude brain-vulkan`.

This closes out the vision/detection/segmentation domain bucket's
architecture coverage (detection, segmentation, depth all have SDK
pipelines; only `label` remains, correctly excluded as a workflow rather
than an architecture). Remaining fully uncovered SDK domain buckets: TTS/
music, video generation, 3D/world models - see the priority order note
above, last in line since `SplatPipeline`-shaped types are closer to
`Creature` (stateful, steppable) than to any `image(+opts) -> T` pipeline
built so far, and world models have no settled domain object yet.

### Phase 4.1 - `TtsPipeline` (Qwen3-TTS) - the TTS/music bucket's first pipeline, ONE type over three voice-selection call shapes

Covers **Qwen3-TTS only**, over the already-existing `qwen3tts::spec::
Qwen3TtsSpec` (a real `ArchSpec` since before this session, unlike YOLOv8/
ZipDepth's from-scratch specs) - picked over the other two served
architectures in this bucket (cosyvoice, minimaxmusic3) because it is the
only one with a genuinely shrinkable full-graph forward pass (`TalkerConfig::
tiny()`/`MtpConfig::tiny()` exist; cosyvoice's flow/diffusion stage and
minimaxmusic3's 5-component chain have no shrink lever at all - a future
milestone there pays a `RestorePipeline`/`SegmentPipeline`-class fixed cost,
not a design blocker, just a heavier one).

**ONE pipeline type, not three, and not a fork of `ImagePipeline`'s own
"one type, several backends" shape either**: `speak`/`clone_voice`/`design`
are three voice-selection call shapes over the SAME capability
(`qwen3tts::caps::manifest`'s own doc already frames `synth`/`clone`/
`design` as "each a thin wrapper over the same pipeline"), not three
different capabilities the way restoration and upscaling are - rule 2
("one pipeline type per capability") says keep them on one `TtsPipeline`
type, which is what this does.

**No new typed-core split needed - the crate already had one**: every
action in `qwen3tts::caps.rs` was ALREADY a thin wrapper over
`qwen3tts::pipeline::{synth,clone,design}`, and `qwen3tts::pipeline::
TtsPaths::from_assembly` already existed, built for exactly this seam. This
is the first phase in the campaign where the target crate needed zero
production-code changes at all - `crates/sdk/src/tts.rs` calls the same
functions `crates/cli/src/tts_cli.rs` and `qwen3tts::caps::{Synth,Clone,
Design}Action::run` already call.

**Unlike every other pipeline in this crate, `TtsPipelineBuilder::load`
builds no resident GPU state at all**: Qwen3-TTS's own design is stateless
per call (`qwen3tts::caps`'s own module doc - "the weights load per call...
there is nothing resident to cache"), so `TtsPipeline` is just a resolved,
existence-checked `TtsPaths` handle; every `.speak()`/`.clone_voice()`/
`.design()` call pays the same load cost the CLI/D-Bus path already pays.
The builder still checks `talker`/`mtp`/`codec` exist at `load()` time
(mirroring `qwen3tts::caps::common_run`'s own existence check), so a broken
checkpoint fails at construction rather than on the first call.

**`Audio`, the audio bucket's first domain object**: interleaved-nothing
(mono) `f32` PCM samples plus the sample rate they were generated at
(`SAMPLE_RATE = 24_000`, the same constant every existing caller
hardcodes - `qwen3tts::pipeline::{synth,clone,design}` return raw `Vec<f32>`
with no rate attached at all, so there was nothing to derive it from without
a deeper change to that crate, out of scope here). `.save(path)` writes a
WAV via `audio::wav::write`, the same codec every CLI/D-Bus caller in this
workspace already writes.

**A real ceiling this milestone's own test discipline confirmed applies
here too**: `qwen3tts::pipeline::synth`/`clone`/`design` all need a REAL
tokenizer (`data::qwen_tokenizer::QwenBpe`, via `prompt::load_tokenizer`) -
the same "cannot synthesize a meaningful BPE vocab from scratch" ceiling
`TranscribePipeline`'s/`TextGenerationPipeline`'s/`EmbeddingPipeline`'s own
tests already document and stop short of. `crates/sdk/tests/tts_pipeline.rs`
reuses `qwen3tts::spec::tests`' own `write_qwen3tts_checkpoint` fixture
shape (a converted-checkpoint directory with real `brain.manifest.json`
compound-manifest roles, fake tensor content) to prove resolution reaches
`TtsPipelineBuilder::load`'s own existence check and `.speak()` reaches
`qwen3tts::pipeline::synth` itself, which then fails CLEANLY
(`Error::Backend`, never a panic) on the fake tokenizer/checkpoint content -
the same "resolution proven, full forward pass not" scope those three
pipelines' own tests already accepted.

**A real fixture-building gap found and fixed while writing the "missing
role" test**: naively reusing every OTHER spec's `mark_locally_present`
helper (a single-role `{"weights": "weights.stub"}` manifest) initially hit
`Error::Download` instead of `Error::Missing`, because `Store::local_compound`
(`crates/modelstore/src/lib.rs:312`) checks that EVERY role path a manifest
DECLARES actually exists on disk before considering a repo "locally
present" - so a manifest naming a role `Qwen3TtsSpec` does not even look
for (`"weights"`, not `"weights_dir"`/`"ckpt"`) still needs that role's own
file to exist for `store.local()` to succeed at all, or the builder
attempts a network fetch instead of ever reaching the resolver. Fixed by
writing `weights.stub` (satisfying `local_compound`'s existence check) under
a manifest whose `family` is `"qwen3tts"` (so `classify_compound_manifest`
does not skip it) but whose ONE role is named `"weights"` (so neither of
`Qwen3TtsSpec`'s two real roles, `weights_dir`/`ckpt`, ever classifies) -
the same trick `RESOLVER_MIGRATED_ARCHS`' single-role specs already use,
adapted to a two-role compound manifest for the first time.

**Not done, tracked for later**: `cosyvoice`/`minimaxmusic3` (this bucket's
other two served architectures) have no SDK pipeline yet - `cosyvoice`
would need a genuinely different call shape anyway (its one `synth` action
always requires `ref_audio`+`ref_text`, no speaker-free mode, closer to
`clone_voice` alone than to `speak`). `mimi`/`ecapatdnn` (consumed as plain
files inside `qwen3tts`'s own `weights_dir`, not independently resolvable)
and `campplus`/`s3tokenizer` (have `spec.rs` but no `caps.rs`, cosyvoice-
internal only) remain real "no independently-servable `ArchSpec`" gaps, not
new ones this milestone introduced. No CLI migration gap here either -
`tts_cli.rs` was never on the resolver-migrated list to begin with (its own
`--weights-dir`/`--ckpt` flags predate `RESOLVER_MIGRATED_ARCHS`), so there
is nothing this phase silently left worse than it found.

Verified with `cargo build -p brain --features audio` (clean), `bash
scripts/gates/check-sdk-features.sh` (OK, `audio` compiles standalone -
confirmed clean on the FIRST attempt this time, unlike `vision`/`image` in
Phase 3.3), `cargo test -p brain --features audio --test tts_pipeline`
(3 passed), `cargo test -p brain --features audio --tests` (full audio
surface suite, no regressions), `bash scripts/gates/check-no-doc-citations.sh`
(clean), and a full `cargo build --workspace --exclude brain-vulkan`.

### Phase 4.2 - CosyVoice joins `TtsPipeline`, a real "Store::local never recognizes it" bug found and fixed, plus `samples/tts/clone` (a real Rust sample requested mid-sweep, not a Phase item)

Covers **CosyVoice 2 only** (`cosyvoice2` variant), over the already-existing
`cosyvoice::spec::CosyVoiceSpec` (a real `ArchSpec` since before this
session, same as Qwen3-TTS's own `Qwen3TtsSpec` was for Phase 4.1) - `variant`
defaults to `cosyvoice2` because `cosyvoice::caps::manifest`'s own doc
already says "only cosyvoice2 is servable today"; `cosyvoice3` is reachable
through `TtsOptions::variant("cosyvoice3")` for a caller who has that
checkpoint, unvalidated by this milestone's own test the way `cosyvoice2` is.
minimaxmusic3 (this bucket's third served architecture) stays deferred -
still no shrink lever for a full forward pass, the same reason Phase 4.1
picked qwen3tts first.

**`TtsPipeline` forks internally, not into a sibling type** - `resolve_arch`
tries `qwen3tts` first (unchanged order/tie-break reasoning from
`ForecastPipeline`'s own kronos-first precedent: it is the more complete of
the two), then `cosyvoice` only when qwen3tts did not resolve. Past
`TtsPipelineBuilder::load`, every method dispatches on an internal
`Backend` enum - the same shape `ImagePipeline` (flux2 vs s3dit) and
`ForecastPipeline` (kronos vs timesfm3) already established, rule 2's "one
pipeline type per capability."

**CosyVoice's one action does not fit `speak`/`design` at all - only
`clone_voice`, at REDUCED fidelity.** `cosyvoice::caps::manifest`'s own doc
frames `synth` as "zero-shot voice cloning: target text + a reference audio
clip and its transcript" - there is no speaker-free mode (unlike Qwen3-TTS's
`speak`) and no VoiceDesign/CustomVoice action at all. A CosyVoice-resolved
pipeline's `speak`/`design` return `Error::MissingArgument` - knowable
before any backend call, the SAME class of caller-programming error M6
introduced for `CreatureBuilder`'s required-field checks, not a new
variant. `clone_voice`'s own `ref_text: Option<&str>` stays optional in the
public signature (Qwen3-TTS genuinely supports x-vector-only cloning with
it unset) but a CosyVoice-resolved pipeline rejects `None` with the same
`Error::MissingArgument`, since CosyVoice has no x-vector-only mode either -
documented on the method, not silently coerced to an empty string the way
an earlier draft of this milestone briefly considered (that would have
handed CosyVoice's `generate` an empty transcript and produced a confusing
downstream failure instead of a clear one at the call boundary).

**`TtsOptions` grows two CosyVoice-only knobs** (`variant`, `n_timesteps`)
that Qwen3-TTS silently ignores - the same "the other backend's fields are
inert" shape already accepted elsewhere in this crate for split call
surfaces. `seed` is the one field both backends' own `GenOpts` already
carry, so it stays shared.

**A real, independent judgment call, not a bug**: `CosyVoicePaths::
from_assembly` still reads `BRAIN_S3TOKENIZER_V2`/`BRAIN_CAMPPLUS_DIR`
directly from the environment (that function's own doc: neither role is
resolver-migrated yet) - this milestone did not migrate them, since doing so
would mean giving `s3tokenizer`/`campplus` their own `ArchSpec`s, a
prerequisite-crate-first shape like Phase 2.6's `CodeFormerSpec` was for
Phase 2.7, not a small addition to an SDK pipeline. `TtsPipelineBuilder::
load`'s own existence check only verifies `llm.pt`/`flow.pt`/`hift.pt`
under their resolved directories (mirroring qwen3tts's own `talker`/`mtp`/
`codec` check) - `s3tokenizer`/`campplus` stay a call-time failure inside
`cosyvoice::pipeline::generate` itself, the same ceiling qwen3tts's own
`speaker` (clone-only) role already accepts.

**A real, independent bug found reaching this crate's first genuine
end-to-end resolution of a real local cosyvoice fixture - `Store::local`
never recognizes ANY cosyvoice checkpoint at all, even a real one.**
`TtsPipelineBuilder::load` (copied wholesale from `ForecastPipeline`'s own
"check `Store::local`, fetch-if-missing, THEN resolve" order) hit
`ModelNotFound("FunAudioLLM/CosyVoice2-0.5B: config.json has no
architecture")` against the test fixture below, despite every file
`CosyVoiceSpec::classify` needs being genuinely present and content-correct.
Root cause: `brain_modelstore` has no `FilesRecipe` entry for cosyvoice (no
conversion step either, unlike qwen3tts's `brain tts import`), so
`Store::local` - which only recognizes a compound `brain.manifest.json` or a
bare `model.brain.safetensors`, neither of which a raw cosyvoice checkpoint
ever has - returns `None`, and `plan()` falls through to
`TransformersRecipe`'s catch-all, which reads `config.json` for an
`architectures` field cosyvoice's repo does not carry and fails outright.
This is not fixture-only: a real user who `hf download`s
`FunAudioLLM/CosyVoice2-0.5B` into `$BRAIN_MODELS_DIR/FunAudioLLM/
CosyVoice2-0.5B` (the same layout every other architecture in this workspace
uses) and calls `TtsPipeline::from_pretrained` would hit the exact same
failure with real weights sitting right there. Fixed IN THIS CRATE ONLY
(not `brain_modelstore`'s shared recipe/plan machinery, out of scope here -
a cosyvoice `FilesRecipe` that can actually FETCH a checkpoint is real,
separate future work) by reordering `TtsPipelineBuilder::load` to try
`resolve_arch` FIRST - a real, content-based scan that reads raw file
content directly (`CosyVoiceSpec::classify`'s own per-file torch-checkpoint
reads), a strictly wider net than `Store::local`'s narrow "converted
checkpoint" shapes - and only fall through to `Store::local`/`plan`/download
when `resolve_arch` reports the model genuinely missing everywhere. This is
the REVERSE of every other pipeline in this crate (`ImagePipeline`/
`ForecastPipeline`/`VideoPipeline` all check-then-resolve, never
resolve-then-check) - deliberate, documented inline, and verified not to
regress any of them: it does not touch their code, and `TtsPipeline`'s own
existing qwen3tts tests (which already satisfied `Store::local` via its
compound manifest, so resolving on the first try was always possible) still
pass unchanged, now doing STRICTLY LESS work (skipping a redundant
`Store::local`/`plan` round trip) rather than different work.

**Test fixture**: `crates/sdk/tests/tts_pipeline.rs` reproduces
`cosyvoice::spec::tests`' own (private) `fixture`/`write_{llm,flow,hift}_pt`
shape - a CosyVoice 2 checkpoint under `FunAudioLLM/CosyVoice2-0.5B/` with a
compatible tokenizer under the sibling real-world repo
`FunAudioLLM/CosyVoice-BlankEN/` - proving resolution reaches a real,
constructed `TtsPipeline`, then `.clone_voice(...)` reaches
`cosyvoice::pipeline::generate`, which fails cleanly (`Error::Backend`) on
the fake CAM++/S3Tokenizer weight content (both env vars point at an empty
scratch directory) rather than resolution itself failing - the same
"resolution proven, full forward pass not" ceiling every pipeline in this
crate accepts. Two more tests prove the `Error::MissingArgument` dispatch
(`speak`/`design` unconditionally, `clone_voice` when `ref_text` is `None`)
without needing a real forward pass at all.

**`samples/tts/clone`, a genuinely new Rust sample** (not itself a roadmap
Phase item - added because it was asked for directly while this milestone
was in progress, and it is the first real end-to-end USE of
`TtsPipeline::clone_voice` this campaign has produced): `--voice PATH
--text TEXT [--ref-text TEXT] [--out PATH] [--model ID]`, defaulting to
Qwen3-TTS (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`, the same real model id
`crates/qwen3tts/src/spec.rs`'s own test fixture and `docs/models/
qwen3tts.md` already use). With no `--out`, the cloned clip plays on the
default audio output device instead of being written to disk - the first
sample in this workspace to need real audio PLAYBACK (every existing
sample either writes a file or drives a window). `rodio` (pure Rust,
`default-features = false, features = ["playback"]`) is the one new
third-party dependency this needed; `brain::Audio`'s own samples
(`samples/README.md`'s own doc: "already decoded f32 PCM") feed
`rodio::buffer::SamplesBuffer` directly, no WAV encode/decode round-trip
through a temp file. `samples/README.md` rule 1 ("a sample may depend
freely on third-party crates") covers this with no gate change needed; the
27 new third-party crates `rodio`'s `playback` feature pulls in (cpal +
platform backends) do not count against `max-brain-crates`, which only
counts `brain-*` crates.

**A real, pre-existing gap this sample's own `make docs` prep found and
fixed in passing**: `docs/manifest.txt`'s generated Samples part was
already stale before this milestone touched it -
`samples/decision/{arena,salesagent,triage}/README.md` existed on disk but
were never listed, silently invisible to `make docs`. Re-running
`scripts/build/gen-samples-manifest.py` (this milestone's own new sample
made that necessary anyway) picked up all three alongside the new
`samples/tts/clone/README.md` - not a regression this milestone introduced,
a decay from whenever those three samples themselves landed without the
generator being re-run.

Verified with `cargo build -p brain --features audio` (clean), `cargo build
-p sample-tts-clone` (clean on the first attempt, once `max-brain-crates`
was corrected from an initial wrong guess of 30 to the real measured 65 -
`audio` now pulls in BOTH qwen3tts and cosyvoice transitively), `bash
scripts/gates/check-samples.sh` (OK - 7 Rust samples including
`sample-tts-clone` at 65 brain crates/budget 66, plus the incremental-rebuild
assertion against `sample-tts-clone` itself: rebuild compiled exactly 1
crate, 0 of them brain crates), `cargo test -p brain --features audio --test
tts_pipeline` (6 passed: the 3 existing qwen3tts tests unchanged, plus 3 new
cosyvoice ones - a real local-fixture resolve-through-to-a-clean-
`Error::Backend`, and two `Error::MissingArgument` dispatch tests),
`cargo test -p brain --features audio --tests` (full audio surface suite -
transcribe/tts/upscale/video all still pass, no regressions from the
`TtsPipelineBuilder::load` reordering), `bash scripts/gates/
check-sdk-features.sh` (OK, every surface including `audio` still compiles
standalone), `bash scripts/gates/check-no-doc-citations.sh` (clean, after
fixing one real citation this milestone's own doc comment introduced), `bash
scripts/gates/check-doc-links.sh` (169 pages resolve), `python3
scripts/build/gen-samples-manifest.py --check` (clean after the
regeneration above), and a full `cargo build --workspace --exclude
brain-vulkan`.

### Phase 5.1 - `VideoPipeline` (Wan2.1 T2V) - the video generation bucket's first pipeline, and a real "invisible to the scanner" gap found while reusing the crate's own fixture

Covers **Wan2.1 T2V only**, over the already-existing `wan::spec::WanSpec`
(a real, already-migrated `ArchSpec` - `wan` was already resolver-based in
`crates/catalog`, unlike most archs this campaign has picked up cold).
`ltxv` (this bucket's other served architecture) is deferred, the same
"pick the more complete/migrated architecture in the bucket first" reasoning
Phase 4.1 used for qwen3tts over cosyvoice/minimaxmusic3 - `ltxv`'s catalog
registration still uses `always!()`, not a resolved `Assembly`.

**`VideoPipelineBuilder::load` never asks for a variant**: `WanSpec::
assemble`'s own comment already establishes that Wan's variant is NEVER
unrecoverable from the DiT's own tensor shapes (`dit_config_from_shapes`
matches an EXACT (dim, layer-count) pair against a real, named variant or
fails) - unlike flux2's klein-vs-base ambiguity, a resolved Wan `dit` always
names one full variant. `assembly.variant` is read straight off the
resolved `Assembly` and fed to `wan::caps::config_from_name`, the same
"the checkpoint itself is the variant" pattern the CodeFormer/SAM2/RRDBNet
pipelines already established for their own single-preset or
shape-derived configs.

**`VideoPipeline` is the second pipeline in this crate (after
`SegmentPipeline`) to hold resident device state behind a `Mutex`**:
`wan::caps`'s own module doc explains why residency matters here more than
anywhere else in this crate - "a 480p run is measured in tens of minutes...
a cold call pays a slow load plus 5.7 GB of upload; a second request at the
same size pays neither." `generate_with` locks a `Mutex<Option<wan::
pipeline::HotDit>>` and calls the SAME `wan::pipeline::generate_hot`
`WanProvider`'s own `generate_on` helper calls - one implementation of
generation, shared by CLI/D-Bus and the SDK, per rule 10.

**Scope this pipeline deliberately does NOT cover**: `lora_train` (a
training entry point - the same open, per-pipeline "no training call yet"
gap rule 9 tracks for every pipeline in this crate, not a narrower cut this
milestone introduced) and I2V (`wan::caps`'s own module doc says the crate
has no 36-channel input path or CLIP vision tower yet - advertising an
action that cannot run would be worse than not advertising it, the same
reasoning that module doc gives for the served `t2v` action itself).
Progress/cancellation stay at M5's SIMPLE default (`CancelToken::default()`,
a no-op progress closure) - `generate_hot`'s own `progress`/`cancel`
parameters are real and load-bearing (a 480p run needs both to be a servable
action at all), but wiring them through to a public SDK-level progress API
is the same tracked M5b gap every other pipeline in this crate still has,
not a new one.

**A real, independent "invisible to the real scanner" gap found while
building this milestone's own fixture, in a DIFFERENT spec than any prior
phase's bug**: `wan::spec::tests`' own `t2v_1_3b_fixture` (the fixture this
milestone's SDK test reproduces) writes its "prove `resolve()`'s root
inference lands on the right directory" sibling file as `other-vendor/
unrelated.bin` - but that test hand-builds its `ArtifactRecord`s directly
(`ArtifactRecord { path, kind: ArtifactKind::Opaque, .. }`), the SAME
"never goes through the real scanner" gap this campaign already found in
`RrdbnetSpec`/`Sam2Spec`, just manifesting differently here: going through
the REAL `brain_modelstore::inventory::scan` showed that `inventory::scan`
never emits a record for a `.bin` file at all (`kind_of_extension` does not
recognize the extension), so the sibling silently vanishes from the scanned
inventory, `brain_modelstore::resolve::common_root` collapses onto the
single vendor directory instead of the intended common ancestor, and
`vendor_dir`-based cross-role matching (`WanSpec`'s own tokenizer/
text_encoder vocab-compatibility check) breaks - the `tokenizer` role
stopped classifying, silently, only reachable through the FULL `resolve()`
call, not `WanSpec::classify` alone. This crate's own test suite has never
caught this because it never goes through the real scanner either. Not
fixed in `wan::spec::tests` itself (out of scope for this SDK milestone,
which does not otherwise touch that crate) - fixed only in this milestone's
own fixture, by using `.gguf` (a recognized extension, the same one `tests/
tts_pipeline.rs`'s own sibling file already uses) instead of `.bin`. Also
found and worked around: a `brain.manifest.json` anywhere in a directory
collapses the WHOLE directory into one `ArtifactKind::Compound` record,
hiding every other file in it from a content-based, per-file classifier
like `WanSpec::classify` - so this fixture satisfies `Store::local`'s own
presence check by naming the DiT file `model.brain.safetensors`
(`Store`'s own `BASE_WEIGHTS_FILE` fallback name) instead of adding a
manifest, keeping every real file individually visible to the scanner.

**A fast, real end-to-end RESOLUTION test** (not a full forward pass -
see below): because `dit_config_from_shapes`/`is_wan_vae`/
`is_wan_text_encoder` all read shapes/marker tensors rather than full
weight content, `video_pipeline.rs` reaches a real `Resolved` assembly and
a real, constructed `VideoPipeline` in under a second, reusing
`wan::spec::tests`' own minimal-tensor fixture discipline (one marker
tensor per DiT block, a 4-channel toy VAE, a 100-token toy vocabulary).

**Not a full forward pass, by design**: `wan::pipeline::generate_hot` needs
every DiT block's REAL weights to build the transformer at all (this
fixture carries only one marker tensor per block, matching `wan::spec::
tests`' own discipline for staying fast), so `.generate()` fails cleanly
(`Error::Backend`) rather than running a genuine multi-minute video
generation - the same "resolution proven, full forward pass not" scope
`TtsPipeline`'s/`TranscribePipeline`'s own tests already accepted, for a
different underlying ceiling (there: a real tokenizer; here: a
would-be-genuinely-huge complete tensor set with no practical shrink
lever, `WanConfig`'s two presets being fixed, named, resolver-gated exact
shapes, not derived like RRDBNet/YOLOv8/ZipDepth's).

Verified with `cargo build -p brain --features video` (clean), `bash
scripts/gates/check-sdk-features.sh` (OK, `video` compiles standalone,
clean on the first attempt), `cargo test -p brain --features video --test
video_pipeline` (3 passed, under 10s total), `bash scripts/gates/
check-no-doc-citations.sh` (clean), and a full `cargo build --workspace
--exclude brain-vulkan`.

### Phase 6.1 - `VisionLanguagePipeline` (Qwen3-VL) - the Multimodal/VLM/OCR domain bucket's first pipeline

Covers **Qwen3-VL only**, over the already-existing `qwen3vl::spec::
Qwen3VlSpec` (a real, resolver-ready `ArchSpec` since before this session -
one `weights` role, a checkpoint directory classified by real
`config.json` content, `architectures[0] == "Qwen3VLForConditionalGeneration"`,
never the directory's name). Picked first among this bucket's 8 other served
architectures for the same reason Phase 4.1 picked qwen3tts over
cosyvoice/minimaxmusic3: a real, already-exercised `generate` path
(`qwen3vl::caps`'s own module doc: "Real, working, but validation-tier",
backed by a deterministic regression test with a hardcoded pre-change-output
assertion), not the "no checkpoint or reference dump exists in this
environment, treat a first real generation as the actual test" ceiling
`crates/flux1`'s own pipeline module doc carries for its own (unrelated)
architecture - a real, meaningfully different risk class this milestone
explicitly checked for and steered around before committing to a target.

**ONE pipeline type, reusing `qwen3vl::caps::Resident::generate` directly -
not a second implementation.** `Resident::generate`'s own doc already states
its purpose exactly: "The body `GenerateAction::run` used to hold directly,
extracted so a residency-scheduled instance and the direct provider execute
byte-for-byte the same code." `VisionLanguagePipeline::ask_multi_with`
builds a plain `capability::Invocation` (prompt + numbered image blobs) and
calls that SAME function - the identical shape
[`crate::TextGenerationPipeline`] already established (build an
`Invocation`, call into `qwen3::chat`'s shared parsing/generation), not
"capability-dispatch machinery in the loop" (no `Provider`/`Action` trait
object, no routing - just the plain data types and the shared function every
other caller of this checkpoint already goes through).

**`GeneratedText` reused, not reinvented** - `qwen3vl::caps::Resident::
generate` runs `qwen3::chat::SeqState::finish` internally (the SAME
completion-outcome shape `TextGenerationPipeline`'s own `qwen3::chat::
parse_request`+`SeqState`+`generate_kv_stream` sequence produces, since
Qwen3-VL is a vision tower spliced onto a qwen3 decoder), so this pipeline's
`ask`/`ask_multi` return the EXISTING `brain::GeneratedText` type via the
now-`pub(crate)` `crate::text::generated_text_from_outcome` helper, rather
than a parallel, near-identical struct. The `multimodal` Cargo feature
selects `text` for exactly this reason (documented inline in `Cargo.toml`,
not just "needed to compile") - a real dependency relationship, not a
feature picked only to satisfy the compiler.

**ONE capability, not three call shapes** - `ask(image, prompt)` is a
one-image special case of `ask_multi(images, prompt)` (`std::slice::
from_ref`), never a separate implementation: `qwen3vl::caps`'s own doc
frames multi-image as numbered blob keys (`image`, `image1`, ...) contiguous
from `image`, so a single image is just the one-element case of the same
convention, not a different one.

**A real, independent bug this milestone would have reintroduced a THIRD
time, caught before it shipped by checking for it up front this time**
(rather than discovering it via a failing test, as Phase 4.2/2.2b each did
once): a real, already-downloaded qwen3vl checkpoint directory (`hf
download Qwen/Qwen3-VL-4B-Instruct --local-dir $BRAIN_MODELS_DIR/Qwen/
Qwen3-VL-4B-Instruct`, the standard HF layout - `config.json` +
`model.safetensors` shards + `tokenizer.json`) satisfies neither of
`Store::local`'s two recognized shapes (a compound `brain.manifest.json`, or
a bare `model.brain.safetensors`), so the naive "check `Store::local`,
fetch-if-missing, then resolve" order every OTHER pipeline in this crate
uses would fail the exact same way cosyvoice (Phase 4.2) and a raw qwen3
GGUF release (Phase 2.2b) both did. `VisionLanguagePipelineBuilder::load`
tries `resolve_structured` FIRST, from the start, applying the now-
three-times-confirmed fix directly rather than waiting to hit it.

**A real, confirmed ceiling this milestone's own fixture-building hit
empirically, not assumed**: unlike every OTHER resolver-backed pipeline's
own test (`tts_pipeline.rs`/`video_pipeline.rs`/...), which reach a real
CONSTRUCTED pipeline against fake-content-but-real-shape tensors and then
show the first call failing cleanly, `qwen3vl`'s own weight-upload path
(`brain_paramstore`) PANICS - `missing init weight tok.weight: ... not
present in this source` - the instant a declared tensor is absent from the
source file, confirmed by actually building a tiny (if minimal) real HF
config + fake safetensors fixture and running it through
`VisionLanguagePipeline::from_pretrained`. This mirrors
`TextGenerationPipelineBuilder::load`'s own documented reason for
pre-checking `WeightReader::open` before calling the equally panicking
`Qwen::load_inference` - the difference is `text.rs` has an earlier public
stopping point (checkpoint-open, then a clean tokenizer-precedence error)
this pipeline does not, since resolution and construction are one call with
no earlier hook. Reproducing a full, real tensor manifest for both the
vision tower AND the text decoder (unlike qwen3tts's/wan's own three-or-four
-role fixtures) is real, separate fixture-building work this milestone does
not take on - not fixed here (`brain_paramstore`'s panic-on-missing-tensor
is a workspace-wide, intentional "this should never happen with a real
checkpoint" invariant, the same class every other model-loading call in
this crate already accepts panicking on, not a `qwen3vl`-specific bug).

**Scope this pipeline deliberately does NOT cover**: video input and
tool-calling, both REAL, already-implemented capabilities of `Resident::
generate` itself (`video_frames`/`tool_choice`/`tools` parameters this
pipeline always passes as `None`/`ToolChoice::Auto`/`&[]`) - tracked future
extensions, the same class of narrowing `TtsPipeline`'s own `lora_train`/
progress gaps already document, not silently unsupported. The other 8
served Multimodal/VLM/OCR architectures (qwen3omnimoe, fastvlm, llava,
moondream3, deepseekocr2/deepseek2ocr, qwen35/qwen35moe's own VLM shape)
stay tracked future extensions too.

**Test fixture**: `crates/sdk/tests/vlm_pipeline.rs` proves the reachable
ceiling - an unparseable model id (`Error::ModelNotFound`) and a real,
content-classifiable checkpoint with no sibling `tokenizer.json`
(`Error::Missing`, via `Qwen3VlSpec::validate`'s own rejection, fully
offline via the same generic `Store::local`-satisfying manifest trick
`tests/text_pipeline.rs`'s own hub-id-missing test uses). `check_image_count`
(the images.is_empty()/too-many-images gate) is factored out and unit-tested
directly inside `crates/sdk/src/vlm.rs`'s own `#[cfg(test)]` module - the
same "testable with no real pipeline in hand" shape `crate::pipeline::
check_s3dit_size`/`adapter_source_path` already established - since it is an
instance method's precondition with no way to reach it from an integration
test without first building a working pipeline (out of reach - see above).

Verified with `cargo build -p brain --no-default-features --features
multimodal` (clean, one pre-existing dead-code warning shared with the
`vision`-alone build, not introduced here), `bash scripts/gates/
check-sdk-features.sh` (OK, `multimodal` compiles standalone), `cargo test
-p brain --features multimodal --lib vlm::` (4 passed), `cargo test -p
brain --features multimodal --test vlm_pipeline` (2 passed, both under
0.2s - confirms no accidental network reach), `bash scripts/gates/
check-no-doc-citations.sh` (clean), `bash scripts/gates/check-doc-links.sh`
(169 pages resolve), `bash scripts/gates/check-scripts.sh` (PASS).

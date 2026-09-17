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
| Multimodal/VLM/OCR | 9 | 8 | none | `omni_cli.rs`, `document_study_cli.rs` + 5 `resident_*.rs` | yes (qwen3vl) |
| Image generation | 6 | 6 | **YES - `ImagePipeline`** (flux2+s3dit only) | `flux2_cli.rs` + `s3dit::caps::ZAction` | yes (flux2, s3dit) |
| Restoration/upscaling/VAE | 5 | 5 | **PARTIAL - `UpscalePipeline`** (RRDBNet) + **`RestorePipeline`** (CodeFormer; SUPIR/VQGAN deferred, see Phase 2.5/2.7) | no dedicated CLI; `resident_restore/upscale/supir.rs` | no |
| Video generation | 2 | 2 | none | `wan_cli.rs`, `ltxv_cli.rs` | yes (wan) |
| ASR | 2 | 2 | none | **no CLI at all** - `resident_asr.rs` only | no |
| TTS/music/speech codec | 7 | 3 | none | `tts_cli.rs` + `tts_serve.rs` | no |
| Vision/detection/segmentation | 4 | 4 | **PARTIAL - `DetectionPipeline`** (YOLOv8) + **`SegmentPipeline`** (SAM2; depth/label deferred, see Phase 3.1/3.2) | `yolo_cli.rs`, `sam2_cli.rs`, `depth_cli.rs`, `label_cli.rs` | no |
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
| 4 | 8 | `crates/sdk/src/pipeline.rs:300,481,504` | download and s3dit-build progress are likewise discarded | open (M5b - split out, see below: three different progress shapes to plumb, not one) |
| 5 | 6 | `crates/sdk/src/error.rs:70-71` | `Error::Cancelled` is a dead public variant - unreachable with no public cancel entry point | fixed (M5) |
| 6 | 8 | crate-wide | no `.capabilities()`/manifest introspection anywhere in `crates/sdk` | fixed (M8) |
| 7 | 9 | crate-wide | no training/finetune entry point at all in `crates/sdk` (the adjacent Dataset-layer gap is tracked in `sdk.md`; the missing training call itself was not) | open (Phase 2, per-pipeline) |
| 8 | 9/4 | `crates/sdk/src/creature.rs:492-500` | `set_plasticity`/`reward` mutate learned synapse weights with no `save()`/`load()` counterpart - all learning dies with the process | open (backlog) |
| 9 | 7 | `crates/sdk/src/creature.rs:157` | `Creature::build` acquires its GPU via `gpu_core::testgpu::dev` - TEST-SUPPORT infra, weak-reference lifetime, shipping on the production SDK path | fixed (M3) |
| 10 | 7/13 | `crates/sdk/src/creature.rs` + `Cargo.toml` | `CreatureBuilder` has no `Device` knob at all, yet the `creature` feature's doc comment claims it "selects `device`" | fixed (M3) |
| 11 | 5 | `crates/sdk/src/pipeline.rs:416-424,432-435` | `ImagePipelineBuilder::size` is silently IGNORED on a flux2-backed pipeline (the s3dit half of this asymmetry is tracked in `sdk.md`; the flux2 silent no-op was not) | open (backlog) |
| 12 | 6 | `crates/sdk/src/error.rs:62-66` | `Error::Backend` is an untyped catch-all for ~8 semantically distinct failures (license refusal, no models dir, size mismatch, bad extension, missing builder arg, GPU/MuJoCo failure...) | partially fixed (M6 - see below) |
| 13 | 6 | `crates/sdk/src/creature.rs:129-130` | a caller-programming error (missing required builder field) is typed as `Error::Backend`, indistinguishable from a real backend crash | fixed (M6) |
| 14 | 8/4 | `crates/sdk/src/creature.rs:276-278,414-416` | `wiring()`/`wing_wiring()` return a pre-rendered human summary string; no structured/programmatic accessor | open (backlog) |
| 15 | 3 | `crates/sdk/src/creature.rs:257-268` | `Creature::fruit_fly()` has two mandatory runtime-checked fields and no simple zero-arg path | **wontfix (M7)** - see below |
| 16 | 13/10 | `crates/fly/examples/watch.rs:18-60` | hand-builds `Fly`/`SdlWindow`/`Renderer` independently of `Creature`/`View` | **not a violation on reconsideration (M7)** - see below |
| 17 | 13 | `docs/` | no user-facing SDK page exists for either surface (rustdoc itself is compliant) | fixed (M9) |

### Minor / stylistic (backlog, not milestoned individually - fold into whichever nearby milestone touches that file)

18. `ImagePipelineBuilder::load()` vs `CreatureBuilder::build()` - two terminal verbs for the same act in one crate.
19. No `Creature::builder()` alongside `Creature::fruit_fly()` (arguably justified - the species names the assembly).
20. Unprefixed setters (`drive`, `turn`, `reward`) beside `set_`-prefixed ones (`set_wing_power`, `set_plasticity`) on the same type.
21. ~~`creature.rs:283-285` - `drive()`'s doc documents a `turn` param it doesn't take; `turn()` next to it has no doc.~~ fixed in M2.
22. ~~`error.rs:82` - `Error::Backend` renders with no `brain: `-style prefix.~~ **wontfix**: `error.rs`'s own module doc documents this as an intentional VERBATIM passthrough for in-process callers (the underlying flux2/s3dit message routinely already self-identifies, e.g. `"flux2: assemble: no dit chosen"`), and `backend_carries_the_original_message_verbatim` pins exactly that contract. A generic prefix would contradict the documented behavior and break the test for no real gain.
23. `creature.rs:63` - `pub use flybody::Arena;` promoted into the compatibility surface with no doc comment justifying it, unlike `Device`/`DType`.
24. `view.rs:86` - `View::open(creature, title, width, height)` is four positional args with no options type.

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
- [ ] **M5b** - download progress (`ImagePipelineBuilder::load`'s
      `execute_plan` call, finding 4) and s3dit build progress
      (`HotPipeline::build_adapted`'s `impl FnMut(&str)`) are still
      discarded. Split out because these are two MORE distinct progress
      closure shapes (`FnMut(&str, u64, Option<u64>)` for downloads,
      `FnMut(&str)` for the s3dit build) on top of generate's
      `FnMut(u32, u32, &str)`, plumbed through a builder that is consumed by
      value rather than a `&self` call - needs its own design pass for
      where a boxed closure lives on `ImagePipelineBuilder`, not a
      copy-paste of M5's pattern.
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
- [ ] **Phase 2** - new pipelines, in the priority order above: Forecast, Text, Embedding, ASR, then the rest. Each gets its own sub-roadmap section here (or its own file, linked from here) when it starts, written against the full `sdk-design.md` checklist from day one - including an end-to-end test, learning from M10 rather than repeating the `flux2_cli.rs` duplication gap a second time. Design each pipeline's progress/cancellation surface (rule 8) toward the `run.start()/subscribe()/cancel()/result()` shape `.agents/roadmap/orchestration-hsm.md` proposes, rather than reinventing M5's synchronous `generate_with_progress(cancel, on_progress)` a second time - M5's shape stays the right SIMPLE default, but a new pipeline's ADVANCED tier should point at where this is heading.
  - [x] **Phase 2.1** - `ForecastPipeline` (kronos, timesfm3).
  - [x] **Phase 2.2** - `TextGenerationPipeline` (qwen3 only).
  - [x] **Phase 2.3** - `EmbeddingPipeline` (CLIP text towers only).
  - [x] **Phase 2.4** - `TranscribePipeline` (qwen3-asr only).
  - [x] **Phase 2.5** - `UpscalePipeline` (RRDBNet only) - see its own section above for the real `RrdbnetSpec` bug this one found and fixed, and the new `Image::open`/`Image::from_rgb8` public API it needed.
  - [x] **Phase 2.6** - `codeformer::spec::CodeFormerSpec`, the prerequisite Phase 2.5 named - see its own section below.
  - [x] **Phase 2.7** - `RestorePipeline` (CodeFormer) - see its own section below for two real forward-pass infrastructure bugs this one found AND FIXED (a duplicate kernel registration that broke the CPU JIT backend; a `backend-wgpu` buffer-reclaim ceiling from two unpolled `Builder` scopes) - this pipeline's test reaches a genuine, complete `.restore()` forward pass at CodeFormer's real fixed geometry, the strongest end-to-end proof of any pipeline in this crate so far. SUPIR and VQGAN stay deferred with the reasons already on record.
  - [x] **Phase 3.1** - `DetectionPipeline` (YOLOv8 only) - the vision/detection domain bucket's first pipeline, and its first NEW domain object (`Detection`, not `Image`). See its own section below.
  - [x] **Phase 3.2** - `SegmentPipeline` (SAM 2.1) - see its own section below for a real, independent `Sam2Spec` bug found and fixed (the SAME `ArtifactKind::Opaque`-vs-`Torch` mistake `RrdbnetSpec` had), and a genuine end-to-end `.segment()` forward pass at SAM 2.1's real fixed geometry.
  - [ ] Still entirely uncovered domain buckets: TTS/music, video generation, 3D/world models. See the domain inventory table.

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

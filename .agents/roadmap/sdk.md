# sdk - roadmap

The rules this crate (and every future pipeline/CLI/Python surface) is held to
are `.agents/rules/sdk-design.md`. Several "not yet done" items below are that
document's own review checklist failing today, tracked here rather than
re-stated there: no `AutoPipeline` and no multi-architecture dispatch on most
of the pipelines that DO exist (rule 2 - see `.agents/roadmap/
sdk-design-sweep.md` for the per-pipeline scope), the CLI building its own
`flux2::Pipeline`/`HotPipeline` instead of calling `ImagePipeline` (rule 10),
and `brain-py` driving a subprocess instead of a direct binding (rule 10).

crates/sdk (package `brain`, the deliberate sole exception to this workspace's
`brain-<short>` naming convention) is brain's public embeddable SDK facade:
`ImagePipeline` resolves a local model store through `crates/loader`'s
resolver -- the SAME resolver the CLI uses -- and builds a real, resident
model behind ONE public type, dispatching internally on the resolved
`capability::Assembly::arch` (`pipeline::resolve_arch`, private) to either a
flux2-backed or an s3dit-backed variant, with no CLI process and no
capability-dispatch machinery in the loop. `from_pretrained`/`builder`/
`generate`/`generate_with`/`load_lora`/`.save()` are uniform across both
backends even though their underlying `generate()` calls return structurally
different shapes (flux2: `(Vec<u8> RGB8, u32, u32)`; s3dit: a float HWC
`Image { hwc: Vec<f32> in [0,1], w, h }`) -- normalized into one `brain::Image`
via `imaging::pixels::hwc_to_rgb8` for the s3dit case, the crate's existing
f32-HWC-to-u8-RGB8 helper reused rather than duplicated. `ImagePipelineBuilder`
now also carries `.size(width, height)`, s3dit's load-time-only build shape.
Covered by unit tests (the adapter-source-path gate, generation-option
defaulting, the s3dit per-call/build-time size validation) and by resolution +
dispatch integration tests against local, synthetic, fully offline fixtures
for BOTH backends (`crates/sdk/tests/image_pipeline.rs`); neither integration
test reaches an actual successful `.generate()` -- see the first item below
for why.

## Not yet done

- [ ] Full end-to-end generation test coverage: neither the flux2-backed nor
      the s3dit-backed integration test reaches a successful
      `.generate()?.save()`. flux2's `Flux2Config::from_name` only ever
      returns one of four REAL, several-billion-parameter variants -- there
      is no tiny/injectable config reachable through this SDK's public API.
      s3dit's own `dit_config`/`QwenConfig::qwen3_4b` are hardcoded to the
      one shipped Z-Image-Turbo/Qwen3-4B shape at every `HotPipeline::build*`
      call site, with no config-injection seam at all (not even the kind
      flux2's own `Pipeline::build_with(cfg: &Flux2Config, ...)` offers past
      `from_name`'s four names) -- a second checkpoint shape would need
      `s3dit::pipeline::dit_config` taught to read it first. Both tests
      instead prove resolution, dispatch, and construction all the way to
      that real, pre-existing weight-size ceiling, then assert a clean
      `Error::Backend`; real success is exercised only by each backend's own
      `#[ignore]`-gated real-checkpoint tests elsewhere in the workspace.
- [ ] Adapters via store `owner/name` refs - bigger than it first looked;
      re-diagnosed while scoping it as a Phase item, not mechanical wiring.
      The model store's adapter convention
      (`brain_modelstore::Store::adapter_weights_path`, `ModelRef::adapter()`,
      files under `<base>/adapters/<owner>/<name>/<tag>/`) is not an
      `ArchSpec`-role gap at all - it is a SEPARATE, `ModelRef`-level
      mechanism `Store::local` resolves directly, architecture-agnostic in
      principle. The REAL gap: nothing ever WRITES a flux2/s3dit adapter into
      that convention. `crates/cli/src/qwen_cli.rs`'s `finetune_lora` is the
      only writer that exists, qwen3-only; `flux2::lora::save_adapter`/
      `s3dit`'s own finetune save to a caller-given path with no store
      integration at all. So wiring `ImagePipeline::load_lora` to accept a
      store reference needs an answer to a real design question FIRST: is
      there a `brain flux2 finetune --lora`-shaped writer to point it at, or
      does this stay path-only until `ImagePipeline` gets a real
      training/finetune entry point (finding 7, still open) that could write
      one? Picking a convention for EXTERNALLY-sourced LoRAs (downloaded, not
      trained by this workspace) is a second, related question the qwen3 path
      never had to answer. Not scoped for a mechanical fix.
- [x] TextGenerationPipeline/EmbeddingPipeline/TranscribePipeline/
      ForecastPipeline/UpscalePipeline: built (Phase 2.1-2.5 - see
      `.agents/roadmap/sdk-design-sweep.md`, the live tracker for this line
      of work; this bullet was stale until fixed in the UpscalePipeline
      milestone that also added the fifth). Scoped to one architecture (or
      two, for Image/Forecast) each, not the full multi-architecture
      dispatch rule 2 asks for long-term - see that document's own
      "Not done, tracked for later" note per pipeline, and the domain
      inventory table for what has NO pipeline yet.
- [ ] The semantic Dataset layer: no file-backed dataset loader exists for
      any training objective this SDK could eventually expose. DPO is the
      concrete example: `crates/rl/src/objective/dpo.rs`'s
      `pairs_from_group` only ever derives `DpoPair`s from a live rollout
      `Completion` group (`chosen`/`rejected` picked by reward sign within
      one in-memory group) -- there is no way to load pre-collected
      preference pairs from a file.
- [ ] AutoPipeline: not built. `ImagePipeline::from_pretrained` does its own
      two-architecture dispatch internally, but there is no public
      multi-modal `AutoPipeline`-style type that a caller could hand ANY
      supported model id -- image, text, embedding -- and get the right
      pipeline TYPE back; today a caller must already know they want
      `ImagePipeline` specifically.
- [ ] CLI migration onto the SDK: `crates/cli/src/flux2_cli.rs` and
      `s3dit::caps::ZAction`/`crates/cli/src/resident.rs` each still
      construct their own `flux2::Pipeline`/`s3dit::pipeline::HotPipeline`
      inline, independently of `crates/sdk` -- the CLI does not call
      `brain::ImagePipeline` at all, so the resolve/build logic genuinely
      exists in two places (the SDK's copy and the CLI's own), not one
      shared one. There are actually SIX independent build sites on the
      flux2 side alone (`flux2::caps` and `resident_flux2.rs` are two more,
      and neither calls `effective_dit_precision` -- a real bug, not just
      duplication: a served `.gguf` DiT misses the fp32-to-packed-int8
      correction the CLI and SDK both apply). Full site-by-site map and the
      planned extraction: `.agents/roadmap/sdk-design-sweep.md`.
- [x] Per-domain Cargo features: done. `crates/sdk/Cargo.toml` now gates
      `brain-flux2`/`brain-s3dit`/`brain-imaging`/`brain-model`/
      `brain-capability`/`brain-loader`/`brain-modelref`/`brain-modelstore`/
      `brain-gpu-core` behind `optional = true`, selected per surface
      (`image`, `creature`); an embedder who only wants one backend no
      longer links the other's dependency closure. Enforced by
      `scripts/gates/check-sdk-features.sh` (`make check/sdk-features`).
- [ ] s3dit's build-time size/cap_len/hifi coupling: `s3dit::pipeline::
      HotPipeline::build_adapted` records its DiT/VAE graphs for exactly one
      `(width, height, cap_len, hifi)` shape at construction
      (`check_build_shape`), unlike flux2 where size is a per-generate-call
      `GenOpts` field. This milestone resolved the asymmetry pragmatically:
      `ImagePipelineBuilder::size` is a LOAD-time-only option for an
      s3dit-backed pipeline, and `ImageGenerationOptions::size` on
      `generate_with` is VALIDATED against it (a mismatch is a named
      `Error::Backend`, never silently ignored or silently resized) rather
      than applied. `cap_len` (the caption token capacity) and `hifi` (fp32
      vs int8 DiT) are similarly build-time-only and not exposed as SDK
      builder knobs at all yet -- `ImagePipelineBuilder::load` always builds
      s3dit at `s3dit::pipeline::DEFAULT_CAP_LEN` and `hifi = (dtype ==
      DType::F32)`, a literal-but-surprising reading of `DType::F32` that
      maps to the HEAVIER 2-GPU fp32 build (`brain do z-image text2image`'s
      own CLI default is `int8`, not `fp32`) -- an embedder relying on this
      SDK's `DType::F32` default gets a different, heavier build than the
      CLI's own default for the same architecture.
- [x] `DownloadPolicy`: done - see `.agents/roadmap/sdk-design-sweep.md`
      Phase 6.3/6.4/6.5. Every pipeline builder in the crate now has a
      `.download_policy(...)` knob selecting `Offline`/`IfMissing`/
      `AlwaysCheck`: the 12 single-architecture builders (depth, embedding,
      restore, ground, music, video, vlm, text, detect, segment, upscale,
      asr) share `crate::resolve_policy::resolve_with_policy`; `TtsPipeline`
      (qwen3tts/cosyvoice) and `ForecastPipeline` (kronos/timesfm3) share the
      new two-architecture `crate::resolve_policy::resolve_two_with_policy`;
      `ImagePipeline` (flux2/s3dit) keeps its own inline version because it
      alone also reports download/build progress through caller-supplied
      closures, a capability neither shared helper carries.
- [ ] `resolve_arch`'s two-architecture tie-break: when a store resolves
      NEITHER flux2 nor s3dit, the reported `Error::Ambiguous`/
      `Error::Missing` prefers whichever architecture found real (if
      ambiguous) evidence over one that found none, and falls back to
      flux2's own outcome when both are plain `Missing` -- an arbitrary
      tie-break (flux2 is tried first), not a claim that flux2 is the more
      likely answer. A third image architecture would need this dispatch
      generalized past its current two-way `if`/`if` shape.
- [ ] `steps` has no upper bound anywhere in the call chain, on either
      backend: neither `ImageGenerationOptions`, nor flux2's
      `resolved_steps`, nor s3dit's `HotPipeline::generate` (which only
      floors it to `.max(1)`) refuses an absurd value -- a caller who passes
      `steps(u32::MAX)` gets a denoise loop that many iterations long, not a
      clean refusal. Pre-existing in both backends' own APIs; deliberately
      not given an SDK-only clamp here, which would only hide the same gap
      from every other caller of `flux2`/`s3dit` directly. Recorded in
      detail in the security-audit section below.
- [ ] Prompt-length handling is INCONSISTENT between the two backends this
      one SDK type now fronts: flux2 silently TRUNCATES an overlong prompt
      to `cfg.txt_len` tokens (an `eprintln!` warning, never an `Err`,
      `crates/flux2/src/pipeline.rs`'s encode path), while s3dit cleanly
      REFUSES one past its built `cap_len` capacity (`fit_caption`). A
      caller cannot tell, from `brain::Error` alone, which behavior they are
      going to get for an overlong prompt -- fixing flux2's own truncation
      behavior is out of this milestone's scope (it is flux2's, not the
      SDK's, to change), but the inconsistency is now doubly visible with
      one facade type fronting both.

## Security audit (Part 3, this milestone)

Applied `.agents/rules/api-security.md`'s checklist to `crates/sdk`'s new
public surface. This crate is a local, in-process embeddable library, not a
network service, so several sections are not applicable the way they are for
`crates/apiserve`/`crates/dbus`:

- **§1 Authentication & authorization -- N/A.** No route, no key, no bus
  name; nothing in this crate accepts a connection from anywhere.
- **§7 Transport -- N/A.** No server binds; there is no interface to default
  to localhost.
- **§6 Output/data exposure -- N/A.** No listing or stats surface exists in
  `crates/sdk`.

The sections that DO apply, with real attention paid:

- **§2 Input handling & DoS.** `width`/`height` are bounded by real,
  pre-existing checks in each backend's own code (flux2's fixed build-time
  forward-token ceiling; s3dit's `check_build_shape` RoPE-table
  addressable-size check, now reached at `ImagePipelineBuilder::size` time)
  -- verified, not assumed, by reading both call paths. `steps` has NO
  upper bound in either backend (see "Not yet done" above) -- a real,
  pre-existing gap this milestone deliberately did not paper over with an
  SDK-only clamp, because that would mask the same gap from `flux2`/`s3dit`
  callers who bypass this crate entirely. Prompt length is bounded, but
  INCONSISTENTLY (flux2 truncates silently; s3dit refuses cleanly) -- also
  recorded above rather than fixed here, since the fix belongs in flux2's
  own encode path, not this facade.
- **§3 Resource safety & backpressure.** This crate has no admission
  control, queue, or concurrency limit of its own -- by design, since it is
  a library, not a server (documented in `crates/sdk/src/lib.rs`'s own
  module doc now). Each `ImagePipeline` holds real multi-gigabyte GPU/host
  memory for as long as it lives; nothing here stops a caller from building
  several concurrently. A service built on top of this crate that accepts
  requests from an untrusted network needs its own admission/backpressure
  layer in front of it, the same way `apiserve`/`dbus` provide one in front
  of everything else in this workspace -- this crate does not, and should
  not, try to provide that itself.
- **§4 SSRF / egress.** Traced end to end:
  `ImagePipelineBuilder::load`'s auto-fetch path reuses
  `brain_modelstore::plan`/`brain_modelstore::HfHub`/
  `brain_modelref::ModelRef::parse` VERBATIM -- the exact same,
  already-audited machinery `crates/apiserve`/`crates/dbus` route their own
  auto-fetch dispatch through. `model_id` is parsed and validated
  (path-traversal/reserved-vendor rejection) before it ever reaches `plan`,
  and `HfHub`'s destination host is fixed to `huggingface.co`, never
  request-derived. This crate introduces no new fetch mechanism and no new
  attacker-controlled URL -- confirmed, not merely assumed.
- **§5 Error hygiene.** `Error::Backend`/`Error::Download` carry the
  underlying flux2/s3dit/modelstore crate's own message VERBATIM, which
  routinely includes real on-disk paths. That is the correct, intended
  contract for an in-process caller who already has filesystem access and
  needs the real reason a build failed -- but it is exactly what a
  network-facing surface must never do, and nothing in this crate warned an
  embedder of that before this audit. Fixed in this milestone: `Error`'s own
  module doc (`crates/sdk/src/error.rs`) now says so explicitly, so a
  caller who re-exposes `Error::Display` to an untrusted network client
  knows they are inheriting a path-disclosure surface and must add their
  own translation layer first.

No finding from this pass required a code-behavior change beyond the two doc
additions above (`crates/sdk/src/lib.rs`'s resource-safety note,
`crates/sdk/src/error.rs`'s error-hygiene note); every other real finding
(`steps` unbounded, the flux2/s3dit prompt-truncation inconsistency) is
recorded as a "Not yet done" item above rather than given an SDK-only
band-aid that would hide the same gap from every other caller of
`flux2`/`s3dit` directly.

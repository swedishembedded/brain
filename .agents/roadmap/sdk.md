# sdk - roadmap

The rules this crate (and every future pipeline/CLI/Python surface) is held to
are `.agents/rules/sdk-design.md`. Several "not yet done" items below are that
document's own review checklist failing today, tracked here rather than
re-stated there: no multi-architecture dispatch on most of the pipelines
that DO exist (rule 2 - see `.agents/roadmap/sdk-design-sweep.md` for the
per-pipeline scope; `AutoPipeline` itself is done - see that file's Phase
6.6), the CLI building its own
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
- [x] AutoPipeline: done - see `.agents/roadmap/sdk-design-sweep.md` Phase
      6.6. `brain::AutoPipeline::from_pretrained` tries every known
      architecture's resolver against the current model store and hands
      back the matching concrete pipeline type, boxed, as one of 15 enum
      variants - a new `auto` surface feature (not a `Domain` variant,
      `scripts/gates/check-sdk-features.sh` now names it an explicit
      cross-cutting exception) that pulls in every other resolver-backed
      surface as a hard prerequisite.
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
- [x] s3dit's build-time size/cap_len/hifi coupling: the `cap_len`/`hifi`
      half is done. `ImagePipelineBuilder::cap_len(u32)` (default
      `s3dit::pipeline::DEFAULT_CAP_LEN`, unchanged) and
      `ImagePipelineBuilder::hifi(bool)` (default `None`, still deriving
      `dtype == DType::F32` when unset - the pre-existing reading, only
      named now, not changed) are new builder knobs threaded through BOTH
      s3dit build sites (`ImagePipelineBuilder::load`/`load_with_progress`
      and `ImagePipeline::load_lora`'s rebuild), stored on `S3ditBackend`
      the same way `width`/`height` already are so an adapter reload never
      silently drops a caller-requested capacity/precision back to the
      default. Purely additive - no default changed, so this does not yet
      touch the OTHER half of this gap (the surprising DEFAULT itself:
      `DType::F32` mapping onto the heavier 2-GPU hifi build when the CLI's
      own default is int8 - now escapable via `.hifi(false)`, but still the
      out-of-the-box behavior for a caller who never calls either knob).
      Proven with two new fixture tests:
      `cap_len_reaches_the_s3dit_build_and_is_validated_before_the_dit_is_opened`
      (`cap_len(0)` fails `check_cap_len`'s own message BEFORE the DiT
      checkpoint opens - an earlier, distinct wall from the default
      cap_len's "incomplete tensor set" failure the pre-existing tests
      reach, proving the value is genuinely threaded through and not
      silently defaulted) and `hifi_override_still_reaches_a_real_build_
      attempt` (`.hifi(true)` with no `.dtype(...)` call reaches the same
      clean build failure). Size/`ImageGenerationOptions::size` validation
      (the part already fixed) is untouched.
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
- [x] `resolve_arch`'s two-architecture tie-break: done. `resolve_arch` now
      calls `crate::resolve_policy::try_two` (made `pub(crate)`) instead of
      carrying its own byte-for-byte copy of the same `if`/`if` tie-break -
      the duplication `resolve_two_with_policy` (Phase 6.5) left in place
      because it only needed `try_two`'s LOGIC, not its policy/fetch
      wrapper (`ImagePipelineBuilder` keeps its own progress-reporting fetch
      path). Same observable behavior (flux2 tried first, `Ambiguous` beats
      `Missing`, two `Missing`s report flux2's), now one implementation
      instead of two. `cargo test -p brain --features image --test
      image_pipeline` (6 passed, including both backends' own dispatch
      tests) and `cargo clippy -p brain --features full --all-targets`
      (clean on both touched files) confirm no behavior change.
- [x] `steps` has no upper bound anywhere in the call chain: done, fixed in
      BOTH backends directly (not with an SDK-only clamp, which would have
      hidden the gap from every other caller of `flux2`/`s3dit`). Neither
      backend's own capability manifest declares this as decorative:
      `crates/flux2/src/caps.rs` and `crates/s3dit/src/caps.rs` each already
      independently declared `.max(150.0)` on the `steps` `ParamSpec` --
      `capability::ActionSpec::validate` never reads `ParamSpec::min`/`max`
      (only discovery UIs do), so that ceiling was never actually enforced.
      `150` is therefore not invented here; it is the one number both
      backends had already converged on. `flux2::pipeline::MAX_STEPS` and
      `s3dit::pipeline::MAX_STEPS` (both `pub const 150`) are now the single
      source of truth each crate's own `caps.rs` reads via
      `crate::pipeline::MAX_STEPS as f64` instead of a repeated literal, and
      each backend's own `generate` path refuses a request over the ceiling
      BEFORE the sampling loop: flux2's `generate_batch_on` per-request (same
      `out[i] = Err(..); continue` convention `encode_image` failures already
      use), s3dit's new `check_steps` free function (mirrors the sibling
      `check_cap_len`/`fit_caption` shape, called from `HotPipeline::generate`
      right after the existing `.max(1)` floor). Proven with
      `an_absurd_step_count_is_refused_before_the_sampling_loop` (flux2, via
      the existing `Stub` denoiser harness -- asserts the stub's sampling
      loop never ran) and `an_absurd_step_count_is_refused_by_name` (s3dit,
      a pure unit test of `check_steps`). `capability::ActionSpec::validate`
      itself still does not generically enforce `min`/`max` for every
      `ParamSpec` -- a separate, cross-cutting gap affecting every model with
      a ranged param, not scoped into this fix.
- [x] Prompt-length handling is now CONSISTENT between the two backends this
      one SDK type fronts: fixed in flux2 directly (not with an SDK-only
      translation, which would have left direct `flux2` callers with the old
      silent-truncation behavior). `Pipeline::encode_prompt` used to
      `eprintln!` a warning and `ids.truncate(cfg.txt_len)` - conditioning on
      a PREFIX of the user's prompt (audit F18) - and now returns
      `Result<Vec<f32>, String>`, refusing via the new
      `check_prompt_length(token_count, txt_len)` free function (mirrors
      `s3dit::pipeline::check_cap_len`/`fit_caption`'s own shape) exactly the
      way s3dit's `fit_caption` already refused past `cap_len`. Both call
      sites in `generate_batch_on` (the real prompt and the CFG-uncond empty
      one) propagate the error through the same `out[i] = Err(e); continue`
      per-request convention `encode_image` failures already used, so a
      caller now gets the SAME `brain::Error::Backend` shape for an overlong
      prompt regardless of which backend resolved. Confirmed no caller
      outside `crates/flux2/src/pipeline.rs` called `encode_prompt` directly
      (grep across `crates/cli`, `crates/sdk`, `crates/flux2/tests`) and no
      test exercised the truncate-and-continue path, so this was a contained
      change: the `Denoiser` trait declaration, its one production impl, and
      three test-stub impls, plus the two call sites - no ripple into
      `crates/sdk`/`crates/cli`, both of which already propagate `Result`
      end to end. Proven with `an_overlong_prompt_is_refused_not_truncated`
      (a pure unit test of `check_prompt_length`, mirroring
      `an_absurd_step_count_is_refused_by_name`'s shape on the s3dit side).

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
  -- verified, not assumed, by reading both call paths. `steps` is now
  bounded in both backends themselves (`flux2::pipeline::MAX_STEPS`/
  `s3dit::pipeline::MAX_STEPS`, see "Not yet done" above for detail) rather
  than with an SDK-only clamp, so the fix also covers `flux2`/`s3dit`
  callers who bypass this crate entirely. Prompt length is now bounded
  CONSISTENTLY: fixed in flux2's own encode path (`Pipeline::encode_prompt`
  now refuses via `check_prompt_length`, matching s3dit's pre-existing
  `fit_caption` refusal, see "Not yet done" above for detail) rather than
  patched over in this facade, so the fix covers direct `flux2` callers too.
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

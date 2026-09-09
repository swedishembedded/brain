# pulid - roadmap

PuLID-FLUX identity conditioning (`crates/pulid`): the `IDFormer` Perceiver
resampler (face embedding → 32 ID tokens) and the injected
`PerceiverAttentionCA`, cross-attended into the FLUX.1 image stream at 20
sites. Composes `clip::EvaVision` and `flux1::{Flux, inject}`; adds no kernel
and no shared block.

Parity-gated on both backends against a hooked reference - IDFormer 29 taps,
the CA unit 8, and the conditioned FLUX.1 forward 10, worst 1−cos 1.44e-11.
The image → `id_cond` path exists (`idcond::IdCond::from_image` /
`idcond::compose`), and the serving contract is met: `pulid::caps`
(`text2image`), `resident_pulid::PulidResident`, a `catalog.rs` entry, D-Bus
`Run`, `examples/imagegen/pulid_generate.py`.

## Not yet done

- [x] Reference-grade face preprocessing - **`crates/bisenet`**, a new crate:
      the real official `parsing_bisenet` weights (ResNet18 context path +
      2 `AttentionRefinementModule`s + `FeatureFusionModule`, 19-class
      output), imported via a re-serialization of facexlib's release
      (`bisenet::import`'s module docs explain why - the released `.pth` is
      torch's pre-1.6 legacy pickle format), plus the SAME FFHQ-512 5-point
      alignment template and `grid_sample` warp `arcface::align` uses at its
      own 112px template (`bisenet::align`), plus the background-whiten/
      face-grayscale mask (`bisenet::mask`). `pulid::caps::Bundle::
      face_embeds` now runs the full chain - SCRFD's own landmarks (already
      computed for ArcFace's alignment) realigned to 512px, parsed, masked,
      THEN bicubic-resized to EVA-CLIP-L/336 - replacing the old plain
      resize. **Verified against real `facexlib` output on a real photo**
      (`crates/bisenet/tests/parity.rs`): the BiSeNet forward itself is
      cosine 1.0000000000 / 100.0000% per-pixel class agreement; the full
      align+mask chain (which also carries the warp-interpolation/padding
      divergence `arcface::align`'s own doc already documents at its
      template) is cosine 0.9999958920 (alignment) and 0.9997812905 (the
      final mask, where a hard argmax boundary pixel flip from that small
      alignment difference shows up as a large per-pixel delta despite the
      images being visually identical - not a bug). Needs `BRAIN_BISENET_DIR`
      (a directory holding `parsing_bisenet.safetensors` -
      `tools/goldens/pulid_face_parsing_dump_reference.py` produces one from
      a real `facexlib` install, converting the legacy-format release once).
- [x] `start_step` - `Flux1::generate_injected` now takes `GenerateOptions
      .start_step` (steps before it forward WITHOUT injection, identical to
      `inject: None`); `pulid::caps`'s `text2image` exposes it (default 0 =
      every step, matching the prior always-inject behavior). Still open:
      sweeping it (and `id_weight`, whose default/range now match upstream's
      own 1.0/0..3 rather than an unexplained 0.8/0..2) against a real
      identity-fidelity metric - `examples/imagegen/identity_score.sh`'s
      ArcFace-cosine method is the tool, no real run has swept it yet.
- [ ] Full-depth conditioning run across all injection sites - only a
      reduced-depth run has been exercised. **The real FLUX.1-dev checkpoint
      landed in this workspace for the first time and two real, pre-existing
      defects surfaced on first contact** (neither caused by this session's
      changes - both block a first real generation from completing at all,
      independent of BiSeNet):
      1. `data::unigram::UnigramTokenizer` does not implement SentencePiece's
         `Precompiled` normalizer (`unigram.rs`'s own module docs and a test
         already document this as a deliberate, known gap - reimplementing
         sentencepiece's normalizer + its protobuf charsmap is a separate,
         substantial undertaking, not attempted here). FLUX.1-dev's released
         `tokenizer_2/tokenizer.json` (T5-XXL) uses exactly this normalizer,
         so `brain flux1 text2image` fails at TOKENIZATION, before any
         denoising step - this blocks EVERY flux1/pulid generation with the
         real released tokenizer, not something specific to identity
         conditioning.
      2. Independent of (1): `pulid::caps::Bundle::load` panics inside
         `clip::model::EvaVision::new_on` (a wgpu bind-group validation
         error, "3 bindings vs a 5-binding layout") when built with real
         weights. Diagnostic instrumentation confirmed this happens BEFORE
         `bisenet::align::norm_crop_512` (or any other BiSeNet code) ever
         runs - `EvaVision::new_on` merely uploads `ParamStore` weights, no
         kernel dispatch of its own, so this is most likely a wgpu
         asynchronous-error-reporting artifact surfacing a fault from an
         EARLIER dispatch (elsewhere in `Bundle::load`'s sequence of
         `gpu.new_like(...)` calls sharing one ambient device) at the next
         GPU operation, not a bug in `EvaVision` itself. Not root-caused
         further - `BRAIN_NO_KERNEL_UPGRADE=1` and swapping BiSeNet's
         `new_like` for a fresh `Gpu::new` both leave it unchanged, ruling
         out the kernel-upgrade table and BiSeNet-specific device sharing as
         the cause.
      Neither defect touches this crate's own parity-gated code (IDFormer/CA/
      FLUX forward, or `crates/bisenet`'s own bit-exact-verified graph) -
      both are pre-existing infrastructure gaps this workspace's first real
      end-to-end run happened to be the first thing to reach.
- [x] Multi-image identity conditioning - `pulid::caps::text2image` accepts
      `face_image` (primary, required) plus up to three optional
      `face_image1/2/3` auxiliary photos, mean-pooled per-representation
      (raw ArcFace embedding, EVA-CLIP CLS, each of the 5 taps) before the
      one `idcond::compose`/`IdFormer` call. Explicitly NOT a transcription
      of upstream PuLID v1.1's own fusion algorithm (no source/reference for
      that in this workspace) - a real, useful capability, not a parity
      claim for the >1-image case. Single-image case is unaffected/still
      parity-gated.
- [x] True CFG - `flux1::pipeline::GenerateOptions.true_cfg` runs the second,
      un-injected forward on a negative prompt's own conditioning and
      combines `neg + scale*(pos-neg)`, gated by `cfg_start_step`; both
      `flux1::caps` and `pulid::caps` expose `negative_prompt`/`true_cfg`/
      `cfg_start_step` action params (0 = off, the default). No reference
      dump for true CFG specifically exists in this workspace, so this is
      NOT parity-gated against upstream's own true-CFG branch - a real,
      structurally-correct implementation of the documented formula, not a
      verified-equivalent one.
- [x] `FLUX.1-Krea-dev` variant wired (`Flux1Config::krea_dev`, byte-identical
      to `dev` per BFL's own release) into both crates' variant enums.
      **Still NOT a validation claim** - neither crate has the Krea-dev
      checkpoint or a reference dump; this only makes the variant name
      accepted and architecturally plausible, same honest status
      `kontext-dev`/`schnell` already carried before this.
- [ ] Backward pass / gradient check for the adapter (`check_pulid`)
- [ ] Batch > 1 - serial `run_batch`, same reason as `flux1`'s (one
      multi-step sample per request)

## Explicitly out of scope for this crate

**PuLID v1 / v1.1 on SDXL is a separate model, not a missing feature of this
one.** Upstream's SDXL branch is its own attention-processor/IDFormer
integration (community-checkpoint support, up to three auxiliary identity
images, its own CFG/`num_zero`/orthogonalization/sampler options) built on
`crates/sdxlunet`, not on `crates/flux1`. If it is ever wanted, it is a new
crate with its own import/parity ladder and its own roadmap entry under this
repo's normal porting discipline - not an extension of `crates/pulid`, which
stays FLUX.1-only.

The current forward pass reuses buffers across layers in an inference-shaped
way, so a backward pass needs a distinct training-mode forward with per-layer
buffer allocation rather than a flag on the existing one. That refactor is
the prerequisite for `check_pulid`, not an afterthought to it.

Only `dev` is validated against a PuLID reference, and no reference dump of a
full ID-conditioned *generation* exists in this workspace - so "PuLID works"
end to end is not a claim this crate supports.

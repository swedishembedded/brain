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
- [x] Full-depth conditioning run across all injection sites - the real
      FLUX.1-dev + PuLID-FLUX checkpoints landed in this workspace for the
      first time and THREE real, pre-existing infrastructure defects
      surfaced in sequence on first contact (none in this crate's own
      parity-gated code - IDFormer/CA/FLUX forward and `crates/bisenet`'s
      bit-exact-verified graph were never at fault). All three are now
      fixed:
      1. `data::unigram::UnigramTokenizer` did not implement SentencePiece's
         `Precompiled` normalizer (plus two other steps in the same
         `Sequence` - `Strip` and a `" {2,}"` `Replace` whose content is
         `U+2581`, not a space). FLUX.1-dev's released `tokenizer_2/
         tokenizer.json` (T5-XXL) uses all three, so tokenization failed
         before any denoising step. Fixed by a faithful port of HF's
         `spm_precompiled`, verified 27/27 exact token-id parity against
         real HuggingFace `tokenizers` output on the real tokenizer.
      2. `pulid::caps::Bundle::load` built the EVA-CLIP vision tower on
         `clip::model::TEXT_PIPELINES` instead of `VISION_PIPELINES` - every
         index still resolved to a real (wrong) kernel, and the first arity
         mismatch crashed wgpu's `create_bind_group` ("3 bindings vs a
         5-binding layout") inside `EvaVision::new_on`. Fixed by resolving
         `EvaVision`'s kernels BY NAME instead of positionally; bind groups
         are now labelled with their kernel's name so this class of error
         self-identifies.
      3. `Flux1::n_max` sized the DiT for image tokens only ("no txt/refs
         headroom yet"), but the forward asserts `nt + ni <= n_max` over the
         JOINT sequence - the first real T5-XXL-conditioned forward panicked
         "sized for 1024 joint tokens, got 1536". Fixed by adding the new
         `flux1::pipeline::MAX_TXT_LEN` constant to `n_max`, which both
         crates' `caps.rs` now also use for their `max_len` param's upper
         bound instead of an independently hardcoded `512`.
      4. `Bundle::load`'s identity-extraction stack (ArcFace/BiSeNet/
         EVA-CLIP/IDFormer) built its `Gpu` unscoped, landing on whatever the
         ambient default device was rather than an explicit home - on a
         2-card box this coincided with "te" (a ~21 GiB fp32 T5-XXL), and
         the two together OOM'd. Fixed by pinning this stack to "dit"'s home
         via the same `Homes` `plan_flux1` already computed (real headroom:
         an int8 DiT + resident `PulidCa` is ~17.5 GiB against 24).
      With all four fixed, `brain pulid text2image` now runs a real
      generation end to end for the first time: FLUX.1-dev `dev` + PuLID
      v0.9.1, 512x512, 8 steps, int8 DiT on one P40, fp32 T5-XXL on the
      other. Measured identity fidelity against the source photo (ArcFace
      cosine, `examples/imagegen/identity_score.sh`): **0.5107** - per that
      script's own documented anchors, "the same person to a human, most of
      the time," not a coincidence and not "family resemblance." This is
      the first real, numeric confirmation that PuLID's identity injection
      does something in this workspace, not just that the glue compiles.
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

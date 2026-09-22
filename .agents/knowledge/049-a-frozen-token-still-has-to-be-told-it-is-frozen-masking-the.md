<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 49. A frozen token still has to be TOLD it is frozen - masking the output is only half the mechanism

Image conditioning on a real 22B video DiT (`--start-frame`/`--end-frame`:
hold a real still fixed at the clip's first and/or last pixel frame and
generate the motion in between) produced a video whose conditioned frame was
a flawless reproduction of the source image and whose every OTHER frame was
a fine horizontal-line mesh texture with no recognizable structure. The
same prompt, seed, resolution and schedule with NO image conditioning
produced clean, coherent, temporally consistent video. The token geometry
had already been derived from the reference line by line and unit-tested:
positions (`VideoConditionByKeyframeIndex`'s `get_pixel_coords` + `+=
frame_idx` + single-pixel-frame narrowing), the frozen `denoise_mask`, the
`clean_latent` content, the append-vs-overwrite split between `frame_idx ==
0` and `frame_idx != 0`. All of it was correct.

**The bug:** the denoise loop fed the model `vec![sigma; t]` - the
schedule's current sigma broadcast to every token - as its timesteps. The
reference builds `Modality.timesteps` as `timesteps_from_mask(denoise_mask,
sigma)`, i.e. the per-token *product*. A frozen conditioning token holds
clean VAE content and therefore has timestep `0`, not `sigma`.

**Why that is catastrophic rather than cosmetic.** The DiT's AdaLN builds
one shift/scale/gate row PER TOKEN out of that per-token timestep. A token
carrying clean image content but labelled "as noisy as everything else" gets
modulated as if it were pure noise, and self-attention then mixes that
mis-modulated token into every genuinely-noisy token in the sequence. The
conditioning block, meant to be the strongest signal in the clip, becomes
the strongest *corruption* in it.

**Why the symptom pointed away from the cause.** `post_process_latent`
re-pins frozen tokens to `clean` after every step regardless, so the
conditioned frame decodes perfectly no matter how badly the model was
misled about it. "The frame I pinned is perfect, everything else is
garbage" reads as *the conditioning works and the model cannot cope with
what is left* - a capacity story (too few latent frames, too much temporal
compression, an over-constrained loop) that is entirely plausible, entirely
wrong, and expensive to disprove by generating at larger shapes.

**A second bug of the same family, same commit.** The append path rebuilt
`keyframes_mask` from scratch as all-zeros over the whole (base + appended)
sequence. The reference's `extend_keyframes_mask` *extends* the existing
mask: the appended conditioning tokens are `marked=false`, but the base
video keeps `_first_frame_keyframes_mask`'s unconditional first-latent-frame
ones. Rebuilding instead of extending meant turning image conditioning ON
also silently turned OFF a positional-embedding term the same clip had in
the unconditioned path - a per-feature regression of the very fix in #48.

**Consequences worth keeping:**

* **Conditioning has two halves: what the sampler does with the output, and
  what the model is told about the input.** Freezing a token by masking the
  x0 estimate makes the token come out right; it does nothing to stop that
  token from poisoning everything else. When porting any masked/partial
  denoising feature, find every place the reference derives something from
  `denoise_mask` - not just the one that writes the latent back.
* **"The conditioned part is perfect, the generated part is broken" is
  evidence about the model's INPUTS, not about its capacity.** A pinned
  output is pinned by the sampler and tells you nothing about whether the
  model understood it. Resist the capacity hypothesis until the per-token
  inputs have been diffed against the reference field by field.
* **A per-token tensor passed as a scalar-broadcast is invisible to shape
  checks and to every weight-free geometry test.** `vec![sigma; t]` has the
  right length, the right dtype and the right values on the code path that
  had existed until then (nothing frozen). Where a reference computes a
  tensor as a product of two things, port the product - not the
  simplification that happens to be equal while the other factor is all
  ones.
* **When an append-style conditioning item builds a new state, audit every
  field it reconstructs.** The reference's items use `dataclasses.replace`
  or explicit `extend_*` helpers precisely because rebuilding a field from
  scratch silently discards whatever earlier stages put there. A field the
  port rebuilds as a constant is a field the port has stopped tracking.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 50. A pixel-delta gate with no calibrated threshold passes on decode dither, and "the conditioned clip does not move" turned out to be the request rather than the bug

`--start-frame X --end-frame X` on the real 22B distilled checkpoint - the
same still pinned at both ends, i.e. a seamless loop - produced a clip in
which every frame is the pinned still. The dog is in the identical
fully-extended pose in all 25 frames; the flying card does not move a pixel.
The immediately preceding fix (#49) had verified itself with "mean
consecutive frame delta went from 14.77 to 2.64-4.59", read as garbage ->
coherent. It was neither.

**The metric was the first bug.** Mean consecutive-frame delta on this
pipeline, 384x192, 9 frames: a visually frozen clip scores **2.2**, a real
one **5.4**. VAE decode dither on static content is not zero, so "nonzero"
proves nothing, and 2.4x reads as "less motion, but motion". The metric that
separates them is **peak excursion from frame 0**, `max_i mean|frame_i -
frame_0|`: the same two runs score **5.1** and **26.6**. A static clip cannot
accumulate - its wobble is uncorrelated noise around one image - while real
motion walks away from frame 0 and stays away. (Not `|frame_last - frame_0|`
either: a loop is supposed to come back.) `crates/ltxv/tests/motion_real.rs`.

**Then localize by neighbouring request, not by reading the port.** Sixteen
real-weight runs, one variable each, peak excursion, seed 42:

| request | 384x192 9f | 384x192 25f | 640x320 25f | 640x320 49f |
|---|---|---|---|---|
| text-to-video, no stills | 26.6 | 18.1 | 20.5 | - |
| one still at frame 0 | 35.1 | - | - | - |
| two DIFFERENT stills | 35.2 | - | 40.0 | - |
| the SAME still at both ends | 4.6 | 32.4 | 7.3 | 7.2 |

Everything at or above 18.1 visibly animates; everything at or below 8.8
reproduces the pinned still in every frame. One row is static. The engine
generates motion from nothing, from one anchor, and between two different
anchors - at the SAME shape, seed, prompt, sampler and conditioning code.
That row also survives every lever the reference exposes: `strength`
1.0/0.8/0.5 (4.6/4.8/5.0), the deterministic and the ancestral sampler
(4.6/5.1, and 7.3/8.8 at 640x320), the in-place and the appended frame-0
conditioning mechanism (5.1/4.6), a CRF-33 re-compressed conditioning still
(8.4 vs 7.3), and doubling the clip length (7.2). The decisive control is
the third column: changing ONLY the end still from "the same image" to
"that image mirrored" moves the score from 7.3 to 40.0. "Start at this
image and end at this same image" has a correct trivial solution and the
model returns it. Five plausible root causes, each with a real reference
citation, each refuted by one run.

**Real port defects the hunt did find, none of which was the reported
symptom:**

* **A refusal inherited from the wrong reference function.** `denoise`
  asserted `eta == 0` whenever any token was conditioned, on the grounds
  that a frozen token's ancestral renoise term needs a per-token sigma. That
  is true of `ltx_pipelines.utils.samplers._inject_sde_noise` (the res2s
  loop, which really does build `stack([timesteps_from_mask(mask, sigma),
  timesteps_from_mask(mask, sigma_next)])`) and false of
  `_ancestral_euler_denoising_loop`, which is what
  `euler_ancestral_denoising_loop` - and therefore LTX-2.5 distilled's own
  stage 1 (`ltx_pipelines.distilled`: `ANCESTRAL_SAMPLER_SINCE_VERSION =
  (2, 5)`, `ANCESTRAL_ETA = 1.0`) - actually runs. That loop steps the whole
  latent with the SCALAR schedule and re-applies `post_process_latent` to
  the STEPPED result. The port had deleted the checkpoint's own sampler from
  the conditioned path to satisfy a constraint that path never had.
* **The two loops apply the conditioning mask in different places.**
  `euler_denoising_loop`/`_step_state` masks the x0 ESTIMATE before
  stepping; `_ancestral_euler_denoising_loop` masks `x_next` AFTER the
  renoise term (`if draw_noise: x_next = post_process_latent(...)`) and
  short-circuits the terminal sigma to the raw estimate with no mask at all.
  Porting one ordering and using it for both leaves freshly injected noise
  sitting on a token that is supposed to be clean.
* **The wrong conditioning builder for interpolation.** `--start-frame` +
  `--end-frame` is `ltx_pipelines.keyframe_interpolation`, which uses
  `helpers.image_conditionings_by_adding_guiding_latent` - every image,
  `frame_idx == 0` included, becomes an APPENDED
  `VideoConditionByKeyframeIndex`. The port reused
  `combined_image_conditionings`' image-to-video branch, whose
  `frame_idx == 0` special case (`VideoConditionByLatentIndex`) OVERWRITES
  the generated video's own first latent frame. At 121 frames that freezes 1
  latent frame in 16; at 9 it freezes half the clip.
* **A required reference argument hardcoded to its most extreme value.**
  `ImageConditioningInput.strength` has no default upstream -
  `--image PATH FRAME_IDX STRENGTH` requires all three and the help text's
  own example is `0.8`. The port pinned it to `1.0` with no knob.
* **A mandatory preprocessing step missing entirely** (still missing, and
  measured not to matter for motion): `helpers.load_image_and_preprocess`
  runs `preprocess(image, crf)` - an H.264 re-encode/decode round trip at
  `DEFAULT_IMAGE_CRF = 33` - before the resize, "to match the compression
  the model was trained on", and raises rather than defaulting when `crf` is
  unresolved. The port feeds the VAE a pristine PNG. It also resizes with
  `resize_exact` (aspect-distorting, Lanczos) where the reference does
  aspect-preserving `resize_and_center_crop` (bilinear).

**Consequences worth keeping:**

* **Calibrate a threshold against a known-BAD run, not only a known-good
  one.** A floor picked from the good run alone has no idea how close the
  bad run gets. Both numbers, from real weights, or the gate is decoration -
  and record them in the test so the next person can see the separation
  instead of re-deriving it.
* **A generative "this looks wrong" is localized fastest by the nearest
  request that looks RIGHT.** Vary one input at a time - drop a
  conditioning, change one anchor, change the shape - and let the boundary
  fall out. Reading the port for a bug that is not there costs more than the
  GPU time; here the decisive evidence was one run with two different stills
  instead of one still twice.
* **A refusal is a feature deletion, and it inherits its justification from
  whichever reference function was open at the time.** When a port refuses
  a combination the reference supports, check that the constraint belongs to
  the code path the reference actually takes for that combination.
* **When the reference makes an argument REQUIRED, the port must not pick a
  value.** A required argument means upstream believes there is no safe
  default. Hardcoding one - especially the extreme end of its range - turns
  a user decision into an invisible property of the port.
* **Look at the frames.** Nine evenly spaced PNGs read by eye settled in one
  minute what the pixel-delta number had been arguing about for a whole
  session.

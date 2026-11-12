# LTX-2.5 (audio + video generation)

LTX-2.5 turns a text prompt into a short clip **with synchronized audio**: a
single diffusion transformer denoises a video-latent stream and an audio-latent
stream together, coupled every block by cross-attention between the two, rather
than generating picture and sound separately and muxing them afterward. It is
Lightricks' second video model in brain after [wan](wan.md), and structurally the
harder one - two token streams instead of one, per-token (not per-request)
timesteps, and a 12B text encoder rather than an 11 GB one.

`ltxv` names the whole upstream family the way `wan` does: LTX-2.3/2.4/2.5 share
one Hugging Face class (`AVTransformer3DModel`) and one GGUF architecture tag
(`ltxv`); which release you have is a configuration, not a different model.

**Status: the real 22B checkpoint generates video, sound, or both, from
text, stills, a soundtrack or an existing clip.** With the checkpoint in the
model store, `brain ltxv t2v --prompt ...` runs the real distilled DiT
(streamed, int8, device-resident), the real Gemma-4 text encoder and the real
VAEs at the reference pipelines' own defaults; without it, the same command
falls back to a tiny random-weight DiT that proves the wiring and nothing
else. `.agents/roadmap/ltxv.md` is the milestone ledger, with parity numbers
per component and the open gaps.

## Support

| Capability | Supported |
|---|---|
| Text to video | [x] `brain ltxv t2v` / the `t2v` action. Unnamed settings take the reference defaults (768x512, 121 frames, 24 fps) when the checkpoint is configured; a clip past one denoising window rolls a latent context across windows, past the single-stage ceiling it runs the reference's two stages |
| Text to video + audio | [x] `--audio` / `audio: true`: both streams denoised jointly, 48 kHz stereo through the vocoder's bandwidth extension, any length (see "Generating sound") |
| Text to audio | [x] `brain ltxv t2a` / the `t2a` action: the audio stream alone, the video half never read |
| Audio to video | [x] `--audio-source <wav>`: the soundtrack held frozen while the picture denoises against it, any length (each window holds its own slice) |
| Retake, video to audio | [x] `brain ltxv v2v --window S:E` / the `v2v` action: regenerate a time window of the picture, the sound or both; `--hold-video --audio` generates sound for a picture |
| Masked video to video | [x] `brain ltxv v2v --mask <maskseq>`: regenerate a tracked region, hold the rest token-exactly; `--keyframe F:<image>` guides it with edited stills (see "Replacing a person") |
| Character replacement | [x] `brain ltxv replace`: a clip, a character sheet and a click on a person in; the clip with the person replaced out (see "Replacing a person") |
| Extension | [x] `brain ltxv extend` / the `extend` action: continue a clip from its last 57 frames, with its soundtrack |
| Image / keyframe conditioning | [x] `--start-frame`, `--keyframe` (any number), `--mid-frame`, `--end-frame` (see "Anchoring a clip on real images") |
| Clip length from the prompt | [x] the duration head, when it is in the model store (see "Clip length") |
| Guidance | [x] the reference's multimodal guider: CFG, STG, rescale and isolated-modality guidance per stream, `--guidance-preset reference` (see "Guidance and samplers") |
| Samplers | [x] ancestral Euler (the distilled pipeline's) and res_2s (the reference's high-quality pipeline's), `--sampler` |
| Post-hoc upscale | [x] `brain ltxv upscale` / the `upscale` action |
| DFR | [x] `brain ltxv dfr` on the real checkpoint: keyframe-segment canvas, stage 1 with generated keyframe slots, spatial upscale, the detailing pass under the Refine-Details IC-LoRA, 0-2 temporal rounds and the spatial epilogue (`--spatial-upscalings 2`) |
| LoRA inference | [x] `--adapter <path>` (repeatable): ComfyUI/PEFT `.safetensors` from the official trainer, published IC-LoRAs (with `--reference-video`), or brain-trained adapters, applied beside the int8 weights |
| LoRA / full fine-tune | [~] video-only DiT; the audio-visual DiT has no training support |
| CLI | [x] `brain ltxv {t2v,t2a,v2v,replace,extend,upscale,dfr}` |
| Capability actions (D-Bus, `brain do`) | [x] `t2v`, `t2a`, `v2v`, `extend`, `upscale`, `dfr`; the clip actions take the source as a `video` input blob and its sound as a `soundtrack` WAV blob |
| HTTP API | [ ] the OpenAI/Anthropic routes have no video route |
| Multi-device | [~] the CFG pair runs on two cards concurrently; block sharding across cards is implemented for the video-only DiT |
| NPU | [ ] out of scope for a model this large |

## Architecture, in brief

Two independent token streams, video (4096-dim, 32 heads x 128) and audio
(2048-dim, 32 heads x 64), 48 shared-depth blocks. Per block: self-attention,
then a *separate* cross-attention into Gemma-4 text features (SDXL/Wan topology,
never a concatenated joint sequence), then bidirectional audio<->video
cross-attention, then a plain GELU-tanh feed-forward - no MMDiT-style joint
attention anywhere. Modulation is PixArt adaLN-single, but **per-token**: every
latent cell carries its own timestep (`denoise_mask * sigma`), which is the
mechanism behind image/keyframe conditioning and multi-shot generation in one
forward pass rather than a separate code path. RoPE is fractional (position
normalized to `[-1,1]` by an axis-specific max), GPT-NeoX split/rotate-half
layout, video using 3 axes (frame/height/width) and audio 1 (time in seconds) -
the two streams' cross-attention shares a common time-only RoPE space so a video
frame and an audio window at the same moment attend to each other correctly.

The video VAE is causal 3D convolution at (8, 32, 32) stride down to 128 latent
channels, replicating (not zero-padding) the first frame across every temporal
receptive field. Two decoders ship for it: a conventional conv mirror, and a
"diffusion decoder" that is itself a small 3D neighborhood-attention transformer
with no convolutions at all, denoised in 1-2 steps. The audio VAE runs on
log-mel spectrograms (not raw waveform, not complex STFT) through 2D causal
convolution to an 8-channel continuous latent; a BigVGAN-style vocoder with
snake-beta activations turns that back into 16 kHz audio, then a bandwidth-
extension stage lifts it to 48 kHz.

See `.agents/roadmap/ltxv.md` for exact dimensions, the settled convention
questions (chunk orders, norm epsilons, padding modes - the kind of detail that
is easy to get subtly wrong porting a two-decoder, two-VAE, two-stream model),
and the current gap list.

## Getting the weights

Model id: `brain/ltxv`. The default checkpoint is
`Lightricks/LTX-2.5` on Hugging Face, gated - fetching it requires a Hugging
Face account with access accepted and `hf auth login` run first. It ships as
several independent files rather than one directory-style checkpoint. Put
them in the model store and `brain ltxv` finds each one by its content: the
DiT GGUF, the video and audio VAEs, the Gemma-4 text encoder (the file that
carries LTX's `text_embedding_projection`; its tokenizer is embedded in it) and
the spatial x2 latent upscaler. A role flag names a file outright when the
store holds more than one candidate:

```bash
brain ltxv t2v \
  --dit …/ltx-2.5-22b-distilled-transformer-Q8_0.gguf \
  --vae …/ltx-2.5-video-vae-conv-bf16.safetensors \
  --text-encoder …/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --spatial-upsampler …/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors ...
```

The DiT alone is 42 GB at bf16 and the text encoder 26 GB - together they need
roughly 70+ GB of addressable memory just to hold weights, before activations.
Confirm your hardware can hold what you point at before fetching it.

An LTX GGUF (`general.architecture = "ltxv"`) converts with `brain import
<file>`; it carries the full architecture config as GGUF metadata, so no
separate config file is needed. The VAEs, text encoder and tokenizer still come
from the repository above.

## Running it

```bash
brain ltxv t2v --prompt "a cat walking along the shoreline at dusk" --output-path cat.mp4
```

Every weight comes from the model store. Settings the command leaves unnamed
take the reference pipelines' defaults when the real checkpoint is
configured; `--dit-config tiny` (or a store without the checkpoint) runs the
random-weight smoke test instead, which proves the wiring and carries no
semantics. `--steps` and the schedule flags apply to a non-distilled
checkpoint; the distilled one has a fixed schedule.

### Guidance and samplers

The distilled checkpoint denoises with one conditional forward per step,
which is the default. `--guidance-preset reference` selects the reference
pipeline's non-distilled guidance (CFG 3 video / 7 audio, STG 1 on block 28,
rescale 0.7, isolated-modality guidance 3), `--guidance-preset hq` its
high-quality preset (no STG, rescale 0.45 video / 1.0 audio, run with
`--sampler res2s`), and each term has its own flag
per stream (`--guidance`, `--stg`, `--stg-blocks`, `--rescale`,
`--modality-guidance`, and their `--audio-*` counterparts). The terms combine
on the x0 estimate exactly as `MultiModalGuider` does: STG bypasses
self-attention in the named blocks, the isolated-modality pass cuts the
audio/video cross-attention. CFG guides against `--negative-prompt`, which
defaults to the reference pipelines' own list of artifacts; it is encoded
only when some CFG scale is not 1.

`--sampler res2s` runs the reference's second-order res_2s loop: two guided
evaluations per step (the second at the geometric-midpoint sigma), SDE noise
at both, and a final evaluation that removes the last noise. `--eta` is each
sampler's noise strength, 1.0 for ancestral Euler and 0.5 for res_2s by
default.

### Clip length

When the request names no length and the duration head
(`ltx-2.5-duration-head-bf16.safetensors`) is in the model store, the head
predicts it from the prompt: it pools the prompt's connector outputs into a
length between one and twenty seconds, snapped onto the VAE's `8k + 1` frame
grid. `t2v` without `--frames` and `t2a` without `--seconds` both use it; the
prompt is encoded once either way. A video clip, with or without sound, is
predicted from both streams' connector outputs as every reference video
pipeline predicts it, and `t2a` from the audio stream's alone.

### Anchoring a clip on real images

A generation can be anchored to stills you supply at its first frame, at any
number of interior frames and at its last: `--start-frame`, `--keyframe
<frame>:<image>` (repeatable; `--mid-frame` is one keyframe at the clip's
midpoint, or at `--mid-frame-at`) and `--end-frame`. Each still is VAE-encoded
and appended to the sequence as a clean guiding block at its own instant -
held at sigma 0 (`denoise_mask = 0`, per-token timestep 0) - while every
token of the generated clip denoises freely around it. The clip moves
through each still and reaches it at its frame, rather than freezing on it or
cutting to it.

```bash
brain -v --device gpu0 ltxv t2v \
  --prompt "a fishing boat crosses the harbour mouth, camera tracking left" \
  --frames 121 --width 768 --height 448 --fps 24 --output-path boat.mp4 \
  --start-frame dawn.png --keyframe 40:turning.png --keyframe 80:midway.png \
  --end-frame open-sea.png
```

* **Any interior frame is a legal position; nothing is snapped to the `1 +
  8k` grid.** A guide carries its own RoPE position
  (`VideoConditionByKeyframeIndex.apply_to` adds `frame_idx` straight onto the
  time coordinate and divides by fps), not a slot on the generated video's
  latent grid. The `1 + 8k` rule constrains the clip's *length*. Keyframes
  must be in timeline order, strictly between the first and last frames; the
  resolved frames are printed in the run's first line.
* **`--mid-frame` without `--mid-frame-at`** lands on `(frames - 1) / 2`, the
  reference's own position for a single interior keyframe
  (`ltx_pipelines.utils.helpers.evenly_spaced_keyframe_positions(1, 121) ==
  [60]`).
* **One still at frame 0 and nothing else is still image-to-video**, which
  overwrites latent frame 0 in place. The moment a second still is given,
  every still becomes an appended guide instead, including the one at frame
  0. That is the reference's own split between `combined_image_conditionings`
  and `image_conditionings_by_adding_guiding_latent`.
* **Why a guide and not a pin.** Overwriting the latent slot that contains a
  keyframe's instant with the still, and freezing it there, freezes that
  slot's eight pixel frames and makes the clip jump into and out of the
  freeze. Measured on the real checkpoint at 768x448 x 97 frames with stills
  from an unconditioned run of the same prompt, the worst frame-to-frame step
  was 3.0x its neighbours with the pin, and 1.09x with the guide (1.11x for
  the unconditioned run itself). The guide also reached the still more
  closely (mean |delta| 2.28 at its frame, against 4.95 for the pin).
* **`--conditioning-strength`** (default `1.0`) applies to every still given.
* **Two identical anchors make a static clip.** Passing the same image to
  `--start-frame` and `--end-frame` asks for "start here, end here", which has
  a correct trivial answer; see `crates/ltxv/tests/it/motion_real.rs` for the
  measured table. Keyframes follow the same logic: anchor *different*
  instants with stills a plausible motion apart.
* **Long clips route every anchor to the window that owns it.** A clip
  generated as several windows (see below) conditions each keyframe in the
  window whose emitted frames contain it, and every earlier window carries the
  anchors still ahead of it as guides, so each window bends toward the next
  keyframe. The `t2v` capability action takes the same stills as
  `start_frame`, `keyframes` (`frame:path` pairs, comma-separated),
  `mid_frame`/`mid_frame_at` and `end_frame`.
* **Not supported for a multi-scene clip**: a scene boundary is a cut that
  carries nothing across, so a continuous shot with keyframes is one scene.

### Clips longer than one denoising window

`--frames` takes any legal `1 + 8k` length. Past what a single denoising window
holds, `brain ltxv t2v` generates the clip as several consecutive windows and
carries **the previous window's own last latent frames** across each boundary:
they are sliced out of the denoised latent before anything is decoded, and
frozen at sigma 0 (`denoise_mask = 0`, per-token timestep 0, re-pinned every
step) at the head of the next window while only the new frames get a denoising
schedule. It is the same freezing mechanism `--start-frame` uses for latent
frame 0, over N latent frames instead of one.

```bash
brain -v --device gpu0 ltxv t2v \
  --prompt "..." --frames 481 --width 1280 --height 704 --fps 24 \
  --output-path long.mp4
```

* **Why not just chain clips on their last frame.** Decoding a clip, taking its
  last RGB frame and re-encoding it as the next clip's `--start-frame` is
  continuous in *position* and discontinuous in *velocity*: a single frame
  carries no information about what was moving, in which direction, or how
  fast, so the model re-invents the motion at every seam - visibly as stutters,
  changes of direction, or motion running backwards. The rolling latent context
  carries 57 pixel frames of real motion history instead.
* **`--context-frames`** (default `57`) sets how much is carried, in pixel
  frames, and must be `1 + 8k`. 57 frames is 8 latent frames, which is the
  reference's own prefix size for temporal extension
  (`packages/ltx-trainer/configs/video_extend_lora.yaml`'s
  `temporal_boundary: 8`, whose validation samples spell the same number as
  `num_frames: 57`). It is *not* derived from the VAE's temporal receptive
  field, which is ±14 latent frames and includes lookahead a rolling window
  cannot have - see `crates/ltxv/src/longform.rs`'s module doc.
* **The context costs tokens.** A continuation window spends
  `context_latent_frames × lh × lw` of its budget before it generates anything,
  so a longer context means fewer new frames per window and more windows. The
  per-window ceiling is `ltxv::longform::LONGFORM_MAX_TOKENS` (13200, the
  largest single-window generation this crate has a recorded real run at),
  overridable with `--max-window-tokens`.
* **A request that fits one window is unchanged.** It is handed straight to the
  single-window path, bit for bit, and none of this runs.
* **The window-major loop is the default.** `--stage-major` runs a video-only
  clip's stage 1 at half resolution across the whole clip and then refines it
  at full resolution window by window; its refinement windows each regenerate
  their own slice and do not stitch (measured on the real checkpoint: a cut to
  another scene at a refinement seam), so it is opt-in. Audio-visual clips
  always use the window-major loop.
* **The windows share the clip's new frames evenly.** The fewest windows the
  token ceiling allows, each generating the same number of new latent frames
  (the first never fewer than the carried context). A continuation of only
  one or two new latent frames after its context only holds its history - the
  clip froze for its last third of a second before the split was evened.
* **`--audio` crosses the same seams**, on its own time grid and with its own
  constraint on where a seam may fall - see "Generating sound" below.
* **Anchors are routed per window**: `--start-frame` conditions the first
  window, each keyframe the window that owns its frame, `--end-frame` the
  last, and every window also carries the anchors still ahead of it as
  guides.

The same routing applies to the `t2v` capability action (`brain do brain/ltxv
t2v`), with the default context and no new action parameter.

### Several scenes in one clip

Everything above keeps **one shot** continuous, which is the wrong answer if the
clip is supposed to become something else part way through: every window shares
the prompt, and every continuation window is hard-conditioned on the real
content before it. Repeat `--scene <frames>:<prompt>` to write a clip that
changes.

```bash
brain --device gpu0 ltxv t2v \
  --width 768 --height 448 --fps 24 --output-path story.mp4 \
  --scene 121:"a fishing boat leaves a harbour at dawn, camera tracking" \
  --scene 121:"the open sea under heavy rain, waves breaking over the bow" \
  --scene 57:"a close-up of a gull on a wet railing"
```

One command, one file, 299 frames.

* **Inside a scene, nothing changes.** The rolling latent context above still
  carries across every window boundary, so a scene longer than one window is
  the same continuous shot it would be on its own.
* **At a scene boundary the context resets.** The next scene's first window
  carries `context = 0`, exactly like the first window of any clip, so it is
  free to be a different subject, setting or action instead of a forced
  continuation. A multi-scene run is the single-scene machinery run once per
  scene with nothing carried between the runs, and the decoded frames
  concatenated - so scene *n*'s pixels do not depend on scene *n-1* at all
  (gated:
  `crates/ltxv/tests/it/longform.rs::a_two_scene_request_is_one_clip_whose_second_scene_owes_nothing_to_its_first`).
* **It is also how a scene ends deliberately.** Long single-prompt
  autoregressive generation is documented to degrade - drift and error
  accumulation, and a strong clean history makes "copy the last frame" a cheap
  solution, so motion can go static. Cutting to a new scene on the caller's
  schedule is a way to not depend on how long one shot survives.
* **The separator is the first colon only**, so a prompt may contain colons.
  Each scene's frame count is its own `1 + 8k`; the clip's length is their sum.
* **`--scene` cannot be combined with `--prompt`/`--frames`** - those two *are*
  the single-scene spelling, and a run that mixed them would have two ways to
  say where the clip starts. Every seed is derived per scene, so two scenes
  never draw the same initial noise.
* **`--start-frame` conditions the first scene's opening only**, and
  `--end-frame`/`--keyframe`/`--mid-frame` are refused for a multi-scene clip:
  a scene boundary carries nothing across, so a keyframe belongs to a
  continuous shot.
* **Not exposed on the `t2v` capability action** - `--scene` is a CLI flag; the
  action still takes one prompt.

### Applying a LoRA adapter

`--adapter <path>` applies a LoRA to the transformer for the whole run, at the
strength set by the `--lora-scale` that follows it (default 1.0, ComfyUI's
`strength_model`). Repeat the pair to stack adapters; every `t2v`, `upscale`,
`v2v` and `dfr` run takes them.

```bash
brain -v --device gpu0 ltxv t2v \
  --prompt "..." --frames 121 --width 768 --height 448 --output-path out.mp4 \
  --adapter my-style.safetensors --lora-scale 0.8
```

* **The file format is the reference's own**: `diffusion_model.<module>.lora_A.weight`
  / `.lora_B.weight` pairs, as the official LTX trainer exports them and the
  published IC-LoRAs ship. The delta is `strength * B @ A`, exactly the
  reference loader's; an adapter brain trained records `alpha/rank` in its
  header and that multiplies in on top.
* **The checkpoint is never rewritten.** The real transformer runs at int8,
  and folding a trained delta into int8 weights rounds most of it away (it is
  typically a fraction of one quantization level), so every block linear
  applies the correction beside its packed weight from the fp32 activation.
  Targets outside the blocks (such as the embeddings connector's attention)
  are fp32 and fold exactly.
* **Every target must exist.** An adapter naming a tensor this model does not
  have, or one of the wrong shape, is refused before any weight is read - a
  run that silently skipped part of an adapter would look adapted and not be.
* LoKr adapters are refused: their delta has no low-rank form to apply beside
  a quantized weight.

**IC-LoRA reference videos.** An IC-LoRA (the published control, restoration
and transformation adapters) reads a *reference clip* beside the clip it
generates. `--reference-video <path>` supplies it: its first `--frames` frames
are VAE-encoded at the geometry the adapter declares in its own metadata
(`reference_downscale_factor`, `reference_temporal_scale_factor`) and appended
as clean tokens at their own positions, which the adapter learned to attend
across. `--reference-strength` (default 1.0) sets how strongly they are held.
`--reference-attention-strength` (default 1.0) sets how strongly the reference
and the generated clip attend to each other - 0 cuts the reference off - and
`--reference-attention-mask <video>` sets that weight per region and frame from
a grayscale video (white attends, black does not), scaled by the strength.

```bash
brain --device gpu0 ltxv t2v \
  --prompt "..." --frames 121 --width 1280 --height 704 --output-path out.mp4 \
  --adapter ltx-2.5-22b-ic-lora-depth.safetensors --reference-video depth.mp4
```

* It needs the IC-LoRA: the base checkpoint has no use for reference tokens,
  so a reference without `--adapter` is refused.
* A two-stage run follows the reference recipe: stage 1 runs in context
  (adapters and reference), stage 2 refines with the bare checkpoint.
  `--stage-2-ic-lora` keeps both on stage 2 as well.
* One window only: a clip long enough to need several windows, `upscale`,
  `v2v` and `dfr` refuse a reference rather than drop it.

### Upscaling a clip that already exists

`brain ltxv upscale` takes a rendered video file and re-renders it at twice the
spatial resolution. It is not a pixel-space resampler: the clip is VAE-encoded,
carried up by the **official LTX-2.5 x2 latent spatial upscaler**, refined by a
short diffusion pass at the new size, and VAE-decoded - the exact tail a
two-stage generation already runs (see the Stages section of
`brain ltxv t2v --help`), reached through the same code.

```bash
brain -v --device gpu0 ltxv upscale \
  --input clip.mp4 --output-path clip_upscaled.mp4 \
  --prompt "the prompt the clip was generated from"
```

Four things worth knowing before running it:

* **`--prompt` is not optional in practice.** The refinement step is a
  diffusion pass conditioned on text, so the model is answering "what should
  this look like with more detail" against whatever context it is given.
  Omitting the flag refines against an empty prompt and costs detail; the
  command says so on stderr when you do. A video file carries no record of
  the prompt it was generated from, so it has to be supplied.
* **The input's shape has to be VAE-representable**: width and height a
  multiple of 32, frame count of the form `1 + 8k`. Anything `brain ltxv t2v`
  wrote already satisfies both.
* **Length has a ceiling per refinement pass, and past it the passes carry a
  rolling latent context.** An upscaled clip carries four times the video
  tokens per frame that its input did, so a length that was fine at the source
  resolution need not fit at the target one. Past
  `ltxv::pipeline::REFINE_MAX_TOKENS` in one pass the clip is refined in
  several, planned by the same `longform::window_plan` a multi-window
  generation uses: each pass freezes the previous pass's own last
  `--context-frames` of *refined* latent at the head of its sequence, so the
  passes are one continuous clip. A refinement starts at sigma 0.909 - it
  keeps under a tenth of the content it is handed - so passes with no shared
  history do not merely step in fine detail, they come back as separately
  re-imagined clips.
* **That context costs pass budget, and at a dense grid it is capped.** A pass
  holds `REFINE_MAX_TOKENS / (tokens per latent frame)` latent frames and
  spends the carried ones before it refines anything. At 2560x1408 that is 3
  latent frames a pass, so the reference's 8-frame context cannot fit: the plan
  carries the most the grid allows (2) and says so on stderr. Lower
  `--context-frames` to buy back passes at the cost of continuity;
  `--context-frames 1` carries a single latent frame and costs about what an
  uncarried plan did. A clip that fits one pass is never split and carries
  nothing, and a grid with no room for one carried frame plus one new one is
  refused before any weight is read.

`--refine-steps` picks a SUFFIX of the distilled refinement table (default: all
3 steps), not a resampling of it - the distilled checkpoint only denoises
correctly at the sigma values distillation baked in, so fewer steps means
starting further down the same table rather than taking the same span in
bigger hops.

`brain ltxv dfr` runs DFR (Diffusion Fidelity Rendering) on the real
checkpoint. The clip is laid out on a canvas of whole 24- or 32-frame
segments; stage 1 generates it at half resolution together with one sharp
keyframe at every segment border, and stage 2 carries both up with the
spatial x2 upscaler and re-denoises at full resolution under the
Refine-Details IC-LoRA (strength 0.5), the half-resolution video as its
reference. The canvas is trimmed back to the requested length; with
`--audio` the soundtrack is stage 1's. It takes `t2v`'s sampling, guidance,
adapter and duration-head behaviour, and needs the
`ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors` IC-LoRA
(`Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details`) in the model store or
`--detailing-lora <path>`; the width and height must be multiples of 64.

`--temporal-upsample-rounds 1|2` then doubles the frame rate once or twice:
each round upscales the clip x2 in time with the temporal latent upscaler,
splits it into keyframe-seam tiles and densifies each - its seams held as
keyframes, a new keyframe generated mid-segment, a later tile starting on
the keyframe before its seam with the earlier tile's frames pinned, the
stage-1 soundtrack frozen beside it. A request of `N` frames comes back as
`(N - 1) * 2^rounds + 1` frames at `fps * 2^rounds`.

`--spatial-upscalings 2` adds one more detailing pass at the output size:
stage 1 runs at a quarter of it, the detailing pass and the rounds at half,
and the epilogue rebuilds every keyframe at the output size (decoded,
stretched x2 with Lanczos, encoded again), upscales the clip, and details it
window by window on the last round's seams - each window holding its
keyframes, with the half-size clip as its reference, the earlier window's
frames pinned, and spatial tiles of 32 latent cells (1024 px) blended after
every step on a side longer than that. The size must then be a multiple of
128:

```bash
brain pull https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details/blob/main/ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors
brain -v --device gpu0 ltxv dfr --audio --prompt "a cat walking on a beach" \
  --frames 49 --output-path out.mp4
```

### Generating sound

LTX-2.5 is one model that denoises a video-latent stream and an audio-latent
stream **together**, coupled every block by cross-attention. `--audio` runs
both; without it only the model's video half runs and the clip is silent.

```bash
brain -v --device gpu0 ltxv t2v --audio \
  --prompt "a blacksmith hammers glowing steel on an anvil, rhythmic clangs" \
  --frames 57 --width 1280 --height 704 --fps 24 --output-path forge.mp4
```

Every weight, the audio VAE and the text encoder included, comes from the model
store (see "Getting the weights").

* **The distilled checkpoint carries the whole model.** Two thirds of its
  tensors are the audio stream and the bidirectional A<->V cross-attention
  (`audio_attn1/2`, `audio_ff`, `audio_to_video_attn`, `video_to_audio_attn`,
  both `*_adaln_single` families, `audio_embeddings_connector`). Sound was
  never a missing-weights problem; it was a wiring one.
* **The text conditioning is per stream.** The encoder checkpoint carries
  `text_embedding_projection.{video,audio}_aggregate_embed` side by side and
  each stream's own embeddings connector is built for its own head's width, so
  one text-tower forward produces two projections. A real
  `--text-encoder` is therefore required: the stub context has nothing to
  project.
* **Alignment is arithmetic, not a later resample.** The audio latent's length
  is `round(frames / fps * 25)` and every audio token carries a `[start, end)`
  bound in SECONDS on the same axis the video tokens use, which is the space
  the cross-attention's shared RoPE is built in. The decode lands exactly
  three mel frames short of the clip (the causal audio VAE's first latent
  frame covers one mel frame rather than four); the last sample is held over
  that gap so the two tracks are the same length.
* **Off by default because of COST, not correctness - but the cost is a
  fraction of what it was.** The audio-extended transformer block has the same
  streamed/quantized/device-resident implementation the video-only one does
  (`LtxAvBlockQ`, `CachedQAvBlockWeights`, `AvDitSession`), and that is what an
  `--audio` generation denoises through: the checkpoint runs as int8 with as
  many blocks resident on the card as its VRAM budget allows, instead of being
  expanded to host fp32 and re-uploaded block by block on every forward.
  At the real single-window width on one discrete GPU, a warm audio-visual
  forward is modestly more expensive than the same clip's video-only forward
  rather than several times, and holds roughly half the host RAM; the two
  arms are reproducible side by side with the `ltxv_bench streamed-av` and
  `ltxv_bench av` harnesses. It stays `[~]` and off by default because the
  audio-visual path has no training support.
* **`--av-fp32-reference` picks the other arm.** The eager host-fp32
  audio-visual denoiser is the reference the quantized one is gated and
  measured against, and it is still reachable - the same model at higher
  precision, needing the whole checkpoint expanded to fp32 in host RAM and
  several times the wall clock. It is an A/B switch, not a quality dial; each
  run's own opening line names whichever arm it is on.
* **48 kHz stereo.** The base vocoder synthesizes 16 kHz; the checkpoint's
  own bandwidth-extension stage (`vocoder.bwe_generator.*`, driven by a
  causal log-mel of the base waveform through `vocoder.mel_stft.*`) adds the
  upper band on top of a x3 resample, exactly as the reference's
  `VocoderWithBWE` does.
* **Muxed in process.** brain writes the clip itself: AV1 video and an Opus
  sound track in `.mp4`, `.mov`, `.mkv` or `.webm`. A `.gif` is written
  silent with a line on stderr, since the container holds no audio stream.
* **A clip longer than one denoising window keeps its sound.** Past the
  window ceiling the audio latent crosses each seam exactly as the video
  latent does: the previous window's own last audio tokens are sliced out of
  the denoised latent and frozen at sigma 0 at the head of the next window's
  audio sequence, while only the new tokens get a denoising schedule. The
  windows contribute tokens to ONE audio latent, decoded once at the end, so
  there is no vocoder boundary at a seam to hear.
* **The seams have to land on whole audio tokens, and a window plan with
  `--audio` is built so they do.** The two streams do not share a time
  resolution: one video latent frame is 8 pixel frames, so it is `8 * 25 /
  fps` audio tokens - 25/3 of one at 24 frames a second. A seam placed at an
  arbitrary latent frame would fall part way into a token, and two thirds of a
  token cannot be frozen. So an audio-visual plan advances in multiples of the
  smallest number of latent frames worth a whole number of tokens (3 at 24 and
  30 frames a second, 1 at 25), which is a different window split from the
  same clip without `--audio` - one more window at some lengths. Every seam is
  then
  exact, and the clip's total token count is the same
  `round(frames / fps * 25)` a single-window clip of that length would have,
  so the container's two stream durations do not move. A shape whose seams
  cannot be placed is refused with the frame counts that would work, never
  approximated: audio that drifts against the picture does not announce
  itself.
* **`--context-frames` is what to lower if a plan is refused.** The carried
  context costs both streams' budget, and it is the one knob that changes
  where a seam can fall.
* **Not for a multi-scene clip.** A scene boundary deliberately carries
  nothing across, which is what frees the next scene to be a different
  subject - so the sound would restart at every cut, and the per-scene token
  counts do not sum to the clip's own. What sound should do at a deliberate
  visual cut is a design question rather than an arithmetic one, and it is
  refused rather than guessed. A long SINGLE-scene clip has no such limit.

The same `audio` switch is a parameter on the `t2v` capability action, and the
action declares an `audio` output blob (a complete 48 kHz stereo WAV), so the
D-Bus/served surface returns the sound too rather than only the CLI.

### Text to audio

```bash
brain ltxv t2a --prompt "rain on a tin roof, distant thunder" --output-path rain.wav
```

The audio stream runs alone, as the reference's `t2a_one_stage` builds it:
every block is the audio stream's self-attention, text cross-attention and
feed-forward, and none of the video half's weights is read. The audio
guidance flags apply; the cross-modal term is off, as there is no picture to
guide against.

### Retakes, sound for a picture, and a picture for a sound

`brain ltxv v2v --window S:E` regenerates the seconds between `S` and `E` of
an existing clip against `--prompt` and holds the rest. With `--audio` and the
clip's soundtrack as `--audio-source`, the sound in the window is regenerated
with the picture; `--hold-video` keeps every frame and generates sound for
them (video to audio), `--hold-audio` keeps the soundtrack. `t2v --audio
--audio-source <wav>` goes the other way: the soundtrack is held frozen and
the picture denoises against it.

### Replacing a person

```bash
brain ltxv replace --input street.mp4 --character sheet.png --point 405,195 \
  --prompt "a woman with short copper-red hair in a mustard raincoat walks toward the camera" \
  --output-path replaced.mp4
```

The person clicked on is replaced by the character on the sheet; their
path, their timing and the background stay the source's. In order:

1. SAM 2.1 tracks the person through the clip (`--point`, repeatable, and
   `--exclude` are the clicks).
2. FLUX.2 paints the character over the person on frame 0 and on guide
   frames (`--guide-frames`, default every 40th frame and the last). It
   repaints only the person's grown silhouette on a canvas around them and
   keeps the rest exactly, so the character has to take the person's
   place, size and pose instead of being staged somewhere else. Where the
   frame cuts the person off, the canvas extends below it: the model draws
   a whole body at the person's scale there rather than shrinking the
   figure to fit, and the rows below the frame are discarded. An edit whose figure overlaps the person by less than
   half (intersection over union) is retried with another seed. With
   ArcFace in the model store every attempt is made, and of those that
   cover the person the one whose face is closest to the sheet's is kept
   (`--no-identity` keeps the first that covers the person); a guide
   frame with no edit that passes is dropped and reported, because a guide
   standing somewhere else pulls the clip away from the person's motion.
   Frame 0 with no passing edit stops the run.
3. Masked video to video regenerates the person's region and holds every
   other latent token: the region is the person's silhouette joined with the
   painted figure's (carried along the track and scaled with the person),
   grown to whole latent tokens. The conditioning is a mean over each
   token's 32 x 32 pixels and pixel frames, so a token the region covers
   only in part is held in part - a sleeve of the old costume survives.
   Frame 0 is the painted still, held inside the figure and redrawn in a
   ring across its outline, where the edit blends the clothes into the
   street. The other stills are keyframe guides held at half strength
   (`--guide-strength`): held fully, the clip cross-fades from one still's
   figure into the next instead of moving it.
4. The generated clip is composited over the source across a feathered
   edge: outside the region every pixel is the source's own.

Any clip brain decodes is accepted: one that is not 1 + 8k frames at
multiples of 32 is trimmed at its end and centre-cropped onto that grid,
the clicks moving with the crop, and the run says what it did
(`--no-conform` refuses instead). `--actor-mask <dir>` gives the person's
track as a mask sequence instead of clicks. brain reads no sound out of a
container, so the result is silent unless `--soundtrack <wav>` carries one.
`--adapter`/`--lora-scale` apply LoRAs to the video pass and
`--edit-adapter`/`--edit-lora-scale` to the still edits; every threshold
and margin above is a flag (`brain ltxv replace --help`).

Every intermediate - the track, each edit attempt, the stills, the region
mask sequence, the clip the video pass sees and the generated clip - is
kept in `--work-dir` (default `<output>.work`).
`samples/shell/videogen/character_replace` runs this, or one step of it,
with every model and setting as a flag, and lists the pull commands for
each model it can use.

`brain ltxv v2v --mask <maskseq> --keyframe F:<image>` is the video step on
its own: guides append clean blocks at their frames' instants and compose
with the mask; a start, mid or end still would overwrite the tokens the
mask holds and is refused.

### Extending a clip

```bash
brain ltxv extend --input clip.mp4 --seconds 3 --prompt "the cat turns and walks away" \
  --output-path longer.mp4
```

The clip's last 57 frames (the reference extension recipe's 8 latent frames)
are held as a clean prefix while new frames after them denoise; the output is
the clip itself, untouched, followed by the new frames. `--audio` with the
clip's soundtrack extends the sound too. An extension LoRA trained with the
reference's `video_extend_lora` recipe applies with `--adapter`.

### Seeing what it did

Any of these runs takes `--trace-ltxv <0-5>` (see `brain help` for the shared
tracing options). Level 3 reports the run's phases and timings, 4 adds each
denoise step's sigma and seconds-per-step, and 5 adds every individual
forward plus, on the real streamed 22B path, every transformer block with
whether its weights came from the per-generation cache or were re-read and
re-quantized:

```bash
brain --trace-ltxv 4 ltxv t2v --prompt "a cat" --frames 9 --width 64 \
  --height 64 --steps 4 --output-path out.mp4
brain --trace-ltxv 5 --trace-format json --trace-output run.jsonl ltxv t2v …
```

Levels 1 and 2 are the "only tell me what went wrong" settings: 1 is errors
only, 2 adds the warnings that say a run is not what it looks like (a
random-weight DiT, a stub text context, a `--steps` the distilled schedule
ignores).

`ltxv` has a dedicated CLI module, so the resolver gives it precedence over
generic capability dispatch - the same routing `wan` uses. Every command is
also a capability action, discoverable with `brain caps brain/ltxv`, callable
with `brain do`, and served as a cancellable streaming job over D-Bus
(`brain serve --dbus`); the clip actions take their source as a `video` input
blob.

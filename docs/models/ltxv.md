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

**Status: in progress - `brain ltxv t2v`/`brain ltxv dfr` run end to end as
SMOKE TESTS, not a quality claim yet.** This page describes the target shape;
see `.agents/roadmap/ltxv.md` for the live, milestone-by-milestone checklist
of what has actually landed (it goes into far more numeric detail - exact
cosine parity per component, exact gaps - than this page tries to). The 22B
transformer and the 12B Gemma-4 text encoder are large enough that real-weight
validation needs a machine this port was not built on, and neither exists in
this repo yet - what runs today is the real scheduler (`LTX2Scheduler` + the
rectified-flow ancestral Euler step), the real causal 3D video VAE decode
(plus, for `dfr`, real latent upscalers), and the real CLI/capability/
residency wiring, composed with a **tiny, random-weight** DiT and a **stub**
text context (no real Gemma-4 encoder in the pipeline yet, even though a
tiny-config-parity-proven `crates/gemma4` crate exists on its own). See
`crates/ltxv/src/pipeline.rs`'s module doc for exactly what is real and what
is a placeholder in each command. What has real-weight parity coverage is
every small, self-contained piece this port has ported so far - both video
VAE decoders (the conv mirror and the 3D neighborhood-attention "diffusion
decoder"), the audio VAE, the vocoder, both latent upscalers, the duration
head - plus the schedule math; the DiT (video-only and audio+video) and
Gemma-4 are only tiny-config-parity-proven, not real-weight-proven (see
"Getting the weights" below for why).

## Support

| Capability | Supported |
|---|---|
| Inference (text to video) | [~] smoke test - tiny random-weight DiT, stub text context (`brain ltxv t2v`), see above |
| Inference (DFR, higher-res multi-stage) | [~] smoke test - real spatial/temporal upscalers and VAE decode, still the tiny DiT (`brain ltxv dfr`) |
| Post-hoc upscale of a finished clip | [x] `brain ltxv upscale` - VAE-encode an existing video file, run the official x2 latent spatial upscaler, refine on the distilled refinement schedule, VAE-decode. Shares the upscale+refine implementation with the internal two-stage generation path. CLI only so far, no capability action (see below) |
| Inference (text to video+audio) | [~] `brain ltxv t2v --audio` / `audio: true` on the `t2v` action - the real audio+video DiT, both streams denoised jointly, decoded through the real audio VAE + vocoder and muxed into the container. Works; it is `[~]` rather than `[x]` because the audio-extended block has no streamed/quantized/resident path (see "Generating sound" below), so it runs the whole checkpoint as host fp32 and is far more expensive per step than the video-only path. 16 kHz stereo (no bandwidth extension); single-window clips only |
| Inference (image to video / keyframe conditioning) | [x] `--start-frame`, `--mid-frame` (+ `--mid-frame-at`) and `--end-frame` - up to three real stills VAE-encoded and held at sigma 0 in ONE generation pass, with `--conditioning-strength` for how hard. Refused for a multi-window or multi-scene clip; see "Anchoring a clip on real images" below |
| LoRA fine-tune | [~] video-only DiT, host-math/gradcheck-proven (FD < 1e-4), single- and whole-batch overfit drives loss to ~0 at tiny-config scale - the audio-extended DiT has no training support |
| Full fine-tune | [~] same scope/caveat as LoRA fine-tune above |
| INT8 | [~] storage format only for the video-only DiT's weights (`crate::int8`) - not wired into any checkpoint loader, no compute-time kernel |
| CLI (`brain ltxv {t2v,upscale,dfr}`) | [x] |
| HTTP API | [ ] the OpenAI/Anthropic routes cover chat, embeddings and text-to-IMAGE; there is no video route, and `api_caps` derives exposure from action shape, so a video action is simply not advertised |
| D-Bus | [~] `t2v` and `dfr` are reachable as actions, not just CLI subcommands, via the generalized `capability::Provider`/residency surface - `upscale` is **not**: it takes a whole video file as INPUT, which is the first action in this model that would need an input blob rather than parameters alone, and that action shape has not been designed. Recorded rather than quietly skipped, because "a bespoke CLI subcommand is never the only entry point" is this repo's serving contract |
| Batched serving | [ ] nothing resident to batch yet - see `crates/cli/src/resident_ltxv.rs`'s module doc |
| Multi-device sharding | [~] `model::Shardable` plumbing for the video-only DiT is implemented and tested (partition planning, weight-subset loading, the single-shard and sequential-two-stage cases) - no real multi-device execution has been run against two physical accelerators yet |
| NPU | [ ] deliberate scope exclusion, not a gap expected to close later: no existing `NpuModel` implementation pattern fits a model this large, and this model's realistic deployment target is GPU/CPU |

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

Model id (once served): `brain/ltxv`. The default checkpoint is
`Lightricks/LTX-2.5` on Hugging Face, gated - fetching it requires a Hugging
Face account with access accepted and `hf auth login` run first. It ships as
several independent files rather than one directory-style checkpoint:

```bash
export BRAIN_LTXV_DIT=…/ltx-2.5-22b-distilled-transformer-bf16.safetensors
export BRAIN_LTXV_VAE=…/ltx-2.5-video-vae-bf16.safetensors
export BRAIN_LTXV_AUDIO_VAE=…/ltx-2.5-audio-vae-bf16.safetensors
export BRAIN_LTXV_TEXT_ENCODER=…/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
export BRAIN_LTXV_TOKENIZER=…                # extracted from the text-encoder file
```

The DiT alone is 42 GB at bf16 and the text encoder 26 GB - together they need
roughly 70+ GB of addressable memory just to hold weights, before activations.
Confirm your hardware can hold what you point at before fetching it.

An LTX GGUF (`general.architecture = "ltxv"`) converts with `brain import
<file>`; it carries the full architecture config as GGUF metadata, so no
separate config file is needed. The VAEs, text encoder and tokenizer still come
from the repository above.

## Running it

The smoke-test path runs today, needing only the real VAE (no DiT/text-encoder
weights - the DiT is always tiny-config with fresh random weights):

```bash
export BRAIN_LTXV_VAE=…/ltx-2.5-video-vae-conv-bf16.safetensors
brain ltxv t2v --prompt "a cat walking on a beach" --frames 9 --width 64 \
  --height 64 --steps 4 --output-path out.mp4
```

This proves the pipeline WIRING - real scheduler, real VAE decode, a real mp4
out the other end - not generation quality; see `crates/ltxv/src/pipeline.rs`'s
module doc.

### Anchoring a clip on real images

A generation can be pinned to stills you supply, at up to three instants at
once: `--start-frame`, `--mid-frame` and `--end-frame`. Each is VAE-encoded and
held at sigma 0 (`denoise_mask = 0`, per-token timestep 0, re-pinned every step)
while the rest of the clip denoises around it.

```bash
brain -v --device gpu0 ltxv t2v --dit-config ltx25_22b \
  --prompt "a fishing boat crosses the harbour mouth, camera tracking left" \
  --frames 121 --width 768 --height 448 --fps 24 --output-path boat.mp4 \
  --start-frame dawn.png --mid-frame midway.png --end-frame open-sea.png
```

* **`--mid-frame-at <N>`** picks the pixel frame the middle still anchors,
  strictly between `0` and `--frames - 1`. Left off, it is the clip's own
  midpoint, `(frames - 1) / 2` - the reference's own position for a single
  interior keyframe (`ltx_pipelines.utils.helpers.
  evenly_spaced_keyframe_positions(1, 121) == [60]`). The resolved frame is
  printed in the run's first line.
* **Any frame is a legal position; nothing is snapped to the `1 + 8k` grid.**
  A middle still is an *appended guiding block* carrying its own RoPE position
  (`VideoConditionByKeyframeIndex.apply_to` adds `frame_idx` straight onto the
  time coordinate and divides by fps), not a slot on the generated video's
  latent grid, so there is no grid for it to land on. The `1 + 8k` rule
  constrains the clip's *length*, not where a guide may point inside it.
* **One still at frame 0 and nothing else is still image-to-video**, which
  overwrites latent frame 0 in place. The moment a second still is given -
  middle, end, or both - every still becomes an appended guide instead,
  including the one at frame 0, and no token of the generated video is frozen.
  That is the reference's own split between `combined_image_conditionings` and
  `image_conditionings_by_adding_guiding_latent`.
* **`--conditioning-strength`** (default `1.0`) applies to every still given.
* **Two identical anchors make a static clip.** Passing the same image to
  `--start-frame` and `--end-frame` asks for "start here, end here", which has
  a correct trivial answer; see `crates/ltxv/tests/motion_real.rs` for the
  measured table. A middle anchor is subject to the same logic - anchor
  *different* instants.
* **Not supported for a multi-window or multi-scene clip.** `--mid-frame` names
  a pixel frame of the whole clip, and routing it means finding the window
  whose emitted range covers it and re-expressing it in that window's own
  frame numbering. That is refused rather than silently dropped.

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
brain -v --device gpu0 ltxv t2v --dit-config ltx25_22b \
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
  overridable with `BRAIN_LTXV_LONGFORM_MAX_TOKENS`.
* **A request that fits one window is unchanged.** It is handed straight to the
  single-window path, bit for bit, and none of this runs.
* **`--end-frame` and `--mid-frame` are refused for a multi-window clip** - the
  first pins the last frame of one window, and pinning the end of a rolling
  plan has not been designed; the second names a pixel frame of the whole clip,
  which would have to be routed to whichever window covers it.
  `--start-frame` conditions the first window as usual.

The same routing applies to the `t2v` capability action (`brain do brain/ltxv
t2v`), with the default context and no new action parameter.

### Several scenes in one clip

Everything above keeps **one shot** continuous, which is the wrong answer if the
clip is supposed to become something else part way through: every window shares
the prompt, and every continuation window is hard-conditioned on the real
content before it. Repeat `--scene <frames>:<prompt>` to write a clip that
changes.

```bash
brain --device gpu0 ltxv t2v --dit-config ltx25_22b \
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
  `crates/ltxv/tests/longform.rs::a_two_scene_request_is_one_clip_whose_second_scene_owes_nothing_to_its_first`).
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
  `--end-frame`/`--mid-frame` are refused for a multi-scene clip for the same
  reasons they are refused for a multi-window one.
* **Not exposed on the `t2v` capability action** - `--scene` is a CLI flag; the
  action still takes one prompt.

### Upscaling a clip that already exists

`brain ltxv upscale` takes a rendered video file and re-renders it at twice the
spatial resolution. It is not a pixel-space resampler: the clip is VAE-encoded,
carried up by the **official LTX-2.5 x2 latent spatial upscaler**, refined by a
short diffusion pass at the new size, and VAE-decoded - the exact tail a
two-stage generation already runs (see the Stages section of
`brain ltxv t2v --help`), reached through the same code.

```bash
export BRAIN_LTXV_VAE=[path/to/ltx-2.5-video-vae-conv-bf16.safetensors]
export BRAIN_LTXV_UPSAMPLER_SPATIAL=[path/to/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors]
export BRAIN_LTXV_DIT=[path/to/ltx-2.5-22b-distilled-transformer-Q8_0.gguf]
export BRAIN_LTXV_TEXT_ENCODER=[path/to/gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf]
brain -v --device gpu0 ltxv upscale --dit-config ltx25_22b \
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

`brain ltxv dfr` runs the same smoke-test DiT through DFR (Diffusion Fidelity
Rendering): a half-res base generation with generated keyframe slots, a REAL
spatial x2 latent upscale, a full-res re-noised detailing pass (no IC-LoRA -
none exists in this repo), and 0-2 REAL temporal x2 upsample rounds with
tile-based stitching, needing the VAE plus both real latent-upscaler
checkpoints:

```bash
export BRAIN_LTXV_VAE=…/ltx-2.5-video-vae-conv-bf16.safetensors
export BRAIN_LTXV_UPSAMPLER_SPATIAL=…/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
export BRAIN_LTXV_UPSAMPLER_TEMPORAL=…/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors
brain ltxv dfr --prompt "a cat walking on a beach" --frames 9 --width 64 \
  --height 64 --steps 4 --output-path out.mp4
```

### Generating sound

LTX-2.5 is one model that denoises a video-latent stream and an audio-latent
stream **together**, coupled every block by cross-attention. `--audio` runs
both; without it only the model's video half runs and the clip is silent.

```bash
export BRAIN_LTXV_VAE=[path/to/ltx-2.5-video-vae-conv-bf16.safetensors]
export BRAIN_LTXV_DIT=[path/to/ltx-2.5-22b-distilled-transformer-Q8_0.gguf]
export BRAIN_LTXV_TEXT_ENCODER=[path/to/gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf]
export BRAIN_LTXV_AUDIO_VAE=[path/to/ltx-2.5-audio-vae-bf16.safetensors]
brain -v --device gpu0 ltxv t2v --dit-config ltx25_22b --audio   --prompt "a blacksmith hammers glowing steel on an anvil, rhythmic clangs"   --frames 57 --width 1280 --height 704 --fps 24 --output-path forge.mp4
```

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
* **Off by default because of COST, not correctness.** The audio-extended
  transformer block has no streamed/quantized/device-resident implementation
  the way the video-only one does - no `LtxAvBlockQ`, no AV
  `CachedQBlockWeights`, no AV `DitSession`. So an audio-visual run expands
  the whole checkpoint to host fp32 and re-uploads every block to the card on
  every forward, which makes it markedly slower per step than the same clip
  without sound and needs most of a large machine's RAM. The command refuses
  up front, with both numbers, on a machine that cannot hold it. Closing that
  gap is the tracked next step, and it is what would make sound cheap.
* **16 kHz stereo.** The bandwidth-extension stage that lifts the base
  vocoder to 48 kHz (`vocoder.bwe_generator.*`) is present in the checkpoint
  and not implemented - it needs an ISTFT this port does not have.
* **Muxed, or handed back.** `.mp4`/`.mkv`/`.mov` get an AAC track,
  `.webm` an Opus one; a `.gif` is written silent with a line on stderr since
  the container holds no audio stream. Without `ffmpeg` the sound is written
  as `audio.wav` beside the numbered frames and the printed command muxes
  both - the same "never throw away a generation for want of an encoder"
  contract the video path already had.
* **Single-window clips only.** A multi-window or multi-scene request is
  refused: what crosses a window seam today is a video latent prefix, and the
  audio stream's counterpart has not been designed. Generating per window and
  concatenating would restart the sound at every seam.

The same `audio` switch is a parameter on the `t2v` capability action, and the
action declares an `audio` output blob (a complete 16 kHz stereo WAV), so the
D-Bus/served surface returns the sound too rather than only the CLI.

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
generic capability dispatch (`brain ltxv {t2v,upscale,dfr}` runs that module, not a
generic action call) - the same routing `wan` uses. Both actions are still
reachable the same way every model in this repo is over other transports:
discovery (`brain caps brain/ltxv`), `brain do brain/ltxv {t2v,dfr}` (not
`upscale` - see the Support table), and a
cancellable streaming job over D-Bus (`brain serve --dbus`) - a bespoke CLI
subcommand is never the only entry point, per this repo's serving contract. A
real 22B DiT and the Gemma-4 text encoder are tracked gaps (see the roadmap) -
once they land, this same CLI/capability surface serves them with no shape
change, only `--dit-config`/`dit_config` growing a second value.

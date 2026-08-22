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
| Inference (text to video+audio) | [ ] the audio-extended DiT (`LtxAvDit`) and A<->V cross-attention exist at tiny-config parity as a library, but nothing wires them into a pipeline/CLI action yet |
| Inference (image to video) | [ ] |
| LoRA fine-tune | [~] video-only DiT, host-math/gradcheck-proven (FD < 1e-4), single- and whole-batch overfit drives loss to ~0 at tiny-config scale - the audio-extended DiT has no training support |
| Full fine-tune | [~] same scope/caveat as LoRA fine-tune above |
| INT8 | [~] storage format only for the video-only DiT's weights (`crate::int8`) - not wired into any checkpoint loader, no compute-time kernel |
| CLI (`brain ltxv {t2v,upscale,dfr}`) | [x] |
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
  --input clip.mp4 --output-path clip_2x.mp4 \
  --prompt "the prompt the clip was generated from"
```

Three things worth knowing before running it:

* **`--prompt` is not optional in practice.** The refinement step is a
  diffusion pass conditioned on text, so the model is answering "what should
  this look like with more detail" against whatever context it is given.
  Omitting the flag refines against an empty prompt and costs detail; the
  command says so on stderr when you do. A video file carries no record of
  the prompt it was generated from, so it has to be supplied.
* **The input's shape has to be VAE-representable**: width and height a
  multiple of 32, frame count of the form `1 + 8k`. Anything `brain ltxv t2v`
  wrote already satisfies both.
* **Length has a ceiling per refinement pass.** An upscaled clip carries four
  times the video tokens per frame that its input did, so a length that was
  fine at the source resolution need not fit at the target one. Past
  `ltxv::pipeline::REFINE_MAX_TOKENS` in one pass the clip is refined in
  several consecutive segments that share one frame at each boundary, and
  fine detail can step where two segments meet - a bounded, visible artefact,
  reported on stderr, not a silent corruption. A clip that fits is never
  split, and a request that cannot be split into anything runnable is refused
  before any weight is read.

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

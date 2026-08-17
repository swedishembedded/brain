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

**Status: in progress, not yet runnable.** This page describes the target shape;
see `.agents/roadmap/ltxv.md` for the live checklist of what has actually landed.
The 22B transformer and the 12B Gemma-4 text encoder are large enough that
real-weight validation needs a machine this port was not built on - what has
parity-gated real-weight coverage today is only the small, self-contained pieces
(the two video VAE decoders, the audio VAE, the vocoder, the latent upscalers).

## Support

| Capability | Supported |
|---|---|
| Inference (text to video) | [ ] |
| Inference (text to video+audio) | [ ] |
| Inference (image to video) | [ ] |
| LoRA fine-tune | [ ] |
| Full fine-tune | [ ] |
| INT8 | [ ] |
| CLI (`brain ltxv t2v`) | [ ] |
| D-Bus | [ ] |
| Batched serving | [ ] |
| NPU | [ ] |

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

Not yet - see the roadmap. Once the video-only milestone lands, the shape will
match `wan`'s: `brain ltxv t2v --prompt … --output-path out.mp4`, plus discovery
and a cancellable streaming job over the generalized capability interface
(`brain caps brain/ltxv`, `brain serve --dbus`, `examples/videogen/`).

# MiniMax-H3 (joint video + audio generation)

A 33B rectified-flow diffusion transformer that denoises video and audio
latents together in **one packed self-attention stack** - unlike
[`ltxv`](ltxv.md)'s two-stream architecture, H3 has no separate audio blocks
and no A/V cross-attention; modality differences come entirely from
input/output projections and modality-tagged AdaLN conditioning. Text
conditioning is a Qwen3-VL encoder truncated to an intermediate hidden
layer. Released tasks: `t2va` (text -> video+audio), `fl2va`
(first/last-frame-conditioned), `ref2va` (up to 12 mixed image/video/audio
references).

**This port is in progress.** See `.agents/roadmap/minimaxh3.md` for the
phase ledger and which convention questions are settled vs. still open. No
capability below is claimed working until its own phase's real-weight parity
gate is green.

## Support

| Task | Status |
|---|:---:|
| `t2va` (text -> video+audio) | [ ] in progress |
| `fl2va` (first/last-frame -> video+audio) | [ ] in progress |
| `ref2va` (image/video/audio references -> video+audio) | [~] implemented and golden-checked (layout, reference normalization and encoding, Qwen3-VL presentation); not yet run on the `transformer_ref/` weights |
| Training (LoRA) | [ ] in progress |

## Getting the weights

**Weights are never auto-fetched.** `minimaxh3` has no `default_ref` in
`arch::ARCHS` - the same deliberate omission `supir`/`fincast` use, here for
a different reason (see Licensing below). Put a checkout you obtained
yourself in the model store, laid out as the release ships it: the root
`MiniMaxH3ModularPipeline` (`model_index.json` beside `transformer/`, `vae/`,
`audio_vae/`, `text_encoder/` and `tokenizer/`). `brain minimaxh3` finds every
component from that root; `--dit`, `--video-vae`, `--vocoder`,
`--text-encoder` and `--tokenizer` name one outright. The served adapter
reads `BRAIN_MINIMAXH3_DIR`, the same root.

## Running it

```
BRAIN_MINIMAXH3_ALLOW_COMMUNITY=1 brain minimaxh3 fl2va \
  --prompt "a red kayak plunges through whitewater rapids, spray everywhere" \
  --first_frame still.png --width 704 --height 384 --num_frames 121 \
  --steps 20 --out video=kayak.mp4
```

`t2va` takes the same flags without `--first_frame`/`--last_frame`. The clip
carries the sound the same forward generated as its own track; `--out
audio=…` also writes it as a WAV.

`ref2va` takes `--references`, a comma-separated list of up to 9 images, 3
videos and 3 soundtracks (12 in all): a WAV is a soundtrack, a PNG/JPEG/WebP
an image, anything else a video, whose own sound is attached as
`clip.mp4+clip.wav`. A soundtrack needs a visual reference beside it. The
prompt refers to them as the reference pipeline labels them - `<Picture 1>`,
`<Video 1>`, `<Audio 1>` in order of kind:

```
BRAIN_MINIMAXH3_ALLOW_COMMUNITY=1 brain minimaxh3 ref2va \
  --prompt "the dog from <Picture 1> runs down the street from <Video 1>" \
  --references dog.png,street.mp4+street.wav --width 704 --height 384 \
  --num_frames 121 --steps 20 --out video=dog.mp4
```

Images are scaled to a 2048-pixel short edge and videos resampled to 24 fps
as the reference does; each reference is VAE-encoded and packed ahead of the
generated audio and video under the same rotary clock, and the prompt is
encoded with the references themselves presented to Qwen3-VL. Every run
reports per-step progress and stops at the next step boundary when cancelled.

* **The text encoder is placed per prompt.** The first 50 layers of the
  Qwen3-VL language tower (~27 GB at int8) are spread over as many of the
  `--device` cards as hold them, run once, and dropped before the DiT loads.
  A selection that cannot hold them is refused, naming the bytes.
* **The DiT is held resident at int8 when the cards hold it**, with the
  request's forward workspace free on each (~25 GB of packed linears and
  norms, split over two cards), so every step reads weights already on the
  cards. Otherwise it streams one block at a time at fp32, which any card
  holds but which reads the whole checkpoint every step. On two P40s with
  21.9 GiB free each, 704x384 x 121 frames (~11k packed rows) holds resident
  and peaks at ~21.9 GB per card; 768x448 (~13.6k rows) does not - its first
  step needs ~10.9 GB of workspace beside ~12.5 GB of weights per card.

## Hardware and limits

2xTesla P40 (24GB each) + 184GB RAM is what this port targets. The 66GB
bf16 checkpoint is 132GB at fp32, too large to hold resident; at int8 the
DiT's linears sit across both cards. Of the 33B parameters, 13.0B are the
per-block `adaln_proj` projections, which never load as weights: each step's
modulation rows are projected once per distinct timestep and cached. The Qwen3-VL text encoder is a
separate ~33B-class model and is never co-resident with the DiT: generation
runs in sequential phases (encode -> free -> denoise -> free -> decode).

License: **MiniMax H3 Community License Agreement** - a territorial
carve-out (the ordinary grant excludes the EU, UK, South Korea and the US;
an organization in one of those regions needs MiniMax's separate
authorization), a >$20M/yr revenue registration clause, and a mandatory
"MiniMax H3" UI attribution requirement for any commercial product or
service. This port's Rust code is Apache-2.0, ported from the Apache-2.0
`diffusers`/`transformers` reference implementation - MiniMax's own
(community-licensed) repository is not translated into this crate. The
weights themselves are never vendored or auto-fetched; running against a
checkpoint you obtained yourself additionally requires
`BRAIN_MINIMAXH3_ALLOW_COMMUNITY=1` (`minimaxh3::caps::check_license`),
confirming you have cleared the license's terms for your own use. See the
checkpoint's own `LICENSE` file for the authoritative text.

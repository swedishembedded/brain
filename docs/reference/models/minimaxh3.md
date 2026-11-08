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
`--text-encoder` and `--tokenizer` name one outright. `brain serve` serves
the same model-store checkout.

## Running it

```
brain minimaxh3 fl2va --allow-community \
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
brain minimaxh3 ref2va --allow-community \
  --prompt "the dog from <Picture 1> runs down the street from <Video 1>" \
  --references dog.png,street.mp4+street.wav --width 704 --height 384 \
  --num_frames 121 --steps 20 --out video=dog.mp4
```

Images are scaled to a 2048-pixel short edge and videos resampled to 24 fps
as the reference does; each reference is VAE-encoded and packed ahead of the
generated audio and video under the same rotary clock, and the prompt is
encoded with the references themselves presented to Qwen3-VL. Every run
reports per-step progress and stops at the next step boundary when cancelled.

Every task takes a LoRA adapter for its DiT with `--adapter` (`adapter` on a
served request, a path the model store resolves) and its strength with
`--lora_scale` (1.0 as trained, on top of the file's own alpha/rank). A
diffusers PEFT save reads as it is, including one that keeps its adapter name
in each key and its alpha in the header, such as a few-step distillation; run
it with the step count it was distilled for:

```
brain minimaxh3 ref2va --allow_community true \
  --prompt "the woman from <Picture 1> turns to the camera and laughs" \
  --references woman.png --width 768 --height 448 --num_frames 121 \
  --adapter lightx2v/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --steps 4 --out video=woman.mp4
```

Each target is checked against the DiT by name and shape before a weight is
read. A resident int8 DiT evaluates each correction beside its linear on the
fp32 input, never folded into the int8 weights; a streamed fp32 DiT folds it
exactly as each block is read. The corrections are held on the cards with the
blocks they belong to and count toward the placement.

`--vsa_sparsity` turns on video sparse attention (FastVideo's VSA for this
packed sequence): the video is cut into 4x4x4 tiles of 64 rows, every video
query keeps the best `1 - sparsity` of each video's key tiles by tile-mean
score, and the text, references and audio always stay dense. `--vsa_gate`
names the trained `to_gate_compress` of every block - a FastVideo VSA
checkpoint or a gate-only extract - which adds the coarse branch the
sparse attention was distilled with:

```
brain minimaxh3 ref2va --allow_community true \
  --prompt "the woman from <Picture 1> turns to the camera and laughs" \
  --references woman.png --width 768 --height 448 --num_frames 121 \
  --adapter lightx2v/Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --vsa_gate barelymining/ComfyUI-MiniMax-H3-FastVideo/fasth3_vsa_gate.safetensors \
  --vsa_sparsity 0.75 --steps 4 --out video=woman.mp4
```

A served `ref2va` request, which cannot name files on the server, picks the
same setup by `preset`: `base` (the release alone), `turbo-4step` (the
adapter above) or `turbo-4step-vsa` (the adapter and the gates, at
`vsa_sparsity` 0.75 unless one is asked). Each preset runs pinned files of
`lightx2v/Minimax-h3-Turbo` and `barelymining/ComfyUI-MiniMax-H3-FastVideo`
from the model store that holds the release; one the store lacks is
refused naming the repo.

Sparse attention needs a GPU (its attention pass is workgroup-cooperative).
The gates are held at the DiT's tier beside each block (~1.9 GB at int8)
and count toward the placement.

* **The text encoder is placed per prompt.** The first 50 layers of the
  Qwen3-VL language tower (~27 GB at int8) are spread over as many of the
  `--device` cards as hold them, run once, and dropped before the DiT loads.
  A selection that cannot hold them is refused, naming the bytes.
* **The DiT is held resident at int8 when the cards hold it**, with the
  request's forward workspace free on each (~25 GB of packed linears and
  norms, split over two cards), so every step reads weights already on the
  cards. Otherwise it streams one block at a time at fp32, which any card
  holds but which reads the whole checkpoint every step. On two P40s,
  768x448 x 121 frames (~13.6k packed rows) holds resident and peaks at
  ~19.0 GB per card; a 20-step `fl2va` clip at that size takes ~50 min.
* **`--int8-activation-levels 2`** (`int8_activation_levels: 2` on a
  request) quantizes each int8 linear's input in two levels: the second
  quantizes what the first rounds away, so the activation side is exact to
  ~1/(127·254) of its group scale and the DiT computes what its int8 weights
  do. It costs a second product per linear and a second int8 copy of every
  linear's input in the workspace, which the placement reserves. The default,
  1, is the int8 tier's ordinary arithmetic.

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
checkpoint you obtained yourself additionally requires `--allow-community`
(`allow_community: true` on a served request; `minimaxh3::caps::check_license`),
confirming you have cleared the license's terms for your own use. Every
request carries its own confirmation, and one without it is refused. See the
checkpoint's own `LICENSE` file for the authoritative text.

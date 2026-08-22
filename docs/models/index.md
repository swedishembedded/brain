# Model catalog

Every architecture below has one canonical id (see [The CLI](../using/cli.md))
that names it everywhere: the CLI (`brain <id> <verb>` / `brain <verb> <id>`),
`brain caps`, and its own page here. A checked box means the capability is
reachable through a real, documented command - nothing here is aspirational.
**⤓** means brain fetches and converts the weights itself on first use;
everything else needs a local checkpoint, pointed at by a `BRAIN_*` variable
named on the model's own page (see
[Getting the weights](../using/models-and-weights.md)).

## Text

| Architecture | Solves | Infer | Train | LoRA |
|---|---|:---:|:---:|:---:|
| [`qwen3`](qwen3.md) ⤓ | instruct/chat LLM, tool calls, paged-KV serving | [x] | [x] | [x] |
| [`qwen35moe`](qwen35moe.md) | hybrid GDN/GQA sparse-MoE decoder | [x] | [ ] | [x] |
| [`qwen35`](qwen35.md) | dense hybrid GDN/GQA decoder + MTP + vision splice | [x] | [x] | [x] |
| [`gpt2`](gpt2.md) | dense nanoGPT-parity baseline | [x] | [x] | [ ] |
| [`glmdsa`](glmdsa.md) | MLA + sigmoid noaux_tc MoE decoder | [x] | [x] | [ ] |

## Embeddings

| Architecture | Solves | Infer |
|---|---|:---:|
| [`lfm2`](lfm2.md) ⤓ | text embeddings + fill-mask, 8k context | [x] |
| [`clip`](clip.md) | text/image embeddings | [x] |
| [`scrfd`](scrfd.md) | face detection | [x] |
| [`arcface`](arcface.md) | face identity embedding | [x] |

## Vision-language

| Architecture | Solves | Infer |
|---|---|:---:|
| [`fastvlm`](fastvlm.md) ⤓ | image captioning | [x] |
| [`qwen3vl`](qwen3vl.md) ⤓ | image + text → text | [x] |
| [`deepseek2ocr`](deepseek2ocr.md) ⤓ | document image → text/markdown (OCR, tables, grounding) | [x] |
| [`qwen3omnimoe`](qwen3omnimoe/readme.md) | text/audio/image/video → text, plus spoken output | [x] |

## Speech

| Architecture | Solves | Infer | LoRA |
|---|---|:---:|:---:|
| [`nemotronasr`](nemotronasr.md) ⤓ | streaming speech-to-text | [x] | [ ] |
| [`qwen3asr`](qwen3asr.md) ⤓ | offline speech-to-text | [x] | [ ] |
| [`qwen3tts`](qwen3tts.md) ⤓ | voice cloning / speech synthesis | [x] | [x] |
| [`cosyvoice`](cosyvoice.md) | zero-shot voice cloning TTS (LM + flow-matching mel decoder + HiFT vocoder) | [x] (CosyVoice 2 only - CosyVoice 3 forward-parity-proven, not yet composed into a pipeline) | [x] (LM only) |

## Music generation

| Architecture | Solves | Infer | LoRA |
|---|---|:---:|:---:|
| [`minimaxmusic3`](minimaxmusic3.md) | lyrics+caption -> full song (Qwen3-8B AR + flow-matching DiT + DAC vocoder) | [x] (unvalidated end-to-end on this machine - RAM) | library only |

## Image generation and editing

| Architecture | Solves | Infer | LoRA |
|---|---|:---:|:---:|
| [`s3dit`](s3dit.md) ⤓ | text-to-image | [x] | [x] |
| [`flux2`](flux2.md) | text-to-image + reference-image editing | [x] | [x] |
| [`codeformer`](codeformer.md) | blind face restoration | [x] | |
| [`rrdbnet`](rrdbnet.md) ⤓ | 4x super-resolution | [x] | |
| [`vqgan`](vqgan.md) | image ↔ codebook encode/decode | [x] | |
| [`imgpipe`](imgpipe.md) | composed segment → restore → upscale, one call | [x] | |

## Video generation

| Architecture | Solves | Infer | LoRA |
|---|---|:---:|:---:|
| [`wan`](wan.md) ⤓ | text-to-video (image-to-video not implemented) | [x] | library only |
| [`ltxv`](ltxv.md) | text-to-video+audio (two-stream A/V DiT) -- in progress, not yet runnable | [ ] | [ ] |

## Vision and 3D

| Architecture | Solves | Infer | Train |
|---|---|:---:|:---:|
| [`yolov8`](yolov8/readme.md) ⤓ | anchor-free object detection | [x] | [x] |
| [`zipdepth`](zipdepth.md) | monocular relative depth | [x] | [x] |
| [`sam2`](sam2.md) ⤓ | promptable image segmentation | [x] | |
| [`worldmirror2`](worldmirror2.md) | multi-view photos → 3D scene | [x] | |
| [`splat`](splat.md) | 3D Gaussian Splatting render/fit | [x] | [x] |

## Forecasting

| Architecture | Solves | Infer | Train | LoRA |
|---|---|:---:|:---:|:---:|
| [`chronos2`](chronos2.md) | probabilistic time-series forecasting | [x] | | |
| [`fincast`](fincast.md) | probabilistic time-series forecasting | [x] | | |
| [`kronos`](kronos.md) | OHLCV bar forecasting | [x] | [x] | [x] |

## World models and control

| Architecture | Solves | Infer | Train |
|---|---|:---:|:---:|
| [`diamond`](diamond.md) | playable, action-conditioned video (Atari-100k) | [x] | [x] |
| [`genieredux`](genieredux.md) | playable, action-conditioned video (CoinRun) | [ ] | [ ] |

## Not yet servable

These are parity-gated architecture ports with no CLI/HTTP/D-Bus serving
surface yet - real, verified code, but not something you can run as a model
today. Each has its own page with the full status:

- [`t5encoder`](t5encoder.md) - a text conditioner
- [`sdxlunet`](sdxlunet.md) - a diffusion backbone
- [`controlnet`](controlnet.md) - spatial conditioning for diffusion backbones
- [`flux1`](flux1.md) - text-to-image + image editing
- [`pulid`](pulid.md) - identity-conditioned image generation
- [`instantid`](instantid.md) - identity-conditioned image generation (SDXL)
- [`moondream3`](moondream3.md) - a third vision-language model

## Components

Not independently servable - reached only as part of another architecture's
composed pipeline: [`deepseek2`](deepseek2.md), [`sam1`](sam1.md),
[`mimi`](mimi.md), [`ecapatdnn`](ecapatdnn.md), [`autoencoderkl`](autoencoderkl.md),
[`s3tokenizer`](s3tokenizer.md) (`cosyvoice`'s FSQ speech tokenizer),
[`campplus`](campplus.md) (`cosyvoice`'s 192-d speaker encoder).

Notes on the columns:

- **LoRA** columns are omitted for model families where fine-tuning isn't
  offered at all (rather than shown as an empty box everywhere).
- **QLoRA is not implemented anywhere in brain** - INT8/GGUF paths are
  inference-only.

# Model catalog

Every model id below is what you pass to `brain do`, or set as the `model`
field over HTTP/D-Bus. A checked box means the capability is reachable through
a real, documented command — nothing here is aspirational. **⤓** means brain
fetches and converts the weights itself on first use; everything else needs a
local checkpoint, pointed at by a `BRAIN_*` variable named on the model's own
page (see [Getting the weights](../using/models-and-weights.md)).

## Text

| Model id | Solves | Infer | Train | LoRA |
|---|---|:---:|:---:|:---:|
| [`Qwen/Qwen3-0.6B`](qwen3.md) ⤓ | instruct/chat LLM, tool calls, paged-KV serving | [x] | [x] | [x] |
| [`brain/gpt`](gpt.md) | dense nanoGPT-parity baseline | [x] | [x] | [ ] |
| [`brain/glm`](glm.md) | mixture-of-experts decoder | [x] | [x] | [ ] |
| [`brain train`](moe.md) | sparse top-2-of-4 MoE toy next-token rule | [x] | [x] | [ ] |
| [`brain seq2seq`](seq2seq.md) | general encoder-decoder Transformer | [x] | [x] | [ ] |

## Embeddings

| Model id | Solves | Infer |
|---|---|:---:|
| [`LiquidAI/LFM2.5-350M`](lfm.md) ⤓ | text embeddings + fill-mask, 8k context | [x] |
| [`brain/clip`](clip.md) | text/image embeddings | [x] |
| [`brain/facenet`](face.md) | face detection + identity embedding | [x] |

## Vision-language

| Model id | Solves | Infer |
|---|---|:---:|
| [`brain/fastvlm`](vlm.md) | image captioning | [x] |
| [`brain/qwenvl`](vlm.md) | image + text → text | [x] |
| [`deepseek-ai/DeepSeek-OCR`](deepseek-ocr.md) | document image → text/markdown (OCR, tables, grounding) | [x] |
| [`brain/omni`](omni/readme.md) | text/audio/image/video → text, plus spoken output | [x] |

## Speech

| Model id | Solves | Infer | LoRA |
|---|---|:---:|:---:|
| [`brain/nemotron`](asr.md) | streaming speech-to-text | [x] | |
| [`brain/qwen-asr`](asr.md) | offline speech-to-text | [x] | |
| [`brain/tts`](tts.md) | voice cloning / speech synthesis | [x] | [x] |

## Image generation and editing

| Model id | Solves | Infer | LoRA |
|---|---|:---:|:---:|
| [`Tongyi-MAI/Z-Image-Turbo`](zimage.md) ⤓ | text-to-image | [x] | [x] |
| [`brain/flux2-klein`](flux2.md) | text-to-image + reference-image editing | [x] | [x] |
| [`brain/restore`](restore.md) | blind face restoration | [x] | |
| [`brain/upscale`](upscale.md) | 4x super-resolution | [x] | |
| [`brain/vqgan`](vqgan.md) | image ↔ codebook encode/decode | [x] | |
| [`brain/imgpipe`](imgpipe.md) | composed segment → restore → upscale, one call | [x] | |

## Vision and 3D

| Model id | Solves | Infer | Train |
|---|---|:---:|:---:|
| [`Ultralytics/YOLOv8`](yolo/readme.md) ⤓ | anchor-free object detection | [x] | [x] |
| [`brain/depth`](depth.md) | monocular relative depth | [x] | [x] |
| [`brain/sam2`](sam2.md) | promptable image segmentation | [x] | |
| [`brain mirror`](mirror.md) | multi-view photos → 3D scene | [x] | |
| [`brain splat`](splat.md) | 3D Gaussian Splatting render/fit | [x] | [x] |

## Forecasting

| Model id | Solves | Infer | Train | LoRA |
|---|---|:---:|:---:|:---:|
| [`brain/chronos2`](chronos2.md) | probabilistic time-series forecasting | [x] | | |
| [`brain/fincast`](fincast.md) | probabilistic time-series forecasting | [x] | | |
| [`brain/kronos`](kronos.md) | OHLCV bar forecasting | [x] | [x] | [x] |

## World models and control

| Model id | Solves | Infer | Train |
|---|---|:---:|:---:|
| [`brain wm`](world-models.md) | playable, action-conditioned video | [x] | [x] |
| [`brain pid`](pid.md) | control policy imitation | [x] | [x] |

## Not yet servable

These are parity-gated architecture ports with no `brain do`/HTTP/D-Bus surface
yet — real, verified code, but not something you can run as a model today.
Each has a public roadmap of what's left:

- T5-XXL encoder — a text conditioner
- SDXL UNet — a diffusion backbone
- ControlNet — spatial conditioning for diffusion backbones
- FLUX.1/Kontext — text-to-image + image editing
- PuLID — identity-conditioned image generation
- Moondream 3 — a second vision-language model (see [the VLM page](vlm.md))

Notes on the columns:

- **LoRA** columns are omitted for model families where fine-tuning isn't
  offered at all (rather than shown as an empty box everywhere).
- **QLoRA is not implemented anywhere in brain** — INT8/GGUF paths are
  inference-only.

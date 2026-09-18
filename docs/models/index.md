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
| [`t5encoder`](t5encoder.md) | T5-XXL / umT5-XXL text-conditioning embeddings | [x] |

## Vision-language

| Architecture | Solves | Infer |
|---|---|:---:|
| [`fastvlm`](fastvlm.md) ⤓ | image captioning | [x] |
| [`llava`](llava.md) | image captioning (also [SUPIR](supir.md)'s optional auto-caption input) | [x] (untested against real weights - see the model's own page) |
| [`qwen3vl`](qwen3vl.md) ⤓ | image + text → text | [x] |
| [`deepseek2ocr`](deepseek2ocr.md) ⤓ | document image → text/markdown (OCR, tables, grounding) | [x] |
| [`deepseekocr2`](deepseekocr2.md) | document image → text/markdown, global view only (no auto-fetch - no vendor GGUF exists yet) | [x] |
| [`qwen3omnimoe`](qwen3omnimoe/readme.md) | text/audio/image/video → text, plus spoken output | [x] |
| [`moondream3`](moondream3.md) | image + instruction → text (captioning) | [x] |
| [`florence2`](florence2.md) | visual grounding: a phrase + an image → a bounding box | [x] |

## Speech

| Architecture | Solves | Infer | LoRA |
|---|---|:---:|:---:|
| [`nemotronasr`](nemotronasr.md) ⤓ | streaming speech-to-text | [x] | [ ] |
| [`qwen3asr`](qwen3asr.md) ⤓ | offline speech-to-text | [x] | [ ] |
| [`qwen3tts`](qwen3tts.md) ⤓ | voice cloning / speech synthesis | [x] | [x] |
| [`cosyvoice`](cosyvoice.md) | zero-shot voice cloning TTS (LM + flow-matching mel decoder + HiFT vocoder) | [x] (both generations; CosyVoice 3 has no real-weight end-to-end run recorded) | [x] (LM only) |

## Music generation

| Architecture | Solves | Infer | LoRA |
|---|---|:---:|:---:|
| [`minimaxmusic3`](minimaxmusic3.md) | lyrics+caption -> full song (Qwen3-8B AR + flow-matching DiT + DAC vocoder) | [x] (no end-to-end run on real weights recorded yet - the five components together exceed the host memory it has been run on) | library only |

## Image generation and editing

| Architecture | Solves | Infer | LoRA |
|---|---|:---:|:---:|
| [`s3dit`](s3dit.md) ⤓ | text-to-image | [x] | [x] |
| [`flux2`](flux2.md) | text-to-image + reference-image editing | [x] | [x] |
| [`sdxlunet`](sdxlunet.md) | text-to-image (SDXL UNet backbone) | [x] | |
| [`flux1`](flux1.md) | text-to-image + Kontext image editing | [x] | |
| [`controlnet`](controlnet.md) | text-to-image conditioned on a control image (SDXL ControlNet) | [x] | |
| [`pulid`](pulid.md) | identity-conditioned image generation (FLUX.1) | [x] | |
| [`codeformer`](codeformer.md) | blind face restoration | [x] | |
| [`supir`](supir.md) | photo-realistic blind image restoration (SDXL + GLVControl + ZeroSFT/ZeroCrossAttn) | [x] (untested end to end on real weights - device memory, see the model's own page) | [x] |
| [`rrdbnet`](rrdbnet.md) ⤓ | 4x super-resolution | [x] | |
| [`vqgan`](vqgan.md) | image ↔ codebook encode/decode | [x] | |
| [`imgpipe`](imgpipe.md) | composed segment → restore → upscale, one call | [x] | |

## Video generation

| Architecture | Solves | Infer | LoRA |
|---|---|:---:|:---:|
| [`wan`](wan.md) ⤓ | text-to-video (image-to-video not implemented) | [x] | library only |
| [`ltxv`](ltxv.md) | text-to-video+audio (two-stream A/V DiT); the pipeline runs end to end as a **smoke test** only - the real 22B transformer and the Gemma-4 text encoder are not yet imported, so it does not generate usable video | [ ] | [ ] |

## Vision and 3D

| Architecture | Solves | Infer | Train |
|---|---|:---:|:---:|
| [`yolov8`](yolov8/readme.md) ⤓ | anchor-free object detection | [x] | [x] |
| [`zipdepth`](zipdepth.md) | monocular relative depth | [x] | [x] |
| [`sam2`](sam2.md) ⤓ | promptable segmentation: a mask from a click, on an image or tracked through a video | [x] | |
| [`worldmirror2`](worldmirror2.md) | multi-view photos → 3D scene | [x] | |
| [`splat`](splat.md) | 3D Gaussian Splatting render/fit | [x] | [x] |

## Forecasting

| Architecture | Solves | Infer | Train | LoRA |
|---|---|:---:|:---:|:---:|
| [`chronos2`](chronos2.md) | probabilistic time-series forecasting | [x] | | |
| [`fincast`](fincast.md) | probabilistic time-series forecasting | [x] | | |
| [`kronos`](kronos.md) | OHLCV bar forecasting | [x] | [x] | [x] |
| [`timesfm3`](timesfm3.md) | natively multivariate probabilistic forecasting | [x] | | |

## World models and control

| Architecture | Solves | Infer | Train |
|---|---|:---:|:---:|
| [`diamond`](diamond.md) | playable, action-conditioned video (Atari-100k) | [x] | [x] |
| [`genieredux`](genieredux.md) | playable, action-conditioned video (CoinRun) | [ ] | [ ] |

## A digital animal

Not a checkpoint port, and not servable - a real *Drosophila* connectome run as
a spiking network, driving a body in MuJoCo through the animal's own motor
neurons. What is gated at the end is a behaviour rather than a tensor.

- [the fruit fly](fly.md) - 178,860 neurons, 15,902,235 synapses, closed
  sensorimotor loop; see the page for what is and is not achieved.

## Toy architectures

brain's own tasks, with no upstream reference to import or parity-check
against. They are real - gradient-checked and benchmarked like anything else -
and they exist so a change to the engine can be tested against an architecture
that is fully understood. They are excluded from `brain caps` and
`brain --help`.

| Architecture | Solves | Infer | Train |
|---|---|:---:|:---:|
| [`toymoe`](toymoe.md) | sparse Mixture-of-Experts decoder (RMSNorm/RoPE, top-k) | [x] | [x] |
| [`toypid`](toypid.md) | PID event/effect transformer; backs the WebGPU browser demo | [x] | [x] |
| [`toyseq2seq`](toyseq2seq.md) | encoder-decoder transformer (bidirectional encoder + cross-attention decoder) | [x] | [x] |

## Not yet servable

These are parity-gated architecture ports with no CLI/HTTP/D-Bus serving
surface yet - real, verified code, but not something you can run as a model
today. Each has its own page with the full status:

- [`instantid`](instantid.md) - identity-conditioned image generation (SDXL)

## Components

Not independently servable - reached only as part of another architecture's
composed pipeline: [`deepseek2`](deepseek2.md), [`sam1`](sam1.md),
[`mimi`](mimi.md), [`ecapatdnn`](ecapatdnn.md), [`autoencoderkl`](autoencoderkl.md),
[`s3tokenizer`](s3tokenizer.md) (`cosyvoice`'s FSQ speech tokenizer),
[`gemma4`](gemma4.md) (`ltxv`'s text tower),
[`campplus`](campplus.md) (`cosyvoice`'s 192-d speaker encoder).

Notes on the columns:

- **LoRA** columns are omitted for model families where fine-tuning isn't
  offered at all (rather than shown as an empty box everywhere).
- **QLoRA is not implemented anywhere in brain** - INT8/GGUF paths are
  inference-only.

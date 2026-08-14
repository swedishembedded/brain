# BRaiN

![brain](docs/banner.jpg)

Modern AI infrastructure is fragmented. Knowledge is spread across many frameworks. Nothing can be optimized all in one place. 

Training usually means Python and PyTorch. Production inference means another
runtime. Edge deployment means another toolchain. Browser inference means
WebGPU. Intel NPUs mean OpenVINO. Distributed execution adds another layer
again. Every new model arrives with its own assumptions, dependencies, code, and
serving path.

**BRaiN replaces that stack with one engine.**

It trains and runs models locally, in pure Rust, across GPUs, CPUs, Intel NPUs,
and WebGPU - without embedding Python, PyTorch, CUDA, or another deep-learning
framework into the runtime.

The same primitives, model definitions, execution graph, and interfaces are used
from training through deployment. GPU kernels are implemented directly,
backpropagation is verified independently with finite-difference gradient
checking, and models are exposed uniformly through CLI, HTTP, and D-Bus.

## What brain solves today

* **One runtime from training to serving** - train, fine-tune, evaluate, and serve without moving the model between unrelated frameworks.
* **One hardware abstraction** - run the same model code on GPUs, CPUs, Intel NPUs, or in the browser through WebGPU.
* **One serving stack** - local models behind OpenAI-, Anthropic-, and OpenRouter-compatible APIs, with paged KV cache and continuous batching.
* **Training without PyTorch** - forward pass, backward pass, optimizers, and GPU kernels implemented directly in brain.
* **One interface across model types** - language, vision, speech, image generation, forecasting, and other models.
* **Fine-tuning on your own data** - including LoRA (support pending for more models)
* **Multimodal workloads without separate runtimes** - detection, depth, segmentation, recognition, restoration, upscaling, image generation, speech recognition, TTS, and voice cloning.
* **Distributed execution built into the engine** - data parallelism, pipeline parallelism, and tensor-parallel primitives over the same transport abstraction whether workers are threads, local processes, or machines on a network.
* **Weight streaming** - brain implements a residency engine that can efficiently stream weights and allows concurrent on demand model serving of multiple models at the same time with eviction control.

The goal of brain is to make model training and inference easily accessible, to
make it run on absolutely anything, and to keep all knowledge in one place and
to keep optimizing it relentless so that it becomes an extremely efficient AI
workload runtime that scales across both accelerators and nodes.

## Quick start

```bash
make release                          # build the optimized ./target/release/brain
make test                             # full test suite
make gradcheck                        # backprop correctness gate (finite differences)
```

Every architecture is reachable through one grammar: `brain <verb>
<architecture>` and `brain <architecture> <verb>` are the same command. For an
architecture with a known small default checkpoint, `infer` needs no other
flags at all -- brain fetches the weights, converts them, and runs:

```bash
brain infer qwen3 --prompt "The capital of France is"     # auto-fetches Qwen/Qwen3-0.6B
brain infer yolov8 --image photo.ppm                       # auto-fetches Ultralytics/YOLOv8
brain infer lfm2 --text "The capital of France is <|mask|>."  # auto-fetches LiquidAI/LFM2.5-350M
```

The first two are validated end to end on this box: `brain infer yolov8
--image ...` fetched Ultralytics/YOLOv8 and returned real detections; `brain
infer qwen3 --prompt ...` fetched Qwen/Qwen3-0.6B and generated a correct
completion. `brain infer lfm2` runs the identical code path (LiquidAI/LFM2.5-350M
is the smallest checkpoint of the three, ~650 MB) but its download did not
finish inside this session's network budget -- expect it to work the same way
qwen3 does, just note it hasn't been observed completing end to end here.

Every other architecture is reachable the same way once you point it at
weights (`brain caps <arch>` prints exactly what each takes; not every
architecture has a confirmed small default checkpoint to auto-fetch yet):

```bash
brain caps                                                  # every architecture + its actions
brain infer scrfd --in image=photo.ppm --json                # face detection (needs BRAIN_SCRFD_DIR)
brain infer glmdsa --weights F --prompt "..."                 # GLM-5.2 MoE decoder
brain qwen3tts synth --text "..." --out out.wav               # voice synthesis
brain zipdepth --image photo.ppm --weights zipdepth.pth       # monocular depth
```

Large architectures (Qwen3.5-35B-A3B, Qwen3-Omni-30B, FLUX.1/FLUX.2, SDXL) are
deliberately not part of this list -- they need tens of GB of disk and are not
something to fetch by accident. See their own pages in
[`docs/models/`](docs/models/index.md) for sizing before you run them.

See [`docs/introduction/quickstart.md`](docs/introduction/quickstart.md) for a
fuller walkthrough, including running a local API in one command, and
[`docs/using/cli.md`](docs/using/cli.md) for the complete CLI grammar.

## Model support

Every model `brain caps` reports, with what it does and where its full page
is. Toy architectures (`toymoe`, `toypid`, `toyseq2seq`, `toyautoencoder` --
brain's own tasks, no upstream reference) are excluded here; see
[`docs/models/index.md`](docs/models/index.md) for the complete catalog
including those.

| Model id | Domain | What it does |
|---|---|---|
| [`Qwen/Qwen3-0.6B`](docs/models/qwen3.md) | Text | dense decoder chat/tool-calling, paged continuous-batching serving |
| [`brain/qwen35moe`](docs/models/qwen35moe.md) | Text | Qwen3.5-35B-A3B hybrid GDN/GQA MoE decoder |
| [`gpt2`](docs/models/gpt2.md) | Text | nanoGPT-style baseline, from-scratch training reference |
| [`glmdsa`](docs/models/glmdsa.md) | Text | GLM-5.2 (MLA + sigmoid noaux_tc MoE + DSA) |
| [`LiquidAI/LFM2.5-350M`](docs/models/lfm2.md) | Text | bidirectional encoder, fill-mask + embeddings, 8k context |
| [`qwen3omnimoe`](docs/models/qwen3omnimoe/readme.md) | Multimodal | text/audio/image/video in, text + speech out (Thinker+Talker+Code2Wav) |
| [`brain/qwenvl`](docs/models/qwen3vl.md) | Multimodal | general image + text -> text |
| [`brain/fastvlm`](docs/models/fastvlm.md) | Multimodal | dedicated fast image captioning |
| [`deepseek-ai/DeepSeek-OCR`](docs/models/deepseek2ocr.md) | Multimodal | document image -> text/markdown |
| [`brain/nemotron`](docs/models/nemotronasr.md) | Audio | streaming speech-to-text (FastConformer + RNN-T) |
| [`brain/qwen-asr`](docs/models/qwen3asr.md) | Audio | offline speech-to-text |
| [`brain/tts`](docs/models/qwen3tts.md) | Audio | voice cloning / text-to-speech (Talker + MTP + codec) |
| [`Ultralytics/YOLOv8`](docs/models/yolov8/readme.md) | Vision | from-scratch anchor-free object detection |
| [`brain/depth`](docs/models/zipdepth.md) | Vision | monocular depth (pure-conv, realtime webcam) |
| [`brain/sam2`](docs/models/sam2.md) | Vision | promptable segmentation |
| [`brain/scrfd`](docs/models/scrfd.md) | Vision | face detection (boxes, scores, 5-point landmarks) |
| [`brain/arcface`](docs/models/arcface.md) | Vision | face identity embedding (512-d, cosine-ready) |
| [`brain/clip`](docs/models/clip.md) | Vision | text/image embeddings |
| [`Tongyi-MAI/Z-Image-Turbo`](docs/models/s3dit.md) | Image | text-to-image diffusion (S3-DiT) |
| [`brain/flux2-klein`](docs/models/flux2.md) | Image | text-to-image + editing (MMDiT) |
| [`brain/restore`](docs/models/codeformer.md) | Image | blind face restoration |
| [`brain/upscale`](docs/models/rrdbnet.md) | Image | super-resolution |
| [`brain/vqgan`](docs/models/vqgan.md) | Image | VQ autoencoder (CodeFormer's codebook) |
| [`worldmirror2`](docs/models/worldmirror2.md) | 3D | multi-view images -> 3D Gaussian Splatting scene |
| [`splat`](docs/models/splat.md) | 3D | 3D Gaussian Splatting viewer/renderer |
| [`brain/chronos2`](docs/models/chronos2.md) | Forecasting | probabilistic time-series forecasting |
| [`brain/fincast`](docs/models/fincast.md) | Forecasting | patched decoder + sparse MoE forecasting |
| [`brain/kronos`](docs/models/kronos.md) | Forecasting | OHLCV candlestick forecasting |
| [`diamond`](docs/models/diamond.md) | World models | playable, action-conditioned Atari-100k simulation |
| [`brain/imgpipe`](docs/models/imgpipe.md) | Vision | composable image-processing pipeline (D-Bus only) |

`brain caps --json` is the live, weights-free source of truth this table is
checked against (`tests/e2e/model_table_check.py`); the model id column names
what `brain caps`/D-Bus/HTTP actually serve today. Where that id still reads
as a pre-rename crate name (`brain/qwenvl`, `brain/depth`, `brain/tts`, ...)
rather than its architecture id (`qwen3vl`, `zipdepth`, `qwen3tts`), that is
this catalog's own internal dispatch key, not something you type -- the CLI
column you actually run is the architecture id (`brain infer qwen3vl`, `brain
zipdepth --image ...`, `brain qwen3tts synth ...`); unifying the two is
tracked follow-up work, not yet done.

## Where to go next

| | |
|---|---|
| **Full documentation** | [`docs/readme.md`](docs/readme.md) |
| Install & build | [`docs/introduction/install.md`](docs/introduction/install.md) |
| The `brain` command line | [`docs/using/cli.md`](docs/using/cli.md) |
| Every `BRAIN_*` environment variable | [`docs/using/configuration.md`](docs/using/configuration.md) |
| Model catalog | [`docs/models/index.md`](docs/models/index.md) |
| Scaling across GPUs | [`docs/scaling/overview.md`](docs/scaling/overview.md) |
| Performance | [`docs/performance/overview.md`](docs/performance/overview.md) |
| Kernel catalogue (generated) | [`docs/reference/kernels.md`](docs/reference/kernels.md) |
| Contributing to brain | [`AGENTS.md`](AGENTS.md) |

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

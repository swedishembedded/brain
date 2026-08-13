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

# Train + evaluate the GPT baseline end to end:
make data/calculator                  # generate a dataset
make train/gpt/calculator             # -> out/gpt-calculator.safetensors
make eval/gpt/calculator              # validation perplexity + task exact-match
```

See [`docs/introduction/quickstart.md`](docs/introduction/quickstart.md) for a
walkthrough, including running a real LLM behind a local API in one command.

## Supported models

A sample of what's covered — the full matrix (every model, what it supports,
and its own page) is [`docs/models/index.md`](docs/models/index.md).

| Task | Examples |
|---|---|
| Text | Qwen3 chat/tool-calling, a mixture-of-experts decoder, a nanoGPT-style baseline |
| Vision | YOLOv8-style detection, monocular depth, promptable segmentation, face recognition, document OCR |
| Image generation & editing | Text-to-image diffusion, face restoration, super-resolution |
| Speech | Voice cloning / TTS, streaming and offline speech-to-text |
| Forecasting | Probabilistic time-series and OHLCV forecasters, with LoRA fine-tuning |
| 3D | Multi-view reconstruction, 3D Gaussian Splatting |
| World models | Playable, action-conditioned video simulation |

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

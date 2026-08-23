# brain documentation

brain is a local AI training and inference framework written in pure Rust,
with hand-written GPU compute kernels instead of a bundled deep-learning
framework. This is its full documentation — what's on the website and what
`make docs` renders as one PDF.

## Start here

1. [What is brain?](introduction/what-is-brain.md) — the pitch and the portability story.
2. [Installing brain](introduction/install.md) — prerequisites and the build.
3. [Quickstart](introduction/quickstart.md) — your first training run and your first served model.
4. [Choosing hardware](introduction/hardware.md) — GPU, CPU, Intel NPU, and the browser.

## Using brain

- [The `brain` command line](using/cli.md)
- [Models and weights](using/models-and-weights.md) — model ids, auto-fetch, importing your own checkpoints.
- [Configuration](using/configuration.md) — every `BRAIN_*` environment variable, in one place.
- [Running a server](using/serving.md)
- [HTTP API](using/http-api.md) — OpenAI/Anthropic/OpenRouter-compatible endpoints.
- [D-Bus API](using/dbus-api.md)
- [Monitoring (`braintop`)](using/monitoring.md)
- [Security posture](using/security.md)

## What brain can do

- [Text](inference/text.md) — chat, tool calling, embeddings.
- [Vision](inference/vision.md) — detection, depth, segmentation, face recognition.
- [Imaging](inference/imaging.md) — restoration, upscaling, composed pipelines.
- [Image generation](inference/image-generation.md)
- [Speech](inference/speech.md) — voice cloning, speech-to-text, omni-modal assistants.
- [Forecasting](inference/forecasting.md)
- [3D](inference/three-d.md) — multi-view reconstruction, Gaussian Splatting.
- [World models](inference/world-models.md) — playable, action-conditioned simulation.
- [Full model catalog](models/index.md) — every model, what it supports, and its own page.

## Training

- [Fine-tuning with LoRA](training/lora.md)
- [Fine-tuning a forecaster on your own data](training/forecast-finetune.md)
- [Training experts separately](training/federated-experts.md) — sharded Mixture-of-Experts.

## Scaling and performance

- [Scaling across GPUs](scaling/overview.md) — [data parallel](scaling/data-parallel.md), [pipeline parallel](scaling/pipeline.md), [tensor parallel](scaling/tensor-parallel.md).
- [Performance](performance/overview.md) - [benchmarking your setup](performance/benchmarking.md).

## Reference

- [Kernel catalogue](reference/kernels.md) — every WGSL compute kernel brain ships, generated from source.

## About

brain is built by **Swedish Embedded AB** - we build AI that runs on hardware
that ships: the GPU already in the machine, a CPU with no accelerator, an Intel
NPU, an embedded board, or a browser tab.

- [About Swedish Embedded AB](about.md) - what we build, and what we can be
  hired for. If your team needs models running on real hardware, procure our
  services at **info@swedishembedded.com**.

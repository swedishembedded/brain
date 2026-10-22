# brain documentation

brain is a local AI training and inference framework written in pure Rust,
with its own WGSL and CUDA compute kernels instead of a bundled deep-learning
framework. These pages are the full documentation: the same files render on
GitHub, on the website at <https://swedishembedded.com/docs/brain>, and as one
PDF (`make docs/pdf`).

## Getting started

1. [What is brain?](getting-started/what-is-brain.md) - the pitch and the portability story.
2. [Installing brain](getting-started/install.md) - prerequisites and the build.
3. [Quickstart](getting-started/quickstart.md) - your first training run and your first served model.

## Guides

- **Inference:** [models and weights](guides/inference/models-and-weights.md), then by modality:
  [text](guides/inference/text.md), [vision](guides/inference/vision.md),
  [imaging](guides/inference/imaging.md), [image generation](guides/inference/image-generation.md),
  [speech](guides/inference/speech.md), [forecasting](guides/inference/forecasting.md),
  [3D](guides/inference/three-d.md), [world models](guides/inference/world-models.md).
- **Training:** [LoRA fine-tuning](guides/training/lora.md),
  [fine-tuning a forecaster](guides/training/forecast-finetune.md),
  [labeling](guides/training/labeling.md),
  [training experts separately](guides/training/federated-experts.md).
- **Hardware:** [choosing hardware](guides/hardware/choosing-hardware.md),
  [NVIDIA GPUs (CUDA)](guides/hardware/cuda.md),
  [benchmarking your setup](guides/hardware/benchmarking.md).
- **Serving:** [running a server](guides/serving/serving.md),
  [monitoring with `braintop`](guides/serving/monitoring.md),
  [security posture](guides/serving/security.md).

## Concepts

- [Architecture](concepts/architecture.md) - the crate layers, the backends and the model registry.
- [Performance](concepts/execution/performance.md) - profiling, kernel selection, INT8.
- [Scaling across GPUs](concepts/scaling/overview.md): [data parallel](concepts/scaling/data-parallel.md),
  [pipeline parallel](concepts/scaling/pipeline.md), [tensor parallel](concepts/scaling/tensor-parallel.md).

## Reference

- [The `brain` command line](reference/cli.md)
- [The Rust SDK](reference/sdk.md) - the embeddable `brain` crate: no CLI process, no server.
- [Configuration](reference/configuration.md) - every `BRAIN_*` environment variable.
- [HTTP API](reference/api/http.md) - OpenAI, Anthropic and OpenRouter compatible endpoints.
- [D-Bus API](reference/api/dbus.md)
- [Model catalog](reference/models/index.md) - every model, what it supports, one page each.
- [WGSL kernel catalogue](reference/kernels/index.md) and
  [native CUDA kernel catalogue](reference/kernels/cuda.md), both generated from source.

## Examples

[Samples](../samples/README.md) are standalone programs built on the SDK, each with
its own README, published under examples.

## Contributing, compliance and about

- [Contributing](contributing/index.md) - building, checks, and how these docs are built.
- [Third-party model licences](compliance/third-party-models.md)
- [About Swedish Embedded AB](about.md) - what we build, and what we can be hired for.
  If your team needs models running on real hardware, procure our services at
  **info@swedishembedded.com**.

# What is brain?

brain trains and runs neural networks **locally**, in Rust. There is no PyTorch
and no Python to build it, run it, or check that it learns correctly.

## One set of kernels, every kind of hardware

brain's operations are written once, as portable WGSL compute kernels, and the
same kernels run on four backends. All four (wgpu, Vulkan, CUDA and CPU) are
compiled into every native build and are chosen at run time with `--backend`:

- **wgpu** (the default GPU backend) - any modern GPU through Vulkan, Metal or
  DirectX 12, and the browser through WebGPU;
- **native Vulkan** - Vulkan driven directly;
- **CUDA** - NVIDIA GPUs through the NVIDIA driver: the portable kernels are
  translated to CUDA and compiled on first use, and the paths that matter most
  for speed have native CUDA kernels beside them
  ([Running on NVIDIA GPUs](../guides/hardware/cuda.md));
- **CPU** - the same kernels compiled to native code for the CPU's cores, so no
  GPU is needed at all.

A **web browser** is a build target of the wgpu backend: a wasm build runs the
same kernel source as the native one.

A portable kernel either gives the same answer on every backend or it is a
bug. Each native CUDA kernel is checked against the portable kernel it
replaces, or against a reference computation where it has no portable twin.

Separately, an **Intel NPU** path exists for a subset of models, opt-in per
model. It is not one of the four eager backends and does not share
the WGSL kernels - OpenVINO is a whole-graph compiler, so reaching the NPU
means exporting a model to ONNX, compiling it with OpenVINO, and running the
compiled graph, rather than dispatching brain's own kernels op by op. See
[Hardware](../guides/hardware/choosing-hardware.md) for which models have an NPU path today.

## Correctness without a Python oracle

Most from-scratch training code proves its backward pass is right by diffing
against a PyTorch reference. brain doesn't have one to diff against - instead,
the analytic gradients are checked against an in-repo
finite-difference gradient checker (`make test/gradcheck`), on its own forward pass.
That gate is what "correct" means here, and it runs as part of the normal test
suite.

## What it actually does

Under one consistent toolset - a CLI, an HTTP API (OpenAI/Anthropic/OpenRouter-
compatible), and a D-Bus surface - brain covers a lot of ground:

- **Large language models** - training from scratch, LoRA fine-tuning, and
  serving (chat, tool calls, batched inference with a paged KV cache).
- **Vision** - object detection, segmentation, monocular depth, face
  restoration, super-resolution.
- **Speech** - text-to-speech voice cloning and speech-to-text.
- **Image generation** - text-to-image and reference-image editing.
- **Time-series forecasting** - probabilistic forecasting over OHLCV and
  general time series.
- **3D** - multi-view reconstruction and Gaussian-splatting fly-throughs.
- **World models** - playable, action-conditioned video generation.

Every one of these is reachable the same way: `brain <architecture> <action>`
(or `brain <action> <architecture>` - both orders dispatch identically),
whether that architecture has its own dedicated CLI module or reaches the
generalized capability interface directly. See the model catalog for the
full, current list of models and what each supports.

## Who builds it

brain is built by **Swedish Embedded AB**. We build AI that runs on hardware
that ships - the GPU already in the machine, a CPU with no accelerator at all,
an Intel NPU, an embedded Linux board, or a browser tab. This engine is that
work done in the open.

If your team is trying to get a model onto real hardware, fit a large model
into a budget it does not currently fit, port a checkpoint to a runtime with no
Python underneath it, or write GPU kernels and prove they are still correct
afterwards, you can procure our services by sending an email to
**info@swedishembedded.com**. See [About Swedish Embedded AB](../about.md).

## Where to go next

- [Install](install.md) - build it.
- [Quickstart](quickstart.md) - train something, then run a real LLM.
- [Hardware](../guides/hardware/choosing-hardware.md) - picking a device (GPU/CPU/NPU).
- [The CLI](../reference/cli.md) - the full command map.
- [About Swedish Embedded AB](../about.md) - who builds brain, and what we can
  be hired for.

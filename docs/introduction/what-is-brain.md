# What is brain?

brain trains and runs neural networks **locally** - pure Rust, with hand-written
WGSL GPU kernels doing the compute. There's no PyTorch and no Python required to
build it, run it, or even to check that it learns correctly.

## One engine, three eager backends, plus a separate NPU path

brain is a self-contained engine of a few hundred WGSL compute kernels. The same
kernel source runs, unchanged, on three eager backends that all compile into
every native build and are selected at runtime:

- **wgpu** (the default GPU backend) - covering Vulkan, Metal, DX12, and
  WebGPU;
- **CPU-JIT**, where the same WGSL is JIT-compiled to native code via a
  Cranelift backend and runs across all your cores, no GPU required at all;
- **native Vulkan**, a separate eager backend targeting Vulkan directly
  (coopmat / tensor-core paths included).

A **web browser** is a build target of the wgpu backend, not a fourth
backend: compiled to wasm32, a wasm build carries only wgpu/WebGPU, running
the identical kernel source as the native wgpu backend.

Nothing is duplicated across these three: there is one implementation of each
op, and it either runs correctly everywhere or it's a bug.

Separately, an **Intel NPU** path exists for a subset of models
(`crates/npu`). It is not one of the three eager backends and does not share
the WGSL kernels - OpenVINO is a whole-graph compiler, so reaching the NPU
means exporting a model to ONNX, compiling it with OpenVINO, and running the
compiled graph, rather than dispatching brain's own kernels op by op. See
[Hardware](hardware.md) for which models have an NPU path today.

## Correctness without a Python oracle

Most from-scratch training code proves its backward pass is right by diffing
against a PyTorch reference. brain doesn't have one to diff against - instead,
every model's analytic WGSL gradients are checked against an in-repo
finite-difference gradient checker (`make gradcheck`), on its own forward pass.
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
- [Hardware](hardware.md) - picking a device (GPU/CPU/NPU).
- [The CLI](../using/cli.md) - the full command map.
- [About Swedish Embedded AB](../about.md) - who builds brain, and what we can
  be hired for.

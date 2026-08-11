# What is brain?

brain trains and runs neural networks **locally** — pure Rust, with hand-written
WGSL GPU kernels doing the compute. There's no PyTorch and no Python required to
build it, run it, or even to check that it learns correctly.

## One engine, three backends

brain is a self-contained engine of a few hundred WGSL compute kernels. The same
kernel source runs, unchanged, on:

- a **GPU**, through [wgpu](https://wgpu.rs) — covering Vulkan, Metal, DX12, and
  WebGPU;
- a **CPU**, where the same WGSL is JIT-compiled to native code and runs across
  all your cores, no GPU required at all;
- and inside a **web browser**, via WebGPU — the same kernels, compiled to wasm.

Nothing is duplicated per backend. There is one implementation of each op, and
it either runs correctly everywhere or it's a bug.

## Correctness without a Python oracle

Most from-scratch training code proves its backward pass is right by diffing
against a PyTorch reference. brain doesn't have one to diff against — instead,
every model's analytic WGSL gradients are checked against an in-repo
finite-difference gradient checker (`make gradcheck`), on its own forward pass.
That gate is what "correct" means here, and it runs as part of the normal test
suite.

## What it actually does

Under one consistent toolset — a CLI, an HTTP API (OpenAI/Anthropic/OpenRouter-
compatible), and a D-Bus surface — brain covers a lot of ground:

- **Large language models** — training from scratch, LoRA fine-tuning, and
  serving (chat, tool calls, batched inference with a paged KV cache).
- **Vision** — object detection, segmentation, monocular depth, face
  restoration, super-resolution.
- **Speech** — text-to-speech voice cloning and speech-to-text.
- **Image generation** — text-to-image and reference-image editing.
- **Time-series forecasting** — probabilistic forecasting over OHLCV and
  general time series.
- **3D** — multi-view reconstruction and Gaussian-splatting fly-throughs.
- **World models** — playable, action-conditioned video generation.

Every one of these is reachable the same way: a dedicated `brain <model>`
subcommand, or the uniform `brain do <model> <action>` entry point that works
identically across all of them. See the model catalog for the full, current
list of models and what each supports.

## Where to go next

- [Install](install.md) — build it.
- [Quickstart](quickstart.md) — train something, then run a real LLM.
- [Hardware](hardware.md) — picking a device (GPU/CPU/NPU).
- [The CLI](../using/cli.md) — the full command map.

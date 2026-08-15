# Hardware support

brain runs the same model code across a wide range of hardware: discrete
GPUs old and new, plain CPU cores, Intel NPUs, and even a browser tab via
WebGPU. This page covers what that portability rests on and how to pick a
device for your workload.

## Why one binary runs everywhere

brain's architecture is a **portable baseline plus capability-gated fast
tiers, with automatic fallback** - not a single fixed set of GPU features.
Every kernel is written once, in WGSL, against a portable fp32 baseline that
every backend (old desktop cards, current-generation GPUs, CPUs, and
browsers alike) can run unchanged. Faster numeric tiers - bf16, f16, INT8,
and block-quantized formats - are declared per kernel and dispatched only on
a device that actually supports them; a device that can't run a given tier
correctly falls back to the fp32 baseline rather than producing wrong
results or failing outright. brain's capability layer
(`backend-api::NumericSupport`) is what makes that selection mechanical
instead of guessed: it records what a backend measured, not what a device's
marketing claims, and a runtime selector reads it to pick the fastest tier
that's actually safe on the hardware present.

Beyond that baseline, `NumericSupport::int8_dot` is real and already
load-bearing, not aspirational: the packed-int8 DP4A dot kernels
(`matmul_i8*`, WGSL `dot4I8Packed`) execute on every wgpu-class backend, and
several models default to them - Qwen3.5's paged KV cache serves in int8 by
default, and `qwen3`/`s3dit`/`qwen35moe` all take `--precision int8` for a
~4× smaller weight footprint on the same card. `bf16`/`f16`/cooperative-matrix
tiers are the ones still gated off (`NumericSupport::BASELINE` reports them
unsupported and every such dtype promotes to fp32); wiring those is tracked
separately. So "portable fp32 baseline" describes the *arithmetic contract*
every kernel is written against, not a claim that fp32 is the only thing
brain computes in today - see `.agents/rules/architecture.md` for the same
correction against the (also easy to misread) "fp32-only" phrasing.

Within that fp32 baseline, the kernels also stay off a further set of GPU
features - no atomics, no subgroup operations, no kernel that needs a large
number of bindings - because those vary enough across old desktop cards,
current GPUs, CPUs, and browsers that relying on any of them would break
portability somewhere. That's a deliberately chosen baseline, not a ceiling
on what brain will ever compute in: it's what lets the exact same baseline
kernel source run on a decade-old GPU, a modern one, a CPU, or in WebGPU in a
browser tab today, while the capability-gated tiers above it are where
device-specific speed is meant to come from as they land.

## Listing your hardware

```
$ brain devices
canonical device registry (source: vulkan enumeration, PCI-bus order)
index  name        pci bus        uuid      vram      backends
gpu0   Tesla P40   0000:04:00.0   0c223d47  24.0 GiB  vulkan+wgpu
gpu1   Tesla P40   0000:82:00.0   ce637d48  24.0 GiB  vulkan+wgpu
ambient selection: none pinned - Gpu::new lands on gpu0 (Tesla P40)
```

`brain devices` prints the canonical GPU table: index, PCI bus id, name,
VRAM, and which backends can see the card.

GPU indices are stable and PCI-bus-ordered, so `gpu0` and `gpu1` always name
the same physical card, regardless of the order tools like `nvidia-smi`
happen to report them in, and regardless of driver updates or reboots.

## Selecting a device

Pass `--device` to any command that runs a model:

- `--device gpu0` - a specific GPU, by its canonical index.
- `--device gpu` - all GPUs.
- `--device cpu` - all CPU cores. brain JIT-compiles the same WGSL kernels
  to native code for the CPU backend, so no GPU is required at all.
- `--device npu` - the Intel NPU, via a separate compile path from the
  GPU/CPU kernels.
- `--device vulkan` - the native Vulkan backend instead of the default wgpu
  backend.

Device selectors can be combined as a comma-separated union, e.g.
`--device gpu1,cpu0-3` to use one GPU alongside a slice of CPU cores.

The `BRAIN_DEVICE` and `BRAIN_GPU_INDEX` environment variables select a
device the same way, ambiently, for cases where passing `--device` on every
invocation isn't convenient.

## Picking hardware for your workload

- **Local LLM serving** benefits most from more VRAM - it determines how
  large a model (and how much context) you can hold resident.
- **Training** benefits from more GPUs. See [Scaling](../scaling/overview.md)
  for how brain splits training across multiple devices.
- **CPU-only** works everywhere, with no GPU dependency, but is slower than
  running on a GPU.
- **The Intel NPU path** is opt-in per model today, not automatic - check a
  given model's own docs for NPU support before relying on it.

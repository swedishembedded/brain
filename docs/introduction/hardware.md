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

Today, that baseline is also the whole story on the compute side: every
backend executes fp32 only, so `NumericSupport`'s faster-tier fields all
report unsupported and every dtype currently promotes to fp32 regardless of
what a checkpoint declares. Wiring real bf16/f16/INT8/q4 compute paths
through the kernel set is in progress (tracked separately); this page
describes the direction the capability layer is built for, not a claim that
those faster tiers execute today.

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
ambient selection: none pinned — Gpu::new lands on gpu0 (Tesla P40)
```

`brain devices` prints the canonical GPU table: index, PCI bus id, name,
VRAM, and which backends can see the card.

GPU indices are stable and PCI-bus-ordered, so `gpu0` and `gpu1` always name
the same physical card, regardless of the order tools like `nvidia-smi`
happen to report them in, and regardless of driver updates or reboots.

## Selecting a device

Pass `--device` to any command that runs a model:

- `--device gpu0` — a specific GPU, by its canonical index.
- `--device gpu` — all GPUs.
- `--device cpu` — all CPU cores. brain JIT-compiles the same WGSL kernels
  to native code for the CPU backend, so no GPU is required at all.
- `--device npu` — the Intel NPU, via a separate compile path from the
  GPU/CPU kernels.
- `--device vulkan` — the native Vulkan backend instead of the default wgpu
  backend.

Device selectors can be combined as a comma-separated union, e.g.
`--device gpu1,cpu0-3` to use one GPU alongside a slice of CPU cores.

The `BRAIN_DEVICE` and `BRAIN_GPU_INDEX` environment variables select a
device the same way, ambiently, for cases where passing `--device` on every
invocation isn't convenient.

## Picking hardware for your workload

- **Local LLM serving** benefits most from more VRAM — it determines how
  large a model (and how much context) you can hold resident.
- **Training** benefits from more GPUs. See [Scaling](../scaling/overview.md)
  for how brain splits training across multiple devices.
- **CPU-only** works everywhere, with no GPU dependency, but is slower than
  running on a GPU.
- **The Intel NPU path** is opt-in per model today, not automatic — check a
  given model's own docs for NPU support before relying on it.

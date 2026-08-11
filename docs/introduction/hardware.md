# Hardware support

brain runs the same model code across a wide range of hardware: discrete
GPUs old and new, plain CPU cores, Intel NPUs, and even a browser tab via
WebGPU. This page covers what that portability rests on and how to pick a
device for your workload.

## Why one binary runs everywhere

brain's kernels are written once, in WGSL, and run unchanged on every
backend. That's only possible because the kernels are deliberately kept to a
conservative subset of GPU features: fp32 only (no f16), no atomics, no
subgroup operations, and no kernels that need a large number of bindings.
None of those features are exotic, but they vary enough across old desktop
cards, current-generation GPUs, CPUs, and browsers that relying on any of
them would break portability somewhere. Staying inside the common subset is
what lets the exact same kernel source run on a decade-old GPU, a modern
one, a CPU, or in WebGPU in a browser tab. This restraint is brain's core
hardware-portability pitch — it's also why a specific kernel occasionally
looks less clever than it could be on any one backend: it's optimized to be
correct and fast everywhere, not maximal on one target.

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

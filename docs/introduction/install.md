# Install

## Prerequisites

The only hard prerequisite is a **Rust toolchain** (install via
[rustup](https://rustup.rs)) — brain is a self-contained Cargo workspace, and
the core build/test path needs nothing else: no Python, no PyTorch, no
external C/C++ dependencies.

A few things are optional, needed only for specific paths:

- **A GPU with a Vulkan/Metal/DX12 driver** — for the GPU backend. Not
  required: brain also runs entirely on CPU (see [Hardware](hardware.md)).
- **Node.js 18+ and a WebGPU-capable browser** (Chrome/Edge 113+) — only for
  the in-browser WebGPU demo (`make web/dev`).
- **Python + OpenVINO** — only for the Intel NPU export/run path
  (`brain npu …`) and a couple of benchmark reference scripts. `make
  requirements` installs this tooling; the Rust engine itself never needs it,
  and `make build`/`make test` stay green without it installed.

## Build

```bash
make release          # build the optimized ./target/release/brain
```

`make build` also exists for an unoptimized debug build, but `make release`
is what you want for anything performance-sensitive — training and inference
both.

## Test

```bash
make test             # full test suite (unit + integration)
```

`make test` needs a GPU by default. To run the entire suite on CPU only:

```bash
BRAIN_DEVICE=cpu make test
```

This runs the same tests against the CPU backend (the same WGSL kernels,
JIT-compiled), so a machine with no GPU at all can still validate the full
build.

## Correctness gate

```bash
make gradcheck         # finite-difference backprop correctness gate
```

This is the check that stands in for a PyTorch reference: every model's
analytic WGSL gradients are verified against finite differences of its own
forward pass. Run it after any change that touches a model's forward or
backward pass.

## Next

- [Hardware](hardware.md) — choosing and selecting a device (GPU/CPU/NPU).
- [Quickstart](quickstart.md) — train something, then run a real LLM.

# Architecture

brain is a Cargo workspace of small, single-responsibility crates (about a
hundred). Each crate may depend only on crates in the layers above it, and
`scripts/gates/check-crate-layers.sh` enforces the layering where a violation
would be a hard Cargo cycle.

## Layers

1. **Kernels and the architecture registry.** `kernels` holds every compute
   kernel as WGSL source, embedded as constants and looked up by name; nothing
   duplicates kernel text. `arch` is the one table of architectures (id, display
   name, domain, Hugging Face class names, GGUF spelling, default checkpoint)
   that the CLI resolver, importers and model cards all read.
2. **The accelerator seam.** `backend-api` defines the backend traits.
   `gpu-core` is the device facade: one `Gpu`, buffer and step API, with the
   backend chosen at run time by `--device` and `--backend`. There is no
   per-backend model code.
3. **Training substrate.** Parameter and optimizer state, the checkpoint
   container (safetensors with a hashed manifest), data loading and tokenizers,
   the architecture-agnostic `Model` trait and trainer, paged KV cache, and the
   multi-GPU parallelism layer (see [scaling](scaling/overview.md)).
4. **Models.** One crate per architecture, composed from the shared kernels and
   blocks.
5. **Cross-cutting services.** Evaluation, benchmarking, the finite-difference
   gradient checker that gates every backward pass, federated expert shards, and
   the ONNX export plus Intel NPU runtime.
6. **Serving and front-ends.** Typed capability manifests shared by the CLI and
   the event API, the served-model catalog, residency (weights tiered across
   GPU, RAM and disk under a budget), the HTTP, D-Bus and stdio/TCP servers,
   the public SDK (`brain`) and the `brain` binary.

## Backends

Four eager backends are compiled into every native build and share the same
WGSL kernels:

| Backend | Crate | Notes |
|---|---|---|
| wgpu (default) | `backend-wgpu` | Vulkan, Metal, DirectX 12, OpenGL and WebGPU |
| Vulkan | `backend-vulkan` | Vulkan driven directly through `ash`; WGSL is compiled to SPIR-V |
| CUDA | `backend-cuda` | the CUDA driver API loaded at run time (no build-time CUDA dependency); WGSL is translated to CUDA C++ and compiled lazily, with native CUDA kernels in `kernels-cuda` for the hottest operations |
| CPU | `backend-cpu` | WGSL compiled to native code with a Cranelift JIT, spread across cores |

The Intel NPU is not a fifth backend. It is a separate whole-graph path
(`onnx` exports, `npu` runs through OpenVINO) that exists for a subset of
models and does not share the WGSL kernels. A browser is a build target of the
wgpu backend (`crates/web`).

The portable kernels are core-compute only: a single bind group, no atomics, no
subgroups and no `f16`, so the same source runs on old desktop GPUs and on
WebGPU. Activations are fp32; INT8 (including DP4A dot-product kernels) is a
first-class compute path rather than storage-only quantization.

## Conventions

- WGSL is the source of truth for kernels; a generated catalogue is published
  in the [kernel reference](../reference/kernels/index.md).
- Backpropagation is gated by `gradcheck` (finite differences), not by a Python
  oracle, and each native CUDA kernel is checked against the portable
  kernel it replaces.
- Every model is reachable the same way: `brain <model> <action>` on the CLI,
  and through HTTP, D-Bus and the SDK, because all of them dispatch the same
  typed capability manifest.

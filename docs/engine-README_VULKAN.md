# Native-Vulkan cooperative-matrix path (`vulkan-coopmat`)

A **separate**, opt-in execution path (ash + native Vulkan) that can run matmul
on NVIDIA tensor cores via `VK_KHR_cooperative_matrix`, with a runtime
capability query and a scalar fallback for GPUs without it (e.g. Pascal sm_61).

It does **not** replace the default wgpu/WGSL pipeline (`src/gpu.rs`). The
default build is unchanged and pulls in neither `ash` nor `naga`. WGSL remains
the source of truth: the scalar kernels are the same `src/shaders/*.wgsl` text,
compiled to SPIR-V at runtime by `naga`. Only the cooperative-matrix matmul is
authored separately, in GLSL (`src/shaders_vk/matmul_coopmat.comp`).

## Scope (what is implemented)

- `src/vulkan/context.rs` — `VkContext`: ash instance, physical-device
  selection, the `VkPhysicalDeviceCooperativeMatrixFeaturesKHR` +
  `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` capability query (M/N/K
  shapes + component types), logical device + compute queue, command-pool,
  host-visible buffer alloc/upload/download, and single-dispatch record/submit/
  fence-wait. Conceptually mirrors `Gpu` in `src/gpu.rs`.
- `src/vulkan/shader.rs` — `wgsl_to_spirv()`: WGSL → `naga::front::wgsl` →
  `naga::back::spv::write_vec` → SPIR-V, plus `vk::ShaderModule` creation. naga
  maps each WGSL `@group(N) @binding(M)` directly to a SPIR-V `DescriptorSet=N,
  Binding=M`, so the descriptor layout matches the kernel by construction.
- `src/shaders_vk/matmul_coopmat.comp` — GLSL compute shader using
  `GL_KHR_cooperative_matrix`, tile-based `out = x @ W^T`, f16×f16→f32
  accumulate (16×16×16 tile). The i8×i8→i32 variant is documented in the file
  header.
- `src/vulkan/matmul.rs` — `MatmulBackend::select()` (Coopmat vs Scalar),
  backend-agnostic `matmul()`, and `cooperative_matmul_demo()` (the smoke
  entry).
- CLI (feature-gated): `moe pid vk-info`, `moe pid vk-matmul`.

**Out of scope (documented follow-up):** porting the full PID forward pass
(embed / layernorm / attention / SwiGLU / head / CE) onto this runtime. This
deliverable is the matmul + runtime + capability + fallback slice only. The
scalar-kernel bring-up (`wgsl_to_spirv` + the shared descriptor layout in
`build_pipeline`) is the reusable foundation for that port: each `forward_steps`
dispatch in `src/pid.rs` maps to one `wgsl_to_spirv` + `build_pipeline` +
`dispatch`, with the same `[uniform@0, storage@1..]` bind-group convention.

## Backend selection

```
Coopmat  iff  device reports a usable f16×f16→f32 coop-matrix shape
              AND a precompiled matmul_coopmat.spv was baked in at build time
Scalar   otherwise   (WGSL matmul.wgsl via naga; runs anywhere, exact fp32)
```

The coopmat SPIR-V is only baked in when `build.rs` finds a GLSL compiler at
build time (see below). Absent that, or on a device without cooperative matrix,
the runtime silently uses the scalar fallback.

## Building

```sh
# default build is unaffected:
CARGO_HOME=/tmp/cargo-moe cargo build --release

# opt into the Vulkan path:
CARGO_HOME=/tmp/cargo-moe cargo build --release --features vulkan-coopmat
```

`build.rs` tries `glslc` (shaderc) then `glslangValidator` to compile
`matmul_coopmat.comp` → `matmul_coopmat.spv` into `OUT_DIR`. If neither is on
`PATH` it prints a `cargo:warning` note and **skips** (it never fails the
build); the tensor-core kernel is then unavailable and the scalar fallback is
used.

### Installing a GLSL compiler (to enable the tensor-core kernel)

- **shaderc / glslc** (recommended): part of the Vulkan SDK
  (<https://vulkan.lunarg.com/sdk/home>), or `apt install glslc` /
  `pacman -S shaderc` / `brew install shaderc`.
- **glslangValidator**: `apt install glslang-tools`, or from the Vulkan SDK.

After installing, rebuild with `--features vulkan-coopmat`; `build.rs` will
detect the compiler, emit `matmul_coopmat.spv`, and set the `have_coopmat_spv`
cfg so the SPIR-V is `include_bytes!`'d into the binary.

Manual compile (for inspection / a Slang-based pipeline):

```sh
glslc -fshader-stage=compute --target-env=vulkan1.3 -O \
  src/shaders_vk/matmul_coopmat.comp -o matmul_coopmat.spv
# or
glslangValidator -V --target-env vulkan1.3 -S comp \
  src/shaders_vk/matmul_coopmat.comp -o matmul_coopmat.spv
```

(Slang's `slangc` can also emit SPIR-V for cooperative matrix; the build.rs
detection can be extended to call it the same way.)

## Running

```sh
moe pid vk-info       # adapter + cooperative-matrix capabilities + chosen backend
moe pid vk-matmul     # run the matmul demo with whichever backend is selected
```

### What `vk-info` prints

- On **llvmpipe** (this dev machine) or **Pascal sm_61**: no
  `VK_KHR_cooperative_matrix`, feature disabled, no shapes, backend `Scalar`.
- On **NVIDIA Turing (sm_75) / Ampere / Hopper** with the SDK present:
  extension present, feature enabled, and a list of supported shapes, e.g.

  ```
  supported cooperative-matrix shapes (M x N x K  A*B->C/result  scope):
     16 x  16 x  16   f16*f16->f32/f32   sat=false  scope=Subgroup
     16 x  16 x  32   i8*i8->i32/i32     sat=false  scope=Subgroup
     ...
  f16*f16->f32 tensor-core usable: true
  ```

### What `vk-matmul` prints

- **Scalar** backend (llvmpipe / Pascal): exact result, `max abs error = 0`.
- **Coopmat** backend (NVIDIA + baked SPIR-V): result computed on tensor cores;
  expect ~1e-2 absolute error vs the fp32 reference from f16 rounding of the
  inputs. The demo computes `out = x @ I^T` (W = identity), so `out ≈ x`.

## What remains to validate on NVIDIA hardware

This machine has only software Vulkan (llvmpipe), which lacks cooperative
matrix, so **only the scalar fallback has actually executed here**. The
tensor-core path is correct-by-construction. To validate:

1. Install the Vulkan SDK (for `glslc`/`glslangValidator`) on an NVIDIA box.
2. `cargo build --release --features vulkan-coopmat` — confirm the build log
   shows `compiled matmul_coopmat.comp` (i.e. SPIR-V was baked in).
3. `moe pid vk-info` — confirm the f16×f16→f32 16×16×16 shape is listed and
   `f16*f16->f32 tensor-core usable: true`, backend `Coopmat`.
4. `moe pid vk-matmul` — confirm `max abs error` is small (~1e-2, f16 rounding).
5. Cross-check the coopmat result against the scalar backend for a few random
   matrices, and against `nn.Linear` in PyTorch, to confirm the `out = x @ W^T`
   layout and the column-major load of `W` (the `W^T` trick) are correct.
6. If the device only exposes a different supported shape, adjust `TILE` in
   `matmul.rs` and the `const TILE_*` in `matmul_coopmat.comp` to match a shape
   reported by `vk-info`.

After matmul is validated, the documented follow-up is to port the remaining
PID `forward_steps` kernels onto `VkContext` using `wgsl_to_spirv` +
`build_pipeline`, reusing the WGSL in `src/shaders/`.

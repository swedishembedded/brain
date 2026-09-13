# backend-cuda - roadmap

Ledger for native CUDA support. brain reaches NVIDIA hardware today through
WGSL -> SPIR-V (`backend-wgpu`/`backend-vulkan`); the goal is a real
`backend-cuda` on the CUDA Driver API, free to use warp intrinsics, DP4A,
tensor cores where present, stream overlap and CUDA Graphs - **across whatever
compute capability is actually plugged in**.

## The rule this whole effort is written under

**No hardware is hardcoded, anywhere.** Device count, compute capability, VRAM,
SM count and tensor-core presence come from a runtime query
(`cuDeviceGetCount`/`cuDeviceGetAttribute`/`cuDeviceTotalMem`), never from a
constant, a comment, a table or an entry in this file. Development and testing
currently happen against 2x Tesla P40 (cc 6.1, driver 570.195.03) - that is
*incidental test hardware and a citable measurement*, never a target, never a
baseline assumption, and never a permanent ceiling. Wherever a capability is
named below, read it as "the properties of whatever is attached, exemplified by
what was measured", and write any new gate so it is correct on a card with
tensor cores even though none is available to verify against today.

The single fixed constant in the CUDA code is NVIDIA's PCI vendor id, which is
a property of the vendor whose driver the library *is*, not of any card.

## Delivered

### Toolchain

CUDA 12.x toolkit installed from the distribution (`nvidia-cuda-toolkit`);
`nvcc --version` reports 12.2.140, and `libnvrtc.so.12` is present. **12.x is
required for Pascal-class cards** (CUDA 13 dropped them), but that is a
build-matrix concern, not a code assumption: nothing in the tree names a CUDA
version.

### `crates/backend-cuda` - device identity only

A new workspace member (default-members, like every other backend crate), with
`driver.rs`: `libcuda.so.1` is `dlopen`ed via `libloading` at run time, never
linked. `cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`,
`cuDeviceGetUuid`, `cuDeviceTotalMem_v2` and `cuDeviceGetAttribute` are
resolved AT their function-pointer types (not `transmute`d from `*mut c_void`),
and the load attempt plus `cuInit` happen exactly once per process with the
outcome - failure included - cached. A box with no NVIDIA driver gets an
ordinary `Err` naming the reason; nothing in the build path references a CUDA
symbol, so a machine with no driver and no toolkit still builds and tests
green.

`enumerate_physical_gpus()` returns `Vec<backend_api::GpuIdentity>` in CUDA
ordinal order, the same shape `backend_vulkan`/`backend_wgpu` already return.

### CUDA as a third device-identity registry source

`gpu_core::devices::registry()` now tries native Vulkan, then wgpu, then CUDA.
CUDA is a *fallback*, not a preference: both existing sources are empty on a
headless driver-only box, which would otherwise have an empty registry and no
`gpu0` at all.

Identity is keyed on `cuDeviceGetUuid`, which returns the same 16 bytes Vulkan
reports as `deviceUUID`. `devices::cuda_ordinal(index)` resolves a canonical
`gpu<i>` to the CUDA ordinal naming the same physical card, through
`GpuIdentity::same_device` - never through either enumeration's position, since
`CUDA_VISIBLE_DEVICES` renumbers CUDA's per process.

### `--backend`, a new flag over the existing `ComputeSet.backend` field

| Axis | Question | Values |
|---|---|---|
| `--device` | which hardware is schedulable | `cpu`, `gpu`, `npu`, `gpu0`, `gpu1,cpu0-3`, unions |
| `--backend` | how that hardware is driven | `wgpu` (default), `vulkan`, `cuda`, `cpu` |

`ComputeSet.backend` already existed, so this is a surface addition, not a
refactor. `Backend::parse`/`Backend::name` are the one spelling of the four
tokens; `ComputeSet::set_backend` applies the override without touching the
device set and refuses a GPU backend over a GPU-less set rather than demoting
silently. The CLI resolves the device set, applies the override, then
publishes - in that order, because the ambient `OnceLock` is first-writer-wins.

**An explicitly requested backend is a hard contract**: `--backend cuda` panics
naming exactly what is missing (no buffers, no kernel compilation, no dispatch)
instead of falling through to wgpu. That wrong-but-quiet fall-through was what
an unmatched match arm did before.

The `--device` grammar is deliberately UNCHANGED: `vulkan`/`wgpu` still parse
as backend-setting device tokens. `BRAIN_DEVICE=vulkan` is load-bearing in
`scripts/gates/parity-gate.sh` and documented as a performance instruction for
cards without resizable BAR, so retiring those tokens is a change of its own.

### Gate

`crates/gpu-core/tests/cuda_device_identity.rs` -
`cuda_uuid_matches_vulkan_device_uuid`: every CUDA-enumerated device UUID must
match a Vulkan/wgpu-enumerated device for the same physical card, and the
canonical index must resolve back to that CUDA ordinal. Skip-if-absent via
`brain_testutil::skip_unavailable` (no NVIDIA driver, or no independent
enumeration to cross-check against) - `make test` is unaffected on a
non-NVIDIA machine. Verified to have teeth by perturbing one UUID byte: the
test fails and prints both sides.

## Not delivered - what is still missing

**Device identity only. Everything that executes work is deferred.**

- **No `backend_api::Backend` impl.** No buffers, no `write`/`read`, no
  `step`, no caps. `--backend cuda` cannot run a model and says so.
- **No kernel source or compilation.** `crates/kernels-cuda` does not exist;
  no NVRTC path, no cubin cache, no lazy per-`kind` compilation.
- **No tier reporting.** `ImplChoice`, the `const POLICY` table, its coverage
  ratchet test, and `make cuda-table/check` are not written. Until they are,
  nothing can report *why* a given op ran on the implementation it ran on -
  and a generated tier that cannot be distinguished from a hand-tuned one is
  the silent-fallback defect class, not a cosmetic gap.
- **No WGSL -> CUDA generator**, and so none of its six known
  silent-wrong-number hazards is handled yet (early `return` before
  `__syncthreads` on the barrier-using kernels; WGSL 16-byte uniform layout vs
  C++ default; PTX shift clamping vs Cranelift's mask; NVRTC's default
  `--fmad=true` against a 1e-6 parity assertion; `__restrict__` vs deliberately
  aliasing `DeviceBuffer` clones; `var<workgroup>` zero-init vs `__shared__`).
- **No provider wiring, no CUDA Graphs, no allocator.**
- **`brain devices` does not show CUDA visibility per card.** The backends
  column still reports only `vulkan`/`wgpu`. The data is available
  (`devices::cuda_ordinal`); the column is not wired.
- **No `BRAIN_BACKEND` environment variable.** `--backend` is a flag only. If
  one is added it owes `docs/using/configuration.md` an entry
  (`check-env-docs.sh`) and must be read in the single `crates/gpu-core/src/`
  file that already reads `BRAIN_DEVICE` (`check-device-env-single-source.sh`).
- **CUDA is not in `scripts/gates/parity-gate.sh`**, and must not be until the
  catalogue and the backward kernels support it.
- **No cost formulas** for CUDA steps in `gpu-core/src/cost.rs` (a coverage
  ratchet), and `step_native` carries no `StepMeta`, so profiling would show
  `<no-meta>`.
- **memauth**: a CUDA device on the same physical card as a Vulkan device must
  share a `PoolId` or VRAM is double-counted. Not yet arranged - no CUDA
  allocation exists to double-count.

## Design decisions worth not re-litigating

- **CUDA kernels will live in `crates/kernels-cuda`, not `crates/kernels/cuda/`.**
  `scripts/build/kernels-regen.sh` derives const names as UPPER_SNAKE of a
  `.wgsl` stem, so `cuda/matmul.cu` collides with the existing `MATMUL`;
  `kernels::ALL` is asserted to compile on the test device kernel-by-kernel;
  `cost.rs`'s ratchet demands a cost formula per entry; and all five WGSL
  metadata cross-checks are WGSL-text-specific with zero purchase on CUDA C++.
  A sibling crate with its own registry and its own gate is cheaper than one
  script hosting two disjoint validators.
- **Caps must NOT report the honest numbers.** `max_storage_binding_bytes` is
  used as a tile-budget *divisor*, so reporting the card's real VRAM there
  makes chunked pipelines size slabs they cannot allocate. Keep `2 GiB - 1`,
  and expose a `cuMemGetInfo`-derived `max_buffer_bytes` separately.
- **Tier policy is a `const` + ratchet test, not a `.toml`.** A tier table is a
  status ledger; those live in `.agents/`, and a number nothing checks goes
  stale.
- **CUDA Graphs land last.** Keying the uniform allocation on
  `(kind, bufs, threads)` *excluding* params, with a pinned-staging
  `cuMemcpyHtoDAsync` as each step's first graph node and
  `cuGraphExecKernelNodeSetParams` for grid growth. Note `read()`/`poll_wait()`
  are illegal inside a captured region and brain reads logits every token.

## Notes for whoever picks this up

- Build only through the Makefile. A targeted run is
  `make test CARGO_TEST="cargo test --release --offline -p brain-gpu-core" TEST_THREADS="1 cuda"`.
- Keep GPU tests at the existing tiny-shape gradcheck/parity scale. This is a
  shared box; a sustained benchmark contends with whatever else is resident.
- `.agents/rules/lessons.md` #107 (the `_v2` symbol-name trap in any `dlopen`ed
  C API) and #108 (why a CUDA ordinal is not an identity) came out of this
  work and are the two things most likely to be re-learned the hard way.

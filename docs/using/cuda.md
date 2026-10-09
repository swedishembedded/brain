<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Running on NVIDIA GPUs (CUDA)

brain drives NVIDIA hardware through the CUDA Driver API. The backend loads
`libcuda.so.1` at run time (nothing is linked at build time), so a build never
needs a CUDA toolkit, and a machine without an NVIDIA driver simply does not
offer the backend. It runs on Linux x86-64 and Linux aarch64, including Grace
Hopper systems.

> Swedish Embedded AB implements GPU compute stacks and the validation that
> shows which parts of a model catalogue run on which accelerator. If your team
> needs expertise in bringing models to NVIDIA hardware, you can procure our
> services by sending an email to info@swedishembedded.com.

## What it needs

- **The NVIDIA driver.** That is all a model needs to *run* once its kernels
  are compiled.
- **NVRTC** (`libnvrtc`, part of the CUDA toolkit) to compile kernels the first
  time each is used; the result is cached on disk (`BRAIN_PIPELINE_CACHE_DIR`).
  Without it a machine can still run every catalogue kernel and native kernel
  from ahead-of-time images (below); a kernel with no image and no NVRTC is
  reported by name.

brain finds NVRTC in this order: the library named by `BRAIN_NVRTC` (the only
one tried when set), then the toolkit under `CUDA_PATH` (`lib64`, then `lib`),
then the system loader, trying `libnvrtc.so.13`, `.12`, `.11` and the
unversioned name.

### Compile targets

Kernels are compiled for the capability the driver reports, as real machine
code (`sm_90`, not a virtual architecture). A kernel that uses
architecture-specific instructions (Hopper's warpgroup MMA and TMA multicast)
declares it, and is compiled for the suffixed target (`sm_90a`) only on a
device and toolkit that provide it: a kernel that merely prefers the suffix
falls back to the plain target elsewhere, and one that requires it is declined
on any other device rather than failing inside the compiler. The on-disk
cubin cache is keyed on the target (suffix included), the compiler version, the
flags, any specialization macros and the launch ABI version, so changing any of
them recompiles instead of reusing a stale binary.

### Machines with a driver and no toolkit

`make cuda/aot` compiles the whole WGSL catalogue (through the CUDA
translation) and the hand-written kernels offline into one cubin per
architecture, plus portable PTX, and a manifest. The CUDA backend reads that
directory when it exists, so such a machine runs those kernels with the driver
alone:

```bash
make cuda/aot LANE=12     # sm_61 70 80 86 89 90 + PTX (Pascal through Hopper)
make cuda/aot LANE=13     # sm_90 100 120 + PTX (Hopper and newer)
```

Run both lanes to fill one directory for a mixed fleet; copy the directory to
the target machine and point `BRAIN_CUDA_AOT_DIR` at it. The images are build
products and are never written into the repository; the directory defaults to
`cuda-aot` under `BRAIN_PIPELINE_CACHE_DIR`. For one kernel on one device the
backend takes, in order: the cubin for the device's architecture (the
arch-specific one first when the kernel asks for it), a cubin for a lower minor
of the same major, the disk cache or NVRTC, and finally the PTX, which the
driver compiles itself. Images are matched by a hash of the generated source,
the entry point, the target, the flags and the launch ABI version, so a stale
image is never used for edited source, and each image is checked against the
manifest's checksum before the driver sees it. `BRAIN_CUDA_AOT=0` ignores the
directory. Kernels created at run time by dtype or KV-tier specialisation are
not in the catalogue and still need NVRTC.

### Installing a toolkit without root

`make cuda/install LANE=12` or `LANE=13` downloads NVIDIA's redistributable
archives (checksum-verified) into `$BRAIN_CUDA_PREFIX` (default
`~/.local/cuda`), with cuDNN and NCCL; `LANE=nsight` adds the profilers.
Activate a lane in a shell with:

```bash
eval "$(scripts/build/install-cuda-userspace.py env 13)"
```

CUDA 13 dropped offline compilation for Maxwell, Pascal and Volta, so those
cards need the 12 lane. A newer glibc than the toolkit was built against
conflicts with a few of its math headers; the installer patches them, so
`nvcc` works for host-side builds too.

## Choosing the backend

| How | Effect |
| --- | --- |
| `--backend cuda` | drive the selected GPUs with CUDA; fails by name if it cannot, never falls back |
| `BRAIN_BACKEND=cuda` | the same, for a process with no CLI (an embedding application, an older test binary); the flag wins |
| `cargo test ... -- --device gpu1 --backend cuda` | a native-kernel gate (`gpu_core::card_tests!`): the card and backend are its arguments, and without `--device` it opens no card |
| nothing | wgpu where Vulkan or wgpu sees the GPU; **CUDA where CUDA is the only API that does** (a driver-only machine); the CPU without a GPU |

`--device gpu0` still says *which* card; the backend says *how* it is driven.

`--profile-replays N` (a global flag, like `--device`) makes the passes that
support it - the FLUX.2 DiT forward, the Qwen3 encoder forward, the VAE
encode and decode graphs - also replay their recorded dispatches `N` times as
one captured program and print each kernel's device time at its per-dispatch
minimum over the replays. That is the ranking to trust on a GPU another
process is using: a time-sliced neighbour only ever lengthens a dispatch.

## What runs on it

Every kernel is written once, in WGSL. The CUDA backend translates each to CUDA
C++ and compiles it with NVRTC, so the same source runs on every backend and
the answers are held to the CPU reference. A kernel the translator cannot yet
express is refused **by name at the dispatch that needs it** rather than
approximated or run elsewhere. The translator handles scalars, vectors,
matrix products and determinants, structs (nested in the parameter block, as locals,
as helper arguments and results, and as records in a storage binding), arrays
as values, workgroup memory and barriers (including inside loops, when every
thread of the workgroup reaches them), user helper functions, and the
register-tiled matrix multiplies and fused attention kernels. Every kernel in
the catalogue is translated; `make cuda-coverage` below shows it for the
device in front of you.

An out-of-range array index never touches memory outside the array, as WGSL
requires of every implementation: the translated kernel clamps the index to the
last element of the buffer range it was bound to (the slice, for a sliced
dispatch), the same as the reference backends do. Kernels rely on this at
ragged tile edges, where they read a little past the end and mask the result.

A few hot kernels also have a hand-written CUDA version that replaces the
generated one on a CUDA device (`docs/reference/kernels-cuda.md` lists them).
The int8 decode GEMV reads weights at about three times the rate of the
generated kernel, and a handful of fused kernels each replace a whole chain of
small ones in a decode step - the residual add + norm + int8 quantiser, the
SwiGLU and attention-gate epilogues, a Gated DeltaNet layer's step, a gated-
attention layer's prep, and a layer's projections of one activation. All of
them return bit-identical results to the chains they replace.
`--no-native-kernels` (a global flag, like `--device`) keeps every dispatch on
the generated tier - the A/B switch the native kernels are measured with.

A repeated submission is also recorded into a CUDA graph and replayed, and a
decode step is replayed whole, from recycled scratch buffers, in a few chunks so
the card starts before the host has built the rest of the step.
`BRAIN_CUDA_GRAPHS=0` turns that off. `qwen35_decode_profile` reports the
per-token spread and the host calls per token (a steady-state token makes no
allocation and no individual launch).

`make cuda-coverage` generates and compiles every kernel for the device in
front of it and writes which are runnable and which are refused, and why, to
`cuda-coverage.json` (`OUT=path` to move it).

Floating-point contraction is switched off in the generated code, so a kernel
rounds the way the reference rounds; results agree with the CPU to
floating-point reduction-order differences, and many are bit-identical.

## Memory limits

A CUDA device reports four separate memory limits (`Gpu::memory_limits()`), because
one number cannot answer all four questions:

| limit | what it answers | CUDA value |
|---|---|---|
| allocation | largest single allocation | the device's free memory when asked |
| binding | largest range one dispatch binds | `min(device memory, 4 GiB - 1)`, the range a 32-bit index reaches |
| workspace | largest scratch or staging block a pass takes | a sixty-fourth of the device, 64 MiB to 1 GiB |
| working set | slab a tiling pipeline aims for; tile budgets derive from it | a sixteenth of the device, from 2 GiB - 1 up to the binding |

A device of 32 GiB or less keeps the slab it always had; a larger one tiles with a
larger slab, so a vocabulary head that needed two tiles fits in one. Buffers larger
than the binding are allocatable and are reached through sub-range bindings; the
generator does its address arithmetic in 64 bits, and a test addresses a 5 GiB
buffer past the 4 GiB mark. The Vulkan and wgpu backends keep the limits they
report.

## Memory placement

`Gpu::placement_facts()` reports what the driver says about sharing memory with
the host: managed memory, pageable-memory access (the device reading ordinary
host allocations), access through the host's page tables, native host atomics,
concurrent managed access, direct host access to managed memory and host
registration. They are driver attributes, not inferred from the integrated flag:
a Grace-Hopper node is not integrated and reports all of them.

Placement is opt-in. `Gpu::try_alloc_placed(label, bytes, policy)` takes
`AllocPolicy::Device` (the default; the only policy used for model weights and hot
buffers), `Managed` (the driver migrates pages between host and device) or
`System` (ordinary host memory used directly by kernels). A device that cannot
honour a policy refuses it; nothing is quietly allocated on the card instead.
`Gpu::prefetch` and `Gpu::advise` move or advise a managed or system range.
Managed memory is charged to the card's pool and system memory to the host's. The
buffers are zero-filled; managed and system counts have their own entries in
`backend_cuda::live_resources()`.

`make gh200/placement` (`BUFFER_MIB` sizes the buffer) measures the same workload
against each placement. On a Grace-Hopper node, 1 GiB, CUDA 13.0:

| | device | managed | system |
|---|---|---|---|
| GPU first-touch write (GB/s) | 1877 | 45 | 6 |
| steady read (GB/s) | 3736 | 3716 | 3725 |
| CPU first touch, GPU first read (GB/s) | | 5.6 | 202 |
| steady read of CPU-placed pages (GB/s) | | 3716 | 3716 |
| prefetch to device (ms) | | 11 | 124 |

The driver moves host-placed pages to the device as the GPU reads them, so repeated
reads converge on HBM speed and the cost is the first pass: faults for managed
memory, the link for system memory. A prefetch pays that cost up front. The
oversubscription run (`OVERSUBSCRIBE=1`, which takes all free device memory) is
available and was not run on a shared card.

### Counting HBM and Grace memory separately

The process memory ceiling (`--limit-vram-total`, `--limit-ram-total`) can account
for a second tier per card: host memory the card reads coherently. It is off by
default, so a ceiling governs the GPU-only run it always did. `BRAIN_COHERENT_TIER=1`
declares the host as every card's second tier; only a managed allocation
(`try_alloc_placed`) then spills into it when the card's ceiling refuses it, and
ordinary device allocations never do. `memauth::MemoryAuthority::request_tiered`
takes the tier policy directly (`DeviceOnly`, `AllowCoherentSpill`,
`CoherentOnly`), grants report the tier they are charged to, and
`residency::Budgets::set_coherent_tier` gives the residency planner the same
second budget, charged only by a policy that allows it.

## Pooled scratch allocation

A pass that repeats - a prefill round, a decode step's temporaries - frees and
allocates the same kind of block every iteration, and `cuMemFree` waits for the
device. While a handle is holding freed blocks (`Gpu::hold_freed_blocks`, which
`scratch_scope` switches on) its allocations come from a stream-ordered pool
(`cuMemAllocFromPoolAsync`, `cuMemFreeAsync`): a free is enqueued instead of
waited for, and any block that fits is reissued, whatever its size. The pool keeps
at most the retention cap of idle bytes (a sixteenth of the card, 256 MiB to
4 GiB; `BRAIN_CUDA_BLOCK_CACHE_MB` overrides) once its stream has run, returns all
of it when holding stops, and is destroyed with the handle. A block larger than
half the cap, a model weight, goes to the driver directly.

On a Grace-Hopper node one prefill-shaped round (268 blocks) cost the host about
2000 microseconds through the driver, 200 through the pool, and 32 through the
older exact-size cache on an idle device. `BRAIN_CUDA_MEMPOOL=0` selects that
cache, which is used automatically where the driver has no memory pools. Freeing
a block that no captured graph names no longer discards graphs.

## Memory is returned when a model is dropped

Everything the CUDA backend takes from the driver - device allocations, page-locked
staging, memory pools, events, streams, captured graphs and loaded kernels - is released when
the object that owns it is dropped, including while the device handle lives on
to serve the next model. A loaded Qwen3.8-27B returns its ~28 GiB when the
instance is dropped.

`backend_cuda::live_resources()` reports what the process currently holds, by
kind; two readings around a workload must be equal once everything the workload
created is gone. `cargo test -p brain-backend-cuda --test leaks` runs that check
over every owner, and `compute-sanitizer --leak-check full` over the same test
binaries reports no leaked allocations.

## Native kernel launch contract

A hand-written CUDA kernel registers with `NativeSpec::Cuda` and a
`CudaLaunch` that declares how it is launched, beyond its source:

- `dynamic_shared_bytes` and `shared_opt_in`: dynamic shared memory per block,
  and the opt-in to more than the 48 KiB every device grants by default. The
  device's queried limit is the ceiling; a registration above it is declined,
  not truncated.
- `cluster`: thread-block cluster dimensions (at most 8 blocks), launched
  through `cuLaunchKernelEx`. The block count of a dispatch must be a multiple
  of the cluster size. A device that cannot launch clusters declines the
  kernel.
- `scalars`: typed by-value arguments after the buffer pointers (32- and 64-bit
  integers and floats). Their values may change between dispatches, including
  inside a replayed CUDA graph.
- `arch`: whether the kernel prefers or requires the `sm_XYa` target.

`Gpu::native_max_active_blocks` returns the number of blocks of a registered
kernel one multiprocessor keeps resident, as the driver computes it for that
kernel's register and shared-memory use; times the multiprocessor count it is
the grid that fills the device. `CudaLaunch::NONE` is the plain launch.

## Container image for Arm servers

`make container` builds an aarch64 image for Grace-Hopper and other Arm + CUDA
servers from `tools/container/aarch64-cuda.Containerfile`. The runtime image
holds the `brain` binary and the ahead-of-time kernel images of both toolkit
lanes and no CUDA toolkit: the driver comes from the host through the
container runtime, and kernels load without NVRTC. `make container/ci` builds
the CI stage, which runs the gates that need no GPU (paths, scope, file size,
license headers, the kernel tables, `cargo check`, and the CUDA crates' tests
on the CPU), so a CI runner and a developer machine run the same recipe.
`make container/smoke` runs `brain devices` in the image on the host's GPU.
Set `CONTAINER_ENGINE` to choose between docker and podman and `BRAIN_IMAGE`
for the tag.

## Checking a machine

```bash
make test/cuda-matrix                       # every crate's tests, per-crate JSON report
make test/cuda-matrix CRATES="qwen3 gpt2"   # a few
brain devices                               # which cards brain sees
```

A test that needs a GPU says so and skips when there is none; the report's
zero failures on such a machine mean "did not run", which is why the report
lists passed and ignored counts as well.

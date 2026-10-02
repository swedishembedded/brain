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
  Without it the backend reports why and nothing runs on it: there are no
  ahead-of-time binaries.

brain finds NVRTC in this order: the library named by `BRAIN_NVRTC` (the only
one tried when set), then the toolkit under `CUDA_PATH` (`lib64`, then `lib`),
then the system loader, trying `libnvrtc.so.13`, `.12`, `.11` and the
unversioned name.

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
| `BRAIN_BACKEND=cuda` | the same, for a process with no CLI (a test binary, an embedding application); the flag wins |
| nothing | wgpu where Vulkan or wgpu sees the GPU; **CUDA where CUDA is the only API that does** (a driver-only machine); the CPU without a GPU |

`--device gpu0` still says *which* card; the backend says *how* it is driven.

## What runs on it

Every kernel is written once, in WGSL. The CUDA backend translates each to CUDA
C++ and compiles it with NVRTC, so the same source runs on every backend and
the answers are held to the CPU reference. A kernel the translator cannot yet
express is refused **by name at the dispatch that needs it** rather than
approximated or run elsewhere. The translator handles scalars and vectors,
workgroup memory and barriers (including inside loops, when every thread of
the workgroup reaches them), user helper functions, and the register-tiled
matrix multiplies and fused attention kernels. Not yet translated: a few
3D-reconstruction and splatting kernels that use nested structures.

An out-of-range array index never touches memory outside the array, as WGSL
requires of every implementation: the translated kernel clamps the index to the
last element of the buffer range it was bound to (the slice, for a sliced
dispatch), the same as the reference backends do. Kernels rely on this at
ragged tile edges, where they read a little past the end and mask the result.

A few hot kernels also have a hand-written CUDA version that replaces the
generated one on a CUDA device (`docs/reference/kernels-cuda.md` lists them).
Today that is the int8 decode GEMV, which reads weights at about three times
the rate of the generated kernel and returns bit-identical results.
`BRAIN_NO_NATIVE_KERNELS=1` keeps every dispatch on the generated tier.

`make cuda-coverage` generates and compiles every kernel for the device in
front of it and writes which are runnable and which are refused, and why, to
`cuda-coverage.json` (`OUT=path` to move it).

Floating-point contraction is switched off in the generated code, so a kernel
rounds the way the reference rounds; results agree with the CPU to
floating-point reduction-order differences, and many are bit-identical.

## Memory is returned when a model is dropped

Everything the CUDA backend takes from the driver - device allocations, page-locked
staging, events, streams, captured graphs and loaded kernels - is released when
the object that owns it is dropped, including while the device handle lives on
to serve the next model. A loaded Qwen3.8-27B returns its ~28 GiB when the
instance is dropped.

`backend_cuda::live_resources()` reports what the process currently holds, by
kind; two readings around a workload must be equal once everything the workload
created is gone. `cargo test -p brain-backend-cuda --test leaks` runs that check
over every owner, and `compute-sanitizer --leak-check full` over the same test
binaries reports no leaked allocations.

## Checking a machine

```bash
make test/cuda-matrix                       # every crate's tests, per-crate JSON report
make test/cuda-matrix CRATES="qwen3 gpt2"   # a few
brain devices                               # which cards brain sees
```

A test that needs a GPU says so and skips when there is none; the report's
zero failures on such a machine mean "did not run", which is why the report
lists passed and ignored counts as well.

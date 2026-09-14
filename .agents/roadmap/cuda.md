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

### Gate - device identity

`crates/gpu-core/tests/cuda_device_identity.rs` -
`cuda_uuid_matches_vulkan_device_uuid`: every CUDA-enumerated device UUID must
match a Vulkan/wgpu-enumerated device for the same physical card, and the
canonical index must resolve back to that CUDA ordinal. Skip-if-absent via
`brain_testutil::skip_unavailable` (no NVIDIA driver, or no independent
enumeration to cross-check against) - `make test` is unaffected on a
non-NVIDIA machine. Verified to have teeth by perturbing one UUID byte: the
test fails and prints both sides.

### Tier reporting, before any tuned kernel exists

The instrumentation that makes a silent slow fallback impossible landed
BEFORE the first native kernel, on purpose: numerically the tiers are
interchangeable, so a backend serving a mechanically translated kernel where a
hand-written one was promised is indistinguishable from a backend that is
simply slower than hoped. Four pieces:

**`backend_api::ImplSource`** - `Reference` (the portable WGSL) < `Generated`
(mechanically derived from it) < `Tuned` (hand-written, optionally specialised
to a queried capability). The ORDERING is the contract - `satisfies` is the one
comparison a policy makes - not an incidental derive.

**`crates/kernels-cuda`** - a source-only leaf crate (`brain-backend-api` only,
so it builds anywhere including wasm) holding the native kernels and their
metadata: name, op, tier, compute-capability floor, entry point, embedded
source. `best_for(table, op, cc)` resolves the HIGHEST floor at or below the
capability the caller queried. `check_table` states the per-entry invariants -
unique names, a tier a source file can honestly claim, an entry point that
exists in the text, a floor consistent with the instructions used (a body using
`__dp4a` may not declare a floor below the capability that introduced the
instruction), and no `__restrict__` unless deliberately exempted, since brain's
device buffers alias by design.

`ALL` is **empty**. Nothing can compile or launch a kernel yet, so a `.cu` file
here would be source nothing builds, runs or checks - the unverified claim this
whole mechanism exists to prevent. The registry, its invariants and its gate
exist first so the first real kernel lands into something that checks it.

**`ImplChoice` on the `OperatorProvider` seam** - `gpu-core`'s
`ProviderRegistry::resolve_choice` now returns, alongside the chosen provider,
a record of the op, the shape, the provider that RAN, its tier, the arch it was
resolved for, and one `Decline` per skipped provider carrying its own reason
(`Disabled` / `CapsUnmet(Requirement)` / `NotAccepted` / `LowerFailed(msg)`).
The decline reason is the datum that did not exist: `dispatch` previously
`tracing::warn`ed on a failed lower and discarded everything else at a
`continue`. `dispatch` attaches the record to every `Lowered`, and after a
failed lower it names the provider that actually ran, with the failure recorded
against the one that did not. `resolve_choice` touches no device, so a gate can
run it with nothing attached.

`ImplChoice::arch` reports the device class plus a compute capability that is
`None` until some backend publishes one - `DeviceCaps` has no such field and the
CUDA backend publishes no caps at all yet. `None` means *not reported* and may
never be read as a version or a default; `ArchTag::of` is the single place to
wire it when that changes.

**`backend_cuda::policy`** - `const POLICY: &[PolicyEntry]`, an entry being
"from compute capability `min_cc` upward, this op must reach at least this
tier". `required_in`/`violation_in` take the table as an argument so the rule is
testable independently of what the shipped table says. Not a `.toml`: a tier
table is a status ledger, and a ledger nothing reads rots. `POLICY` is
**empty** - no operator can honestly be required to be tuned while nothing can
run at all.

### Gate - the tier ratchet

`crates/gpu-core/tests/cuda_impl_policy.rs`. It lives in `gpu-core` because the
policy is in `backend-cuda` and `gpu-core` already depends on that crate, so the
test that joins a policy to a real dispatch record cannot live on the other
side. Seven assertions, none needing a GPU (one builds the CPU device):

- an op a policy declares `Tuned` that resolves to `Generated` is a violation
  naming both tiers and the capability it was judged at - the milestone's red
  test, driven by a synthetic fixture policy declared in the test file and
  marked as such, since the shipped one is empty;
- the same op resolved to a tuned impl satisfies it (so the ratchet is not
  vacuously red), and the portable reference does NOT - "it still computes the
  right answer" is what makes this failure invisible;
- a threshold applies upward only and the highest applicable one wins, asserted
  at capabilities no device here has, above and below every threshold;
- every skipped provider is recorded with its own reason, in chain order;
- a failed lower appears on the record as `LowerFailed`, not only in a
  `tracing::warn` nobody subscribes to;
- the shipped policy is backed by the shipped registry, with the count of
  operators under contract pinned (0 today). An equality rather than
  `cost.rs`'s floor, deliberately: a tier requirement is a reviewed performance
  contract, so having to edit the number is the point.

`make cuda-table` / `make cuda-table/check`
(`scripts/build/gen-cuda-kernel-table.py`, in `test/full`) regenerate and gate
`docs/reference/kernels-cuda.md` from the registry, and fail when a `.cu` file
exists that no entry embeds - which `include_str!` cannot catch, since it proves
registry -> file and never file -> registry. Verified to have teeth in both
directions (an unregistered source, and a registered kernel missing from the
page). A sibling gate, NOT a mode of the WGSL one: that generator derives const
names from `.wgsl` stems, compiles every entry on the test device, demands a
cost formula per entry and cross-checks five WGSL-text-specific fields - none of
which can read CUDA C++.

### `crates/wgsl-cuda` - the generated (T0) tier, for a named subset

A new leaf crate: naga IR in, CUDA C++ text out. It depends on `naga` and
nothing else - no driver, no toolkit - so it builds and its tests run on any
box, and the dependency edge to the backend stays one-way (backend -> generator).

One WGSL work-group is one CUDA block, so no index rewriting happens:
`blockDim = (@workgroup_size, 1, 1)`, `gridDim = (grid_x, grid_y, 1)`,
`local_invocation_id = threadIdx`, `workgroup_id = blockIdx`,
`num_workgroups = gridDim`. The entry point is `extern "C" __global__` (a flat
symbol table has no mangling to undo) and takes the uniform stream first, then
one pointer per storage binding in ascending binding order.

**Covered kernels (6 of 474):** `add2`, `mul`, `gelu`, `add_inplace`,
`quant_group_sum`, `gradnorm_part`. That is the milestone's bar and not a
coverage claim - breadth was the explicitly adjustable dial, and correctness on
a small set was spent instead. The six were chosen to reach every mechanism the
emitter has: elementwise f32, a math intrinsic (`tanh`), an aliasing in-place
binding, packed int8 with `dot4I8Packed` and u32/i32 arithmetic, and a
cooperative reduction with `var<workgroup>`, a barrier and a pre-barrier early
return. Anything outside the supported IR subset is an error naming what was
found; nothing is approximated.

### The six codegen hazards - which were EXERCISED and which were reasoned about

All six are handled, and five are held by a test that fails when the handling
is removed (verified by removing it, not by inspection):

| Hazard | Handling | Exercised by |
|---|---|---|
| early `return` before a barrier | guard flag + re-guarded statements, barriers outside the guard; plus a post-condition that a barrier-using kernel contains no `return` at all | **yes** - a padded grid where surplus work-groups return; caught by dropping the guard (measured: they wrote their output) |
| uniform layout | every member read at naga's own byte offset; no struct is transliterated | **yes** - a `vec3<u32>` member forces WGSL offset 28 where C++ packs to 16; caught by using a naive offset |
| shift semantics | `(amount) & 31u` written into the emitted code | **yes** - shift amounts straddling the word width; caught by removing the mask |
| FMA contraction | `--fmad=false`, and WGSL's explicit `fma()` still emits `fmaf` | **yes** - inputs asserted on the host to distinguish one rounding from two; caught by flipping the flag |
| `var<workgroup>` zero-init | zeroed by the whole block at entry, published by a barrier | **yes** - a reduction whose tail lanes never write their slot, run after a kernel that deliberately dirties the same shared window; caught by removing the zeroing |
| `__restrict__` | never emitted for any generated kernel | **partly** - the emitted text is asserted to carry none (caught by adding one), and an aliased-binding run agrees with the reference. A run that would MISCOMPILE under `__restrict__` needs a cross-invocation alias, which no kernel in the covered subset has, so that half is reasoned about rather than measured |

### Gate - golden agreement against the CPU reference

`crates/backend-cuda/tests/wgsl_cuda_golden.rs`: one WGSL source, two
independent code generators - `wgsl-cpu`'s Cranelift JIT and `wgsl-cuda` ->
NVRTC -> a real device - compared per kernel and per shape, at a whole-workgroup
shape and at one with a partial tail. Elementwise and packed-int8 results are
required to be **bit-identical** (tolerance 0); `tanh` and the reduction are
held to maxabs < 1e-6, the same floor the cross-backend parity assertions use.
Skip-if-absent, so a box with no driver or no NVRTC is unaffected.

### `backend-cuda` grew an execution substrate (not a `Backend`)

- `driver.rs`: the context/memory/module/launch entry points, resolved
  SEPARATELY from the identity ones and allowed to fail on their own - a driver
  missing one of them must still report device identity. `_v2` names where
  `cuda.h` defines them.
- `exec.rs`: `Context` (primary context retained per device, made current
  before every call), `DeviceMem`, `Module`, `Function`. `cuLaunchKernel` takes
  pointers TO the argument values, so a buffer argument is a pointer to the
  `CUdeviceptr`.
- `nvrtc.rs`: `libnvrtc` dlopened (it ships with the toolkit, not the driver,
  so a box that can RUN CUDA cannot necessarily compile it), compiling to a
  **cubin** for the capability the device reported - never a written-down one.
  The compile log is part of the error, because a generated kernel that does
  not compile is a defect in the generator.

The cubin cache key is a sha256 over length-prefixed fields: source, entry
name, compute capability, NVRTC version, and the exact compile flags. Files are
published with `rename(2)` from a process-unique temporary. `backend_api::cache_dir()`
is now the one cache-directory ladder; `gpu_core::tune::cache_dir` delegates to
it rather than keeping a second copy, since a backend crate cannot depend on
`gpu-core`.

### Two defects in the CPU reference, found by the comparison and fixed

The golden gate is only as good as the side it compares against, and it
immediately found `wgsl-cpu` wrong about the same two WGSL guarantees the CUDA
emitter has to handle:

- an invocation that returned before the barrier still ran the segment AFTER
  it, so every padded-grid guard was ineffective past the barrier. Fixed with a
  per-invocation "still running" mask in work-group scratch (a per-invocation
  SSA local cannot survive the split, which is what the existing `f6` check
  refuses);
- `var<workgroup>` was a stack slot allocated once and reused, so the second
  work-group read the first one's values. Fixed by zeroing it per work-group.

Both are pinned by `crates/wgsl-cpu/tests/workgroup_semantics.rs`, which fails
on the unfixed compiler. Neither was reachable from a model run today (the CPU
backend dispatches exactly `n_wg` work-groups, and the kernels in the tree
write every slot they read), which is why they survived.

## Not delivered - what is still missing

**Device identity only. Everything that executes work is deferred.**

- **No `backend_api::Backend` impl.** No buffers, no `write`/`read`, no
  `step`, no caps. `--backend cuda` cannot run a model and says so.
- **No kernel source and no compilation.** `crates/kernels-cuda` exists but
  its registry is empty, and nothing compiles CUDA at all: no NVRTC path, no
  cubin cache, no lazy per-`kind` compilation. `make cuda-table/check` is
  therefore a structure gate only - a declared kernel failing to COMPILE under
  NVRTC for its own declared floor is the check the plan wants there, and it
  needs a toolkit, so it is an addition to the existing checks, never a
  replacement.
- **No CUDA provider, so nothing yet produces a CUDA `ImplChoice`.** The
  record, the tier vocabulary, the policy and the ratchet are all in place and
  exercised against fixture providers; the first real producer arrives with the
  backend. Until then the shipped policy is empty by necessity, not by
  oversight.
- **Tier coverage is not surfaced anywhere.** `brain devices` shows no per-op
  tier coverage, there is no `--trace-impl` flag, and nothing carries the
  choice into `braintop` (which would go through a `DeviceBudget`-side field:
  the accelerator rows are built from budgets, not from a snapshot map). The
  record exists; nothing reads it back out yet.
- **`ProviderRegistry::dispatch` is still inert in production.** Only
  `Ops::matmul` routes through the seam, and `Ops::with_providers` remains
  test-only - so the `ImplChoice` now attached to every `Lowered` describes
  real dispatches only in tests. Wiring it up is the provider milestone's job.
- **The generator covers 6 of 474 kernels.** Everything with a vector or
  matrix type, a texture, an atomic, a `switch`, a function call, a
  multi-dimensional `@workgroup_size`, a barrier under control flow, a `return`
  inside a loop in a barrier-using kernel, or f16 is refused with a message
  naming the construct - refused, not approximated, but refused all the same.
  Vectors are the first thing any breadth work needs: the emitter is
  scalar-only today.
- **Nothing DISPATCHES a generated kernel.** The golden gate compiles and
  launches them directly; there is no provider, no `kind` -> kernel mapping, no
  lazy per-`kind` compilation cache in front of the cubin cache, and
  `kernels_cuda::ALL` is still empty (these are generated, not hand-written, so
  they do not belong in that registry).
- **`make cuda-table/check` still does not compile anything under NVRTC.** The
  machinery now exists (`nvrtc::compile`), but wiring a compile check into the
  gate means deciding what a box without a toolkit does, which is a gate
  question rather than a code one.
- **No ahead-of-time cubins.** NVRTC is the only compilation path, so a
  deployment with a driver and no toolkit cannot run the generated tier at all.
- **No provider wiring, no CUDA Graphs, no allocator.**
- **`brain devices` does not show CUDA visibility per card.** The backends
  column still reports only `vulkan`/`wgpu`. The data is available
  (`devices::cuda_ordinal`); the column is not wired.
- **No `BRAIN_BACKEND` environment variable.** `--backend` is a flag only -
  re-confirmed: the string appears nowhere in the tree, and
  `check-device-env-single-source.sh` still reports exactly one reader
  (`crates/gpu-core/src/devices.rs`, for `BRAIN_DEVICE`). If one is added it
  owes `docs/using/configuration.md` an entry (`check-env-docs.sh`) and must
  be read in that same single file.
- **The CUDA crates are not in the workspace overview yet.** Neither
  `AGENTS.md`'s crate table nor the layer diagram in `.agents/rules/architecture.md`
  names `backend-cuda`, `kernels-cuda` or `wgsl-cuda`. That is deliberate for
  now - those documents describe what a reader can USE, and none of the three
  runs a model - but it is an edit owed at the point the backend does.
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
  C API), #108 (why a CUDA ordinal is not an identity), #109 (why a ratchet
  that starts at zero cannot be a floor), #110 (`include_str!` proves registry
  -> file and never the reverse), #111 (record a skip where it happens), #112
  (the two WGSL work-group guarantees no target gives for free), #113 (an idle
  device makes an uninitialised-memory test pass), #114 (a generated tier is
  held to the REFERENCE's answer, not the language's) and #115 (materialise
  generated expressions where the IR says they are evaluated) came out of this
  work and are the things most likely to be re-learned the hard way.
- Adding the first tuned kernel is four edits, in this order: the `.cu` file,
  its `kernels-cuda` registry entry, `make cuda-table`, then the `POLICY` entry
  plus the contract count in the ratchet test. Doing the policy entry first
  makes the ratchet red, which is the correct order to discover in.
- Pre-existing and NOT caused by this work, so a run that shows them is still
  clean: `brain-backend-api`'s
  `select::tests::paged_attention_fused_only_offers_the_fused_kernel_at_causal_chunk_f32`
  fails on the base commit too, and `crates/gpu-core/tests/native_f16_provider.rs`'s
  three tests fail with "the `f16` extension is not supported in the current
  environment" on hardware without it.

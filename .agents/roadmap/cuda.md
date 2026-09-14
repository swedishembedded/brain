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

**An explicitly requested backend is a hard contract**: `--backend cuda` builds
the CUDA backend or panics naming the driver's own reason it could not, never
falling through to wgpu. That wrong-but-quiet fall-through was what an
unmatched match arm did before.

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

`ALL` was **empty** at this point, on purpose: nothing could compile or launch
a kernel yet, so a `.cu` file would have been source nothing built, ran or
checked. The registry, its invariants and its gate existed first so the first
real kernel landed into something that checked it - which is what then
happened (see the tuned-tier section below, and lesson #118 for the invariant
that caught the first kernel for a sentence in its own header).

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

### `backend_api::Backend` on the CUDA Driver API

`crates/backend-cuda/src/backend.rs` - `CudaBackend`. Allocations
(`storage`/`storage_init`/`buffer`/`uniform_dynamic`), host transfers
(`write`/`write_at`/`read`), recorded dispatches
(`step`/`step_sliced`/`step_buf`/`submit`), `poll_wait`, `kind` = `"cuda"`,
`identity` (the UUID-keyed one M0 established) and `caps`.

`Gpu::try_new_cuda` builds one; `--backend cuda` now builds one too, on the
card `--device gpu<i>` pinned if any. The placeholder that panicked
"cannot run kernels yet" is gone - the panic that replaced it fires only when
the backend genuinely cannot be built, and names the driver's own reason.

**Compilation is lazy per `kind`, and that is the point.** `backend-vulkan`
builds a pipeline for the whole registered catalogue at `Factory` time; a few
hundred NVRTC invocations up front would be minutes of cold start before the
first token. `CudaBackend` reads each kernel's `@workgroup_size` at
registration and compiles nothing; the first dispatch of a `kind` runs
`wgsl-cuda` -> NVRTC -> `cuModuleLoadData` and caches the module under that
`kind`, in front of the existing on-disk cubin cache. A kernel the generator
refuses is a **panic naming the kernel and the construct** at the dispatch that
needed it - not a fallback to another device, not an approximation.

Ordering is the context's default stream: every launch and every transfer
serialises against the ones before it, which is the same guarantee the wgpu
backend gets from the barriers it inserts between passes. `submit` is eager;
`read`/`poll_wait` are where the host synchronises.

Three questions the driver is asked that it was not asked before
(`cuDeviceGetAttribute` for max threads per block, shared memory per block and
warp size) plus `cuMemsetD8` and `cuMemGetInfo`. Nothing about a card is
written down.

### Caps - two different ceilings, deliberately answered differently

`max_storage_binding_bytes` stays the portable `2 GiB - 1`. It is read as a
tile-budget **divisor** (`model::block::tile_budget_words_for`, and the same
shape in `wan` and `s3dit`), so reporting a large card's real memory there does
not unlock a bigger binding - it makes those pipelines size slabs the device
cannot allocate.

`max_buffer_bytes` answers the other question (the largest single allocation)
and has no divisor semantics attached, so it reports `cuMemGetInfo`'s free
figure honestly.

Everything else in `DeviceCaps`/`ArchDesc` is a query: SM count, max threads
per block, shared memory per block, warp size, integrated-or-discrete.
`workgroup_reductions` is `true` (one work-group is one block; `__syncthreads`
is what a `workgroupBarrier` becomes). The roofline fields stay `None` -
measured, never queried. `f32` is `Native`; `i8` is **`Emulated`**, not
`Native`: `wgsl-cuda` writes `dot4I8Packed` out as a four-lane loop valid on
every capability rather than emitting the card's DP4A instruction, and
reporting `Native` would tell a selector this backend has dedicated int8
hardware behind that kernel when what it has is a loop. `f16` stays `Absent` -
the generator refuses `enable f16;` outright rather than widening it to fp32,
so no f16 arithmetic runs here whatever the card could do.

### Gate - one model's forward, on every backend this box has

`crates/gpt2/tests/cuda_backend_parity.rs` -
`the_cuda_forward_agrees_with_every_other_backend_on_this_box`. A dense GPT
decoder at `GptConfig::tiny()` (vocab 65, 2 layers, d_model 32, 4 heads, ff
128) over `b=1, t=6`, forward logits only, compared CPU vs CUDA and Vulkan vs
CUDA at maxabs < 1e-6, plus per-position argmax equality. **Measured: 8.94e-8
on both pairs**, over 390 logits. Skip-if-absent on both the CUDA driver and
the Vulkan ICD.

Why that model and that shape: `wgsl-cuda` refuses the register-blocked GEMMs
and the flash-attention family (a barrier inside a loop has no sound
guarded-body form), and `b * t` below `select::GEMM_TILE_MIN_ROWS` makes every
linear select the naive `matmul` **without the test forcing a kernel choice**.
The whole forward - `embed`, `pos_add`, `ln_stats`/`layernorm`, `matmul`,
`bias_add`, `attn_scores`/`attn_softmax`/`attn_apply`, `gelu`, `add2` - then
lands inside the supported subset. It is the smallest complete forward pass in
the workspace that does. Nothing is wired into `make parity`.

The generator's real reach turned out to be much wider than the six kernels M2
golden-tested: **448 of the 474** catalogue kernels generate. The 26 that do
not are 25 with a barrier inside a loop (every `matmul_*_reg*`/`matmul_tiled`/
`matmul_i8`/`flash_attn_*`/`paged_flash_*`) and one with a function call
(`matmul_kq_gemv_reg`). "Generates" is not "is correct" - only the six golden
kernels and this one model's forward have been held to the reference - but it
does mean breadth is now a validation problem rather than an emitter problem.

### Gate - the plumbing the model test cannot see

`crates/backend-cuda/tests/backend_contract.rs`, five assertions, added because
the model test above was MEASURED to be blind to four separate mutations of
this backend (see lessons #116). Four have verified teeth - each fails when the
mechanism it names is removed:

| assertion | mutation that makes it fail |
|---|---|
| a sliced step binds its sub-range, not the head | drop the `+ 4 * word_offset` |
| ... and the grid covers the whole window | lay the grid out at 256 instead of the kernel's 64 (an UNDER-dispatch; over-dispatching is invisible, every kernel self-masks) |
| a 256-wide kernel is launched 256 wide | launch `(64,1,1)` - and the group has to be wider than 64 elements, or a 64-thread block still reduces the right total |
| `write_at` starts at its word offset | drop the offset |
| new storage is zeroed | **none found** - see below |
| a kernel compiles on first dispatch and only once | (a regression to eager compilation is otherwise invisible) |

The zeroing assertion is stated honestly in its own doc comment as NOT known to
discriminate: with the zeroing removed the driver still returned zeros across
twenty dirty-free-reallocate rounds, because it scrubs a freed allocation
before reissuing it. That is a driver's courtesy, not an API guarantee, so the
zeroing stays and the test says what it is worth (lesson #117).

### The first tuned (T2) kernel, and the provider that dispatches it

`crates/kernels-cuda/cu/matmul_f32_tiled.cu` - fp32 `out = x @ Wt`, the
hand-written form of the portable `matmul.wgsl`, reading the identical
buffers and the identical `[m, k, n]` uniform. 64x64 output tile per block,
16-deep staged reduction, 256 threads each owning a 4x4 register block; both
operands staged through `__shared__` with one float of row padding so the
transposing stores do not serialise on a bank.

**Measured, on this box's cards (2x Tesla P40, cc 6.1, one of them also
running an unrelated job throughout), 512x512x512 f32, median of five trials
of eight dispatches:** the generated tier took 33.6 ms and the tuned kernel
779 us - a **43.1x** ratio. A second run under heavier contention measured
27.4x. The gate asserts a floor of 8x, well under both, because the number
worth defending is "the hand-written kernel was worth writing", not one box's
best minute; both sides contend equally so the RATIO is far steadier than
either absolute.

**And the delta was exactly 0 over 262144 outputs.** Not the tolerance the
milestone was planned around. The generated tier's problem at this shape is
not its arithmetic: neighbouring threads differ in the output COLUMN, so
their weight addresses are `k` floats apart and every lane of a warp touches
a different cache line on each of the `k` iterations. Fixing the access
pattern needed no reassociation, so the tuned kernel accumulates in exactly
the reference's order - one register, k ascending - and both tiers compile
with the same `--fmad=false`. The assertion is still the project-wide 1e-6
absolute bar rather than the zero observed: a later tuned kernel may
legitimately reassociate, and pinning the observed value would make that look
like a regression (lesson #119).

Registry metadata grew what a launcher needs and cannot read out of CUDA C++:
`block_dim`, the output `tile` a block covers, declared `shared_bytes`, and
`reported` (the `native:`-qualified name a dispatch record shows, pinned by
`check_table` to match `name`). The capability floor is
`BASELINE_MIN_CC` = 5.0 - the lowest a CUDA 12.x toolchain emits for at all,
and an honest statement about what this source uses rather than about any
card. `min_cc` may now be written as a NAMED constant; the catalogue
generator resolves `pub const NAME: Cc = (a, b);` from the registry itself.

### `gpu_core::provider::cuda::CudaProvider`

`OperatorProvider` for `Op::MatMul`, forward, plain f32 only. Modelled on
`coopmat` (the closest existing non-WGSL provider) but reaching the device
through a new `NativeSpec::Cuda { src, entry, block_dim, bindings,
shared_bytes }` - source text, never a pre-compiled image, because the
capability it is compiled for is whatever the backend queried from its own
device.

Three gates decide whether it runs, and the DEVICE answers all three: the
compute capability the driver reported (met against each kernel's declared
floor by `kernels_cuda::best_for`, highest eligible floor first); the
device's own queried threads-per-block and shared-memory-per-block limits;
and whether the backend can compile CUDA C++ at all. The last needs no
backend-name check - every other backend answers `None` to a
`NativeSpec::Cuda`, which is a recorded decline rather than a silence.

`accepts` refuses structurally, on the operand bundle, not just on the
declared dtype: a quantized tier binds five operands with two scale planes,
and feeding those to a three-pointer kernel would read a scale plane as
weights.

### `ArchDesc::compute_capability` - the queried fact this all keys on

`ArchDesc` grew `compute_capability: Option<(u32, u32)>`, filled by the CUDA
backend from `cuDeviceGetAttribute` and `None` on every other backend (they
describe themselves by feature bits; none has an ordered version number).
`provider::ArchTag::of` now reports it, so every `ImplChoice` carries the
capability its dispatch was resolved FOR - which is what the M1 tier ratchet
was built to read and previously could only get from a fixture.

### Native kernels are `Step`s like any other

`backend-cuda` implements `register_native`/`step_native`, plus a new
`Backend::step_native_sliced`. Slicing is not exotic - a model's activations
live at a row offset inside one buffer - so a native provider without it
could only ever serve a synthetic call site. The trait's DEFAULT declines any
non-zero offset rather than ignoring it: a backend with no native slicing
path answers `None` and the provider falls back, instead of silently reading
from the head of the buffer.

Native kernels occupy `kind` values from `kernels.len()` upward. Two things
that differ from the catalogue path and bit once each:

- `threads` is the BLOCK count for a native step and the INVOCATION count for
  a catalogue one. Dividing a block count by `block_dim` a second time
  launches a fraction of the blocks and leaves the output's tail unwritten -
  no crash, since every kernel self-masks (lesson #120).
- `n_bindings` counts STORAGE bindings only. A `NativeSpec`'s `bindings` list
  includes the uniform and a generated kernel's does not, so counting it made
  every dispatch look one buffer short.

Unlike the catalogue, a native kernel compiles EAGERLY inside
`register_native` - a compile failure has to be answerable with "this device
declined it", and by first dispatch the provider has already promised to
serve the request.

### Production wiring - `ProviderRegistry::for_gpu`

`Ops::with_selector` (and therefore `Ops::new`, and therefore every model)
now builds `ProviderRegistry::for_gpu` instead of `::reference`. It is the
ONE place a non-reference provider enters a real run, and everything it adds
is gated on a queried device fact rather than a build flag or an environment
variable. Today that is the CUDA provider, added whenever the device reports
a compute capability at all. On every device that reports none - the CPU JIT,
wgpu, Vulkan - the chain is still exactly the reference provider and the
constructor's behaviour is unchanged bit-for-bit.

`coopmat`, `native_f16` and `cpu_isa` are deliberately NOT in that chain:
adding a provider to it is a claim that a real forward pass through it was
checked on hardware that admits it, and none of the three has one.

### Gate - the tuned tier earns its place

`crates/gpu-core/tests/cuda_provider_matmul.rs`, three tests, skip-if-absent:

- the speedup floor AND the 1e-6 agreement, in one test, because a tuned
  kernel that is wrong is worthless and one that is right but no faster is a
  maintenance cost with a `Tuned` claim attached;
- the shared `provider::parity` case table (decode-shaped, the tile
  crossover, a multi-tile shape, a non-tile-multiple shape, a non-zero row
  offset), driven through the tuned provider against the WGSL oracle;
- the PRODUCTION registry, asserted from the dispatch record - `provider ==
  "cuda"`, `source == Tuned`, `arch.compute_capability` equal to what the
  device was queried for, `kernels == ["native:matmul_f32_tiled"]` - plus a
  host oracle on the output, because a record naming a kernel that produced
  garbage is worse than no record.

Largest allocation in the whole file is 1 MiB; the timed region is a few
milliseconds of device time.

### Four more operators onto the `OperatorProvider` seam

`Ops::embed`, `Ops::moe_linear`, `Ops::matmul_dx` and `Ops::matmul_dw` now
dispatch through `providers.dispatch` instead of calling `Gpu::step` by hand,
joining `Ops::matmul`. Three new `select::Op` variants carry them: `Embed`,
`MatMulDx`, `MatMulDw` (`MoeExpertLinear` already existed for the selector's
own use).

**This changed no dispatch and no number**, and that is the point: each of
the four has exactly ONE physical kernel shape per dtype in the WGSL
catalogue - no cooperative sibling, no register-tiled sibling, nothing for a
shape gate to switch between - so `candidates` returns `vec![Reference]` for
all three new Ops and the bound kernel is the one each `bind_*` already
picked. What changed is that a NON-WGSL provider can now answer them, which
it could not before however capable it was. WGSL remains the fallback for all
four, and `CudaProvider` declines every one of them today.

Two deliberate choices worth not re-deriving:

- **`MatMulDx` and `MatMulDw` are separate `Op`s, not `Op::MatMul` at
  `Pass::Backward`.** The two backward GEMMs of one linear are different
  computations over different operands - `dX` contracts over `n` against the
  weight, `dW` over `m` against the activation - so a provider asked for
  "MatMul, backward" could only tell them apart by inspecting the operand
  bundle, which is exactly the kind of implicit contract this seam exists to
  replace.
- **Every operand binds the WHOLE buffer (`range == (0, 0)`).** All four were
  on unsliced `Gpu::step` before, and a computed extent would have been a new
  claim rather than a move: a caller may legitimately hand any of them a
  buffer larger than the logical tensor, and a binding sized to `m * n` would
  have started refusing those.

`WgslProvider` grew `lower_fixed` alongside `lower_matmul`. The split is
about variant CHOICE, not importance: `Op::MatMul`'s thread count depends on
which variant the selector returned, the other four's is a function of the
output's extent alone. Both still ask the selector, so a cooperative sibling
landing for any of them is a `candidates` arm plus a `bind` arm and nothing
in the provider.

### Parity fixtures - what is covered and what deliberately is not

`provider::parity::cases_for` gained `Op::MatMulDx` and `Op::MatMulDw` (two
shapes each: one that divides the work-group size evenly and one that divides
nothing, since an off-by-one in the thread count only shows in the tail), and
the per-case comparison moved into one `compare` so every builder holds its
provider to the same bar. A host-oracle test sits alongside the self-parity
one: two providers can agree on a tail neither of them wrote, and only an
oracle sees that.

`Op::Embed` and `Op::MoeExpertLinear` have NO fixture, on purpose. A fixture
for either would be a second implementation of its semantics rather than a
shape - `Embed` needs a u32 index buffer bounded by a vocabulary the fixture
would also have to invent, and `MoeExpertLinear`'s output depends on the
VALUES of a routing gate, so a randomly seeded one would assert parity over
whatever subset happened to route. Both already have a host-oracle test
against the `Ops` façade in `crates/model` covering exactly the tiers they
ship, and `Ops::matmul_dx`/`matmul_dw` are additionally exercised by
`brain-gradcheck`'s finite-difference suite, which is on the parity gate.
When a non-WGSL provider claims either, the fixture it needs should be
written then, against that provider's actual operand bundle.

## Not delivered - what is still missing

**One model's forward, one tuned kernel. Backward, breadth and every other
tuned kernel are deferred.**

- **The tuned tier is ONE operator at ONE dtype.** Plain f32 `Op::MatMul`,
  forward. Every quantized weight tier (`I8`/`Q4`/K-quant), both backward
  GEMMs and every other operator are still answered by the generated tier -
  correctly, and visibly so in the dispatch record.
- **No DP4A / int8 tuned kernel.** The milestone was named for one. It was not
  written, and the reason is worth stating rather than leaving as an omission:
  the gate this tier had to clear is agreement with the fp32 WGSL reference to
  1e-6, which an int8 path does not answer at all, and the quantized operand
  bundle (five operands, two scale planes, a per-dtype `k` convention that
  differs between the i8 and q4 families) is a second, independent piece of
  work from the launch/registration plumbing this milestone built. The
  plumbing now exists and is proven, so a DP4A kernel is an addition to it -
  `DP4A_MIN_CC` is already the floor its registry entry would declare, and
  `best_for` would then prefer it over the f32 kernel on any card at or above
  that capability, which is the first time the capability-keyed resolution
  becomes observable in production rather than only in its own test.
- **`POLICY` is still empty, and now for a different reason than before.** A
  tuned kernel exists, but `PolicyEntry` has no dtype axis, so the only entry
  that could be written - "`Op::MatMul` must reach `Tuned`" - would also
  demand it of the quantized tiers and be false at the first quantized linear.
  The next change there is the dtype axis, not an entry. Until then the
  M1 ratchet stays exercised only against fixture tables, and
  `the_shipped_policy_is_backed_by_the_shipped_cuda_registry`'s `CONTRACTS`
  stays 0.
- **The tuned kernel is not in `make parity`.** It is covered by its own
  gate and by `parity-gate.sh`'s CUDA line (which runs that gate), not by the
  gradcheck package - that would demand the full catalogue and backward
  kernels, which this backend does not have.

- **Forward only, and one model.** `crates/gpt2/tests/cuda_backend_parity.rs`
  is the whole of what has been proved against a model: `GptConfig::tiny()`,
  `b=1, t=6`, forward logits. No backward pass, no decode/KV tape, no MoE, no
  attention variant beyond `attn_scores`/`attn_softmax`/`attn_apply`, no
  quantized weights, no second model. Everything the generator refuses -
  `matmul_*_reg*`, `matmul_tiled`, `matmul_i8`, `flash_attn_*`,
  `paged_flash_*`, `matmul_kq_gemv_reg` - is unreachable on this backend, so
  any shape that selects one of them fails by name at that dispatch. `b * t >=
  select::GEMM_TILE_MIN_ROWS` with an output width >= `GEMM_TILE_MIN_COLS` is
  exactly such a shape, which is why this gate's own shape is small.
- **448 of 474 kernels GENERATE; 7 have been checked against the reference.**
  The six of M2's golden gate plus what this model's forward touches. Assume
  nothing about the other 441: they emit plausible CUDA and nobody has compared
  the numbers. Breadth from here is a validation problem, not an emitter one.
- **No allocator and no uniform reuse.** Every `step` does a `cuMemAlloc` +
  `cuMemcpyHtoD` for its uniform stream and frees it when the step drops, and
  every buffer is its own `cuMemAlloc`. Correct, and the wrong shape for a
  decode loop; it is also what CUDA Graphs cannot capture (the plan's keyed
  `(kind, bufs, threads)` uniform with pinned staging is the replacement).
- **`cuModuleGetFunction` per launch.** The module is cached under its `kind`,
  the entry-point lookup inside it is not. A symbol lookup per dispatch is
  cheap next to a launch and free to fix; it is named here so it is not
  rediscovered as a mystery.
- **No hand-written kernel source.** `crates/kernels-cuda`'s registry is still
  empty: every dispatch this backend serves is `Generated`, and there is no
  `Tuned` entry for any op on any capability. `make cuda-table/check` is
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
- **The generator refuses 26 of 474 kernels.** 25 for a barrier inside a loop
  (`matmul_reg`/`reg2`/`reg3`/`reg4` and their `splitk`/`grouped`/`tn`
  variants, `matmul_tiled`, `matmul_dx_reg`, `matmul_dw_reg*`, `matmul_i8`,
  `matmul_i8_dyn`, `matmul_kq_dyn`, `matmul_q4_dyn_reg`, every `flash_attn_*`
  and every `paged_flash_*`), one for a function call
  (`matmul_kq_gemv_reg`). Refused by name, never approximated - but refused all
  the same, and they are precisely the fast kernels, so the generated tier is
  slower than Vulkan by construction. A barrier inside a loop needs a guarded
  form the emitter does not have; a `Call` needs function emission.
- **`make cuda-table/check` still does not compile anything under NVRTC.** The
  machinery now exists (`nvrtc::compile`), but wiring a compile check into the
  gate means deciding what a box without a toolkit does, which is a gate
  question rather than a code one.
- **No ahead-of-time cubins.** NVRTC is the only compilation path, so a
  deployment with a driver and no toolkit cannot run the generated tier at all.
- **No provider wiring and no CUDA Graphs.**
- **`brain devices` does not show CUDA visibility per card.** The backends
  column still reports only `vulkan`/`wgpu`. The data is available
  (`devices::cuda_ordinal`); the column is not wired.
- **No `BRAIN_BACKEND` environment variable.** `--backend` is a flag only -
  re-confirmed: the string appears nowhere in the tree, and
  `check-device-env-single-source.sh` still reports exactly one reader
  (`crates/gpu-core/src/devices.rs`, for `BRAIN_DEVICE`). If one is added it
  owes `docs/using/configuration.md` an entry (`check-env-docs.sh`) and must
  be read in that same single file.
- **CUDA is not in `scripts/gates/parity-gate.sh`**, and must not be until the
  catalogue and the backward kernels support it.
- **No cost formulas** for CUDA steps in `gpu-core/src/cost.rs` (a coverage
  ratchet), and `step_native` carries no `StepMeta`, so profiling would show
  `<no-meta>`.
- **memauth**: a CUDA device on the same physical card as a Vulkan device must
  share a `PoolId` or VRAM is double-counted. Still not arranged, and now it
  MATTERS: this backend allocates, so a process holding both handles on one
  card double-counts under `--limit-vram-total`. `Gpu::try_new_cuda` goes
  through `Gpu::wrap`, which resolves the pool from the ambient selection; a
  CUDA handle built on an explicitly pinned card does get `Device::Gpu(i)`
  through `Gpu::new_on`, but the two paths have not been reconciled with the
  Vulkan handle for the same card.

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
- The generated tier answers everything EXCEPT plain f32 `Op::MatMul`, which
  the tuned kernel now takes. Anything else that looks slow on this backend is
  expected to be: see the refused-kernel list above - every fast GEMM and
  every flash-attention kernel is among them.
- `.agents/rules/lessons.md` #107 (the `_v2` symbol-name trap in any `dlopen`ed
  C API), #108 (why a CUDA ordinal is not an identity), #109 (why a ratchet
  that starts at zero cannot be a floor), #110 (`include_str!` proves registry
  -> file and never the reverse), #111 (record a skip where it happens), #112
  (the two WGSL work-group guarantees no target gives for free), #113 (an idle
  device makes an uninitialised-memory test pass), #114 (a generated tier is
  held to the REFERENCE's answer, not the language's), #115 (materialise
  generated expressions where the IR says they are evaluated), #116 (a
  whole-model parity test is weak evidence about a backend's plumbing - with
  the MEASURED list of mutations it does not catch), #117 (some contract
  assertions cannot be given teeth, and must say so), #118 (a metadata gate
  that scans source text reads the header prose as code), #119 (a hand-written
  kernel can be much faster without reassociating anything) and #120
  (`threads` means invocations to a catalogue kernel and blocks to a native
  one), #121 (two providers agreeing on a tail neither of them wrote is not
  parity) and #122 (moving a call site onto a dispatch seam must not narrow
  its bindings) came out of this work and are the things most likely to be
  re-learned the hard way.
- Adding a tuned kernel is: the `.cu` file, its `kernels-cuda` registry entry,
  `make cuda-table`, and - once `PolicyEntry` has a dtype axis - the `POLICY`
  entry plus the contract count in the ratchet test. Doing the policy entry
  first makes the ratchet red, which is the correct order to discover in. The
  `CudaProvider` needs an edit only if the kernel serves an operator or an
  operand bundle it does not already accept.
- Pre-existing and NOT caused by this work, so a run that shows them is still
  clean: `brain-backend-api`'s
  `select::tests::paged_attention_fused_only_offers_the_fused_kernel_at_causal_chunk_f32`
  fails on the base commit too, and `crates/gpu-core/tests/native_f16_provider.rs`'s
  three tests fail with "the `f16` extension is not supported in the current
  environment" on hardware without it.

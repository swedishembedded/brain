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

Two toolkit lanes install into a user prefix without root
(`make cuda/install LANE=12|13`, `scripts/build/install-cuda-userspace.py`;
see `gh200.md`). **The 12.x lane is required for Pascal-class cards** (CUDA 13
dropped them), but that is a build-matrix concern, not a code assumption:
nothing in the tree names a CUDA version. NVRTC is found by `BRAIN_NVRTC`, then
under `CUDA_PATH`, then by SONAME (`.so.13`, `.so.12`, `.so.11`, unversioned).

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

### Memory limits - four questions, four answers

One `max_storage_binding_bytes` used to answer every memory question, and
because chunked pipelines read it as a tile-budget divisor, CUDA reported the
portable `2 GiB - 1` so they would not size slabs the device could not bind. The
limits are now separate (`backend_api::MemoryLimits`, `Backend::memory_limits`,
`Gpu::memory_limits`): the largest allocation (`cuMemGetInfo` free memory), the
bindable range (`min(total, u32::MAX)`: what a 32-bit element index reaches
whatever the element type; larger buffers are reached by sub-range bindings), the
workspace (a sixty-fourth of the card, 64 MiB to 1 GiB) and the working set (a
sixteenth of the card, from the old `2 GiB - 1` up to the binding), which is what
`model::block::tile_budget_words_for` and its siblings read. A card of 32 GiB or
less keeps its old slab. `max_storage_binding_bytes` stays as the binding.
Backends that do not tell the limits apart (wgpu, Vulkan) report their one
number in every field.

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

### The int8 decode GEMV - a native kernel behind the upgrade seam

Qwen3.8-27B INT8 decode streams 27 GiB of weights once per token and ~55% of its
device time was `matmul_i8_gemv_reg#MREG=1` reading them at about a quarter of
HBM bandwidth: the generated tier loads one 32-bit word per lane per step.
`crates/kernels-cuda/cu/matmul_i8_gemv.cu` is the hand-written replacement, and
it is the first native kernel that is dispatched through the *transparent
upgrade seam* (`gpu_core::native_upgrade`) rather than through an
`OperatorProvider`: the int8 linears bind `matmul_i8_gemv` by kernel index and
never go through `Ops::matmul`'s provider chain, so a provider would never see
them.

<!-- perf-number: measured on one GH200 (cc 9.0), see the ledger below -->
**What it does.** 16-byte `ld.global.nc` weight loads (L1 no-allocate), `__dp4a`,
and every load of a stage - weights, scales and the matching vectors of x -
issued before any is consumed. Sixteen threads serve one weight row and thread
`j` owns virtual lanes `4j..4j+3` of the WGSL kernel's 64-lane layout, so the
vector load IS the four lanes' words and the fold walks the 16 threads in
ascending order with warp shuffles. Products and sums are explicit
round-to-nearest operations. The result is **bit-identical** to the WGSL tier,
not within a tolerance, which is why every existing int8 gate passes unchanged.
Up to eight rows of x are served per weight pass (the WGSL `MREG` ladder up to
8); more rows keep the WGSL ladder, because a second tile row would re-stream
the weights from HBM.

**How it is selected.** `native_upgrade::resolve` runs after
`upgrade::resolve` for every `Gpu` handle and activates a row only when (1) the
WGSL upgrade for the same kernel is active, so the capability policy has already
chosen a workgroup-per-output GEMV, (2) the device reports a compute capability
and the registry has a kernel for this operator AND weight tier at or below it,
and (3) the backend accepts the source in `register_native`. Every other
backend answers `None` there, so wgpu, Vulkan and the CPU JIT are untouched
without a backend-name test. The decision per dispatch is `native_upgrade::
apply` (m in 1..=8, `kg % 8 == 0`); `Gpu::native_kernel_for` exposes it to
tests. `BRAIN_NO_NATIVE_KERNELS=1` pins the WGSL tier.

Two things the work needed in shared code: the registry resolved by operator
and capability alone, so `CudaKernel` gained a `weight` tier and `find`/
`best_for` take it (otherwise the fp32 provider would have been handed the
int8 kernel); and `register_native` appended a module per call to a registry
every sibling handle shares, so each `Gpu` built over an int8 model would have
loaded one more module for the life of the device - registration is now
idempotent per spec.

**Gates.** `crates/gpu-core/tests/i8_gemv_native.rs`: the redirect fires exactly
for the shapes it serves; raw-bit equality with the WGSL tier over tails, ragged
`n`, unaligned and aligned binding windows (the scalar-load path), sentinels
around every window, 60 seeded random shapes, and the real 27B projection
widths. A mutation of the fold order fails it. `i8_gemv_native_leak.rs`
builds and drops 24 handles that dispatch the kernel and requires the device to
return every byte; `compute-sanitizer --leak-check full` over both files
reported 0 errors and 0 bytes leaked.

**What was measured, and what it says** (device-timed with the backend's
per-launch events, each shape cycling >= 768 MiB of distinct weight copies so
nothing is read from L2; m = 1, `i8_gemv_native_bench`, best of 7):

<!-- perf-number: ledger of one measurement session on one shared GH200 -->
| shape (K x N) | WGSL GB/s | native GB/s |
|---|---|---|
| 5120 x 1024 | 317 | 538 |
| 5120 x 6144 / 6144 x 5120 | 807 / 796 | 1819 / 1954 |
| 5120 x 10240 | 910 | 2436 |
| 5120 x 12288 | 812-947 | 2517 |
| 17408 x 5120 (ffn down) | 1030 | 2828 |
| 5120 x 17408 (ffn gate/up) | 1003 | 2764 |
| 5120 x 248320 (lm head) | 950-1069 | 3752 |

The ceilings on this card, from `tools/gh200-probe`: 3818 GB/s read, 3437 GB/s
copy (read plus write counted), 3677 GB/s write. In the production dispatch
path the 100 MB shapes reach 72-74% of the probe's read ceiling (69-71% of the
nominal 4 TB/s) and the head 98% (94%). The smaller projections are short enough (tens of microseconds) that
a per-launch floor every kernel in this path pays (a trivial elementwise kernel
measures about 8 us in the same profile) is a large share of them; timed on
their own with back-to-back launches the same kernel reads 3.1-3.2 TB/s on the
100 MB shapes. At m = 8 the kernel is arithmetic-bound, not bandwidth-bound
(a dp4a, a convert, a multiply and an add per word per row, none of which may be
contracted if the bits are to match): 1.4-1.6 TB/s against the WGSL tier's
0.3 TB/s.

End to end, `qwen35_decode_profile 4` on the real Q8_0 checkpoint (one GH200,
shared with other jobs, so only the device-timed rows are compared): the GEMV
went from 32.2-32.7 ms/token (55-56% of device kernel time) to 12.8 ms/token
(34%); the whole production decode pass from 99-120 ms/token to 82-84 ms/token
(host-side time varies between runs, which is why the pass is not used to
judge the kernel). `cargo test --release -p brain-qwen35` (92 tests in the
main lane) passes unchanged.

Not done: the generated tier's other decode kernels (`rmsnorm_rows`,
`max_abs_rows`, `quant_pack`, `bmm`) are now the larger share of what is left;
the q4 and K-quant GEMVs have no native kernel; a weight-tier axis on
`PolicyEntry` still has to land before a contract can be written for this
kernel.

### CUDA Graphs - a repeated submission is captured once and replayed

`submit` no longer issues one driver call per dispatch when the submission's
shape repeats. The second identical submission is recorded into a CUDA graph;
every one after that is a single `cuGraphLaunch`.

**What that is worth, measured rather than assumed.** The driver-call
collapse is exact and asserted: at 64 dispatches per submission the host makes
128 calls (a parameter upload and a launch each) on the unbatched path and one
on the replay path. The host-TIME saving is smaller than that ratio, because
the calls are no longer the largest term: both paths first resolve every step
to its kernel and argument list and build the submission's signature to
compare against the captured one, and at this shape that bookkeeping now costs
more than the driver calls did. Replaying has measured between 0.62 and 0.78
of the unbatched host time, repeatably, and the test's bar is 0.9. Reducing
the bookkeeping - it allocates per step, per submission - is where the next
host-cost work is, and it would benefit both paths.

Four things had to be true, and each is asserted in
`crates/backend-cuda/tests/cuda_graphs.rs` rather than argued for:

1. **A per-step parameter change must not move a device address.** The uniform
   allocation is keyed on `(kind, buffers+offsets, parameter word count)` -
   the dispatch's structure, with the parameter VALUES excluded. A captured
   graph then gives every step its own slice of ONE graph-private parameter
   block, and the graph's first node copies the whole block from page-locked
   host staging in one go (see the decode section below for why it is one node
   and not one per step). Pinned because a copy from pageable memory is
   rejected during capture. The key also excludes `threads`, which the plan
   listed: the grid is not a property of the storage, and keying on it would
   hand every advancing position a fresh uniform address, which is exactly
   the event the next point exists to survive.
2. **A grid that grows must not force a second graph.** It is re-pointed with
   `cuGraphExecKernelNodeSetParams` inside the already instantiated graph.
   Node handles come from `cuStreamGetCaptureInfo_v2` read immediately after
   each recorded launch - `cuGraphGetNodes` returns nodes in an unspecified
   order and cannot say which node came from which launch.
3. **A free of a block the graph names must invalidate the graph.** `Context`
   carries a `graph_epoch` bumped in `DeviceMem`'s `Drop` when the block was
   marked as named by a captured graph, and a captured graph records the epoch
   it was captured at. (It started as a coarse `alloc_epoch` bumped by any
   free; that epoch remains, for sweeps of dead uniforms, but a prefill pass's
   temporaries freed between decode steps must not cost the decode graph its
   replay.)
4. **`read`/`poll_wait` must never land inside a captured region.** Capture
   begins and ends inside one `submit`, so a read can only fall between two
   submissions; kernel compilation, module loading and entry-point resolution
   are hoisted out of the region for the same reason. The confinement is
   checked, not asserted in prose: the backend refuses a `read` or a
   `poll_wait` that arrives while the capture flag is set (which can only be
   another thread), and one test drives the read-per-submission loop a decoder
   actually performs. Capture uses `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`, so
   the restriction never reaches brain's other threads, devices or backends.

The capture stream is created (`cuStreamCreate` with `CU_STREAM_DEFAULT`)
rather than being the legacy default stream, which cannot be captured. It is
created *blocking* so the synchronous host transfers that still use the legacy
stream stay ordered against it exactly as they were when everything shared one
stream.

Capture is **on by default** where the driver exposes the entry points, and
`CudaBackend::with_graph_capture(false)` turns it off. It is validated end to
end, not only in isolation: `crates/gpt2/tests/cuda_backend_parity.rs` runs a
real forward through this path and still reports `maxabs 8.940697e-8` against
both the CPU and the Vulkan backends, unchanged from before graphs existed.

Two counters that did not exist were built first, because the claim is not
observable without them: `CudaBackend::launch_stats` reports `host_launches`
(driver launch calls the host actually made, which `DeviceStats::dispatches`
cannot distinguish) and `host_nanos` (wall-clock inside `submit`, which never
waits for the device), alongside `graph_captures`, `graph_replays`,
`grid_updates` and `staging_waits`.

**`staging_waits` is the honest cost, and it is counted rather than hidden.**
A replay that must CHANGE something the in-flight graph is still using - new
parameter words in a staging block, or a new grid in the instantiated graph -
has to wait for the device first. Neither loop this exists for pays it: a
decoder's parameters change every token but it reads its logits between
submissions, which already drains the device, and a loop that resubmits an
unchanged shape has nothing to change. The counter is what makes a third kind
of caller - one that changes parameters every submission and never reads -
visible as a number rather than as unexplained slowness; it gets correct
answers at roughly the unbatched cost.

A handle keeps up to eight captured graphs rather than one, and that is a
measurement rather than a preference: instantiating costs milliseconds, so a
caller alternating between two shapes with a single slot re-instantiates on
every switch and ends up slower than launching each dispatch. (Four until a
decode token became several graphs of its own - see below.)

Two defects were found and fixed while doing this, both in code that predates
it: `CudaBackend` listed its `Context` as its FIRST field, so Rust dropped the
context before the device memory, modules and graphs that are resources of it
(it now drops last, and the field carries the reason); and every dispatch paid
a `cuModuleGetFunction` for an answer that cannot change while the module is
loaded, which is now resolved once at compile time.

A third was introduced by this work and caught by a whole-suite run rather
than a targeted one, which is worth recording because the targeted run could
not have caught it. Sharing the uniform allocation means the parameters can no
longer be uploaded when a step is recorded, so they moved into `submit` - and
a *synchronous* host-to-device copy runs on the legacy default stream, which
is ordered against every blocking stream in the context. One of those per
dispatch does not cost a driver call, it drains the device between every pair
of dispatches, and eight back-to-back dispatches of one kernel took an order
of magnitude longer than before. The uploads are now enqueued on the dispatch
stream from a pool of page-locked blocks that are lent to a submission and
returned when the device next drains (`StagingPool`). The unbatched path is
now faster than it was before this work, not slower: it also no longer
allocates a device uniform per step.

### Decode throughput - Qwen3.8-27B int8 on one card, from 81 to 12.4 ms/token

Measured with `qwen35_decode_profile` (real 27B Q8_0 GGUF, one GH200, a 12-token
prompt, best of several runs on a quiet window - the card is shared and a busy
neighbour doubles every number, so quote best-of and the spread it prints).
`nsys profile --trace=cuda --cuda-graph-trace=node` plus the sqlite kernel table
gives device-busy time against the token's span.

| stage | ms/token | kernels/token | notes |
|---|---|---|---|
| starting point (native int8 GEMV landed) | 81 | 2500 | device busy ~25 ms; 2500 alloc/free pairs, no graph |
| whole token one replayed graph | 32 | 2500 | arena + pass + one parameter copy |
| `add_rms_quant`, `quant_epilogue` | 24.5 | 1893 | 4 launches -> 1 at each norm boundary |
| `gdn_decode` | 17.0 | 985 | 19 kernels -> 1 per GDN layer |
| `gqa_decode_prep` + chunked issue | 14.5 | 865 | the card starts while the host builds |
| first chunk early | 14.2 | 865 | |
| multi GEMV | 14.0 | 629 | projections of one activation, one launch |
| `quant_epilogue` on 1024 threads | 12.4 | 629 | |

Device-busy time per token went 25 ms -> ~13.5 ms; the GEMVs are now ~9.4 ms of it
(27 GiB at about 2.9 TB/s against a ~7.7 ms floor at the 3.75 TB/s the LM head
reaches). What each step was, in the order it was found - the full account is in
knowledge #205 and #206:

1. **The graph path did not engage on the real tape.** A decode submits per
   layer and a capture needed one shape twice in a row, so a run made one
   capture. `begin_pass`/`end_pass` (`Backend`, `Gpu::pass_scope`) hold the
   per-layer submissions and issue the token as one; the capture trigger
   remembers a set of recent shapes (a token is the layer stack and then the
   head, alternating).
2. **Every temporary was a `cuMemAlloc` and a `cuMemFree`**, the latter waiting
   for the whole device and discarding every graph. The decode runs in named
   scratch arenas (`Gpu::scratch_scope_in`; the layer stack and the head share a
   handle and used to evict each other), the per-token inputs are recycled
   buffers that are written, and the attention scratch stride is bucketed to 128
   keys so it is constant token to token. `qwen35_decode_profile` prints
   `0 cuMemAlloc, 0 individual launches, N graph replays` per token as the check.
3. **One copy node per step cost ~12 us of device time between kernels**, enough
   that a replayed token was slower than launching each kernel. A graph now has
   one parameter block and one host-to-device copy.
4. **Native fused kernels** (CUDA only, selected by name through
   `Gpu::fused_step`, `BRAIN_NO_NATIVE_KERNELS=1` keeps the WGSL chains), each
   gated BYTE-for-byte against the chain it replaces on the same device:

   | kernel | replaces | gate |
   |---|---|---|
   | `add_rms_quant` | add2, rmsnorm_rows, max_abs_rows, quant_pack | `gpu-core/tests/add_rms_quant_native.rs` |
   | `quant_epilogue` | silu_mul / sigmoid+mul, max_abs_rows, quant_pack | `gpu-core/tests/quant_epilogue_native.rs` |
   | `gdn_decode` | 19 kernels of a GDN layer's step | `model/tests/gdn_decode_native.rs` |
   | `gqa_decode_prep` | split, QK norm, rope, KV append (8 kernels) | `model/tests/gqa_decode_prep_native.rs` |
   | `matmul_i8_gemv_multi` | up to 4 projections of one activation | `gpu-core/tests/i8_gemv_multi_native.rs` |

   and through the whole decode tape in `qwen35/tests/decode_fusion.rs`
   (`Qwen35::set_decode_fusion` is the A/B switch). Identity is achievable and is
   the bar because every one of them keeps the reference's own reduction order
   (the 64-lane sum of squares, the ascending L2/RMS sums, the contraction over
   the key index) and its own `expf`/`1/sqrtf`/`rint` expressions.
5. **The card sat idle while the host built the token** (~2.7 ms of recording and
   resolving ~860 steps in front of ~13 ms of device work). `flush` inside a
   pass issues the held steps once 64 (first chunk) / 256 (later chunks) are
   held, never by timing, so the chunks repeat and replay.

What is left, ordered by size: the GEMVs (~2 ms above the floor, mid-size
projections run at 2.7-2.8 TB/s), the SwiGLU epilogue (`silu`'s `expf` and two
IEEE divisions an element, ~5.7 us at 17408 wide), `gdn_decode` (16 blocks on 132
SMs, ~16 us), and the host's ~2.5 ms step build, which is hidden behind device
work except for the first chunk. Attention is the one part that scales with
context and is NOT yet fused: at 12 positions a token is 12.4 ms, at 1500 it is
~30 ms because the decode still runs the three-kernel score/softmax/apply triad
(one thread per key in the scores kernel) rather than the split-key flash decode
the WGSL catalogue has.

### Prefill on tensor cores - Qwen3.8-27B INT8, GH200 (cc 9.0)

Cold prefill of the resident INT8 GGUF was 155 tok/s at 512 tokens. The
per-kernel profile (`qwen35_gguf_prefill_profile`, device-timed) said 85% of it
was `matmul_i8_dyn`, a DP4A GEMM at ~10 TOPS on a part with int8 tensor cores,
and that the next three costs were the Gated DeltaNet chunk math
(`gdn_ut_step` x 63 launches per layer, the one-thread-per-output `bmm`) and the
scalar fp32 flash-prefill. Each was replaced behind a capability gate (CUDA
compute capability >= 8.0, `BRAIN_NO_PROVIDER=cuda` / `BRAIN_NO_GDN_FAST=1` /
`BRAIN_CUDA_BLOCK_CACHE_MB=0` switch them off), and the portable path stays the
fallback and the reference every new path is held to.

**Why not cuBLASLt.** The weights carry one scale per 32 elements of K (GGUF
Q8_0); a library int8 GEMM scales per output row or column. The scale can only
be applied where the int32 sum ends, which is every `k32` - one MMA. That is
not expressible as a library call, and requantising to per-row scales would
change what the model computes. So the GEMM is native (`matmul_i8_mma`,
`mma.sync.m16n8k32`): every MMA starts from an accumulator seeded with the bit
pattern of 1.5 * 2^23, so the integer result comes back as the float
`1.5 * 2^23 + i` (exact, `|i| <= 32*127*128`); one FADD removes the bias and one
FFMA applies the group scale into an fp32 running sum in ascending group order.
No int->float conversion, no library, no handle to create or leak, and it runs
inside the same stream, graphs and fences as every other step.

What changed, each gated by its own test (all skip without a CUDA device):

| piece | file | held to |
|---|---|---|
| int8 GEMM on int8 MMA, 64x64 tile, 4-stage `cp.async`, registry entry keyed by weight tier, routed through `CudaProvider` above the decode regime for group-32 scales and `K % 64 == 0` | `kernels-cuda/cu/matmul_i8_mma.cu`, `gpu-core/tests/cuda_provider_matmul_i8.rs` | the portable DP4A kernel on the real 27B shapes to 2e-5 of the output RMS (measured 2.4e-6, fp32 rounding of the group fold) and an f64 oracle |
| causal paged flash-prefill, `head_dim` 256, fp16 MMA, fp32 pool and accumulators; rounds of at most 32 rows (speculative verify, ragged tails) keep the fp32 kernel so those rounds still compute what the token-by-token tape computes | `kernels-cuda/cu/flash_prefill_f16_hd256.cu`, `gpu-core/tests/cuda_flash_prefill.rs` | the fp32 portable kernel and an f64 oracle to 1.2e-2 of the output RMS (measured worst element 5.8e-3: fp16 rounding of Q, K, V, P) |
| UT transform in one workgroup-per-matrix dispatch | `kernels/wgsl/gdn_ut_fwd.wgsl`, `model/tests/gdn_ut_fwd.rs` | the `gdn_ut_step` loop, bit for bit |
| 64x64-tile register-blocked batched matmul for `bmm`/`bmm_acc` | `kernels/wgsl/bmm_tiled.wgsl`, `model/tests/bmm_tiled.rs` | `bmm`/`bmm_acc`, bit for bit |
| the whole across-chunk recurrence in one persistent launch | `kernels-cuda/cu/gdn_chunk_loop_f32.cu`, `model/tests/gdn_chunk_fwd_fast.rs` | the nine-dispatch-per-chunk walk, bit for bit, at every chunk length 64..1 |
| a bounded block cache so a freed block is reissued instead of waiting in `cuMemFree` | `backend-cuda/src/exec.rs`, `backend-cuda/tests/block_cache.rs`, `qwen35/tests/cuda_leaks.rs` | counters back to baseline exactly (opt-in per handle; see lesson 207) |

`gpu_core::provider::cuda` carries the seam: the GEMM is a provider request;
flash-prefill and the chunk recurrence are drop-ins at their one call site
(`model::block::gqa_chunk_step`, `model::gdn::gdn_chunk_fwd`) through
`paged_flash_prefill_step` / `gdn_chunk_loop_step`, which answer `None` - and
the caller dispatches the portable kernel - for any decline. Kernels carry
their own binding list; `Gpu::native_kernel` offers a native kernel to the
backend once per handle.

A tensor-core stack also runs 1024-row prefill rounds instead of 256: what a
round pays per dispatch is a real fraction of it once the arithmetic is cheap
(table at `TENSOR_CORE_PREFILL_TOKENS`).

**Measured** (GH200, one card shared with other jobs, so the figures are the
fastest of repeated runs taken while the card was idle; `brain perf run longctx
--target qwen35-gguf --ladder 1 --steps 1`, "before" is the same binary with the
paths above switched off):

| prefill tokens | before tok/s | after tok/s |
|---|---|---|
| 512 | 149 | 1837 |
| 4096 | 145 | 1902 |
| 16384 | 130 | 1668 |

Per-kernel table of one 256-row round at depth 256 (device time, ms):
`matmul_i8_dyn` 1257 -> `matmul_i8_mma` 60.5; `bmm` + `bmm_acc` 91 -> `bmm_tiled`
7.2 + the recurrence kernel 11.1; `gdn_ut_step` x 3024 launches 42 -> `gdn_ut_fwd`
3.6; `paged_flash_prefill_hd256` 85 (2.1 TFLOPS) -> 4.5 (55 TFLOPS at 2k keys). A
round is device-bound again: 188 ms wall for 150 ms of kernels, the rest being
the ~7,800 stream operations (a launch and a parameter upload per step).

**Not done, in the order it would pay:**

- The per-step parameter upload. Every step is a launch preceded by a
  `cuMemcpyHtoDAsync` of its few uniform words, and the stream pays a bubble
  between the two (median 4.8 us over ~7,800 operations a round, ~40 ms).
  Passing the uniform as a `__grid_constant__` kernel argument would remove
  the copy, the uniform allocation and the per-step bookkeeping, but it changes
  the generated kernels' signature and the graph path's parameter update.
- The int8 GEMM is at ~225 TOPS and issue-bound on the group fold (eight ALU
  instructions per MMA); a wider warp tile and two k-tiles per barrier are the
  untried levers. Per-row weight scales (not what GGUF Q8_0 carries) would let a
  library GEMM run at several times that.
- The ~25 small elementwise kernels (layout permutes, row scales, norms, the
  quantise pair) are ~75 ms of a round, each bound by its own launch; fusing
  `max_abs_rows` + `quant_pack` and the GDN glue is where that goes.
- Flash-prefill is 55 TFLOPS, hd256 only, fp32 pool. An fp16 KV shadow would
  halve its staging and let it use `cp.async`.
- `head_dim != 256`, `dk/dv != 128`, group-16 (Q6_K) scales and chunks above 64
  decline to the portable path by construction.

### Dense convolution training - YOLOv8n fine-tune, P40 (cc 6.1)

`brain yolov8 fine-tune` at batch 8, 512 x 512 spent nearly all of its device
time in the three naive WGSL conv kernels (`conv2d`, `conv2d_dx`, `conv2d_dw`:
one thread per output, a serial walk of the whole reduction in global memory,
and a weight gradient with one thread per weight element reducing over every
position of the batch). `kernels-cuda/cu/conv2d_f32.cu` replaces all three:

| piece | how it is reached | held to |
|---|---|---|
| forward, implicit GEMM, 16/32/64-channel x 128-position tile, double-buffered 16-deep k slices | `native_upgrade` row for `conv2d` (same uniform, same bindings) | f64 oracle at the fp32 summation bound + bit-exact on integer data at the real geometry (`gpu-core/tests/conv2d_native.rs`) |
| input gradient, implicit GEMM per stride class (only the live taps of each `(h mod s, w mod s)` class) | `native_upgrade` row for `conv2d_dx` | same |
| weight gradient, implicit GEMM split over output positions, fixed-order reduction (no atomics) | `Gpu::conv2d_dw_steps`, called by `vision::blocks::Conv` (it needs a scratch plane the WGSL slot cannot carry) | same |

`BRAIN_NO_NATIVE_KERNELS=1` restores the WGSL kernels (the A/B switch).

<!-- perf-number: ledger of one measurement session on a shared P40 -->
Measured on one P40 that another process kept at 100% utilisation throughout,
so every per-kernel figure is the fastest of seven single-step submissions
(see knowledge #212 for why a busy neighbour inflates anything else), at every
conv of the network (`conv2d_native_bench`): forward 20.9 ms, input gradient
24.6 ms, weight gradient 15.2 ms per step - 60.8 ms for 124 GFLOP of conv
arithmetic, 2.0 TFLOP/s. The same step's WGSL conv kernels took 7.5 s of
device time under the same contention (5.7 s on a quiet card). First-step loss
is unchanged at the printed precision (50.8109) and 40 steps converge
(50.81 -> 6.56).

## Not delivered - what is still missing

**One model's forward, one tuned kernel. Backward, breadth and every other
tuned kernel are deferred.**

- **The tuned tier is a short list.** Plain f32 `Op::MatMul`, the int8
  group-scaled GEMM, the fp16 flash-prefill and the Gated DeltaNet chunk
  recurrence (see "Prefill on tensor cores" above), all forward. The other
  quantized weight tiers (`Q4`/K-quant), both backward GEMMs and every other
  operator are still answered by the generated tier - correctly, and visibly so
  in the dispatch record.
- **No DP4A int8 kernel** (superseded for cc >= 8.0 by the int8-MMA one above;
  below that capability the portable DP4A WGSL kernel is still what runs). The
  original milestone was not written, and the reason is worth stating rather than leaving as an omission:
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
- **How many of the registry's kernels GENERATE has not been re-measured.**
  The registry is now 533 kernels (an older count of 474, with 448 generating,
  predates it), and the numbers have not been compared against the reference
  except for the six of M2's golden gate plus what one model's forward touches.
  Breadth from here is a validation problem, not an emitter one; the plan to
  enumerate and check every kernel is phase 3 of `gh200.md`.
- **No allocator.** Every buffer is its own `cuMemAlloc`. The per-step uniform
  copy is no longer one: unbatched dispatches stage their uniform words in a
  pool of pinned blocks, and a captured graph shares a keyed uniform allocation.
  What is still missing is a device allocator in the shape a decode loop wants
  (and any device free discards every captured graph).
- **One hand-written kernel.** `crates/kernels-cuda` registers
  `matmul_f32_tiled` (FP32 forward GEMM, preserving the reference reduction
  order); every other dispatch is `Generated`, and `POLICY` is empty. A native
  tensor-core kernel does not exist. `make cuda-table/check` is a structure
  gate only - compiling each declared kernel under NVRTC for its own floor
  needs a toolkit, so it would be an addition, not a replacement.
- **`CudaProvider` produces CUDA `ImplChoice`s only for plain F32 forward
  matmul.** Quantized, half-precision, backward, embed and MoE shapes decline
  to the reference provider, and attention, softmax, convolution and the
  model-local GEMM selection never reach the provider seam at all.
- **Tier coverage is not surfaced anywhere.** `brain devices` shows no per-op
  tier coverage, there is no `--trace-impl` flag, and nothing carries the
  choice into `braintop` (which would go through a `DeviceBudget`-side field:
  the accelerator rows are built from budgets, not from a snapshot map). The
  record exists; nothing reads it back out yet.
- **The provider seam covers five `Ops` methods** (`matmul`, `embed`,
  `moe_linear`, `matmul_dx`, `matmul_dw`) and `ProviderRegistry::for_gpu`
  prepends `CudaProvider` in production whenever a compute capability is
  reported. Most model crates choose kernels without it.
- **The generator refuses a set of kernels** (26 of 474 when last counted, not
  re-measured against 533): 25 for a barrier inside a loop
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
- **`brain devices` does not show CUDA visibility per card.** The backends
  column still reports only `vulkan`/`wgpu`. The data is available
  (`devices::cuda_ordinal`); the column is not wired.
- **`BRAIN_BACKEND` is the environment twin of `--backend`.** It is read in the
  one file that reads `BRAIN_DEVICE` (`gpu-core`'s `devices.rs`), applied to the
  same `ComputeSet`, and the CLI flag wins over it; `check-device-env-single-source.sh`
  holds both variables to a single reader. A test binary or an embedding
  application can therefore be pointed at CUDA without a CLI.
- **Eight captured graphs at most, per handle.** Instantiation costs
  milliseconds, so it only pays amortised over many replays; a caller
  alternating between more than eight shapes evicts and re-instantiates, and
  would be slower than launching each dispatch. The number is a judgement,
  not a measurement of where the knee is: a decode token needs five or six
  (its layer-stack chunks plus the head).
- **Graph capture is switched off with `BRAIN_CUDA_GRAPHS=0`** (or
  `CudaBackend::with_graph_capture(false)` from Rust). There is still no CLI flag
  for it.
- **Nothing reports capture state to a human.** `brain devices` does not say
  whether a device is batching submissions, `braintop` shows no
  `graph_replays`, and `launch_stats` is reachable only from Rust. The
  `staging_waits` counter in particular is a field diagnostic that currently
  nothing in the field can read.
- **Two same-shaped steps with different parameters share a uniform when
  issued one at a time.** On the unbatched path the second's upload must wait
  for the first's kernel. A captured graph gives every step its own parameter
  slice, so replay has no such serialisation (asserted by
  `cuda_pass.rs::two_steps_of_one_shape_with_different_parameters_keep_their_own`).
- **Graph capture is validated on one model's forward and one model's decode.**
  `gpt2`'s tiny cross-backend parity runs through it and still agrees to
  `8.940697e-8`, `cuda_graphs.rs`/`cuda_pass.rs` cover the mechanism directly,
  and Qwen3.8-27B's decode runs through it (`decode_steady_state.rs`,
  `gguf_resident_real.rs`). No backward pass has been driven through it.
- **`scripts/gates/parity-gate.sh` runs one CUDA step**
  (`cuda_provider_matmul`, which skips without a device); the wider catalogue
  and the backward kernels are not in it.
- **No cost formulas** for CUDA steps in `gpu-core/src/cost.rs` (a coverage
  ratchet), and `step_native` carries no `StepMeta`, so profiling would show
  `<no-meta>`.
- **memauth**: a CUDA and a Vulkan handle on one physical card both charge
  `Device::Gpu(i)`, and under `--limit-vram-total` every GPU shares one pool, so
  they do not double-count. What is still wrong: `Gpu::try_new_cuda` opens CUDA
  ordinal 0 but charges the ambient selection's index, so on a multi-GPU box
  the two can disagree; and the ceiling is a flag, not a per-card capacity.

## Roadmap to full, extremely fast coverage (M6+)

M0-M5 proved the mechanism end to end (identity, tier reporting, a generated
tier, a `Backend`, one tuned kernel, graph replay) on one model's forward.
What is missing is BREADTH (every op, forward and backward, every dtype) and
DEPTH (the fast kernel families the generator refuses, and whatever a queried
card's generation can do that a portable kernel cannot). Both must ship
without weakening the rule already built and gated: an op this backend cannot
answer at the tier its policy demands is a named, recorded gap, never a
quiet drop to a slower tier or another device.

### The resolution principle that keeps duplication down while chasing peak speed

Three ways to answer one `Op`, ranked by preferring whichever is both
correct and fastest for the capability actually queried, cheapest to add
last:

1. **Library** - a vendor's own tuned kernel (cuBLASLt, cuDNN, NCCL) for
   whatever generation the driver reports, reached by `dlopen` exactly like
   NVRTC - optional, declined cleanly where the shared object is absent, and
   never linked at build time. This is the mechanism that gets tensor-core
   and per-generation tuning "for free" - the vendor has already written and
   tuned the kernel for every architecture brain would otherwise have to
   hand-port one at a time.
2. **Tuned (native)** - brain's own hand-written `.cu`, for whatever a
   library does not cover well: brain's own `PagedAttention` KV layout,
   decode-shaped GEMV the general GEMM libraries do not specialise for, or
   any op a library simply has no kernel for.
3. **Generated** - the existing `wgsl-cuda` T0 path, the correctness floor
   every op already has today.

This is a **provider chain ordering**, not a change to how a chain resolves -
`ProviderRegistry` already tries providers in order and records a `Decline`
per skip (`resolve_choice`, delivered in M1). A library provider is simply
placed before the native `CudaProvider`; on any box without the library it
declines every op it would have claimed, in the record, and `CudaProvider`
or the generated tier answers instead - never a silent gap. `ImplSource`
gains a fourth value, **`Library`**, ranked above `Tuned` (`Reference <
Generated < Tuned < Library`): tier reporting must be able to say "a vendor
kernel answered this," not fold it into the same bucket as a hand-written
one, because the two have different maintenance and portability properties.

Net effect on duplication: brain hand-writes a kernel only for what a queried
card's library does not already do well - not one kernel per architecture
generation, and not a WGSL translation for anything performance-critical.

### Milestones

| M | Deliverable | Genuine red test |
|---|---|---|
| **6** | Real allocator: a suballocating arena per device, replacing one `cuMemAlloc` per step/buffer; `cuModuleGetFunction` cached at compile time, not per launch | an allocation-churn test asserting no repeated `cuMemAlloc` for a repeated size class across N churn cycles, plus a graph-replay run across a churn cycle that stays within one `alloc_epoch` |
| **7** | Backward pass: `brain-gradcheck`'s finite-difference suite runs on `--backend cuda` (skip-if-absent) for every op the forward gate already covers | the suite, currently unable to select CUDA at all, turns green for the covered subset and is wired into `.agents/roadmap/cuda.md`'s ledger the same commit it turns green |
| **8** | `CudaLibraryProvider`: cuBLASLt for the GEMM family (plain, register-tiled, and quantized), `dlopen`ed and declined cleanly where absent | golden agreement vs the WGSL reference at the project's 1e-6 floor, skip-if-library-absent, plus a decline-is-recorded test with the library deliberately not loaded |
| **9** | Attention family: cuDNN's fused attention where the operand layout matches its contract; a hand-written flash-attention/paged-attention T2 kernel as the fallback that always exists, retiring every `flash_attn_*`/`paged_flash_*` refusal | decode-shaped and long-context shapes at 1e-6 vs the WGSL reference, run through whichever of the two answered, recorded distinctly in the dispatch trace |
| **10** | Quantized tiers: `PolicyEntry` grows a dtype axis; a DP4A int8 T2 kernel; Q4/K-quant register-tiled kernels mirroring their WGSL counterparts | parity fixtures at each dtype's existing WGSL tolerance (not 1e-6 - the quant tiers already accept a looser bar there), plus the now-non-empty `POLICY` ratchet exercising a real entry for the first time |
| **11** | Per-generation tensor-core packs (WMMA sm_70+, MMA sm_80+, WGMMA/TMA sm_90+) for the hottest kernels, each a `kernels-cuda` entry with its own `min_cc` floor, resolved automatically by `best_for` - no branch anywhere names an architecture | `make cuda-table/check` gains an NVRTC/nvcc compile-only check per declared floor via `-arch=compute_XX`, including floors above any card present; a numeric golden test runs wherever hardware exists to run it and the entry is marked "compiles, unverified numerically" otherwise - never claimed as verified `Tuned` parity that never ran |
| **12** | Multi-GPU: NCCL as a `dlopen`ed provider for `Collective`; `Shard.gpu_index`/`check-multi-gpu-sharding.sh` cover CUDA-identified devices; one `memauth::PoolId` shared between a CUDA and a Vulkan handle on the same physical card | a two-device collective test (skip-if-fewer-than-2) and a memauth test asserting one `PoolId` per physical card across both backend handles, so `--limit-vram-total` stops double-counting |
| **13** | Surfacing and gate graduation: `brain devices` per-op tier column and per-card CUDA visibility; `--trace-impl`; `braintop`'s `DeviceBudget`-side field; cost formulas for CUDA steps in `gpu-core/src/cost.rs`; `StepMeta` for `step_native`; CI-built AOT cubins alongside the NVRTC dev path; CUDA joins `scripts/gates/parity-gate.sh` as a real line, not a skip | `cost.rs`'s existing coverage ratchet extended to CUDA steps and made red before the formulas land; `parity-gate.sh`'s CUDA line green on a full run, gated on M7-M11 landing first |

Ordering rationale: the allocator (M6, done for transient lifetimes as an owned stream-ordered
pool behind `hold_freed_blocks`, see `mem_pool.rs`) is a prerequisite for everything after
it - graph replay, backward's extra live buffers and every library call all
need address-stable, reusable memory, not a `cuMemAlloc` per step. Backward
(M7) comes before the fast hand-written kernels so correctness gates exist
before anything gets exotic. Library providers (M8-M9) precede the
hand-written quantized/tensor-core work (M10-M11) because a vendor's kernel
is the cheapest correct answer wherever it applies, and only writing brain's
own kernel for what a library leaves uncovered keeps hand-ported surface
area to the minimum the "no silent slow fallback" rule can still make fast.
Multi-GPU (M12) is largely independent of M8-M11 but depends on the settled
device-identity and allocator work, so it is sequenced after both. Surfacing
and gate graduation (M13) is last because it reports coverage that does not
exist yet, and because promoting CUDA into `parity-gate.sh` before backward
and the fast kernel families land would either gate a partial backend or
silently skip the parts that are not ready - exactly what this whole effort
exists to make impossible.

Every milestone owes the same repo obligations M0-M5 did: SPDX +
`Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>` on every new
file, a `.agents/knowledge/` entry in the same commit as any
non-obvious finding, build only through the Makefile, zero warnings
including pre-existing ones in touched files, and one self-contained commit
per verified milestone on a linear history. No milestone may name a card,
a compute capability as a permanent ceiling, or a measured number as a
promise for different hardware - every threshold is a queried fact, checked
again at the point it is used.

## Design decisions worth not re-litigating

- **CUDA kernels will live in `crates/kernels-cuda`, not `crates/kernels/cuda/`.**
  `scripts/build/kernels-regen.sh` derives const names as UPPER_SNAKE of a
  `.wgsl` stem, so `cuda/matmul.cu` collides with the existing `MATMUL`;
  `kernels::ALL` is asserted to compile on the test device kernel-by-kernel;
  `cost.rs`'s ratchet demands a cost formula per entry; and all five WGSL
  metadata cross-checks are WGSL-text-specific with zero purchase on CUDA C++.
  A sibling crate with its own registry and its own gate is cheaper than one
  script hosting two disjoint validators.
- **The binding is not the tile budget.** Reporting the card's real VRAM as the
  one binding figure made chunked pipelines size slabs they cannot bind, and
  reporting `2 GiB - 1` starved them on a large card. Superseded by the split
  limits (see "Memory limits"): tile budgets read the working set.
- **Tier policy is a `const` + ratchet test, not a `.toml`.** A tier table is a
  status ledger; those live in `.agents/`, and a number nothing checks goes
  stale.
- **CUDA Graphs are keyed on the dispatch's structure, not its parameters.**
  Done, with one deliberate departure from the plan's wording: the uniform
  key excludes `threads` as well as the parameter values. Including it would
  give every advancing sequence position a fresh uniform address and so
  invalidate the graph at precisely the boundary
  `cuGraphExecKernelNodeSetParams` exists to survive - the grid is a property
  of the launch, not of the storage. Everything else stands: pinned staging
  and a `cuMemcpyHtoDAsync` as each step's first node, exec-update rather than
  one instantiated graph per position, `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`,
  and `read`/`poll_wait` kept out of the captured region by confining capture
  to a single `submit`.
- **Capture is triggered by a repeated submission SHAPE, never by a step-cache
  hit streak.** The rejected trigger could not fire: `StepCache::Key` includes
  the parameters, and the position-carrying steps can never repeat. The shape
  of a submission - its kernels, its buffers and its parameter block sizes -
  repeats every token by construction, and it is also the exact thing a graph
  records, so it is both a trigger that fires and one that means something.

## Notes for whoever picks this up

- Build only through the Makefile. A targeted run is
  `make test CARGO_TEST="cargo test --release --offline -p brain-gpu-core" TEST_THREADS="1 cuda"`.
- Keep GPU tests at the existing tiny-shape gradcheck/parity scale. This is a
  shared box; a sustained benchmark contends with whatever else is resident.
- **`cuda_provider_matmul`'s speedup floor is sensitive to the FIRST run of a
  freshly built binary.** Measured, not suspected: back-to-back alternating
  runs with graph capture forced on and forced off gave 5.57 / 5.13 on the
  first pair and then 34.80 / 36.21 and 34.87 / 37.11 - the two settings agree
  with each other every time, and the first invocation after a rebuild is slow
  whichever one is in force. Five trials and a median are not enough to
  absorb it, because the whole test is about a second long and all five trials
  land inside the slow window. If that test fails at around 5x, re-run it
  before concluding anything; the floor has NOT been lowered to accommodate
  this, because a floor moved to fit a flake stops being a floor.
- Captured graphs are keyed on `Arc` identity, so a test that wants to force a
  re-capture must actually free a buffer (`drop`), not merely build a
  same-shaped submission.
- The generated tier answers everything EXCEPT plain f32 `Op::MatMul`, which
  the tuned kernel now takes. Anything else that looks slow on this backend is
  expected to be: see the refused-kernel list above - every fast GEMM and
  every flash-attention kernel is among them.
- `.agents/knowledge/` #107 (the `_v2` symbol-name trap in any `dlopen`ed
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
  kernel can be much faster without reassociating anything) #120
  (`threads` means invocations to a catalogue kernel and blocks to a native
  one), #121 (two providers agreeing on a tail neither of them wrote is not
  parity), #122 (moving a call site onto a dispatch seam must not narrow
  its bindings), #123 (a shared allocation makes a record-time upload a
  write-after-write race), #124 (`_v2` in a `dlopen`ed C API cuts both ways),
  #125 (Rust drops struct fields in declaration order, and a device context
  is a resource's parent) and #126 (a batching win has to be measured in the
  pattern the caller actually uses) came out of this work and are the things most likely to be
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

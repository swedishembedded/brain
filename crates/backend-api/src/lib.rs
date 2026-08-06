// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Backend abstraction API — the seam every brain compute backend plugs into.
//!
//! Two contracts live here:
//!
//! * [`Backend`] — the *eager, per-step* compute device (wgpu, native CPU JIT,
//!   native Vulkan). It mirrors the historical `gpu_core::Gpu` method surface:
//!   allocate buffers, record dispatches (`step*`), and run them (`submit`) with
//!   blocking read-back. `brain-gpu-core` is a thin facade over a
//!   `Box<dyn Backend>`.
//! * [`GraphBackend`] — the *whole-graph compile→run* contract (the OpenVINO NPU
//!   path). A serialized graph (ONNX bytes) is compiled once for a target device
//!   and then run; nothing here is per-step.
//!
//! The neutral handle types [`DeviceBuffer`] and [`Step`] are opaque: each
//! backend stores its native buffer / dispatch record inside and downcasts on
//! use, so the trait methods are object-safe (`dyn Backend`) and adding a backend
//! never touches the dispatch core.
//!
//! A backend is registered by name (see [`register_backend`]); the facade
//! constructs one via [`create_backend`]. That is what makes "add a backend" a
//! new crate that depends only on this one — no edits to `brain-gpu-core`.

use std::any::Any;
use std::sync::Arc;

/// Kernel selection: which implementation of an op runs, given shape + device.
pub mod select;

// The neutral handles and the `Backend` trait are `Send + Sync` on native (the
// CPU backend hands disjoint buffer sub-ranges to rayon workers, and models cross
// threads), but NOT on wasm: WebGPU's `wgpu::Buffer`/`Device` are `Rc`-based and
// single-threaded, so requiring `Send + Sync` there would be unsatisfiable. The
// cfg alias below carries that difference in one place.

/// The erased-handle inner type: `Send + Sync` on native, bare on wasm.
#[cfg(not(target_arch = "wasm32"))]
type Erased = dyn Any + Send + Sync;
#[cfg(target_arch = "wasm32")]
type Erased = dyn Any;

/// `Send + Sync` on native, empty on wasm — the bound a value must meet to be
/// wrapped in a neutral handle. (A blanket impl makes it automatic.)
#[cfg(not(target_arch = "wasm32"))]
pub trait ThreadSafe: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> ThreadSafe for T {}
#[cfg(target_arch = "wasm32")]
pub trait ThreadSafe {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> ThreadSafe for T {}

/// Max workgroups per grid dimension (downlevel/Vulkan guarantee). Every backend
/// reproduces the same tiling so the kernels' index math is identical.
pub const MAX_GROUPS_PER_DIM: u32 = 65535;

/// The workgroup size a kernel gets when its source declares none — and the
/// size all but the register-tiled GEMMs use.
pub const DEFAULT_WORKGROUP_SIZE: u32 = 64;

/// Workgroup grid for `threads` invocations at `@workgroup_size(64)`: 1D until the
/// count exceeds the per-dimension limit, then tiled into Y. Shared by every
/// backend so the kernels' `gid.y*(nwg.x*64)+gid.x` reconstruction is identical.
pub fn grid(threads: u32) -> (u32, u32) {
    grid_ws(threads, DEFAULT_WORKGROUP_SIZE)
}

/// [`grid`] for a kernel whose `@workgroup_size` is `wg` rather than 64.
///
/// A kernel is free to declare a different workgroup size — a register-tiled
/// GEMM wants 256 invocations per tile so it can hold a 128×128 output block —
/// but then *every* backend must lay out the grid with that same `wg`, and the
/// kernel must reconstruct its flat id as `gid.y*(nwg.x*WG)+gid.x` using its own
/// `WG`. [`workgroup_size_of`] is how the backends learn it, so the WGSL source
/// stays the single place the number is written down.
pub fn grid_ws(threads: u32, wg: u32) -> (u32, u32) {
    let wg = wg.max(1);
    let groups = threads.div_ceil(wg).max(1);
    if groups <= MAX_GROUPS_PER_DIM {
        (groups, 1)
    } else {
        (MAX_GROUPS_PER_DIM, groups.div_ceil(MAX_GROUPS_PER_DIM))
    }
}

/// The `x` extent of a WGSL kernel's `@workgroup_size(...)` attribute, or
/// [`DEFAULT_WORKGROUP_SIZE`] if the source has none.
///
/// A deliberately dumb scan rather than a naga parse: every backend needs this
/// number at registration time (the CPU backend has no naga module at that
/// point, and the wgpu backend would have to re-parse), the attribute is a
/// literal in every in-repo kernel, and a wrong answer here would show up
/// immediately as a wrong dispatch size in the cross-backend parity tests.
/// Only decimal literals are recognised — `@workgroup_size(WG)` with a `const`
/// would silently fall back to 64, so kernels spell the number out.
pub fn workgroup_size_of(src: &str) -> u32 {
    let Some(at) = src.find("@workgroup_size") else { return DEFAULT_WORKGROUP_SIZE };
    let rest = &src[at + "@workgroup_size".len()..];
    let Some(open) = rest.find('(') else { return DEFAULT_WORKGROUP_SIZE };
    let digits: String = rest[open + 1..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().unwrap_or(DEFAULT_WORKGROUP_SIZE)
}

/// Per-kernel workgroup sizes for a `(name, wgsl)` registration list, in the
/// same order — what a backend stores alongside its compiled pipelines.
pub fn workgroup_sizes(kernels: &[(&str, &str)]) -> Vec<u32> {
    kernels.iter().map(|(_, src)| workgroup_size_of(src)).collect()
}

/// Pack an f32 into the u32 uniform stream (kernels read it back with bitcast).
pub fn f(x: f32) -> u32 {
    x.to_bits()
}

// ---- device capability model ------------------------------------------------
//
// What the device can actually do — the inputs a kernel selector needs to pick
// a variant, and what `brain perf` records so a result is machine-comparable.
// One struct, plain data, filled by each backend at construction and cached;
// where a value is unknowable it is `None`, and a consumer must cope — an
// unknown capability is never assumed present.

/// Broad device class — the coarsest input a kernel selector keys on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeviceClass {
    /// Host CPU execution — the Cranelift JIT backend, or a software
    /// rasteriser behind a GPU API (llvmpipe/lavapipe): the work runs on cores
    /// either way, and tiles sized for thousands of GPU lanes thrash both.
    Cpu,
    IntegratedGpu,
    DiscreteGpu,
    /// Whole-graph compiled accelerator (OpenVINO NPU). Not an eager
    /// [`Backend`]; the class exists so caps recorded from that path share the
    /// same vocabulary.
    Npu,
    /// WebGPU in a browser — the portability floor: no subgroups, no f16,
    /// no native queries beyond the WebGPU limits.
    Browser,
}

/// Numeric paths a device supports beyond the always-present fp32 baseline.
///
/// Semantics are "brain's kernels for this tier execute on this device", as
/// established by the backend at construction — never assumed from the device's
/// marketing. Speed *within* a supported tier is the autotuner's question, not
/// this struct's: a device may expose f16 at 1/64 rate (Pascal), which is why
/// `f16` here means *fast* f16 and stays `false` until measured.
///
/// `f16`/`bf16` mean *fast compute*; `f16_storage`/`bf16_storage` mean only
/// "can hold bytes of this width, dequantizing to fp32 to compute" -- the
/// distinction [`DType::promote`] needs. A card can store bf16 without a fast
/// bf16 compute path (e.g. a Pascal card widens on load and computes fp32);
/// storage-only support is never sufficient to skip that promotion.
#[derive(Clone, Copy, Debug)]
pub struct NumericSupport {
    /// Always true — the portable baseline and the numerical reference every
    /// other tier is parity-gated against.
    pub f32: bool,
    /// The packed-int8 dot kernels (`matmul_i8*`, WGSL `dot4I8Packed`)
    /// execute. Hardware DP4A where the driver has it; the 4× weight-byte
    /// saving holds regardless.
    pub int8_dot: bool,
    /// *Fast* f16 arithmetic. Deliberately not "f16 is exposed": Pascal
    /// exposes f16 at 1/64 rate, so availability without a measured rate is
    /// exactly the trap. Stays false until the autotuner (S5) measures it.
    pub f16: bool,
    /// *Fast* bf16 arithmetic, same "measured, not marketed" rule as `f16`.
    pub bf16: bool,
    /// The device can hold f16 bytes at all (even without a fast compute
    /// path for them) -- "exists" as opposed to `f16`'s "fast".
    pub f16_storage: bool,
    /// The device can hold bf16 bytes at all -- "exists" as opposed to
    /// `bf16`'s "fast".
    pub bf16_storage: bool,
    /// Cooperative-matrix / tensor-core matmul (the optional
    /// `VK_KHR_cooperative_matrix` path).
    pub coop_matrix: bool,
}

impl NumericSupport {
    /// The portable floor: fp32 only.
    pub const BASELINE: NumericSupport = NumericSupport {
        f32: true,
        int8_dot: false,
        f16: false,
        bf16: false,
        f16_storage: false,
        bf16_storage: false,
        coop_matrix: false,
    };
}

/// A checkpoint's on-disk/in-memory element width -- the input to
/// [`DType::promote`]'s placement-budgeting decision. Deliberately separate
/// from any per-backend storage type; this is what a *checkpoint* declares,
/// not what a kernel computes in. No `Ord`: F16 and BF16 are the same width
/// with no natural ordering between them, so "widens" is checked via
/// [`DType::bytes`], not variant order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    /// Bytes per element on disk/in host memory for this dtype.
    pub const fn bytes(self) -> u64 {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    /// The dtype a checkpoint declared as `self` actually executes as on a
    /// device with `numeric` support -- **only ever widens**, never
    /// narrows: there is deliberately no `demote`, so "never auto-quantize
    /// down" is structural rather than a policy a caller could get wrong.
    /// fp32 is the guaranteed ceiling (every device supports it), so this
    /// always returns *some* dtype rather than an `Option`.
    ///
    /// Today every backend dequantizes to fp32 on load (see `crate::gguf`'s
    /// `deq_*` family and every model crate's `from_reader`), so this
    /// currently returns `F32` for every input regardless of `numeric` --
    /// honestly reflecting that no execution path can compute f16/bf16 yet.
    /// The value here is the *placement budgeting* this makes correct now
    /// (a bf16 checkpoint promoted to fp32 costs 2x, not 1x); the value
    /// later is that a real f16/bf16 compute path landing cannot silently
    /// mis-budget or mis-execute on a card that lacks it.
    pub fn promote(self, numeric: &NumericSupport) -> DType {
        let _ = numeric;
        DType::F32
    }
}

#[cfg(test)]
mod dtype_tests {
    use super::*;

    #[test]
    fn promote_never_narrows_and_always_lands_at_or_above_the_input() {
        let none = NumericSupport::BASELINE;
        let full = NumericSupport { f16: true, bf16: true, f16_storage: true, bf16_storage: true, ..NumericSupport::BASELINE };
        for numeric in [none, full] {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let promoted = dtype.promote(&numeric);
                assert!(
                    promoted.bytes() >= dtype.bytes(),
                    "{dtype:?} ({} bytes) promoted to {promoted:?} ({} bytes) under {numeric:?} -- narrowed",
                    dtype.bytes(),
                    promoted.bytes()
                );
            }
        }
    }

    #[test]
    fn fp32_is_the_guaranteed_ceiling() {
        // Every promotion lands at F32 today (see promote's doc comment for
        // why); this pins that honestly rather than asserting an aspiration.
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            assert_eq!(dtype.promote(&NumericSupport::BASELINE), DType::F32);
        }
    }

    #[test]
    fn bytes_matches_element_width() {
        assert_eq!(DType::F32.bytes(), 4);
        assert_eq!(DType::F16.bytes(), 2);
        assert_eq!(DType::BF16.bytes(), 2);
    }
}

/// What the device can actually do. Filled once at backend construction,
/// cached on the backend, exposed via [`Backend::caps`].
#[derive(Clone, Debug)]
pub struct DeviceCaps {
    pub class: DeviceClass,
    /// SMs / CUs / cores. `None` where the API does not expose it (wgpu).
    pub compute_units: Option<u32>,
    /// Largest `@workgroup_size` a kernel may declare on this device.
    pub max_workgroup_size: u32,
    /// Workgroup (shared) memory per workgroup, bytes.
    pub workgroup_mem_bytes: u32,
    /// SIMD/subgroup width. `None` = not exposed (the WebGPU baseline).
    pub subgroup_size: Option<u32>,
    /// No host<->device copy cost (integrated GPUs, CPU).
    pub unified_memory: bool,
    /// Kernels that stage partial sums in workgroup memory behind a
    /// `workgroupBarrier()` execute *correctly* on this device. True on every
    /// real GPU path; false on the CPU JIT, whose split-at-barrier execution
    /// model mis-executes the decode-regime reduction kernels (its native
    /// fast paths own that regime instead). A selector must not choose a
    /// cooperative variant where this is false — that is a correctness gate,
    /// not a tuning preference.
    pub workgroup_reductions: bool,
    /// Peak memory bandwidth, GB/s. `None` = unknown.
    ///
    /// No graphics/compute API reports this, so it is filled by *measurement*
    /// (`gpu_core::roof`), not by a query — and it stays `None` until something
    /// measures it. It is the denominator every memory-bound kernel is judged
    /// against; reporting a memory-bound kernel as a FLOP rate is meaningless.
    pub peak_bandwidth_gbs: Option<f32>,
    /// Peak fp32 arithmetic rate, GFLOP/s. `None` = unknown.
    ///
    /// Same rule as [`Self::peak_bandwidth_gbs`]: measured, never queried and
    /// never derived from a marketing figure. Together the two are the roofline
    /// this engine's kernels are graded on, and having them here — rather than
    /// as a `PEAK_TFLOPS` literal in each bench — is what makes a "% of peak"
    /// claim a statement about *the device that ran*, on any hardware.
    pub peak_gflops: Option<f32>,
    pub numeric: NumericSupport,
}

impl DeviceCaps {
    /// Where a kernel sits relative to the ridge point, given the work it did.
    ///
    /// The ridge is `peak_gflops / peak_bandwidth_gbs` FLOP/byte: below it a
    /// kernel cannot be compute-bound however it is tiled, above it bandwidth
    /// cannot be the limit. `None` when either roof is unmeasured — an
    /// unknown capability is never assumed present.
    pub fn ridge_flops_per_byte(&self) -> Option<f32> {
        match (self.peak_gflops, self.peak_bandwidth_gbs) {
            (Some(f), Some(b)) if b > 0.0 => Some(f / b),
            _ => None,
        }
    }
}

impl DeviceCaps {
    /// The WebGPU-guaranteed floor for `class`: conservative limits, fp32
    /// only, nothing assumed. What a consumer gets when a backend knows
    /// nothing better.
    pub fn portable_baseline(class: DeviceClass) -> DeviceCaps {
        DeviceCaps {
            class,
            compute_units: None,
            // The engine's own invariants: kernels declare at most
            // @workgroup_size(256) and stage at most 16 KiB (the downlevel
            // default) unless the adapter reports more.
            max_workgroup_size: 256,
            workgroup_mem_bytes: 16 * 1024,
            subgroup_size: None,
            unified_memory: false,
            // WebGPU-conformant devices execute workgroup barriers correctly —
            // this is part of the floor, not an extension.
            workgroup_reductions: true,
            peak_bandwidth_gbs: None,
            peak_gflops: None,
            numeric: NumericSupport::BASELINE,
        }
    }
}

/// Stable identity of one physical GPU, established by the canonical device
/// registry (`gpu_core::devices`) and consumed by backends to select the same
/// card by *identity*, never by enumeration position or a process-global env
/// var. Identity key priority: PCI bus id (stable across boots and shared with
/// NVML/nvidia-smi) → Vulkan `deviceUUID` (== the NVML GPU UUID on NVIDIA) →
/// `(vendor:device, ordinal)` where `ordinal` counts devices with the same
/// vendor:device pair in `vkEnumeratePhysicalDevices` order (the tiebreaker for
/// identical twins when neither PCI nor UUID is available).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuIdentity {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    /// `VkPhysicalDeviceIDProperties::deviceUUID` (Vulkan 1.1 core).
    pub uuid: Option<[u8; 16]>,
    /// `"domain:bus:device.function"` lowercase hex (`VK_EXT_pci_bus_info`).
    pub pci_bus: Option<String>,
    /// Ordinal within this identity's `(vendor_id, device_id)` pair, in the
    /// source enumeration order.
    pub ordinal: usize,
    /// Largest DEVICE_LOCAL heap, bytes (0 = unknown).
    pub vram_bytes: u64,
    pub class: DeviceClass,
}

impl GpuIdentity {
    /// Whether `other` names the same physical card, using the strongest key
    /// both sides carry: UUID, else PCI bus, else (vendor, device, ordinal).
    pub fn same_device(&self, other: &GpuIdentity) -> bool {
        if let (Some(a), Some(b)) = (self.uuid, other.uuid) {
            return a == b;
        }
        if let (Some(a), Some(b)) = (&self.pci_bus, &other.pci_bus) {
            return a == b;
        }
        self.vendor_id == other.vendor_id
            && self.device_id == other.device_id
            && self.ordinal == other.ordinal
    }
}

/// Backend-neutral buffer usage flags. Mirrors the subset of `wgpu::BufferUsages`
/// the kernels need; the wgpu backend maps it back to `wgpu::BufferUsages`, the
/// CPU backend ignores it (all allocations are plain host memory).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BufUsage(pub u32);

impl BufUsage {
    pub const STORAGE: BufUsage = BufUsage(1);
    pub const COPY_DST: BufUsage = BufUsage(2);
    pub const COPY_SRC: BufUsage = BufUsage(4);
    pub const UNIFORM: BufUsage = BufUsage(8);
    pub fn contains(self, other: BufUsage) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for BufUsage {
    type Output = BufUsage;
    fn bitor(self, rhs: BufUsage) -> BufUsage {
        BufUsage(self.0 | rhs.0)
    }
}

/// An opaque device buffer on whichever backend created it. Model code holds
/// these and passes `&DeviceBuffer` to the dispatch methods without knowing the
/// backend; the backend downcasts back to its native buffer type. Cloning is
/// cheap (an `Arc` bump) and aliases the same underlying buffer — every backend's
/// native buffer is already reference-counted or a no-op-drop handle.
#[derive(Clone)]
pub struct DeviceBuffer(Arc<Erased>);

impl DeviceBuffer {
    /// Wrap a backend-native buffer.
    pub fn new<T: Any + ThreadSafe>(inner: T) -> DeviceBuffer {
        DeviceBuffer(Arc::new(inner))
    }
    /// Identity of the underlying allocation. Two `DeviceBuffer`s alias the
    /// same memory iff their ids are equal (clones share one `Arc`).
    pub fn alloc_id(&self) -> *const () {
        Arc::as_ptr(&self.0) as *const ()
    }
    /// Recover the native buffer. Panics on a backend mismatch (a buffer from one
    /// backend handed to another) — the same fail-fast the enum dispatch had.
    pub fn downcast_ref<T: Any>(&self) -> &T {
        self.0
            .downcast_ref::<T>()
            .expect("DeviceBuffer/backend mismatch")
    }
}

/// Backend-neutral record of WHAT a dispatch computes: the kernel slot it runs
/// and the uniform params it was recorded with. Attached by the `gpu_core`
/// facade at `step*` time, so cost accounting (offline `cost_of`, online
/// per-submit counters) never reaches into a backend's native dispatch record.
/// `params` is `None` for `step_buf` dispatches — their uniform lives in a
/// caller-owned buffer whose contents the facade cannot see.
#[derive(Clone, Debug)]
pub struct StepMeta {
    /// Index into the `(name, wgsl)` kernel set the device was built with.
    pub kernel: usize,
    /// The `params` slice as recorded, or `None` (`step_buf`).
    pub params: Option<Vec<u32>>,
    /// Dispatch thread count.
    pub threads: u32,
}

/// An opaque recorded dispatch, tagged by the backend that built it. Held by
/// callers between `step*` and `submit`; `submit` downcasts it back.
#[derive(Clone)]
pub struct Step {
    inner: Arc<Erased>,
    meta: Option<Arc<StepMeta>>,
}

impl Step {
    /// Wrap a backend-native dispatch record.
    pub fn new<T: Any + ThreadSafe>(inner: T) -> Step {
        Step { inner: Arc::new(inner), meta: None }
    }
    /// Attach the neutral dispatch record (facade `step*` does this; backends
    /// never need to).
    pub fn with_meta(mut self, meta: StepMeta) -> Step {
        self.meta = Some(Arc::new(meta));
        self
    }
    /// The neutral dispatch record, if the step came through the facade.
    pub fn meta(&self) -> Option<&StepMeta> {
        self.meta.as_deref()
    }
    /// Recover the native dispatch record. Panics on a backend mismatch.
    pub fn downcast_ref<T: Any>(&self) -> &T {
        self.inner.downcast_ref::<T>().expect("Step/backend mismatch")
    }
}

/// The eager, per-step compute device. Implemented by wgpu, the native CPU JIT,
/// and native Vulkan. All methods take `&self` and use only neutral handle types,
/// so `dyn Backend` is object-safe and the facade is a trivial forwarder.
///
/// `read`/`poll_wait` are native-only: on wasm the browser drives the device via
/// its event loop (no blocking poll) and read-back is async, handled by the
/// wgpu backend's inherent `read_async` rather than a trait method. The trait is
/// `Send + Sync` on native (backends run across threads); on wasm it is not,
/// because WebGPU handles are single-threaded (see the `Erased`/`ThreadSafe`
/// note above).
#[cfg(not(target_arch = "wasm32"))]
pub trait Backend: Send + Sync {
    /// Allocate `n` f32 words of zeroed device storage.
    fn storage(&self, n: u64) -> DeviceBuffer;
    /// Allocate device storage initialised from host `data`.
    fn storage_init(&self, name: &str, data: &[f32]) -> DeviceBuffer;
    /// Allocate a buffer of `size` bytes with the given usage.
    fn buffer(&self, label: &str, size: u64, usage: BufUsage) -> DeviceBuffer;
    /// A writable uniform buffer sized for `len` u32s, updated later via `write`.
    fn uniform_dynamic(&self, len: usize) -> DeviceBuffer;
    /// Overwrite `buf`'s contents with host `data` (after prior compute completes).
    fn write(&self, buf: &DeviceBuffer, data: &[u32]);
    /// Record a dispatch with a fresh single-use uniform buffer.
    fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step;
    /// Record a dispatch where each buffer binds the sub-range `offsets[i]`.
    fn step_sliced(
        &self,
        kind: usize,
        bufs: &[&DeviceBuffer],
        offsets: &[(u64, u64)],
        params: &[u32],
        threads: u32,
    ) -> Step;
    /// Record a dispatch around an already-allocated uniform buffer.
    fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step;
    /// Clear the given buffers, then run all recorded steps.
    fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]);
    /// Block, copy `buf` back to the host, and return it as f32.
    fn read(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32>;
    /// Block until all submitted device work has completed.
    fn poll_wait(&self);
    /// Send recorded-but-unsubmitted work to the device WITHOUT waiting for
    /// completion — the frame-pipelining hook: start the device on frame n,
    /// overlap the host's preprocessing of frame n+1, synchronise at the next
    /// `read`. Backends that execute eagerly at `submit` (CPU) have nothing
    /// pending, so the default no-op is correct for them.
    fn flush(&self) {}
    /// The largest a single storage-buffer binding may be on this device, in
    /// bytes — the hardware limit a kernel's biggest buffer must fit under.
    /// Card-dependent (wgpu reports the adapter's value); the default is the
    /// common ~2 GiB so callers that never query a real device stay conservative.
    /// A second handle onto the **same** device: same queue and compiled
    /// pipelines, its own command stream. `None` when the backend cannot share
    /// (the caller then builds a fresh device, which is correct, just not free).
    ///
    /// Exists because building a device is expensive and several concurrent
    /// devices on one physical card are hostile to the driver, so a process
    /// running many models wants one device and many handles.
    /// Which execution backend this is: `"wgpu" | "cpu" | "vulkan"`. Kernel
    /// selection needs it — e.g. the decode-regime workgroup kernels help every
    /// GPU but are mis-executed by the CPU JIT's barrier-split model (and the
    /// CPU backend has its own native fast paths), so they are gated off there.
    fn kind(&self) -> &'static str;

    fn share(&self) -> Option<Box<dyn Backend>> {
        None
    }

    /// A weak handle onto this backend's shared device state, for pools and
    /// fixtures that must NOT keep the device alive: a device should die with
    /// its last real handle (an orderly, in-process `vkDestroyDevice`), never
    /// at process exit — a device torn down during exit crashes the NVIDIA
    /// driver's worker threads. `None` when the backend has no shared state.
    fn downgrade(&self) -> Option<Box<dyn WeakBackend>> {
        None
    }

    /// A backend for a **different kernel set** on the **same device** — the
    /// multi-model form of [`Backend::share`]. One process serving N models
    /// wants one device carrying N pipeline sets, not N devices: many
    /// concurrent devices on one card is both slow (a full device init each)
    /// and hazardous (it deadlocked the test suite). `None` when the backend
    /// has no shareable device; the caller then builds a fresh backend.
    fn new_like(&self, _kernels: &[(&str, &str)]) -> Option<Box<dyn Backend>> {
        None
    }

    fn max_storage_binding_bytes(&self) -> u64 {
        2 * 1024 * 1024 * 1024 - 1
    }

    /// What this device can actually do — see [`DeviceCaps`]. Filled at
    /// construction; querying is a cached read, never a device round-trip.
    fn caps(&self) -> DeviceCaps;

    /// Device-op accounting for THIS handle since its creation, if the backend
    /// counts (relaxed atomics, negligible next to a dispatch). `None` = not
    /// counted — a consumer must report null, never zero.
    fn stats(&self) -> Option<DeviceStats> {
        None
    }

    /// Turn per-kernel DEVICE timing on or off for this device, returning
    /// whether it is now on. `false` means the backend cannot time kernels.
    ///
    /// This exists because host wall-clock around a drained slice is not a
    /// measurement of a kernel — it measures launch + execute + fence, whose
    /// floor is roughly constant and therefore inflates small kernels in inverse
    /// proportion to their size (up to 29x measured; `docs/lessons.md` #31).
    /// A profiler that attributes time between kernels must use device time.
    fn set_kernel_timing(&self, _on: bool) -> bool {
        false
    }

    /// Per-kernel accumulated DEVICE time since the last [`Self::reset_kernel_times`],
    /// as `(kernel name, milliseconds, calls)`. `None` where the backend cannot
    /// time kernels — a consumer must then say so, never substitute host time
    /// silently.
    fn kernel_times(&self) -> Option<Vec<(String, f64, u64)>> {
        None
    }

    /// Zero the per-kernel accumulators.
    fn reset_kernel_times(&self) {}

    /// Print the per-kernel `BRAIN_PROFILE` timing table NOW (stderr). The
    /// dump otherwise fires only at drop — which a RESIDENT model held in a
    /// static never reaches, so its profile was unreadable by construction.
    /// No-op when profiling is off or the backend does not time kernels.
    fn dump_profile(&self) {}
}

/// Per-handle device-op counters — the queryable form of what
/// `BRAIN_PROFILE` used to print only to stderr. What a benchmark records so
/// "how many submits/readbacks did this run cost" is machine-readable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceStats {
    pub submits: u64,
    pub dispatches: u64,
    pub readbacks: u64,
    pub bind_groups: u64,
    pub uniform_allocs: u64,
}

/// wasm variant: no `Send + Sync` (WebGPU is single-threaded) and no blocking
/// `read`/`poll_wait` (read-back is async via the wgpu backend's `read_async`).
#[cfg(target_arch = "wasm32")]
pub trait Backend {
    /// Allocate `n` f32 words of zeroed device storage.
    fn storage(&self, n: u64) -> DeviceBuffer;
    /// Allocate device storage initialised from host `data`.
    fn storage_init(&self, name: &str, data: &[f32]) -> DeviceBuffer;
    /// Allocate a buffer of `size` bytes with the given usage.
    fn buffer(&self, label: &str, size: u64, usage: BufUsage) -> DeviceBuffer;
    /// A writable uniform buffer sized for `len` u32s, updated later via `write`.
    fn uniform_dynamic(&self, len: usize) -> DeviceBuffer;
    /// Overwrite `buf`'s contents with host `data` (after prior compute completes).
    fn write(&self, buf: &DeviceBuffer, data: &[u32]);
    /// Record a dispatch with a fresh single-use uniform buffer.
    fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step;
    /// Record a dispatch where each buffer binds the sub-range `offsets[i]`.
    fn step_sliced(
        &self,
        kind: usize,
        bufs: &[&DeviceBuffer],
        offsets: &[(u64, u64)],
        params: &[u32],
        threads: u32,
    ) -> Step;
    /// Record a dispatch around an already-allocated uniform buffer.
    fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step;
    /// Clear the given buffers, then run all recorded steps.
    fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]);
    /// What this device can actually do — see [`DeviceCaps`]. The default is
    /// the browser floor, which is exactly what wasm is.
    fn caps(&self) -> DeviceCaps {
        DeviceCaps::portable_baseline(DeviceClass::Browser)
    }
}

/// The whole-graph compile→run contract (the OpenVINO NPU path). A serialized
/// graph is compiled once for a target device, then run. Not object-safe by
/// design (associated IO types differ per backend); selected concretely.
pub trait GraphBackend: Sized {
    /// Backend-specific compile/run configuration (e.g. target device, perf hint).
    type Config;
    /// One inference's outputs (e.g. the raw model head tensors).
    type Output;
    /// Backend-specific error type.
    type Error: std::error::Error;

    /// Compile a serialized graph (ONNX bytes) for the configured device.
    fn compile(onnx: &[u8], cfg: &Self::Config) -> Result<Self, Self::Error>;
    /// Run one inference over `input` with the NCHW `shape`.
    fn run(&mut self, input: &[f32], shape: [usize; 4]) -> Result<Self::Output, Self::Error>;
    /// The device this session actually resolved to (e.g. "NPU", or a fallback).
    fn device(&self) -> &str;
}


/// A weak reference to a backend's shared device state — see [`Backend::downgrade`].
pub trait WeakBackend: ThreadSafe {
    /// A fresh strong handle, if the device is still alive.
    fn upgrade(&self) -> Option<Box<dyn Backend>>;
}

// ---- backend registry -------------------------------------------------------
//
// A backend registers a factory under a name; the facade constructs one by name.
// This is what lets a new backend be a standalone crate — implement `Backend`,
// call `register_backend`, and the dispatch core never changes. Native-only: the
// wasm build has exactly one backend (wgpu) and the facade holds it concretely.

#[cfg(not(target_arch = "wasm32"))]
mod registry {
    use super::Backend;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    /// Builds a backend for the given `(name, wgsl_source)` kernel set, or returns
    /// an error (e.g. no device present) so the caller can fall back.
    pub type Factory = fn(&[(&str, &str)]) -> Result<Box<dyn Backend>, String>;

    static REGISTRY: OnceLock<Mutex<HashMap<&'static str, Factory>>> = OnceLock::new();

    fn registry() -> &'static Mutex<HashMap<&'static str, Factory>> {
        REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Register `factory` under `name` (idempotent; last write wins). A backend
    /// crate calls this so the facade can build it by name without naming its type.
    pub fn register_backend(name: &'static str, factory: Factory) {
        registry().lock().unwrap().insert(name, factory);
    }

    /// True iff a backend is registered under `name`.
    pub fn backend_registered(name: &str) -> bool {
        registry().lock().unwrap().contains_key(name)
    }

    /// Construct the backend registered under `name`, compiling `kernels`.
    pub fn create_backend(name: &str, kernels: &[(&str, &str)]) -> Result<Box<dyn Backend>, String> {
        let factory = registry().lock().unwrap().get(name).copied();
        match factory {
            Some(factory) => factory(kernels),
            None => Err(format!("no backend registered under '{name}'")),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use registry::{backend_registered, create_backend, register_backend, Factory};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_workgroup_size_forms() {
        assert_eq!(workgroup_size_of("@compute @workgroup_size(64)\nfn main() {}"), 64);
        assert_eq!(workgroup_size_of("@workgroup_size(256)"), 256);
        assert_eq!(workgroup_size_of("@workgroup_size( 128 , 1, 1 )"), 128);
        // no attribute, or a non-literal extent => the engine default
        assert_eq!(workgroup_size_of("fn main() {}"), DEFAULT_WORKGROUP_SIZE);
        assert_eq!(workgroup_size_of("@workgroup_size(WG)"), DEFAULT_WORKGROUP_SIZE);
    }

    #[test]
    fn grid_ws_matches_grid_at_64() {
        for t in [1u32, 63, 64, 65, 4096, 4_194_240, 4_194_241] {
            assert_eq!(grid(t), grid_ws(t, 64), "threads={t}");
        }
    }

    #[test]
    fn grid_ws_covers_every_thread() {
        for wg in [64u32, 128, 256] {
            for t in [1u32, wg - 1, wg, wg + 1, 100_000] {
                let (gx, gy) = grid_ws(t, wg);
                assert!((gx as u64) * (gy as u64) * (wg as u64) >= t as u64, "wg={wg} threads={t}");
                assert!(gx <= MAX_GROUPS_PER_DIM && gy >= 1);
            }
        }
    }

    /// Past the 65535-group limit the grid tiles into Y — the case every
    /// kernel's `gid.y*(nwg.x*WG)+gid.x` reconstruction depends on.
    #[test]
    fn grid_ws_tiles_into_y_past_the_limit() {
        let (gx, gy) = grid_ws(MAX_GROUPS_PER_DIM * 256 + 256, 256);
        assert_eq!((gx, gy), (MAX_GROUPS_PER_DIM, 2));
    }
}

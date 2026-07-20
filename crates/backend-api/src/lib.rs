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

/// Workgroup grid for `threads` invocations at `@workgroup_size(64)`: 1D until the
/// count exceeds the per-dimension limit, then tiled into Y. Shared by every
/// backend so the kernels' `gid.y*(nwg.x*64)+gid.x` reconstruction is identical.
pub fn grid(threads: u32) -> (u32, u32) {
    let groups = threads.div_ceil(64).max(1);
    if groups <= MAX_GROUPS_PER_DIM {
        (groups, 1)
    } else {
        (MAX_GROUPS_PER_DIM, groups.div_ceil(MAX_GROUPS_PER_DIM))
    }
}

/// Pack an f32 into the u32 uniform stream (kernels read it back with bitcast).
pub fn f(x: f32) -> u32 {
    x.to_bits()
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

/// An opaque recorded dispatch, tagged by the backend that built it. Held by
/// callers between `step*` and `submit`; `submit` downcasts it back.
#[derive(Clone)]
pub struct Step(Arc<Erased>);

impl Step {
    /// Wrap a backend-native dispatch record.
    pub fn new<T: Any + ThreadSafe>(inner: T) -> Step {
        Step(Arc::new(inner))
    }
    /// Recover the native dispatch record. Panics on a backend mismatch.
    pub fn downcast_ref<T: Any>(&self) -> &T {
        self.0.downcast_ref::<T>().expect("Step/backend mismatch")
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

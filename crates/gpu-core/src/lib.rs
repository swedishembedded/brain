// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Compute-device facade.
//!
//! `Gpu` is a thin wrapper over a [`backend_api::Backend`] — the eager, per-step
//! compute contract. Model code holds a `Gpu` and calls `storage`/`step`/`submit`/
//! `read` without knowing which backend is behind it. The concrete backends live
//! in their own crates (`brain-backend-wgpu`, `brain-backend-cpu`,
//! `brain-backend-vulkan`), each implementing `Backend` and registering itself by
//! name; adding a backend is a new crate that depends only on `brain-backend-api`,
//! never this facade or its dispatch.
//!
//! The neutral [`DeviceBuffer`], [`Step`], and [`BufUsage`] types (re-exported
//! from `backend_api`) are what let the rest of the workspace avoid naming
//! `wgpu::*` / `ash::*` directly.
//!
//! ## Backend selection
//!
//! Native builds carry all three backends; a given `Gpu` uses exactly one, chosen
//! by [`set_default_backend`] (the CLI's `--device` flag) or `BRAIN_DEVICE`. On
//! wasm only the wgpu/WebGPU backend exists, so `Gpu` wraps it directly (no
//! `dyn`, and async device init / read-back via `new_async` / `read_async`).

pub use backend_api::{f, BufUsage, DeviceBuffer, Step};

/// `--device` parsing and resolution: which compute is *schedulable*.
#[cfg(not(target_arch = "wasm32"))]
pub mod devices;
#[cfg(not(target_arch = "wasm32"))]
pub use devices::{ComputeSet, DeviceSpec, Inventory};


/// A process-wide device fixture for **test binaries** — explicit, documented,
/// and torn down before exit.
///
/// libtest runs tests concurrently in one process, and each GPU test that
/// builds its own `Gpu` puts another live device on the card. On the NVIDIA
/// driver this box runs, that shape fails two ways, both measured:
/// many concurrent devices deadlock (~50% of runs, every thread in futex
/// wait), and a device *leaked in a static* at process exit segfaults the
/// driver's worker thread during teardown — after every test has passed.
///
/// So: one parent device per test binary, one handle per kernel set via
/// [`Gpu::new_like`], and an `atexit` hook that drops everything in an orderly
/// fashion (drain, destroy pipelines, destroy device — under the same lock as
/// creation) before the process tears the driver's threads down.
///
/// Production code must NOT use this: it states its sharing explicitly at its
/// own call sites (`Gpu::share` / `Gpu::new_like`), and a process that exits
/// drops its engines first.
#[cfg(not(target_arch = "wasm32"))]
pub mod testgpu {
    use super::{Gpu, WeakGpu};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    /// Keyed by the kernel slice's address: kernel sets are `'static` consts,
    /// so the pointer is a stable identity.
    static POOL: OnceLock<Mutex<HashMap<usize, WeakGpu>>> = OnceLock::new();

    /// A handle for `kernels` on this test binary's shared device.
    ///
    /// The pool holds only **weak** references: the device stays alive exactly
    /// as long as some test holds a handle, and dies in an orderly in-process
    /// destruction with the last one. That lifecycle is the entire point — a
    /// device that survives into process exit (leaked static) crashed the
    /// driver's worker threads intermittently, and tearing one down from an
    /// `atexit` hook crashed every run. Overlapping tests share one device;
    /// a gap in the schedule lets it die and the next test rebuilds it.
    pub fn dev(kernels: &'static [(&'static str, &'static str)]) -> Gpu {
        let cell = POOL.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = cell.lock().unwrap_or_else(|e| e.into_inner());
        let key = kernels.as_ptr() as usize;
        if let Some(g) = map.get(&key).and_then(|w| w.upgrade()) {
            return g;
        }
        // Any live entry can parent a new kernel set on the same device.
        let parent = map.values().find_map(|w| w.upgrade());
        let g = match parent {
            Some(p) => p.new_like(kernels),
            None => Gpu::new(kernels),
        };
        if let Some(w) = g.downgrade() {
            map.insert(key, w);
        }
        g
    }
}

// ---- native facade ----------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
mod native_facade {
    use super::{DeviceBuffer, Step};
    use backend_api::BufUsage;
    use std::sync::atomic::{AtomicU8, Ordering};

    /// Which compute backend a `Gpu` uses. A selection enum for the CLI — distinct
    /// from the [`backend_api::Backend`] *trait* the backends implement.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Backend {
        Wgpu,
        Cpu,
        /// Native Vulkan compute (ash + naga WGSL->SPIR-V). Falls back to wgpu if
        /// no Vulkan device/ICD is present.
        Vulkan,
    }

    static DEFAULT_BACKEND: AtomicU8 = AtomicU8::new(0); // 0=unset, 1=Wgpu, 2=Cpu, 3=Vulkan

    /// Set the process-wide default backend that `Gpu::new` builds. The CLI sets
    /// this from its `--device` flag. When unset, `BRAIN_DEVICE=cpu|vulkan`
    /// selects that backend; otherwise the default is wgpu.
    pub fn set_default_backend(b: Backend) {
        DEFAULT_BACKEND.store(
            match b {
                Backend::Wgpu => 1,
                Backend::Cpu => 2,
                Backend::Vulkan => 3,
            },
            Ordering::Relaxed,
        );
    }

    /// True iff a backend was explicitly selected via [`set_default_backend`]
    /// (i.e. the CLI saw a `--device` flag). Lets callers that historically
    /// defaulted to CPU (yolo) opt into the selected backend only when one was
    /// actually chosen.
    pub fn backend_selected() -> bool {
        DEFAULT_BACKEND.load(Ordering::Relaxed) != 0
    }

    /// The registry name of the backend that will be (or was) selected —
    /// `"wgpu" | "cpu" | "vulkan"`. Public so callers that record *what actually
    /// ran* (the perf suite's result fingerprint) don't have to re-derive it.
    pub fn backend_name() -> &'static str {
        resolve_backend_name()
    }

    /// How many distinct physical discrete GPUs this machine has. `0` on a
    /// GPU-less box (where `--device gpu` still works, via a software
    /// rasteriser). Multi-GPU tests gate on this so they skip instead of
    /// faulting inside the driver.
    pub fn discrete_gpu_count() -> usize {
        backend_wgpu::discrete_gpu_count()
    }

    /// The wgpu adapter this process selected, if a wgpu backend was built:
    /// `(description, is_software)`. `None` on a pure CPU/Vulkan run.
    ///
    /// A box with no real GPU still serves `--device gpu` through a software
    /// rasteriser, so any recorded performance number must carry this to be
    /// interpretable.
    pub fn adapter_info() -> Option<(String, bool)> {
        backend_wgpu::adapter_desc().map(|a| (a.description, a.software))
    }

    /// The registry name of the selected backend.
    fn resolve_backend_name() -> &'static str {
        match DEFAULT_BACKEND.load(Ordering::Relaxed) {
            1 => "wgpu",
            2 => "cpu",
            3 => "vulkan",
            _ => {
                if let Ok(v) = std::env::var("BRAIN_DEVICE") {
                    if v.eq_ignore_ascii_case("cpu") {
                        return "cpu";
                    }
                    if v.eq_ignore_ascii_case("vulkan") {
                        return "vulkan";
                    }
                }
                "wgpu"
            }
        }
    }

    /// Register the built-in backends' factories once (idempotent).
    fn register_builtins() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            backend_wgpu::register();
            backend_cpu::register();
            backend_vulkan::register();
        });
    }


    /// A weak `Gpu` handle: holds the device's compiled state without keeping it
    /// alive. The point is lifecycle: a pooled device must die with its **last
    /// real handle** — an orderly in-process destruction — because a device that
    /// survives into process exit (leaked in a static, or torn down from an
    /// `atexit` hook) crashes the NVIDIA driver's worker threads during
    /// teardown. Measured on the test suite: in-test drops never crash; a
    /// leaked static crashed intermittently; atexit teardown crashed every run.
    pub struct WeakGpu(Box<dyn backend_api::WeakBackend>);

    impl WeakGpu {
        /// A fresh strong handle, if the device is still alive.
        pub fn upgrade(&self) -> Option<Gpu> {
            self.0.upgrade().map(|inner| Gpu { inner })
        }
    }

    /// The compute device: a runtime-selected eager backend behind a trait object.
    /// All are compiled in; a given instance uses exactly one. Adding a backend
    /// never touches this type — the dispatch just forwards to `self.inner`.
    pub struct Gpu {
        inner: Box<dyn backend_api::Backend>,
    }

    impl Gpu {
        /// Build the default backend (see [`set_default_backend`] / `BRAIN_DEVICE`).
        /// Vulkan falls back to wgpu when no Vulkan device/ICD is present, so the
        /// build stays usable everywhere.
        pub fn new(kernels: &[(&str, &str)]) -> Gpu {
            register_builtins();
            let name = resolve_backend_name();
            match backend_api::create_backend(name, kernels) {
                Ok(inner) => Gpu { inner },
                Err(e) if name == "vulkan" => {
                    eprintln!("brain: Vulkan backend unavailable ({e}); falling back to wgpu");
                    Gpu {
                        inner: backend_api::create_backend("wgpu", kernels)
                            .expect("wgpu backend must be available"),
                    }
                }
                Err(e) => panic!("failed to build backend '{name}': {e}"),
            }
        }

        /// A second handle onto **this** device: same adapter, queue and compiled
        /// pipelines, its own command stream.
        ///
        /// Building a device costs seconds (device init + one shader compile per
        /// kernel), and several concurrent devices on one card are hostile to the
        /// driver. A process that needs many `Gpu`s — a server running several
        /// models, a test binary — should build one and `share()` it, rather than
        /// calling `new()` per model. Sharing is explicit so the number of real
        /// devices stays answerable by reading the code.
        ///
        /// Only the wgpu backend can share today; other backends fall back to
        /// building a fresh device, which is correct, just not free.
        pub fn share(&self) -> Gpu {
            match self.inner.share() {
                Some(inner) => Gpu { inner },
                None => panic!(
                    "this backend cannot share a device; build a new Gpu instead"
                ),
            }
        }

        /// [`Gpu::share`] when the backend supports it, else a fresh build of
        /// `kernels` — for callers that must work on every backend (the CPU JIT
        /// has no shareable device; building fresh there is cheap and correct).
        pub fn share_or_new(&self, kernels: &[(&str, &str)]) -> Gpu {
            match self.inner.share() {
                Some(inner) => Gpu { inner },
                None => Gpu::new(kernels),
            }
        }

        /// A `Gpu` for a **different kernel set** on the **same device** — how a
        /// process holds many models on one card. Backends without a shareable
        /// device (the CPU JIT compiles per kernel set anyway) just build fresh,
        /// which is correct and cheap there.
        pub fn new_like(&self, kernels: &[(&str, &str)]) -> Gpu {
            match self.inner.new_like(kernels) {
                Some(inner) => Gpu { inner },
                None => Gpu::new(kernels),
            }
        }

        /// Which backend this handle executes on: `"wgpu" | "cpu" | "vulkan"`.
        pub fn kind(&self) -> &'static str {
            self.inner.kind()
        }

        /// A weak handle for pools/fixtures — see [`WeakGpu`]. `None` when the
        /// backend has no shared state to weakly reference.
        pub fn downgrade(&self) -> Option<WeakGpu> {
            self.inner.downgrade().map(WeakGpu)
        }

        /// Build on the native CPU backend regardless of the default selection.
        pub fn new_cpu(kernels: &[(&str, &str)]) -> Gpu {
            Gpu { inner: Box::new(backend_cpu::CpuBackend::new(kernels)) }
        }

        /// Build on the wgpu backend regardless of the default selection.
        pub fn new_wgpu(kernels: &[(&str, &str)]) -> Gpu {
            Gpu { inner: Box::new(backend_wgpu::WgpuBackend::new(kernels)) }
        }

        /// Build `count` wgpu devices on DISTINCT physical GPUs from one adapter
        /// enumeration — the reliable multi-GPU path (two separate `new_wgpu`
        /// calls can reorder and collide on one card). Returns one [`Gpu`] per card.
        #[cfg(not(target_arch = "wasm32"))]
        pub fn new_wgpu_multi(kernels: &[(&str, &str)], count: usize) -> Vec<Gpu> {
            backend_wgpu::WgpuBackend::new_multi(kernels, count)
                .into_iter()
                .map(|b| Gpu { inner: Box::new(b) })
                .collect()
        }

        /// Build on the native Vulkan backend, or `Err` if no Vulkan device is present.
        pub fn try_new_vulkan(kernels: &[(&str, &str)]) -> Result<Gpu, String> {
            backend_vulkan::VulkanBackend::try_new(kernels).map(|g| Gpu { inner: Box::new(g) })
        }

        pub fn storage(&self, n: u64) -> DeviceBuffer {
            self.inner.storage(n)
        }
        pub fn storage_init(&self, name: &str, data: &[f32]) -> DeviceBuffer {
            self.inner.storage_init(name, data)
        }
        pub fn buffer(&self, label: &str, size: u64, usage: BufUsage) -> DeviceBuffer {
            self.inner.buffer(label, size, usage)
        }
        pub fn uniform_dynamic(&self, len: usize) -> DeviceBuffer {
            self.inner.uniform_dynamic(len)
        }
        pub fn write(&self, buf: &DeviceBuffer, data: &[u32]) {
            self.inner.write(buf, data)
        }
        pub fn read(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
            self.inner.read(buf, n)
        }
        pub fn poll_wait(&self) {
            self.inner.poll_wait()
        }
        /// Largest single storage-buffer binding this device allows, in bytes —
        /// the hardware ceiling a kernel's biggest buffer must fit under. Used to
        /// pick attention backends per-card instead of assuming a fixed limit.
        pub fn max_storage_binding_bytes(&self) -> u64 {
            self.inner.max_storage_binding_bytes()
        }
        /// Start all recorded work on the device without waiting (frame
        /// pipelining: overlap host work with device compute; a later `read`
        /// synchronises). No-op on eager backends (CPU).
        pub fn flush(&self) {
            self.inner.flush()
        }
        pub fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step {
            crate::assert_no_output_alias(bufs);
            self.inner.step(kind, bufs, params, threads)
        }
        pub fn step_sliced(&self, kind: usize, bufs: &[&DeviceBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> Step {
            // NB: sliced views of ONE buffer at disjoint offsets are legal and common
            // here, so no alias check — wgpu validates the concrete ranges.
            self.inner.step_sliced(kind, bufs, offsets, params, threads)
        }
        pub fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step {
            self.inner.step_buf(kind, ubuf, bufs, threads)
        }
        pub fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]) {
            self.inner.submit(clears, steps)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native_facade::{
    adapter_info, backend_name, backend_selected, discrete_gpu_count, set_default_backend, Backend,
    Gpu, WeakGpu,
};

// ---- wasm facade ------------------------------------------------------------

// On wasm there is only the wgpu/WebGPU backend; `Gpu` wraps it concretely (no
// `dyn`), and device init / read-back are async (no blocking executor / poll in
// the browser). Model code that uses `gpu_core::{Gpu, DeviceBuffer, Step}`
// compiles unchanged against either target.
#[cfg(target_arch = "wasm32")]
mod wasm_facade {
    use super::{DeviceBuffer, Step};
    use backend_api::{Backend, BufUsage};
    use backend_wgpu::WgpuBackend;

    pub struct Gpu {
        inner: WgpuBackend,
    }

    impl Gpu {
        /// Async device init + pipeline compile (the browser has no blocking
        /// executor, so there is no synchronous `new`).
        pub async fn new_async(kernels: &[(&str, &str)]) -> Gpu {
            Gpu { inner: WgpuBackend::new_async(kernels).await }
        }

        pub fn storage(&self, n: u64) -> DeviceBuffer {
            Backend::storage(&self.inner, n)
        }
        pub fn storage_init(&self, name: &str, data: &[f32]) -> DeviceBuffer {
            Backend::storage_init(&self.inner, name, data)
        }
        pub fn buffer(&self, label: &str, size: u64, usage: BufUsage) -> DeviceBuffer {
            Backend::buffer(&self.inner, label, size, usage)
        }
        pub fn uniform_dynamic(&self, len: usize) -> DeviceBuffer {
            Backend::uniform_dynamic(&self.inner, len)
        }
        pub fn write(&self, buf: &DeviceBuffer, data: &[u32]) {
            Backend::write(&self.inner, buf, data)
        }
        pub fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step {
            crate::assert_no_output_alias(bufs);
            Backend::step(&self.inner, kind, bufs, params, threads)
        }
        pub fn step_sliced(&self, kind: usize, bufs: &[&DeviceBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> Step {
            Backend::step_sliced(&self.inner, kind, bufs, offsets, params, threads)
        }
        pub fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step {
            Backend::step_buf(&self.inner, kind, ubuf, bufs, threads)
        }
        pub fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]) {
            Backend::submit(&self.inner, clears, steps)
        }

        /// Async buffer readback (browser): awaits the map callback.
        pub async fn read_async(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
            self.inner.read_async_buf(buf, n).await
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm_facade::Gpu;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_packs_float_bits() {
        assert_eq!(f(1.0), 1.0f32.to_bits());
        assert_eq!(f(-2.5), (-2.5f32).to_bits());
        assert_eq!(f32::from_bits(f(3.14159)), 3.14159f32);
    }

    #[test]
    fn dispatch_storage_and_readback() {
        // Exercises the whole plumbing: storage_init, step, submit, read. Uses the
        // CPU backend so it runs on headless CI with no GPU (both backends share
        // this dispatch contract, so it covers the wgpu path's plumbing too).
        let gpu = Gpu::new_cpu(&[("add2", kernels::ADD2)]);
        let a = gpu.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
        let b = gpu.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
        let out = gpu.storage(4);
        let step = gpu.step(0, &[&a, &b, &out], &[4], 4);
        gpu.submit(&[], &[step]);
        assert_eq!(gpu.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// Same dispatch contract on the native **Vulkan** backend, when a Vulkan
    /// device/ICD is present. Skips (does not fail) on hosts without Vulkan, so CI
    /// stays green; on a real device it validates the ash pipeline/descriptor/
    /// barrier/readback path against the CPU reference result.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn vulkan_dispatch_storage_and_readback() {
        let gpu = match Gpu::try_new_vulkan(&[("add2", kernels::ADD2)]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping vulkan test (no Vulkan device): {e}");
                return;
            }
        };
        let a = gpu.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
        let b = gpu.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
        let out = gpu.storage(4);
        let step = gpu.step(0, &[&a, &b, &out], &[4], 4);
        gpu.submit(&[], &[step]);
        assert_eq!(gpu.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);

        // Multi-dispatch with a read-after-write dependency exercises the inter-
        // dispatch memory barrier: out2 = (a+b) + b.
        let out2 = gpu.storage(4);
        let s1 = gpu.step(0, &[&a, &b, &out], &[4], 4);
        let s2 = gpu.step(0, &[&out, &b, &out2], &[4], 4);
        gpu.submit(&[], &[s1, s2]);
        assert_eq!(gpu.read(&out2, 4), vec![21.0, 42.0, 63.0, 84.0]);
    }
}

/// Reject a dispatch that binds its OUTPUT buffer as an input as well.
///
/// wgpu treats `STORAGE_READ_WRITE` as exclusive within one dispatch, so
/// `f(x, y, x)` is a validation error there even though the CPU backend would
/// happily run it — the failure then shows up only on GPU, often surfacing at
/// an unrelated later call. Binding a buffer twice READ-ONLY is fine (the
/// chunked-attention trio deliberately binds one fused qkv buffer as both the
/// q and kv views), so only the output slot is checked. Kernels that must
/// accumulate into themselves have a dedicated `*_inplace` form with a single
/// read_write binding.
#[track_caller]
fn assert_no_output_alias(bufs: &[&DeviceBuffer]) {
    // The output is the last binding (for the `*_inplace` family it is the
    // first, but then the *other* operand is last, so the same test holds).
    if let Some(out) = bufs.last() {
        if bufs.iter().filter(|b| b.alloc_id() == out.alloc_id()).count() > 1 {
            panic!(
                "dispatch binds its output buffer as an input as well; this is a \
                 wgpu usage-scope violation (STORAGE_READ_WRITE is exclusive). \
                 Use the kernel's *_inplace form, or write to a distinct buffer."
            );
        }
    }
}

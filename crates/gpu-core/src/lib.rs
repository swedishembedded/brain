// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Compute-device facade.
//!
//! `Gpu` is a thin wrapper over a [`backend_api::Backend`] - the eager, per-step
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

pub use backend_api::{
    f, BufUsage, DeviceBuffer, DeviceCaps, DeviceClass, DeviceStats, NumericSupport, Step,
    StepMeta,
};
pub use backend_api::select;

/// Per-kernel FLOP/int-OPS/bytes formulas + step-list accounting (offline
/// `Gpu::cost_of`, online `Gpu::ops_counters`).
pub mod cost;

/// File-backed persistence for measured kernel choices (S5).
#[cfg(not(target_arch = "wasm32"))]
pub mod tune;

/// The device's own measured roofline - the denominator every "% of peak"
/// claim in this engine divides by, so that claim is about the device that ran.
#[cfg(not(target_arch = "wasm32"))]
pub mod roof;

/// Per-kernel-kind profiling of a recorded pass - the one implementation the
/// model benches share.
#[cfg(not(target_arch = "wasm32"))]
pub mod profile;

/// Drop-in fast kernels a model inherits without editing its dispatch sites.
mod upgrade;

/// `--device` parsing and resolution: which compute is *schedulable*.
#[cfg(not(target_arch = "wasm32"))]
pub mod devices;
#[cfg(not(target_arch = "wasm32"))]
pub use devices::{
    ambient_compute_set, publish_compute_set, published_compute_set, ComputeSet, DeviceSpec,
    Inventory,
};

/// Declare that the process is exiting, after which GPU devices are leaked
/// rather than destroyed - see `backend_wgpu::set_process_exiting` for the
/// measured multi-device teardown fault this exists to avoid. Call it once,
/// from a process's own shutdown path, never from library code.
#[cfg(not(target_arch = "wasm32"))]
pub fn set_process_exiting() {
    backend_wgpu::set_process_exiting();
}


/// A process-wide device fixture for **test binaries** - explicit, documented,
/// and torn down before exit.
///
/// libtest runs tests concurrently in one process, and each GPU test that
/// builds its own `Gpu` puts another live device on the card. On the NVIDIA
/// driver this box runs, that shape fails two ways, both measured:
/// many concurrent devices deadlock (~50% of runs, every thread in futex
/// wait), and a device *leaked in a static* at process exit segfaults the
/// driver's worker thread during teardown - after every test has passed.
///
/// So: one parent device per test binary, one handle per kernel set via
/// [`Gpu::new_like`], and an `atexit` hook that drops everything in an orderly
/// fashion (drain, destroy pipelines, destroy device - under the same lock as
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
    /// destruction with the last one. That lifecycle is the entire point - a
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
    use backend_api::{BufUsage, StepMeta};
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Arc, Mutex};

    /// Which compute backend a `Gpu` uses. A selection enum for the CLI - distinct
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

    /// The registry name of the backend that will be (or was) selected -
    /// `"wgpu" | "cpu" | "vulkan"`. Public so callers that record *what actually
    /// ran* (the perf suite's result fingerprint) don't have to re-derive it.
    pub fn backend_name() -> &'static str {
        resolve_backend_name()
    }

    /// How many distinct physical discrete GPUs this machine has. `0` on a
    /// GPU-less box (where `--device gpu` still works, via a software
    /// rasteriser). Multi-GPU tests gate on this so they skip instead of
    /// faulting inside the driver. Answered by the canonical device registry -
    /// one enumeration, cached.
    pub fn discrete_gpu_count() -> usize {
        crate::devices::gpus()
            .iter()
            .filter(|d| d.identity.class == backend_api::DeviceClass::DiscreteGpu)
            .count()
    }

    /// How many GPUs the wgpu backend can actually schedule work on - discrete
    /// cards, or (only when no discrete card is present) the integrated GPU.
    /// This is the canonical device registry's own count, which already applies
    /// that discrete-preferred precedence (see `devices::DeviceRegistry`).
    /// Unlike [`discrete_gpu_count`], an integrated-only box (e.g. Intel Arc on
    /// Meteor Lake) reports 1 here, not 0 - use this wherever "is there GPU
    /// hardware brain should schedule on" is the real question (the `--device`
    /// default set, the no-nvidia-smi budgeting fallback); keep
    /// `discrete_gpu_count` where multi-GPU sharding specifically needs distinct
    /// discrete cards.
    pub fn visible_gpu_count() -> usize {
        crate::devices::gpus().len()
    }

    /// Identities of the physical GPUs the wgpu backend can bind, in its own
    /// enumeration order - for `brain devices` to report per-card backend
    /// visibility against the canonical registry (matched by identity).
    pub fn wgpu_visible_gpus() -> Vec<backend_api::GpuIdentity> {
        backend_wgpu::enumerate_gpus()
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

    /// The capabilities of the first device this process built, if any - the
    /// machine-readable sibling of [`adapter_info`], for result fingerprints
    /// (`brain perf`'s env block). Per-handle callers use [`Gpu::caps`].
    pub fn device_caps() -> Option<backend_api::DeviceCaps> {
        CAPS.get().cloned()
    }

    static CAPS: std::sync::OnceLock<backend_api::DeviceCaps> = std::sync::OnceLock::new();

    /// Remember the first-built device's caps for [`device_caps`] (first write
    /// wins, mirroring the adapter record).
    fn record_caps(inner: &dyn backend_api::Backend) {
        let _ = CAPS.set(inner.caps());
    }

    /// The registry name of the selected backend.
    ///
    /// When no backend was explicitly selected via [`set_default_backend`]
    /// (the CLI's `--device` path), this goes through
    /// [`crate::devices::ambient_compute_set`] - the single, strong-grammar
    /// resolution of `BRAIN_DEVICE` shared with `--device`'s own parser -
    /// rather than re-deriving a second, weaker interpretation here. See
    /// `ambient_compute_set`'s doc for why: a bare env re-parse used to
    /// understand only `"cpu"`/`"vulkan"` and silently mangled everything
    /// else (`gpu0`, `npu`, `cpu0-7`, …) to "just use wgpu, ambient card".
    fn resolve_backend_name() -> &'static str {
        match DEFAULT_BACKEND.load(Ordering::Relaxed) {
            1 => "wgpu",
            2 => "cpu",
            3 => "vulkan",
            _ => match crate::devices::ambient_compute_set().backend {
                crate::devices::Backend::Wgpu => "wgpu",
                crate::devices::Backend::Cpu => "cpu",
                crate::devices::Backend::Vulkan => "vulkan",
            },
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
    /// real handle** - an orderly in-process destruction - because a device that
    /// survives into process exit (leaked in a static, or torn down from an
    /// `atexit` hook) crashes the NVIDIA driver's worker threads during
    /// teardown. Measured on the test suite: in-test drops never crash; a
    /// leaked static crashed intermittently; atexit teardown crashed every run.
    pub struct WeakGpu {
        weak: Box<dyn backend_api::WeakBackend>,
        names: Arc<Vec<String>>,
    }

    impl WeakGpu {
        /// A fresh strong handle, if the device is still alive.
        pub fn upgrade(&self) -> Option<Gpu> {
            self.weak.upgrade().map(|inner| Gpu::wrap(inner, self.names.clone()))
        }
    }

    /// The compute device: a runtime-selected eager backend behind a trait object.
    /// All are compiled in; a given instance uses exactly one. Adding a backend
    /// never touches this type - the dispatch just forwards to `self.inner`.
    pub struct Gpu {
        inner: Box<dyn backend_api::Backend>,
        /// Kernel names of this handle's pipeline set, index-aligned with the
        /// `kind` passed to `step*` - what resolves a recorded step back to a
        /// cost formula.
        names: Arc<Vec<String>>,
        /// Online OPS counters for THIS handle, folded in at `submit`. Handles
        /// are per model/stage by construction (`share`/`new_like` start fresh
        /// counters), so these are per-device, per-model numbers.
        counters: Mutex<crate::cost::CostReport>,
        /// Whether `submit` folds steps into `counters` at all. OFF by default:
        /// the online tally is a mutex lock + a ~110-arm string match PER
        /// DISPATCH, which a production decode loop must not pay for a ledger
        /// nothing reads. Armed by [`Gpu::reset_ops_counters`] - the call every
        /// measuring consumer (flops CLI, flops tests, benches) already makes
        /// before its measured run, so measurement flows are unchanged.
        cost_enabled: std::sync::atomic::AtomicBool,
        /// Active transparent kernel upgrades for THIS device:
        /// `(registered slot, faster slot, thread multiplier)`. Usually empty;
        /// see [`crate::upgrade`] for what qualifies and why the seam exists.
        upgrades: Vec<(usize, usize, u32)>,
    }

    impl Gpu {
        fn wrap(inner: Box<dyn backend_api::Backend>, names: Arc<Vec<String>>) -> Gpu {
            // Per handle, once: the policy is a pure function of names + caps,
            // so `step` costs one compare against a usually-empty list.
            let upgrades = crate::upgrade::resolve(&names, &inner.caps());
            Gpu { inner, names, counters: Mutex::new(Default::default()), cost_enabled: std::sync::atomic::AtomicBool::new(false), upgrades }
        }

        fn kernel_names(kernels: &[(&str, &str)]) -> Arc<Vec<String>> {
            Arc::new(kernels.iter().map(|(n, _)| n.to_string()).collect())
        }

        /// The caller's kernel list with any drop-in fast variants **appended**
        /// (see [`crate::upgrade`]). Every constructor funnels through this, so
        /// a model inherits the faster kernel by registering the slow one -
        /// which is what `crates/vae` did not get when `gn_stats` was fixed
        /// for DIAMOND.
        /// Returns a borrowed view when there is nothing to add.
        fn expanded<'a>(kernels: &'a [(&'a str, &'a str)]) -> std::borrow::Cow<'a, [(&'a str, &'a str)]> {
            match crate::upgrade::expand(kernels) {
                Some(v) => std::borrow::Cow::Owned(v),
                None => std::borrow::Cow::Borrowed(kernels),
            }
        }
        /// Build the default backend (see [`set_default_backend`] / `BRAIN_DEVICE`).
        /// Vulkan falls back to wgpu when no Vulkan device/ICD is present, so the
        /// build stays usable everywhere.
        ///
        /// GPU placement is ambient, resolved through the canonical device
        /// registry: a scoped [`crate::devices::with_gpu`] selection, else the
        /// `--device gpu<i>` pin, else `BRAIN_GPU_INDEX` (user input, parsed
        /// once), else canonical card 0. Explicit placement uses [`Gpu::new_on`].
        pub fn new(kernels: &[(&str, &str)]) -> Gpu {
            let kernels = &Self::expanded(kernels);
            register_builtins();
            let name = resolve_backend_name();
            let inner: Box<dyn backend_api::Backend> = match name {
                "wgpu" => Self::build_wgpu(kernels, crate::devices::selected_device()),
                "vulkan" => {
                    let dev = crate::devices::selected_device();
                    let built = match dev {
                        Some(d) => backend_vulkan::VulkanBackend::try_new_on(kernels, &d.identity),
                        None => backend_vulkan::VulkanBackend::try_new(kernels),
                    };
                    match built {
                        Ok(b) => Box::new(b),
                        Err(e) => {
                            eprintln!("brain: Vulkan backend unavailable ({e}); falling back to wgpu");
                            Self::build_wgpu(kernels, dev)
                        }
                    }
                }
                _ => match backend_api::create_backend(name, kernels) {
                    Ok(inner) => inner,
                    Err(e) => panic!("failed to build backend '{name}': {e}"),
                },
            };
            record_caps(inner.as_ref());
            Gpu::wrap(inner, Self::kernel_names(kernels))
        }

        /// wgpu backend on `dev` (identity-matched) or wgpu's own default when
        /// no physical card exists (the software-rasteriser fallback).
        fn build_wgpu(
            kernels: &[(&str, &str)],
            dev: Option<&crate::devices::DeviceId>,
        ) -> Box<dyn backend_api::Backend> {
            Box::new(match dev {
                Some(d) => backend_wgpu::WgpuBackend::new_on(kernels, &d.identity),
                None => backend_wgpu::WgpuBackend::new(kernels),
            })
        }

        /// Build on the specific physical card `dev` - explicit placement, no
        /// ambient/env input consulted. Respects the selected backend *class*:
        /// under `--device cpu` / `BRAIN_DEVICE=cpu` this still builds the CPU
        /// backend (there is one CPU; the card is moot), so device-plumbing
        /// call sites work unchanged on CPU-only runs.
        pub fn new_on(dev: &crate::devices::DeviceId, kernels: &[(&str, &str)]) -> Gpu {
            let kernels = &Self::expanded(kernels);
            register_builtins();
            let inner: Box<dyn backend_api::Backend> = match resolve_backend_name() {
                "cpu" => Box::new(backend_cpu::CpuBackend::new(kernels)),
                "vulkan" => match backend_vulkan::VulkanBackend::try_new_on(kernels, &dev.identity) {
                    Ok(b) => Box::new(b),
                    Err(e) => {
                        eprintln!("brain: Vulkan backend unavailable ({e}); falling back to wgpu");
                        Box::new(backend_wgpu::WgpuBackend::new_on(kernels, &dev.identity))
                    }
                },
                _ => Box::new(backend_wgpu::WgpuBackend::new_on(kernels, &dev.identity)),
            };
            record_caps(inner.as_ref());
            Gpu::wrap(inner, Self::kernel_names(kernels))
        }

        /// [`Gpu::new_on`] by canonical index - the common call shape where the
        /// index came from a `Shard`, `residency::Device::Gpu(i)`, or a parsed
        /// user string. Errors on an out-of-range index when the machine has
        /// cards; on a GPU-less box it falls back to [`Gpu::new`] (the CPU or
        /// software path - placement is moot there).
        pub fn new_on_index(index: u32, kernels: &[(&str, &str)]) -> Result<Gpu, String> {
            if crate::devices::gpus().is_empty() {
                return Ok(Gpu::new(kernels));
            }
            Ok(Gpu::new_on(crate::devices::device(index)?, kernels))
        }

        /// A second handle onto **this** device: same adapter, queue and compiled
        /// pipelines, its own command stream.
        ///
        /// Building a device costs seconds (device init + one shader compile per
        /// kernel), and several concurrent devices on one card are hostile to the
        /// driver. A process that needs many `Gpu`s - a server running several
        /// models, a test binary - should build one and `share()` it, rather than
        /// calling `new()` per model. Sharing is explicit so the number of real
        /// devices stays answerable by reading the code.
        ///
        /// Only the wgpu backend can share today; other backends fall back to
        /// building a fresh device, which is correct, just not free.
        pub fn share(&self) -> Gpu {
            match self.inner.share() {
                Some(inner) => Gpu::wrap(inner, self.names.clone()),
                None => panic!(
                    "this backend cannot share a device; build a new Gpu instead"
                ),
            }
        }

        /// [`Gpu::share`] when the backend supports it, else a fresh build of
        /// `kernels` - for callers that must work on every backend (the CPU JIT
        /// has no shareable device; building fresh there is cheap and correct).
        pub fn share_or_new(&self, kernels: &[(&str, &str)]) -> Gpu {
            match self.inner.share() {
                Some(inner) => Gpu::wrap(inner, self.names.clone()),
                None => Gpu::new(kernels),
            }
        }

        /// A `Gpu` for a **different kernel set** on the **same device** - how a
        /// process holds many models on one card. Backends without a shareable
        /// device (the CPU JIT compiles per kernel set anyway) just build fresh,
        /// which is correct and cheap there.
        pub fn new_like(&self, kernels: &[(&str, &str)]) -> Gpu {
            let kernels = &Self::expanded(kernels);
            match self.inner.new_like(kernels) {
                Some(inner) => Gpu::wrap(inner, Self::kernel_names(kernels)),
                None => Gpu::new(kernels),
            }
        }

        /// Which backend this handle executes on: `"wgpu" | "cpu" | "vulkan"`.
        pub fn kind(&self) -> &'static str {
            self.inner.kind()
        }

        /// What this device can actually do - class, limits, numeric tiers.
        /// Cached at backend construction; reading it is free.
        ///
        /// The two roofline fields are overlaid from whatever
        /// [`crate::roof`] has already *measured* - reading caps never has the
        /// side effect of running probe kernels, so they stay `None` until a
        /// caller asks for them with `roof::ensure`. An unknown capability is
        /// never assumed present.
        pub fn caps(&self) -> backend_api::DeviceCaps {
            let mut c = self.inner.caps();
            if let Some(r) = crate::roof::known() {
                c.peak_gflops = Some(r.gflops);
                c.peak_bandwidth_gbs = Some(r.gbs);
            }
            c
        }

        /// Turn per-kernel DEVICE timing on/off; `false` = this backend cannot
        /// time kernels. See [`Self::kernel_times`].
        pub fn set_kernel_timing(&self, on: bool) -> bool {
            self.inner.set_kernel_timing(on)
        }

        /// Per-kernel accumulated DEVICE time since the last reset, as
        /// `(kernel, ms, calls)`. `None` where the backend cannot time kernels.
        ///
        /// This is the ONLY honest source for attributing time BETWEEN kernels.
        /// Host wall-clock around a drained slice measures launch + execute +
        /// fence, whose floor is roughly constant and therefore inflates small
        /// kernels in inverse proportion to their size - up to 29x measured,
        /// enough to invert a ranking.
        pub fn kernel_times(&self) -> Option<Vec<(String, f64, u64)>> {
            self.inner.kernel_times()
        }

        /// Zero the per-kernel accumulators.
        pub fn reset_kernel_times(&self) {
            self.inner.reset_kernel_times()
        }

        /// Device-op accounting for this handle (submits/dispatches/readbacks)
        /// since creation. `None` where the backend does not count - report
        /// null, never zero.
        pub fn stats(&self) -> Option<backend_api::DeviceStats> {
            self.inner.stats()
        }

        /// Print the per-kernel `BRAIN_PROFILE` table now - the resident-model
        /// escape hatch for a profile that otherwise only prints at drop.
        pub fn dump_profile(&self) {
            self.inner.dump_profile()
        }

        /// A weak handle for pools/fixtures - see [`WeakGpu`]. `None` when the
        /// backend has no shared state to weakly reference.
        pub fn downgrade(&self) -> Option<WeakGpu> {
            self.inner.downgrade().map(|weak| WeakGpu { weak, names: self.names.clone() })
        }

        /// Build on the native CPU backend regardless of the default selection.
        pub fn new_cpu(kernels: &[(&str, &str)]) -> Gpu {
            let kernels = &Self::expanded(kernels);
            let inner: Box<dyn backend_api::Backend> =
                Box::new(backend_cpu::CpuBackend::new(kernels));
            record_caps(inner.as_ref());
            Gpu::wrap(inner, Self::kernel_names(kernels))
        }

        /// Build on the wgpu backend regardless of the default selection.
        /// Placement is ambient (registry-resolved), like [`Gpu::new`].
        pub fn new_wgpu(kernels: &[(&str, &str)]) -> Gpu {
            let kernels = &Self::expanded(kernels);
            let inner = Self::build_wgpu(kernels, crate::devices::selected_device());
            record_caps(inner.as_ref());
            Gpu::wrap(inner, Self::kernel_names(kernels))
        }

        /// Build `count` wgpu devices on DISTINCT physical GPUs - canonical
        /// cards `0..count`, each selected by identity through the registry
        /// (identity matching is what makes repeated builds collision-free;
        /// position-indexed enumeration was observed to reorder between calls).
        /// GPU-less boxes fall back to the single-enumeration position path so
        /// its "need N GPUs" assertion still reports honestly.
        #[cfg(not(target_arch = "wasm32"))]
        pub fn new_wgpu_multi(kernels: &[(&str, &str)], count: usize) -> Vec<Gpu> {
            let kernels = &Self::expanded(kernels);
            let names = Self::kernel_names(kernels);
            let devs = crate::devices::gpus();
            if devs.len() >= count {
                return devs[..count]
                    .iter()
                    .map(|d| {
                        let inner: Box<dyn backend_api::Backend> =
                            Box::new(backend_wgpu::WgpuBackend::new_on(kernels, &d.identity));
                        record_caps(inner.as_ref());
                        Gpu::wrap(inner, names.clone())
                    })
                    .collect();
            }
            backend_wgpu::WgpuBackend::new_multi(kernels, count)
                .into_iter()
                .map(|b| Gpu::wrap(Box::new(b), names.clone()))
                .collect()
        }

        /// Build on the native Vulkan backend, or `Err` if no Vulkan device is present.
        pub fn try_new_vulkan(kernels: &[(&str, &str)]) -> Result<Gpu, String> {
            let kernels = &Self::expanded(kernels);
            backend_vulkan::VulkanBackend::try_new(kernels)
                .map(|g| Gpu::wrap(Box::new(g), Self::kernel_names(kernels)))
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
        /// [`Self::write`] for a host `f32` slice - the bit reinterpretation the
        /// backend's `u32` upload wants.
        ///
        /// The inverse of [`Self::read`], which already returns `Vec<f32>`; this
        /// side was missing, so every model that uploads host floats re-derived
        /// `data.iter().map(f32::to_bits).collect()` at its call site. Two of
        /// those had congealed into byte-identical private `fn write` helpers
        /// (`sdxlunet::model`, `controlnet::model`) - AGENTS.md's "one
        /// implementation" rule, and the reason this lives on the device facade
        /// that owns the `u32` half rather than in any model crate.
        pub fn write_f32(&self, buf: &DeviceBuffer, data: &[f32]) {
            self.write(buf, bytemuck::cast_slice(data))
        }
        /// [`Self::write`] starting at `offset_words` (u32 words) into `buf`,
        /// rather than always the start. See `backend_api::Backend::write_at`
        /// for why this exists: on this engine's non-ReBAR discrete GPUs, one
        /// `write`/`write_f32` call sized to a whole multi-GB tensor leaves a
        /// same-size wgpu staging allocation permanently resident (measured
        /// 2.00x, `crates/gpu-core/tests/vram_overhead.rs`). A caller streaming
        /// a large upload should chunk through this (or [`Self::write_f32_chunked`])
        /// instead of one `write_f32` for the whole tensor.
        pub fn write_at(&self, buf: &DeviceBuffer, offset_words: u64, data: &[u32]) {
            self.inner.write_at(buf, offset_words, data)
        }
        /// [`Self::write_at`] for a host `f32` slice.
        pub fn write_f32_at(&self, buf: &DeviceBuffer, offset_words: u64, data: &[f32]) {
            self.write_at(buf, offset_words, bytemuck::cast_slice(data))
        }
        /// Upload `data` into `buf` (starting at word 0) in `chunk_words`-sized
        /// pieces via repeated [`Self::write_f32_at`] calls, bounding the
        /// resident staging cost of a large weight-tensor upload to roughly
        /// `chunk_words * 4` bytes instead of the whole tensor. See
        /// `write_at`'s doc for why this matters on non-ReBAR discrete GPUs.
        /// `chunk_words` of 0 or larger than `data.len()` degrades to one
        /// `write_f32` call.
        pub fn write_f32_chunked(&self, buf: &DeviceBuffer, data: &[f32], chunk_words: usize) {
            let chunk = if chunk_words == 0 { data.len().max(1) } else { chunk_words };
            for (i, part) in data.chunks(chunk).enumerate() {
                self.write_f32_at(buf, (i * chunk) as u64, part);
            }
        }
        pub fn read(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
            self.inner.read(buf, n)
        }
        pub fn poll_wait(&self) {
            self.inner.poll_wait()
        }
        /// [`Self::poll_wait`], bounded: `false` on timeout instead of blocking
        /// forever. See `backend_api::Backend::poll_wait_timeout`'s doc for the
        /// full contract, including which backends actually bound anything
        /// (currently only `backend-wgpu`) versus inheriting the
        /// always-blocking default.
        pub fn poll_wait_timeout(&self, timeout: std::time::Duration) -> bool {
            self.inner.poll_wait_timeout(timeout)
        }
        /// Largest single storage-buffer binding this device allows, in bytes -
        /// the hardware ceiling a kernel's biggest buffer must fit under. Used to
        /// pick attention backends per-card instead of assuming a fixed limit.
        pub fn max_storage_binding_bytes(&self) -> u64 {
            self.inner.max_storage_binding_bytes()
        }
        /// Largest single ALLOCATION this device allows, in bytes - see
        /// `backend_api::Backend::max_buffer_bytes` for why this is a separate
        /// (and usually larger) number than the binding ceiling above.
        pub fn max_buffer_bytes(&self) -> u64 {
            self.inner.max_buffer_bytes()
        }
        /// Start all recorded work on the device without waiting (frame
        /// pipelining: overlap host work with device compute; a later `read`
        /// synchronises). No-op on eager backends (CPU).
        pub fn flush(&self) {
            self.inner.flush()
        }
        /// Bytes currently buried (dropped, not yet actually freed on-device)
        /// by a backend whose reclaim is deferred - see
        /// `backend_api::Backend::buried_bytes`'s doc. 0 on backends that
        /// reclaim eagerly.
        pub fn buried_bytes(&self) -> u64 {
            self.inner.buried_bytes()
        }
        /// Total queue submissions (each a blocking submit + fence wait) since
        /// construction - see `backend_api::Backend::queue_submits`'s doc. 0 on
        /// backends that don't count (CPU/eager backends have no separate
        /// submit step).
        pub fn queue_submits(&self) -> u64 {
            self.inner.queue_submits()
        }
        /// Count of actual deferred-reclaim events since construction - see
        /// `backend_api::Backend::reclaim_event_count`'s doc. 0 on backends
        /// that reclaim eagerly.
        pub fn reclaim_event_count(&self) -> u64 {
            self.inner.reclaim_event_count()
        }
        /// Record one dispatch of pipeline `kind`.
        ///
        /// `kind`/`threads` pass through [`crate::upgrade`] first: where a
        /// contract-identical faster kernel is registered and this device wants
        /// it, the *dispatch* is redirected and its thread count scaled.
        ///
        /// The recorded [`StepMeta`] deliberately keeps the CALLER's `kind` and
        /// `threads`. The upgrade is transparent by definition, so it must not
        /// leak into the caller's index space: profilers and cost harnesses map
        /// `meta.kernel` through *their own* kernel list (`flux2_bench` does
        /// exactly that), and an appended pipeline slot would index past the end
        /// of it. Which kernel physically ran is the backend's record -
        /// `BRAIN_PROFILE=1` names the real pipeline.
        pub fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step {
            crate::assert_no_output_alias(bufs);
            let (k, t) = crate::upgrade::apply(&self.upgrades, kind, threads);
            self.inner
                .step(k, bufs, params, t)
                .with_meta(StepMeta { kernel: kind, params: Some(params.to_vec()), threads })
        }
        pub fn step_sliced(&self, kind: usize, bufs: &[&DeviceBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> Step {
            // NB: sliced views of ONE buffer at disjoint offsets are legal and common
            // here, so no alias check - wgpu validates the concrete ranges.
            let (k, t) = crate::upgrade::apply(&self.upgrades, kind, threads);
            self.inner
                .step_sliced(k, bufs, offsets, params, t)
                .with_meta(StepMeta { kernel: kind, params: Some(params.to_vec()), threads })
        }
        pub fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step {
            // The uniform lives in a caller-owned buffer: shape params unknown
            // here, so the cost side gets only kernel + threads.
            let (k, t) = crate::upgrade::apply(&self.upgrades, kind, threads);
            self.inner
                .step_buf(k, ubuf, bufs, t)
                .with_meta(StepMeta { kernel: kind, params: None, threads })
        }
        pub fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]) {
            // Only when armed (see `cost_enabled`): tallying is a mutex lock
            // plus a per-dispatch string match - measurement machinery, not a
            // cost every production decode step should pay.
            if !steps.is_empty() && self.cost_enabled.load(std::sync::atomic::Ordering::Relaxed) {
                let mut ctr = self.counters.lock().unwrap_or_else(|e| e.into_inner());
                crate::cost::tally(&mut ctr, &self.names, steps);
            }
            self.inner.submit(clears, steps)
        }

        // ---- FLOP/OPS accounting (see `gpu_core::cost`) ---------------------

        /// OFFLINE cost of a recorded step list - no execution. The steps must
        /// have been recorded through a handle of this kernel set (kernel
        /// indices resolve through this handle's pipeline names).
        pub fn cost_of(&self, steps: &[Step]) -> crate::cost::CostReport {
            let mut r = crate::cost::CostReport::default();
            crate::cost::tally(&mut r, &self.names, steps);
            r
        }

        /// ONLINE counters: everything submitted through THIS handle since the
        /// last [`Gpu::reset_ops_counters`] - which is also what ARMS the
        /// tally (it is off by default; see `cost_enabled`). One handle is one
        /// device context, so a sharded pipeline reads per-stage numbers from
        /// each stage's handle.
        pub fn ops_counters(&self) -> crate::cost::CostReport {
            self.counters.lock().unwrap_or_else(|e| e.into_inner()).clone()
        }

        /// Reset the online counters and ENABLE per-submit tallying on this
        /// handle (after warm-up, before a measured run - the call every
        /// measuring consumer already makes first). Tallying starts disabled
        /// so production serving never pays for a ledger nothing reads.
        pub fn reset_ops_counters(&self) {
            self.cost_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
            *self.counters.lock().unwrap_or_else(|e| e.into_inner()) = Default::default();
        }

        /// The registered name of pipeline slot `kind` on this handle.
        pub fn kernel_name(&self, kind: usize) -> Option<&str> {
            self.names.get(kind).map(|s| s.as_str())
        }

        /// The name of the pipeline a dispatch of slot `kind` PHYSICALLY runs
        /// on this device - the redirected slot's name when a transparent
        /// [`crate::upgrade`] applies, otherwise `kind`'s own name. Backend
        /// device-timing maps ([`Gpu::kernel_times`]) are keyed by physical
        /// names, so profile attribution translates through this; everything
        /// else keeps seeing the caller's own index space (see `step`'s doc).
        pub fn physical_kernel_name(&self, kind: usize) -> Option<&str> {
            let (k, _) = crate::upgrade::apply(&self.upgrades, kind, 1);
            self.names.get(k).map(|s| s.as_str())
        }

        /// The pipeline slot a kernel name occupies on this handle, or `None`
        /// if this model did not register it.
        ///
        /// The inverse of [`Gpu::kernel_name`], for **shared block builders**
        /// that want to use an optional faster kernel variant without forcing
        /// every model's `KernelIds` literal to grow a field: the builder asks
        /// whether the variant is present and falls back when it is not, so a
        /// model opts in purely by adding the kernel to its PIPELINES list.
        /// A model with fixed indices should still pass them explicitly - this
        /// is for the shared builders, not for hot per-element lookups.
        pub fn kernel_index(&self, name: &str) -> Option<usize> {
            self.names.iter().position(|n| n == name)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native_facade::{
    adapter_info, backend_name, backend_selected, device_caps, discrete_gpu_count,
    set_default_backend, visible_gpu_count, wgpu_visible_gpus, Backend, Gpu, WeakGpu,
};

// ---- wasm facade ------------------------------------------------------------

// On wasm there is only the wgpu/WebGPU backend; `Gpu` wraps it concretely (no
// `dyn`), and device init / read-back are async (no blocking executor / poll in
// the browser). Model code that uses `gpu_core::{Gpu, DeviceBuffer, Step}`
// compiles unchanged against either target.
#[cfg(target_arch = "wasm32")]
mod wasm_facade {
    use super::{DeviceBuffer, Step};
    use backend_api::{Backend, BufUsage, StepMeta};
    use backend_wgpu::WgpuBackend;
    use std::sync::Mutex;

    pub struct Gpu {
        inner: WgpuBackend,
        names: Vec<String>,
        counters: Mutex<crate::cost::CostReport>,
        /// See the native facade: online tallying is off until
        /// `reset_ops_counters` arms it.
        cost_enabled: std::sync::atomic::AtomicBool,
        /// See the native facade / [`crate::upgrade`].
        upgrades: Vec<(usize, usize, u32)>,
    }

    impl Gpu {
        /// Async device init + pipeline compile (the browser has no blocking
        /// executor, so there is no synchronous `new`).
        pub async fn new_async(kernels: &[(&str, &str)]) -> Gpu {
            let expanded = crate::upgrade::expand(kernels);
            let kernels: &[(&str, &str)] = expanded.as_deref().unwrap_or(kernels);
            let inner = WgpuBackend::new_async(kernels).await;
            let names: Vec<String> = kernels.iter().map(|(n, _)| n.to_string()).collect();
            let upgrades = crate::upgrade::resolve(&names, &Backend::caps(&inner));
            Gpu { inner, names, counters: Mutex::new(Default::default()), cost_enabled: std::sync::atomic::AtomicBool::new(false), upgrades }
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
        /// [`Self::write`] for a host `f32` slice. Kept in step with the native
        /// facade so a model is written once against one `Gpu` surface.
        pub fn write_f32(&self, buf: &DeviceBuffer, data: &[f32]) {
            self.write(buf, bytemuck::cast_slice(data))
        }
        pub fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step {
            crate::assert_no_output_alias(bufs);
            let (k, t) = crate::upgrade::apply(&self.upgrades, kind, threads);
            Backend::step(&self.inner, k, bufs, params, t)
                .with_meta(StepMeta { kernel: kind, params: Some(params.to_vec()), threads })
        }
        pub fn step_sliced(&self, kind: usize, bufs: &[&DeviceBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> Step {
            let (k, t) = crate::upgrade::apply(&self.upgrades, kind, threads);
            Backend::step_sliced(&self.inner, k, bufs, offsets, params, t)
                .with_meta(StepMeta { kernel: kind, params: Some(params.to_vec()), threads })
        }
        pub fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step {
            let (k, t) = crate::upgrade::apply(&self.upgrades, kind, threads);
            Backend::step_buf(&self.inner, k, ubuf, bufs, t)
                .with_meta(StepMeta { kernel: kind, params: None, threads })
        }
        pub fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]) {
            // Off until armed - see the native facade's `submit`.
            if !steps.is_empty() && self.cost_enabled.load(std::sync::atomic::Ordering::Relaxed) {
                let mut ctr = self.counters.lock().unwrap_or_else(|e| e.into_inner());
                crate::cost::tally(&mut ctr, &self.names, steps);
            }
            Backend::submit(&self.inner, clears, steps)
        }

        /// OFFLINE cost of a recorded step list - see the native facade.
        pub fn cost_of(&self, steps: &[Step]) -> crate::cost::CostReport {
            let mut r = crate::cost::CostReport::default();
            crate::cost::tally(&mut r, &self.names, steps);
            r
        }

        /// ONLINE counters accumulated at `submit` - see the native facade.
        pub fn ops_counters(&self) -> crate::cost::CostReport {
            self.counters.lock().unwrap_or_else(|e| e.into_inner()).clone()
        }

        /// Reset the online counters and arm tallying - see the native facade.
        pub fn reset_ops_counters(&self) {
            self.cost_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
            *self.counters.lock().unwrap_or_else(|e| e.into_inner()) = Default::default();
        }

        /// Async buffer readback (browser): awaits the map callback.
        pub async fn read_async(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
            self.inner.read_async_buf(buf, n).await
        }

        /// Device capabilities (the browser floor plus whatever WebGPU reports).
        pub fn caps(&self) -> backend_api::DeviceCaps {
            Backend::caps(&self.inner)
        }

        /// The registered name of pipeline slot `kind` on this handle.
        pub fn kernel_name(&self, kind: usize) -> Option<&str> {
            self.names.get(kind).map(|s| s.as_str())
        }

        /// The pipeline slot a kernel name occupies - see the native facade's
        /// `kernel_index`; shared block builders use it to pick an optional
        /// kernel variant without changing every model's `KernelIds`.
        pub fn kernel_index(&self, name: &str) -> Option<usize> {
            self.names.iter().position(|n| n == name)
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm_facade::Gpu;

/// Reject a dispatch that binds its OUTPUT buffer as an input as well.
///
/// wgpu treats `STORAGE_READ_WRITE` as exclusive within one dispatch, so
/// `f(x, y, x)` is a validation error there even though the CPU backend would
/// happily run it - the failure then shows up only on GPU, often surfacing at
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_packs_float_bits() {
        assert_eq!(f(1.0), 1.0f32.to_bits());
        assert_eq!(f(-2.5), (-2.5f32).to_bits());
        assert_eq!(f32::from_bits(f(7.2589)), 7.2589f32);
    }

    /// The capability contract every consumer relies on: fp32 is always
    /// available, the CPU backend reports itself honestly (class, cores,
    /// unified memory), and nothing unknowable is assumed present.
    #[test]
    fn cpu_caps_are_honest() {
        let gpu = Gpu::new_cpu(&[("add2", kernels::ADD2)]);
        let c = gpu.caps();
        assert_eq!(c.class, DeviceClass::Cpu);
        assert!(c.numeric.f32, "fp32 is the portable baseline - always true");
        assert!(c.compute_units.unwrap_or(0) >= 1);
        assert!(c.unified_memory, "host memory IS device memory on the CPU");
        // The CPU JIT cannot run the multi-barrier packed-int8 GEMMs, and
        // fast-f16/coop-matrix don't exist there: claiming any of these would
        // send a selector down a path that cannot execute.
        assert!(!c.numeric.int8_dot && !c.numeric.f16 && !c.numeric.coop_matrix);
    }

    /// A template-specialised kernel (S3) is just another (name, source) pair:
    /// it compiles and runs beside its base through the ordinary backend path,
    /// and the tuned constant actually changes what executes.
    #[test]
    fn specialised_kernel_variant_runs_beside_its_base() {
        static SRC: &str = "const SCALE: u32 = 2u;\n\
            struct P { n: u32 };\n\
            @group(0) @binding(0) var<uniform> p: P;\n\
            @group(0) @binding(1) var<storage, read> x: array<f32>;\n\
            @group(0) @binding(2) var<storage, read_write> y: array<f32>;\n\
            @compute @workgroup_size(64)\n\
            fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n\
                    @builtin(num_workgroups) nwg: vec3<u32>) {\n\
                let i = gid.y * (nwg.x * 64u) + gid.x;\n\
                if (i >= p.n) { return; }\n\
                y[i] = x[i] * f32(SCALE);\n\
            }";
        let (name, spec) = kernels::template::interned("scale", SRC, &[("SCALE", 5)]).unwrap();
        let gpu = Gpu::new_cpu(&[("scale", SRC), (name, spec)]);
        let x = gpu.storage_init("x", &[1.0, 2.0, 3.0]);
        let y = gpu.storage(3);
        let s0 = gpu.step(0, &[&x, &y], &[3], 3);
        gpu.submit(&[], &[s0]);
        assert_eq!(gpu.read(&y, 3), vec![2.0, 4.0, 6.0], "base kernel: SCALE=2");
        let s1 = gpu.step(1, &[&x, &y], &[3], 3);
        gpu.submit(&[], &[s1]);
        assert_eq!(gpu.read(&y, 3), vec![5.0, 10.0, 15.0], "variant: SCALE=5");
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

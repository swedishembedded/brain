// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! wgpu eager [`Backend`] — the portable GPU backend (Vulkan/Metal/DX12/GL on
//! native, WebGPU on wasm). WGSL kernels are compiled into compute pipelines at
//! init; dispatches are lazily accumulated and flushed into a single compute pass
//! per forward (see [`WgpuBackend::submit`]).
//!
//! The inherent methods operate on native `wgpu::Buffer`/[`WgpuStep`]; the thin
//! `impl Backend` downcasts the neutral [`DeviceBuffer`]/[`Step`] handles and
//! delegates to them, so `brain-gpu-core` can treat this as a `dyn Backend`.

use backend_api::{Backend, BufUsage, DeviceBuffer, Step};
use wgpu::util::DeviceExt;

/// A recorded dispatch: (pipeline index, bind group, grid_x, grid_y). The grid is
/// 1D (grid_y = 1) until the workgroup count exceeds the per-dimension limit, then
/// it tiles into Y; shaders reconstruct the linear thread index from
/// `num_workgroups`, so the split is transparent.
pub type WgpuStep = (usize, wgpu::BindGroup, u32, u32);

/// What adapter this process actually got. Captured the first time a backend is
/// built, so callers can *record* it rather than only see it on stderr.
///
/// `software` is the load-bearing field: a box with no real GPU still serves
/// `--device gpu` through a software rasteriser (llvmpipe/lavapipe/SwiftShader).
/// A benchmark that reports such a run as a "GPU number" is worse than useless,
/// so `brain perf` embeds this in every result artifact.
#[derive(Clone, Debug)]
pub struct AdapterDesc {
    /// `"<name> (<device_type>, <backend>)"` — the same text `log_adapter` prints.
    pub description: String,
    pub name: String,
    pub device_type: String,
    pub backend: String,
    pub software: bool,
}

static ADAPTER: std::sync::OnceLock<AdapterDesc> = std::sync::OnceLock::new();

/// The instance options every backend is built from.
///
/// wgpu's default `InstanceFlags` are `from_build_config()`, which turns on
/// `DEBUG | VALIDATION` whenever `debug_assertions` is set — so a plain
/// `cargo test` / `make build` silently enables the Vulkan validation layers and
/// `VK_EXT_debug_utils` object naming, while `make release` does not.
///
/// That divergence is not harmless: on a software Vulkan ICD (lavapipe, the only
/// Vulkan device on a box with no real GPU) `vkSetDebugUtilsObjectNameEXT`
/// faults inside the loader, so every debug-profile test that writes a buffer
/// from more than one thread dies with SIGSEGV in `libvulkan.so.1`. Validation
/// also costs real time in exactly the builds used for iterating.
///
/// So: opt **in**, never by default. `BRAIN_GPU_VALIDATION=1` restores the
/// layers when you actually want them, and debug and release now build the same
/// instance.
fn instance_descriptor() -> wgpu::InstanceDescriptor {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    // PRIMARY (Vulkan/Metal/DX12/WebGPU) — never the GL fallback. brain reaches
    // old GPUs through Vulkan and browsers through WebGPU, so GL adds nothing;
    // enumerating it initialises EGL, and the EGL driver cores are the crash
    // site of two distinct suite failures: Mesa's loader faulted under
    // concurrent debug-build init (lavapipe, SIGSEGV in libvulkan), and
    // NVIDIA's eglcore worker thread ("[vkps] Update") segfaulted
    // intermittently while tests ran concurrently. BRAIN_GPU_GL=1 restores the
    // old behaviour if a GL-only machine ever actually needs it.
    desc.backends = if std::env::var("BRAIN_GPU_GL").as_deref() == Ok("1") {
        wgpu::Backends::all()
    } else {
        wgpu::Backends::PRIMARY
    };
    desc.flags = match std::env::var("BRAIN_GPU_VALIDATION") {
        Ok(v) if v != "0" => wgpu::InstanceFlags::debugging(),
        _ => wgpu::InstanceFlags::empty(),
    };
    desc
}

/// Serialises backend construction across threads.
///
/// Building a backend creates a `wgpu::Instance`, which enumerates **every**
/// graphics backend — including GL via EGL. Mesa's EGL/GL loader is not safe to
/// enter concurrently from several threads in one process: doing so faults
/// inside the driver (seen as `MESA: error: ZINK: failed to choose pdev`
/// followed by SIGSEGV) whenever more than one test thread builds a device at
/// the same time. Device creation happens once per engine and never on a hot
/// path, so serialising it costs nothing measurable and makes multi-threaded
/// construction safe — which the test suite and any multi-model host both do.
#[cfg(not(target_arch = "wasm32"))]
fn init_lock() -> std::sync::MutexGuard<'static, ()> {
    static INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A poisoned lock only means some other thread panicked while building a
    // device; the data is `()`, so recovering is always correct.
    INIT.lock().unwrap_or_else(|e| e.into_inner())
}

/// The adapter this process selected, or `None` if no wgpu backend was built
/// (e.g. a pure `--device cpu` run).
pub fn adapter_desc() -> Option<AdapterDesc> {
    ADAPTER.get().cloned()
}

/// How many **distinct physical discrete GPUs** this machine exposes.
///
/// Counted the same way the multi-GPU paths select cards: discrete adapters,
/// narrowed to one graphics backend (Vulkan when present) so a single card
/// enumerated once per backend is not counted twice.
///
/// Multi-GPU tests use this to *skip* rather than fault: without it they assume
/// cards 0 and 1 exist and die inside the driver on any single-GPU or GPU-less
/// box, which reads as a real failure and hides actual regressions.
#[cfg(not(target_arch = "wasm32"))]
pub fn discrete_gpu_count() -> usize {
    let _guard = init_lock();
    pollster::block_on(async {
        let instance = wgpu::Instance::new(instance_descriptor());
        let mut adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY).await;
        adapters.retain(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
        if adapters.iter().any(|a| a.get_info().backend == wgpu::Backend::Vulkan) {
            adapters.retain(|a| a.get_info().backend == wgpu::Backend::Vulkan);
        }
        adapters.len()
    })
}

fn record_adapter(info: &wgpu::AdapterInfo) {
    let _ = ADAPTER.set(AdapterDesc {
        description: format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend),
        name: info.name.clone(),
        device_type: format!("{:?}", info.device_type),
        backend: format!("{:?}", info.backend),
        // Cpu = a software rasteriser. Unknown adapters are not assumed real.
        software: matches!(info.device_type, wgpu::DeviceType::Cpu),
    });
}

/// Log the selected adapter. Native prints to stderr; wasm has no stderr, so it
/// goes to the browser console.
#[cfg(not(target_arch = "wasm32"))]
fn log_adapter(info: &wgpu::AdapterInfo) {
    record_adapter(info);
    // Several engine instances may be built in one process (the TTS pipeline makes
    // one per component); log the adapter line only once.
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| eprintln!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend));
}
#[cfg(target_arch = "wasm32")]
fn log_adapter(info: &wgpu::AdapterInfo) {
    record_adapter(info);
    web_sys::console::log_1(
        &format!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend).into(),
    );
}

/// The **expensive, shareable** half of a wgpu backend: the instance/adapter/
/// device/queue and every compiled pipeline.
///
/// Split out from [`WgpuBackend`] because building it is what costs ~2-3 s (device
/// init plus one shader compile per kernel), and because creating many of them on
/// one physical GPU is actively harmful: several concurrent Vulkan devices on a
/// single card deadlocked the test suite roughly half the time (all threads in
/// futex wait) and made every model activation pay a full device init. One
/// process now builds this once per distinct kernel set and shares it.
struct DeviceShared {
    // ManuallyDrop so teardown can control ORDER and LOCKING: everything —
    // pipelines, queue, device — is destroyed inside `drop` under the same lock
    // that guards creation. Dropping a device while its driver worker thread
    // ("[vkps] Update") is still optimising pipelines, or while another thread
    // creates on a sibling device, segfaults this NVIDIA driver.
    device: std::mem::ManuallyDrop<wgpu::Device>,
    queue: std::mem::ManuallyDrop<wgpu::Queue>,
    /// Serialises submission and readback across every handle on this device.
    ///
    /// One device, one lane — the same policy the residency executor applies at
    /// the scheduling layer. Mechanically: this NVIDIA driver's worker threads
    /// ("[vkps] Update") segfault intermittently under concurrent
    /// submit + poll(wait) from many host threads on one device, and many
    /// devices per process instead deadlocks it. Serialising the short
    /// encode/submit/poll critical sections is cheap (production serving drives
    /// the GPU from one scheduler thread anyway) and makes multi-threaded use —
    /// the test suite, multi-model hosts — safe on this driver.
    io: std::sync::Mutex<()>,
    pipelines: Vec<wgpu::ComputePipeline>,
    /// Each kernel's declared `@workgroup_size` (parallel to `pipelines`). Almost
    /// every kernel is the engine's default 64; the register-tiled GEMMs use 256.
    /// The dispatch grid must be laid out with the kernel's OWN size, because the
    /// kernel reconstructs its flat invocation id from it.
    wgsizes: Vec<u32>,
    /// `BRAIN_PROFILE` per-kernel GPU timing (native only, and only when the
    /// adapter has TIMESTAMP_QUERY): each dispatch runs in its own compute pass
    /// with begin/end timestamps, resolved and accumulated per kernel name.
    /// Pure observability — the non-profiling flush path is untouched.
    #[cfg(not(target_arch = "wasm32"))]
    gpu_profile: Option<GpuProfile>,
}

impl DeviceShared {
    /// Compile `kernels` into pipelines on an existing device+queue. Used both
    /// when a device is first created and by [`WgpuBackend::new_like`], which
    /// compiles a *different* kernel set onto the **same** device —
    /// `wgpu::Device` is a cheap Arc-backed handle, so cloning it refers to the
    /// same underlying device, never a second one on the card.
    fn compile(
        device: wgpu::Device,
        queue: wgpu::Queue,
        kernels: &[(&str, &str)],
        want_ts: bool,
    ) -> DeviceShared {
        // BRAIN_GPU_CHECKED=1 restores wgpu's runtime bounds checks for kernel
        // debugging; the default trusts the kernels — no clamp instruction on
        // every buffer load/store — matching the CPU backend, whose Cranelift
        // JIT has always run with `MemFlags::trusted()`. The safety argument is
        // the same on both backends: every kernel self-bounds on its uniform
        // (`if (idx >= total) return`) and buffer sizes are fixed by the model.
        let checked = std::env::var("BRAIN_GPU_CHECKED").map(|v| v != "0").unwrap_or(false);
        let runtime_checks = if checked {
            wgpu::ShaderRuntimeChecks::checked()
        } else {
            wgpu::ShaderRuntimeChecks::unchecked()
        };
        let pipelines = kernels
            .iter()
            .map(|(name, src)| {
                // SAFETY: kernels self-bound on their uniform and contain no
                // unbounded loops (every loop is counted by a uniform field);
                // buffer sizes are fixed by the model — the identical contract
                // the CPU JIT has always relied on with `MemFlags::trusted()`.
                let module = unsafe {
                    device.create_shader_module_trusted(
                        wgpu::ShaderModuleDescriptor {
                            label: Some(name),
                            source: wgpu::ShaderSource::Wgsl((*src).into()),
                        },
                        runtime_checks,
                    )
                };
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            })
            .collect();
        #[cfg(not(target_arch = "wasm32"))]
        let gpu_profile = if want_ts {
            Some(GpuProfile {
                names: kernels.iter().map(|(n, _)| n.to_string()).collect(),
                period_ns: queue.get_timestamp_period(),
                acc: std::sync::Mutex::new(vec![(0.0, 0); kernels.len()]),
            })
        } else {
            None
        };
        #[cfg(target_arch = "wasm32")]
        let _ = want_ts;
        DeviceShared {
            device: std::mem::ManuallyDrop::new(device),
            queue: std::mem::ManuallyDrop::new(queue),
            io: std::sync::Mutex::new(()),
            pipelines,
            wgsizes: backend_api::workgroup_sizes(kernels),
            #[cfg(not(target_arch = "wasm32"))]
            gpu_profile,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for DeviceShared {
    fn drop(&mut self) {
        // Tear down under the same lock that guards creation, in a safe order:
        // drain the queue, drop the pipelines, then the queue, then the device.
        // Without the drain + serialisation, destroying a device races the
        // driver's background pipeline-optimiser thread and other threads'
        // device-level calls — observed as intermittent SIGSEGV in
        // "[vkps] Update" while the test suite ran concurrently.
        let _guard = init_lock();
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        self.pipelines.clear();
        // SAFETY: drop() runs exactly once; the fields are never used again.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.queue);
            std::mem::ManuallyDrop::drop(&mut self.device);
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for DeviceShared {
    fn drop(&mut self) {
        // SAFETY: drop() runs exactly once; no locking needed on wasm (single
        // threaded) and no driver worker to race.
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.queue);
            std::mem::ManuallyDrop::drop(&mut self.device);
        }
    }
}

/// The wgpu compute device: a handle onto a shared [`DeviceShared`] plus **its
/// own** command stream.
///
/// The command stream stays per handle on purpose. `submit` batches dispatches
/// and `flush` records the whole batch as one compute pass, so two handles that
/// shared a pending list would interleave each other's batches. Sharing the
/// device while keeping the stream private gives one Vulkan device per process
/// *and* unchanged batching semantics for every existing caller.
pub struct WgpuBackend {
    shared: std::sync::Arc<DeviceShared>,
    /// Lazily-accumulated dispatches: `submit` appends its steps here instead of
    /// encoding+submitting immediately, and `flush` records the WHOLE batch into a
    /// single compute pass + one `queue.submit` (on the next read/write/poll). So
    /// a forward's ~130 block dispatches become ONE submission and ONE compute
    /// pass — instead of ~one queue.submit and ~one pass *per block*, each of
    /// which is a GPU pipeline barrier that serialises an integrated GPU.
    /// `Mutex` keeps `WgpuBackend: Sync`; it is only ever locked single-threaded.
    pending: std::sync::Mutex<Vec<WgpuStep>>,
    /// `BRAIN_PROFILE` op counters: (uniform buffers, bind groups, submits,
    /// dispatches, readbacks) — surfaces per-frame GPU resource churn / sync.
    stats: Option<std::sync::atomic::AtomicU64>,
    stats_bg: std::sync::atomic::AtomicU64,
    stats_submit: std::sync::atomic::AtomicU64,
    stats_dispatch: std::sync::atomic::AtomicU64,
    stats_read: std::sync::atomic::AtomicU64,
}

impl WgpuBackend {
    // Field shims so the rest of this file (and its tests) keep reading
    // `self.device()` / `self.queue()` / … unchanged.
    #[inline]
    fn device(&self) -> &wgpu::Device { &self.shared.device }
    #[inline]
    fn queue(&self) -> &wgpu::Queue { &self.shared.queue }
    #[inline]
    fn pipelines(&self) -> &[wgpu::ComputePipeline] { &self.shared.pipelines }
    #[inline]
    fn wgsizes(&self) -> &[u32] { &self.shared.wgsizes }
}

/// Per-kernel GPU-time accumulator for the `BRAIN_PROFILE` timestamp path.
#[cfg(not(target_arch = "wasm32"))]
struct GpuProfile {
    names: Vec<String>,
    /// Nanoseconds per timestamp tick (`Queue::get_timestamp_period`).
    period_ns: f32,
    /// Per-pipeline (total_ms, calls).
    acc: std::sync::Mutex<Vec<(f64, u64)>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for WgpuBackend {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        if let Some(uni) = &self.stats {
            eprintln!(
                "=== GPU op counts (BRAIN_PROFILE) === uniforms={} bind_groups={} submits={} dispatches={} readbacks={}",
                uni.load(Relaxed),
                self.stats_bg.load(Relaxed),
                self.stats_submit.load(Relaxed),
                self.stats_dispatch.load(Relaxed),
                self.stats_read.load(Relaxed),
            );
        }
        if let Some(p) = &self.shared.gpu_profile {
            let acc = p.acc.lock().unwrap();
            let mut rows: Vec<(usize, f64, u64)> =
                acc.iter().enumerate().filter(|(_, (_, c))| *c > 0).map(|(i, (ms, c))| (i, *ms, *c)).collect();
            rows.sort_by(|a, b| b.1.total_cmp(&a.1));
            let total: f64 = rows.iter().map(|r| r.1).sum();
            eprintln!("=== GPU kernel time (BRAIN_PROFILE, timestamp queries, total {total:.1} ms) ===");
            for (i, ms, c) in rows {
                eprintln!(
                    "  {:<20} {:8.1} ms  {:5} calls  ({:4.1}%)",
                    p.names[i],
                    ms,
                    c,
                    ms / total.max(1e-9) * 100.0
                );
            }
        }
    }
}

impl WgpuBackend {
    /// Initialise the device and compile `kernels` (name, WGSL source) into
    /// pipelines indexed by their position in the slice.
    ///
    /// Native blocking entry: wraps the async core in `pollster::block_on` so
    /// the existing synchronous call sites (CLI, training, tests) are unchanged.
    /// On wasm the browser has no blocking executor, so use `new_async` instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(kernels: &[(&str, &str)]) -> WgpuBackend {
        let _guard = init_lock();
        pollster::block_on(WgpuBackend::new_async(kernels))
    }

    /// A second handle onto **this** backend's device: same instance, adapter,
    /// queue and compiled pipelines, but its own command stream.
    ///
    /// Explicit on purpose. Building a device costs seconds and several
    /// concurrent devices on one card are hostile to the driver, so callers that
    /// need more than one handle — a serving process running several models, a
    /// test binary — should create the device once and share it. Making that a
    /// hidden process-global cache would be less code at the call site but would
    /// tie every `Gpu` in the process together invisibly, and make "how many
    /// devices exist" unanswerable from the code.
    ///
    /// The command stream stays per handle: `submit` batches dispatches and
    /// `flush` records the batch as one compute pass, so two handles sharing a
    /// pending list would interleave each other's batches.
    /// A backend for a **different kernel set** on **this same device**: clones
    /// the (Arc-backed) `wgpu::Device`/`Queue` handles and compiles the new
    /// pipelines onto them. This is what lets one process hold every model's
    /// pipelines on a single device instead of one device per model —
    /// many concurrent devices on one card is what deadlocked the suite.
    pub fn new_like_device(&self, kernels: &[(&str, &str)]) -> WgpuBackend {
        // Serialised like `new`: pipeline creation is a device-level operation,
        // and the NVIDIA driver's worker threads segfault under many concurrent
        // pipeline-set compilations/teardowns on one device.
        let _guard = init_lock();
        let profile_on = std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false);
        #[cfg(not(target_arch = "wasm32"))]
        let want_ts = profile_on && self.shared.gpu_profile.is_some();
        #[cfg(target_arch = "wasm32")]
        let want_ts = false;
        let shared = std::sync::Arc::new(DeviceShared::compile(
            (*self.shared.device).clone(),
            (*self.shared.queue).clone(),
            kernels,
            want_ts,
        ));
        let stats = profile_on.then(|| std::sync::atomic::AtomicU64::new(0));
        WgpuBackend::from_shared(shared, stats)
    }

    pub fn share_device(&self) -> WgpuBackend {
        let stats =
            self.stats.as_ref().map(|_| std::sync::atomic::AtomicU64::new(0));
        WgpuBackend::from_shared(self.shared.clone(), stats)
    }

    /// Blocking `count` backends on distinct physical cards (see [`new_multi_async`]).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_multi(kernels: &[(&str, &str)], count: usize) -> Vec<WgpuBackend> {
        let _guard = init_lock();
        pollster::block_on(WgpuBackend::new_multi_async(kernels, count))
    }

    /// Async device init + pipeline compile. This is the portable core used on
    /// both targets: native wraps it in `pollster::block_on` (see `new`), wasm
    /// awaits it from the wasm-bindgen entry point.
    pub async fn new_async(kernels: &[(&str, &str)]) -> WgpuBackend {
        let instance = wgpu::Instance::new(instance_descriptor());
        // `BRAIN_GPU_INDEX=<i>` pins a specific physical GPU (the box has two
        // Tesla P40s); without it the high-performance default is used. This is
        // what lets one process drive card 0 and another card 1, and is the hook
        // multi-GPU data-parallel / sharding builds on.
        let forced = std::env::var("BRAIN_GPU_INDEX").ok().and_then(|v| v.parse::<usize>().ok());
        #[cfg(not(target_arch = "wasm32"))]
        let adapter = match forced {
            Some(i) => {
                let mut adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY).await;
                // Prefer discrete GPUs, stable order, then index into them.
                adapters.retain(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
                // `enumerate_adapters(all())` lists each physical card ONCE PER
                // graphics backend (Vulkan, GL, …), so indexing across the mixed
                // list can select the same card via a different backend
                // (`BRAIN_GPU_INDEX=1` landing back on card 0). Narrow to a single
                // backend (prefer Vulkan on this Pascal/Linux box) so the index
                // maps to distinct physical cards — the invariant multi-GPU
                // sharding / data-parallel relies on.
                if adapters.iter().any(|a| a.get_info().backend == wgpu::Backend::Vulkan) {
                    adapters.retain(|a| a.get_info().backend == wgpu::Backend::Vulkan);
                }
                if adapters.is_empty() {
                    panic!("BRAIN_GPU_INDEX set but no discrete GPU adapters enumerated");
                }
                adapters.into_iter().nth(i).unwrap_or_else(|| panic!("BRAIN_GPU_INDEX={i} out of range"))
            }
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("no suitable GPU adapter found"),
        };
        #[cfg(target_arch = "wasm32")]
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");
        Self::from_adapter(&adapter, kernels).await
    }

    /// Create `count` backends on DISTINCT physical cards from ONE adapter
    /// enumeration (Vulkan discrete GPUs, in enumerated order). This is the
    /// reliable multi-GPU path: two separate `new_async` calls each re-enumerate
    /// and wgpu can reorder the list, landing a second in-process device back on
    /// card 0 (observed on the 2×P40 box). A single enumeration guarantees
    /// `adapters[i]` → distinct physical card.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn new_multi_async(kernels: &[(&str, &str)], count: usize) -> Vec<WgpuBackend> {
        let instance = wgpu::Instance::new(instance_descriptor());
        let mut adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY).await;
        adapters.retain(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
        if adapters.iter().any(|a| a.get_info().backend == wgpu::Backend::Vulkan) {
            adapters.retain(|a| a.get_info().backend == wgpu::Backend::Vulkan);
        }
        assert!(adapters.len() >= count, "need {count} discrete GPUs, found {}", adapters.len());
        let mut out = Vec::with_capacity(count);
        for a in adapters.iter().take(count) {
            out.push(Self::from_adapter(a, kernels).await);
        }
        out
    }

    /// Build a backend from an already-selected adapter (shared by [`new_async`]
    /// and [`new_multi_async`]).
    async fn from_adapter(adapter: &wgpu::Adapter, kernels: &[(&str, &str)]) -> WgpuBackend {
        let info = adapter.get_info();
        log_adapter(&info);

        // downlevel defaults => broad compatibility (incl. sm_61 / Pascal); bump
        // the storage-buffer count to 8 (LayerNorm-dgamma and attention use up to 5).
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 8;
        // Raise the buffer/binding SIZE caps to whatever this adapter actually
        // supports. The downlevel defaults cap at 256MB buffer / 128MB binding,
        // which on a big-VRAM card (e.g. 24GB P40) needlessly rejects large
        // batches even with plenty of free memory. Requesting the adapter's own
        // reported maxima is always valid, and on WebGPU it stays whatever the
        // browser allows.
        let adapter_limits = adapter.limits();
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        // The weight-tiled conv stages an output channel's weights in workgroup
        // memory (up to 32 KiB); the downlevel default caps that at 16 KiB, so
        // request whatever the adapter actually supports.
        limits.max_compute_workgroup_storage_size = adapter_limits.max_compute_workgroup_storage_size;
        // BRAIN_PROFILE: per-kernel GPU timing wants timestamp queries. Opt-in
        // and adapter-gated, so the default request stays feature-empty (the
        // portability invariant) and profiling degrades to op counts where the
        // feature is absent (WebGPU, old drivers).
        let profile_on = std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false);
        let want_ts = profile_on && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features =
            if want_ts { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("moe-rs-device"),
                required_features,
                required_limits: limits,
                // MemoryUsage (tight suballocation) not Performance (large blocks):
                // a large resident model has hundreds of medium weight buffers, and
                // Performance's big blocks waste ~2× (12 GB → 23 GB, OOMs a P40).
                // BRAIN_GPU_MEM_PERF=1 restores the perf-first blocks.
                memory_hints: if std::env::var("BRAIN_GPU_MEM_PERF").as_deref() == Ok("1") {
                    wgpu::MemoryHints::Performance
                } else {
                    wgpu::MemoryHints::MemoryUsage
                },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("request_device failed");

        #[cfg(not(target_arch = "wasm32"))]
        {
            let l = device.limits();
            let mib = |b: u64| b / (1024 * 1024);
            eprintln!(
                "limits: max_buffer_size {} MiB, max_storage_buffer_binding_size {} MiB \
                 (adapter caps: {} / {} MiB)",
                mib(l.max_buffer_size),
                mib(l.max_storage_buffer_binding_size as u64),
                mib(adapter_limits.max_buffer_size),
                mib(adapter_limits.max_storage_buffer_binding_size as u64),
            );
        }

        // `BRAIN_GPU_CHECKED=1` restores wgpu's injected per-access bounds
        // clamps (the debugging default). Otherwise shaders compile TRUSTED —
        // no clamp instruction on every buffer load/store — matching the CPU
        // backend, whose Cranelift JIT has always run with
        // `MemFlags::trusted()` and no bounds checks anywhere. The safety
        // argument is the same on both backends: every kernel self-bounds on
        // its uniform (`if (idx >= total) return`), and buffer sizes are fixed
        // by the model at build time. The conv inner loops do ~1 load per
        // 2-3 FMAs, so the removed clamps are directly measurable.
        use std::sync::atomic::AtomicU64;
        let stats = if profile_on { Some(AtomicU64::new(0)) } else { None };
        let shared = std::sync::Arc::new(DeviceShared::compile(device, queue, kernels, want_ts));
        WgpuBackend::from_shared(shared, stats)
    }

    /// A fresh handle (own command stream) onto an already-built device.
    fn from_shared(
        shared: std::sync::Arc<DeviceShared>,
        stats: Option<std::sync::atomic::AtomicU64>,
    ) -> WgpuBackend {
        use std::sync::atomic::AtomicU64;
        WgpuBackend {
            shared,
            pending: std::sync::Mutex::new(Vec::new()),
            stats,
            stats_bg: AtomicU64::new(0),
            stats_submit: AtomicU64::new(0),
            stats_dispatch: AtomicU64::new(0),
            stats_read: AtomicU64::new(0),
        }
    }

    fn max_storage_binding_bytes(&self) -> u64 {
        self.device().limits().max_storage_buffer_binding_size as u64
    }

    /// Record all pending dispatches into ONE compute pass and submit. Idempotent.
    /// wgpu inserts the necessary inter-dispatch barriers within the pass, so the
    /// per-block read-after-write dependencies are preserved.
    fn flush(&self) {
        let _io = self.shared.io.lock().unwrap_or_else(|e| e.into_inner());
        self.flush_inner();
    }

    /// [`Self::flush`] body, with the device's io lock already held.
    fn flush_inner(&self) {
        let steps: Vec<WgpuStep> = std::mem::take(&mut *self.pending.lock().unwrap());
        if steps.is_empty() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.shared.gpu_profile.is_some() {
            self.flush_profiled(&steps);
            if self.stats.is_some() {
                self.stats_submit.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            return;
        }
        let mut enc = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for (kind, bg, gx, gy) in &steps {
                pass.set_pipeline(&self.pipelines()[*kind]);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*gx, *gy, 1);
            }
        }
        self.queue().submit(Some(enc.finish()));
        if self.stats.is_some() {
            self.stats_submit.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The `BRAIN_PROFILE` timestamp flush: one compute pass PER dispatch, each
    /// bracketed by begin/end timestamps, resolved and read back synchronously,
    /// accumulated per kernel. Slower than the production single-pass flush (the
    /// readback stalls the queue once per flush) but numerically identical —
    /// same pipelines, same bind groups, same order; wgpu barriers between
    /// passes preserve the read-after-write dependencies exactly like within
    /// one pass.
    #[cfg(not(target_arch = "wasm32"))]
    fn flush_profiled(&self, steps: &[WgpuStep]) {
        let p = self.shared.gpu_profile.as_ref().unwrap();
        // Chunked to stay comfortably under the per-query-set limit (8192).
        for chunk in steps.chunks(2048) {
            let n = chunk.len() as u32;
            let qs = self.device().create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("brain-profile"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * n,
            });
            let bytes = (2 * n * 8) as u64;
            let resolve = self.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("brain-profile-resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = self.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("brain-profile-staging"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = self.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for (i, (kind, bg, gx, gy)) in chunk.iter().enumerate() {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &qs,
                        beginning_of_pass_write_index: Some(2 * i as u32),
                        end_of_pass_write_index: Some(2 * i as u32 + 1),
                    }),
                });
                pass.set_pipeline(&self.pipelines()[*kind]);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*gx, *gy, 1);
            }
            enc.resolve_query_set(&qs, 0..2 * n, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
            self.queue().submit(Some(enc.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            self.device().poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let ticks: Vec<u64> = bytemuck::cast_slice::<u8, u64>(&slice.get_mapped_range()).to_vec();
            staging.unmap();

            let mut acc = p.acc.lock().unwrap();
            for (i, (kind, ..)) in chunk.iter().enumerate() {
                let dt = ticks[2 * i + 1].saturating_sub(ticks[2 * i]);
                let e = &mut acc[*kind];
                e.0 += dt as f64 * p.period_ns as f64 / 1e6;
                e.1 += 1;
            }
        }
    }

    pub fn storage(&self, n: u64) -> wgpu::Buffer {
        self.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn storage_init(&self, name: &str, data: &[f32]) -> wgpu::Buffer {
        self.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    pub fn buffer(&self, label: &str, size: u64, usage: BufUsage) -> wgpu::Buffer {
        let mut u = wgpu::BufferUsages::empty();
        if usage.contains(BufUsage::STORAGE) {
            u |= wgpu::BufferUsages::STORAGE;
        }
        if usage.contains(BufUsage::COPY_DST) {
            u |= wgpu::BufferUsages::COPY_DST;
        }
        if usage.contains(BufUsage::COPY_SRC) {
            u |= wgpu::BufferUsages::COPY_SRC;
        }
        if usage.contains(BufUsage::UNIFORM) {
            u |= wgpu::BufferUsages::UNIFORM;
        }
        self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: u,
            mapped_at_creation: false,
        })
    }

    fn uniform(&self, data: &[u32]) -> wgpu::Buffer {
        if let Some(s) = &self.stats {
            s.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let mut bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        while bytes.len() % 16 != 0 {
            bytes.push(0);
        }
        self.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// A writable uniform buffer sized for `len` u32s (16-byte aligned), to be
    /// updated later via `write`. Lets a caller build a dispatch once and reuse
    /// its bind group across many submits, changing only the uniform contents —
    /// avoiding the per-dispatch buffer/bind-group churn that otherwise exhausts
    /// the GPU memory aperture in long training loops.
    pub fn uniform_dynamic(&self, len: usize) -> wgpu::Buffer {
        let size = (((len * 4) + 15) / 16 * 16).max(16) as u64;
        self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Build one dispatch around an already-allocated uniform buffer: bind group
    /// 0 = [ubuf, buffers...], grid = ceil(threads/64). The bind group keeps
    /// `ubuf` and `bufs` alive for the lifetime of the returned `WgpuStep`.
    pub fn step_buf(&self, kind: usize, ubuf: &wgpu::Buffer, bufs: &[&wgpu::Buffer], threads: u32) -> WgpuStep {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: ubuf.as_entire_binding(),
        }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }
        let layout = self.pipelines()[kind].get_bind_group_layout(0);
        let bg = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        if self.stats.is_some() {
            self.stats_bg.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let (gx, gy) = backend_api::grid_ws(threads, self.wgsizes()[kind]);
        (kind, bg, gx, gy)
    }

    /// Build one dispatch with a fresh single-use uniform buffer. Convenient for
    /// one-shot work; for hot loops prefer `uniform_dynamic` + `step_buf` so the
    /// uniform/bind group are allocated once and reused.
    pub fn step(&self, kind: usize, bufs: &[&wgpu::Buffer], params: &[u32], threads: u32) -> WgpuStep {
        self.step_buf(kind, &self.uniform(params), bufs, threads)
    }

    /// Like [`step`](Self::step) but each storage buffer binds a sub-range
    /// `(word_offset, word_len)` (`word_len == 0` => to the end). This keeps a
    /// single binding within `max_storage_buffer_binding_size` (e.g. tiling a
    /// >128MB embedding into vocab slices). Offsets must satisfy the adapter's
    /// `min_storage_buffer_offset_alignment` (256B); row-aligned tiles do.
    pub fn step_sliced(&self, kind: usize, bufs: &[&wgpu::Buffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> WgpuStep {
        let ubuf = self.uniform(params);
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }];
        for (i, b) in bufs.iter().enumerate() {
            let (off_w, len_w) = offsets[i];
            let resource = if off_w == 0 && len_w == 0 {
                b.as_entire_binding()
            } else {
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: b,
                    offset: off_w * 4,
                    size: if len_w == 0 { None } else { std::num::NonZeroU64::new(len_w * 4) },
                })
            };
            entries.push(wgpu::BindGroupEntry { binding: (i + 1) as u32, resource });
        }
        let layout = self.pipelines()[kind].get_bind_group_layout(0);
        let bg = self.device().create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
        if self.stats.is_some() {
            self.stats_bg.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let (gx, gy) = backend_api::grid_ws(threads, self.wgsizes()[kind]);
        (kind, bg, gx, gy)
    }

    /// Clear the given buffers, then run all steps as a compute pass. The work is
    /// RECORDED into a lazily-accumulated command encoder (one per forward) and
    /// only sent to the GPU on the next read/write/poll — so a whole forward's
    /// dispatches coalesce into a single `queue.submit` instead of one per call.
    /// wgpu inserts the inter-dispatch barriers within and across the passes.
    pub fn submit(&self, clears: &[&wgpu::Buffer], steps: &[WgpuStep]) {
        // Clears can't go inside a compute pass: flush the accumulated dispatches
        // (so they run before the clear) then clear in a transfer submission. In
        // the eval/inference path there are no clears, so everything accumulates
        // into a single pass flushed at the terminal readback.
        if !clears.is_empty() {
            self.flush();
            let mut enc = self
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            for c in clears {
                enc.clear_buffer(c, 0, None);
            }
            self.queue().submit(Some(enc.finish()));
        }
        self.pending.lock().unwrap().extend(steps.iter().cloned());
        if self.stats.is_some() {
            self.stats_dispatch
                .fetch_add(steps.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn write(&self, buf: &wgpu::Buffer, data: &[u32]) {
        // Flush pending compute first so a host write never races ahead of
        // dispatches recorded before it (queue order: prior compute, then write).
        // One io hold across both, so another handle's submit cannot interleave.
        let _io = self.shared.io.lock().unwrap_or_else(|e| e.into_inner());
        self.flush_inner();
        self.queue().write_buffer(buf, 0, bytemuck::cast_slice(data));
    }

    /// Block until all submitted GPU work has completed, letting wgpu reclaim the
    /// transient per-submit resources (command buffers and `write_buffer` staging
    /// memory) that accrue otherwise. A long training loop that only submits —
    /// never reads back — never triggers this reclaim, so those transients pile
    /// up in the GPU memory aperture until an allocation fails mid-run (on
    /// integrated GPUs the aperture is small, ~355 MiB). Call once per step to
    /// bound the in-flight transient memory to a single iteration.
    ///
    /// Native only: the browser drives the device via its event loop, so there is
    /// no blocking poll to call there.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn poll_wait(&self) {
        self.flush();
        self.device().poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    /// Copy a device buffer into a MAP_READ staging buffer and return its
    /// contents as f32. Native only: it blocks on `device.poll(wait)` + an mpsc
    /// recv, which is impossible in a browser. Wasm uses `read_async`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        if self.stats.is_some() {
            self.stats_read.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let _io = self.shared.io.lock().unwrap_or_else(|e| e.into_inner());
        self.flush_inner(); // ensure all recorded compute is queued before the copy
        let staging = self.read_staging(buf, n);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device().poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        out
    }

    /// Async buffer readback for wasm: awaits the map callback rather than
    /// blocking the thread. In the browser the device's work is driven by the
    /// event loop, so no `device.poll(wait)` is needed -- yielding back to the
    /// JS executor lets the GPU complete and fire the callback.
    #[cfg(target_arch = "wasm32")]
    pub async fn read_async(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        self.flush();
        let staging = self.read_staging(buf, n);
        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Schedule the mapping callbacks; on the WebGPU backend this is a no-op
        // beyond servicing the queue, the actual completion arrives via the
        // browser event loop while we await the oneshot.
        let _ = self.device().poll(wgpu::PollType::Poll);
        rx.await.expect("map_async channel dropped").expect("buffer map failed");
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        out
    }

    /// Shared staging-buffer copy used by both `read` (native) and `read_async`
    /// (wasm).
    fn read_staging(&self, buf: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
        let staging = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, (n * 4) as u64);
        self.queue().submit(Some(enc.finish()));
        staging
    }
}

/// Neutral-handle bridge: downcast the opaque [`DeviceBuffer`]/[`Step`] back to
/// `wgpu::Buffer`/[`WgpuStep`] and delegate to the inherent methods. Inherent
/// methods take resolution priority, so `WgpuBackend::method(self, …)` is
/// unambiguous.

/// Weak handle onto a [`DeviceShared`] — see `backend_api::Backend::downgrade`.
/// Holds no strong count, so the device still dies with its last real handle.
struct WeakWgpu(std::sync::Weak<DeviceShared>);

impl backend_api::WeakBackend for WeakWgpu {
    fn upgrade(&self) -> Option<Box<dyn Backend>> {
        let shared = self.0.upgrade()?;
        let profile_on = std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false);
        let stats = profile_on.then(|| std::sync::atomic::AtomicU64::new(0));
        Some(Box::new(WgpuBackend::from_shared(shared, stats)))
    }
}

impl Backend for WgpuBackend {
    fn share(&self) -> Option<Box<dyn Backend>> {
        Some(Box::new(self.share_device()))
    }
    fn new_like(&self, kernels: &[(&str, &str)]) -> Option<Box<dyn Backend>> {
        Some(Box::new(self.new_like_device(kernels)))
    }
    fn downgrade(&self) -> Option<Box<dyn backend_api::WeakBackend>> {
        Some(Box::new(WeakWgpu(std::sync::Arc::downgrade(&self.shared))))
    }
    fn storage(&self, n: u64) -> DeviceBuffer {
        DeviceBuffer::new(WgpuBackend::storage(self, n))
    }
    fn storage_init(&self, name: &str, data: &[f32]) -> DeviceBuffer {
        DeviceBuffer::new(WgpuBackend::storage_init(self, name, data))
    }
    fn buffer(&self, label: &str, size: u64, usage: BufUsage) -> DeviceBuffer {
        DeviceBuffer::new(WgpuBackend::buffer(self, label, size, usage))
    }
    fn uniform_dynamic(&self, len: usize) -> DeviceBuffer {
        DeviceBuffer::new(WgpuBackend::uniform_dynamic(self, len))
    }
    fn write(&self, buf: &DeviceBuffer, data: &[u32]) {
        WgpuBackend::write(self, buf.downcast_ref::<wgpu::Buffer>(), data)
    }
    fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step {
        let bs: Vec<&wgpu::Buffer> = bufs.iter().map(|b| b.downcast_ref::<wgpu::Buffer>()).collect();
        Step::new(WgpuBackend::step(self, kind, &bs, params, threads))
    }
    fn step_sliced(&self, kind: usize, bufs: &[&DeviceBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> Step {
        let bs: Vec<&wgpu::Buffer> = bufs.iter().map(|b| b.downcast_ref::<wgpu::Buffer>()).collect();
        Step::new(WgpuBackend::step_sliced(self, kind, &bs, offsets, params, threads))
    }
    fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step {
        let bs: Vec<&wgpu::Buffer> = bufs.iter().map(|b| b.downcast_ref::<wgpu::Buffer>()).collect();
        Step::new(WgpuBackend::step_buf(self, kind, ubuf.downcast_ref::<wgpu::Buffer>(), &bs, threads))
    }
    fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]) {
        let cs: Vec<&wgpu::Buffer> = clears.iter().map(|b| b.downcast_ref::<wgpu::Buffer>()).collect();
        let ss: Vec<WgpuStep> = steps.iter().map(|s| s.downcast_ref::<WgpuStep>().clone()).collect();
        WgpuBackend::submit(self, &cs, &ss);
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn read(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
        WgpuBackend::read(self, buf.downcast_ref::<wgpu::Buffer>(), n)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn poll_wait(&self) {
        WgpuBackend::poll_wait(self)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn flush(&self) {
        // Submit the accumulated compute pass; no wait — the point is overlap.
        WgpuBackend::flush(self);
    }
}

/// Async read-back keyed by the neutral [`DeviceBuffer`] handle (wasm facade).
/// Keeps `wgpu::Buffer` knowledge inside this crate so `brain-gpu-core` needs no
/// direct wgpu dependency on wasm.
#[cfg(target_arch = "wasm32")]
impl WgpuBackend {
    pub async fn read_async_buf(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
        self.read_async(buf.downcast_ref::<wgpu::Buffer>(), n).await
    }
}

/// Register this backend under `"wgpu"` so the facade can build it by name.
#[cfg(not(target_arch = "wasm32"))]
pub fn register() {
    backend_api::register_backend("wgpu", |kernels| Ok(Box::new(WgpuBackend::new(kernels))));
}

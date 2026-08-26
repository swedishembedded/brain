// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--device` - **which compute is schedulable**.
//!
//! The flag does not pick "a backend"; it declares the set of compute units
//! brain may schedule work onto. Everything else (which `Gpu` a model builds,
//! which residency budgets exist, how many CPU threads run, whether the NPU path
//! is allowed) follows from that set.
//!
//! ```text
//! (absent)        every device present - GPUs + CPU + NPU, scheduled together
//! cpu             CPU only (all cores)
//! gpu             every GPU, and nothing else
//! npu             NPU only
//! gpu,cpu         both, scheduled together
//! gpu0            only physical GPU 0
//! gpu0,gpu1       those two GPUs
//! cpu21           only CPU core 21
//! cpu0-7          CPU cores 0..=7
//! gpu1,cpu0-3     one GPU plus four cores
//! ```
//!
//! Host RAM and disk are always available as *cache/spill* tiers - restricting
//! compute to the GPU does not stop weights from spilling to RAM or being
//! memory-mapped from disk. `--device` bounds **where work executes**, not where
//! bytes may rest.
//!
//! # NPU caveat
//!
//! The NPU is not a `gpu-core` backend: OpenVINO is a *whole-graph* compiler, so
//! reaching it is a separate export → quantise → compile → run path
//! (`crates/npu`) that a model must be explicitly built for. This module
//! therefore *inventories* NPUs and honours `npu` in a spec, but a plain `brain
//! <cmd>` with no `--device` will not silently route work there - only an
//! explicit request does. Transparent NPU scheduling needs the per-model
//! export path first.
//!
//! Parsing ([`DeviceSpec::parse`]) is pure and total; resolution
//! ([`DeviceSpec::resolve`]) probes the machine. They are separate so the
//! grammar is testable without hardware.

use std::fmt;

pub use backend_api::GpuIdentity;

// ---- canonical device registry ---------------------------------------------
//
// THE process-wide enumeration of physical GPUs, performed once, with stable
// identity. Canonical index = position after sorting by PCI bus id (stable
// across boots and shared with NVML/nvidia-smi), so `gpu0`/`gpu1` in --device,
// `Shard.gpu_index`, and `residency::Device::Gpu(i)` all name the same physical
// card - and nvidia-smi order maps to it via PCI bus id, never by assumption.
//
// Identity comes from the ash (native Vulkan) enumeration when an ICD is
// present: PCI bus id (VK_EXT_pci_bus_info) and deviceUUID (Vulkan 1.1 core;
// equals the NVML GPU UUID on NVIDIA). Without an ICD the wgpu enumeration
// fills in, with identity read through the `Adapter::as_hal` escape hatch where
// its backend is Vulkan and the fallback key (vendor:device, ordinal) elsewhere.
// Backends select adapters by matching these identities - never by enumeration
// position across independent enumerations, and never via env mutation.

/// One physical GPU in the canonical registry.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceId {
    /// Canonical index - what `gpu<i>` means everywhere in brain.
    pub index: u32,
    pub identity: GpuIdentity,
}

pub struct DeviceRegistry {
    devices: Vec<DeviceId>,
    /// Which enumeration established identity: `"vulkan"` (ash) or `"wgpu"`.
    source: &'static str,
}

static REGISTRY: std::sync::OnceLock<DeviceRegistry> = std::sync::OnceLock::new();

/// The registry, enumerated on first use.
pub fn registry() -> &'static DeviceRegistry {
    REGISTRY.get_or_init(|| {
        let (ids, source) = match backend_vulkan::enumerate_physical_gpus() {
            Ok(v) if !v.is_empty() => (v, "vulkan"),
            Err(e) => {
                tracing::debug!(error = %e, "native Vulkan enumeration unavailable; falling back to wgpu");
                (backend_wgpu::enumerate_gpus(), "wgpu")
            }
            Ok(_) => (backend_wgpu::enumerate_gpus(), "wgpu"),
        };
        let reg = DeviceRegistry::from_identities(ids, source);
        tracing::info!(source, gpus = reg.devices().len(), "device registry built (one-time, process lifetime)");
        reg
    })
}

impl DeviceRegistry {
    fn from_identities(mut ids: Vec<GpuIdentity>, source: &'static str) -> DeviceRegistry {
        use backend_api::DeviceClass;
        // Physical cards only: a software rasteriser (llvmpipe) is not a card
        // and must never occupy a canonical index.
        ids.retain(|d| d.class != DeviceClass::Cpu);
        // When any discrete GPU exists, indices cover only discrete cards -
        // the set `--device gpu` schedules and `Inventory::probe` counts.
        if ids.iter().any(|d| d.class == DeviceClass::DiscreteGpu) {
            ids.retain(|d| d.class == DeviceClass::DiscreteGpu);
        }
        // Canonical order: PCI bus id when every card reports one (stable
        // across boots and driver updates); otherwise the enumeration order,
        // whose per-(vendor,device) ordinals are at least stable per boot.
        if !ids.is_empty() && ids.iter().all(|d| d.pci_bus.is_some()) {
            ids.sort_by(|a, b| a.pci_bus.cmp(&b.pci_bus));
        }
        DeviceRegistry {
            devices: ids
                .into_iter()
                .enumerate()
                .map(|(i, identity)| DeviceId { index: i as u32, identity })
                .collect(),
            source,
        }
    }

    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }
    pub fn source(&self) -> &'static str {
        self.source
    }
}

/// Every physical GPU, in canonical (PCI-bus) order.
pub fn gpus() -> &'static [DeviceId] {
    registry().devices()
}

/// The card behind canonical index `index`. Out-of-range is an error, never a
/// silent clamp.
pub fn device(index: u32) -> Result<&'static DeviceId, String> {
    let devs = gpus();
    devs.get(index as usize)
        .ok_or_else(|| format!("gpu{index} requested but this machine has {} GPU(s)", devs.len()))
}

/// The registry entry whose PCI bus id matches `pci` (case-insensitive,
/// tolerant of nvidia-smi's `00000000:81:00.0` zero-padding).
pub fn device_by_pci(pci: &str) -> Option<&'static DeviceId> {
    let norm = |s: &str| -> String {
        // "domain:bus:dev.fn" hex fields, minus leading zeros per field.
        s.to_ascii_lowercase()
            .split([':', '.'])
            .map(|f| f.trim_start_matches('0').to_string())
            .collect::<Vec<_>>()
            .join(":")
    };
    let want = norm(pci);
    gpus().iter().find(|d| d.identity.pci_bus.as_deref().map(|p| norm(p) == want).unwrap_or(false))
}

// ---- ambient + scoped selection --------------------------------------------
//
// Placement inputs, strongest first:
//   1. a scoped selection (`with_gpu`) - thread-local, race-free; what the
//      residency executor and the multi-GPU helpers use;
//   2. the pin `ComputeSet::apply` recorded from `--device gpu<i>`;
//   3. `BRAIN_GPU_INDEX` - user input only, parsed ONCE at first use;
//   4. none: canonical device 0 when the registry has cards, else the
//      backend's own default (the software-rasteriser fallback path).

static AMBIENT_PIN: std::sync::Mutex<Option<Option<u32>>> = std::sync::Mutex::new(None);
static ENV_AMBIENT: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();

std::thread_local! {
    static SCOPED: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Record the `--device` pin (`Some(i)` for a single-card selection, `None` to
/// clear an inherited `BRAIN_GPU_INDEX`). Called by [`ComputeSet::apply`].
pub fn set_ambient_gpu(pin: Option<u32>) {
    *AMBIENT_PIN.lock().unwrap_or_else(|e| e.into_inner()) = Some(pin);
}

fn env_ambient() -> Option<u32> {
    *ENV_AMBIENT.get_or_init(|| {
        std::env::var("BRAIN_GPU_INDEX").ok().and_then(|v| v.parse::<u32>().ok())
    })
}

/// The ambient canonical index (pin > env), before any scoped override.
pub fn ambient_gpu() -> Option<u32> {
    match *AMBIENT_PIN.lock().unwrap_or_else(|e| e.into_inner()) {
        Some(pin) => pin,
        None => env_ambient(),
    }
}

/// Run `f` with GPU `index` as this thread's device selection: every
/// `Gpu::new` under it lands on that card. Thread-local, so concurrent scopes
/// on other threads (residency lanes) cannot race. Errors on an out-of-range
/// index when the machine has cards; on a GPU-less/CPU-backend run the scope is
/// recorded but placement is moot.
pub fn with_gpu<R>(index: u32, f: impl FnOnce() -> R) -> Result<R, String> {
    if !gpus().is_empty() {
        device(index)?; // validate, never clamp
    }
    SCOPED.with(|s| s.borrow_mut().push(index));
    struct Pop;
    impl Drop for Pop {
        fn drop(&mut self) {
            SCOPED.with(|s| {
                s.borrow_mut().pop();
            });
        }
    }
    let _pop = Pop;
    Ok(f())
}

/// The selection a `Gpu::new` on this thread resolves to (scoped > pin > env).
pub fn current_gpu() -> Option<u32> {
    SCOPED.with(|s| s.borrow().last().copied()).or_else(ambient_gpu)
}

/// The registry entry a `Gpu::new` on this thread builds on, or `None` when no
/// physical card exists (backend default / software fallback applies). Panics
/// on an out-of-range explicit selection - never a silent clamp.
pub fn selected_device() -> Option<&'static DeviceId> {
    let devs = gpus();
    match current_gpu() {
        Some(i) => {
            if devs.is_empty() {
                // Preserve the old forced-selection strictness on GPU-less
                // boxes: an explicit index cannot be honoured there. The CPU
                // backend never consults this, so pure-CPU runs are unaffected.
                return None;
            }
            Some(devs.get(i as usize).unwrap_or_else(|| {
                panic!("gpu{i} selected but this machine has {} GPU(s)", devs.len())
            }))
        }
        None => devs.first(),
    }
}

/// One requested class of compute, before it is resolved against real hardware.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    /// Every GPU present.
    AllGpus,
    /// A specific physical GPU index.
    Gpu(u32),
    /// Every CPU core.
    AllCpu,
    /// A specific set of CPU core ids (`cpu21`, `cpu0-7`).
    CpuCores(Vec<usize>),
    /// Every NPU present.
    AllNpus,
    /// A specific NPU index.
    Npu(u32),
    /// Force the native-Vulkan backend rather than wgpu, for GPU work.
    VulkanBackend,
    /// Force the wgpu backend explicitly.
    WgpuBackend,
}

/// A parsed `--device` value. `None` (the flag absent) means *use everything*.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DeviceSpec {
    pub requests: Vec<Request>,
    /// The original text, recorded in benchmark artifacts.
    pub source: String,
}

impl DeviceSpec {
    /// Parse a comma-separated `--device` value.
    ///
    /// Grammar per token: `cpu | gpu | npu | vulkan | wgpu`, each optionally
    /// suffixed by an index (`gpu0`) or, for `cpu`, an inclusive range
    /// (`cpu0-7`). Empty input yields an empty spec, which resolves to *all
    /// hardware*.
    pub fn parse(s: &str) -> Result<DeviceSpec, String> {
        let source = s.trim().to_string();
        let mut requests = Vec::new();
        if source.is_empty() {
            return Ok(DeviceSpec { requests, source });
        }
        for raw in source.split(',') {
            let tok = raw.trim().to_ascii_lowercase();
            if tok.is_empty() {
                continue;
            }
            requests.push(parse_token(&tok)?);
        }
        Ok(DeviceSpec { requests, source })
    }

    /// True when nothing was requested - schedule on everything available.
    pub fn is_all(&self) -> bool {
        self.requests.is_empty()
    }

    /// Resolve against the machine. `probe` supplies the hardware inventory so
    /// this is testable without a GPU.
    pub fn resolve(&self, probe: &Inventory) -> Result<ComputeSet, String> {
        let mut set = ComputeSet {
            gpus: Vec::new(),
            cpu_cores: Vec::new(),
            npus: Vec::new(),
            backend: Backend::Wgpu,
            source: if self.source.is_empty() { "all".into() } else { self.source.clone() },
            explicit: !self.is_all(),
        };

        if self.is_all() {
            // Everything present. GPUs first (the scheduler prefers them), CPU
            // always usable, NPU when one exists.
            set.gpus = (0..probe.gpus).collect();
            set.cpu_cores = (0..probe.cpu_cores).collect();
            set.npus = (0..probe.npus).collect();
            set.backend = if probe.gpus > 0 { Backend::Wgpu } else { Backend::Cpu };
            return Ok(set);
        }

        let mut want_vulkan = false;
        let mut any_gpu_request = false;
        let mut any_npu_request = false;
        for r in &self.requests {
            match r {
                Request::AllGpus => {
                    any_gpu_request = true;
                    set.gpus.extend(0..probe.gpus);
                }
                Request::Gpu(i) => {
                    any_gpu_request = true;
                    if *i >= probe.gpus {
                        return Err(format!(
                            "gpu{i} requested but this machine has {} GPU(s)",
                            probe.gpus
                        ));
                    }
                    set.gpus.push(*i);
                }
                Request::AllCpu => set.cpu_cores.extend(0..probe.cpu_cores),
                Request::CpuCores(cores) => {
                    for &c in cores {
                        if c >= probe.cpu_cores {
                            return Err(format!(
                                "cpu{c} requested but this machine has {} core(s)",
                                probe.cpu_cores
                            ));
                        }
                        set.cpu_cores.push(c);
                    }
                }
                Request::AllNpus => {
                    any_npu_request = true;
                    set.npus.extend(0..probe.npus);
                }
                Request::Npu(i) => {
                    any_npu_request = true;
                    if *i >= probe.npus {
                        return Err(format!(
                            "npu{i} requested but this machine has {} NPU(s)",
                            probe.npus
                        ));
                    }
                    set.npus.push(*i);
                }
                Request::VulkanBackend => {
                    want_vulkan = true;
                    any_gpu_request = true;
                    if set.gpus.is_empty() {
                        set.gpus.extend(0..probe.gpus);
                    }
                }
                Request::WgpuBackend => {
                    any_gpu_request = true;
                    if set.gpus.is_empty() {
                        set.gpus.extend(0..probe.gpus);
                    }
                }
            }
        }

        set.gpus.sort_unstable();
        set.gpus.dedup();
        set.cpu_cores.sort_unstable();
        set.cpu_cores.dedup();
        set.npus.sort_unstable();
        set.npus.dedup();

        if any_gpu_request && set.gpus.is_empty() {
            return Err("GPU compute requested but no GPU was found".to_string());
        }
        if any_npu_request && set.npus.is_empty() {
            return Err(
                "NPU compute requested but no NPU was found (expected /dev/accel/accel*)".to_string(),
            );
        }
        if set.gpus.is_empty() && set.cpu_cores.is_empty() && set.npus.is_empty() {
            return Err(format!("--device {:?} selects no usable compute", self.source));
        }

        // The host backend a `Gpu::new()` builds. NPU-only still needs a host
        // backend for pre/post-processing, and that host work is CPU work.
        set.backend = if !set.gpus.is_empty() {
            if want_vulkan {
                Backend::Vulkan
            } else {
                Backend::Wgpu
            }
        } else {
            Backend::Cpu
        };
        Ok(set)
    }
}

fn parse_token(tok: &str) -> Result<Request, String> {
    let split = |name: &str| tok.strip_prefix(name).map(|rest| rest.trim().to_string());

    if let Some(rest) = split("gpu") {
        return match rest.as_str() {
            "" => Ok(Request::AllGpus),
            n => n
                .parse::<u32>()
                .map(Request::Gpu)
                .map_err(|_| format!("bad GPU index in {tok:?} (expected e.g. gpu0)")),
        };
    }
    if let Some(rest) = split("npu") {
        return match rest.as_str() {
            "" => Ok(Request::AllNpus),
            n => n
                .parse::<u32>()
                .map(Request::Npu)
                .map_err(|_| format!("bad NPU index in {tok:?} (expected e.g. npu0)")),
        };
    }
    if let Some(rest) = split("cpu") {
        return match rest.as_str() {
            "" => Ok(Request::AllCpu),
            n => parse_cores(n).map(Request::CpuCores).map_err(|e| format!("{e} in {tok:?}")),
        };
    }
    if tok == "vulkan" {
        return Ok(Request::VulkanBackend);
    }
    if tok == "wgpu" {
        return Ok(Request::WgpuBackend);
    }
    Err(format!(
        "unknown device {tok:?} - expected cpu | gpu | npu | vulkan | wgpu, \
         each optionally indexed (gpu0, cpu21, cpu0-7)"
    ))
}

/// `21` or `0-7` (inclusive).
fn parse_cores(s: &str) -> Result<Vec<usize>, String> {
    if let Some((a, b)) = s.split_once('-') {
        let lo: usize = a.trim().parse().map_err(|_| format!("bad core range start {a:?}"))?;
        let hi: usize = b.trim().parse().map_err(|_| format!("bad core range end {b:?}"))?;
        if hi < lo {
            return Err(format!("core range {lo}-{hi} is inverted"));
        }
        return Ok((lo..=hi).collect());
    }
    s.parse::<usize>().map(|c| vec![c]).map_err(|_| format!("bad core id {s:?}"))
}

/// What the machine actually has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inventory {
    pub gpus: u32,
    pub cpu_cores: usize,
    pub npus: u32,
}

impl Inventory {
    /// Probe the current machine.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn probe() -> Inventory {
        Inventory {
            // Not `discrete_gpu_count`: the default (`--device` absent) set must
            // include an integrated GPU when that's the only card present, or an
            // iGPU-only box silently loses GPU scheduling entirely.
            gpus: crate::visible_gpu_count() as u32,
            cpu_cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            npus: npu_count(),
        }
    }
}

/// Intel NPUs expose `/dev/accel/accel*`; OpenVINO is loaded at run time, so
/// presence is a device-node question, not a linkage one.
#[cfg(not(target_arch = "wasm32"))]
fn npu_count() -> u32 {
    let mut n = 0;
    if let Ok(rd) = std::fs::read_dir("/dev/accel") {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with("accel") {
                n += 1;
            }
        }
    }
    n
}

/// Which host backend a resolved set implies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Wgpu,
    Cpu,
    Vulkan,
}

/// The resolved, schedulable compute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeSet {
    /// Physical GPU indices work may be scheduled on.
    pub gpus: Vec<u32>,
    /// CPU core ids work may run on. A strict subset means affinity is pinned
    /// and the thread pool is sized to match.
    pub cpu_cores: Vec<usize>,
    /// NPU indices work may be scheduled on.
    pub npus: Vec<u32>,
    /// The host backend a model instantiates.
    pub backend: Backend,
    /// The `--device` text, or `"all"`.
    pub source: String,
    /// False when `--device` was absent (i.e. "everything").
    pub explicit: bool,
}

impl ComputeSet {
    pub fn gpu_enabled(&self) -> bool {
        !self.gpus.is_empty()
    }
    pub fn cpu_enabled(&self) -> bool {
        !self.cpu_cores.is_empty()
    }
    pub fn npu_enabled(&self) -> bool {
        !self.npus.is_empty()
    }
    /// True when the CPU is restricted to a subset of cores.
    pub fn cpu_pinned(&self, total_cores: usize) -> bool {
        self.cpu_enabled() && self.cpu_cores.len() < total_cores
    }
    /// A single GPU index when exactly one is schedulable - what pins the
    /// registry's ambient device selection (see [`set_ambient_gpu`]).
    pub fn single_gpu(&self) -> Option<u32> {
        (self.gpus.len() == 1).then(|| self.gpus[0])
    }
}

/// What [`ComputeSet::apply_backend`] records in the ambient pin for a
/// resolved set: `None` means "record nothing", which is what keeps
/// `BRAIN_GPU_INDEX` reachable as the ladder's level 3.
///
/// `narrowed` is [`ComputeSet::explicit`] - false exactly when the request
/// was "everything". The distinction matters because recording a pin
/// unconditionally shadows level 3 whether or not the user asked for
/// anything, which discards an exported `BRAIN_GPU_INDEX` in silence.
///
/// Clearing is still right for a set the user DID narrow: `--device cpu` or
/// `--device gpu0,gpu1` states where work may go, and an inherited env pin
/// must not override that. It is not needed to make multi-GPU scheduling
/// safe - [`current_gpu`] resolves `SCOPED` ahead of the ambient selection,
/// so a `with_gpu` lane already beats both the pin and the env.
///
/// Pure, so the ladder is testable without the `OnceLock`/`Mutex` globals a
/// real run resolves through.
fn ambient_pin_for(single: Option<u32>, narrowed: bool) -> Option<Option<u32>> {
    match (single, narrowed) {
        (Some(i), _) => Some(Some(i)),
        (None, true) => Some(None),
        (None, false) => None,
    }
}

impl ComputeSet {
    /// The backend + ambient GPU pin half of [`Self::apply`] - side-effect
    /// light and safe to call from a library/test context (e.g. from
    /// [`ambient_compute_set`] when a plain `cargo test` binary lazily
    /// resolves `BRAIN_DEVICE` with no CLI in the loop): it never touches
    /// the process-wide rayon pool size or CPU affinity, so a library caller
    /// building a `Gpu` never triggers those as a side effect.
    ///
    /// * selects the host backend;
    /// * pins the registry's ambient GPU selection when exactly one card is
    ///   schedulable, so every later `Gpu::new` binds that physical card.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn apply_backend(&self) -> Result<(), String> {
        crate::set_default_backend(match self.backend {
            Backend::Wgpu => crate::Backend::Wgpu,
            Backend::Cpu => crate::Backend::Cpu,
            Backend::Vulkan => crate::Backend::Vulkan,
        });

        // One card selected: pin it in the registry's ambient selection, so
        // every later `Gpu::new` binds that physical card. A set the user
        // narrowed to something other than one card clears the pin instead,
        // so an inherited `BRAIN_GPU_INDEX` cannot override the restriction.
        // A set that narrows nothing records nothing, which leaves that
        // variable reachable as the ladder's level 3 (see `ambient_pin_for`).
        if let Some(pin) = ambient_pin_for(self.single_gpu(), self.explicit) {
            set_ambient_gpu(pin);
        }
        Ok(())
    }

    /// Make this set the process's actual schedulable compute. CLI-only: on
    /// top of [`Self::apply_backend`], sizes the rayon pool to the selected
    /// core count and pins the process's CPU affinity to those cores, so
    /// `cpu21` really is one core rather than one core's worth of threads
    /// spread over the machine. Called once at start-up, before any model is
    /// built, so every later `Gpu::new()` and every rayon pool observes it.
    ///
    /// Note this bounds **compute**, not memory: host RAM and disk remain
    /// available as cache/spill tiers regardless.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn apply(&self) -> Result<(), String> {
        self.apply_backend()?;

        if self.cpu_enabled() {
            // Only override when the user narrowed the CPU; otherwise respect an
            // existing RAYON_NUM_THREADS.
            let total = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            if self.cpu_cores.len() < total {
                std::env::set_var("RAYON_NUM_THREADS", self.cpu_cores.len().to_string());
                pin_to_cores(&self.cpu_cores)?;
            }
        }
        Ok(())
    }
}

// ---- process-wide ambient compute set --------------------------------------

/// The process's resolved `--device`/`BRAIN_DEVICE` compute set: published
/// explicitly by the CLI ([`publish_compute_set`]), or lazily resolved by
/// [`ambient_compute_set`] on first use by any other caller. Exactly one
/// `OnceLock`, so there is exactly one resolution no matter which path gets
/// there first - a second `publish_compute_set` (or a lazy resolution that
/// loses the race to an explicit publish) is a no-op, first writer wins.
static AMBIENT_COMPUTE: std::sync::OnceLock<ComputeSet> = std::sync::OnceLock::new();

/// Publish the CLI's resolved `--device` set as the process-wide ambient
/// compute set. Called once by `crates/cli`'s `select_backend`, before any
/// model is built, so [`ambient_compute_set`] returns exactly what the CLI
/// resolved rather than re-deriving it from `BRAIN_DEVICE` a second time.
#[cfg(not(target_arch = "wasm32"))]
pub fn publish_compute_set(set: ComputeSet) {
    let _ = AMBIENT_COMPUTE.set(set);
}

/// The set the CLI already published via [`publish_compute_set`], or `None`
/// before that has happened (a process with no CLI in the loop at all, or a
/// caller running before `select_backend` - use [`ambient_compute_set`]
/// there instead, which always resolves to something).
#[cfg(not(target_arch = "wasm32"))]
pub fn published_compute_set() -> Option<&'static ComputeSet> {
    AMBIENT_COMPUTE.get()
}

/// The process's ambient compute set - the single source of truth every
/// non-CLI caller (a test binary, a library caller that never goes through
/// `crates/cli`) now resolves `--device`/`BRAIN_DEVICE` through, instead of
/// re-deriving it with a second, weaker parser.
///
/// Returns whatever [`publish_compute_set`] already recorded (the CLI path);
/// otherwise resolves `BRAIN_DEVICE` through the exact same strong grammar
/// `--device` uses ([`DeviceSpec::parse`] + [`DeviceSpec::resolve`] against
/// [`Inventory::probe`]), applies only the backend/GPU-pin half
/// ([`ComputeSet::apply_backend`] - never the CLI-only thread-pool/affinity
/// side effects of [`ComputeSet::apply`]), and caches the result for the
/// process lifetime (deliberately - the same one-shot-per-process treatment
/// this module already gives `BRAIN_GPU_INDEX`, see [`ambient_gpu`]'s doc).
///
/// A `BRAIN_DEVICE` value that fails to parse or fails to resolve against
/// this machine's real hardware (an out-of-range `gpu99`, an NPU request on
/// a box with none, …) prints ONE warning to stderr and falls back to the
/// default "all devices" set - this never panics, and never silently
/// reinterprets an unrecognised token as "just use wgpu, ambient card" the
/// way the old weak ladder did.
#[cfg(not(target_arch = "wasm32"))]
pub fn ambient_compute_set() -> &'static ComputeSet {
    AMBIENT_COMPUTE.get_or_init(|| {
        let text = std::env::var("BRAIN_DEVICE").unwrap_or_default();
        let probe = Inventory::probe();
        let resolved = DeviceSpec::parse(&text).and_then(|spec| spec.resolve(&probe));
        let set = match resolved {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "brain: BRAIN_DEVICE={text:?}: {e}; falling back to the default all-devices set"
                );
                DeviceSpec::default()
                    .resolve(&probe)
                    .expect("the empty/default device spec always resolves")
            }
        };
        // Best-effort: only the backend/GPU-pin half applies here (see the
        // doc above); a failure has nothing more to do than leave the
        // process on whatever backend it already had.
        let _ = set.apply_backend();
        set
    })
}

/// How many GPUs a machine-shape decision may actually use RIGHT NOW - the
/// `--device`/`BRAIN_DEVICE` restriction applied, not the machine's raw
/// physical card count ([`gpus`]/`visible_gpu_count`). Every "should this
/// model shard across N GPUs, or fall back to a single-GPU streaming window"
/// decision (`s3dit::pipeline::hifi_needs_window` and its siblings) must read
/// THIS, never `gpus().len()` directly - otherwise `--device gpu0` restricts
/// nothing: the decision still sees every card the machine physically has,
/// picks the multi-GPU shape, and the kernels that actually dispatch (which DO
/// honour the restriction) either fail outright or silently touch a card
/// outside the requested set.
///
/// Reads [`ambient_compute_set`] (the same single resolution `--device` goes
/// through everywhere else), so it agrees with what `Gpu::new` can actually
/// reach - a caller never needs its own env/CLI parsing to answer this.
#[cfg(not(target_arch = "wasm32"))]
pub fn schedulable_gpu_count() -> usize {
    ambient_compute_set().gpus.len()
}

/// Restrict this process to `cores` via `sched_setaffinity`.
///
/// Uses the `libc` crate rather than a hand-rolled `extern "C"`. brain's build
/// constraint is *no `bindgen`/libclang in the build path* (see
/// `crates/capture/src/v4l2.rs`, which avoids `v4l`/`nokhwa` for exactly that
/// reason) - `libc` is pure Rust with no build-time C toolchain and is already
/// a transitive dependency here via `cranelift-jit` and `wgpu-hal`, so it costs
/// nothing and gets `cpu_set_t`'s real size and proper errno reporting right.
#[cfg(all(target_os = "linux", not(target_arch = "wasm32")))]
pub fn pin_to_cores(cores: &[usize]) -> Result<(), String> {
    let ncpu = libc::CPU_SETSIZE as usize;
    // SAFETY: zeroed cpu_set_t is the documented empty set; CPU_SET only writes
    // within the set for indices < CPU_SETSIZE, which is checked below.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &c in cores {
        if c >= ncpu {
            return Err(format!("core id {c} exceeds the {ncpu}-cpu affinity mask"));
        }
        unsafe { libc::CPU_SET(c, &mut set) };
    }
    // pid 0 == the calling process.
    let rc = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        return Err(format!("sched_setaffinity({cores:?}) failed: {err}"));
    }
    Ok(())
}

/// Non-Linux: affinity is not portable, so the core *count* still sizes the
/// thread pool but placement is left to the OS.
#[cfg(all(not(target_os = "linux"), not(target_arch = "wasm32")))]
pub fn pin_to_cores(_cores: &[usize]) -> Result<(), String> {
    Ok(())
}

impl fmt::Display for ComputeSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if !self.gpus.is_empty() {
            parts.push(format!(
                "gpu[{}]",
                self.gpus.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",")
            ));
        }
        if !self.npus.is_empty() {
            parts.push(format!(
                "npu[{}]",
                self.npus.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",")
            ));
        }
        if !self.cpu_cores.is_empty() {
            parts.push(format!("cpu[{} core(s)]", self.cpu_cores.len()));
        }
        write!(f, "{}", parts.join(" + "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inv(gpus: u32, cores: usize, npus: u32) -> Inventory {
        Inventory { gpus, cpu_cores: cores, npus }
    }

    fn resolve(s: &str, i: Inventory) -> Result<ComputeSet, String> {
        DeviceSpec::parse(s)?.resolve(&i)
    }

    #[test]
    fn absent_uses_every_device() {
        let set = resolve("", inv(2, 48, 1)).unwrap();
        assert_eq!(set.gpus, vec![0, 1]);
        assert_eq!(set.cpu_cores.len(), 48);
        assert_eq!(set.npus, vec![0]);
        assert!(!set.explicit);
        assert_eq!(set.backend, Backend::Wgpu);
    }

    /// The placement ladder this module documents (see the "ambient + scoped
    /// selection" note) puts `BRAIN_GPU_INDEX` at level 3, below a `--device
    /// gpu<i>` pin. A run that narrows nothing must therefore leave level 3
    /// reachable: recording an unconditional "no pin" shadows the variable
    /// entirely, and a user who exports it is ignored in silence.
    ///
    /// Asserted through a pure function so the ladder is checked without
    /// touching the process-wide `AMBIENT_PIN`/`ENV_AMBIENT` globals - a
    /// `OnceLock` and a `Mutex` that a test cannot re-resolve per case.
    #[test]
    fn narrowing_nothing_leaves_brain_gpu_index_reachable() {
        // `--device gpu1`: exactly one card, so pin it. Level 2 beats level 3.
        assert_eq!(ambient_pin_for(Some(1), true), Some(Some(1)));
        // `--device gpu` / `--device cpu`: narrowed, but not to a single card.
        // An inherited env pin must not leak into a set the user restricted.
        assert_eq!(ambient_pin_for(None, true), Some(None));
        // No `--device` at all: record nothing, so `ambient_gpu` falls through
        // to `BRAIN_GPU_INDEX` exactly as the ladder promises.
        assert_eq!(ambient_pin_for(None, false), None);
    }

    /// Ties the ladder to what `resolve` really produces, so the rule cannot
    /// drift away from the `explicit` flag it reads.
    #[test]
    fn a_bare_run_does_not_shadow_the_env_pin_but_a_single_card_does() {
        let bare = resolve("", inv(2, 48, 1)).unwrap();
        assert_eq!(
            ambient_pin_for(bare.single_gpu(), bare.explicit),
            None,
            "a run with no --device must leave BRAIN_GPU_INDEX reachable"
        );
        let one = resolve("gpu1", inv(2, 48, 1)).unwrap();
        assert_eq!(ambient_pin_for(one.single_gpu(), one.explicit), Some(Some(1)));
    }

    #[test]
    fn absent_on_a_gpuless_box_is_cpu_backed_but_still_all() {
        let set = resolve("", inv(0, 8, 0)).unwrap();
        assert!(set.gpus.is_empty());
        assert_eq!(set.cpu_cores.len(), 8);
        assert_eq!(set.backend, Backend::Cpu);
    }

    #[test]
    fn gpu_means_all_gpus_and_nothing_else() {
        let set = resolve("gpu", inv(2, 48, 1)).unwrap();
        assert_eq!(set.gpus, vec![0, 1]);
        assert!(!set.cpu_enabled(), "--device gpu must not schedule CPU compute");
        assert!(!set.npu_enabled(), "--device gpu must not schedule NPU compute");
    }

    #[test]
    fn cpu_means_cpu_only() {
        let set = resolve("cpu", inv(2, 48, 1)).unwrap();
        assert!(!set.gpu_enabled());
        assert!(!set.npu_enabled());
        assert_eq!(set.cpu_cores.len(), 48);
        assert_eq!(set.backend, Backend::Cpu);
    }

    #[test]
    fn npu_means_npu_only() {
        let set = resolve("npu", inv(2, 48, 1)).unwrap();
        assert_eq!(set.npus, vec![0]);
        assert!(!set.gpu_enabled());
        assert!(!set.cpu_enabled());
    }

    #[test]
    fn comma_separated_selects_several_classes() {
        let set = resolve("gpu,cpu", inv(2, 16, 0)).unwrap();
        assert_eq!(set.gpus, vec![0, 1]);
        assert_eq!(set.cpu_cores.len(), 16);
        assert!(!set.npu_enabled());
        assert_eq!(set.backend, Backend::Wgpu, "a GPU in the set implies the GPU backend");
    }

    #[test]
    fn gpu0_pins_one_card() {
        let set = resolve("gpu0", inv(2, 48, 0)).unwrap();
        assert_eq!(set.gpus, vec![0]);
        assert_eq!(set.single_gpu(), Some(0));
        let set1 = resolve("gpu1", inv(2, 48, 0)).unwrap();
        assert_eq!(set1.gpus, vec![1]);
        assert_eq!(set1.single_gpu(), Some(1));
    }

    #[test]
    fn two_indexed_gpus_are_not_a_single_gpu_pin() {
        let set = resolve("gpu0,gpu1", inv(2, 48, 0)).unwrap();
        assert_eq!(set.gpus, vec![0, 1]);
        assert_eq!(set.single_gpu(), None);
    }

    #[test]
    fn cpu21_selects_exactly_that_core() {
        let set = resolve("cpu21", inv(0, 48, 0)).unwrap();
        assert_eq!(set.cpu_cores, vec![21]);
        assert!(set.cpu_pinned(48));
    }

    #[test]
    fn cpu_range_is_inclusive() {
        let set = resolve("cpu0-7", inv(0, 48, 0)).unwrap();
        assert_eq!(set.cpu_cores, (0..=7).collect::<Vec<_>>());
        assert_eq!(set.cpu_cores.len(), 8);
    }

    #[test]
    fn mixed_index_forms_combine() {
        let set = resolve("gpu1,cpu0-3", inv(2, 48, 0)).unwrap();
        assert_eq!(set.gpus, vec![1]);
        assert_eq!(set.cpu_cores, vec![0, 1, 2, 3]);
    }

    #[test]
    fn duplicates_collapse() {
        let set = resolve("gpu0,gpu0,cpu1,cpu1", inv(2, 48, 0)).unwrap();
        assert_eq!(set.gpus, vec![0]);
        assert_eq!(set.cpu_cores, vec![1]);
    }

    #[test]
    fn requesting_an_npu_on_a_box_without_one_says_so() {
        let e = resolve("npu", inv(2, 48, 0)).unwrap_err();
        assert!(e.contains("no NPU was found"), "{e}");
    }

    #[test]
    fn out_of_range_indices_are_errors_not_silent_clamps() {
        let e = resolve("gpu5", inv(2, 48, 0)).unwrap_err();
        assert!(e.contains("gpu5") && e.contains("2 GPU"), "{e}");
        let e = resolve("cpu99", inv(2, 48, 0)).unwrap_err();
        assert!(e.contains("cpu99"), "{e}");
        let e = resolve("npu0", inv(2, 48, 0)).unwrap_err();
        assert!(e.contains("npu0"), "{e}");
    }

    #[test]
    fn requesting_gpu_on_a_gpuless_box_is_an_error() {
        let e = resolve("gpu", inv(0, 48, 0)).unwrap_err();
        assert!(e.contains("no GPU"), "{e}");
    }

    #[test]
    fn unknown_tokens_are_rejected_with_the_grammar() {
        let e = DeviceSpec::parse("tpu").unwrap_err();
        assert!(e.contains("unknown device"), "{e}");
        assert!(e.contains("cpu"), "the error should teach the grammar: {e}");
    }

    #[test]
    fn vulkan_selects_the_native_backend_over_all_gpus() {
        let set = resolve("vulkan", inv(2, 48, 0)).unwrap();
        assert_eq!(set.backend, Backend::Vulkan);
        assert_eq!(set.gpus, vec![0, 1]);
    }

    #[test]
    fn whitespace_and_case_are_tolerated() {
        let set = resolve(" GPU0 , CPU0-1 ", inv(2, 48, 0)).unwrap();
        assert_eq!(set.gpus, vec![0]);
        assert_eq!(set.cpu_cores, vec![0, 1]);
    }

    #[test]
    fn inverted_core_range_is_rejected() {
        let e = DeviceSpec::parse("cpu7-0").unwrap_err();
        assert!(e.contains("inverted"), "{e}");
    }

    #[test]
    fn display_is_readable() {
        let set = resolve("gpu0,cpu0-3", inv(2, 48, 0)).unwrap();
        let s = set.to_string();
        assert!(s.contains("gpu[0]"), "{s}");
        assert!(s.contains("4 core"), "{s}");
    }

    #[test]
    fn source_is_recorded_for_the_artifact() {
        assert_eq!(resolve("gpu0", inv(1, 4, 0)).unwrap().source, "gpu0");
        assert_eq!(resolve("", inv(1, 4, 0)).unwrap().source, "all");
    }

    // ---- registry (pure parts) ----------------------------------------------

    fn ident(name: &str, pci: Option<&str>, ordinal: usize, class: backend_api::DeviceClass) -> GpuIdentity {
        GpuIdentity {
            name: name.into(),
            vendor_id: 0x10de,
            device_id: 0x1b38,
            uuid: None,
            pci_bus: pci.map(|s| s.to_string()),
            ordinal,
            vram_bytes: 24 << 30,
            class,
        }
    }

    #[test]
    fn canonical_order_is_pci_sorted() {
        use backend_api::DeviceClass::DiscreteGpu;
        // Enumeration order deliberately reversed vs PCI order.
        let reg = DeviceRegistry::from_identities(
            vec![
                ident("B", Some("0000:82:00.0"), 0, DiscreteGpu),
                ident("A", Some("0000:04:00.0"), 1, DiscreteGpu),
            ],
            "vulkan",
        );
        let d = reg.devices();
        assert_eq!(d[0].identity.name, "A");
        assert_eq!(d[0].index, 0);
        assert_eq!(d[1].identity.name, "B");
        assert_eq!(d[1].index, 1);
    }

    #[test]
    fn software_rasterisers_never_get_an_index() {
        use backend_api::DeviceClass::{Cpu, DiscreteGpu};
        let reg = DeviceRegistry::from_identities(
            vec![ident("llvmpipe", None, 0, Cpu), ident("P40", Some("0000:04:00.0"), 0, DiscreteGpu)],
            "vulkan",
        );
        assert_eq!(reg.devices().len(), 1);
        assert_eq!(reg.devices()[0].identity.name, "P40");
    }

    #[test]
    fn discrete_cards_shadow_integrated_ones() {
        use backend_api::DeviceClass::{DiscreteGpu, IntegratedGpu};
        let reg = DeviceRegistry::from_identities(
            vec![
                ident("iGPU", Some("0000:00:02.0"), 0, IntegratedGpu),
                ident("dGPU", Some("0000:04:00.0"), 0, DiscreteGpu),
            ],
            "vulkan",
        );
        assert_eq!(reg.devices().len(), 1, "--device gpu means discrete when one exists");
        assert_eq!(reg.devices()[0].identity.name, "dGPU");
    }

    #[test]
    fn missing_pci_keeps_enumeration_order() {
        use backend_api::DeviceClass::DiscreteGpu;
        let reg = DeviceRegistry::from_identities(
            vec![ident("first", None, 0, DiscreteGpu), ident("second", None, 1, DiscreteGpu)],
            "wgpu",
        );
        assert_eq!(reg.devices()[0].identity.name, "first");
        assert_eq!(reg.devices()[1].identity.name, "second");
    }

    #[test]
    fn identity_matching_prefers_strongest_key() {
        use backend_api::DeviceClass::DiscreteGpu;
        let mut a = ident("P40", Some("0000:04:00.0"), 0, DiscreteGpu);
        let mut b = ident("P40", Some("0000:82:00.0"), 1, DiscreteGpu);
        // Twins: same (vendor,device), different PCI - must not match.
        assert!(!a.same_device(&b));
        // UUID wins over PCI when both sides carry one.
        a.uuid = Some([1; 16]);
        b.uuid = Some([1; 16]);
        assert!(a.same_device(&b));
        // Fallback key: (vendor:device, ordinal) when neither uuid nor pci.
        let c = ident("P40", None, 1, DiscreteGpu);
        let d = ident("P40", None, 1, DiscreteGpu);
        let e = ident("P40", None, 0, DiscreteGpu);
        assert!(c.same_device(&d));
        assert!(!c.same_device(&e));
    }
}

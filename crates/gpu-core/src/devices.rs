// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--device` — **which compute is schedulable**.
//!
//! The flag does not pick "a backend"; it declares the set of compute units
//! brain may schedule work onto. Everything else (which `Gpu` a model builds,
//! which residency budgets exist, how many CPU threads run, whether the NPU path
//! is allowed) follows from that set.
//!
//! ```text
//! (absent)        every device present — GPUs + CPU + NPU, scheduled together
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
//! Host RAM and disk are always available as *cache/spill* tiers — restricting
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
//! <cmd>` with no `--device` will not silently route work there — only an
//! explicit request does. Transparent NPU scheduling needs the per-model export
//! path first; see `docs/models/yolo/npu.md`.
//!
//! Parsing ([`DeviceSpec::parse`]) is pure and total; resolution
//! ([`DeviceSpec::resolve`]) probes the machine. They are separate so the
//! grammar is testable without hardware.

use std::fmt;

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

    /// True when nothing was requested — schedule on everything available.
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
        "unknown device {tok:?} — expected cpu | gpu | npu | vulkan | wgpu, \
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
            gpus: crate::discrete_gpu_count() as u32,
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
    /// A single GPU index when exactly one is schedulable — what pins
    /// `BRAIN_GPU_INDEX`.
    pub fn single_gpu(&self) -> Option<u32> {
        (self.gpus.len() == 1).then(|| self.gpus[0])
    }
}

impl ComputeSet {
    /// Make this set the process's actual schedulable compute.
    ///
    /// Called once at start-up, before any model is built, so every later
    /// `Gpu::new()` and every rayon pool observes it:
    ///
    /// * selects the host backend;
    /// * pins `BRAIN_GPU_INDEX` when exactly one GPU is schedulable, so the wgpu
    ///   backend binds that physical card;
    /// * sizes the rayon pool to the selected core count and pins the process's
    ///   CPU affinity to those cores, so `cpu21` really is one core rather than
    ///   one core's worth of threads spread over the machine.
    ///
    /// Note this bounds **compute**, not memory: host RAM and disk remain
    /// available as cache/spill tiers regardless.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn apply(&self) -> Result<(), String> {
        crate::set_default_backend(match self.backend {
            Backend::Wgpu => crate::Backend::Wgpu,
            Backend::Cpu => crate::Backend::Cpu,
            Backend::Vulkan => crate::Backend::Vulkan,
        });

        match self.single_gpu() {
            // One card selected: bind it. Multi-GPU scheduling picks cards per
            // job instead, so a global pin would be wrong there.
            Some(i) => std::env::set_var("BRAIN_GPU_INDEX", i.to_string()),
            None => std::env::remove_var("BRAIN_GPU_INDEX"),
        }

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

/// Restrict this process to `cores` via `sched_setaffinity`.
///
/// Uses the `libc` crate rather than a hand-rolled `extern "C"`. brain's build
/// constraint is *no `bindgen`/libclang in the build path* (see
/// `crates/capture/src/v4l2.rs`, which avoids `v4l`/`nokhwa` for exactly that
/// reason) — `libc` is pure Rust with no build-time C toolchain and is already
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
}

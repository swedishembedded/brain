// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native **Vulkan compute** eager [`Backend`] — selected at runtime
//! (`--device vulkan` / `BRAIN_DEVICE=vulkan`), with graceful fallback to wgpu
//! when no Vulkan device/ICD is present.
//!
//! WGSL stays the single source of truth: each kernel is compiled WGSL -> SPIR-V
//! by naga at init ([`vulkan::shader::wgsl_to_spirv`]), reflected for its
//! `@group(0)` bindings ([`vulkan::shader::wgsl_bindings`]) to build a matching
//! descriptor-set layout (uniform at binding 0, storage at 1..), and compiled to a
//! compute pipeline once. It honours the same engine invariants as the other
//! backends — one bind group, `@workgroup_size(64)`, the >65535-group 2D-grid
//! tiling ([`backend_api::grid`]) — so every existing model runs unmodified.
//!
//! Execution model mirrors the wgpu backend: `submit` lazily accumulates
//! dispatches; the batch is flushed (recorded into one command buffer with a
//! storage memory barrier between dependent dispatches, submitted, fence-waited)
//! on the next `read`/`write`/`poll_wait`. Transient per-dispatch uniform buffers
//! and descriptor sets are reclaimed after each flush; model storage buffers are
//! host-visible and live until the process exits (a one-shot CLI never frees them
//! mid-run — see the note on `VkOwnedBuffer`).
//!
//! Buffers are host-visible+coherent (no DEVICE_LOCAL/staging split yet), which is
//! correct and simple; a perf pass can add a staged device-local path later.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;
use vulkan::context::{VkBuffer, VkContext};
use vulkan::shader;

use backend_api::{Backend, BufUsage, DeviceBuffer, Step};

/// Ceiling for one `wait_for_fences` call, nanoseconds. Generous — a legitimate
/// prefill dispatch is slow — but finite: `u64::MAX` (the previous value) made a
/// wedged queue block the process forever rather than error, which is why past
/// hangs (`omni_bench encode-vision`, `gpu_core::roofline`) presented as unkillable
/// instead of as a reported failure. Override with `BRAIN_GPU_WAIT_S`.
fn gpu_wait_timeout_ns() -> u64 {
    const DEFAULT_S: f64 = 30.0;
    let secs = std::env::var("BRAIN_GPU_WAIT_S")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_S);
    (secs * 1e9) as u64
}

/// A recorded dispatch: (pipeline index, descriptor set, grid_x, grid_y). All
/// fields are `Copy`, so `VkStep` is `Clone` like the wgpu backend's.
#[derive(Clone, Copy)]
pub struct VkStep {
    kind: usize,
    set: vk::DescriptorSet,
    gx: u32,
    gy: u32,
    /// True when a storage buffer is bound at a non-zero (sub-range) offset.
    /// Intel ANV mis-handles a compute-compute pipeline barrier across such
    /// bindings (flaky stale reads); a submit+fence boundary is honoured, so a
    /// batch containing a sliced step is serialized in `flush`.
    sliced: bool,
    /// True for steps built by `step`/`step_sliced` (backend-owned uniform +
    /// descriptor set, recycled after the flush that runs them). False for
    /// `step_buf` steps (caller-owned uniform, e.g. the `uniform_dynamic`
    /// training-loop reuse pattern), which stay valid across flushes.
    ///
    /// CONTRACT: a transient step is submit-once — re-submitting it after a
    /// flush may read a recycled (rewritten) uniform/descriptor set. Every
    /// in-repo model builds its transient steps and submits them immediately;
    /// hold-and-resubmit code must use `uniform_dynamic` + `step_buf`.
    transient: bool,
}

/// A device buffer handle. Memory is freed when the owning [`VulkanBackend`] is
/// dropped (or the process exits); individual drops are no-ops, matching how model
/// code holds its buffers for the whole run.
pub struct VkOwnedBuffer {
    inner: VkBuffer,
    /// The device this buffer belongs to, so dropping it can hand the handles
    /// back (`VkContext::bury`). Keeping the `Arc` alive also makes the
    /// buffer-outlives-device ordering impossible to get wrong.
    ctx: Arc<VkContext>,
}

impl VkOwnedBuffer {
    fn bytes(&self) -> u64 {
        self.inner.size
    }
}

/// **Dropping a device buffer used to free nothing on this backend**: there was
/// no `Drop` at all, so every buffer any model ever allocated stayed on the card
/// until the whole `VkContext` was destroyed. A resident model never noticed
/// (its weights are meant to live forever), but two real paths did:
///
/// * `omni`'s bf16 Thinker streams the decoder layers that do not fit, dropping
///   each layer's ~2.4 GiB of expert weights before loading the next. On wgpu
///   (where `Drop` frees) its live set is one layer; here it was every layer, so
///   a single request walked a 24 GB card to `ERROR_OUT_OF_DEVICE_MEMORY` - the
///   exact failure `paramstore::upload::Uploader::drain` exists to prevent,
///   silently un-prevented by the backend underneath it.
/// * Every per-token transient (residual stream, RoPE tables, logits) on ANY
///   resident model leaked for the life of the process, so a server's VRAM use
///   grew with the number of requests served instead of staying flat.
///
/// Freeing on the spot would be a use-after-free - [`VulkanBackend::step`]
/// records dispatches into a pending list naming raw `vk::Buffer` handles that
/// are submitted later - so the handles are buried and destroyed at the next
/// point the device is provably done with them (see [`VulkanBackend::flush`]).
impl Drop for VkOwnedBuffer {
    fn drop(&mut self) {
        // Handed over field by field rather than by making `VkBuffer` `Copy`:
        // the handles must have exactly ONE owner responsible for destroying
        // them, and a `Copy` `VkBuffer` would make an accidental double-free a
        // silent one-character mistake.
        let VkBuffer { buffer, memory, size, host_visible } = self.inner;
        self.ctx.bury(VkBuffer { buffer, memory, size, host_visible });
    }
}

/// A compiled kernel: its SPIR-V module, descriptor-set/pipeline layout, the
/// compute pipeline, and the reflected `(binding, is_uniform)` list.
struct KernelPipeline {
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    bindings: Vec<(u32, bool)>,
}

/// One kernel set's compiled pipelines, plus the `Arc<VkContext>` that keeps
/// their owning device/instance alive for at least as long as they do.
///
/// This is the unit [`VulkanBackend::share`]/[`VulkanBackend::new_like`] pass
/// around: `share()` clones the `Arc` (same pipelines, same device — a second
/// handle with its own command stream); `new_like()` compiles a FRESH set
/// against the SAME `ctx` (new pipelines, same device). Before this type
/// existed, `VulkanBackend` had no way to hand out a second handle onto its
/// device at all, so `Backend::share`/`Backend::new_like` fell through to the
/// trait's `None` default and every caller asking this backend to share
/// (`gpu_core::testgpu::dev`, `gpu_core::roof::measure`'s internal probe
/// device) silently built a WHOLE NEW Vulkan device instead — the exact
/// "many concurrent devices on one card" shape `device_sharing.rs` documents
/// as deadlocking the NVIDIA driver roughly half the time.
struct VkPipelineSet {
    ctx: Arc<VkContext>,
    pipelines: Vec<KernelPipeline>,
    wgsizes: Vec<u32>,
    /// Kernel names, parallel to `pipelines`/`wgsizes` — carried alongside the
    /// compiled pipelines so a `share()`'d handle (which has no `kernels` list
    /// of its own) can still report per-kernel names via `kernel_times()`.
    names: Vec<String>,
}

impl Drop for VkPipelineSet {
    fn drop(&mut self) {
        let dev = &self.ctx.device;
        unsafe {
            for kp in &self.pipelines {
                dev.destroy_pipeline(kp.pipeline, None);
                dev.destroy_pipeline_layout(kp.layout, None);
                dev.destroy_descriptor_set_layout(kp.set_layout, None);
                dev.destroy_shader_module(kp.module, None);
            }
        }
    }
}

/// The Vulkan compute device.
/// Relaxed device-op counters, mirroring what the wgpu backend records so the
/// two are directly comparable.
#[derive(Default)]
struct OpCounters {
    submits: AtomicU64,
    dispatches: AtomicU64,
    readbacks: AtomicU64,
    bind_groups: AtomicU64,
    uniform_allocs: AtomicU64,
}

/// The largest batch this backend can timestamp in one flush. Timestamps
/// bracket every dispatch (`n+1` marks), so this is the query-pool size.
/// A flush with more steps than this simply skips timing for that batch
/// (`kernel_times` stays honest about what it did and did not see) rather
/// than crashing or silently truncating — real model forwards are dozens to
/// low hundreds of dispatches, far under this.
const MAX_TIMED_DISPATCHES: usize = 8192;

/// Per-kernel-kind device timing, mirroring `backend-wgpu`'s `GpuProfile`
/// shape so `gpu_core::profile` gets the same `(name, ms, calls)` contract
/// from either backend. `None`/`false` everywhere when this device's
/// compute queue cannot write timestamps at all (`timestamp_valid_bits ==
/// 0`) — never a substituted host time.
struct VkProfile {
    enabled: std::sync::atomic::AtomicBool,
    /// Per pipeline index: (accumulated ms, calls).
    acc: Mutex<Vec<(f64, u64)>>,
    pool: Mutex<Option<vk::QueryPool>>,
}

impl VkProfile {
    fn new(n_kernels: usize) -> VkProfile {
        VkProfile {
            enabled: std::sync::atomic::AtomicBool::new(backend_api::profile_enabled()),
            acc: Mutex::new(vec![(0.0, 0); n_kernels]),
            pool: Mutex::new(None),
        }
    }
}

pub struct VulkanBackend {
    /// The device/instance/queue. `Arc`-shared: `share()` clones this handle,
    /// `new_like()` clones it too (same device, fresh pipelines) — the device
    /// is destroyed exactly once, when the last `VulkanBackend`/`WeakVulkan`
    /// referencing it drops.
    ctx: Arc<VkContext>,
    /// What this device can do — queried once at construction from the
    /// physical device (see `backend_api::DeviceCaps`).
    caps: backend_api::DeviceCaps,
    /// The device's real `maxStorageBufferRange`. Overrides the trait's fixed
    /// ~2 GiB default, which over-reports on devices with a smaller range
    /// (an oversized binding fails at dispatch, not allocation).
    max_storage_binding: u64,
    /// This handle's compiled kernel set. `Arc`-shared with every `share()`
    /// sibling; a fresh `Arc` (refcount 1) for every `new_like()` result, so
    /// each kernel set's pipelines are destroyed independently of the others'
    /// lifetime — see [`VkPipelineSet`].
    pipelines: Arc<VkPipelineSet>,
    /// Device-op counters, always maintained (relaxed atomics are negligible
    /// next to a dispatch) — the same contract `backend-wgpu` and
    /// `backend-cpu` implement. Without them `Backend::stats()` fell through to
    /// the trait's `None` default and every caller counting device ops was
    /// blind on this backend, which is how a `.unwrap_or(0)` at a call site
    /// silently turned "not counted" into "zero".
    stats: OpCounters,
    /// Descriptor pools, grown on demand. A descriptor set's lifetime is tied to
    /// its `VkStep` (which a caller may hold and re-submit, e.g. the
    /// `uniform_dynamic` + `step_buf` reuse pattern), so sets are NEVER reset
    /// mid-run — that would invalidate a reused set, exactly as the wgpu backend
    /// keeps its `BindGroup` alive per-Step. When the active pool is exhausted a
    /// new one is appended; all pools are freed when the backend drops. (A
    /// one-shot CLI is bounded; an unbounded training loop should reuse Steps,
    /// which keeps the set count flat.)
    pools: Mutex<Vec<vk::DescriptorPool>>,
    /// Accumulated dispatches, flushed as one command submission.
    pending: Mutex<Vec<VkStep>>,
    /// Transient uniforms of steps built but not yet passed to `submit`.
    uniforms: Mutex<Vec<VkBuffer>>,
    /// Transient uniforms of submitted-but-not-yet-flushed steps. Moved from
    /// `uniforms` at `submit`, recycled into `free_uniforms` after the flush's
    /// fence wait — never earlier, or an in-flight dispatch could see its
    /// params rewritten.
    inflight_uniforms: Mutex<Vec<VkBuffer>>,
    /// Recycled transient uniform buffers, keyed by byte size. A steady-state
    /// frame allocates ZERO uniforms (and performs zero queue submits building
    /// its steps — the uniforms are host-visible, written by direct map).
    free_uniforms: Mutex<std::collections::HashMap<u64, Vec<VkBuffer>>>,
    /// Recycled descriptor sets of flushed transient steps, keyed by pipeline
    /// index (sets are layout-specific). Rewriting an idle set via
    /// `update_descriptor_sets` is legal; the flush's fence wait is what makes
    /// them idle.
    free_sets: Mutex<std::collections::HashMap<usize, Vec<vk::DescriptorSet>>>,
    /// Per-kernel-kind device timestamp timing (`BRAIN_PROFILE`). See
    /// `VkProfile`'s doc — degrades to "unavailable" on a device/queue that
    /// cannot write timestamps, never substitutes host time.
    profile: VkProfile,
    /// Kernel names, parallel to `pipelines`/`wgsizes` — `kernel_times()`
    /// reports by name, matching the wgpu backend's contract.
    names: Vec<String>,
}

// ash handles are Send+Sync; all interior mutation goes through the Mutexes above.
unsafe impl Send for VulkanBackend {}
unsafe impl Sync for VulkanBackend {}

const POOL_MAX_SETS: u32 = 16384;

// ---- physical-GPU identity (the canonical registry's enumeration) -----------

/// Map a Vulkan device type onto the backend-neutral class vocabulary.
fn class_of(t: vk::PhysicalDeviceType) -> backend_api::DeviceClass {
    use backend_api::DeviceClass;
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceClass::DiscreteGpu,
        vk::PhysicalDeviceType::CPU => DeviceClass::Cpu,
        _ => DeviceClass::IntegratedGpu,
    }
}

/// Identity of one enumerated physical device. `ordinal` is this device's
/// position among devices sharing its (vendor, device) pair, in
/// `vkEnumeratePhysicalDevices` order — the tiebreaker for identical twins
/// when UUID and PCI are both unavailable.
///
/// # Safety
/// `instance` must be valid and `pd` one of its enumerated physical devices.
unsafe fn pd_identity(instance: &ash::Instance, pd: vk::PhysicalDevice, ordinal: usize) -> backend_api::GpuIdentity {
    let props = instance.get_physical_device_properties(pd);
    let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();

    // deviceUUID (Vulkan 1.1 core; == the NVML GPU UUID on NVIDIA).
    let mut idp = vk::PhysicalDeviceIDProperties::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut idp);
    instance.get_physical_device_properties2(pd, &mut p2);
    let uuid = (idp.device_uuid != [0u8; 16]).then_some(idp.device_uuid);

    // PCI bus id via VK_EXT_pci_bus_info — only queried where the device
    // advertises the extension (chaining an unsupported struct is UB per spec).
    let has_pci = instance
        .enumerate_device_extension_properties(pd)
        .map(|exts| {
            exts.iter().any(|e| {
                std::ffi::CStr::from_ptr(e.extension_name.as_ptr())
                    == ash::ext::pci_bus_info::NAME
            })
        })
        .unwrap_or(false);
    let pci_bus = has_pci.then(|| {
        let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut pci);
        instance.get_physical_device_properties2(pd, &mut p2);
        format!(
            "{:04x}:{:02x}:{:02x}.{:x}",
            pci.pci_domain, pci.pci_bus, pci.pci_device, pci.pci_function
        )
    });

    let mem = instance.get_physical_device_memory_properties(pd);
    let vram_bytes = mem.memory_heaps[..mem.memory_heap_count as usize]
        .iter()
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size)
        .max()
        .unwrap_or(0);

    backend_api::GpuIdentity {
        name,
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        uuid,
        pci_bus,
        ordinal,
        vram_bytes,
        class: class_of(props.device_type),
    }
}

/// The process's ONE enumeration instance, created on first use and never
/// destroyed - the same rule, and for the same reason, as
/// `brain_vulkan::context`'s `shared_instance`: destroying a Vulkan instance
/// makes the loader unload every ICD it opened, and an ICD that has been
/// unloaded and reloaded a handful of times stops resolving `vkCreateInstance`
/// altogether, leaving the process blind to hardware that is still physically
/// present.
///
/// Deliberately separate from the compute context's instance rather than
/// shared with it: this one asks for nothing beyond Vulkan 1.1 (enough for
/// `VkPhysicalDeviceIDProperties` / `properties2`), so the registry can still
/// enumerate cards on a driver that would refuse the compute instance's higher
/// API version and cooperative-matrix extension. Two long-lived instances cost
/// a handful of driver handles; instance CHURN is what does damage.
fn enumeration_instance() -> Result<&'static (ash::Entry, ash::Instance), String> {
    static SHARED: std::sync::OnceLock<Result<(ash::Entry, ash::Instance), String>> =
        std::sync::OnceLock::new();
    SHARED
        .get_or_init(|| unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("failed to load Vulkan loader: {e}"))?;
            // 1.1 for VkPhysicalDeviceIDProperties / properties2.
            let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
            let info = vk::InstanceCreateInfo::default().application_info(&app_info);
            let instance =
                entry.create_instance(&info, None).map_err(|e| format!("vkCreateInstance failed: {e}"))?;
            Ok((entry, instance))
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// Enumerate every Vulkan physical device with stable identity — the canonical
/// enumeration `gpu_core::devices` builds the process-wide registry from. Uses
/// the shared [`enumeration_instance`]; no logical device, no pipelines. `Err`
/// when no loader/ICD is present (the registry then falls back to the wgpu
/// enumeration).
pub fn enumerate_physical_gpus() -> Result<Vec<backend_api::GpuIdentity>, String> {
    let (_entry, instance) = enumeration_instance()?;
    unsafe {
        let pds = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate_physical_devices failed: {e}"))?;
        let mut ordinals: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
        let ids = pds
            .iter()
            .map(|&pd| {
                let props = instance.get_physical_device_properties(pd);
                let ord = ordinals.entry((props.vendor_id, props.device_id)).or_insert(0);
                let id = pd_identity(instance, pd, *ord);
                *ord += 1;
                id
            })
            .collect();
        Ok(ids)
    }
}

impl VulkanBackend {
    /// Try to build the Vulkan backend, compiling every kernel to a pipeline.
    /// Returns `Err` (so the caller can fall back to wgpu) if no Vulkan device is
    /// available or a kernel fails to compile.
    pub fn try_new(kernels: &[(&str, &str)]) -> Result<VulkanBackend, String> {
        let r = Self::try_new_impl(kernels, None);
        if let Err(e) = &r {
            tracing::warn!(error = %e, "native Vulkan backend unavailable; caller falls back to wgpu");
        }
        r
    }

    /// [`VulkanBackend::try_new`] on the specific physical card `target`,
    /// selected by identity (UUID → PCI → (vendor:device, ordinal)) — the
    /// registry-resolved placement path.
    pub fn try_new_on(kernels: &[(&str, &str)], target: &backend_api::GpuIdentity) -> Result<VulkanBackend, String> {
        tracing::trace!(name = %target.name, pci = ?target.pci_bus, "opening device (native Vulkan)");
        let r = Self::try_new_impl(kernels, Some(target));
        if let Err(e) = &r {
            tracing::warn!(name = %target.name, pci = ?target.pci_bus, error = %e, "native Vulkan backend unavailable for the requested card; caller falls back to wgpu");
        }
        r
    }

    fn try_new_impl(kernels: &[(&str, &str)], target: Option<&backend_api::GpuIdentity>) -> Result<VulkanBackend, String> {
        let ctx = match target {
            None => VkContext::new()?,
            Some(t) => VkContext::new_select(&|instance, pds| {
                let mut ordinals: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
                for (i, &pd) in pds.iter().enumerate() {
                    // SAFETY: instance/pd come from VkContext's own enumeration.
                    let id = unsafe {
                        let props = instance.get_physical_device_properties(pd);
                        let ord = ordinals.entry((props.vendor_id, props.device_id)).or_insert(0);
                        let id = pd_identity(instance, pd, *ord);
                        *ord += 1;
                        id
                    };
                    if t.same_device(&id) {
                        return Ok(i);
                    }
                }
                Err(format!("physical GPU {:?} (pci {:?}) not found by the Vulkan ICD", t.name, t.pci_bus))
            })?,
        };
        log_adapter(&ctx.adapter_name);
        let ctx = Arc::new(ctx);
        let pipelines = Self::compile_pipeline_set(ctx.clone(), kernels)?;
        let caps = Self::query_caps(&ctx);
        let max_storage_binding = unsafe {
            ctx.instance.get_physical_device_properties(ctx.physical_device)
        }
        .limits
        .max_storage_buffer_range as u64;
        Self::from_shared(ctx, Arc::new(pipelines), caps, max_storage_binding)
    }

    /// Compile `kernels` into a [`VkPipelineSet`] against an already-built
    /// `ctx` — the shared core of both initial construction and
    /// [`VulkanBackend::new_like`] (a fresh pipeline set on an EXISTING
    /// device, never a new `VkContext`).
    fn compile_pipeline_set(ctx: Arc<VkContext>, kernels: &[(&str, &str)]) -> Result<VkPipelineSet, String> {
        let dev = &ctx.device;
        let mut pipelines = Vec::with_capacity(kernels.len());
        for (name, src) in kernels {
            let spirv = shader::wgsl_to_spirv(src).map_err(|e| format!("{name}: {e}"))?;
            let bindings = shader::wgsl_bindings(src).map_err(|e| format!("{name}: {e}"))?;
            unsafe {
                let module = shader::make_shader_module(dev, &spirv).map_err(|e| format!("{name}: {e}"))?;
                let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
                    .iter()
                    .map(|&(b, is_uniform)| {
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(b)
                            .descriptor_type(if is_uniform {
                                vk::DescriptorType::UNIFORM_BUFFER
                            } else {
                                vk::DescriptorType::STORAGE_BUFFER
                            })
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    })
                    .collect();
                let set_layout = dev
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings),
                        None,
                    )
                    .map_err(|e| format!("{name}: set layout: {e}"))?;
                let set_layouts = [set_layout];
                let layout = dev
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                        None,
                    )
                    .map_err(|e| format!("{name}: pipeline layout: {e}"))?;
                let entry = std::ffi::CString::new("main").unwrap();
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(&entry);
                let pipeline = dev
                    .create_compute_pipelines(
                        vk::PipelineCache::null(),
                        &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
                        None,
                    )
                    .map_err(|(_, e)| format!("{name}: compute pipeline: {e}"))?[0];
                pipelines.push(KernelPipeline { module, set_layout, layout, pipeline, bindings });
            }
        }
        let wgsizes = backend_api::workgroup_sizes(kernels);
        let names = kernels.iter().map(|(n, _)| n.to_string()).collect();
        Ok(VkPipelineSet { ctx, pipelines, wgsizes, names })
    }

    /// Build a `VulkanBackend` handle around an existing `ctx`/`pipelines` —
    /// the common tail of fresh construction, [`VulkanBackend::share`] and
    /// [`VulkanBackend::new_like`]. Every handle gets its OWN descriptor pool
    /// and command-stream state (`pending`/`uniforms`/`free_sets`/...): those
    /// must never be shared, or two handles' batches would interleave.
    fn from_shared(
        ctx: Arc<VkContext>,
        pipelines: Arc<VkPipelineSet>,
        caps: backend_api::DeviceCaps,
        max_storage_binding: u64,
    ) -> Result<VulkanBackend, String> {
        let pool = unsafe { new_pool(&ctx.device)? };
        let profile = VkProfile::new(pipelines.pipelines.len());
        let names = pipelines.names.clone();
        Ok(VulkanBackend {
            ctx,
            caps,
            max_storage_binding,
            pipelines,
            stats: OpCounters::default(),
            pools: Mutex::new(vec![pool]),
            pending: Mutex::new(Vec::new()),
            uniforms: Mutex::new(Vec::new()),
            inflight_uniforms: Mutex::new(Vec::new()),
            free_uniforms: Mutex::new(std::collections::HashMap::new()),
            free_sets: Mutex::new(std::collections::HashMap::new()),
            profile,
            names,
        })
    }

    /// Fill [`backend_api::DeviceCaps`] from the physical device. Everything
    /// here is queried, never assumed: `int8_dot` comes from the measured
    /// DP4A property the context established at device creation, and fast-f16
    /// stays false until a measured rate says otherwise (Pascal exposes f16 at
    /// 1/64 rate — availability is not speed).
    fn query_caps(ctx: &VkContext) -> backend_api::DeviceCaps {
        use backend_api::{DeviceCaps, DeviceClass, NumericSupport};
        let props = unsafe { ctx.instance.get_physical_device_properties(ctx.physical_device) };
        let class = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => DeviceClass::DiscreteGpu,
            vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceClass::IntegratedGpu,
            // A software rasteriser executes on host cores.
            vk::PhysicalDeviceType::CPU => DeviceClass::Cpu,
            // Unknown/virtual: conservative middle; unified memory is decided
            // separately so this assumes no zero-copy.
            _ => DeviceClass::IntegratedGpu,
        };
        // Subgroup width (Vulkan 1.1 core; the context already requires 1.1+).
        let mut sub = vk::PhysicalDeviceSubgroupProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sub);
        unsafe { ctx.instance.get_physical_device_properties2(ctx.physical_device, &mut p2) };
        DeviceCaps {
            class,
            compute_units: None, // core Vulkan exposes no SM/CU count
            max_workgroup_size: props.limits.max_compute_work_group_invocations,
            workgroup_mem_bytes: props.limits.max_compute_shared_memory_size,
            subgroup_size: (sub.subgroup_size > 0).then_some(sub.subgroup_size),
            unified_memory: matches!(
                props.device_type,
                vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::CPU
            ),
            workgroup_reductions: true, // real barrier semantics (SPIR-V)
            // Measured by `gpu_core::roof`, never reported by Vulkan.
            peak_bandwidth_gbs: None,
            peak_gflops: None,
            numeric: NumericSupport {
                int8_dot: ctx.prec.dp4a,
                coop_matrix: ctx.caps.feature_supported && !ctx.caps.shapes.is_empty(),
                ..NumericSupport::BASELINE
            },
        }
    }

    /// Allocate one descriptor set with `set_layout`, growing the pool list when
    /// the active pool is exhausted (never resets — see [`VulkanBackend::pools`]).
    fn alloc_set(&self, set_layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        self.stats.bind_groups.fetch_add(1, Ordering::Relaxed);
        let dev = &self.ctx.device;
        let set_layouts = [set_layout];
        let mut pools = self.pools.lock().unwrap_or_else(|e| e.into_inner());
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(*pools.last().unwrap())
            .set_layouts(&set_layouts);
        unsafe {
            match dev.allocate_descriptor_sets(&info) {
                Ok(s) => s[0],
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) | Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                    let pool = new_pool(dev).expect("grow descriptor pool");
                    pools.push(pool);
                    let info = vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&set_layouts);
                    dev.allocate_descriptor_sets(&info).expect("allocate from fresh pool")[0]
                }
                Err(e) => panic!("allocate_descriptor_sets: {e}"),
            }
        }
    }

    // ---- buffers ----

    pub fn storage(&self, n: u64) -> VkOwnedBuffer {
        let b = self.ctx.storage(n * 4, vk::BufferUsageFlags::empty());
        self.ctx.zero(&b); // zero-init like wgpu/CPU storage
        VkOwnedBuffer { inner: b, ctx: self.ctx.clone() }
    }

    pub fn storage_init(&self, _name: &str, data: &[f32]) -> VkOwnedBuffer {
        let b = self.ctx.storage((data.len() * 4).max(4) as u64, vk::BufferUsageFlags::empty());
        if data.is_empty() {
            self.ctx.zero(&b);
        } else {
            self.ctx.upload(&b, bytemuck::cast_slice(data)); // fully covers the buffer
        }
        VkOwnedBuffer { inner: b, ctx: self.ctx.clone() }
    }

    pub fn buffer(&self, _label: &str, size: u64, usage: BufUsage) -> VkOwnedBuffer {
        let extra = if usage.contains(BufUsage::UNIFORM) {
            vk::BufferUsageFlags::UNIFORM_BUFFER
        } else {
            vk::BufferUsageFlags::empty()
        };
        let b = self.ctx.storage(size.max(4), extra);
        self.ctx.zero(&b);
        VkOwnedBuffer { inner: b, ctx: self.ctx.clone() }
    }

    pub fn uniform_dynamic(&self, len: usize) -> VkOwnedBuffer {
        // Host-visible: `write` then updates it by direct map (no staging
        // submit), matching its purpose — a caller-owned uniform rewritten
        // every iteration of a hot loop.
        let size = ((len * 4).div_ceil(16) * 16).max(16) as u64;
        let b = self.ctx.storage_host(size, vk::BufferUsageFlags::UNIFORM_BUFFER);
        self.ctx.zero(&b);
        VkOwnedBuffer { inner: b, ctx: self.ctx.clone() }
    }

    pub fn write(&self, buf: &VkOwnedBuffer, data: &[u32]) {
        // Ensure prior recorded compute completes before a host write, matching
        // the wgpu backend's flush-before-write ordering.
        self.flush();
        self.ctx.upload(&buf.inner, bytemuck::cast_slice(data));
    }

    /// [`Self::write`] at a word offset — see `Backend::write_at`.
    pub fn write_at(&self, buf: &VkOwnedBuffer, offset_words: u64, data: &[u32]) {
        self.flush();
        self.ctx.upload_at(&buf.inner, bytemuck::cast_slice(data), offset_words * 4);
    }

    pub fn read(&self, buf: &VkOwnedBuffer, n: usize) -> Vec<f32> {
        self.stats.readbacks.fetch_add(1, Ordering::Relaxed);
        self.flush();
        let bytes = self.ctx.download(&buf.inner, n * 4);
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    }

    pub fn poll_wait(&self) {
        self.flush();
    }

    // ---- dispatch ----

    /// Build a dispatch with a fresh single-use uniform buffer (tracked transient).
    pub fn step(&self, kind: usize, bufs: &[&VkOwnedBuffer], params: &[u32], threads: u32) -> VkStep {
        let ubuf = self.make_uniform(params);
        let step = self.record(kind, &ubuf, bufs, &[], threads, true);
        self.uniforms.lock().unwrap_or_else(|e| e.into_inner()).push(ubuf);
        step
    }

    /// Build a dispatch around a caller-owned uniform buffer (reused across runs).
    pub fn step_buf(&self, kind: usize, ubuf: &VkOwnedBuffer, bufs: &[&VkOwnedBuffer], threads: u32) -> VkStep {
        self.record(kind, &ubuf.inner, bufs, &[], threads, false)
    }

    /// Build a dispatch where each storage buffer binds the sub-range
    /// `offsets[i] = (word_offset, word_len)` (`word_len == 0` => to end).
    pub fn step_sliced(
        &self,
        kind: usize,
        bufs: &[&VkOwnedBuffer],
        offsets: &[(u64, u64)],
        params: &[u32],
        threads: u32,
    ) -> VkStep {
        let ubuf = self.make_uniform(params);
        let step = self.record(kind, &ubuf, bufs, offsets, threads, true);
        self.uniforms.lock().unwrap_or_else(|e| e.into_inner()).push(ubuf);
        step
    }

    /// Total queue submissions (each a blocking submit + fence wait) since
    /// construction. Perf observability for `tests/perf_contract.rs`.
    pub fn queue_submits(&self) -> u64 {
        self.ctx.submits.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Transient uniform buffers currently alive (unsubmitted + in-flight +
    /// recycled). Bounded by the largest single frame, not by frame count.
    pub fn transient_uniform_count(&self) -> usize {
        self.uniforms.lock().unwrap_or_else(|e| e.into_inner()).len()
            + self.inflight_uniforms.lock().unwrap_or_else(|e| e.into_inner()).len()
            + self.free_uniforms.lock().unwrap_or_else(|e| e.into_inner()).values().map(Vec::len).sum::<usize>()
    }

    /// A transient per-dispatch uniform: recycled from `free_uniforms` when one
    /// of the right size is idle, else a fresh HOST-VISIBLE allocation. Both the
    /// zero (pad bytes) and the params write go through a direct map — building
    /// a dispatch performs NO queue submits. (The old DEVICE_LOCAL version cost
    /// a fill submit + a staged-copy submit — two blocking GPU round trips — per
    /// dispatch per frame, which serialized inference by orders of magnitude.)
    fn make_uniform(&self, params: &[u32]) -> VkBuffer {
        self.stats.uniform_allocs.fetch_add(1, Ordering::Relaxed);
        let size = ((params.len() * 4).div_ceil(16) * 16).max(16) as u64;
        let b = self
            .free_uniforms
            .lock()
            .unwrap()
            .get_mut(&size)
            .and_then(Vec::pop)
            .unwrap_or_else(|| self.ctx.storage_host(size, vk::BufferUsageFlags::UNIFORM_BUFFER));
        self.ctx.zero(&b); // pad bytes beyond `params` must be 0 (mapped memset)
        if !params.is_empty() {
            self.ctx.upload(&b, bytemuck::cast_slice(params)); // mapped memcpy
        }
        b
    }

    /// Allocate a descriptor set for `kind`, wire the uniform (binding 0) and the
    /// storage buffers (bindings 1..) — optionally sub-ranged — and return the step.
    fn record(
        &self,
        kind: usize,
        ubuf: &VkBuffer,
        bufs: &[&VkOwnedBuffer],
        offsets: &[(u64, u64)],
        threads: u32,
        transient: bool,
    ) -> VkStep {
        let kp = &self.pipelines.pipelines[kind];
        let dev = &self.ctx.device;
        // Transient sets recycle through `free_sets` (same pipeline => same
        // layout; the flush's fence wait made them idle, so rewriting below via
        // `update_descriptor_sets` is legal). Caller-held `step_buf` sets must
        // stay valid across flushes, so they always allocate fresh.
        let set = if transient {
            self.free_sets.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&kind).and_then(Vec::pop)
        } else {
            None
        }
        .unwrap_or_else(|| self.alloc_set(kp.set_layout));

        // Build the binding metadata first (binding 0 = uniform; the storage
        // bindings consume `bufs` in order), then materialise the buffer-info
        // slices in a fixed Vec so the `WriteDescriptorSet`s can borrow stable
        // addresses (no further pushes after this point).
        let mut meta: Vec<(u32, vk::DescriptorType, vk::DescriptorBufferInfo)> =
            Vec::with_capacity(kp.bindings.len());
        let mut storage_i = 0usize;
        for &(binding, is_uniform) in &kp.bindings {
            let (vkbuf, off_b, range_b, ty) = if is_uniform {
                (ubuf.buffer, 0u64, ubuf.size, vk::DescriptorType::UNIFORM_BUFFER)
            } else {
                let b = bufs[storage_i];
                let (off_w, len_w) = offsets.get(storage_i).copied().unwrap_or((0, 0));
                let off = off_w * 4;
                let range = if len_w == 0 { b.bytes() - off } else { len_w * 4 };
                storage_i += 1;
                (b.inner.buffer, off, range, vk::DescriptorType::STORAGE_BUFFER)
            };
            meta.push((
                binding,
                ty,
                vk::DescriptorBufferInfo::default().buffer(vkbuf).offset(off_b).range(range_b),
            ));
        }
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = meta.iter().map(|&(_, _, bi)| [bi]).collect();
        let writes: Vec<vk::WriteDescriptorSet> = meta
            .iter()
            .zip(infos.iter())
            .map(|(&(binding, ty, _), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(ty)
                    .buffer_info(info)
            })
            .collect();
        unsafe { dev.update_descriptor_sets(&writes, &[]) };
        // The set now names these raw handles, and will keep naming them until
        // it is retired (`recycle_transients`) or, for a caller-held
        // `step_buf` set, for as long as this backend lives. Registering HERE
        // rather than at `submit` is the whole point: a buffer dropped between
        // building a step and submitting it must not be destroyed - see
        // `VkContext::reclaim_dead`. The uniform is included because
        // `uniform_dynamic` hands the caller an ownable buffer that can be
        // dropped just like a storage one.
        let named: Vec<vk::Buffer> =
            std::iter::once(ubuf.buffer).chain(bufs.iter().map(|b| b.inner.buffer)).collect();
        self.ctx.set_names(set, &named);
        let (gx, gy) = backend_api::grid_ws(threads, self.pipelines.wgsizes[kind]);
        let sliced = offsets.iter().any(|&(off, _)| off > 0);
        VkStep { kind, set, gx, gy, sliced, transient }
    }

    pub fn submit(&self, clears: &[&VkOwnedBuffer], steps: &[VkStep]) {
        self.stats.submits.fetch_add(1, Ordering::Relaxed);
        self.stats.dispatches.fetch_add(steps.len() as u64, Ordering::Relaxed);
        if !clears.is_empty() {
            // Match wgpu: complete prior work, then zero the buffers, before the
            // new steps (which may read them) are queued.
            self.flush();
            self.run_clears(clears);
        }
        // Recorded but not yet submitted: until these run, the buffers they
        // name must not be destroyed even if their Rust owners drop (see
        // `VkContext::reclaim_dead`).
        self.ctx.steps_recorded(steps.len() as u64);
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).extend_from_slice(steps);
        // These steps' transient uniforms are now in flight: eligible for
        // recycling once the flush that runs them has fence-waited. Uniforms of
        // steps NOT yet submitted stay in `uniforms`, untouched by a flush that
        // races between their creation and their own submit.
        self.inflight_uniforms.lock().unwrap_or_else(|e| e.into_inner()).append(&mut self.uniforms.lock().unwrap_or_else(|e| e.into_inner()));
    }

    fn run_clears(&self, clears: &[&VkOwnedBuffer]) {
        let dev = &self.ctx.device;
        unsafe {
            let (cmd, guard) = self.begin_cmd();
            for c in clears {
                dev.cmd_fill_buffer(cmd, c.inner.buffer, 0, c.inner.size, 0);
            }
            self.end_and_wait(cmd, guard);
        }
    }

    /// Record all pending dispatches into one command buffer (with a storage
    /// memory barrier between consecutive dispatches), submit, fence-wait, then
    /// reclaim the batch's descriptor sets + transient uniform buffers.
    fn flush(&self) {
        let steps: Vec<VkStep> = std::mem::take(&mut *self.pending.lock().unwrap_or_else(|e| e.into_inner()));
        if steps.is_empty() {
            // This handle has nothing recorded; `reclaim_dead` still checks the
            // DEVICE-wide count before freeing. It is the only reclaim point a
            // caller that allocates and drops without ever dispatching reaches.
            self.ctx.reclaim_dead();
            return;
        }
        let dev = &self.ctx.device;
        // Serialize (submit+fence per dispatch) when the batch contains a sliced
        // (sub-range) binding: **Intel ANV's** compute-compute pipeline barrier
        // does not reliably make a prior dispatch's writes visible across a
        // non-zero descriptor offset, but a queue-submit/fence boundary does. This
        // is an Intel driver bug — on other vendors the standard memory barrier is
        // correct, and serializing there is pure waste (a per-frame model with a
        // vocab-tiled embedding does ~one submit+fence per *dispatch* instead of
        // one per *frame*: on an NVIDIA P40 the TTS Talker forward measured slower,
        // not faster).
        // Gate the workaround to Intel (vendor 0x8086). `BRAIN_VK_SERIAL` forces
        // it everywhere (diagnostic); `BRAIN_VK_NO_SERIAL` disables it (to confirm
        // the Intel bug on that hardware).
        const VENDOR_INTEL: u32 = 0x8086;
        let force_serial = std::env::var("BRAIN_VK_SERIAL").is_ok();
        let no_serial = std::env::var("BRAIN_VK_NO_SERIAL").is_ok();
        let vendor_needs_serial = self.ctx.vendor_id == VENDOR_INTEL;
        if !no_serial && (force_serial || (vendor_needs_serial && steps.iter().any(|s| s.sliced))) {
            let time_this = self.timing_active();
            unsafe {
                for s in &steps {
                    let (cmd, guard) = self.begin_cmd();
                    let qp = time_this.then(|| self.timestamp_pool());
                    if let Some(qp) = qp {
                        dev.cmd_reset_query_pool(cmd, qp, 0, 2);
                        dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
                    }
                    let kp = &self.pipelines.pipelines[s.kind];
                    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kp.pipeline);
                    dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, kp.layout, 0, &[s.set], &[]);
                    dev.cmd_dispatch(cmd, s.gx, s.gy, 1);
                    if let Some(qp) = qp {
                        dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, qp, 1);
                    }
                    self.end_and_wait(cmd, guard);
                    // Each dispatch here already pays its own submit+fence, so
                    // timing it individually (unlike the batched branch below)
                    // costs nothing extra and distorts nothing — the isolation
                    // this branch exists for is already the unit being timed.
                    if let Some(qp) = qp {
                        let ts = self.read_timestamps(qp, 2);
                        self.record_timing(&[s.kind], &ts);
                    }
                }
            }
            self.recycle_transients(&steps);
            return;
        }
        // Only time a batch this backend can actually bracket in one query
        // pool — an oversized batch silently skips timing rather than
        // truncating or panicking (`kernel_times` stays honest: it reports
        // what it measured, not a guess for what it didn't).
        let time_this_batch = self.timing_active() && steps.len() < MAX_TIMED_DISPATCHES;
        let query_pool = time_this_batch.then(|| unsafe { self.timestamp_pool() });
        unsafe {
            let (cmd, guard) = self.begin_cmd();
            if let Some(qp) = query_pool {
                dev.cmd_reset_query_pool(cmd, qp, 0, (steps.len() + 1) as u32);
                dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
            }
            // Conservative full memory barrier between dependent dispatches: every
            // prior shader write is made available/visible to subsequent shader
            // reads/writes. (A finer per-buffer barrier is a later optimisation.)
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            for (i, s) in steps.iter().enumerate() {
                if i > 0 {
                    dev.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[barrier],
                        &[],
                        &[],
                    );
                }
                let kp = &self.pipelines.pipelines[s.kind];
                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kp.pipeline);
                dev.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    kp.layout,
                    0,
                    &[s.set],
                    &[],
                );
                dev.cmd_dispatch(cmd, s.gx, s.gy, 1);
                if let Some(qp) = query_pool {
                    dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, qp, (i + 1) as u32);
                }
            }
            self.end_and_wait(cmd, guard);
        }
        if let Some(qp) = query_pool {
            let ts = unsafe { self.read_timestamps(qp, (steps.len() + 1) as u32) };
            let kinds: Vec<usize> = steps.iter().map(|s| s.kind).collect();
            self.record_timing(&kinds, &ts);
        }
        self.recycle_transients(&steps);
    }

    /// Bytes currently buried (dropped, not yet `vkFreeMemory`'d) on this
    /// device - see [`VkContext::buried_bytes`]'s doc.
    pub fn buried_bytes(&self) -> u64 {
        self.ctx.buried_bytes()
    }

    /// Count of actual reclaim events on this device - see
    /// [`VkContext::reclaim_event_count`]'s doc.
    pub fn reclaim_event_count(&self) -> u64 {
        self.ctx.reclaim_event_count()
    }

    /// Recycle the flushed batch's TRANSIENT resources — the fence wait in
    /// `end_and_wait` has just proven them idle. Uniforms go back to the
    /// size-keyed pool, descriptor sets to the per-pipeline pool (deduped: the
    /// same step submitted twice in one batch must not donate its set twice).
    /// `step_buf` steps (transient = false) are caller-owned and left alone, so
    /// the `uniform_dynamic` reuse pattern keeps working across flushes.
    fn recycle_transients(&self, steps: &[VkStep]) {
        // These dispatches have now run to completion (the caller's
        // `end_and_wait` fence-waited), so they no longer name anything. The
        // count is dropped HERE rather than when the batch left the pending
        // list, so it never reads zero while this batch's buffers are in use.
        self.ctx.steps_submitted(steps.len() as u64);
        for u in std::mem::take(&mut *self.inflight_uniforms.lock().unwrap_or_else(|e| e.into_inner())) {
            self.free_uniforms.lock().unwrap_or_else(|e| e.into_inner()).entry(u.size).or_default().push(u);
        }
        {
            let mut seen = std::collections::HashSet::new();
            let mut free = self.free_sets.lock().unwrap_or_else(|e| e.into_inner());
            for s in steps {
                if s.transient && seen.insert(s.set) {
                    // Idle and about to be rewritten before any reuse, so it
                    // stops pinning the buffers it named - which is what lets
                    // the reclaim below actually free this batch's scratch.
                    self.ctx.set_released(s.set);
                    free.entry(s.kind).or_default().push(s.set);
                }
            }
        }
        // Buffers dropped while this batch was recorded: with nothing left
        // recorded anywhere and this batch's sets released just above, this is
        // where they are actually destroyed (see `impl Drop for
        // VkOwnedBuffer`). Strictly after the release, or every buffer this
        // batch touched would still read as referenced and stay buried an
        // extra flush.
        self.ctx.reclaim_dead();
    }

    /// Whether this flush should record per-dispatch timestamps: profiling is
    /// on AND this queue family can actually write them.
    fn timing_active(&self) -> bool {
        self.profile.enabled.load(std::sync::atomic::Ordering::Relaxed) && self.ctx.timestamp_valid_bits != 0
    }

    /// The reusable timestamp query pool, created on first use, sized for
    /// `MAX_TIMED_DISPATCHES + 1` marks (enough to bracket the largest batch
    /// this backend will time).
    unsafe fn timestamp_pool(&self) -> vk::QueryPool {
        let mut slot = self.profile.pool.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = *slot {
            return p;
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count((MAX_TIMED_DISPATCHES + 1) as u32);
        let pool = self.ctx.device.create_query_pool(&info, None).expect("create_query_pool");
        *slot = Some(pool);
        pool
    }

    /// Block for and read back `count` timestamps. Only called after
    /// `end_and_wait` has already fence-waited the command buffer that wrote
    /// them, so `WAIT` here is a formality, not an extra stall.
    unsafe fn read_timestamps(&self, pool: vk::QueryPool, count: u32) -> Vec<u64> {
        let mut buf = vec![0u64; count as usize];
        self.ctx
            .device
            .get_query_pool_results(pool, 0, &mut buf, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
            .expect("get_query_pool_results");
        buf
    }

    /// Fold `n+1` bracketing timestamps for `n` dispatches (`kinds[i]` ran
    /// between `ts[i]` and `ts[i+1]`) into the per-kernel-kind accumulator.
    fn record_timing(&self, kinds: &[usize], ts: &[u64]) {
        let period_ns = self.ctx.timestamp_period_ns;
        let mut acc = self.profile.acc.lock().unwrap_or_else(|e| e.into_inner());
        for (i, &kind) in kinds.iter().enumerate() {
            let dt_ns = ts[i + 1].saturating_sub(ts[i]) as f64 * period_ns;
            let entry = &mut acc[kind];
            entry.0 += dt_ns / 1e6;
            entry.1 += 1;
        }
    }

    /// Begins recording and returns the lock guarding `ctx.queue`/
    /// `ctx.command_pool` for the whole record-submit-wait-free sequence —
    /// held from here until [`VulkanBackend::end_and_wait`] drops it. Both
    /// are shared, non-thread-safe Vulkan objects once `share()`/`new_like()`
    /// hand out a second live handle onto the same `ctx` (see
    /// `VkContext::queue_lock`'s doc comment).
    unsafe fn begin_cmd(&self) -> (vk::CommandBuffer, std::sync::MutexGuard<'_, ()>) {
        let guard = self.ctx.queue_guard();
        let dev = &self.ctx.device;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = dev.allocate_command_buffers(&alloc).expect("alloc cmd")[0];
        dev.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .expect("begin cmd");
        (cmd, guard)
    }

    unsafe fn end_and_wait(&self, cmd: vk::CommandBuffer, _guard: std::sync::MutexGuard<'_, ()>) {
        self.ctx.submits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dev = &self.ctx.device;
        dev.end_command_buffer(cmd).expect("end cmd");
        let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).expect("fence");
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        dev.queue_submit(self.ctx.queue, &[submit], fence).expect("queue_submit");
        let timeout_ns = gpu_wait_timeout_ns();
        match dev.wait_for_fences(&[fence], true, timeout_ns) {
            Ok(()) => {}
            // Do NOT silently retry -- report which submit wedged rather than
            // masking it with another indefinite wait.
            Err(vk::Result::TIMEOUT) => panic!(
                "GPU submit did not complete within {:.1}s (BRAIN_GPU_WAIT_S) -- \
                 device likely wedged",
                timeout_ns as f64 / 1e9
            ),
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                panic!("GPU device lost while waiting for a submit to complete")
            }
            Err(e) => panic!("wait_for_fences: {e:?}"),
        }
        dev.destroy_fence(fence, None);
        dev.free_command_buffers(self.ctx.command_pool, &cmds);
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        let dev = &self.ctx.device;
        unsafe {
            let _ = dev.device_wait_idle();
            for u in std::mem::take(&mut *self.uniforms.lock().unwrap_or_else(|e| e.into_inner())) {
                self.ctx.destroy_buffer(u);
            }
            for u in std::mem::take(&mut *self.inflight_uniforms.lock().unwrap_or_else(|e| e.into_inner())) {
                self.ctx.destroy_buffer(u);
            }
            for (_, us) in std::mem::take(&mut *self.free_uniforms.lock().unwrap_or_else(|e| e.into_inner())) {
                for u in us {
                    self.ctx.destroy_buffer(u);
                }
            }
            for &pool in self.pools.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                dev.destroy_descriptor_pool(pool, None);
            }
            // Pipelines are NOT destroyed here: `self.pipelines` is an
            // `Arc<VkPipelineSet>`, possibly shared with a `share()` sibling
            // still alive. `VkPipelineSet::drop` destroys them exactly once,
            // when the last handle referencing this kernel set drops.
            if let Some(qp) = self.profile.pool.lock().unwrap_or_else(|e| e.into_inner()).take() {
                dev.destroy_query_pool(qp, None);
            }
        }
        // Model storage buffers (host-visible) are reclaimed by the OS on process
        // exit; a one-shot CLI never frees them mid-run.
    }
}

/// Create a fresh descriptor pool sized for `POOL_MAX_SETS` sets (uniform + up to
/// 8 storage buffers each). Pools are grown, never reset (see
/// [`VulkanBackend::pools`]).
unsafe fn new_pool(dev: &ash::Device) -> Result<vk::DescriptorPool, String> {
    let sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(POOL_MAX_SETS),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(POOL_MAX_SETS * 8),
    ];
    dev.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::default().max_sets(POOL_MAX_SETS).pool_sizes(&sizes),
        None,
    )
    .map_err(|e| format!("descriptor pool: {e}"))
}

fn log_adapter(name: &str) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| eprintln!("adapter: {name} (Vulkan compute, ash + naga WGSL->SPIR-V)"));
}

/// A weak handle onto a Vulkan device's shared state (`ctx` via the kernel
/// set's own `Arc`, plus the kernel set itself): keeps neither alive.
/// `upgrade()` reconstructs a real, fully-functional handle sharing both, iff
/// something else still holds them live — the same "the device dies with its
/// last real handle, never at process exit" contract `WeakWgpu` implements
/// for the wgpu backend, so `gpu_core::testgpu::dev`'s pool can track a
/// Vulkan device without itself keeping it alive.
struct WeakVulkan {
    pipelines: std::sync::Weak<VkPipelineSet>,
    caps: backend_api::DeviceCaps,
    max_storage_binding: u64,
}

// SAFETY: mirrors `VulkanBackend`'s own `unsafe impl Send/Sync` just below —
// the ash handles reachable through `Weak<VkPipelineSet>` are themselves
// Send+Sync (thin wrappers over a function-pointer table + handle), and this
// type holds no interior-mutable state of its own to race on.
unsafe impl Send for WeakVulkan {}
unsafe impl Sync for WeakVulkan {}

impl backend_api::WeakBackend for WeakVulkan {
    fn upgrade(&self) -> Option<Box<dyn Backend>> {
        let pipelines = self.pipelines.upgrade()?;
        let ctx = pipelines.ctx.clone();
        let backend = VulkanBackend::from_shared(ctx, pipelines, self.caps.clone(), self.max_storage_binding)
            .unwrap_or_else(|e| panic!("vulkan: upgrading a shared device handle: {e}"));
        Some(Box::new(backend))
    }
}

/// Neutral-handle bridge: downcast the opaque [`DeviceBuffer`]/[`Step`] back to
/// `VkOwnedBuffer`/[`VkStep`] and delegate to the inherent methods.
impl Backend for VulkanBackend {
    fn kind(&self) -> &'static str {
        "vulkan"
    }
    /// A second handle onto the SAME device (instance/device/queue) and the
    /// SAME compiled pipelines, with its own command stream — the Vulkan
    /// sibling of `WgpuBackend::share_device`. Before this existed, this
    /// method fell through to the trait's `None` default and every caller of
    /// `Gpu::share`/`gpu_core::testgpu::dev` on the Vulkan backend silently
    /// built a whole new device instead of truly sharing one (see
    /// [`VkPipelineSet`]'s doc comment).
    fn share(&self) -> Option<Box<dyn Backend>> {
        let backend = Self::from_shared(self.ctx.clone(), self.pipelines.clone(), self.caps.clone(), self.max_storage_binding)
            .unwrap_or_else(|e| panic!("vulkan share: {e}"));
        Some(Box::new(backend))
    }
    /// A handle for a DIFFERENT kernel set on the SAME device — compiles
    /// fresh pipelines against the existing `ctx` rather than building a new
    /// `VkContext` (new instance + device + queue). This is the fix for the
    /// "many concurrent Vulkan devices deadlock the driver" class: before
    /// this existed, `new_like` fell through to the trait's `None` default,
    /// and `Gpu::new_like` (the caller's contract on that `None`) built an
    /// entirely separate device as a "fallback" that was actually the
    /// dangerous path, not a safe one.
    fn new_like(&self, kernels: &[(&str, &str)]) -> Option<Box<dyn Backend>> {
        let pipelines = Self::compile_pipeline_set(self.ctx.clone(), kernels)
            .unwrap_or_else(|e| panic!("vulkan new_like: {e}"));
        let backend = Self::from_shared(self.ctx.clone(), Arc::new(pipelines), self.caps.clone(), self.max_storage_binding)
            .unwrap_or_else(|e| panic!("vulkan new_like: {e}"));
        Some(Box::new(backend))
    }
    fn downgrade(&self) -> Option<Box<dyn backend_api::WeakBackend>> {
        Some(Box::new(WeakVulkan {
            pipelines: Arc::downgrade(&self.pipelines),
            caps: self.caps.clone(),
            max_storage_binding: self.max_storage_binding,
        }))
    }
    /// Device-op accounting — the same counters the other two backends report,
    /// so "how many submits/readbacks did this run cost" has a real answer on
    /// every backend rather than `None` on this one.
    fn stats(&self) -> Option<backend_api::DeviceStats> {
        Some(backend_api::DeviceStats {
            submits: self.stats.submits.load(Ordering::Relaxed),
            dispatches: self.stats.dispatches.load(Ordering::Relaxed),
            readbacks: self.stats.readbacks.load(Ordering::Relaxed),
            bind_groups: self.stats.bind_groups.load(Ordering::Relaxed),
            uniform_allocs: self.stats.uniform_allocs.load(Ordering::Relaxed),
        })
    }

    /// Per-kernel-kind device timing via `vkCmdWriteTimestamp`, mirroring
    /// `backend-wgpu`'s contract. Returns `false` (timing NOT enabled) when
    /// this device's compute queue cannot write timestamps at all — the
    /// caller then knows to expect `kernel_times() == None`, never a
    /// silently-substituted host time.
    fn set_kernel_timing(&self, on: bool) -> bool {
        if self.ctx.timestamp_valid_bits == 0 {
            return false;
        }
        self.profile.enabled.store(on, Ordering::Relaxed);
        true
    }

    fn kernel_times(&self) -> Option<Vec<(String, f64, u64)>> {
        if self.ctx.timestamp_valid_bits == 0 {
            return None;
        }
        let acc = self.profile.acc.lock().unwrap_or_else(|e| e.into_inner());
        Some(
            self.names
                .iter()
                .zip(acc.iter())
                .filter(|(_, (_, calls))| *calls > 0)
                .map(|(name, (ms, calls))| (name.clone(), *ms, *calls))
                .collect(),
        )
    }

    fn reset_kernel_times(&self) {
        for e in self.profile.acc.lock().unwrap_or_else(|e| e.into_inner()).iter_mut() {
            *e = (0.0, 0);
        }
    }

    fn dump_profile(&self) {
        let Some(mut rows) = self.kernel_times() else {
            eprintln!("=== GPU kernel time (BRAIN_PROFILE, vulkan) === unavailable: this queue cannot write timestamps");
            return;
        };
        let total: f64 = rows.iter().map(|(_, ms, _)| ms).sum();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("=== GPU kernel time (BRAIN_PROFILE, timestamp queries, total {total:.1} ms) ===");
        for (name, ms, calls) in &rows {
            let pct = if total > 0.0 { 100.0 * ms / total } else { 0.0 };
            eprintln!("  {name:<28} {ms:8.1} ms  {calls:6} calls  ({pct:4.1}%)");
        }
    }

    fn caps(&self) -> backend_api::DeviceCaps {
        self.caps.clone()
    }
    fn max_storage_binding_bytes(&self) -> u64 {
        self.max_storage_binding
    }
    fn storage(&self, n: u64) -> DeviceBuffer {
        DeviceBuffer::new(VulkanBackend::storage(self, n))
    }
    fn storage_init(&self, name: &str, data: &[f32]) -> DeviceBuffer {
        DeviceBuffer::new(VulkanBackend::storage_init(self, name, data))
    }
    fn buffer(&self, label: &str, size: u64, usage: BufUsage) -> DeviceBuffer {
        DeviceBuffer::new(VulkanBackend::buffer(self, label, size, usage))
    }
    fn uniform_dynamic(&self, len: usize) -> DeviceBuffer {
        DeviceBuffer::new(VulkanBackend::uniform_dynamic(self, len))
    }
    fn write(&self, buf: &DeviceBuffer, data: &[u32]) {
        VulkanBackend::write(self, buf.downcast_ref::<VkOwnedBuffer>(), data)
    }
    fn write_at(&self, buf: &DeviceBuffer, offset_words: u64, data: &[u32]) {
        VulkanBackend::write_at(self, buf.downcast_ref::<VkOwnedBuffer>(), offset_words, data)
    }
    fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step {
        let bs: Vec<&VkOwnedBuffer> = bufs.iter().map(|b| b.downcast_ref::<VkOwnedBuffer>()).collect();
        Step::new(VulkanBackend::step(self, kind, &bs, params, threads))
    }
    fn step_sliced(&self, kind: usize, bufs: &[&DeviceBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> Step {
        let bs: Vec<&VkOwnedBuffer> = bufs.iter().map(|b| b.downcast_ref::<VkOwnedBuffer>()).collect();
        Step::new(VulkanBackend::step_sliced(self, kind, &bs, offsets, params, threads))
    }
    fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step {
        let bs: Vec<&VkOwnedBuffer> = bufs.iter().map(|b| b.downcast_ref::<VkOwnedBuffer>()).collect();
        Step::new(VulkanBackend::step_buf(self, kind, ubuf.downcast_ref::<VkOwnedBuffer>(), &bs, threads))
    }
    fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]) {
        let cs: Vec<&VkOwnedBuffer> = clears.iter().map(|b| b.downcast_ref::<VkOwnedBuffer>()).collect();
        let ss: Vec<VkStep> = steps.iter().map(|s| *s.downcast_ref::<VkStep>()).collect();
        VulkanBackend::submit(self, &cs, &ss);
    }
    fn read(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
        VulkanBackend::read(self, buf.downcast_ref::<VkOwnedBuffer>(), n)
    }
    fn poll_wait(&self) {
        VulkanBackend::poll_wait(self)
    }
    fn flush(&self) {
        // This backend's flush fence-waits (its batch submission is
        // synchronous by design), so there is no overlap win here — but the
        // semantics ("all recorded work reaches the device") hold.
        VulkanBackend::flush(self);
    }
    fn buried_bytes(&self) -> u64 {
        VulkanBackend::buried_bytes(self)
    }
    fn queue_submits(&self) -> u64 {
        VulkanBackend::queue_submits(self)
    }
    fn reclaim_event_count(&self) -> u64 {
        VulkanBackend::reclaim_event_count(self)
    }
}

/// Register this backend under `"vulkan"`. The factory returns `Err` when no
/// Vulkan device/ICD is present, so the facade can fall back to wgpu.
pub fn register() {
    backend_api::register_backend("vulkan", |kernels| {
        VulkanBackend::try_new(kernels).map(|g| Box::new(g) as Box<dyn Backend>)
    });
}

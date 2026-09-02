// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `VkContext`: a minimal native-Vulkan (ash) compute runtime that mirrors the
//! conceptual API of `crate::gpu::Gpu` (storage buffers, dispatch, read-back),
//! plus the capability query for `VK_KHR_cooperative_matrix`.
//!
//! Scope: enough plumbing to bring up a single compute kernel (the matmul) end
//! to end -- instance, physical-device selection + cooperative-matrix feature/
//! shape query, logical device + compute queue, descriptor set layout/pool,
//! command-buffer record/submit/fence-wait, and host-visible buffer upload/
//! download. The full PID forward port onto this runtime is a documented
//! follow-up (see README_VULKAN.md); this is deliberately the matmul + runtime
//! + capability + fallback slice only.
//!
//! SAFETY / VALIDATION: this machine has only software Vulkan (llvmpipe), which
//! does not expose cooperative matrix. The code is correct-by-construction and
//! must be validated on NVIDIA hardware (Turing sm_75+ for f16 tensor cores;
//! Pascal sm_61 will report no cooperative-matrix support and take the scalar
//! fallback). Everything here compiles cleanly under
//! `cargo check --features vulkan-coopmat`.

use std::ffi::{CStr, CString};

use ash::vk;

/// Debug-messenger callback for `BRAIN_VK_VALIDATE`: prints validation /
/// synchronization-hazard messages to stderr.
unsafe extern "system" fn vk_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if !data.is_null() && !(*data).p_message.is_null() {
        let msg = CStr::from_ptr((*data).p_message).to_string_lossy();
        eprintln!("[VK {severity:?}] {msg}");
    }
    vk::FALSE
}

/// One supported cooperative-matrix shape, decoded from
/// `VkCooperativeMatrixPropertiesKHR` into something printable / matchable.
#[derive(Clone, Copy, Debug)]
pub struct CoopMatShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: vk::ComponentTypeKHR,
    pub b_type: vk::ComponentTypeKHR,
    pub c_type: vk::ComponentTypeKHR,
    pub result_type: vk::ComponentTypeKHR,
    pub saturating_accumulation: bool,
    pub scope: vk::ScopeKHR,
}

impl CoopMatShape {
    /// True for the f16 x f16 -> f32 accumulate shape the GLSL kernel authors
    /// (subgroup scope). This is the shape `matmul_coopmat.comp` targets.
    pub fn is_f16_f16_f32(&self) -> bool {
        self.a_type == vk::ComponentTypeKHR::FLOAT16
            && self.b_type == vk::ComponentTypeKHR::FLOAT16
            && self.c_type == vk::ComponentTypeKHR::FLOAT32
            && self.result_type == vk::ComponentTypeKHR::FLOAT32
            && self.scope == vk::ScopeKHR::SUBGROUP
    }

    /// True for the i8 x i8 -> i32 accumulate shape (documented integer variant).
    #[allow(dead_code)]
    pub fn is_i8_i8_i32(&self) -> bool {
        self.a_type == vk::ComponentTypeKHR::SINT8
            && self.b_type == vk::ComponentTypeKHR::SINT8
            && self.c_type == vk::ComponentTypeKHR::SINT32
            && self.result_type == vk::ComponentTypeKHR::SINT32
            && self.scope == vk::ScopeKHR::SUBGROUP
    }
}

/// Capability report for the selected adapter.
pub struct CoopMatCaps {
    /// `VkPhysicalDeviceCooperativeMatrixFeaturesKHR::cooperativeMatrix`.
    pub feature_supported: bool,
    /// Whether the `VK_KHR_cooperative_matrix` device extension is present.
    pub extension_present: bool,
    /// All enumerated supported shapes (empty if unsupported).
    pub shapes: Vec<CoopMatShape>,
}

impl CoopMatCaps {
    /// The kernel can run on tensor cores iff the feature is on AND at least one
    /// queried shape matches the authored f16xf16->f32 tile semantics.
    pub fn supports_f16_tensorcore(&self) -> bool {
        self.feature_supported && self.shapes.iter().any(|s| s.is_f16_f16_f32())
    }
}

/// Picks which enumerated physical device to bind, by index — how
/// `backend-vulkan` resolves a card by canonical identity rather than position.
///
/// The `'a` is load-bearing: a bare `dyn Fn` in a type alias defaults to
/// `+ 'static`, which would force every caller's closure to outlive the
/// process — the elided lifetime of the `&dyn Fn` argument this replaced.
pub type PhysicalDeviceSelect<'a> =
    dyn Fn(&ash::Instance, &[vk::PhysicalDevice]) -> Result<usize, String> + 'a;

// Some fields (entry, physical_device, queue_family_index) are retained for
// ownership/lifetime and for the documented follow-up forward port; they are
// not all read by the matmul slice yet.
#[allow(dead_code)]
pub struct VkContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub adapter_name: String,
    /// PCI vendor id (0x10de NVIDIA, 0x8086 Intel, 0x1002 AMD). Used to gate
    /// vendor-specific workarounds (e.g. the Intel-ANV sliced-binding serialize).
    pub vendor_id: u32,
    /// Nanoseconds per `vkCmdWriteTimestamp` tick (`VkPhysicalDeviceLimits::
    /// timestampPeriod`). May be `0.0` on a device that reports no meaningful
    /// value — `timestamp_valid_bits` is the field that actually gates whether
    /// timestamps are usable at all.
    pub timestamp_period_ns: f64,
    /// Valid bits in a timestamp written by the compute queue family this
    /// context uses (`VkQueueFamilyProperties::timestampValidBits`). `0` means
    /// this queue family cannot write timestamps — per-kernel device timing
    /// must degrade to unavailable, never substitute host time.
    pub timestamp_valid_bits: u32,
    pub caps: CoopMatCaps,
    /// Which non-fp32 arithmetic the logical device was created able to execute
    /// (queried from the physical device, enabled at `create_device` when
    /// present). The peak-throughput bench uses these to skip a precision the
    /// hardware/driver does not expose rather than fail pipeline creation.
    pub prec: PrecisionCaps,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    /// Reusable host-visible staging buffer for device-local up/downloads (grown
    /// on demand; transfers are fence-serialized so a single buffer suffices).
    staging: std::sync::Mutex<Option<VkBuffer>>,
    /// Buffers whose Rust owner has dropped, waiting for a moment the device is
    /// provably done with them - see [`Self::bury`] / [`Self::reclaim_dead`].
    dead: std::sync::Mutex<Vec<VkBuffer>>,
    /// Count of [`Self::reclaim_dead`] calls that actually freed something -
    /// see [`Self::reclaim_event_count`]'s doc.
    reclaim_events: std::sync::atomic::AtomicU64,
    /// Dispatches recorded against THIS DEVICE by any backend handle and not
    /// yet submitted. Device-wide rather than per-handle because
    /// `Backend::share`/`new_like` hand out sibling handles with their own
    /// command streams, and a buffer dropped by one of them may still be named
    /// by a dispatch another has recorded - so "is anything recorded anywhere"
    /// is the question a safe reclaim has to answer.
    pending_steps: std::sync::atomic::AtomicU64,
    /// Raw `vk::Buffer` handles each LIVE descriptor set currently names.
    ///
    /// [`Self::pending_steps`] alone cannot answer "is this buffer still
    /// referenced": it counts dispatches that reached `submit`, but a
    /// descriptor set is written (and starts naming raw buffer handles that
    /// outlive their Rust owner) at *record* time, which is strictly earlier.
    /// A caller that records a batch of steps, drops a scratch buffer, and
    /// only then submits - or that holds a `step_buf` step across flushes -
    /// leaves the counter at zero with live sets still naming the buffer, so
    /// `reclaim_dead` destroyed memory a queued dispatch went on to read.
    /// Keyed by set rather than counted globally so a caller-held step pins
    /// only the buffers IT names, not every buffer on the device.
    set_refs: std::sync::Mutex<std::collections::HashMap<vk::DescriptorSet, Vec<vk::Buffer>>>,
    /// Total queue submissions issued through this context (each is a blocking
    /// submit + fence wait). Perf observability: inference must keep this O(1)
    /// per frame, not O(dispatches) — see `backend-vulkan/tests/perf_contract.rs`.
    pub submits: std::sync::atomic::AtomicU64,
    /// Guards every use of `queue` and `command_pool`: the Vulkan spec requires
    /// host access to a `VkQueue` (`vkQueueSubmit`) AND to a `VkCommandPool`
    /// (`vkAllocateCommandBuffers`/recording/`vkFreeCommandBuffers`) to be
    /// EXTERNALLY synchronized — concurrent calls from two threads are
    /// undefined behaviour, not merely a data race Rust's type system would
    /// catch (both are `Copy` handles, so nothing stops two threads compiling
    /// cleanly). `VulkanBackend::share`/`new_like` (backend-vulkan) hand out
    /// exactly this shape — several live handles onto one `ctx`, each with its
    /// own command-stream STATE but the SAME underlying queue/pool — so every
    /// call site that touches `queue` or `command_pool` must hold this lock
    /// for its whole allocate/reset-record-submit sequence. Confirmed load-
    /// bearing, not theoretical: omitting it reproduced a real
    /// `ERROR_DEVICE_LOST` on P40 hardware within seconds of 8 threads sharing
    /// one device (`crates/gpu-core/tests/device_sharing.rs`'s
    /// `concurrent_shared_handles_do_not_deadlock`).
    ///
    /// **No longer implies a synchronous fence wait.** This lock used to be
    /// held across the WHOLE submit+fence-wait+free sequence, which is what
    /// made every submit here synchronous and un-pipelined regardless of this
    /// lock's own purpose (host-race prevention). `backend-vulkan`'s
    /// `VulkanBackend::flush_async` (M6.2) now releases this lock immediately
    /// after `vkQueueSubmit` returns, signals [`Self::timeline`] instead of a
    /// one-shot `VkFence`, and only calls [`Self::timeline_wait`] - outside
    /// this lock, since a semaphore wait touches neither `queue` nor
    /// `command_pool` - when a caller actually needs the result (`read`) or
    /// when a reused command-buffer slot must be proven idle before it is
    /// re-recorded. GPU execution order was always strictly FIFO per queue
    /// (Vulkan's own submission-order guarantee, unrelated to this lock); what
    /// changes is that the HOST no longer blocks between back-to-back
    /// `submit()`+`flush()` cycles that never read the result in between.
    pub queue_lock: std::sync::Mutex<()>,
    /// Timeline semaphore backing asynchronous submission (`backend-vulkan`'s
    /// N-submissions-in-flight scheme, M6.2). `None` when the device does not
    /// report the `timelineSemaphore` feature (defensive fallback only - every
    /// driver this workspace has actually run on, including this box's P40 at
    /// Vulkan 1.3, reports it; Vulkan 1.2's own conformance requirements
    /// mandate it). Monotonically increasing: a submission signals
    /// [`Self::timeline_next`]'s returned value, and [`Self::timeline_wait`]
    /// blocks until the semaphore's counter reaches a given value.
    timeline: Option<vk::Semaphore>,
    /// Next value a submission through [`Self::timeline`] should signal.
    /// Starts at 1 - the semaphore's own initial value is 0, and a value of 0
    /// would be indistinguishable from "never submitted" in a fresh command-
    /// buffer-ring slot.
    timeline_counter: std::sync::atomic::AtomicU64,
}

/// Non-fp32 arithmetic the device exposes and this context enabled. fp32 is
/// always available and needs no flag.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrecisionCaps {
    /// `shaderFloat64` (core) — WGSL `f64` arithmetic. NVIDIA: true (1/32 rate).
    pub f64: bool,
    /// `shaderFloat16` (VK_KHR_shader_float16_int8 / Vulkan 1.2) — WGSL `f16`.
    /// P40: exposed via the extension even though the base 1.0 bit is false.
    pub f16: bool,
    /// `shaderIntegerDotProduct` (Vulkan 1.3) + a `4x8BitPacked*Accelerated`
    /// device property — WGSL `dot4I8Packed` (DP4A). P40: accelerated.
    pub dp4a: bool,
}

/// A device buffer + its backing memory. Storage buffers prefer `DEVICE_LOCAL`
/// (compute reads/writes proper GPU memory). `host_visible` is true when the
/// bound memory type is ALSO `HOST_VISIBLE | HOST_COHERENT` — the llvmpipe
/// fallback (no device-local heap) always qualifies, and so does the single
/// unified heap on an integrated GPU. When true, `upload`/`zero` map the
/// buffer directly (no staging, no submit); `download` always stages
/// regardless of `host_visible` — see its doc for the measured driver race
/// that makes a direct-map readback unsafe.
pub struct VkBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub host_visible: bool,
}

/// The process's ONE Vulkan instance (with the loader entry it came from, and
/// whether the cooperative-matrix instance extension is present on it),
/// created on first use and **never destroyed**.
///
/// Not an optimisation - a correctness requirement. Destroying the last
/// Vulkan instance makes the loader unload every ICD it opened, and some
/// vendor ICDs do not survive being unloaded and reloaded: after a handful of
/// rounds the loader still finds the shared object but can no longer resolve
/// `vkCreateInstance` through it, and from then on the process enumerates
/// zero physical devices while every other process still sees the cards. A
/// process that builds one context per forward call - the shape every "fresh
/// device per call" model has - reaches that point in single digits, and what
/// follows is not a clean error but a silent demotion to a software adapter
/// with a fraction of the real card's limits.
///
/// So the instance is created once and outlives every context. Contexts still
/// create and destroy their own logical devices normally; only the instance is
/// permanent, and it owns no VRAM.
///
/// # Safety
/// Wraps unsafe ash FFI. The instance is deliberately leaked, which is what
/// makes handing out `&'static` references to it sound.
fn shared_instance() -> Result<&'static (ash::Entry, ash::Instance, bool), String> {
    static SHARED: std::sync::OnceLock<Result<(ash::Entry, ash::Instance, bool), String>> =
        std::sync::OnceLock::new();
    SHARED
        .get_or_init(|| unsafe {
            tracing::info!("creating the process-lifetime Vulkan instance (once, never destroyed)");
            // `vkCreateInstance` makes the loader `dlopen` every installed
            // ICD, which is the same non-re-entrant loader path `backend-wgpu`
            // serialises when it builds its `wgpu::Instance`. Same lock, so
            // the two cannot enter it at the same moment from two processes.
            let _init = backend_api::hardware::device_init_lock();
            let entry = ash::Entry::load().map_err(|e| format!("failed to load Vulkan loader: {e}"))?;

            let app_name = CString::new("brain-vk").unwrap();
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(0)
                .engine_name(&app_name)
                .engine_version(0)
                .api_version(vk::API_VERSION_1_3);

            // Optional: `BRAIN_VK_VALIDATE` enables the Khronos validation layer with
            // SYNCHRONIZATION_VALIDATION + a debug messenger, to catch GPU hazards
            // (the loader's VK_INSTANCE_LAYERS env is unreliable, so we wire it here).
            let validate = std::env::var("BRAIN_VK_VALIDATE").is_ok();
            let val_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();
            let mut layers: Vec<*const std::ffi::c_char> = Vec::new();
            // Validation instance extensions (debug messenger + sync-validation feature)
            // are needed in BOTH the with- and without-coopmat instance variants.
            let mut val_exts: Vec<*const std::ffi::c_char> = Vec::new();
            if validate {
                layers.push(val_layer.as_ptr());
                val_exts.push(ash::ext::debug_utils::NAME.as_ptr());
                val_exts.push(ash::ext::validation_features::NAME.as_ptr());
            }
            // Instance must enable the cooperative-matrix instance extension so we
            // can call vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
            let mut instance_exts: Vec<*const std::ffi::c_char> = vec![ash::khr::cooperative_matrix::NAME.as_ptr()];
            instance_exts.extend_from_slice(&val_exts);
            // BRAIN_VK_VALIDATE=gpu => GPU-assisted (in-shader OOB/descriptor checks);
            // anything else => synchronization validation (race/hazard checks).
            let gpu_av = std::env::var("BRAIN_VK_VALIDATE").map(|v| v == "gpu").unwrap_or(false);
            let sync_feats = if gpu_av {
                vec![vk::ValidationFeatureEnableEXT::GPU_ASSISTED]
            } else {
                vec![vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION]
            };
            let mut val_features = vk::ValidationFeaturesEXT::default().enabled_validation_features(&sync_feats);
            let mut instance_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&instance_exts);
            if validate {
                instance_info = instance_info.push_next(&mut val_features);
            }

            // The cooperative-matrix instance extension may be absent on llvmpipe;
            // retry without it (we then report extension_present = false) - but keep the
            // validation extensions so the debug messenger still loads.
            let (instance, coopmat_instance_ext) = match entry.create_instance(&instance_info, None) {
                Ok(i) => (i, true),
                Err(e) => {
                    tracing::warn!(error = %e, "vkCreateInstance failed with the cooperative-matrix extension; retrying without it");
                    let mut bare = vk::InstanceCreateInfo::default()
                        .application_info(&app_info)
                        .enabled_layer_names(&layers)
                        .enabled_extension_names(&val_exts);
                    if validate {
                        bare = bare.push_next(&mut val_features);
                    }
                    let i = entry
                        .create_instance(&bare, None)
                        .map_err(|e| format!("vkCreateInstance failed: {e}"))?;
                    (i, false)
                }
            };

            if validate {
                let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(vk_debug_callback));
                let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
                // Deliberately never destroyed: the messenger lives for the process
                // lifetime (diagnostic only) and is reclaimed with the instance.
                // Neither the loader nor the handle implements `Drop`, so letting
                // both bindings fall out of scope already leaks the messenger - a
                // `mem::forget` here would be a no-op that only looked deliberate.
                if loader.create_debug_utils_messenger(&info, None).is_err() {
                    eprintln!("[vk] failed to install the debug messenger");
                }
                eprintln!("[vk] validation layer + synchronization validation enabled");
            }
            Ok((entry, instance, coopmat_instance_ext))
        })
        .as_ref()
        .map_err(|e| e.clone())
}

impl VkContext {
    /// Create the instance + device and run the cooperative-matrix capability
    /// query. Selects the first discrete GPU if present, else the first device.
    ///
    /// # Safety
    /// Wraps unsafe ash FFI; the returned object owns all handles and frees them
    /// in `Drop`. Returns `Err` (rather than panicking) so callers can fall back.
    pub fn new() -> Result<VkContext, String> {
        unsafe { Self::new_inner(None) }
    }

    /// Like [`VkContext::new`], but the physical device is chosen by `select`
    /// (an index into the enumerated list) instead of the built-in ranking —
    /// how `backend-vulkan` binds the card the canonical device registry
    /// resolved, by identity rather than position.
    pub fn new_select(select: &PhysicalDeviceSelect<'_>) -> Result<VkContext, String> {
        unsafe { Self::new_inner(Some(select)) }
    }

    unsafe fn new_inner(select: Option<&PhysicalDeviceSelect<'_>>) -> Result<VkContext, String> {
        // Shared and never destroyed - see `shared_instance`. `ash::Entry` and
        // `ash::Instance` are handle wrappers, so cloning them hands this
        // context the SAME instance every other context uses; what it owns
        // (and destroys in `Drop`) is the logical device below, not this.
        let (entry, instance, coopmat_instance_ext) = shared_instance()?;
        let (entry, instance, coopmat_instance_ext) =
            (entry.clone(), instance.clone(), *coopmat_instance_ext);


        let physical_devices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate_physical_devices failed: {e}"))?;
        if physical_devices.is_empty() {
            return Err("no Vulkan physical devices found".into());
        }
        // Prefer a real GPU over a software rasteriser (llvmpipe): rank by device
        // type discrete > integrated > virtual > other, and only fall back to a
        // CPU device if nothing else exists. `BRAIN_VK_DEVICE=<index>` forces a
        // specific physical-device index (overriding the ranking).
        let rank = |t: vk::PhysicalDeviceType| -> i32 {
            match t {
                vk::PhysicalDeviceType::DISCRETE_GPU => 4,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 0,
                _ => 1,
            }
        };
        let forced = std::env::var("BRAIN_VK_DEVICE").ok().and_then(|s| s.parse::<usize>().ok());
        let physical_device = if let Some(select) = select {
            match select(&instance, &physical_devices) {
                Ok(i) if i < physical_devices.len() => physical_devices[i],
                Ok(i) => return Err(format!("device selector returned out-of-range index {i}")),
                Err(e) => return Err(e),
            }
        } else {
            match forced {
                Some(i) if i < physical_devices.len() => physical_devices[i],
                _ => physical_devices
                    .iter()
                    .copied()
                    .max_by_key(|&pd| rank(instance.get_physical_device_properties(pd).device_type))
                    .unwrap_or(physical_devices[0]),
            }
        };

        let props = instance.get_physical_device_properties(physical_device);
        let adapter_name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        let vendor_id = props.vendor_id;
        // Nanoseconds per timestamp tick — 0.0 on a device with no meaningful
        // timestamp support (spec allows this; `timestamp_valid_bits` below is
        // the authoritative "can this queue write timestamps at all" signal).
        let timestamp_period_ns = props.limits.timestamp_period as f64;

        // Pick a queue family with COMPUTE.
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);
        let (queue_family_index, timestamp_valid_bits) = queue_families
            .iter()
            .enumerate()
            .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, q)| (i as u32, q.timestamp_valid_bits))
            .ok_or_else(|| "no compute queue family".to_string())?;

        // ---- capability query ----
        let caps = Self::query_coopmat(&entry, &instance, physical_device, coopmat_instance_ext);

        // ---- logical device ----
        // Enable the device-level coopmat extension + feature only when present,
        // so creation still succeeds on devices that lack it (e.g. llvmpipe,
        // Pascal). The scalar fallback path needs no extension.
        let device_ext_names: Vec<*const i8> = if caps.extension_present {
            vec![ash::khr::cooperative_matrix::NAME.as_ptr()]
        } else {
            vec![]
        };

        let queue_priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)];

        // ---- non-fp32 arithmetic capability query ----
        // Read what the physical device supports, so we enable exactly those at
        // create_device (enabling an unsupported feature makes creation fail).
        let mut sf16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut sdot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut timeline_feat = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut core_feats2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut sf16i8)
            .push_next(&mut sdot)
            .push_next(&mut timeline_feat);
        instance.get_physical_device_features2(physical_device, &mut core_feats2);
        // The `4x8BitPacked` signed-accelerated property confirms DP4A is a fast
        // path (not emulated); pair it with the dot-product feature bit.
        let mut dot_props = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut dot_props);
        instance.get_physical_device_properties2(physical_device, &mut props2);
        let prec = PrecisionCaps {
            f64: core_feats2.features.shader_float64 != 0,
            f16: sf16i8.shader_float16 != 0,
            dp4a: sdot.shader_integer_dot_product != 0
                && dot_props.integer_dot_product4x8_bit_packed_signed_accelerated != 0,
        };
        let timeline_supported = timeline_feat.timeline_semaphore != 0;

        // Enable the coopmat extension (if present) plus the promoted-core
        // extensions that back f16/int8 and the integer dot product. Both were
        // promoted to core (1.2 / 1.3), but requesting the extension name is the
        // portable way to unlock the feature on a 1.3 device.
        let mut device_ext_names: Vec<*const i8> = device_ext_names;
        let dev_exts = instance.enumerate_device_extension_properties(physical_device).unwrap_or_default();
        let has_ext = |name: &CStr| dev_exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == name);
        if prec.f16 && has_ext(ash::khr::shader_float16_int8::NAME) {
            device_ext_names.push(ash::khr::shader_float16_int8::NAME.as_ptr());
        }
        if prec.dp4a && has_ext(ash::khr::shader_integer_dot_product::NAME) {
            device_ext_names.push(ash::khr::shader_integer_dot_product::NAME.as_ptr());
        }

        // Chain the coopmat feature struct only if supported.
        let mut coopmat_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(caps.feature_supported);
        // Enable exactly the non-fp32 arithmetic the device reported, plus
        // `robustBufferAccess`: a core Vulkan 1.0 feature that bounds an
        // out-of-range storage access in HARDWARE. It is what lets
        // `shader::wgsl_to_spirv` drop naga's per-access software clamp (the
        // same trade wgpu makes when its `robust_buffer_access2` private cap
        // selects `BoundsCheckPolicy::Unchecked`) without making an
        // out-of-range access undefined behaviour. Free on this class of
        // hardware - the P40 roofline probes measure identically with it on.
        let core_enabled = vk::PhysicalDeviceFeatures::default()
            .shader_float64(prec.f64)
            .robust_buffer_access(true);
        let mut en_f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(prec.f16);
        let mut en_dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
            .shader_integer_dot_product(prec.dp4a);
        let mut en_timeline =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(timeline_supported);
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_ext_names)
            .enabled_features(&core_enabled);
        if caps.feature_supported {
            device_info = device_info.push_next(&mut coopmat_features);
        }
        if prec.f16 {
            device_info = device_info.push_next(&mut en_f16i8);
        }
        if prec.dp4a {
            device_info = device_info.push_next(&mut en_dot);
        }
        if timeline_supported {
            device_info = device_info.push_next(&mut en_timeline);
        }

        // `vkCreateDevice` racing another THREAD in this same process,
        // simultaneously creating or destroying a device on the same card, is
        // a real driver hazard (Mesa's EGL/GL loader is not re-entrant, and
        // the driver's own worker threads race a concurrent destroy - see
        // `backend_api::hardware`'s module doc). This crate reaches the same
        // physical cards through raw `ash`, so it takes the SAME shared
        // in-process lock `backend-wgpu` does - a second, private lock here
        // would serialise nothing that matters. Scoped to the FFI call pair
        // (device + its command pool) and released immediately: the
        // capability queries above and the buffer work afterwards are not
        // part of the hazard.
        let (device, queue, command_pool) = {
            let _init = backend_api::hardware::device_init_lock();
            let device = instance
                .create_device(physical_device, &device_info, None)
                .map_err(|e| format!("vkCreateDevice failed: {e}"))?;
            let queue = device.get_device_queue(queue_family_index, 0);
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = device
                .create_command_pool(&pool_info, None)
                .map_err(|e| format!("create_command_pool failed: {e}"))?;
            (device, queue, command_pool)
        };

        let mem_props = instance.get_physical_device_memory_properties(physical_device);

        // A `SEMAPHORE_TYPE_TIMELINE` semaphore, initial counter 0. Only
        // created when the device actually enabled the feature above - a
        // semaphore of this type used against a device that never enabled
        // `timelineSemaphore` is invalid, not merely slow.
        let timeline = if timeline_supported {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            Some(
                device
                    .create_semaphore(&info, None)
                    .map_err(|e| format!("create_semaphore (timeline): {e}"))?,
            )
        } else {
            None
        };

        Ok(VkContext {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            adapter_name,
            vendor_id,
            timestamp_period_ns,
            timestamp_valid_bits,
            caps,
            prec,
            mem_props,
            staging: std::sync::Mutex::new(None),
            dead: std::sync::Mutex::new(Vec::new()),
            reclaim_events: std::sync::atomic::AtomicU64::new(0),
            pending_steps: std::sync::atomic::AtomicU64::new(0),
            set_refs: std::sync::Mutex::new(std::collections::HashMap::new()),
            submits: std::sync::atomic::AtomicU64::new(0),
            queue_lock: std::sync::Mutex::new(()),
            timeline,
            timeline_counter: std::sync::atomic::AtomicU64::new(1),
        })
    }

    /// Acquire the lock guarding `queue`/`command_pool` for the caller's whole
    /// allocate-record-submit-wait-free sequence — see [`VkContext::queue_lock`].
    pub fn queue_guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.queue_lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether this device has a live timeline semaphore for asynchronous
    /// submission - false on the (defensive-only) fallback where the device
    /// did not report `timelineSemaphore`.
    pub fn timeline_supported(&self) -> bool {
        self.timeline.is_some()
    }

    /// Claim the next value an asynchronous submission should signal on the
    /// semaphore returned by [`Self::timeline_semaphore`] (chained onto the
    /// caller's own `vk::SubmitInfo` via `.signal_semaphores(&[sem])` plus a
    /// `push_next`ed `vk::TimelineSemaphoreSubmitInfo`). Monotonically
    /// increasing across every caller sharing this `ctx` (siblings via
    /// `share`/`new_like` included), so two handles' submissions never
    /// collide on the same value.
    pub fn timeline_next(&self) -> u64 {
        self.timeline_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// The shared timeline semaphore a caller's submission signals - `None`
    /// when this device has no timeline semaphore, in which case the caller
    /// must fall back to a `VkFence`.
    pub fn timeline_semaphore(&self) -> Option<vk::Semaphore> {
        self.timeline
    }

    /// Non-blocking peek at the timeline's current counter value
    /// (`vkGetSemaphoreCounterValue`) - never waits, so this is the check a
    /// caller uses to skip a wait it can already tell is unnecessary.
    ///
    /// # Safety
    /// Requires [`Self::timeline_supported`].
    pub unsafe fn timeline_peek(&self) -> u64 {
        self.device
            .get_semaphore_counter_value(self.timeline.expect("timeline_peek without a timeline semaphore"))
            .expect("get_semaphore_counter_value")
    }

    /// Block the calling host thread until the timeline's counter reaches
    /// `value` (`vkWaitSemaphores`) - bounded, never `u64::MAX`, matching
    /// every other device wait in this file (`BRAIN_GPU_WAIT_S`). Touches
    /// neither `queue` nor `command_pool`, so - unlike `wait_for_fences` in
    /// `run_cmd`/`dispatch` above - this deliberately does NOT require
    /// [`Self::queue_guard`]: a semaphore wait does not race a concurrent
    /// `vkQueueSubmit`/command-buffer allocation on another handle sharing
    /// this device the way touching the queue or pool would.
    ///
    /// # Safety
    /// Requires [`Self::timeline_supported`].
    pub unsafe fn timeline_wait(&self, value: u64) {
        let sem = self.timeline.expect("timeline_wait without a timeline semaphore");
        let semaphores = [sem];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
        match self.device.wait_semaphores(&wait_info, backend_api::hardware::wait_timeout_ns()) {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => panic!(
                "GPU timeline wait did not reach value {value} within (BRAIN_GPU_WAIT_S) -- device likely wedged"
            ),
            Err(vk::Result::ERROR_DEVICE_LOST) => panic!("GPU device lost while waiting on the timeline semaphore"),
            Err(e) => panic!("wait_semaphores: {e:?}"),
        }
    }

    /// Enumerate cooperative-matrix shapes + read the feature bit.
    unsafe fn query_coopmat(
        entry: &ash::Entry,
        instance: &ash::Instance,
        pd: vk::PhysicalDevice,
        instance_ext_loaded: bool,
    ) -> CoopMatCaps {
        // Is the device extension advertised?
        let extension_present = instance
            .enumerate_device_extension_properties(pd)
            .map(|exts| {
                exts.iter().any(|e| {
                    CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::cooperative_matrix::NAME
                })
            })
            .unwrap_or(false);

        // Read VkPhysicalDeviceCooperativeMatrixFeaturesKHR via Features2.
        let mut coop_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut coop_features);
        if extension_present {
            instance.get_physical_device_features2(pd, &mut features2);
        }
        let feature_supported = extension_present && coop_features.cooperative_matrix == vk::TRUE;

        // Enumerate supported shapes (needs the instance-level extension loader).
        let mut shapes = Vec::new();
        if instance_ext_loaded && extension_present {
            let loader = ash::khr::cooperative_matrix::Instance::new(entry, instance);
            if let Ok(props) = loader.get_physical_device_cooperative_matrix_properties(pd) {
                for pr in props {
                    shapes.push(CoopMatShape {
                        m: pr.m_size,
                        n: pr.n_size,
                        k: pr.k_size,
                        a_type: pr.a_type,
                        b_type: pr.b_type,
                        c_type: pr.c_type,
                        result_type: pr.result_type,
                        saturating_accumulation: pr.saturating_accumulation == vk::TRUE,
                        scope: pr.scope,
                    });
                }
            }
        }

        CoopMatCaps {
            feature_supported,
            extension_present,
            shapes,
        }
    }

    // ---- buffers (mirror Gpu::storage / write / read) ----

    /// Allocate a host-visible+coherent buffer usable as a compute storage
    /// buffer and as a transfer src/dst. `size` is in bytes.
    /// Allocate a buffer with the given usage and memory properties.
    ///
    /// `VkBuffer::host_visible` is derived from the memory type actually
    /// bound, never from caller intent: on a unified-memory device (an
    /// integrated GPU with no separate VRAM) the `DEVICE_LOCAL` heap `storage()`
    /// requests is often *also* `HOST_VISIBLE | HOST_COHERENT`, and hardcoding
    /// `host_visible: false` there (as this used to) meant every upload/zero on
    /// it paid a staging-buffer + `run_cmd` (a full submit+fence) for memory a
    /// direct `memcpy` could have reached. Requires BOTH bits, not just
    /// `HOST_VISIBLE` — `with_mapped` never flushes or invalidates, so treating
    /// a merely-host-visible-but-not-coherent type as mappable-without-a-barrier
    /// would be a real correctness bug, not just a missed optimisation.
    ///
    /// `host_visible` does NOT license a direct-map *readback* — see
    /// `download`'s doc for a measured Intel-ANV coherency race that a direct
    /// map after `vkWaitForFences` does not reliably avoid.
    fn alloc_raw(&self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, props: vk::MemoryPropertyFlags) -> Option<VkBuffer> {
        let size = size.max(4);
        unsafe {
            let info = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).expect("create_buffer");
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mem_type = match self.find_memory_type(req.memory_type_bits, props) {
                Some(t) => t,
                None => {
                    self.device.destroy_buffer(buffer, None);
                    return None;
                }
            };
            let alloc = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mem_type);
            if std::env::var_os("BRAIN_VK_ALLOC_DEBUG").is_some() {
                eprintln!("[vk-alloc-debug] requesting {} bytes ({:.3} GiB), type_index={}", req.size, req.size as f64 / (1u64 << 30) as f64, mem_type);
                if req.size > (1u64 << 30) {
                    eprintln!("[vk-alloc-debug] backtrace for >1GiB request:\n{}", std::backtrace::Backtrace::force_capture());
                }
            }
            let memory = self.device.allocate_memory(&alloc, None).expect("allocate_memory");
            self.device.bind_buffer_memory(buffer, memory, 0).expect("bind_buffer_memory");
            let mappable = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let host_visible = self.mem_props.memory_types[mem_type as usize].property_flags.contains(mappable);
            Some(VkBuffer { buffer, memory, size, host_visible })
        }
    }

    /// Allocate a storage buffer. Prefers `DEVICE_LOCAL` (proper GPU memory — the
    /// compute kernels read/write here); falls back to host-visible only when no
    /// device-local heap exists (e.g. llvmpipe).
    pub fn storage(&self, size: vk::DeviceSize, extra_usage: vk::BufferUsageFlags) -> VkBuffer {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | extra_usage;
        self.alloc_raw(size, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| {
                self.alloc_raw(size, usage, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            })
            .expect("no suitable memory type for storage buffer")
    }

    /// Allocate a HOST_VISIBLE|HOST_COHERENT buffer (`host_visible = true`, so
    /// `upload`/`zero` use a direct map — NO queue submits; `download` still
    /// stages, see its doc). For small, host-written, GPU-read data:
    /// uniform/params buffers. Compute-hot storage stays on [`Self::storage`]'s
    /// DEVICE_LOCAL path; on an integrated GPU host-visible memory is the same
    /// physical RAM, so a uniform read costs the same — what changes is that
    /// writing one stops costing two blocking submits (fill + staged copy) per
    /// dispatch per frame.
    pub fn storage_host(&self, size: vk::DeviceSize, extra_usage: vk::BufferUsageFlags) -> VkBuffer {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | extra_usage;
        self.alloc_raw(size, usage, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            .expect("no host-visible|coherent memory type (spec guarantees one)")
    }

    /// Map a host-visible buffer and run `f` on its pointer.
    unsafe fn with_mapped<R>(&self, buf: &VkBuffer, f: impl FnOnce(*mut u8) -> R) -> R {
        let ptr = self
            .device
            .map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty())
            .expect("map_memory") as *mut u8;
        let r = f(ptr);
        self.device.unmap_memory(buf.memory);
        r
    }

    /// Run `f` to record a one-off command buffer, then submit + fence-wait.
    fn run_cmd(&self, f: impl FnOnce(vk::CommandBuffer)) {
        self.submits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _guard = self.queue_guard();
        unsafe {
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&alloc).expect("alloc cmd")[0];
            self.device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .expect("begin");
            f(cmd);
            self.device.end_command_buffer(cmd).expect("end");
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None).expect("fence");
            let cmds = [cmd];
            self.device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).expect("submit");
            // Bounded, never `u64::MAX`: an unbounded fence wait is what turns
            // a wedged queue into an unkillable process instead of a reported
            // failure. Same `BRAIN_GPU_WAIT_S` ceiling every other wait on a
            // device in this workspace uses.
            self.device
                .wait_for_fences(&[fence], true, backend_api::hardware::wait_timeout_ns())
                .expect("wait_for_fences timed out or failed (BRAIN_GPU_WAIT_S) -- device likely wedged");
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &cmds);
        }
    }

    /// Run `f` with the reusable host-visible staging buffer grown to >= `size`.
    fn with_staging<R>(&self, size: vk::DeviceSize, f: impl FnOnce(&VkBuffer) -> R) -> R {
        // Not `.unwrap()`: a prior panic while this SAME lock was held (e.g. a
        // real "device lost" fault mid-readback, on a DIFFERENT job/thread
        // that used this shared `VkContext`) poisons the mutex permanently --
        // `.unwrap()` here would then panic EVERY future call, on every
        // future request, forever, even ones that have nothing to do with
        // whatever the original fault was (see `queue_guard`'s identical
        // recovery, already the established pattern in this file; `Drop`
        // below needs the same fix for the same reason).
        let mut guard = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        let need = size.max(4);
        if guard.as_ref().map(|b| b.size < need).unwrap_or(true) {
            if let Some(old) = guard.take() {
                self.destroy_buffer(old);
            }
            let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let s = self
                .alloc_raw(need, usage, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
                .expect("staging buffer");
            *guard = Some(s);
        }
        f(guard.as_ref().unwrap())
    }

    fn copy_buffer(&self, src: &VkBuffer, dst: &VkBuffer, size: vk::DeviceSize) {
        self.copy_buffer_at(src, dst, size, 0);
    }

    fn copy_buffer_at(&self, src: &VkBuffer, dst: &VkBuffer, size: vk::DeviceSize, dst_offset: vk::DeviceSize) {
        let region = [vk::BufferCopy::default().size(size).dst_offset(dst_offset)];
        self.run_cmd(|cmd| unsafe { self.device.cmd_copy_buffer(cmd, src.buffer, dst.buffer, &region) });
    }

    /// Upload raw bytes to the start of a storage buffer (via staging for
    /// device-local buffers; direct map for the host-visible fallback).
    pub fn upload(&self, buf: &VkBuffer, bytes: &[u8]) {
        self.upload_at(buf, bytes, 0);
    }

    /// [`Self::upload`] at a byte offset into `buf`. `with_staging` already
    /// reuses one shared staging buffer across calls (growing only when a
    /// larger upload needs more room), so — unlike the wgpu backend's
    /// per-write-sized staging belt (see `Backend::write_at`'s doc) — chunking
    /// through this path bounds the *number* of copy commands, not a resident
    /// staging cost this backend does not have.
    pub fn upload_at(&self, buf: &VkBuffer, bytes: &[u8], offset: vk::DeviceSize) {
        assert!(offset + bytes.len() as vk::DeviceSize <= buf.size, "upload overflows buffer");
        if buf.host_visible {
            unsafe {
                self.with_mapped(buf, |ptr| {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset as usize), bytes.len())
                })
            };
            return;
        }
        let n = bytes.len() as vk::DeviceSize;
        self.with_staging(n, |stg| {
            unsafe { self.with_mapped(stg, |ptr| std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len())) };
            self.copy_buffer_at(stg, buf, n, offset);
        });
    }

    /// Zero-fill a storage buffer (so kernels that read a not-fully-written region
    /// see zeros, matching the wgpu/CPU backends). `cmd_fill_buffer` on the GPU for
    /// device-local; direct memset for the host-visible fallback.
    pub fn zero(&self, buf: &VkBuffer) {
        if buf.host_visible {
            unsafe { self.with_mapped(buf, |ptr| std::ptr::write_bytes(ptr, 0, buf.size as usize)) };
            return;
        }
        self.run_cmd(|cmd| unsafe { self.device.cmd_fill_buffer(cmd, buf.buffer, 0, buf.size, 0) });
    }

    /// Read `len` bytes back from the start of a storage buffer — always via
    /// staging, even on a `host_visible` buffer.
    ///
    /// A direct `with_mapped` read (no staging copy) was measured live on this
    /// box's Intel Arc (MTL) / Mesa ANV 25.0.7 to be genuinely racy: a
    /// dispatch's writes are sometimes NOT visible to a host read performed
    /// immediately after `vkWaitForFences` returns, even though the memory
    /// type is `HOST_VISIBLE | HOST_COHERENT` — `gpu-core`'s
    /// `vulkan_dispatch_storage_and_readback` test (a 2-dispatch RAW chain:
    /// `out = a+b`, then `out2 = out+b`, read back) failed ~85-90% of runs
    /// with a direct-map readback (all-zero `out2`, i.e. the pre-dispatch
    /// zero-init value, never the computed one), 20/20 clean once this
    /// function was changed to always stage, and the failure reproduced
    /// identically with `BRAIN_VK_SERIAL=1` (per-dispatch submit+fence,
    /// ruling out the already-documented compute-barrier bug above as the
    /// cause) and was independent of `backend-vulkan`'s per-kernel timestamp
    /// feature (bisected out separately). A diagnostic direct-map peek taken
    /// immediately after the dispatch's own `vkWaitForFences` returned still
    /// read zero; a *later* read of the same memory (this function's own
    /// staging copy) saw the correct value — i.e. the fence signal does not,
    /// on this driver, reliably imply the write is visible to a host
    /// `vkMapMemory` read yet. The staging path's extra `cmd_copy_buffer` +
    /// its own submit+fence happens to give the driver's cache write-back
    /// enough time; per this repo's own rule (never absorb a driver bug by
    /// loosening a tolerance), that extra round trip is being kept
    /// deliberately, not treated as a missed optimisation. `upload`/`zero`
    /// (host write -> device read) are NOT affected — every write in the
    /// above investigation landed correctly — so they keep the direct-map
    /// fast path.
    pub fn download(&self, buf: &VkBuffer, len: usize) -> Vec<u8> {
        assert!(len as vk::DeviceSize <= buf.size, "download overflows buffer");
        let mut out = vec![0u8; len];
        let n = len as vk::DeviceSize;
        self.with_staging(n, |stg| {
            self.copy_buffer(buf, stg, n);
            unsafe { self.with_mapped(stg, |ptr| std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), len)) };
        });
        out
    }

    fn find_memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            (type_bits & (1 << i)) != 0
                && self.mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
    }

    /// Record + submit a single compute dispatch with one bind group
    /// (set 0 = [uniform at binding 0, storage buffers at bindings 1..]) and
    /// wait on a fence. Mirrors `Gpu::submit` for the one-kernel case.
    ///
    /// `groups` is the workgroup count (x, y, z).
    pub fn dispatch(
        &self,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        groups: (u32, u32, u32),
    ) {
        let _guard = self.queue_guard();
        unsafe {
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = self.device.allocate_command_buffers(&alloc).expect("alloc cmd")[0];

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin).expect("begin");
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_dispatch(cmd, groups.0, groups.1, groups.2);
            self.device.end_command_buffer(cmd).expect("end");

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("fence");
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .expect("queue_submit");
            // Bounded, never `u64::MAX` - see `run_cmd`'s wait.
            self.device
                .wait_for_fences(&[fence], true, backend_api::hardware::wait_timeout_ns())
                .expect("wait_for_fences timed out or failed (BRAIN_GPU_WAIT_S) -- device likely wedged");

            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &cmds);
        }
    }

    pub fn destroy_buffer(&self, buf: VkBuffer) {
        unsafe {
            self.device.destroy_buffer(buf.buffer, None);
            self.device.free_memory(buf.memory, None);
        }
    }

    /// Hand a dropped buffer over to be destroyed later.
    ///
    /// Deferred rather than immediate because a caller's recorded-but-not-yet-
    /// submitted dispatch may still name this buffer's raw handle: the backend
    /// batches `vkCmdDispatch` recording and submits later, so "the Rust owner
    /// dropped" does not imply "the device is done with it".
    pub fn bury(&self, buf: VkBuffer) {
        self.dead.lock().unwrap_or_else(|e| e.into_inner()).push(buf);
    }

    /// Declare that descriptor set `set` now names exactly `buffers`, replacing
    /// whatever it named before (a recycled set is rewritten in place).
    ///
    /// Called from the backend's descriptor-write path, so the reference exists
    /// from the moment the set can be bound - not from the moment its dispatch
    /// is submitted. [`Self::reclaim_dead`] refuses to destroy a buried buffer
    /// any live set still names.
    pub fn set_names(&self, set: vk::DescriptorSet, buffers: &[vk::Buffer]) {
        self.set_refs.lock().unwrap_or_else(|e| e.into_inner()).insert(set, buffers.to_vec());
    }

    /// Declare that descriptor set `set` no longer names anything: its
    /// dispatches have completed (fence-waited) and it is being recycled or
    /// discarded, so it cannot be bound again before being rewritten.
    ///
    /// Legal even though the set still physically holds the old descriptors -
    /// Vulkan requires a descriptor to be valid when a dispatch *uses* it, and
    /// every reuse path rewrites the set first (`set_names` above).
    pub fn set_released(&self, set: vk::DescriptorSet) {
        self.set_refs.lock().unwrap_or_else(|e| e.into_inner()).remove(&set);
    }

    /// Note that `n` dispatches have been recorded against this device, and are
    /// not yet submitted. Pairs with [`Self::steps_submitted`].
    pub fn steps_recorded(&self, n: u64) {
        self.pending_steps.fetch_add(n, std::sync::atomic::Ordering::AcqRel);
    }

    /// Note that `n` previously-recorded dispatches have been submitted (and,
    /// on this backend, fence-waited).
    pub fn steps_submitted(&self, n: u64) {
        self.pending_steps.fetch_sub(n, std::sync::atomic::Ordering::AcqRel);
    }

    /// Bytes currently buried (dropped, not yet `vkFreeMemory`'d) - a
    /// read-only introspection, does NOT reclaim anything. Exists so a test
    /// can assert deferred-reclaim accounting stays bounded (e.g. a
    /// multi-layer forward pass must not leave O(layer count) worth of
    /// scratch buried by the time it returns) without needing to measure
    /// real OS-level VRAM.
    pub fn buried_bytes(&self) -> u64 {
        self.dead.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|b| b.size).sum()
    }

    /// Destroy the buried buffers nothing can still reach, returning the bytes
    /// released. Two conditions gate a buffer, and BOTH are required:
    ///
    /// * nothing is recorded-but-unsubmitted against this device anywhere
    ///   ([`Self::pending_steps`]), and
    /// * no live descriptor set names it ([`Self::set_names`]).
    ///
    /// The second is the one that makes this safe. A descriptor set starts
    /// naming a raw `vk::Buffer` when it is *written*, which happens while the
    /// step is built - before `submit`, and therefore before `pending_steps`
    /// sees anything at all. Guarding on the counter alone let a batch of
    /// recorded steps have its scratch buffers destroyed underneath it the
    /// next time any unrelated `read`/`write`/`poll_wait` flushed an empty
    /// pending list, and the subsequent dispatch read freed device memory -
    /// which this hardware reports as `VK_ERROR_DEVICE_LOST`.
    ///
    /// A buffer that is still referenced stays buried rather than being
    /// dropped from the list, so the next reclaim past the referencing set's
    /// retirement frees it. Callers reach this right after a fence-waited
    /// submit, so a live server hits it every flush and the buried set never
    /// grows unbounded.
    pub fn reclaim_dead(&self) -> u64 {
        if self.pending_steps.load(std::sync::atomic::Ordering::Acquire) != 0 {
            return 0;
        }
        let mut dead = self.dead.lock().unwrap_or_else(|e| e.into_inner());
        if dead.is_empty() {
            return 0;
        }
        let referenced: std::collections::HashSet<vk::Buffer> = self
            .set_refs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .flatten()
            .copied()
            .collect();
        let mut freed = 0u64;
        let mut still_named: Vec<VkBuffer> = Vec::new();
        for b in std::mem::take(&mut *dead) {
            if referenced.contains(&b.buffer) {
                still_named.push(b);
            } else {
                freed += b.size;
                self.destroy_buffer(b);
            }
        }
        *dead = still_named;
        drop(dead);
        if freed > 0 {
            self.reclaim_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        freed
    }

    /// How many times [`Self::reclaim_dead`] has actually freed something
    /// (not counted when it ran but found nothing buried, or bailed early on
    /// outstanding work) - unlike a raw queue-submit count, this is NOT
    /// inflated by one-off staging submits (`upload`/`zero`/`download`, none
    /// of which call `reclaim_dead`), so it isolates deferred-reclaim
    /// activity specifically. A loop that only reclaims once, at its very
    /// end, reports 1 here regardless of how many layers/steps it ran; a
    /// loop that reclaims periodically reports roughly `iterations /
    /// period`.
    pub fn reclaim_event_count(&self) -> u64 {
        self.reclaim_events.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        // Teardown is half the hazard, not an afterthought: destroying a
        // device races the driver's own background worker threads and a
        // concurrent create/destroy on another thread of THIS process. So
        // destruction takes the same in-process lock creation does.
        //
        // Blocking, deliberately, even though this is a `Drop` that can run
        // on the residency dispatcher thread: the alternative - give up after
        // a while and destroy the device anyway - is the unserialised
        // teardown that was observed as an intermittent SIGSEGV inside the
        // driver's own background pipeline thread. Every holder of this lock
        // is itself bounded, so the wait is bounded in practice; the case
        // that is not (an abandoned worker thread from a timed-out
        // `backend_api::hardware::bounded`) needs a supervised, killable
        // child process to fix properly, and cannot be fixed by making
        // teardown less safe.
        let _init = backend_api::hardware::device_init_lock();
        unsafe {
            let _ = self.device.device_wait_idle();
            // Same poisoning risk as `with_staging` above -- Drop must never
            // itself panic (it can run on the DISPATCHER thread, via
            // `ResidencyManager::build_failed` dropping the failed claim's
            // instance; an uncaught panic there is the exact server-wide
            // wedge `crates/residency/src/executor.rs::dispatch_loop`'s
            // per-message catch_unwind exists to survive -- surviving it is
            // not a reason to make it MORE likely).
            if let Some(s) = self.staging.lock().unwrap_or_else(|e| e.into_inner()).take() {
                self.device.destroy_buffer(s.buffer, None);
                self.device.free_memory(s.memory, None);
            }
            // The device is idle (waited above), so anything still buried is
            // safe to destroy - and must be, before the device goes away.
            for b in std::mem::take(&mut *self.dead.lock().unwrap_or_else(|e| e.into_inner())) {
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            }
            if let Some(sem) = self.timeline.take() {
                self.device.destroy_semaphore(sem, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            // The INSTANCE is deliberately not destroyed: it is shared by
            // every context and lives for the process - see `shared_instance`.
        }
    }
}

/// Human-readable name for a `ComponentTypeKHR` (for `vk-info`).
pub fn component_type_name(t: vk::ComponentTypeKHR) -> &'static str {
    match t {
        vk::ComponentTypeKHR::FLOAT16 => "f16",
        vk::ComponentTypeKHR::FLOAT32 => "f32",
        vk::ComponentTypeKHR::FLOAT64 => "f64",
        vk::ComponentTypeKHR::SINT8 => "i8",
        vk::ComponentTypeKHR::SINT16 => "i16",
        vk::ComponentTypeKHR::SINT32 => "i32",
        vk::ComponentTypeKHR::SINT64 => "i64",
        vk::ComponentTypeKHR::UINT8 => "u8",
        vk::ComponentTypeKHR::UINT16 => "u16",
        vk::ComponentTypeKHR::UINT32 => "u32",
        vk::ComponentTypeKHR::UINT64 => "u64",
        _ => "?",
    }
}

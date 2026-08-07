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
    /// for its whole allocate-record-submit-wait-free sequence. Confirmed load-
    /// bearing, not theoretical: omitting it reproduced a real
    /// `ERROR_DEVICE_LOST` on P40 hardware within seconds of 8 threads sharing
    /// one device (`crates/gpu-core/tests/device_sharing.rs`'s
    /// `concurrent_shared_handles_do_not_deadlock`). GPU *execution* stays
    /// exactly as serial as before this lock existed (every submit here is
    /// already synchronous submit+fence-wait, never pipelined), so this only
    /// serializes the HOST-side race, not real device throughput.
    pub queue_lock: std::sync::Mutex<()>,
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

/// A device buffer + its backing memory. Storage buffers are `DEVICE_LOCAL`
/// (compute reads/writes proper GPU memory; up/download go through a host-visible
/// staging buffer + GPU copy). `host_visible` is true only for the llvmpipe
/// fallback (no device-local heap) — then up/download map the buffer directly.
pub struct VkBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub host_visible: bool,
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
        let entry = ash::Entry::load().map_err(|e| format!("failed to load Vulkan loader: {e}"))?;

        let app_name = CString::new("moe-rs-vk").unwrap();
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
        // retry without it (we then report extension_present = false) — but keep the
        // validation extensions so the debug messenger still loads.
        let (instance, coopmat_instance_ext) = match entry.create_instance(&instance_info, None) {
            Ok(i) => (i, true),
            Err(_) => {
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
            // both bindings fall out of scope already leaks the messenger — a
            // `mem::forget` here would be a no-op that only looked deliberate.
            if loader.create_debug_utils_messenger(&info, None).is_err() {
                eprintln!("[vk] failed to install the debug messenger");
            }
            eprintln!("[vk] validation layer + synchronization validation enabled");
        }

        let physical_devices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate_physical_devices failed: {e}"))?;
        if physical_devices.is_empty() {
            instance.destroy_instance(None);
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
                Ok(i) => {
                    instance.destroy_instance(None);
                    return Err(format!("device selector returned out-of-range index {i}"));
                }
                Err(e) => {
                    instance.destroy_instance(None);
                    return Err(e);
                }
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
        let mut core_feats2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut sf16i8)
            .push_next(&mut sdot);
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
        // Enable exactly the non-fp32 arithmetic the device reported.
        let core_enabled = vk::PhysicalDeviceFeatures::default().shader_float64(prec.f64);
        let mut en_f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(prec.f16);
        let mut en_dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
            .shader_integer_dot_product(prec.dp4a);
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

        let mem_props = instance.get_physical_device_memory_properties(physical_device);

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
            submits: std::sync::atomic::AtomicU64::new(0),
            queue_lock: std::sync::Mutex::new(()),
        })
    }

    /// Acquire the lock guarding `queue`/`command_pool` for the caller's whole
    /// allocate-record-submit-wait-free sequence — see [`VkContext::queue_lock`].
    pub fn queue_guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.queue_lock.lock().unwrap_or_else(|e| e.into_inner())
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
    fn alloc_raw(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
        host_visible: bool,
    ) -> Option<VkBuffer> {
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
            let memory = self.device.allocate_memory(&alloc, None).expect("allocate_memory");
            self.device.bind_buffer_memory(buffer, memory, 0).expect("bind_buffer_memory");
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
        self.alloc_raw(size, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL, false)
            .or_else(|| {
                self.alloc_raw(
                    size,
                    usage,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    true,
                )
            })
            .expect("no suitable memory type for storage buffer")
    }

    /// Allocate a HOST_VISIBLE|HOST_COHERENT buffer (`host_visible = true`, so
    /// `upload`/`zero`/`download` use a direct map — NO queue submits). For
    /// small, host-written, GPU-read data: uniform/params buffers. Compute-hot
    /// storage stays on [`Self::storage`]'s DEVICE_LOCAL path; on an integrated
    /// GPU host-visible memory is the same physical RAM, so a uniform read
    /// costs the same — what changes is that writing one stops costing two
    /// blocking submits (fill + staged copy) per dispatch per frame.
    pub fn storage_host(&self, size: vk::DeviceSize, extra_usage: vk::BufferUsageFlags) -> VkBuffer {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | extra_usage;
        self.alloc_raw(
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        )
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
            self.device.wait_for_fences(&[fence], true, u64::MAX).expect("wait");
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &cmds);
        }
    }

    /// Run `f` with the reusable host-visible staging buffer grown to >= `size`.
    fn with_staging<R>(&self, size: vk::DeviceSize, f: impl FnOnce(&VkBuffer) -> R) -> R {
        let mut guard = self.staging.lock().unwrap();
        let need = size.max(4);
        if guard.as_ref().map(|b| b.size < need).unwrap_or(true) {
            if let Some(old) = guard.take() {
                self.destroy_buffer(old);
            }
            let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let s = self
                .alloc_raw(need, usage, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, true)
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

    /// Read `len` bytes back from the start of a storage buffer (via staging for
    /// device-local buffers; direct map for the host-visible fallback).
    pub fn download(&self, buf: &VkBuffer, len: usize) -> Vec<u8> {
        assert!(len as vk::DeviceSize <= buf.size, "download overflows buffer");
        let mut out = vec![0u8; len];
        if buf.host_visible {
            unsafe { self.with_mapped(buf, |ptr| std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), len)) };
            return out;
        }
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
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("wait_for_fences");

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
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(s) = self.staging.lock().unwrap().take() {
                self.device.destroy_buffer(s.buffer, None);
                self.device.free_memory(s.memory, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
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

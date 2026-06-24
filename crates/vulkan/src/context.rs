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
    pub caps: CoopMatCaps,
    mem_props: vk::PhysicalDeviceMemoryProperties,
}

/// A device buffer + its backing memory. Host-visible/coherent so we can upload
/// and read back without staging (sufficient for the smoke demo; a perf port
/// would use a DEVICE_LOCAL + staging split).
pub struct VkBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl VkContext {
    /// Create the instance + device and run the cooperative-matrix capability
    /// query. Selects the first discrete GPU if present, else the first device.
    ///
    /// # Safety
    /// Wraps unsafe ash FFI; the returned object owns all handles and frees them
    /// in `Drop`. Returns `Err` (rather than panicking) so callers can fall back.
    pub fn new() -> Result<VkContext, String> {
        unsafe { Self::new_inner() }
    }

    unsafe fn new_inner() -> Result<VkContext, String> {
        let entry = ash::Entry::load().map_err(|e| format!("failed to load Vulkan loader: {e}"))?;

        let app_name = CString::new("moe-rs-vk").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_3);

        // Instance must enable the cooperative-matrix instance extension so we
        // can call vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
        let instance_exts = [ash::khr::cooperative_matrix::NAME.as_ptr()];
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_exts);

        // The cooperative-matrix instance extension may be absent on llvmpipe;
        // retry without it (we then report extension_present = false).
        let (instance, coopmat_instance_ext) = match entry.create_instance(&instance_info, None) {
            Ok(i) => (i, true),
            Err(_) => {
                let bare = vk::InstanceCreateInfo::default().application_info(&app_info);
                let i = entry
                    .create_instance(&bare, None)
                    .map_err(|e| format!("vkCreateInstance failed: {e}"))?;
                (i, false)
            }
        };

        let physical_devices = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("enumerate_physical_devices failed: {e}"))?;
        if physical_devices.is_empty() {
            instance.destroy_instance(None);
            return Err("no Vulkan physical devices found".into());
        }
        // Prefer a discrete GPU.
        let physical_device = physical_devices
            .iter()
            .copied()
            .find(|&pd| {
                instance.get_physical_device_properties(pd).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap_or(physical_devices[0]);

        let props = instance.get_physical_device_properties(physical_device);
        let adapter_name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();

        // Pick a queue family with COMPUTE.
        let queue_family_index = instance
            .get_physical_device_queue_family_properties(physical_device)
            .iter()
            .enumerate()
            .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
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

        // Chain the coopmat feature struct only if supported.
        let mut coopmat_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(caps.feature_supported);
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_ext_names);
        if caps.feature_supported {
            device_info = device_info.push_next(&mut coopmat_features);
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
            caps,
            mem_props,
        })
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
    pub fn storage(&self, size: vk::DeviceSize, extra_usage: vk::BufferUsageFlags) -> VkBuffer {
        let size = size.max(4);
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | extra_usage,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.device.create_buffer(&info, None).expect("create_buffer");
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mem_type = self
                .find_memory_type(
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .expect("no host-visible memory type");
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type);
            let memory = self.device.allocate_memory(&alloc, None).expect("allocate_memory");
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind_buffer_memory");
            VkBuffer { buffer, memory, size }
        }
    }

    /// Upload raw bytes to the start of a host-visible buffer.
    pub fn upload(&self, buf: &VkBuffer, bytes: &[u8]) {
        assert!(bytes.len() as vk::DeviceSize <= buf.size, "upload overflows buffer");
        unsafe {
            let ptr = self
                .device
                .map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty())
                .expect("map_memory") as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            self.device.unmap_memory(buf.memory);
        }
    }

    /// Read `len` bytes back from the start of a host-visible buffer.
    pub fn download(&self, buf: &VkBuffer, len: usize) -> Vec<u8> {
        assert!(len as vk::DeviceSize <= buf.size, "download overflows buffer");
        let mut out = vec![0u8; len];
        unsafe {
            let ptr = self
                .device
                .map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty())
                .expect("map_memory") as *const u8;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
            self.device.unmap_memory(buf.memory);
        }
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

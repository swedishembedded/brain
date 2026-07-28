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

use std::sync::Mutex;

use ash::vk;
use vulkan::context::{VkBuffer, VkContext};
use vulkan::shader;

use backend_api::{Backend, BufUsage, DeviceBuffer, Step};

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
}

impl VkOwnedBuffer {
    fn bytes(&self) -> u64 {
        self.inner.size
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

/// The Vulkan compute device.
pub struct VulkanBackend {
    ctx: VkContext,
    /// What this device can do — queried once at construction from the
    /// physical device (see `backend_api::DeviceCaps`).
    caps: backend_api::DeviceCaps,
    /// The device's real `maxStorageBufferRange`. Overrides the trait's fixed
    /// ~2 GiB default, which over-reports on devices with a smaller range
    /// (an oversized binding fails at dispatch, not allocation).
    max_storage_binding: u64,
    pipelines: Vec<KernelPipeline>,
    /// Each kernel's declared `@workgroup_size` (parallel to `pipelines`), so the
    /// grid is laid out with the size the kernel itself reconstructs its flat
    /// invocation id from — 64 for almost everything, 256 for the register-tiled
    /// GEMMs.
    wgsizes: Vec<u32>,
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
}

// ash handles are Send+Sync; all interior mutation goes through the Mutexes above.
unsafe impl Send for VulkanBackend {}
unsafe impl Sync for VulkanBackend {}

const POOL_MAX_SETS: u32 = 16384;

impl VulkanBackend {
    /// Try to build the Vulkan backend, compiling every kernel to a pipeline.
    /// Returns `Err` (so the caller can fall back to wgpu) if no Vulkan device is
    /// available or a kernel fails to compile.
    pub fn try_new(kernels: &[(&str, &str)]) -> Result<VulkanBackend, String> {
        let ctx = VkContext::new()?;
        log_adapter(&ctx.adapter_name);
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

        let pool = unsafe { new_pool(dev)? };
        let caps = Self::query_caps(&ctx);
        let max_storage_binding = unsafe {
            ctx.instance.get_physical_device_properties(ctx.physical_device)
        }
        .limits
        .max_storage_buffer_range as u64;
        Ok(VulkanBackend {
            ctx,
            caps,
            max_storage_binding,
            pipelines,
            wgsizes: backend_api::workgroup_sizes(kernels),
            pools: Mutex::new(vec![pool]),
            pending: Mutex::new(Vec::new()),
            uniforms: Mutex::new(Vec::new()),
            inflight_uniforms: Mutex::new(Vec::new()),
            free_uniforms: Mutex::new(std::collections::HashMap::new()),
            free_sets: Mutex::new(std::collections::HashMap::new()),
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
            peak_bandwidth_gbs: None,
            numeric: NumericSupport {
                f32: true,
                int8_dot: ctx.prec.dp4a,
                f16: false,
                coop_matrix: ctx.caps.feature_supported && !ctx.caps.shapes.is_empty(),
            },
        }
    }

    /// Allocate one descriptor set with `set_layout`, growing the pool list when
    /// the active pool is exhausted (never resets — see [`VulkanBackend::pools`]).
    fn alloc_set(&self, set_layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let dev = &self.ctx.device;
        let set_layouts = [set_layout];
        let mut pools = self.pools.lock().unwrap();
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
        VkOwnedBuffer { inner: b }
    }

    pub fn storage_init(&self, _name: &str, data: &[f32]) -> VkOwnedBuffer {
        let b = self.ctx.storage((data.len() * 4).max(4) as u64, vk::BufferUsageFlags::empty());
        if data.is_empty() {
            self.ctx.zero(&b);
        } else {
            self.ctx.upload(&b, bytemuck::cast_slice(data)); // fully covers the buffer
        }
        VkOwnedBuffer { inner: b }
    }

    pub fn buffer(&self, _label: &str, size: u64, usage: BufUsage) -> VkOwnedBuffer {
        let extra = if usage.contains(BufUsage::UNIFORM) {
            vk::BufferUsageFlags::UNIFORM_BUFFER
        } else {
            vk::BufferUsageFlags::empty()
        };
        let b = self.ctx.storage(size.max(4), extra);
        self.ctx.zero(&b);
        VkOwnedBuffer { inner: b }
    }

    pub fn uniform_dynamic(&self, len: usize) -> VkOwnedBuffer {
        // Host-visible: `write` then updates it by direct map (no staging
        // submit), matching its purpose — a caller-owned uniform rewritten
        // every iteration of a hot loop.
        let size = (((len * 4) + 15) / 16 * 16).max(16) as u64;
        let b = self.ctx.storage_host(size, vk::BufferUsageFlags::UNIFORM_BUFFER);
        self.ctx.zero(&b);
        VkOwnedBuffer { inner: b }
    }

    pub fn write(&self, buf: &VkOwnedBuffer, data: &[u32]) {
        // Ensure prior recorded compute completes before a host write, matching
        // the wgpu backend's flush-before-write ordering.
        self.flush();
        self.ctx.upload(&buf.inner, bytemuck::cast_slice(data));
    }

    pub fn read(&self, buf: &VkOwnedBuffer, n: usize) -> Vec<f32> {
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
        self.uniforms.lock().unwrap().push(ubuf);
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
        self.uniforms.lock().unwrap().push(ubuf);
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
        self.uniforms.lock().unwrap().len()
            + self.inflight_uniforms.lock().unwrap().len()
            + self.free_uniforms.lock().unwrap().values().map(Vec::len).sum::<usize>()
    }

    /// A transient per-dispatch uniform: recycled from `free_uniforms` when one
    /// of the right size is idle, else a fresh HOST-VISIBLE allocation. Both the
    /// zero (pad bytes) and the params write go through a direct map — building
    /// a dispatch performs NO queue submits. (The old DEVICE_LOCAL version cost
    /// a fill submit + a staged-copy submit — two blocking GPU round trips — per
    /// dispatch per frame, which serialized inference ~100x.)
    fn make_uniform(&self, params: &[u32]) -> VkBuffer {
        let size = ((params.len() * 4 + 15) / 16 * 16).max(16) as u64;
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
        let kp = &self.pipelines[kind];
        let dev = &self.ctx.device;
        // Transient sets recycle through `free_sets` (same pipeline => same
        // layout; the flush's fence wait made them idle, so rewriting below via
        // `update_descriptor_sets` is legal). Caller-held `step_buf` sets must
        // stay valid across flushes, so they always allocate fresh.
        let set = if transient {
            self.free_sets.lock().unwrap().get_mut(&kind).and_then(Vec::pop)
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
        let (gx, gy) = backend_api::grid_ws(threads, self.wgsizes[kind]);
        let sliced = offsets.iter().any(|&(off, _)| off > 0);
        VkStep { kind, set, gx, gy, sliced, transient }
    }

    pub fn submit(&self, clears: &[&VkOwnedBuffer], steps: &[VkStep]) {
        if !clears.is_empty() {
            // Match wgpu: complete prior work, then zero the buffers, before the
            // new steps (which may read them) are queued.
            self.flush();
            self.run_clears(clears);
        }
        self.pending.lock().unwrap().extend_from_slice(steps);
        // These steps' transient uniforms are now in flight: eligible for
        // recycling once the flush that runs them has fence-waited. Uniforms of
        // steps NOT yet submitted stay in `uniforms`, untouched by a flush that
        // races between their creation and their own submit.
        self.inflight_uniforms.lock().unwrap().append(&mut self.uniforms.lock().unwrap());
    }

    fn run_clears(&self, clears: &[&VkOwnedBuffer]) {
        let dev = &self.ctx.device;
        unsafe {
            let cmd = self.begin_cmd();
            for c in clears {
                dev.cmd_fill_buffer(cmd, c.inner.buffer, 0, c.inner.size, 0);
            }
            self.end_and_wait(cmd);
        }
    }

    /// Record all pending dispatches into one command buffer (with a storage
    /// memory barrier between consecutive dispatches), submit, fence-wait, then
    /// reclaim the batch's descriptor sets + transient uniform buffers.
    fn flush(&self) {
        let steps: Vec<VkStep> = std::mem::take(&mut *self.pending.lock().unwrap());
        if steps.is_empty() {
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
        // one per *frame*: on an NVIDIA P40 the TTS Talker forward was 2× slower).
        // Gate the workaround to Intel (vendor 0x8086). `BRAIN_VK_SERIAL` forces
        // it everywhere (diagnostic); `BRAIN_VK_NO_SERIAL` disables it (to confirm
        // the Intel bug on that hardware).
        const VENDOR_INTEL: u32 = 0x8086;
        let force_serial = std::env::var("BRAIN_VK_SERIAL").is_ok();
        let no_serial = std::env::var("BRAIN_VK_NO_SERIAL").is_ok();
        let vendor_needs_serial = self.ctx.vendor_id == VENDOR_INTEL;
        if !no_serial && (force_serial || (vendor_needs_serial && steps.iter().any(|s| s.sliced))) {
            unsafe {
                for s in &steps {
                    let cmd = self.begin_cmd();
                    let kp = &self.pipelines[s.kind];
                    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kp.pipeline);
                    dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, kp.layout, 0, &[s.set], &[]);
                    dev.cmd_dispatch(cmd, s.gx, s.gy, 1);
                    self.end_and_wait(cmd);
                }
            }
            self.recycle_transients(&steps);
            return;
        }
        unsafe {
            let cmd = self.begin_cmd();
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
                let kp = &self.pipelines[s.kind];
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
            }
            self.end_and_wait(cmd);
        }
        self.recycle_transients(&steps);
    }

    /// Recycle the flushed batch's TRANSIENT resources — the fence wait in
    /// `end_and_wait` has just proven them idle. Uniforms go back to the
    /// size-keyed pool, descriptor sets to the per-pipeline pool (deduped: the
    /// same step submitted twice in one batch must not donate its set twice).
    /// `step_buf` steps (transient = false) are caller-owned and left alone, so
    /// the `uniform_dynamic` reuse pattern keeps working across flushes.
    fn recycle_transients(&self, steps: &[VkStep]) {
        for u in std::mem::take(&mut *self.inflight_uniforms.lock().unwrap()) {
            self.free_uniforms.lock().unwrap().entry(u.size).or_default().push(u);
        }
        let mut seen = std::collections::HashSet::new();
        let mut free = self.free_sets.lock().unwrap();
        for s in steps {
            if s.transient && seen.insert(s.set) {
                free.entry(s.kind).or_default().push(s.set);
            }
        }
    }

    unsafe fn begin_cmd(&self) -> vk::CommandBuffer {
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
        cmd
    }

    unsafe fn end_and_wait(&self, cmd: vk::CommandBuffer) {
        self.ctx.submits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dev = &self.ctx.device;
        dev.end_command_buffer(cmd).expect("end cmd");
        let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).expect("fence");
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        dev.queue_submit(self.ctx.queue, &[submit], fence).expect("queue_submit");
        dev.wait_for_fences(&[fence], true, u64::MAX).expect("wait_for_fences");
        dev.destroy_fence(fence, None);
        dev.free_command_buffers(self.ctx.command_pool, &cmds);
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        let dev = &self.ctx.device;
        unsafe {
            let _ = dev.device_wait_idle();
            for u in std::mem::take(&mut *self.uniforms.lock().unwrap()) {
                self.ctx.destroy_buffer(u);
            }
            for u in std::mem::take(&mut *self.inflight_uniforms.lock().unwrap()) {
                self.ctx.destroy_buffer(u);
            }
            for (_, us) in std::mem::take(&mut *self.free_uniforms.lock().unwrap()) {
                for u in us {
                    self.ctx.destroy_buffer(u);
                }
            }
            for &pool in self.pools.lock().unwrap().iter() {
                dev.destroy_descriptor_pool(pool, None);
            }
            for kp in &self.pipelines {
                dev.destroy_pipeline(kp.pipeline, None);
                dev.destroy_pipeline_layout(kp.layout, None);
                dev.destroy_descriptor_set_layout(kp.set_layout, None);
                dev.destroy_shader_module(kp.module, None);
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

/// Neutral-handle bridge: downcast the opaque [`DeviceBuffer`]/[`Step`] back to
/// `VkOwnedBuffer`/[`VkStep`] and delegate to the inherent methods.
impl Backend for VulkanBackend {
    fn kind(&self) -> &'static str {
        "vulkan"
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
}

/// Register this backend under `"vulkan"`. The factory returns `Err` when no
/// Vulkan device/ICD is present, so the facade can fall back to wgpu.
pub fn register() {
    backend_api::register_backend("vulkan", |kernels| {
        VulkanBackend::try_new(kernels).map(|g| Box::new(g) as Box<dyn Backend>)
    });
}

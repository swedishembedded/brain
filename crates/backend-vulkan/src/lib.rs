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
    pipelines: Vec<KernelPipeline>,
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
    /// Per-dispatch uniform buffers (kept alive for the run; freed on drop).
    uniforms: Mutex<Vec<VkBuffer>>,
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
        Ok(VulkanBackend {
            ctx,
            pipelines,
            pools: Mutex::new(vec![pool]),
            pending: Mutex::new(Vec::new()),
            uniforms: Mutex::new(Vec::new()),
        })
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
        let size = (((len * 4) + 15) / 16 * 16).max(16) as u64;
        let b = self.ctx.storage(size, vk::BufferUsageFlags::UNIFORM_BUFFER);
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
        let step = self.record(kind, &ubuf, bufs, &[], threads);
        self.uniforms.lock().unwrap().push(ubuf);
        step
    }

    /// Build a dispatch around a caller-owned uniform buffer (reused across runs).
    pub fn step_buf(&self, kind: usize, ubuf: &VkOwnedBuffer, bufs: &[&VkOwnedBuffer], threads: u32) -> VkStep {
        self.record(kind, &ubuf.inner, bufs, &[], threads)
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
        let step = self.record(kind, &ubuf, bufs, offsets, threads);
        self.uniforms.lock().unwrap().push(ubuf);
        step
    }

    fn make_uniform(&self, params: &[u32]) -> VkBuffer {
        let size = ((params.len() * 4 + 15) / 16 * 16).max(16) as u64;
        let b = self.ctx.storage(size, vk::BufferUsageFlags::UNIFORM_BUFFER);
        self.ctx.zero(&b); // pad bytes beyond `params` must be 0
        if !params.is_empty() {
            self.ctx.upload(&b, bytemuck::cast_slice(params));
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
    ) -> VkStep {
        let kp = &self.pipelines[kind];
        let dev = &self.ctx.device;
        let set = self.alloc_set(kp.set_layout);

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
        let (gx, gy) = backend_api::grid(threads);
        let sliced = offsets.iter().any(|&(off, _)| off > 0);
        VkStep { kind, set, gx, gy, sliced }
    }

    pub fn submit(&self, clears: &[&VkOwnedBuffer], steps: &[VkStep]) {
        if !clears.is_empty() {
            // Match wgpu: complete prior work, then zero the buffers, before the
            // new steps (which may read them) are queued.
            self.flush();
            self.run_clears(clears);
        }
        self.pending.lock().unwrap().extend_from_slice(steps);
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
        // (sub-range) binding: Intel ANV's compute-compute pipeline barrier does
        // not reliably make a prior dispatch's writes visible across a non-zero
        // descriptor offset, but a queue-submit/fence boundary does. Only the
        // vocab-tiled embedding/lm_head use sliced bindings, so the (large) models
        // that tile pay this; everything else takes the fast single-submit path.
        // `BRAIN_VK_SERIAL` forces it for everything (diagnostic).
        let force_serial = std::env::var("BRAIN_VK_SERIAL").is_ok();
        if force_serial || steps.iter().any(|s| s.sliced) {
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
        // Descriptor sets + uniform buffers are intentionally NOT reclaimed here:
        // a step (and its set) may be re-submitted by the caller, so they live
        // until the backend drops (see `VulkanBackend::pools`).
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
}

/// Register this backend under `"vulkan"`. The factory returns `Err` when no
/// Vulkan device/ICD is present, so the facade can fall back to wgpu.
pub fn register() {
    backend_api::register_backend("vulkan", |kernels| {
        VulkanBackend::try_new(kernels).map(|g| Box::new(g) as Box<dyn Backend>)
    });
}

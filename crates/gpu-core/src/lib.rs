// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared GPU context + dispatch helpers (wgpu / Vulkan-Metal-DX12-GL).
//!
//! Both transformers (the sparse-MoE model and the PID event/effect model) build
//! their own kernel set but share this plumbing: device init, the uniform/bind
//! group/dispatch builder, a single-pass submit, and buffer read-back. Keeping it
//! here removes the device-init + dispatch duplication that used to live in both
//! the inference `Engine` and the training `Trainer`.

//! ## Backends
//!
//! By default this crate is the wgpu backend (`Gpu`). With the `cpu-backend`
//! feature it instead compiles the WGSL kernels to native code via
//! `brain-wgsl-cpu` and runs them across CPU cores — same `Gpu` API, same
//! `Step`/`DeviceBuffer` names, so model code is backend-agnostic. The neutral
//! `DeviceBuffer` alias and `BufUsage` type are what let the rest of the
//! workspace avoid naming `wgpu::*` directly.

/// Max workgroups per grid dimension (downlevel/Vulkan guarantee). The CPU
/// backend reproduces the same tiling so the kernels' index math is identical.
pub const MAX_GROUPS_PER_DIM: u32 = 65535;

/// Pack an f32 into the u32 uniform stream (kernels read it back with bitcast).
pub fn f(x: f32) -> u32 {
    x.to_bits()
}

/// Backend-neutral buffer usage flags. Mirrors the subset of
/// `wgpu::BufferUsages` the kernels need; the wgpu backend maps it back to
/// `wgpu::BufferUsages`, the CPU backend ignores it (all allocations are plain
/// host memory).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BufUsage(pub u32);

impl BufUsage {
    pub const STORAGE: BufUsage = BufUsage(1);
    pub const COPY_DST: BufUsage = BufUsage(2);
    pub const COPY_SRC: BufUsage = BufUsage(4);
    pub const UNIFORM: BufUsage = BufUsage(8);
    pub fn contains(self, other: BufUsage) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for BufUsage {
    type Output = BufUsage;
    fn bitor(self, rhs: BufUsage) -> BufUsage {
        BufUsage(self.0 | rhs.0)
    }
}

#[cfg(feature = "cpu-backend")]
mod cpu_backend;
#[cfg(feature = "cpu-backend")]
pub use cpu_backend::{CpuBackend as Gpu, CpuBuffer as DeviceBuffer, Step};

// The wgpu backend is the default; the CPU backend replaces it wholesale.
#[cfg(not(feature = "cpu-backend"))]
pub use wgpu::Buffer as DeviceBuffer;
#[cfg(not(feature = "cpu-backend"))]
pub use wgpu_backend::{Gpu, Step};

#[cfg(not(feature = "cpu-backend"))]
mod wgpu_backend {
use super::{BufUsage, MAX_GROUPS_PER_DIM};
use wgpu::util::DeviceExt;

/// A recorded dispatch: (pipeline index, bind group, grid_x, grid_y).
/// The grid is 1D (grid_y = 1) until the workgroup count exceeds the per-
/// dimension limit, then it tiles into Y; shaders reconstruct the linear thread
/// index from `num_workgroups`, so the split is transparent.
pub type Step = (usize, wgpu::BindGroup, u32, u32);

/// Log the selected adapter. Native prints to stderr; wasm has no stderr, so it
/// goes to the browser console.
#[cfg(not(target_arch = "wasm32"))]
fn log_adapter(info: &wgpu::AdapterInfo) {
    eprintln!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
}
#[cfg(target_arch = "wasm32")]
fn log_adapter(info: &wgpu::AdapterInfo) {
    web_sys::console::log_1(
        &format!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend).into(),
    );
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipelines: Vec<wgpu::ComputePipeline>,
}

impl Gpu {
    /// Initialise the device and compile `kernels` (name, WGSL source) into
    /// pipelines indexed by their position in the slice.
    ///
    /// Native blocking entry: wraps the async core in `pollster::block_on` so
    /// the existing synchronous call sites (CLI, training, tests) are unchanged.
    /// On wasm the browser has no blocking executor, so use `new_async` instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(kernels: &[(&str, &str)]) -> Gpu {
        pollster::block_on(Gpu::new_async(kernels))
    }

    /// Async device init + pipeline compile. This is the portable core used on
    /// both targets: native wraps it in `pollster::block_on` (see `new`), wasm
    /// awaits it from the wasm-bindgen entry point.
    pub async fn new_async(kernels: &[(&str, &str)]) -> Gpu {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("moe-rs-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
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

        let pipelines = kernels
            .iter()
            .map(|(name, src)| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl((*src).into()),
                });
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

        Gpu { device, queue, pipelines }
    }

    pub fn storage(&self, n: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn storage_init(&self, name: &str, data: &[f32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: u,
            mapped_at_creation: false,
        })
    }

    fn uniform(&self, data: &[u32]) -> wgpu::Buffer {
        let mut bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        while bytes.len() % 16 != 0 {
            bytes.push(0);
        }
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Build one dispatch around an already-allocated uniform buffer: bind group
    /// 0 = [ubuf, buffers...], grid = ceil(threads/64). The bind group keeps
    /// `ubuf` and `bufs` alive for the lifetime of the returned `Step`.
    pub fn step_buf(&self, kind: usize, ubuf: &wgpu::Buffer, bufs: &[&wgpu::Buffer], threads: u32) -> Step {
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
        let layout = self.pipelines[kind].get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let groups = ((threads + 63) / 64).max(1);
        if groups <= MAX_GROUPS_PER_DIM {
            (kind, bg, groups, 1)
        } else {
            // tile into a 2D grid; grid_x must stay a multiple of nothing special,
            // but using the full MAX width keeps the shader's `nwg.x * 64` stride exact.
            let gy = (groups + MAX_GROUPS_PER_DIM - 1) / MAX_GROUPS_PER_DIM;
            (kind, bg, MAX_GROUPS_PER_DIM, gy)
        }
    }

    /// Build one dispatch with a fresh single-use uniform buffer. Convenient for
    /// one-shot work; for hot loops prefer `uniform_dynamic` + `step_buf` so the
    /// uniform/bind group are allocated once and reused.
    pub fn step(&self, kind: usize, bufs: &[&wgpu::Buffer], params: &[u32], threads: u32) -> Step {
        self.step_buf(kind, &self.uniform(params), bufs, threads)
    }

    /// Clear the given buffers, then run all steps in a single compute pass
    /// (wgpu inserts the inter-dispatch barriers).
    pub fn submit(&self, clears: &[&wgpu::Buffer], steps: &[Step]) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for c in clears {
            enc.clear_buffer(c, 0, None);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for (kind, bg, gx, gy) in steps {
                pass.set_pipeline(&self.pipelines[*kind]);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*gx, *gy, 1);
            }
        }
        self.queue.submit(Some(enc.finish()));
    }

    pub fn write(&self, buf: &wgpu::Buffer, data: &[u32]) {
        self.queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
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
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    /// Copy a device buffer into a MAP_READ staging buffer and return its
    /// contents as f32. Native only: it blocks on `device.poll(wait)` + an mpsc
    /// recv, which is impossible in a browser. Wasm uses `read_async`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let staging = self.read_staging(buf, n);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
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
        let staging = self.read_staging(buf, n);
        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Schedule the mapping callbacks; on the WebGPU backend this is a no-op
        // beyond servicing the queue, the actual completion arrives via the
        // browser event loop while we await the oneshot.
        let _ = self.device.poll(wgpu::PollType::Poll);
        rx.await.expect("map_async channel dropped").expect("buffer map failed");
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        out
    }

    /// Shared staging-buffer copy used by both `read` (native) and `read_async`
    /// (wasm).
    fn read_staging(&self, buf: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, (n * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        staging
    }
}
} // mod wgpu_backend

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_packs_float_bits() {
        assert_eq!(f(1.0), 1.0f32.to_bits());
        assert_eq!(f(-2.5), (-2.5f32).to_bits());
        assert_eq!(f32::from_bits(f(3.14159)), 3.14159f32);
    }

    #[test]
    fn dispatch_storage_and_readback() {
        // Exercises the whole plumbing: device init, storage_init, step, submit, read.
        // The CPU backend needs no GPU, so it always runs; the wgpu backend skips
        // on headless CI via MOE_SKIP_GPU_TESTS.
        #[cfg(not(feature = "cpu-backend"))]
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let gpu = Gpu::new(&[("add2", kernels::ADD2)]);
        let a = gpu.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
        let b = gpu.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
        let out = gpu.storage(4);
        let step = gpu.step(0, &[&a, &b, &out], &[4], 4);
        gpu.submit(&[], &[step]);
        assert_eq!(gpu.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);
    }
}

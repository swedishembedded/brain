// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Compute-device facade.
//!
//! `Gpu` is a thin wrapper over a [`backend_api::Backend`] — the eager, per-step
//! compute contract. Model code holds a `Gpu` and calls `storage`/`step`/`submit`/
//! `read` without knowing which backend is behind it. The concrete backends live
//! in their own crates (`brain-backend-wgpu`, `brain-backend-cpu`,
//! `brain-backend-vulkan`), each implementing `Backend` and registering itself by
//! name; adding a backend is a new crate that depends only on `brain-backend-api`,
//! never this facade or its dispatch.
//!
//! The neutral [`DeviceBuffer`], [`Step`], and [`BufUsage`] types (re-exported
//! from `backend_api`) are what let the rest of the workspace avoid naming
//! `wgpu::*` / `ash::*` directly.
//!
//! ## Backend selection
//!
//! Native builds carry all three backends; a given `Gpu` uses exactly one, chosen
//! by [`set_default_backend`] (the CLI's `--device` flag) or `BRAIN_DEVICE`. On
//! wasm only the wgpu/WebGPU backend exists, so `Gpu` wraps it directly (no
//! `dyn`, and async device init / read-back via `new_async` / `read_async`).

pub use backend_api::{f, BufUsage, DeviceBuffer, Step};

// ---- native facade ----------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
mod native_facade {
    use super::{DeviceBuffer, Step};
    use backend_api::BufUsage;
    use std::sync::atomic::{AtomicU8, Ordering};

    /// Which compute backend a `Gpu` uses. A selection enum for the CLI — distinct
    /// from the [`backend_api::Backend`] *trait* the backends implement.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Backend {
        Wgpu,
        Cpu,
        /// Native Vulkan compute (ash + naga WGSL->SPIR-V). Falls back to wgpu if
        /// no Vulkan device/ICD is present.
        Vulkan,
    }

    static DEFAULT_BACKEND: AtomicU8 = AtomicU8::new(0); // 0=unset, 1=Wgpu, 2=Cpu, 3=Vulkan

    /// Set the process-wide default backend that `Gpu::new` builds. The CLI sets
    /// this from its `--device` flag. When unset, `BRAIN_DEVICE=cpu|vulkan`
    /// selects that backend; otherwise the default is wgpu.
    pub fn set_default_backend(b: Backend) {
        DEFAULT_BACKEND.store(
            match b {
                Backend::Wgpu => 1,
                Backend::Cpu => 2,
                Backend::Vulkan => 3,
            },
            Ordering::Relaxed,
        );
    }

    /// True iff a backend was explicitly selected via [`set_default_backend`]
    /// (i.e. the CLI saw a `--device` flag). Lets callers that historically
    /// defaulted to CPU (yolo) opt into the selected backend only when one was
    /// actually chosen.
    pub fn backend_selected() -> bool {
        DEFAULT_BACKEND.load(Ordering::Relaxed) != 0
    }

    /// The registry name of the selected backend.
    fn resolve_backend_name() -> &'static str {
        match DEFAULT_BACKEND.load(Ordering::Relaxed) {
            1 => "wgpu",
            2 => "cpu",
            3 => "vulkan",
            _ => {
                if let Ok(v) = std::env::var("BRAIN_DEVICE") {
                    if v.eq_ignore_ascii_case("cpu") {
                        return "cpu";
                    }
                    if v.eq_ignore_ascii_case("vulkan") {
                        return "vulkan";
                    }
                }
                "wgpu"
            }
        }
    }

    /// Register the built-in backends' factories once (idempotent).
    fn register_builtins() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            backend_wgpu::register();
            backend_cpu::register();
            backend_vulkan::register();
        });
    }

    /// The compute device: a runtime-selected eager backend behind a trait object.
    /// All are compiled in; a given instance uses exactly one. Adding a backend
    /// never touches this type — the dispatch just forwards to `self.inner`.
    pub struct Gpu {
        inner: Box<dyn backend_api::Backend>,
    }

    impl Gpu {
        /// Build the default backend (see [`set_default_backend`] / `BRAIN_DEVICE`).
        /// Vulkan falls back to wgpu when no Vulkan device/ICD is present, so the
        /// build stays usable everywhere.
        pub fn new(kernels: &[(&str, &str)]) -> Gpu {
            register_builtins();
            let name = resolve_backend_name();
            match backend_api::create_backend(name, kernels) {
                Ok(inner) => Gpu { inner },
                Err(e) if name == "vulkan" => {
                    eprintln!("brain: Vulkan backend unavailable ({e}); falling back to wgpu");
                    Gpu {
                        inner: backend_api::create_backend("wgpu", kernels)
                            .expect("wgpu backend must be available"),
                    }
                }
                Err(e) => panic!("failed to build backend '{name}': {e}"),
            }
        }

        /// Build on the native CPU backend regardless of the default selection.
        pub fn new_cpu(kernels: &[(&str, &str)]) -> Gpu {
            Gpu { inner: Box::new(backend_cpu::CpuBackend::new(kernels)) }
        }

        /// Build on the wgpu backend regardless of the default selection.
        pub fn new_wgpu(kernels: &[(&str, &str)]) -> Gpu {
            Gpu { inner: Box::new(backend_wgpu::WgpuBackend::new(kernels)) }
        }

        /// Build `count` wgpu devices on DISTINCT physical GPUs from one adapter
        /// enumeration — the reliable multi-GPU path (two separate `new_wgpu`
        /// calls can reorder and collide on one card). Returns one [`Gpu`] per card.
        #[cfg(not(target_arch = "wasm32"))]
        pub fn new_wgpu_multi(kernels: &[(&str, &str)], count: usize) -> Vec<Gpu> {
            backend_wgpu::WgpuBackend::new_multi(kernels, count)
                .into_iter()
                .map(|b| Gpu { inner: Box::new(b) })
                .collect()
        }

        /// Build on the native Vulkan backend, or `Err` if no Vulkan device is present.
        pub fn try_new_vulkan(kernels: &[(&str, &str)]) -> Result<Gpu, String> {
            backend_vulkan::VulkanBackend::try_new(kernels).map(|g| Gpu { inner: Box::new(g) })
        }

        pub fn storage(&self, n: u64) -> DeviceBuffer {
            self.inner.storage(n)
        }
        pub fn storage_init(&self, name: &str, data: &[f32]) -> DeviceBuffer {
            self.inner.storage_init(name, data)
        }
        pub fn buffer(&self, label: &str, size: u64, usage: BufUsage) -> DeviceBuffer {
            self.inner.buffer(label, size, usage)
        }
        pub fn uniform_dynamic(&self, len: usize) -> DeviceBuffer {
            self.inner.uniform_dynamic(len)
        }
        pub fn write(&self, buf: &DeviceBuffer, data: &[u32]) {
            self.inner.write(buf, data)
        }
        pub fn read(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
            self.inner.read(buf, n)
        }
        pub fn poll_wait(&self) {
            self.inner.poll_wait()
        }
        /// Largest single storage-buffer binding this device allows, in bytes —
        /// the hardware ceiling a kernel's biggest buffer must fit under. Used to
        /// pick attention backends per-card instead of assuming a fixed limit.
        pub fn max_storage_binding_bytes(&self) -> u64 {
            self.inner.max_storage_binding_bytes()
        }
        /// Start all recorded work on the device without waiting (frame
        /// pipelining: overlap host work with device compute; a later `read`
        /// synchronises). No-op on eager backends (CPU).
        pub fn flush(&self) {
            self.inner.flush()
        }
        pub fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step {
            crate::assert_no_output_alias(bufs);
            self.inner.step(kind, bufs, params, threads)
        }
        pub fn step_sliced(&self, kind: usize, bufs: &[&DeviceBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> Step {
            // NB: sliced views of ONE buffer at disjoint offsets are legal and common
            // here, so no alias check — wgpu validates the concrete ranges.
            self.inner.step_sliced(kind, bufs, offsets, params, threads)
        }
        pub fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step {
            self.inner.step_buf(kind, ubuf, bufs, threads)
        }
        pub fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]) {
            self.inner.submit(clears, steps)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native_facade::{backend_selected, set_default_backend, Backend, Gpu};

// ---- wasm facade ------------------------------------------------------------

// On wasm there is only the wgpu/WebGPU backend; `Gpu` wraps it concretely (no
// `dyn`), and device init / read-back are async (no blocking executor / poll in
// the browser). Model code that uses `gpu_core::{Gpu, DeviceBuffer, Step}`
// compiles unchanged against either target.
#[cfg(target_arch = "wasm32")]
mod wasm_facade {
    use super::{DeviceBuffer, Step};
    use backend_api::{Backend, BufUsage};
    use backend_wgpu::WgpuBackend;

    pub struct Gpu {
        inner: WgpuBackend,
    }

    impl Gpu {
        /// Async device init + pipeline compile (the browser has no blocking
        /// executor, so there is no synchronous `new`).
        pub async fn new_async(kernels: &[(&str, &str)]) -> Gpu {
            Gpu { inner: WgpuBackend::new_async(kernels).await }
        }

        pub fn storage(&self, n: u64) -> DeviceBuffer {
            Backend::storage(&self.inner, n)
        }
        pub fn storage_init(&self, name: &str, data: &[f32]) -> DeviceBuffer {
            Backend::storage_init(&self.inner, name, data)
        }
        pub fn buffer(&self, label: &str, size: u64, usage: BufUsage) -> DeviceBuffer {
            Backend::buffer(&self.inner, label, size, usage)
        }
        pub fn uniform_dynamic(&self, len: usize) -> DeviceBuffer {
            Backend::uniform_dynamic(&self.inner, len)
        }
        pub fn write(&self, buf: &DeviceBuffer, data: &[u32]) {
            Backend::write(&self.inner, buf, data)
        }
        pub fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step {
            crate::assert_no_output_alias(bufs);
            Backend::step(&self.inner, kind, bufs, params, threads)
        }
        pub fn step_sliced(&self, kind: usize, bufs: &[&DeviceBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> Step {
            Backend::step_sliced(&self.inner, kind, bufs, offsets, params, threads)
        }
        pub fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step {
            Backend::step_buf(&self.inner, kind, ubuf, bufs, threads)
        }
        pub fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]) {
            Backend::submit(&self.inner, clears, steps)
        }

        /// Async buffer readback (browser): awaits the map callback.
        pub async fn read_async(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
            self.inner.read_async_buf(buf, n).await
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm_facade::Gpu;

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
        // Exercises the whole plumbing: storage_init, step, submit, read. Uses the
        // CPU backend so it runs on headless CI with no GPU (both backends share
        // this dispatch contract, so it covers the wgpu path's plumbing too).
        let gpu = Gpu::new_cpu(&[("add2", kernels::ADD2)]);
        let a = gpu.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
        let b = gpu.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
        let out = gpu.storage(4);
        let step = gpu.step(0, &[&a, &b, &out], &[4], 4);
        gpu.submit(&[], &[step]);
        assert_eq!(gpu.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// Same dispatch contract on the native **Vulkan** backend, when a Vulkan
    /// device/ICD is present. Skips (does not fail) on hosts without Vulkan, so CI
    /// stays green; on a real device it validates the ash pipeline/descriptor/
    /// barrier/readback path against the CPU reference result.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn vulkan_dispatch_storage_and_readback() {
        let gpu = match Gpu::try_new_vulkan(&[("add2", kernels::ADD2)]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping vulkan test (no Vulkan device): {e}");
                return;
            }
        };
        let a = gpu.storage_init("a", &[1.0, 2.0, 3.0, 4.0]);
        let b = gpu.storage_init("b", &[10.0, 20.0, 30.0, 40.0]);
        let out = gpu.storage(4);
        let step = gpu.step(0, &[&a, &b, &out], &[4], 4);
        gpu.submit(&[], &[step]);
        assert_eq!(gpu.read(&out, 4), vec![11.0, 22.0, 33.0, 44.0]);

        // Multi-dispatch with a read-after-write dependency exercises the inter-
        // dispatch memory barrier: out2 = (a+b) + b.
        let out2 = gpu.storage(4);
        let s1 = gpu.step(0, &[&a, &b, &out], &[4], 4);
        let s2 = gpu.step(0, &[&out, &b, &out2], &[4], 4);
        gpu.submit(&[], &[s1, s2]);
        assert_eq!(gpu.read(&out2, 4), vec![21.0, 42.0, 63.0, 84.0]);
    }
}

/// Reject a dispatch that binds its OUTPUT buffer as an input as well.
///
/// wgpu treats `STORAGE_READ_WRITE` as exclusive within one dispatch, so
/// `f(x, y, x)` is a validation error there even though the CPU backend would
/// happily run it — the failure then shows up only on GPU, often surfacing at
/// an unrelated later call. Binding a buffer twice READ-ONLY is fine (the
/// chunked-attention trio deliberately binds one fused qkv buffer as both the
/// q and kv views), so only the output slot is checked. Kernels that must
/// accumulate into themselves have a dedicated `*_inplace` form with a single
/// read_write binding.
#[track_caller]
fn assert_no_output_alias(bufs: &[&DeviceBuffer]) {
    // The output is the last binding (for the `*_inplace` family it is the
    // first, but then the *other* operand is last, so the same test holds).
    if let Some(out) = bufs.last() {
        if bufs.iter().filter(|b| b.alloc_id() == out.alloc_id()).count() > 1 {
            panic!(
                "dispatch binds its output buffer as an input as well; this is a \
                 wgpu usage-scope violation (STORAGE_READ_WRITE is exclusive). \
                 Use the kernel's *_inplace form, or write to a distinct buffer."
            );
        }
    }
}

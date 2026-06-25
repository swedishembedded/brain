// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native CPU backend: runs the WGSL kernels (JIT-compiled by `brain-wgsl-cpu`)
//! across CPU cores. API-compatible with the wgpu `Gpu` so model code is
//! backend-agnostic; selected by the `cpu-backend` feature.
//!
//! A buffer is plain host memory (a `Vec<u32>`; every kernel element is 4 bytes).
//! `submit` runs the recorded steps sequentially — preserving the inter-dispatch
//! ordering the wgpu compute pass guarantees — and parallelises the invocations
//! *within* each step across a rayon pool. Each invocation owns a disjoint output
//! element, so the workers never alias their writes.

use crate::BufUsage;
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::sync::{Arc, Mutex};
use wgsl_cpu::Jit;

/// A recorded dispatch: (kernel index, bind group, grid_x, grid_y).
pub type Step = (usize, BindGroup, u32, u32);

struct BufInner {
    data: UnsafeCell<Vec<u32>>,
}
// A buffer's bytes are mutated single-threaded outside `submit`; inside `submit`
// the dispatcher hands disjoint sub-ranges to workers (upheld invariant).
unsafe impl Send for BufInner {}
unsafe impl Sync for BufInner {}

/// A device buffer: a reference-counted block of 4-byte words in host memory.
#[derive(Clone)]
pub struct CpuBuffer {
    inner: Arc<BufInner>,
}

impl CpuBuffer {
    fn with_words(words: Vec<u32>) -> CpuBuffer {
        CpuBuffer { inner: Arc::new(BufInner { data: UnsafeCell::new(words) }) }
    }
    fn zeros(n: usize) -> CpuBuffer {
        CpuBuffer::with_words(vec![0u32; n.max(1)])
    }
    #[allow(clippy::mut_from_ref)]
    fn words_mut(&self) -> &mut Vec<u32> {
        // Safe per the disjoint-access invariant documented on `BufInner`.
        unsafe { &mut *self.inner.data.get() }
    }
    fn base_ptr(&self) -> *mut u8 {
        self.words_mut().as_mut_ptr() as *mut u8
    }
}

/// The CPU compute backend.
pub struct CpuBackend {
    jit: Jit,
    threads: usize,
    /// Kernel names in index order (mirrors the registry passed to `new`), used
    /// for the optional `BRAIN_PROFILE=1` per-kernel timing breakdown.
    names: Vec<String>,
    /// Per-kernel accumulated wall time + call count for `BRAIN_PROFILE`.
    profile: Option<Mutex<Vec<(std::time::Duration, u64)>>>,
    /// Native fast-path kernel indices, resolved by name once at construction.
    /// All `None` (and the fast path off) under `BRAIN_NO_FASTCONV=1` / non-AVX2.
    fast: FastIdx,
}

/// Indices of the kernels that have a native CPU fast path (see `fast_conv` /
/// `fast_ops`). `None` if the kernel isn't registered for this model.
#[derive(Default)]
struct FastIdx {
    conv2d: Option<usize>,
    conv_act: Option<usize>,
    silu: Option<usize>,
    bn_eval: Option<usize>,
    concat2: Option<usize>,
    concat_split: Option<usize>,
    chan_place: Option<usize>,
    upsample2: Option<usize>,
}

/// Bind group for one dispatch: the uniform stream plus the storage buffers in
/// binding order (binding 1..). Holds `Arc` clones so the buffers outlive the step.
#[derive(Clone)]
pub struct BindGroup {
    uniform: CpuBuffer,
    bufs: Vec<CpuBuffer>,
}

impl CpuBackend {
    pub fn new(kernels: &[(&str, &str)]) -> CpuBackend {
        let jit = Jit::new(kernels).expect("WGSL->CPU JIT compilation failed");
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        eprintln!("adapter: brain-wgsl-cpu (Cranelift JIT, {threads} threads)");
        let names: Vec<String> = kernels.iter().map(|(n, _)| n.to_string()).collect();
        let profile = if std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false) {
            Some(Mutex::new(vec![(std::time::Duration::ZERO, 0u64); names.len()]))
        } else {
            None
        };
        let fast_off = std::env::var("BRAIN_NO_FASTCONV").map(|v| v != "0").unwrap_or(false);
        let fast = if fast_off || !crate::fast_conv::avx2_available() {
            FastIdx::default()
        } else {
            let find = |k: &str| names.iter().position(|n| n == k);
            FastIdx {
                conv2d: find("conv2d"),
                conv_act: find("conv_act"),
                silu: find("silu"),
                bn_eval: find("bn_eval"),
                concat2: find("concat2"),
                concat_split: find("concat_split"),
                chan_place: find("chan_place"),
                upsample2: find("upsample2"),
            }
        };
        CpuBackend { jit, threads, names, profile, fast }
    }

    pub fn storage(&self, n: u64) -> CpuBuffer {
        CpuBuffer::zeros(n as usize)
    }

    pub fn storage_init(&self, _name: &str, data: &[f32]) -> CpuBuffer {
        CpuBuffer::with_words(data.iter().map(|x| x.to_bits()).collect())
    }

    pub fn buffer(&self, _label: &str, size: u64, _usage: BufUsage) -> CpuBuffer {
        CpuBuffer::zeros((size / 4) as usize)
    }

    pub fn uniform_dynamic(&self, len: usize) -> CpuBuffer {
        CpuBuffer::zeros(len.max(4))
    }

    pub fn write(&self, buf: &CpuBuffer, data: &[u32]) {
        let w = buf.words_mut();
        if w.len() < data.len() {
            w.resize(data.len(), 0);
        }
        w[..data.len()].copy_from_slice(data);
    }

    pub fn read(&self, buf: &CpuBuffer, n: usize) -> Vec<f32> {
        let w = buf.words_mut();
        (0..n).map(|i| f32::from_bits(w[i])).collect()
    }

    pub fn poll_wait(&self) {}

    /// Build a dispatch around an already-allocated uniform buffer. Mirrors the
    /// wgpu backend's grid math so the kernels' index reconstruction is identical.
    pub fn step_buf(&self, kind: usize, ubuf: &CpuBuffer, bufs: &[&CpuBuffer], threads: u32) -> Step {
        let bg = BindGroup {
            uniform: ubuf.clone(),
            bufs: bufs.iter().map(|b| (*b).clone()).collect(),
        };
        let (gx, gy) = crate::grid(threads);
        (kind, bg, gx, gy)
    }

    pub fn step(&self, kind: usize, bufs: &[&CpuBuffer], params: &[u32], threads: u32) -> Step {
        let ubuf = CpuBuffer::with_words(params.to_vec());
        self.step_buf(kind, &ubuf, bufs, threads)
    }

    /// Zero the `clears`, then run every step in order (the dependency-preserving
    /// equivalent of wgpu's single compute pass), parallelising invocations within
    /// each step across the rayon pool.
    pub fn submit(&self, clears: &[&CpuBuffer], steps: &[Step]) {
        for c in clears {
            c.words_mut().iter_mut().for_each(|w| *w = 0);
        }
        for (kind, bg, gx, gy) in steps {
            let total = (*gx as u64) * (*gy as u64) * 64;
            let uniform = bg.uniform.base_ptr() as *const u32;
            let bufs: Vec<*mut u8> = bg.bufs.iter().map(|b| b.base_ptr()).collect();
            if let Some(prof) = &self.profile {
                let t = std::time::Instant::now();
                self.dispatch(*kind, total, *gx, *gy, uniform, &bufs);
                let dt = t.elapsed();
                let mut g = prof.lock().unwrap();
                g[*kind].0 += dt;
                g[*kind].1 += 1;
            } else {
                self.dispatch(*kind, total, *gx, *gy, uniform, &bufs);
            }
        }
    }

    /// Print the accumulated per-kernel timing breakdown (only if `BRAIN_PROFILE`
    /// was set). Sorted by total time descending.
    pub fn dump_profile(&self) {
        let Some(prof) = &self.profile else { return };
        let g = prof.lock().unwrap();
        let mut rows: Vec<(usize, std::time::Duration, u64)> =
            g.iter().enumerate().map(|(i, (d, c))| (i, *d, *c)).filter(|r| r.2 > 0).collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        let total: std::time::Duration = g.iter().map(|(d, _)| *d).sum();
        eprintln!("=== BRAIN_PROFILE (CPU backend, total {:.1} ms) ===", total.as_secs_f64() * 1e3);
        for (i, d, c) in rows {
            eprintln!(
                "  {:<16} {:8.1} ms  {:5} calls  ({:4.1}%)",
                self.names[i],
                d.as_secs_f64() * 1e3,
                c,
                d.as_secs_f64() / total.as_secs_f64().max(1e-9) * 100.0,
            );
        }
    }

    fn dispatch(
        &self,
        kind: usize,
        total: u64,
        gx: u32,
        gy: u32,
        uniform: *const u32,
        bufs: &[*mut u8],
    ) {
        if total == 0 {
            return;
        }
        // Native fast paths: same math as the WGSL kernels (validated against the
        // scalar reference), but structured loops / bulk copies / AVX2 instead of
        // the one-invocation-per-element JIT loop. Anything else falls through to
        // the JIT below. All `unsafe` here reconstructs slices from the bound
        // storage bases, each sized to its tensor by the model.
        let f = &self.fast;
        if Some(kind) == f.conv2d && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 10);
                let p = crate::fast_conv::ConvParams::from_u32(pu);
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, p.x_len());
                let w = std::slice::from_raw_parts(bufs[1] as *const f32, p.w_len());
                let y = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, p.y_len());
                crate::fast_conv::conv2d(&p, x, w, y);
            }
            return;
        }
        if Some(kind) == f.conv_act && bufs.len() >= 4 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 10);
                let p = crate::fast_conv::ConvParams::from_u32(pu);
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, p.x_len());
                let w = std::slice::from_raw_parts(bufs[1] as *const f32, p.w_len());
                let sb = std::slice::from_raw_parts(bufs[2] as *const f32, 2 * p.cout);
                let y = std::slice::from_raw_parts_mut(bufs[3] as *mut f32, p.y_len());
                crate::fast_conv::conv2d_act(&p, x, w, sb, y);
            }
            return;
        }
        if Some(kind) == f.silu && bufs.len() >= 2 {
            unsafe {
                let total = *uniform as usize;
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, total);
                let out = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, total);
                crate::fast_ops::silu(x, out);
            }
            return;
        }
        if Some(kind) == f.bn_eval && bufs.len() >= 5 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 4);
                let (n, c, h, w) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize);
                let len = n * c * h * w;
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, len);
                let mv = std::slice::from_raw_parts(bufs[1] as *const f32, 2 * c);
                let gb = std::slice::from_raw_parts(bufs[2] as *const f32, 2 * c);
                let out = std::slice::from_raw_parts_mut(bufs[3] as *mut f32, len);
                crate::fast_ops::bn_eval(pu, x, mv, gb, out);
            }
            return;
        }
        if Some(kind) == f.concat2 && bufs.len() >= 4 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 5);
                let (n, ca, cb, h, w) =
                    (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize);
                let hw = h * w;
                let a = std::slice::from_raw_parts(bufs[0] as *const f32, n * ca * hw);
                let b = std::slice::from_raw_parts(bufs[1] as *const f32, n * cb * hw);
                let y = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, n * (ca + cb) * hw);
                crate::fast_ops::concat2(pu, a, b, y);
            }
            return;
        }
        if Some(kind) == f.concat_split && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (n, ctot, csrc, _off, h, w) = (
                    pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize, pu[5] as usize,
                );
                let hw = h * w;
                let dy = std::slice::from_raw_parts(bufs[0] as *const f32, n * ctot * hw);
                let da = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, n * csrc * hw);
                crate::fast_ops::concat_split(pu, dy, da);
            }
            return;
        }
        if Some(kind) == f.chan_place && bufs.len() >= 2 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (n, ctot, csrc, _off, h, w) = (
                    pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize, pu[5] as usize,
                );
                let hw = h * w;
                let src = std::slice::from_raw_parts(bufs[0] as *const f32, n * csrc * hw);
                let dst = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, n * ctot * hw);
                crate::fast_ops::chan_place(pu, src, dst);
            }
            return;
        }
        if Some(kind) == f.upsample2 && bufs.len() >= 2 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 4);
                let (n, c, h, w) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize);
                let hw = h * w;
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, n * c * hw);
                let y = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, n * c * 4 * hw);
                crate::fast_ops::upsample2(pu, x, y);
            }
            return;
        }
        // ~8 chunks per thread for load balance on divergent kernels (e.g. the
        // softmax row-loops whose trip count varies with the causal mask).
        let span = (self.threads as u64 * 8).max(1);
        let chunk = total.div_ceil(span).max(1);
        let starts: Vec<u64> = (0..total).step_by(chunk as usize).collect();
        let uni = SendConst(uniform);
        let bufs_ptr = SendMut(bufs.as_ptr());
        let jit = &self.jit;
        starts.par_iter().for_each(|&s| {
            // Rebind whole wrappers so the closure captures the `Send` newtypes,
            // not their raw-pointer fields (Rust 2021 disjoint capture).
            let uni = uni;
            let bufs_ptr = bufs_ptr;
            let e = (s + chunk).min(total);
            // SAFETY: each invocation writes a disjoint output element, so the
            // sub-ranges never alias; `bufs` outlives this scoped parallel loop.
            unsafe { jit.run(kind, s, e, gx, gy, uni.0, bufs_ptr.0) };
        });
    }
}

impl Drop for CpuBackend {
    fn drop(&mut self) {
        self.dump_profile();
    }
}

#[derive(Clone, Copy)]
struct SendConst(*const u32);
unsafe impl Send for SendConst {}
unsafe impl Sync for SendConst {}

#[derive(Clone, Copy)]
struct SendMut(*const *mut u8);
unsafe impl Send for SendMut {}
unsafe impl Sync for SendMut {}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native CPU eager [`Backend`]: runs the WGSL kernels (JIT-compiled by
//! `brain-wgsl-cpu`) across CPU cores. API-compatible with the wgpu backend so
//! model code is backend-agnostic.
//!
//! A buffer is plain host memory (a `Vec<u32>`; every kernel element is 4 bytes).
//! `submit` runs the recorded steps sequentially — preserving the inter-dispatch
//! ordering the wgpu compute pass guarantees — and parallelises the invocations
//! *within* each step across a rayon pool. Each invocation owns a disjoint output
//! element, so the workers never alias their writes.
//!
//! The inherent methods operate on native `CpuBuffer`/[`CpuStep`]; the thin
//! `impl Backend` downcasts the neutral [`DeviceBuffer`]/[`Step`] handles and
//! delegates to them.

mod fast_conv;
mod fast_ops;

use backend_api::{Backend, BufUsage, DeviceBuffer, Step};
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::sync::{Arc, Mutex};
use wgsl_cpu::Jit;

/// A recorded dispatch: (kernel index, bind group, grid_x, grid_y).
pub type CpuStep = (usize, BindGroup, u32, u32);

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
    /// Each kernel's declared `@workgroup_size` (parallel to `names`). The CPU
    /// dispatcher needs it for two things: laying out the same grid the GPU
    /// backends do, and turning that grid back into an invocation count.
    wgsizes: Vec<u32>,
}

/// Indices of the kernels that have a native CPU fast path (see `fast_conv` /
/// `fast_ops`). `None` if the kernel isn't registered for this model.
#[derive(Default)]
struct FastIdx {
    matmul: Option<usize>,
    matmul_tiled: Option<usize>,
    matmul_reg: Option<usize>,
    matmul_reg2: Option<usize>,
    matmul_dx: Option<usize>,
    matmul_dx_reg: Option<usize>,
    matmul_dw: Option<usize>,
    matmul_dw_reg: Option<usize>,
    conv2d: Option<usize>,
    conv_act: Option<usize>,
    silu: Option<usize>,
    // Weight-tiled (workgroup-memory) conv variants: on CPU they route to the
    // same native fast paths as conv2d/conv_act (the tiling only helps the GPU).
    conv2d_tiled: Option<usize>,
    conv_act_tiled: Option<usize>,
    conv_act_reg: Option<usize>,
    conv_bias: Option<usize>,
    conv_bias_reg: Option<usize>,
    // Grouped/dilated conv (12-u32 ABI) + its register-tiled GPU variant: both
    // route to the per-group GEMM / depthwise fast path.
    conv2d_gd: Option<usize>,
    conv2d_gd_reg: Option<usize>,
    leaky_relu: Option<usize>,
    bn_eval: Option<usize>,
    gn_stats: Option<usize>,
    gn_part: Option<usize>,
    gn_stats2: Option<usize>,
    gn_apply: Option<usize>,
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
    /// `(buffer, word_offset)` — the offset lets a dispatch bind a sub-range of a
    /// buffer (e.g. a vocab tile of a >128MB embedding) so it stays within a
    /// backend's per-binding size limit, matching the wgpu offset binding.
    bufs: Vec<(CpuBuffer, usize)>,
}

impl CpuBackend {
    /// Kernel `kind`'s declared workgroup size.
    #[inline]
    fn wgsize(&self, kind: usize) -> u32 {
        self.wgsizes.get(kind).copied().unwrap_or(backend_api::DEFAULT_WORKGROUP_SIZE)
    }

    pub fn new(kernels: &[(&str, &str)]) -> CpuBackend {
        let jit = Jit::new(kernels).expect("WGSL->CPU JIT compilation failed");
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        // A process can build several engine instances (e.g. the TTS pipeline
        // makes one per component); log the adapter line only once.
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| eprintln!("adapter: brain-wgsl-cpu (Cranelift JIT, {threads} threads)"));
        let names: Vec<String> = kernels.iter().map(|(n, _)| n.to_string()).collect();
        let profile = if std::env::var("BRAIN_PROFILE").map(|v| v != "0").unwrap_or(false) {
            Some(Mutex::new(vec![(std::time::Duration::ZERO, 0u64); names.len()]))
        } else {
            None
        };
        let fast_off = std::env::var("BRAIN_NO_FASTCONV").map(|v| v != "0").unwrap_or(false);
        let fast = if fast_off || !fast_conv::avx2_available() {
            FastIdx::default()
        } else {
            let find = |k: &str| names.iter().position(|n| n == k);
            FastIdx {
                matmul: find("matmul"),
                matmul_tiled: find("matmul_tiled"),
                matmul_reg: find("matmul_reg"),
                matmul_reg2: find("matmul_reg2"),
                matmul_dx: find("matmul_dx"),
                matmul_dx_reg: find("matmul_dx_reg"),
                matmul_dw: find("matmul_dw"),
                matmul_dw_reg: find("matmul_dw_reg"),
                conv2d: find("conv2d"),
                conv_act: find("conv_act"),
                silu: find("silu"),
                conv2d_tiled: find("conv2d_tiled"),
                conv_act_tiled: find("conv_act_tiled"),
                conv_act_reg: find("conv_act_reg"),
                conv_bias: find("conv_bias"),
                conv_bias_reg: find("conv_bias_reg"),
                conv2d_gd: find("conv2d_gd"),
                conv2d_gd_reg: find("conv2d_gd_reg"),
                leaky_relu: find("leaky_relu"),
                bn_eval: find("bn_eval"),
                gn_stats: find("gn_stats"),
                gn_part: find("gn_part"),
                gn_stats2: find("gn_stats2"),
                gn_apply: find("gn_apply"),
                concat2: find("concat2"),
                concat_split: find("concat_split"),
                chan_place: find("chan_place"),
                upsample2: find("upsample2"),
            }
        };
        let wgsizes = backend_api::workgroup_sizes(kernels);
        CpuBackend { jit, threads, names, profile, fast, wgsizes }
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
    pub fn step_buf(&self, kind: usize, ubuf: &CpuBuffer, bufs: &[&CpuBuffer], threads: u32) -> CpuStep {
        let bg = BindGroup {
            uniform: ubuf.clone(),
            bufs: bufs.iter().map(|b| ((*b).clone(), 0usize)).collect(),
        };
        let (gx, gy) = backend_api::grid_ws(threads, self.wgsize(kind));
        (kind, bg, gx, gy)
    }

    /// Pad a uniform stream to a 16-byte multiple, matching the wgpu/vulkan
    /// backends' uniform padding. This is a safety property, not cosmetics: a
    /// kernel whose Params struct grew a trailing field (e.g. `conv_act`'s act
    /// selector) reads the pad word — in bounds, value 0 — from a caller that
    /// predates the field, instead of reading out of bounds.
    fn pad_uniform(params: &[u32]) -> Vec<u32> {
        let mut v = params.to_vec();
        v.resize(v.len().div_ceil(4).max(1) * 4, 0);
        v
    }

    pub fn step(&self, kind: usize, bufs: &[&CpuBuffer], params: &[u32], threads: u32) -> CpuStep {
        let ubuf = CpuBuffer::with_words(Self::pad_uniform(params));
        self.step_buf(kind, &ubuf, bufs, threads)
    }

    /// Like [`step`](Self::step) but each buffer carries a `(word_offset,
    /// word_len)` — the dispatch sees the buffer starting at `word_offset`
    /// (`word_len` is advisory on CPU; the kernel self-bounds via params).
    pub fn step_sliced(&self, kind: usize, bufs: &[&CpuBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> CpuStep {
        let ubuf = CpuBuffer::with_words(Self::pad_uniform(params));
        let bg = BindGroup {
            uniform: ubuf,
            bufs: bufs.iter().enumerate().map(|(i, b)| ((*b).clone(), offsets[i].0 as usize)).collect(),
        };
        let (gx, gy) = backend_api::grid_ws(threads, self.wgsize(kind));
        (kind, bg, gx, gy)
    }

    /// Zero the `clears`, then run every step in order (the dependency-preserving
    /// equivalent of wgpu's single compute pass), parallelising invocations within
    /// each step across the rayon pool.
    pub fn submit(&self, clears: &[&CpuBuffer], steps: &[CpuStep]) {
        for c in clears {
            c.words_mut().iter_mut().for_each(|w| *w = 0);
        }
        for (kind, bg, gx, gy) in steps {
            let total = (*gx as u64) * (*gy as u64) * self.wgsize(*kind) as u64;
            let uniform = bg.uniform.base_ptr() as *const u32;
            let bufs: Vec<*mut u8> = bg.bufs.iter().map(|(b, off)| unsafe { b.base_ptr().add(off * 4) }).collect();
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
        // matmul{,_tiled,_reg}.wgsl: out[M,N] = A[M,K] @ B[N,K]^T.
        // params = [m, k, n]; bufs = [A, B, out]. Same math for all three; the
        // tiled/register-tiled kernels are GPU-only (multi-barrier work-group
        // structure the JIT does not compile), so on CPU all three route to the
        // AVX2 gemm. That is the one-graph rule: a model may pick whichever
        // variant suits its shapes without forking its CPU path.
        if (Some(kind) == f.matmul || Some(kind) == f.matmul_tiled || Some(kind) == f.matmul_reg || Some(kind) == f.matmul_reg2)
            && bufs.len() >= 3
        {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 3);
                let (m, k, n) = (pu[0] as usize, pu[1] as usize, pu[2] as usize);
                let a = std::slice::from_raw_parts(bufs[0] as *const f32, m * k);
                let b = std::slice::from_raw_parts(bufs[1] as *const f32, n * k);
                let c = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, m * n);
                fast_ops::matmul_abt(a, b, c, m, k, n);
            }
            return;
        }
        // matmul_dx{,_reg}: dX[m,k] = sum_n dY[m,n]·W[n,k].  params = [m,k,n,acc];
        // bufs = [dY, W, dX]. The tiled `_reg` variant is GPU-only, so on CPU both
        // route to the same native backward GEMM (the one-graph rule for backprop).
        if (Some(kind) == f.matmul_dx || Some(kind) == f.matmul_dx_reg) && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 4);
                let (m, k, n, acc) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] != 0);
                let dy = std::slice::from_raw_parts(bufs[0] as *const f32, m * n);
                let w = std::slice::from_raw_parts(bufs[1] as *const f32, n * k);
                let dx = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, m * k);
                fast_ops::matmul_dx(dy, w, dx, m, k, n, acc);
            }
            return;
        }
        // matmul_dw{,_reg}: dW[n,k] += sum_m dY[m,n]·X[m,k].  params = [m,k,n];
        // bufs = [dY, X, dW]. Always accumulates.
        if (Some(kind) == f.matmul_dw || Some(kind) == f.matmul_dw_reg) && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 3);
                let (m, k, n) = (pu[0] as usize, pu[1] as usize, pu[2] as usize);
                let dy = std::slice::from_raw_parts(bufs[0] as *const f32, m * n);
                let x = std::slice::from_raw_parts(bufs[1] as *const f32, m * k);
                let dw = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, n * k);
                fast_ops::matmul_dw(dy, x, dw, m, k, n);
            }
            return;
        }
        if (Some(kind) == f.conv2d || Some(kind) == f.conv2d_tiled) && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 10);
                let p = fast_conv::ConvParams::from_u32(pu);
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, p.x_len());
                let w = std::slice::from_raw_parts(bufs[1] as *const f32, p.w_len());
                let y = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, p.y_len());
                fast_conv::conv2d(&p, x, w, y);
            }
            return;
        }
        if (Some(kind) == f.conv2d_gd || Some(kind) == f.conv2d_gd_reg) && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 12);
                let (p, groups) = fast_conv::ConvParams::from_u32_gd(pu);
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, p.x_len());
                let cin_g = p.cin / groups.max(1);
                let w = std::slice::from_raw_parts(bufs[1] as *const f32, p.cout * cin_g * p.k * p.k);
                let y = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, p.y_len());
                fast_conv::conv2d_gd(&p, groups, x, w, y);
            }
            return;
        }
        if Some(kind) == f.leaky_relu && bufs.len() >= 2 {
            unsafe {
                let total = *uniform as usize;
                let slope = f32::from_bits(*uniform.add(1));
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, total);
                let out = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, total);
                fast_ops::leaky_relu(x, out, slope);
            }
            return;
        }
        if (Some(kind) == f.conv_bias || Some(kind) == f.conv_bias_reg) && bufs.len() >= 4 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 10);
                let p = fast_conv::ConvParams::from_u32(pu);
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, p.x_len());
                let w = std::slice::from_raw_parts(bufs[1] as *const f32, p.w_len());
                let bias = std::slice::from_raw_parts(bufs[2] as *const f32, p.cout);
                let y = std::slice::from_raw_parts_mut(bufs[3] as *mut f32, p.y_len());
                fast_conv::conv2d_bias(&p, x, w, bias, y);
            }
            return;
        }
        if (Some(kind) == f.conv_act || Some(kind) == f.conv_act_tiled || Some(kind) == f.conv_act_reg)
            && bufs.len() >= 4
        {
            unsafe {
                // 11th word = activation selector (0 identity, 1 relu, 2 silu,
                // 3 sigmoid), mirroring the WGSL Params. The uniform buffer is
                // 16-byte padded, so the word exists even for a legacy 10-word
                // caller — and reads 0 (identity), which the vision dispatch
                // never emits (it always appends the act code).
                let pu = std::slice::from_raw_parts(uniform, 11);
                let p = fast_conv::ConvParams::from_u32(pu);
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, p.x_len());
                let w = std::slice::from_raw_parts(bufs[1] as *const f32, p.w_len());
                let sb = std::slice::from_raw_parts(bufs[2] as *const f32, 2 * p.cout);
                let y = std::slice::from_raw_parts_mut(bufs[3] as *mut f32, p.y_len());
                fast_conv::conv2d_act(&p, x, w, sb, y, pu[10]);
            }
            return;
        }
        if Some(kind) == f.silu && bufs.len() >= 2 {
            unsafe {
                let total = *uniform as usize;
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, total);
                let out = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, total);
                fast_ops::silu(x, out);
            }
            return;
        }
        if Some(kind) == f.gn_stats && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (n, c, h, w, g) =
                    (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize);
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, n * c * h * w);
                let stats = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, 2 * n * g);
                fast_ops::gn_stats(pu, x, stats);
            }
            return;
        }
        if Some(kind) == f.gn_part && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (n, c, h, w, g, pp) = (
                    pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize, pu[5] as usize,
                );
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, n * c * h * w);
                let part = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, 2 * n * g * pp);
                fast_ops::gn_part(pu, x, part);
            }
            return;
        }
        if Some(kind) == f.gn_stats2 && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 7);
                let (n, g, pp) = (pu[0] as usize, pu[4] as usize, pu[5] as usize);
                let part = std::slice::from_raw_parts(bufs[0] as *const f32, 2 * n * g * pp);
                let stats = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, 2 * n * g);
                fast_ops::gn_stats2(pu, part, stats);
            }
            return;
        }
        if Some(kind) == f.gn_apply && bufs.len() >= 5 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 5);
                let (n, c, h, w, g) =
                    (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize);
                let len = n * c * h * w;
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, len);
                let stats = std::slice::from_raw_parts(bufs[1] as *const f32, 2 * n * g);
                let gb = std::slice::from_raw_parts(bufs[2] as *const f32, 2 * c);
                let y = std::slice::from_raw_parts_mut(bufs[3] as *mut f32, len);
                fast_ops::gn_apply(pu, x, stats, gb, y);
            }
            return;
        }
        if Some(kind) == f.bn_eval && bufs.len() >= 5 {
            unsafe {
                // 5 words: NCHW + the act selector (pad word 0 for old callers).
                let pu = std::slice::from_raw_parts(uniform, 5);
                let (n, c, h, w) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize);
                let len = n * c * h * w;
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, len);
                let mv = std::slice::from_raw_parts(bufs[1] as *const f32, 2 * c);
                let gb = std::slice::from_raw_parts(bufs[2] as *const f32, 2 * c);
                let out = std::slice::from_raw_parts_mut(bufs[3] as *mut f32, len);
                fast_ops::bn_eval(pu, x, mv, gb, out);
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
                fast_ops::concat2(pu, a, b, y);
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
                fast_ops::concat_split(pu, dy, da);
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
                fast_ops::chan_place(pu, src, dst);
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
                fast_ops::upsample2(pu, x, y);
            }
            return;
        }
        // ~8 chunks per thread for load balance on divergent kernels (e.g. the
        // softmax row-loops whose trip count varies with the causal mask).
        let span = (self.threads as u64 * 8).max(1);
        let mut chunk = total.div_ceil(span).max(1);
        // Work-group kernels (workgroup memory + barriers) must be handed whole
        // workgroups per chunk — a workgroup's invocations share scratch and a
        // barrier, so a chunk boundary may not fall mid-workgroup. Round the chunk
        // up to a multiple of the work-group size (the dispatch `total` is already
        // a whole number of workgroups).
        if let Some(wg) = self.jit.workgroup_size(kind) {
            let wg = wg as u64;
            chunk = chunk.div_ceil(wg) * wg;
        }
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

/// Neutral-handle bridge: downcast the opaque [`DeviceBuffer`]/[`Step`] back to
/// `CpuBuffer`/[`CpuStep`] and delegate to the inherent methods.
impl Backend for CpuBackend {
    fn storage(&self, n: u64) -> DeviceBuffer {
        DeviceBuffer::new(CpuBackend::storage(self, n))
    }
    fn storage_init(&self, name: &str, data: &[f32]) -> DeviceBuffer {
        DeviceBuffer::new(CpuBackend::storage_init(self, name, data))
    }
    fn buffer(&self, label: &str, size: u64, usage: BufUsage) -> DeviceBuffer {
        DeviceBuffer::new(CpuBackend::buffer(self, label, size, usage))
    }
    fn uniform_dynamic(&self, len: usize) -> DeviceBuffer {
        DeviceBuffer::new(CpuBackend::uniform_dynamic(self, len))
    }
    fn write(&self, buf: &DeviceBuffer, data: &[u32]) {
        CpuBackend::write(self, buf.downcast_ref::<CpuBuffer>(), data)
    }
    fn step(&self, kind: usize, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Step {
        let bs: Vec<&CpuBuffer> = bufs.iter().map(|b| b.downcast_ref::<CpuBuffer>()).collect();
        Step::new(CpuBackend::step(self, kind, &bs, params, threads))
    }
    fn step_sliced(&self, kind: usize, bufs: &[&DeviceBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> Step {
        let bs: Vec<&CpuBuffer> = bufs.iter().map(|b| b.downcast_ref::<CpuBuffer>()).collect();
        Step::new(CpuBackend::step_sliced(self, kind, &bs, offsets, params, threads))
    }
    fn step_buf(&self, kind: usize, ubuf: &DeviceBuffer, bufs: &[&DeviceBuffer], threads: u32) -> Step {
        let bs: Vec<&CpuBuffer> = bufs.iter().map(|b| b.downcast_ref::<CpuBuffer>()).collect();
        Step::new(CpuBackend::step_buf(self, kind, ubuf.downcast_ref::<CpuBuffer>(), &bs, threads))
    }
    fn submit(&self, clears: &[&DeviceBuffer], steps: &[Step]) {
        let cs: Vec<&CpuBuffer> = clears.iter().map(|b| b.downcast_ref::<CpuBuffer>()).collect();
        let ss: Vec<CpuStep> = steps.iter().map(|s| s.downcast_ref::<CpuStep>().clone()).collect();
        CpuBackend::submit(self, &cs, &ss);
    }
    fn read(&self, buf: &DeviceBuffer, n: usize) -> Vec<f32> {
        CpuBackend::read(self, buf.downcast_ref::<CpuBuffer>(), n)
    }
    fn poll_wait(&self) {
        CpuBackend::poll_wait(self)
    }
}

/// Register this backend under `"cpu"` so the facade can build it by name.
pub fn register() {
    backend_api::register_backend("cpu", |kernels| Ok(Box::new(CpuBackend::new(kernels))));
}

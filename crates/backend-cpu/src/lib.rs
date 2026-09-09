// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native CPU eager [`Backend`]: runs the WGSL kernels (JIT-compiled by
//! `brain-wgsl-cpu`) across CPU cores. API-compatible with the wgpu backend so
//! model code is backend-agnostic.
//!
//! Swedish Embedded AB implements neural-network inference on machines with no
//! GPU at all, for teams that cannot assume an accelerator is present in the
//! field. If your team needs expertise in running models on CPUs without a
//! deep-learning framework underneath them, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! A buffer is plain host memory (a `Vec<u32>`; every kernel element is 4 bytes).
//! `submit` runs the recorded steps sequentially - preserving the inter-dispatch
//! ordering the wgpu compute pass guarantees - and parallelises the invocations
//! *within* each step across a rayon pool. Each invocation owns a disjoint output
//! element, so the workers never alias their writes.
//!
//! The inherent methods operate on native `CpuBuffer`/[`CpuStep`]; the thin
//! `impl Backend` downcasts the neutral [`DeviceBuffer`]/[`Step`] handles and
//! delegates to them.

mod fast_conv;
pub mod fast_ops;
pub mod host_gemm;
pub mod roofline;

use backend_api::{Backend, BufUsage, DeviceBuffer, NumericSupport, Step};
pub mod par;

use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use wgsl_cpu::Jit;

/// A recorded dispatch: (kernel index, bind group, grid_x, grid_y).
pub type CpuStep = (usize, BindGroup, u32, u32);

/// `kind` values at or above this are a [`backend_api::NativeId`] index into
/// `CpuShared::natives`, not a JIT/`FastIdx` kernel index - `kernel-performance
/// .md` M8.10. No real kernel set gets anywhere near 2^30 entries, so this
/// never collides with a genuine JIT pipeline index; `dispatch` checks it
/// FIRST, before the `total == 0` early-out (a native step's own function
/// decides whether there is anything to do, not the grid math below, which a
/// native dispatch's `threads` never really described in the first place).
const NATIVE_BASE: usize = 1 << 30;

/// One registered native function's signature: the raw uniform pointer and
/// the bound buffer bases in bind order - exactly the shape `dispatch`'s own
/// hidden `FastIdx` if-ladder below already reconstructs slices from. A
/// native step is that SAME reconstruction, just reached by a registered id
/// instead of a kernel-name-index match.
type NativeHostFn = Arc<dyn Fn(*const u32, &[*mut u8]) + Send + Sync>;

/// One provider-registered native function - `kernel-performance.md` M8.10's
/// [`backend_api::Backend::register_native`]/`step_native` seam, the CPU
/// ISA-pack half of it.
struct NativeEntry {
    #[allow(dead_code)] // read only via BRAIN_PROFILE-style diagnostics, not yet wired
    name: &'static str,
    f: NativeHostFn,
}

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
/// The compiled, shareable state of a CPU backend. The CPU backend executes
/// eagerly - there is no per-handle command stream - so a second handle
/// ([`CpuBackend::share`], the `Backend::share` contract) is nothing but
/// another `Arc` onto this.
struct CpuShared {
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
    /// Whether AVX2 is available AND `BRAIN_NO_FASTCONV` did not disable it -
    /// the identical gate `fast` above was built from, reused so
    /// `register_native` refuses every `HostFn` name under the same
    /// conditions the hidden `FastIdx` if-ladder already refuses under (M8.10:
    /// "zero delta" means the ABI path and the backend-internal path agree on
    /// when AVX2 is/isn't used, not just on what it computes when used).
    fast_native_enabled: bool,
    /// Provider-registered native (SPIR-V or host-fn) kernels - `kernel-
    /// performance.md` M8.10/M8.11. Append-only; a [`backend_api::NativeId`]
    /// is this `Vec`'s index at registration time.
    natives: Mutex<Vec<NativeEntry>>,
}

/// Relaxed device-op counters. `submits`/`readbacks` are per call;
/// `dispatches`/`bind_groups`/`uniform_allocs` are per recorded step, matching
/// what the wgpu backend counts so the two are comparable.
#[derive(Default)]
struct OpCounters {
    submits: AtomicU64,
    dispatches: AtomicU64,
    readbacks: AtomicU64,
    bind_groups: AtomicU64,
    uniform_allocs: AtomicU64,
    writes: AtomicU64,
}

pub struct CpuBackend {
    shared: std::sync::Arc<CpuShared>,
    /// Device-op counters, always maintained (relaxed atomics are negligible
    /// next to a dispatch) - the same contract `backend-wgpu` implements.
    ///
    /// PER HANDLE, not per device: `Backend::stats` is documented as
    /// "accounting for THIS handle since its creation", and a caller measuring
    /// what one engine cost needs a counter its neighbours cannot move. That is
    /// why this sits on `CpuBackend` and not on the `Arc`-shared `CpuShared` -
    /// `share()` is an `Arc` clone, so counters living there would be
    /// device-global and every concurrent user would land in everyone's delta.
    ///
    /// They were missing entirely, so `stats()` fell through to the trait's
    /// `None` default and every caller counting device ops was blind here.
    /// Callers must report null for that; the one in-tree consumer wrote
    /// `.unwrap_or(0)` and turned "not counted" into "zero", which made an
    /// engine test pass vacuously. Counting removes the ambiguity at its source
    /// rather than teaching each caller to cope.
    stats: OpCounters,
}

/// Indices of the kernels that have a native CPU fast path (see `fast_conv` /
/// `fast_ops`). `None` if the kernel isn't registered for this model.
#[derive(Default)]
struct FastIdx {
    matmul: Option<usize>,
    matmul_tiled: Option<usize>,
    matmul_reg: Option<usize>,
    matmul_reg2: Option<usize>,
    matmul_reg3: Option<usize>,
    /// `matmul_reg3_grouped`: one dispatch over every expert's compacted row
    /// range. Not part of the `matmul*` equivalence class above - its rows
    /// come from device-written group tables, so it needs its own loop.
    matmul_reg3_grouped: Option<usize>,
    matmul_reg4: Option<usize>,
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
    attn_scores_cross: Option<usize>,
    // The coalesced twin of `attn_scores_cross` and the transpose that feeds
    // it. Without a native path here, a model that adopts the coalesced GPU
    // pair would LOSE the AVX2 GEMM on this backend - a GPU win paid for on
    // CPU.
    attn_scores_cross_kt: Option<usize>,
    kv_k_headt: Option<usize>,
    attn_softmax_cross: Option<usize>,
    attn_apply_cross: Option<usize>,
    // The BIDIRECTIONAL SELF-attention trio (`attn_{scores_qk,softmax_bidir,
    // apply_full}.wgsl`) a packed-sequence diffusion transformer runs - the
    // same three GEMM/softmax shapes as the cross family above, differing only
    // in that query and key length are one `seq_len` and the value buffer has
    // no fused-KV offset. Without these three the whole attention half of such
    // a forward runs one output element per JIT invocation: measured on
    // MiniMax-H3's real block shape (56 heads x 128, seq 960), 1.1 GFLOP/s for
    // `attn_apply_full` against the 29-62 GFLOP/s `matmul_abt` reaches on the
    // same host - a third of the entire DiT forward spent in one kernel.
    attn_scores_qk: Option<usize>,
    attn_softmax_bidir: Option<usize>,
    attn_apply_full: Option<usize>,
    moe_linear_gated: Option<usize>,
    moe_linear_gated_dx: Option<usize>,
    moe_linear_gated_dw: Option<usize>,
    gqa_scores: Option<usize>,
    attn_softmax: Option<usize>,
    gqa_apply: Option<usize>,
    gqa_bwd_dscores: Option<usize>,
    gqa_bwd_dv: Option<usize>,
    gqa_bwd_dq: Option<usize>,
    gqa_bwd_dk: Option<usize>,
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
    silu_mul: Option<usize>,
    scale_add: Option<usize>,
}

/// Bind group for one dispatch: the uniform stream plus the storage buffers in
/// binding order (binding 1..). Holds `Arc` clones so the buffers outlive the step.
#[derive(Clone)]
pub struct BindGroup {
    uniform: CpuBuffer,
    /// `(buffer, word_offset)` - the offset lets a dispatch bind a sub-range of a
    /// buffer (e.g. a vocab tile of a >128MB embedding) so it stays within a
    /// backend's per-binding size limit, matching the wgpu offset binding.
    bufs: Vec<(CpuBuffer, usize)>,
}

impl CpuBackend {
    /// Kernel `kind`'s declared workgroup size.
    #[inline]
    fn wgsize(&self, kind: usize) -> u32 {
        self.shared.wgsizes.get(kind).copied().unwrap_or(backend_api::DEFAULT_WORKGROUP_SIZE)
    }

    pub fn new(kernels: &[(&str, &str)]) -> CpuBackend {
        let jit = Jit::new(kernels).expect("WGSL->CPU JIT compilation failed");
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        // A process can build several engine instances (e.g. the TTS pipeline
        // makes one per component); log the adapter line only once.
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| eprintln!("adapter: brain-wgsl-cpu (Cranelift JIT, {threads} threads)"));
        let names: Vec<String> = kernels.iter().map(|(n, _)| n.to_string()).collect();
        let profile = if backend_api::profile_enabled() {
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
                matmul_reg3: find("matmul_reg3"),
                matmul_reg3_grouped: find("matmul_reg3_grouped"),
                matmul_reg4: find("matmul_reg4"),
                attn_scores_cross: find("attn_scores_cross"),
                attn_scores_cross_kt: find("attn_scores_cross_kt"),
                kv_k_headt: find("kv_k_headt"),
                attn_softmax_cross: find("attn_softmax_cross"),
                attn_apply_cross: find("attn_apply_cross"),
                attn_scores_qk: find("attn_scores_qk"),
                attn_softmax_bidir: find("attn_softmax_bidir"),
                attn_apply_full: find("attn_apply_full"),
                moe_linear_gated: find("moe_linear_gated"),
                moe_linear_gated_dx: find("moe_linear_gated_dx"),
                moe_linear_gated_dw: find("moe_linear_gated_dw"),
                gqa_scores: find("gqa_scores"),
                attn_softmax: find("attn_softmax"),
                gqa_apply: find("gqa_apply"),
                gqa_bwd_dscores: find("gqa_bwd_dscores"),
                gqa_bwd_dv: find("gqa_bwd_dv"),
                gqa_bwd_dq: find("gqa_bwd_dq"),
                gqa_bwd_dk: find("gqa_bwd_dk"),
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
                silu_mul: find("silu_mul"),
                scale_add: find("scale_add"),
            }
        };
        let wgsizes = backend_api::workgroup_sizes(kernels);
        let fast_native_enabled = !fast_off && fast_conv::avx2_available();
        CpuBackend {
            shared: std::sync::Arc::new(CpuShared {
                jit,
                threads,
                names,
                profile,
                fast,
                wgsizes,
                fast_native_enabled,
                natives: Mutex::new(Vec::new()),
            }),
            stats: OpCounters::default(),
        }
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
        self.stats.writes.fetch_add(1, Ordering::Relaxed);
    }

    /// [`Self::write`] at a word offset - the CPU backend has no staging
    /// concept, so this is a plain offset `copy_from_slice`; it exists only so
    /// callers streaming a large upload in chunks stay backend-portable.
    pub fn write_at(&self, buf: &CpuBuffer, offset_words: u64, data: &[u32]) {
        let off = offset_words as usize;
        let w = buf.words_mut();
        if w.len() < off + data.len() {
            w.resize(off + data.len(), 0);
        }
        w[off..off + data.len()].copy_from_slice(data);
        self.stats.writes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn read(&self, buf: &CpuBuffer, n: usize) -> Vec<f32> {
        self.stats.readbacks.fetch_add(1, Ordering::Relaxed);
        let w = buf.words_mut();
        (0..n).map(|i| f32::from_bits(w[i])).collect()
    }

    pub fn poll_wait(&self) {}

    /// Build a dispatch around an already-allocated uniform buffer. Mirrors the
    /// wgpu backend's grid math so the kernels' index reconstruction is identical.
    pub fn step_buf(&self, kind: usize, ubuf: &CpuBuffer, bufs: &[&CpuBuffer], threads: u32) -> CpuStep {
        self.stats.bind_groups.fetch_add(1, Ordering::Relaxed);
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
    /// selector) reads the pad word - in bounds, value 0 - from a caller that
    /// predates the field, instead of reading out of bounds.
    fn pad_uniform(params: &[u32]) -> Vec<u32> {
        let mut v = params.to_vec();
        v.resize(v.len().div_ceil(4).max(1) * 4, 0);
        v
    }

    pub fn step(&self, kind: usize, bufs: &[&CpuBuffer], params: &[u32], threads: u32) -> CpuStep {
        self.stats.uniform_allocs.fetch_add(1, Ordering::Relaxed);
        let ubuf = CpuBuffer::with_words(Self::pad_uniform(params));
        self.step_buf(kind, &ubuf, bufs, threads)
    }

    /// Like [`step`](Self::step) but each buffer carries a `(word_offset,
    /// word_len)` - the dispatch sees the buffer starting at `word_offset`
    /// (`word_len` is advisory on CPU; the kernel self-bounds via params).
    pub fn step_sliced(&self, kind: usize, bufs: &[&CpuBuffer], offsets: &[(u64, u64)], params: &[u32], threads: u32) -> CpuStep {
        self.stats.uniform_allocs.fetch_add(1, Ordering::Relaxed);
        self.stats.bind_groups.fetch_add(1, Ordering::Relaxed);
        let ubuf = CpuBuffer::with_words(Self::pad_uniform(params));
        let bg = BindGroup {
            uniform: ubuf,
            bufs: bufs.iter().enumerate().map(|(i, b)| ((*b).clone(), offsets[i].0 as usize)).collect(),
        };
        let (gx, gy) = backend_api::grid_ws(threads, self.wgsize(kind));
        (kind, bg, gx, gy)
    }

    /// [`backend_api::Backend::register_native`] - `kernel-performance.md`
    /// M8.10. This backend only ever recognises `NativeSpec::HostFn(name)`
    /// (no SPIR-V path exists on the CPU JIT); `name` is looked up against a
    /// FIXED table of functions this crate itself implements - see
    /// `backend_api::NativeSpec::HostFn`'s own doc for why the spec carries
    /// only a name, never a closure ("the backend that accepts this decides
    /// how it actually runs"). Refuses (returns `None`) under the exact same
    /// condition the hidden `FastIdx` if-ladder already refuses under
    /// (`fast_native_enabled`) -
    /// see that field's own doc for why this must track it, not probe AVX2
    /// separately.
    pub fn register_native(&self, spec: &backend_api::NativeSpec) -> Option<backend_api::NativeId> {
        let backend_api::NativeSpec::HostFn(name) = spec else {
            return None; // no SPIR-V path on the CPU JIT
        };
        if !self.shared.fast_native_enabled {
            return None;
        }
        let f: NativeHostFn = match *name {
            // M8.10: the existing AVX2 F32 GEMM, reached through the ABI
            // instead of `dispatch`'s hidden `f.matmul` if-ladder arm - same
            // call, same params layout (`[m, k, n]` + `[A, B, out]`).
            "cpu_matmul_abt" => Arc::new(|uniform: *const u32, bufs: &[*mut u8]| unsafe {
                let pu = std::slice::from_raw_parts(uniform, 3);
                let (m, k, n) = (pu[0] as usize, pu[1] as usize, pu[2] as usize);
                let a = std::slice::from_raw_parts(bufs[0] as *const f32, m * k);
                let b = std::slice::from_raw_parts(bufs[1] as *const f32, n * k);
                let c = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, m * n);
                fast_ops::matmul_abt(a, b, c, m, k, n);
            }),
            // M8.11: the new AVX2 packed-int8 GEMM. `[m, kg, n]` + `[xq, wq,
            // sx, sw, out]` - see `fast_ops::matmul_i8_dyn`'s own doc for the
            // exact buffer shapes and the WGSL kernels this reproduces.
            "cpu_matmul_i8_dyn" => Arc::new(|uniform: *const u32, bufs: &[*mut u8]| unsafe {
                let pu = std::slice::from_raw_parts(uniform, 3);
                let (m, kg, n) = (pu[0] as usize, pu[1] as usize, pu[2] as usize);
                let ng = kg / 8;
                let xq = std::slice::from_raw_parts(bufs[0] as *const u32, m * kg);
                let wq = std::slice::from_raw_parts(bufs[1] as *const u32, n * kg);
                let sx = std::slice::from_raw_parts(bufs[2] as *const f32, m);
                let sw = std::slice::from_raw_parts(bufs[3] as *const f32, n * ng);
                let out = std::slice::from_raw_parts_mut(bufs[4] as *mut f32, m * n);
                fast_ops::matmul_i8_dyn(xq, wq, sx, sw, out, m, kg, n);
            }),
            _ => return None,
        };
        let mut natives = self.shared.natives.lock().unwrap_or_else(|e| e.into_inner());
        natives.push(NativeEntry { name, f });
        Some(backend_api::NativeId((natives.len() - 1) as u32))
    }

    /// [`backend_api::Backend::step_native`] - records a dispatch of an `id`
    /// [`Self::register_native`] returned. Builds a [`CpuStep`] exactly like
    /// [`Self::step`] does, except `kind` is biased by [`NATIVE_BASE`] so
    /// `dispatch` routes it to the registered closure instead of the JIT/
    /// `FastIdx` path - see [`NATIVE_BASE`]'s own doc.
    pub fn step_native(&self, id: backend_api::NativeId, bufs: &[&CpuBuffer], params: &[u32], threads: u32) -> CpuStep {
        let kind = NATIVE_BASE + id.0 as usize;
        self.stats.uniform_allocs.fetch_add(1, Ordering::Relaxed);
        let ubuf = CpuBuffer::with_words(Self::pad_uniform(params));
        self.step_buf(kind, &ubuf, bufs, threads)
    }

    /// Zero the `clears`, then run every step in order (the dependency-preserving
    /// equivalent of wgpu's single compute pass), parallelising invocations within
    /// each step across the rayon pool.
    pub fn submit(&self, clears: &[&CpuBuffer], steps: &[CpuStep]) {
        self.stats.submits.fetch_add(1, Ordering::Relaxed);
        self.stats.dispatches.fetch_add(steps.len() as u64, Ordering::Relaxed);
        for c in clears {
            c.words_mut().iter_mut().for_each(|w| *w = 0);
        }
        for (kind, bg, gx, gy) in steps {
            let total = (*gx as u64) * (*gy as u64) * self.wgsize(*kind) as u64;
            let uniform = bg.uniform.base_ptr() as *const u32;
            let bufs: Vec<*mut u8> = bg.bufs.iter().map(|(b, off)| unsafe { b.base_ptr().add(off * 4) }).collect();
            if let Some(prof) = &self.shared.profile {
                let t = std::time::Instant::now();
                self.dispatch(*kind, total, *gx, *gy, uniform, &bufs);
                let dt = t.elapsed();
                let mut g = prof.lock().unwrap_or_else(|e| e.into_inner());
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
        let Some(prof) = &self.shared.profile else { return };
        let g = prof.lock().unwrap_or_else(|e| e.into_inner());
        let mut rows: Vec<(usize, std::time::Duration, u64)> =
            g.iter().enumerate().map(|(i, (d, c))| (i, *d, *c)).filter(|r| r.2 > 0).collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        let total: std::time::Duration = g.iter().map(|(d, _)| *d).sum();
        eprintln!("=== BRAIN_PROFILE (CPU backend, total {:.1} ms) ===", total.as_secs_f64() * 1e3);
        for (i, d, c) in rows {
            eprintln!(
                "  {:<16} {:8.1} ms  {:5} calls  ({:4.1}%)",
                self.shared.names[i],
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
        // Provider-registered native step (`kernel-performance.md` M8.10) -
        // checked BEFORE `total == 0`: a native function decides its own work
        // from the uniform it reads, not from `total`/`gx`/`gy` (those describe
        // a JIT invocation grid a native step never has - see
        // `CpuBackend::step_native`'s own doc).
        if kind >= NATIVE_BASE {
            let natives = self.shared.natives.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = natives.get(kind - NATIVE_BASE) {
                (entry.f)(uniform, bufs);
            }
            return;
        }
        if total == 0 {
            return;
        }
        // Native fast paths: same math as the WGSL kernels (validated against the
        // scalar reference), but structured loops / bulk copies / AVX2 instead of
        // the one-invocation-per-element JIT loop. Anything else falls through to
        // the JIT below. All `unsafe` here reconstructs slices from the bound
        // storage bases, each sized to its tensor by the model.
        let f = &self.shared.fast;
        // matmul{,_tiled,_reg}.wgsl: out[M,N] = A[M,K] @ B[N,K]^T.
        // params = [m, k, n]; bufs = [A, B, out]. Same math for all three; the
        // tiled/register-tiled kernels are GPU-only (multi-barrier work-group
        // structure the JIT does not compile), so on CPU all of them route to the
        // AVX2 gemm. That is the one-graph rule: a model may pick whichever
        // variant suits its shapes without forking its CPU path. `matmul_reg3`
        // (reg2 with the bank conflicts removed) is bit-identical to reg2 by
        // construction, so it belongs to exactly the same equivalence class, and
        // `matmul_reg4` (reg3 re-laid-out for vec4 shared reads) likewise changes
        // only the shared-memory layout, never the arithmetic.
        if (Some(kind) == f.matmul
            || Some(kind) == f.matmul_tiled
            || Some(kind) == f.matmul_reg
            || Some(kind) == f.matmul_reg2
            || Some(kind) == f.matmul_reg3
            || Some(kind) == f.matmul_reg4)
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
        // matmul_reg3_grouped: ONE dispatch over every expert's compacted row
        // range (`model::moe::expert_fwd_grouped`). params = [k, n, n_experts];
        // bufs = [x, w, out, group_row_start, group_row_count, group_tile_start].
        // The GPU kernel resolves a workgroup's expert from `group_tile_start`
        // and then runs `matmul_reg3`'s body over that expert's rows; on CPU
        // the same work is the per-expert loop below, each block routed to the
        // same AVX2 GEMM the whole `matmul*` family uses. `group_tile_start`
        // exists only to map a fixed GPU tile grid onto variable row counts,
        // so it is read for nothing here - the row tables say everything.
        //
        // The compacted row count is not in the params (it is device-computed),
        // so the buffer bounds come from the tables: the highest row any expert
        // claims. Rows past that are the host's worst-case slack and are left
        // untouched, exactly as the GPU kernel's own bound checks leave them.
        //
        // The expert blocks run in PARALLEL, which is the whole reason this
        // routing is worth having on CPU as well as GPU. `matmul_abt`'s own
        // parallelism is over OUTPUT ROWS, and a decode round gives each
        // routed expert exactly one row - so a sequential walk would run the
        // layer's `top_k` GEMVs one after another on one core while the rest
        // of the pool idles, and each of those GEMVs streams a whole
        // `[out, in]` expert matrix, which is bandwidth work that wants every
        // core. Across experts is the only axis with parallelism at that
        // shape, and it is exactly the axis the per-expert-dispatch kernel
        // this replaced could not use.
        if Some(kind) == f.matmul_reg3_grouped && bufs.len() >= 5 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 3);
                let (k, n, e) = (pu[0] as usize, pu[1] as usize, pu[2] as usize);
                let starts = std::slice::from_raw_parts(bufs[3] as *const u32, e);
                let counts = std::slice::from_raw_parts(bufs[4] as *const u32, e);
                let rows = (0..e).map(|i| starts[i] as usize + counts[i] as usize).max().unwrap_or(0);
                if rows == 0 {
                    return;
                }
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, rows * k);
                let w = std::slice::from_raw_parts(bufs[1] as *const f32, e * n * k);
                let out = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, rows * n);
                let gemm = |ei: usize, o: &mut [f32]| {
                    let (s, c) = (starts[ei] as usize, counts[ei] as usize);
                    if c > 0 {
                        fast_ops::matmul_abt(&x[s * k..(s + c) * k], &w[ei * n * k..(ei + 1) * n * k], o, c, k, n);
                    }
                };
                // `row_start` is an exclusive scan of `row_count`, so the
                // experts' row blocks TILE the compacted batch in expert
                // order - which is what lets `out` be cut into disjoint
                // per-expert pieces with safe `split_at_mut` rather than
                // aliasing raw pointers. Verified rather than assumed: an
                // unexpected table falls back to the sequential walk, which
                // needs no such invariant.
                let mut at = 0u32;
                let tiled = starts.iter().zip(counts).all(|(s, c)| {
                    let ok = *s == at;
                    at += *c;
                    ok
                });
                if tiled {
                    let mut blocks: Vec<(usize, &mut [f32])> = Vec::with_capacity(e);
                    let mut rest: &mut [f32] = out;
                    for (ei, c) in counts.iter().enumerate() {
                        let (head, tail) = rest.split_at_mut(*c as usize * n);
                        blocks.push((ei, head));
                        rest = tail;
                    }
                    blocks.into_par_iter().for_each(|(ei, o)| gemm(ei, o));
                } else {
                    for ei in 0..e {
                        let s = starts[ei] as usize;
                        let c = counts[ei] as usize;
                        gemm(ei, &mut out[s * n..(s + c) * n]);
                    }
                }
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
        // Cross-attention trio (query-chunked bidirectional attention): per-head
        // packed GEMMs through the same AVX2 matmul_abt as the linear layers.
        //
        // Buffer lengths reconstruct from the uniform, and must be the EXACT
        // highest element the kernel touches - `bsz*t*stride` silently assumed
        // the base pointer was already advanced to the span's first row (i.e.
        // that `q_off`/`k_off`/`v_off` were region offsets only). A caller may
        // instead bind the buffer whole and carry the span's row offset in
        // those Params - which is what `model::vit::cross_q_fwd` does, because
        // a storage-binding offset must be 256B-aligned and `row0*stride` is
        // not for a ragged window partition. Both conventions land on the same
        // bound below; the old expression was too short for the second and
        // (marginally) too long for the first.
        let span_len = |n: usize, stride: usize, off: usize, heads: usize, hd: usize| {
            // `n == 0` is an empty dispatch; saturate rather than wrap, because
            // this length goes straight into `slice::from_raw_parts`.
            n.saturating_sub(1) * stride + off + heads * hd
        };
        if Some(kind) == f.attn_scores_cross && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 9);
                let (b, h, tq, tk, hd) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize);
                let (qs, kvs, qo, ko) = (pu[5] as usize, pu[6] as usize, pu[7] as usize, pu[8] as usize);
                let q = std::slice::from_raw_parts(bufs[0] as *const f32, span_len(b * tq, qs, qo, h, hd));
                let kv = std::slice::from_raw_parts(bufs[1] as *const f32, span_len(b * tk, kvs, ko, h, hd));
                let sc = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, b * h * tq * tk);
                fast_ops::attn_scores_cross(q, kv, sc, b, h, tq, tk, hd, qs, kvs, qo, ko);
            }
            return;
        }
        if Some(kind) == f.kv_k_headt && bufs.len() >= 2 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 4);
                let (te, dm, kvs, ko) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize);
                let kv = std::slice::from_raw_parts(bufs[0] as *const f32, span_len(te, kvs, ko, 1, dm));
                let kt = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, dm * te);
                fast_ops::kv_k_headt(kv, kt, te, dm, kvs, ko);
            }
            return;
        }
        if Some(kind) == f.attn_scores_cross_kt && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 7);
                let (b, h, tq, tk, hd) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize);
                let (qs, qo) = (pu[5] as usize, pu[6] as usize);
                let q = std::slice::from_raw_parts(bufs[0] as *const f32, span_len(b * tq, qs, qo, h, hd));
                let kt = std::slice::from_raw_parts(bufs[1] as *const f32, h * hd * tk);
                let sc = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, b * h * tq * tk);
                fast_ops::attn_scores_cross_kt(q, kt, sc, b, h, tq, tk, hd, qs, qo);
            }
            return;
        }
        if Some(kind) == f.attn_softmax_cross && bufs.len() >= 2 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 4);
                let rows = (pu[0] * pu[1] * pu[2]) as usize;
                let tk = pu[3] as usize;
                let s = std::slice::from_raw_parts(bufs[0] as *const f32, rows * tk);
                let p = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, rows * tk);
                fast_ops::attn_softmax_cross(s, p, rows, tk);
            }
            return;
        }
        if Some(kind) == f.attn_apply_cross && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 8);
                let (b, h, tq, tk, hd) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize);
                let (kvs, vo, dm) = (pu[5] as usize, pu[6] as usize, pu[7] as usize);
                let p = std::slice::from_raw_parts(bufs[0] as *const f32, b * h * tq * tk);
                let kv = std::slice::from_raw_parts(bufs[1] as *const f32, span_len(b * tk, kvs, vo, h, hd));
                let o = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, b * tq * dm);
                fast_ops::attn_apply_cross(p, kv, o, b, h, tq, tk, hd, kvs, vo, dm);
            }
            return;
        }
        // Bidirectional self-attention trio. Each is the tq==tk, zero-offset
        // case of the cross kernel directly above it, so they route into the
        // SAME three fast ops rather than growing a second implementation of
        // the same three shapes.
        //
        // `attn_scores_qk` carries two uniforms the cross kernel has no
        // equivalent of: `causal` and an explicit `scale`. The scale is passed
        // through (never assumed to be 1/√hd); `causal != 0` deliberately
        // FALLS THROUGH to the JIT rather than being emulated, because the
        // masked variant is a different kernel shape and a wrong mask is
        // exactly the kind of silently-plausible output this repo's porting
        // rules exist to prevent.
        if Some(kind) == f.attn_scores_qk && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 7);
                let (b, h, s, hd) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize);
                let (qks, causal) = (pu[4] as usize, pu[5]);
                let scale = f32::from_bits(pu[6]);
                if causal == 0 {
                    let span = span_len(b * s, qks, 0, h, hd);
                    let q = std::slice::from_raw_parts(bufs[0] as *const f32, span);
                    let k = std::slice::from_raw_parts(bufs[1] as *const f32, span);
                    let sc = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, b * h * s * s);
                    fast_ops::attn_scores_qk(q, k, sc, b, h, s, hd, qks, scale);
                    return;
                }
            }
        }
        if Some(kind) == f.attn_softmax_bidir && bufs.len() >= 2 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 3);
                let t = pu[2] as usize;
                let rows = (pu[0] * pu[1]) as usize * t;
                let s = std::slice::from_raw_parts(bufs[0] as *const f32, rows * t);
                let p = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, rows * t);
                fast_ops::attn_softmax_cross(s, p, rows, t);
            }
            return;
        }
        if Some(kind) == f.attn_apply_full && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (b, h, t, hd) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize);
                let (vs, dm) = (pu[4] as usize, pu[5] as usize);
                let p = std::slice::from_raw_parts(bufs[0] as *const f32, b * h * t * t);
                let v = std::slice::from_raw_parts(bufs[1] as *const f32, span_len(b * t, vs, 0, h, hd));
                let o = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, b * t * dm);
                fast_ops::attn_apply_cross(p, v, o, b, h, t, t, hd, vs, 0, dm);
            }
            return;
        }
        // moe_linear_gated{,_dx,_dw}: matmul_abt/matmul_dx/matmul_dw with a
        // per-row (or per-summed-row) gate early-exit - see fast_ops.rs's own
        // doc for why this is the decode loop's dominant cost kernel.
        //
        // The trailing `w_off` param each of the three carries is the element
        // offset of THIS expert's `[n, k]` matrix inside the bound weight (or
        // gradient) buffer - 0 when every expert owns its own buffer, and
        // `e_idx * n * k` when the three projections are fused
        // `[n_experts, n, k]` banks. Applied here by advancing the base
        // pointer, which is exactly what the shader's own `p.w_off + ...`
        // index does.
        if Some(kind) == f.moe_linear_gated && bufs.len() >= 4 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (m, k, n, ne, e) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize);
                let w_off = pu[5] as usize;
                let x = std::slice::from_raw_parts(bufs[0] as *const f32, m * k);
                let w = std::slice::from_raw_parts((bufs[1] as *const f32).add(w_off), n * k);
                let gate = std::slice::from_raw_parts(bufs[2] as *const f32, m * ne);
                let out = std::slice::from_raw_parts_mut(bufs[3] as *mut f32, m * n);
                fast_ops::moe_linear_gated_fwd(x, w, gate, out, m, k, n, ne, e);
            }
            return;
        }
        if Some(kind) == f.moe_linear_gated_dx && bufs.len() >= 4 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 7);
                let (m, k, n, ne, e) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize);
                let acc = pu[5] != 0;
                let w_off = pu[6] as usize;
                let dy = std::slice::from_raw_parts(bufs[0] as *const f32, m * n);
                let w = std::slice::from_raw_parts((bufs[1] as *const f32).add(w_off), n * k);
                let gate = std::slice::from_raw_parts(bufs[2] as *const f32, m * ne);
                let dx = std::slice::from_raw_parts_mut(bufs[3] as *mut f32, m * k);
                fast_ops::moe_linear_gated_dx(dy, w, gate, dx, m, k, n, ne, e, acc);
            }
            return;
        }
        if Some(kind) == f.moe_linear_gated_dw && bufs.len() >= 4 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (m, k, n, ne, e) = (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize);
                let w_off = pu[5] as usize;
                let dy = std::slice::from_raw_parts(bufs[0] as *const f32, m * n);
                let x = std::slice::from_raw_parts(bufs[1] as *const f32, m * k);
                let gate = std::slice::from_raw_parts(bufs[2] as *const f32, m * ne);
                let dw = std::slice::from_raw_parts_mut((bufs[3] as *mut f32).add(w_off), n * k);
                fast_ops::moe_linear_gated_dw(dy, x, gate, dw, m, k, n, ne, e);
            }
            return;
        }
        // Self-attention family (gqa_scores / attn_softmax / gqa_apply + the
        // gqa_bwd_{dscores,dv,dq,dk} backward quartet): plain causal GQA
        // self-attention (MHA is `n_kv_heads == n_heads`), the shape every
        // decoder's own attention and SAM/CLIP's windowed/global attention
        // use. All buffers are contiguous, no stride/offset params (unlike
        // the cross-attention family above).
        if Some(kind) == f.gqa_scores && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (b, h, hkv, t, hd, group) =
                    (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize, pu[5] as usize);
                let q = std::slice::from_raw_parts(bufs[0] as *const f32, b * t * h * hd);
                let k = std::slice::from_raw_parts(bufs[1] as *const f32, b * t * hkv * hd);
                let sc = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, b * h * t * t);
                fast_ops::gqa_scores(q, k, sc, b, h, hkv, t, hd, group);
            }
            return;
        }
        if Some(kind) == f.attn_softmax && bufs.len() >= 2 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 3);
                let (b, h, t) = (pu[0] as usize, pu[1] as usize, pu[2] as usize);
                let s = std::slice::from_raw_parts(bufs[0] as *const f32, b * h * t * t);
                let p = std::slice::from_raw_parts_mut(bufs[1] as *mut f32, b * h * t * t);
                fast_ops::attn_softmax_causal(s, p, b, h, t);
            }
            return;
        }
        if Some(kind) == f.gqa_apply && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (b, h, hkv, t, hd, group) =
                    (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize, pu[5] as usize);
                let p = std::slice::from_raw_parts(bufs[0] as *const f32, b * h * t * t);
                let v = std::slice::from_raw_parts(bufs[1] as *const f32, b * t * hkv * hd);
                let ctx = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, b * t * h * hd);
                fast_ops::gqa_apply(p, v, ctx, b, h, hkv, t, hd, group);
            }
            return;
        }
        if Some(kind) == f.gqa_bwd_dscores && bufs.len() >= 4 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (b, h, hkv, t, hd, group) =
                    (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize, pu[5] as usize);
                let d_ctx = std::slice::from_raw_parts(bufs[0] as *const f32, b * t * h * hd);
                let v = std::slice::from_raw_parts(bufs[1] as *const f32, b * t * hkv * hd);
                let probs = std::slice::from_raw_parts(bufs[2] as *const f32, b * h * t * t);
                let d_scores = std::slice::from_raw_parts_mut(bufs[3] as *mut f32, b * h * t * t);
                fast_ops::gqa_bwd_dscores(d_ctx, v, probs, d_scores, b, h, hkv, t, hd, group);
            }
            return;
        }
        if Some(kind) == f.gqa_bwd_dv && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (b, h, hkv, t, hd, group) =
                    (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize, pu[5] as usize);
                let probs = std::slice::from_raw_parts(bufs[0] as *const f32, b * h * t * t);
                let d_ctx = std::slice::from_raw_parts(bufs[1] as *const f32, b * t * h * hd);
                let d_v = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, b * t * hkv * hd);
                fast_ops::gqa_bwd_dv(probs, d_ctx, d_v, b, h, hkv, t, hd, group);
            }
            return;
        }
        if Some(kind) == f.gqa_bwd_dq && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (b, h, hkv, t, hd, group) =
                    (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize, pu[5] as usize);
                let d_scores = std::slice::from_raw_parts(bufs[0] as *const f32, b * h * t * t);
                let k = std::slice::from_raw_parts(bufs[1] as *const f32, b * t * hkv * hd);
                let d_q = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, b * t * h * hd);
                fast_ops::gqa_bwd_dq(d_scores, k, d_q, b, h, hkv, t, hd, group);
            }
            return;
        }
        if Some(kind) == f.gqa_bwd_dk && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 6);
                let (b, h, hkv, t, hd, group) =
                    (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] as usize, pu[5] as usize);
                let d_scores = std::slice::from_raw_parts(bufs[0] as *const f32, b * h * t * t);
                let q = std::slice::from_raw_parts(bufs[1] as *const f32, b * t * h * hd);
                let d_k = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, b * t * hkv * hd);
                fast_ops::gqa_bwd_dk(d_scores, q, d_k, b, h, hkv, t, hd, group);
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
                // caller - and reads 0 (identity), which the vision dispatch
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
        // silu_mul.wgsl: out[i] = SiLU(a[i]) * b[i]. params = [total];
        // bufs = [a, b, out]. See fast_ops::silu_mul's own doc for why this
        // (not just moe_linear_gated/matmul) was the decode loop's next
        // promoted bottleneck.
        if Some(kind) == f.silu_mul && bufs.len() >= 3 {
            unsafe {
                let total = *uniform as usize;
                let a = std::slice::from_raw_parts(bufs[0] as *const f32, total);
                let b = std::slice::from_raw_parts(bufs[1] as *const f32, total);
                let out = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, total);
                fast_ops::silu_mul(a, b, out);
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
        // scale_add.wgsl: the MoE combine step for one expert. params =
        // [seq_len, d_model, n_experts, e_idx, accumulate]; bufs = [gate, src,
        // acc]. See fast_ops::scale_add's own doc - same finding/fix shape as
        // silu_mul above.
        if Some(kind) == f.scale_add && bufs.len() >= 3 {
            unsafe {
                let pu = std::slice::from_raw_parts(uniform, 5);
                let (seq_len, d_model, n_experts, e_idx, accumulate) =
                    (pu[0] as usize, pu[1] as usize, pu[2] as usize, pu[3] as usize, pu[4] != 0);
                let gate = std::slice::from_raw_parts(bufs[0] as *const f32, seq_len * n_experts);
                let src = std::slice::from_raw_parts(bufs[1] as *const f32, seq_len * d_model);
                let acc = std::slice::from_raw_parts_mut(bufs[2] as *mut f32, seq_len * d_model);
                fast_ops::scale_add(gate, src, acc, seq_len, d_model, n_experts, e_idx, accumulate);
            }
            return;
        }
        // ~8 chunks per thread for load balance on divergent kernels (e.g. the
        // softmax row-loops whose trip count varies with the causal mask).
        let span = (self.shared.threads as u64 * 8).max(1);
        let mut chunk = total.div_ceil(span).max(1);
        // Work-group kernels (workgroup memory + barriers) must be handed whole
        // workgroups per chunk - a workgroup's invocations share scratch and a
        // barrier, so a chunk boundary may not fall mid-workgroup. Round the chunk
        // up to a multiple of the work-group size (the dispatch `total` is already
        // a whole number of workgroups).
        if let Some(wg) = self.shared.jit.workgroup_size(kind) {
            let wg = wg as u64;
            chunk = chunk.div_ceil(wg) * wg;
        }
        let starts: Vec<u64> = (0..total).step_by(chunk as usize).collect();
        let uni = SendConst(uniform);
        let bufs_ptr = SendMut(bufs.as_ptr());
        let jit = &self.shared.jit;
        starts.par_iter().for_each(|&s| {
            // Rebind whole wrappers so the closure captures the `Send` newtypes,
            // not their raw-pointer fields (Rust 2021 disjoint capture).
            // Rebinding the whole `Send` newtype is REQUIRED, not redundant: under
            // Rust 2021 disjoint capture a closure that only touches `uni.0`
            // captures that raw pointer directly, which is not `Send`. Verified by
            // deletion - it fails with E0277 `*mut f32` cannot be shared between
            // threads safely.
            #[allow(clippy::redundant_locals)]
            let uni = uni;
            // Rebinding the whole `Send` newtype is REQUIRED, not redundant: under
            // Rust 2021 disjoint capture a closure that only touches `bufs_ptr.0`
            // captures that raw pointer directly, which is not `Send`. Verified by
            // deletion - it fails with E0277 `*mut f32` cannot be shared between
            // threads safely.
            #[allow(clippy::redundant_locals)]
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

// Neutral-handle bridge: downcast the opaque `DeviceBuffer`/`Step` back to
// `CpuBuffer`/`CpuStep` and delegate to the inherent methods.

/// Weak handle onto the compiled JIT state - `backend_api::Backend::downgrade`.
struct WeakCpu(std::sync::Weak<CpuShared>);

impl backend_api::WeakBackend for WeakCpu {
    fn upgrade(&self) -> Option<Box<dyn Backend>> {
        Some(Box::new(CpuBackend { shared: self.0.upgrade()?, stats: OpCounters::default() }))
    }
}

impl Backend for CpuBackend {
    fn kind(&self) -> &'static str {
        "cpu"
    }
    fn dump_profile(&self) {
        CpuBackend::dump_profile(self)
    }
    /// Device-op accounting - the same counters `backend-wgpu` reports, so a
    /// caller asking "how many submits/readbacks did this run cost" gets a real
    /// answer on every backend instead of `None` on this one.
    fn stats(&self) -> Option<backend_api::DeviceStats> {
        let c = &self.stats;
        Some(backend_api::DeviceStats {
            submits: c.submits.load(Ordering::Relaxed),
            dispatches: c.dispatches.load(Ordering::Relaxed),
            readbacks: c.readbacks.load(Ordering::Relaxed),
            bind_groups: c.bind_groups.load(Ordering::Relaxed),
            uniform_allocs: c.uniform_allocs.load(Ordering::Relaxed),
            writes: c.writes.load(Ordering::Relaxed),
        })
    }

    fn caps(&self) -> backend_api::DeviceCaps {
        use backend_api::arch::{ArchDesc, IsaFeatures, TierLevel, TierSupport};
        use backend_api::{DType, DeviceCaps, DeviceClass};

        // I8 is `Native` iff AVX2 is available AND not disabled by
        // `BRAIN_NO_FASTCONV` (`kernel-performance.md` M8.11:
        // `fast_ops::matmul_i8_dyn`'s `_mm256_maddubs_epi16`-based GEMM,
        // registered under `register_native("cpu_matmul_i8_dyn")` - see that
        // function's own doc). `Native`, not `Emulated`: this is genuine
        // dedicated int8 SIMD hardware (real AVX2 instructions), unlike
        // `backend-wgpu`'s `dot4I8Packed` polyfill case M8.1 already
        // distinguishes with `Emulated` - the exact fast/polyfill split
        // `ArchDesc::TierLevel`'s own doc comment describes. `Absent`
        // otherwise (`BRAIN_NO_FASTCONV=1` or a non-x86_64/pre-AVX2 host) -
        // there is genuinely no int8 SIMD path to report then, not merely
        // an unmeasured one.
        // F16/BF16 are `Storage`, never higher: host RAM holds any byte
        // layout, but there is no fast f16/bf16 compute path here.
        let mut arch = ArchDesc::default();
        arch.set_tier(DType::F16, TierSupport { level: TierLevel::Storage, ..Default::default() });
        arch.set_tier(DType::BF16, TierSupport { level: TierLevel::Storage, ..Default::default() });
        arch.set_tier(
            DType::I8,
            TierSupport {
                level: if self.shared.fast_native_enabled { TierLevel::Native } else { TierLevel::Absent },
                ..Default::default()
            },
        );
        // Reuses `fast_conv`'s own runtime CPUID probes - never reprobed
        // here. `avx2_available`/`avx512_available` each already require FMA/
        // VL/DQ alongside the base bit, so both ISA fields mirror the same
        // call rather than inventing a separate FMA-only probe.
        // `avx512_vnni` (M8.12): a real probe now exists
        // (`fast_conv::avx512_vnni_available`), unlike M8.1's own "no VNNI/
        // AMX/NEON probe exists yet" floor - `amx_bf16`/`amx_int8` stay the
        // honest default (`false`, never probed): AMX detection AND
        // intrinsics are both still `#![feature(x86_amx_intrinsics)]`-gated
        // on this stable toolchain (confirmed by trying to compile
        // `is_x86_feature_detected!("amx-tile")` directly - `error[E0658]:
        // use of unstable library feature`), so there is no real probe to
        // wire without switching this crate to nightly, which is out of
        // scope for a single ISA-pack milestone. See `kernel-performance.md`
        // M8.12's own ledger entry for this as a documented follow-up, not a
        // silent omission.
        // `neon`/`neon_dotprod` (M8.13): real probes, `#[cfg(target_arch =
        // "aarch64")]`-gated inside `fast_conv` itself - both always `false`
        // on this x86_64 box (there is no ARM core to detect), and
        // UNVALIDATED anywhere in this campaign (no aarch64 target installed
        // to even compile-check on - see `fast_conv::neon_dotprod_available`'s
        // own honesty note).
        arch.isa = IsaFeatures {
            avx2: fast_conv::avx2_available(),
            fma: fast_conv::avx2_available(),
            avx512f: fast_conv::avx512_available(),
            avx512_vnni: fast_conv::avx512_vnni_available(),
            neon: fast_conv::neon_available(),
            neon_dotprod: fast_conv::neon_dotprod_available(),
            ..IsaFeatures::default()
        };

        DeviceCaps {
            class: DeviceClass::Cpu,
            compute_units: Some(self.shared.threads as u32),
            // The JIT's execution model, not a hardware limit: workgroups run
            // as split-at-barrier loops, and the register-tiled 256-thread
            // kernels are the largest in the tree.
            max_workgroup_size: 256,
            workgroup_mem_bytes: 32 * 1024,
            subgroup_size: None, // no SIMD width is surfaced to WGSL
            unified_memory: true,
            // The split-at-barrier JIT mis-executes the workgroup-cooperative
            // reduction kernels (measured token-for-token); the AVX2 fast
            // paths own the decode regime here instead.
            workgroup_reductions: false,
            // Measured by `gpu_core::roof` like every other device - a CPU has
            // a roofline too, and the JIT's kernels are graded against it.
            peak_bandwidth_gbs: None,
            peak_gflops: None,
            // `fp8_storage` (M8.6) is not one of `ArchDesc`'s modelled tiers
            // (portable FP8 decode is a plain single-top-level-barrier
            // select/bitcast kernel, same JIT-compatibility reasoning as the
            // f16/bf16 storage tiers `numeric_view()` already derives) - set
            // directly here rather than threading a new tier through
            // `ArchDesc` for one orthogonal flag.
            // `int8_dot` is forced `false` here EVEN THOUGH `arch.tier(I8) ==
            // Native` above - a deliberate divergence from `arch.numeric_view()`,
            // the same shape M8.1 already precedented in the opposite direction
            // (`vulkan_no_dp4a_still_executes_i8_unlike_the_old_formula`: ArchDesc
            // and the legacy flattened view are allowed to disagree when they are
            // really answering different questions). Here they are: `arch.tier`
            // answers "does M8.11's own native AVX2 `matmul_i8_dyn` GEMM work"
            // (yes) - reached ONLY through `register_native`/`step_native`, never
            // through `select::candidates`. `numeric.int8_dot` answers a
            // DIFFERENT, older question this backend's `select::candidates` still
            // asks directly: "can `KernelVariant::PackedInt8`'s WGSL kernel
            // (`matmul_i8_dyn.wgsl` et al, each `@cpu no` in its own header - not
            // CPU-JIT-compilable, multi-barrier work-group) run here". It cannot,
            // on ANY shape: `candidates`'s `Dtype::I8 | Q4 | Q4K | Q8K | NF4 |
            // F4E2M1` arm returns `vec![PackedInt8]` alone whenever
            // `!caps.workgroup_reductions` (unconditionally true on this backend),
            // so `int8_dot: true` here would make EVERY int8-family matmul on
            // this backend select a kernel the JIT cannot correctly execute -
            // confirmed by reading `candidates`'s real match arm, not assumed.
            numeric: NumericSupport { int8_dot: false, fp8_storage: true, ..arch.numeric_view() },
            arch,
        }
    }
    fn share(&self) -> Option<Box<dyn Backend>> {
        // Eager execution, no per-handle stream: sharing is an Arc clone.
        // A fresh handle starts its own counters - see `CpuBackend::stats`.
        Some(Box::new(CpuBackend { shared: self.shared.clone(), stats: OpCounters::default() }))
    }
    fn downgrade(&self) -> Option<Box<dyn backend_api::WeakBackend>> {
        Some(Box::new(WeakCpu(std::sync::Arc::downgrade(&self.shared))))
    }

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
    fn write_at(&self, buf: &DeviceBuffer, offset_words: u64, data: &[u32]) {
        CpuBackend::write_at(self, buf.downcast_ref::<CpuBuffer>(), offset_words, data)
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
    fn register_native(&self, spec: &backend_api::NativeSpec) -> Option<backend_api::NativeId> {
        CpuBackend::register_native(self, spec)
    }
    fn step_native(&self, id: backend_api::NativeId, bufs: &[&DeviceBuffer], params: &[u32], threads: u32) -> Option<Step> {
        let bs: Vec<&CpuBuffer> = bufs.iter().map(|b| b.downcast_ref::<CpuBuffer>()).collect();
        Some(Step::new(CpuBackend::step_native(self, id, &bs, params, threads)))
    }
}

/// Register this backend under `"cpu"` so the facade can build it by name.
pub fn register() {
    backend_api::register_backend("cpu", |kernels| Ok(Box::new(CpuBackend::new(kernels))));
}
